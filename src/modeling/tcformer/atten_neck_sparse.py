# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

import numpy as np
from .pvt_v2 import trunc_normal_, DropPath
from .pvt_v2_3h2_density_fix_sparse import MyMlp
from .utils_mine import token2map_agg_sparse_new as token2map_agg_mat
from .utils_mine import downup_sparse_new as downup
from .utils_mine import DPC_flops, token2map_flops, map2token_flops, downup_flops, sra_flops
import math


class MergeAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear

        self.pool = nn.AdaptiveAvgPool2d(7)
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, tar_dict, gather_dict, src_dict=None):
        x = tar_dict['x']
        if src_dict is None:
            src_dict = tar_dict
        x_source = downup(gather_dict, src_dict)

        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x_source = self.norm(x_source)
        x_source = self.act(x_source)
        kv = self.kv(x_source).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MergeBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MergeAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MyMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, tar_dict, gather_dict, src_dict=None):

        x = tar_dict['x']
        tar_dict['x'] = self.norm1(x)
        if src_dict is not None:
            src_dict['x'] = self.norm1(src_dict['x'])

        x = x + self.drop_path(self.attn(tar_dict, gather_dict, src_dict))
        H, W = tar_dict['map_size']
        x = x + self.drop_path(self.mlp(self.norm2(x),
                                        tar_dict['loc_orig'],
                                        tar_dict['idx_agg'],
                                        tar_dict['agg_weight'],
                                        H, W))
        return x

'''
From stage1 to stage4
'''

class AttenNeckS(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels=128,
                 num_outs=1,
                 start_level=0,
                 end_level=-1,
                 num_heads=[2, 2, 2, 2],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,

                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),
                 ):
        super().__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        self.mlp_ratios = mlp_ratios

        self.start_level = start_level
        if end_level == -1:
            end_level = len(in_channels) - 1
        self.end_level = end_level

        self.add_extra_convs = add_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.merge_blocks = nn.ModuleList()

        for i in range(self.start_level, self.end_level + 1):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)

        for i in range(self.start_level, self.end_level):
            merge_block = MergeBlock(
                dim=out_channels, num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
            )
            self.merge_blocks.append(merge_block)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build lateral tokens
        input_dicts = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            tmp = inputs[i + self.start_level]
            input_dicts.append(
                {'x': lateral_conv(tmp[0].unsqueeze(2).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze(2),
                 'map_size': tmp[2],
                 'loc_orig': tmp[3],
                 'idx_agg': tmp[4],
                 'agg_weight': tmp[5]
                 })

        # build gather dict
        tmp = inputs[-1]
        gather_dict = {'x': tmp[0],
                       'map_size': tmp[2],
                       'loc_orig': tmp[3],
                       'idx_agg': tmp[4],
                       'agg_weight': tmp[5]}

        # for i in range(len(input_dicts) - 2, -1, -1):
        #     input_dicts[i]['x'] = input_dicts[i]['x'] + downup(input_dicts[i], input_dicts[i+1])
        #     input_dicts[i]['x'] = self.merge_blocks[i](input_dicts[i], gather_dict, input_dicts[i])
        #
        # out, _ = token2map_agg_mat(
        #     input_dicts[0]['x'],
        #     None,
        #     input_dicts[0]['loc_orig'],
        #     input_dicts[0]['idx_agg'],
        #     input_dicts[0]['map_size'],
        # )

        for i in range(1, len(input_dicts)):
            input_dicts[i]['x'] = input_dicts[i]['x'] + downup(input_dicts[i], input_dicts[i-1])
            input_dicts[i]['x'] = self.merge_blocks[i-1](input_dicts[i], gather_dict, input_dicts[i])
        out = input_dicts[-1]['x']
        return out

    def get_extra_flops(self, h, w):
        flops = 0
        C = self.out_channels
        N0 = h * w
        N4 = N0 // 4 // 4 // 4
        for i in range(3):
            # down up
            flops += downup_flops(N0, C)
            # tokenmap
            flops += token2map_flops(N0, C) + token2map_flops(N0, C * self.mlp_ratios[i]) + map2token_flops(N0, C * self.mlp_ratios[i])
            # attn
            flops += 2 * N0 * N4 * C
        return flops
