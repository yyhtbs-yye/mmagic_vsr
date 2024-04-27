# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmagic.registry import MODELS


from einops import rearrange
from einops.layers.torch import Rearrange

from mmagic.models.basicsr_utils.motion_estimation import get_flow_between_frames
from mmagic.models.basicsr_utils.motion_compensation import get_aligned_image_2frames, get_aligned_feature_2frames

from mmagic.models.basicsr_archs.spynet_arch import SpyNet
from mmagic.models.basicsr_archs.x_attentions import GLUAttention

from mmagic.models.basicsr_archs.upsample import Conv3dPixelShuffle as Upsample


from .dcnv2_modules import DCNv2PackFlowGuided

from .vrt_modules import VRTModule, VRTMod8
@MODELS.register_module()
class VRTNet(BaseModule):
    r""" VRT
        Video Restoration Transformer (VRT),
        A PyTorch impl of : `VRT: A Video Restoration Transformer` 
        https://arxiv.org/pdf/2201.00000

    Args:
        upscale (int): Upscaling factor. Set as 1 for video deblurring, etc. Default: 4.
        in_channels (int): Number of input image channels. Default: 3.
        out_channels (int): Number of output image channels. Default: 3.
        window_size (int | tuple(int)): Window size. Default: (6,8,8).
        depths (list[int]): Depths of each Transformer stage.
        indep_reconsts (list[int]): Layers that extract features of different frames independently.
        embed_dims (list[int]): Number of linear projection output channels.
        n_heads (list[int]): Number of attention head of each stage.
        mul_attn_ratio (float): Ratio of mutual attention layers. Default: 0.75.
        mlp_ratio (float): Ratio of mlp hidden n_channels to embedding n_channels. Default: 2.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (obj): Normalization layer. Default: nn.LayerNorm.
        spynet_path (str): Pretrained SpyNet model path.
        n_frames (float): Number of warpped frames. Default: 2.
        deformable_groups (float): Number of deformable groups. Default: 16.
        recal_all_flows (bool): If True, derive (t,t+2) and (t,t+3) flows from (t,t+1). Default: False.
        no_checkpoint_attn_blocks (list[int]): Layers without torch.checkpoint for attention modules.
        no_checkpoint_ffn_blocks (list[int]): Layers without torch.checkpoint for feed-forward modules.
    """

    def __init__(self,
                 upscale=4,
                 in_channels=3,
                 out_channels=3,
                 window_size=[6, 8, 8],
                 unet_block_depths = 8, 
                 unet_block_channels = 64,
                 unet_n_heads = 4,  
                 lnet_n_blocks=6,
                 lnet_block_depths = 4, 
                 lnet_block_channels = 96,
                 lnet_n_heads = 4,  
                 lnet_indep_layers=[11, 12],
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 spynet_path=None,
                 n_frames=2,
                 deformable_groups=16,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upscale = upscale
        self.n_frames = n_frames

        # conv_first
        self.conv_first = nn.Conv3d(in_channels * (1 + 2 * 4), unet_block_depths, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        self.spynet = SpyNet(spynet_path, [2, 3, 4, 5])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 7*unet_block_depths+lnet_n_blocks*lnet_block_depths)]  # stochastic depth decay rule
        
        self.unet = nn.ModuleList([
            VRTModule(
                in_channels=unet_block_channels,
                out_channels=unet_block_channels,
                n_mma_blocks = unet_block_depths / 4 * 3,
                n_msa_blocks = unet_block_depths / 4 * 1,
                # Confgures for TMSAG
                depth=unet_block_depths, n_heads=unet_n_heads, window_size=window_size, 
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                drop_path=dpr[i*unet_block_depths:(i + 1)*unet_block_depths],
                norm_layer=norm_layer,
                # Confgures for PW
                deformable_groups=16,  kernel_size=3, padding=1, 
                                  max_residue_magnitude=10/scale,
                n_frames=n_frames,
                reshape=reshape,
            )
            for i, (reshape, scale) in enumerate(zip(['none', 'down', 'down', 'down', 'up', 'up', 'up'], 
                                                     [1, 2, 4, 8, 4, 2, 1]))
        ])

        self.lnet = VRTMod8(in_channels=unet_block_channels, out_channels=lnet_block_channels, 
                            n_blocks=lnet_n_blocks, depth=lnet_block_depths, 
                            n_heads=lnet_n_heads, window_size=window_size, 
                            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                            drop_paths=[(i + 1)*unet_block_depths???], norm_layer=norm_layer,
                            indep_layers=lnet_indep_layers)

        # stage 8
        self.stage8 = nn.ModuleList(
            [nn.Sequential(
                Rearrange('n c d h w ->  n d h w c'),
                nn.LayerNorm(embed_dims[6]),
                nn.Linear(embed_dims[6], embed_dims[7]),
                Rearrange('n d h w c -> n c d h w')
            )]
        )
        for i in range(7, len(depths)):
            self.stage8.append(
                RTMSA(n_channels=embed_dims[i],
                      depth=depths[i],
                      n_heads=n_heads[i],
                      window_size=[1, window_size[1], window_size[2]] if i in indep_reconsts else window_size, # T=1 or T=all
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                      norm_layer=norm_layer,
                      )
            )

        self.norm = norm_layer(embed_dims[-1])
        self.conv_after_body = nn.Linear(embed_dims[-1], embed_dims[0])

        # for video sr
        num_feat = 64
        self.conv_before_upsample = nn.Sequential(
            nn.Conv3d(embed_dims[0], num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv3d(num_feat, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, x):
        # x: (N, D, C, H, W)

        b, t, c, h, w = x.shape

        x_lq = x.clone()

        # calculate flows
        flows_backward, flows_forward = self.get_flows(x)

        # warp input
        x_backward, x_forward = get_aligned_image_2frames(x,  flows_backward[0], flows_forward[0])
        x = torch.cat([x, x_backward, x_forward], 2)

        # video sr
        x = self.conv_first(x.transpose(1, 2))
        x = x + self.conv_after_body(
            self.forward_features(x, flows_backward, flows_forward).transpose(1, 4)).transpose(1, 4)
        
        x = self.conv_last(self.upsample(self.conv_before_upsample(x))).transpose(1, 2)

        B, T, C, H, W = x.shape

        return x + torch.nn.functional.interpolate(x_lq.view(-1, c, h, w), size=(H, W), mode='bilinear', align_corners=False).view(B, T, C, H, W)


    def get_flows(self, x):
        ''' Get flows for 2 frames, 4 frames or 6 frames.'''

        if self.n_frames == 2:
            flows_backward, flows_forward = get_flow_between_frames(x, self.spynet)

        return flows_backward, flows_forward


    def forward_features(self, x, flows_backward, flows_forward):
        '''Main network for feature extraction.'''

        x1 = self.stage1(x, flows_backward[0::4], flows_forward[0::4])
        x2 = self.stage2(x1, flows_backward[1::4], flows_forward[1::4])
        x3 = self.stage3(x2, flows_backward[2::4], flows_forward[2::4])
        x4 = self.stage4(x3, flows_backward[3::4], flows_forward[3::4])
        x = self.stage5(x4, flows_backward[2::4], flows_forward[2::4])
        x = self.stage6(x + x3, flows_backward[1::4], flows_forward[1::4])
        x = self.stage7(x + x2, flows_backward[0::4], flows_forward[0::4])
        x = x + x1

        for layer in self.stage8:
            x = layer(x)

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        return x
    