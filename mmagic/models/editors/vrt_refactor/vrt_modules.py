import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from .tmsa_modules import RTMSA, TMSAG

from .dcnv2_modules import DCNv2PackFlowGuided
from mmagic.models.basicsr_archs.x_attentions import GLUAttention
from mmagic.models.basicsr_utils.optical_flow import flow_warp

class VRTModule(nn.Module):

    def __init__(self, in_channels, out_channels, 
                 n_mma_blocks, n_msa_blocks,
                 depth, n_heads, window_size, 
                 mlp_ratio, qkv_bias, qk_scale, 
                 drop_path, norm_layer,
                 deformable_groups, kernel_size, padding, 
                 max_residue_magnitude,
                 n_frames,
                 reshape=None,
                 ):
        super(VRTModule, self).__init__()

        self.n_frames = n_frames

        # reshape the tensor
        if reshape == 'none':
            self.reshape = nn.Sequential(Rearrange('n c d h w -> n d h w c'),
                                         nn.LayerNorm(in_channels), nn.Linear(in_channels, out_channels),
                                         Rearrange('n d h w c -> n c d h w'))
        elif reshape == 'down':
            self.reshape = nn.Sequential(Rearrange('n c d (h neih) (w neiw) -> n d h w (neiw neih c)', neih=2, neiw=2),
                                         nn.LayerNorm(4 * in_channels), nn.Linear(4 * in_channels, out_channels),
                                         Rearrange('n d h w c -> n c d h w'))
        elif reshape == 'up':
            self.reshape = nn.Sequential(Rearrange('n (neiw neih c) d h w -> n d (h neih) (w neiw) c', neih=2, neiw=2),
                                         nn.LayerNorm(in_channels // 4), nn.Linear(in_channels // 4, out_channels),
                                         Rearrange('n d h w c -> n c d h w'))

        # mutual and self attention
        self.residual_group1 = TMSAG(n_channels=out_channels, n_blocks=n_mma_blocks, 
                                     n_heads=n_heads, window_size=window_size, 
                                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                     drop_path=drop_path, norm_layer=norm_layer, mut_attn=True,
        )
        
        self.linear1 = nn.Linear(out_channels, out_channels)

        # only self attention
        self.residual_group2 = TMSAG(n_channels=out_channels, n_blocks=n_msa_blocks, 
                                     n_heads=n_heads, window_size=window_size, 
                                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                     drop_path=drop_path, norm_layer=norm_layer, mut_attn=False,
        )
        self.linear2 = nn.Linear(out_channels, out_channels)

        # parallel warping
        self.fa = RefinedFeatureAligner(out_channels,
                                        n_frames=n_frames, 
                                        deformable_groups=deformable_groups, 
                                        kernel_size=kernel_size, 
                                        padding=padding, 
                                        max_residue_magnituds=max_residue_magnitude)

        self.pa_fuse = GLUAttention(out_channels * (1 + 2), out_channels * (1 + 2), out_channels)

    def forward(self, x, flows_backward, flows_forward):
        # print(x.shape)
        x = self.reshape(x)
        x = self.linear1(self.residual_group1(x).transpose(1, 4)).transpose(1, 4) + x
        x = self.linear2(self.residual_group2(x).transpose(1, 4)).transpose(1, 4) + x

        x = x.transpose(1, 2)
        x_backward, x_forward = self.fa(x, flows_backward, flows_forward)
        x = self.pa_fuse(torch.cat([x, 
                                    x_backward, 
                                    x_forward], 2).permute(0, 1, 3, 4, 2)).permute(0, 4, 1, 2, 3)

        return x

class RefinedFeatureAligner(nn.Module):

    def __init__(self, n_channels, n_frames, 
                 deformable_groups, kernel_size, 
                 padding, max_residue_magnitude):
        
        super(RefinedFeatureAligner, self).__init__()

        self.gdcn = DCNv2PackFlowGuided(n_channels, n_channels, 
                                        n_frames=n_frames, 
                                        deformable_groups=deformable_groups, kernel_size=kernel_size, 
                                        padding=padding, max_residue_magnituds=max_residue_magnitude)

    def forward(self, x, flows_backward, flows_forward):
        n = x.size(1)
        
        # backward pass
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        for i in range(n - 1, 0, -1):
            x_i = x[:, i, ...]
            flow = flows_backward[0][:, i - 1, ...]
            x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), mode='bilinear')
            # Frame i+1 aligned towards i
            processed_feature = self.gdcn(x_i, torch.cat([x_i_warped] + 
                                                         [x[:, i - 1, ...]] + 
                                                         [flow], dim=1))
            x_backward.insert(0, processed_feature)

        # forward pass
        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(n - 1):
            x_i = x[:, i, ...]
            flow = flows_forward[0][:, i, ...]
            x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), mode='bilinear')
            # Frame i-1 aligned towards i
            processed_feature = self.gdcn(x_i, torch.cat([x_i_warped] + 
                                                         [x[:, i + 1, ...]] + 
                                                         [flow], dim=1))
            x_forward.append(processed_feature)

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]