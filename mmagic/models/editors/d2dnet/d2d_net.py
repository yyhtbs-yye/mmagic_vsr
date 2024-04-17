from math import sqrt
import functools
import torch
import torch.nn.functional as F
from torch import nn

from mmagic.registry import MODELS
from mmagic.models.utils import make_layer
from mmagic.models.archs import PixelShufflePack
from mmcv.cnn import ConvModule

from .d2d_modules import ResBlockD2D, ResidualBlockNoBN

@MODELS.register_module()
class D2DNet(nn.Module):
    def __init__(self, upscale_factor=4, in_frames=7,
                 in_channels=3, out_channels=3,
                 preproc_config={'n_blocks': 3, 'n_channels': 256, 'kernel_size': 3},
                 align_config={'n_blocks': 1, 'n_channels': 256},
                 postproc_config={'n_blocks': 10, 'n_channels': 256, 'kernel_size': 5}, 
                ):
        super(D2DNet, self).__init__()
        self.upscale_factor = upscale_factor
        self.in_channels = in_channels

        self.pre_feat_extract = nn.Sequential(
            ConvModule(in_channels, preproc_config['n_channels'], 1, padding=0),
            make_layer(
                ResidualBlockNoBN,
                preproc_config['n_blocks'],
                mid_channels=preproc_config['n_channels'],
                kernel_size=preproc_config['kernel_size'],
            ),
            # The line below is a shape transformer
            nn.Conv2d(preproc_config['n_channels'], align_config['n_channels'], 1, padding=0),
        )

        self.temporal_alignment = nn.Sequential(
            make_layer(ResBlockD2D, 
                align_config['n_blocks'], 
                mid_channels=align_config['n_channels']
            )
        )
        
        self.temporal_aggregation = nn.Conv2d(in_frames * align_config['n_channels'], 
                                              postproc_config['n_channels'], 
                                              1, 1, bias=True)
        
        self.pre_feat_enhancement = nn.Sequential(
            make_layer(ResidualBlockNoBN,
                postproc_config['n_blocks'],
                mid_channels=postproc_config['n_channels'],
                kernel_size=postproc_config['kernel_size'],
            )
        )
        
        self.upscale = nn.Sequential(
            PixelShufflePack(postproc_config['n_channels'], postproc_config['n_channels'], 
                             2, upsample_kernel=3),
            PixelShufflePack(postproc_config['n_channels'], postproc_config['n_channels'], 
                             2, upsample_kernel=3),
            nn.Conv2d(postproc_config['n_channels'], out_channels, 3, 1, 1, bias=False)
        )


    def forward(self, x):

        b, t, c, h, w = x.size()

        residual = F.interpolate(x[:, t // 2, :, :, :], 
                                 scale_factor=self.upscale_factor, 
                                 mode='bilinear',
                                 align_corners=False)
        x_reshaped = x.view(b * t, c, h, w)

        out = self.pre_feat_extract(x_reshaped)
        
        out = out.view(b, t, -1, h, w)

        out = self.temporal_alignment(out)
        out = self.temporal_aggregation(out.view(b, -1, h, w))  # B, C, H, W
        out = self.pre_feat_enhancement(out)

        # Upscaling the output feature using RDB
        out = self.upscale(out)
        out = torch.add(out, residual)

        return out 
