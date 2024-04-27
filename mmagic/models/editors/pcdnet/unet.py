import torch
import torch.nn.functional as F
from torch import nn

from mmcv.ops import DeformConv2d, DeformConv2dPack, deform_conv2d

from einops import rearrange

class Reshaper(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Reshaper, self).__init__()
        self.block = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
        self.acti = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    def forward(self, x):

        return self.acti(self.block(x))

class Downsampler(nn.Module):
    def __init__(self):
        super(Downsampler, self).__init__()

        self.downsample = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), return_indices=True)
    def forward(self, x):
        b, t, _, _, _ = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        output_size = x.shape
        x, indices = self.downsample(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', b=b, t=t)
        return x, indices, output_size

class Upsampler(nn.Module):
    def __init__(self):

        super(Upsampler, self).__init__()
        self.upsample = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
    def forward(self, x, indices, output_size):
        b, t, _, _, _ = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.upsample(x, indices, output_size=output_size)
        x = rearrange(x, '(b t) c h w -> b t c h w', b=b, t=t)
        return x

class UNetPipe(nn.Module):
    def __init__(self, enc_block_def=None, enc_block_args=None,
                       dec_block_def=None, dec_block_args=None,):
        super(UNetPipe, self).__init__()

        self.downsample1 = Downsampler()
        self.downsample2 = Downsampler()

        self.upsample1 = Upsampler()
        self.upsample2 = Upsampler()

        self.encode_block1 = enc_block_def(**enc_block_args)
        self.encode_block2 = enc_block_def(**enc_block_args)
        self.encode_block3 = enc_block_def(**enc_block_args)

        self.decode_block1 = dec_block_def(**dec_block_args)
        self.decode_block2 = dec_block_def(**dec_block_args)
        self.decode_block3 = dec_block_def(**dec_block_args)

    def forward(self, x):
        b, t, c, h, w = x.shape  # Assuming x is a batch with dimensions [Batch, Channels, Height, Width]

        # Encoding path
        x_encoded1 = self.encode_block1(x)
        x_encoded1d, indices1, x_encoded1d_size = self.downsample1(x_encoded1)
        x_encoded2 = self.encode_block2(x_encoded1d)
        x_encoded2d, indices2, x_encoded2d_size = self.downsample2(x_encoded2)
        x_encoded3 = self.encode_block3(x_encoded2d)

        # Decoding path
        x_decoded1 = self.decode_block1(x_encoded3)
        upsampled1 = self.upsample1(x_decoded1, indices2, x_encoded2d_size)
        x_decoded2 = self.decode_block2(upsampled1 + x_encoded2)
        upsampled2 = self.upsample2(x_decoded2, indices1, x_encoded1d_size)
        x_decoded3 = self.decode_block3(upsampled2 + x_encoded1)

        x_out = x_decoded3 + x_encoded1

        return x_out