import torch
import torch.nn.functional as F
from torch import nn

from mmengine.model.weight_init import constant_init
from torch.nn.modules.utils import _pair

from einops import rearrange

from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.cnn import ConvModule


eps = 1e-6
def corr(f1, f2, md=3):
    b, c, h, w = f1.shape
    # Normalize features
    f1 = f1 / (torch.norm(f1, dim=1, keepdim=True) + eps)  # added epsilon to avoid division by zero
    f2 = f2 / (torch.norm(f2, dim=1, keepdim=True) + eps)

    # Compute correlation matrix
    # Unfold patches from f1 with a size of (2*md+1)x(2*md+1) and a padding of md
    f1_unfold = F.unfold(f1, kernel_size=(2*md+1, 2*md+1), padding=(md, md))
    f1_unfold = rearrange(f1_unfold, "b (c q) (h w) -> b c q h w", h=h, w=w, c=c)

    # Expand f2 for broadcasting
    f2 = f2.view(b, c, 1, h, w)

    # Sum over channels to get the correlation volume
    corr_volume = torch.sum(f1_unfold * f2, dim=1)

    return corr_volume


class PWCNetAlignment(nn.Module):
    def __init__(self, n_channels, deform_groups):
        super(PWCNetAlignment, self).__init__()

        act_cfg=dict(type='LeakyReLU', negative_slope=0.1)
        self.md = 3

        self.deform_groups = deform_groups
        self.n_channels = n_channels

        self.downsample = nn.AvgPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2])

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', 
                                    align_corners=False)

        act_cfg=dict(type='LeakyReLU', negative_slope=0.1)


        self.conv2_enc_a = ConvModule(n_channels, n_channels, 3, padding=1, stride=1, act_cfg=act_cfg)
        self.conv2_enc_b = ConvModule(n_channels, n_channels, 3, padding=1, stride=1, act_cfg=act_cfg)

        self.conv3_enc_a = ConvModule(n_channels, n_channels, 3, padding=1, stride=1, act_cfg=act_cfg)
        self.conv3_enc_b = ConvModule(n_channels, n_channels, 3, padding=1, stride=1, act_cfg=act_cfg)

        self.conv3_dec_a = ConvModule((self.md*2+1)**2, n_channels, 3, padding=1, stride=1, act_cfg=act_cfg)
        self.conv3_dec_b = ConvModule(n_channels, n_channels, 3, padding=1, stride=1, act_cfg=act_cfg)
        self.conv3_dec_c = ConvModule(n_channels*2, n_channels, 3, padding=1, stride=1, act_cfg=act_cfg)

        self.conv2_dec_a = ConvModule((self.md*2+1)**2 + 4*n_channels, n_channels, 3, padding=1, stride=1, act_cfg=act_cfg)
        self.conv2_dec_b = ConvModule(n_channels, n_channels, 3, padding=1, stride=1, act_cfg=act_cfg)
        self.conv2_dec_c = ConvModule(n_channels*2, n_channels, 3, padding=1, stride=1, act_cfg=act_cfg)

        self.conv1_dec_a = ConvModule((self.md*2+1)**2 + 4*n_channels, n_channels, 3, padding=1, stride=1, act_cfg=act_cfg)
        self.conv1_dec_b = ConvModule(n_channels, n_channels, 3, padding=1, stride=1, act_cfg=act_cfg)
        self.conv1_dec_c = ConvModule(n_channels*2, n_channels, 3, padding=1, stride=1, act_cfg=act_cfg)


        self.dcn_pack_2 = ModulatedDCNPack(n_channels, n_channels, 3,
                                           padding=1, deform_groups=deform_groups)
        self.dcn_pack_1 = ModulatedDCNPack(n_channels, n_channels, 3,
                                           padding=1, deform_groups=deform_groups)
        self.dcn_pack_f = ModulatedDCNPack(n_channels, n_channels, 3,
                                           padding=1, deform_groups=deform_groups)
        
        self.refiner = nn.Sequential(
            ConvModule(n_channels*2, n_channels, 3, padding=1, stride=1, act_cfg=act_cfg),
            ConvModule(n_channels, n_channels, 3, padding=1, stride=1, act_cfg=act_cfg),
            ConvModule(n_channels, n_channels, 3, padding=1, stride=1, act_cfg=None),
        )
        
    def forward(self, x):
        # x1: level 1, original spatial size
        # x2: level 2, 1/2 spatial size
        # x3: level 3, 1/4 spatial size

        b, t, c, h, w = x.shape  # Extract shape of x1 to variables: batch size, number of frames, channels, height, width

        # Extract center frame features for each level

        # UNet Encoder Part
        feat_center_l1 = x[:, t // 2, :, :, :]  # Feature at the exact middle frame from level 1
        feat_center_l2 = self.downsample(self.conv2_enc_b(self.conv2_enc_a(feat_center_l1)))
        feat_center_l3 = self.downsample(self.conv3_enc_b(self.conv3_enc_a(feat_center_l2)))

        out_feat = [] # List to store aligned features at level 1

        # Coarse Alignment using Deformable Convolution
        for i in range(0, t):
            if i == t // 2:
                # Append the center frame as-is for the middle frame
                out_feat.append(x[:, i, :, :, :])
            else:
                
                # Level 3 Offset Compute ``offset3``
                feat_neig_l1 = x[:, i, :, :, :].contiguous()
                
                feat_neig_l2 = self.conv2_enc_b(self.conv2_enc_a(self.downsample(feat_neig_l1)))
                feat_neig_l3 = self.conv3_enc_b(self.conv3_enc_a(self.downsample(feat_neig_l2)))

                # Align L3
                cost_feat_l3 = F.leaky_relu(input=corr(feat_center_l3, feat_neig_l3, self.md), negative_slope=0.1, inplace=False)
                cost_feat_l3 = self.conv3_dec_a(cost_feat_l3)
                cost_feat_l3 = torch.cat([ self.conv3_dec_b(cost_feat_l3), cost_feat_l3 ], 1)
                offset3 = self.conv3_dec_c(cost_feat_l3)
                # Upsample Flow and Feature
                u_cost_feat_l3 = self.upsample(cost_feat_l3)
                u_offset3 = self.upsample(offset3) * 2

                # Align L2: Warping l2 -> Cost Volume
                feat_align_l2 = self.dcn_pack_2(feat_neig_l2, u_offset3)
                cost_volume_l2 = F.leaky_relu(input=corr(feat_center_l2, feat_align_l2, self.md), negative_slope=0.1, inplace=False)
                cost_feat_l2 = self.conv2_dec_a(torch.cat([ cost_volume_l2, feat_center_l2, u_offset3, u_cost_feat_l3 ], 1))
                cost_feat_l2 = torch.cat([ self.conv2_dec_b(cost_feat_l2), cost_feat_l2 ], 1)
                offset2 = self.conv2_dec_c(cost_feat_l2)
                # Upsample Flow and Feature
                u_cost_feat_l2 = self.upsample(cost_feat_l2)
                u_offset2 = self.upsample(offset2) * 2

                # Align L1
                feat_align_l1 = self.dcn_pack_1(feat_neig_l1, u_offset2)
                cost_volume_l1 = F.leaky_relu(input=corr(feat_center_l1, feat_align_l1, self.md), negative_slope=0.1, inplace=False)
                cost_feat_l1 = self.conv1_dec_a(torch.cat([ cost_volume_l1, feat_center_l1, u_offset2, u_cost_feat_l2 ], 1))
                cost_feat_l1 = torch.cat([ self.conv1_dec_b(cost_feat_l1), cost_feat_l1 ], 1)
                offset1 = self.conv1_dec_c(cost_feat_l1)

                offset = offset1 + self.refiner(cost_feat_l1) * 20

                # Final Output
                feat_align_f = self.dcn_pack_f(feat_neig_l1, offset)

                out_feat.append(feat_align_f)

        return torch.stack(out_feat, dim=1) 

class ModulatedDCNPack(ModulatedDeformConv2d):
    """Modulated Deformable Convolutional Pack.

    Different from the official DCN, which generates offsets and masks from
    the preceding features, this ModulatedDCNPack takes another different
    feature to generate masks and offsets.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True)
        self.init_offset()

    def init_offset(self):
        """Init constant offset."""
        constant_init(self.conv_offset, val=0, bias=0)

    def forward(self, x, extra_feat):
        """Forward function."""
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


if __name__ == "__main__":

    from torch.profiler import profile
    import pyprof
    pyprof.init()

    n_channels = 64  # Number of channels in the input
    deform_groups = 8  # Number of deformable groups
    
    model = MultiscaleAlignment(n_channels, deform_groups).cuda()
    
    # Create a test input tensor
    batch_size = 6
    temporal_dimension = 7  # Number of frames
    height, width = 64, 64  # Spatial dimensions
    input = torch.randn(batch_size, temporal_dimension, n_channels, height, width).cuda()
    
    # Run the model
    output = model(input)
