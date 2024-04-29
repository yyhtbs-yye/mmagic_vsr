import torch
import torch.nn as nn
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from torch.nn.modules.utils import _pair

class DCNv2PackFlowGuided(ModulatedDeformConv2d):

    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, 
                 deformable_groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, 
                         stride, padding, dilation, groups, bias)
        
        self.deform_groups = deformable_groups
        
        self.conv_offset_mask = nn.Conv2d(
            in_channels + 2 * deformable_groups,  # assuming flow fields are appended to input channels
            deformable_groups * 3 * _pair(kernel_size)[0] * _pair(kernel_size)[1],
            kernel_size=_pair(kernel_size),
            stride=_pair(stride),
            padding=_pair(padding),
            bias=True)
        self.init_offset_mask()

    def init_offset_mask(self):
        """Initialize offset and mask to zero."""
        nn.init.constant_(self.conv_offset_mask.weight, 0)
        nn.init.constant_(self.conv_offset_mask.bias, 0)

    def forward(self, x, extra_feat):
        out = self.conv_offset_mask(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)

