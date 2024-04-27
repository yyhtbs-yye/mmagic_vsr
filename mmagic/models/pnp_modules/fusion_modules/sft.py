import torch
import torch.nn as nn
import torch.nn.functional as F
'''
Wang, X., Yu, K., Dong, C. and Loy, C.C., 2018. 
Recovering realistic texture in image super-resolution by deep spatial feature transform. 
In Proceedings of the IEEE conference on computer vision and pattern recognition 
(pp. 606-615).
'''
class SFTBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        # Two ConvBlocks are defined, one for scaling and one for shifting
        # These will learn spatially varying parameters
        self.scale_conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)
        self.shift_conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Compute scaling and shifting parameters
        scale = F.leaky_relu(self.scale_conv(x), 0.1)
        shift = F.leaky_relu(self.shift_conv(x), 0.1)

        # Apply the learned scaling and shifting parameters
        return x * scale + shift
    
class CSFTBlock(nn.Module):
    def __init__(self, n_channels, c_channels):
        super().__init__()
        # Initialize convolutional layers to process the condition input
        self.scale_conv = nn.Conv2d(c_channels, n_channels, kernel_size=3, stride=1, padding=1)
        self.shift_conv = nn.Conv2d(c_channels, n_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, c):
        # cond is the external condition which affects the feature map x
        # Generate scale and shift maps from the condition
        scale = F.leaky_relu(self.scale_conv(c), 0.1)
        shift = F.leaky_relu(self.shift_conv(c), 0.1)

        # Apply the learned scaling and shifting parameters
        return x * scale + shift
