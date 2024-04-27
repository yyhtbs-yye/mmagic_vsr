import torch
import torch.nn as nn
import torch.nn.functional as F
'''
Liu, S., Huang, D. and Wang, Y., 2019. 
Learning spatial fusion for single-shot object detection. 
arXiv preprint arXiv:1911.09516.
'''
class ASFFBlock(nn.Module):
    def __init__(self, n_channels=256):
        super().__init__()

        # Level 2 specific initialization
        self.compress_level_0 = ConvBlock(n_channels, n_channels, 1, 1)
        self.compress_level_1 = ConvBlock(n_channels, n_channels, 1, 1)
        self.compress_level_2 = ConvBlock(n_channels, n_channels, 1, 1)
        self.expand = ConvBlock(n_channels, n_channels, 3, 1)

        compress_c = 16  
        self.weight_level_0 = ConvBlock(n_channels, compress_c, 1, 1)
        self.weight_level_1 = ConvBlock(n_channels, compress_c, 1, 1)
        self.weight_level_2 = ConvBlock(n_channels, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x_level_0, x_level_1, x_level_2):
        level_0_feat = self.compress_level_0(x_level_0)
        level_1_feat = F.interpolate(self.compress_level_1(x_level_1), 
                                        size=level_0_feat.shape[-2:], 
                                        mode='nearest')
        level_2_feat = F.interpolate(self.compress_level_2(x_level_2), 
                                        size=level_0_feat.shape[-2:], 
                                        mode='nearest')

        level_0_weight_v = self.weight_level_0(level_0_feat)
        level_1_weight_v = self.weight_level_1(level_1_feat)
        level_2_weight_v = self.weight_level_2(level_2_feat)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = (level_0_feat * levels_weight[:,0,:,:].unsqueeze(1) +
                             level_1_feat * levels_weight[:,1,:,:].unsqueeze(1) +
                             level_2_feat * levels_weight[:,2,:,:].unsqueeze(1))

        out = self.expand(fused_out_reduced)

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size=3, stride=1):
        super().__init__()
        pad = (kernel_size - 1) // 2  # Calculate padding based on the kernel size
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=pad, bias=False),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        return self.layers(x)

if __name__=="__main__":
    device='cuda'
    # Define input parameters
    n_channels = 64
    kernel_size = 3

    # Create a sample input tensor
    batch_size = 1
    height, width = 32, 32  # Size of the input image
    input_tensor0 = torch.randn(batch_size, n_channels, height, width).to(device)
    height, width = 16, 16  # Size of the input image
    input_tensor1 = torch.randn(batch_size, n_channels, height, width).to(device)
    height, width = 8, 8  # Size of the input image
    input_tensor2 = torch.randn(batch_size, n_channels, height, width).to(device)

    block = ASFFBlock(n_channels=64).to(device)
    print(block(input_tensor0, input_tensor1, input_tensor2).size())
    print(input_tensor0.size())
