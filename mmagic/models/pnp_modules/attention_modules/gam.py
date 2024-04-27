import torch
import torch.nn as nn
import torch.nn.functional as F
'''
Liu, Y., Shao, Z. and Hoffmann, N., 2021. 
Global attention mechanism: Retain information to enhance channel-spatial interactions. 
arXiv preprint arXiv:2112.05561.
'''
class GAMBlock(nn.Module):
    def __init__(self, n_channels, rate=4):
        super().__init__()

        self.sam = SpatialAttention(n_channels, rate=rate)
        self.cam = ChannelAttention(n_channels, rate=rate)

    def forward(self, x):
        
        cam_scale = self.cam(x)

        sam_scale = self.sam(x)

        out = x * cam_scale * sam_scale

        return out

class SpatialAttention(nn.Module):
    def __init__(self, n_channels, rate=4):
        super().__init__()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(n_channels, int(n_channels / rate), kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(n_channels / rate), n_channels, kernel_size=7, padding=3)
        )
    def forward(self, x):
        return self.spatial_attention(x).sigmoid()

class ChannelAttention(nn.Module):
    def __init__(self, n_channels, rate=4):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(n_channels, int(n_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(n_channels / rate), n_channels)
        )
    
    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        return self.channel_attention(x_permute).view(b, h, w, c).permute(0, 3, 1, 2).sigmoid()

if __name__=="__main__":
    device='cuda'
    # Define input parameters
    n_channels = 64
    kernel_size = 3

    # Create a sample input tensor
    batch_size = 1
    height, width = 32, 32  # Size of the input image
    input_tensor = torch.randn(batch_size, n_channels, height, width).to(device)
    block = GAMBlock(n_channels).to(device)
    print(block(input_tensor).size())
    print(input_tensor.size())
