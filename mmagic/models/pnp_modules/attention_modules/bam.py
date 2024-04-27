import torch
import torch.nn as nn
import torch.nn.functional as F
'''
Park, J., Woo, S., Lee, J.Y. and Kweon, I.S., 2018. 
BAM: Bottleneck Attention Module. 
In British Machine Vision Conference (BMVC). 
British Machine Vision Association (BMVA).
'''

class BAMBlock(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16, dilation=4):
        super().__init__()
        self.channel_attention = ChannelAttention(num_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(num_channels, reduction_ratio, dilation=dilation)

    def forward(self, x):
        # Apply both attentions using addition, then apply sigmoid for final attention map
        cam_scale = self.channel_attention(x)
        sam_scale = self.spatial_attention(x)
        bam_scale = torch.sigmoid(cam_scale + sam_scale)
        
        return bam_scale * x + x

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(num_channels // reduction_ratio, num_channels),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.avg_pool(x).view(b, c)
        out = self.fc(out).view(b, c, 1, 1)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16, dilation=4):
        super(SpatialAttention, self).__init__()
        self.reduced_channels = num_channels // reduction_ratio
        
        # 1x1 convolution
        self.conv1 = nn.Conv2d(num_channels, self.reduced_channels, kernel_size=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # First 3x3 convolution with dilation
        self.conv2 = nn.Conv2d(self.reduced_channels, self.reduced_channels, 
                               kernel_size=3, padding=dilation, 
                               dilation=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Second 3x3 convolution with dilation
        self.conv3 = nn.Conv2d(self.reduced_channels, self.reduced_channels, 
                               kernel_size=3, padding=dilation, 
                               dilation=dilation)
        self.relu3 = nn.ReLU(inplace=True)
        
        # 1x1 convolution to get to single channel output
        self.conv4 = nn.Conv2d(self.reduced_channels, 1, 
                               kernel_size=1)
        
    def forward(self, x):
        # Apply the layers with ReLU activations
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        
        # Apply sigmoid to get the attention map
        return self.conv4(x)


if __name__=="__main__":
    device='cuda'
    # Define input parameters
    n_channels = 64
    kernel_size = 3

    # Create a sample input tensor
    batch_size = 1
    height, width = 32, 32  # Size of the input image
    input_tensor = torch.randn(batch_size, n_channels, height, width).to(device)
    block = BAMBlock(n_channels).to(device)
    print(block(input_tensor).size())
    print(input_tensor.size())