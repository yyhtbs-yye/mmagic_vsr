import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Woo, S., Park, J., Lee, J.Y. and Kweon, I.S., 2018. 
Cbam: Convolutional block attention module. 
In Proceedings of the European conference on computer vision (ECCV) 
(pp. 3-19).
'''
class CBAMBlock(nn.Module):
    def __init__(self, n_channels=64, kernel_size=3, 
                 reduction_ratio=16):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size, padding=kernel_size//2, bias=True)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size, padding=kernel_size//2, bias=True)

        self.cam = ChannelAttention(n_channels, reduction_ratio)
        self.sam = SpatialAttention()

        self.init_weights()

    def init_weights(self):
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.cam(out)
        out = self.conv2(out)
        out = self.sam(out)
        return x + out

class ChannelAttention(nn.Module):
    def __init__(self, n_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction_ratio, n_channels),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction_ratio, n_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc1(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc2(self.max_pool(x).view(x.size(0), -1))
        return x * (avg_out.view(x.size(0), x.size(1), 1, 1) + 
                    max_out.view(x.size(0), x.size(1), 1, 1))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        u = torch.cat([avg_out, max_out], dim=1)
        u = self.conv1(u)
        return self.sigmoid(u) * x  

if __name__=="__main__":
    device='cuda'
    # Define input parameters
    n_channels = 64
    kernel_size = 3

    # Create a sample input tensor
    batch_size = 1
    height, width = 32, 32  # Size of the input image
    input_tensor = torch.randn(batch_size, n_channels, height, width).to(device)
    block = CBAMBlock(n_channels).to(device)
    print(block(input_tensor).size())
    print(input_tensor.size())
