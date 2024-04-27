import torch
import torch.nn as nn
import torch.nn.functional as F
'''
Hou, Q., Zhou, D. and Feng, J., 2021. 
Coordinate attention for efficient mobile network design. 
In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition 
(pp. 13713-13722).
'''
class COAMBlock(nn.Module):
    def __init__(self, n_channels, reduction=32):
        super().__init__()

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, n_channels // reduction)

        self.conv1 = nn.Conv2d(n_channels, mip, kernel_size=1, stride=1, padding=0)
        self.act = nn.Hardswish(inplace=True)
        
        self.conv_h = nn.Conv2d(mip, n_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, n_channels, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
    
if __name__=="__main__":
    device='cuda'
    # Define input parameters
    n_channels = 64
    kernel_size = 3

    # Create a sample input tensor
    batch_size = 1
    height, width = 32, 32  # Size of the input image
    input_tensor = torch.randn(batch_size, n_channels, height, width).to(device)
    block = COAMBlock(n_channels).to(device)
    print(block(input_tensor).size())
    print(input_tensor.size())
