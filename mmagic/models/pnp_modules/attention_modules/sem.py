import torch
import torch.nn as nn
import torch.nn.functional as F
'''
Hu, J., Shen, L. and Sun, G., 2018. 
Squeeze-and-excitation networks. 
In Proceedings of the IEEE conference on computer vision and pattern recognition 
(pp. 7132-7141).
'''
class SEMBlock(nn.Module):
    def __init__(self, n_channels=64, 
                 reduction_ratio=16):
        
        super().__init__()
        self.res_scale = 1.0
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=True)
        self.acti = nn.ReLU(inplace=True)

        # Squeeze and Excitation layers
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Squeeze
            nn.Conv2d(n_channels, n_channels // reduction_ratio, 1, bias=True),  # Excitation: Reduction
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // reduction_ratio, n_channels, 1, bias=True),  # Excitation: Expansion
            nn.Sigmoid()
        )

        self.init_weights()

    def init_weights(self):
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.acti(out)
        out = self.conv2(out)

        # Squeeze and Excitation operation
        scale = self.se(out)
        out = out * scale  # Scale the output by the SE layer

        return identity + out * self.res_scale

if __name__=="__main__":
    device='cuda'
    # Define input parameters
    n_channels = 64

    # Create a sample input tensor
    batch_size = 1
    height, width = 32, 32  # Size of the input image
    input_tensor = torch.randn(batch_size, n_channels, height, width).to(device)
    block = SEMBlock(n_channels).to(device)
    print(block(input_tensor).size())
    print(input_tensor.size())