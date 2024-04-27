import torch
import torch.nn as nn
import torch.nn.functional as F
'''
Jaderberg, M., Simonyan, K. and Zisserman, A., 2015. 
Spatial transformer networks. 
Advances in neural information processing systems, 
28.
'''
# Preprocessing Module
class STNBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(n_channels, n_channels//2, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(n_channels//2, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Fully connected layers for transformation prediction
        self.fc_loc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),  # Flatten the output into a single vector per image
            nn.Linear(10, 6)  # Predict six parameters for affine transformation
        )

    def forward(self, x):
        b, c, h, w, = x.shape
        xs = self.localization(x)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # Grid generator and sampler
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x
    
if __name__=="__main__":
    device='cuda'
    # Define input parameters
    n_channels = 64
    kernel_size = 3

    # Create a sample input tensor
    batch_size = 1
    height, width = 32, 32  # Size of the input image
    input_tensor = torch.randn(batch_size, n_channels, height, width).to(device)
    block = STNBlock(n_channels).to(device)
    print(block(input_tensor).size())
    print(input_tensor.size())
