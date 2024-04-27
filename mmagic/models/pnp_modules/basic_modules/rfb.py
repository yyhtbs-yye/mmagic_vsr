import torch
import torch.nn as nn
import torch.nn.functional as F
'''
Liu, S. and Huang, D., 2018. 
Receptive field block net for accurate and fast object detection. 
In Proceedings of the European conference on computer vision (ECCV) 
(pp. 385-400).
'''
class RFBlock(nn.Module):
    def __init__(self,n_channels):
        super().__init__()
        q_channels = n_channels // 4
        self.branch_0 = nn.Sequential(
                    nn.Conv2d(in_channels=n_channels, out_channels=q_channels, 
                              kernel_size=1, stride=1, padding=0),
                    )
        self.branch_1 = nn.Sequential(
                    nn.Conv2d(in_channels=n_channels, out_channels=q_channels, 
                              kernel_size=1, stride=1, padding=0),
                    nn.Conv2d(in_channels=q_channels, out_channels=q_channels, 
                              kernel_size=3, stride=1, padding=1)
                    )
        self.branch_2 = nn.Sequential(
                    nn.Conv2d(in_channels=n_channels, out_channels=q_channels, 
                              kernel_size=1, stride=1, padding=0),
                    nn.Conv2d(in_channels=q_channels, out_channels=q_channels, 
                              kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(in_channels=q_channels, out_channels=q_channels, 
                              kernel_size=3, stride=1, dilation=2, padding=2)
                    )
        self.branch_3 = nn.Sequential(
                    nn.Conv2d(in_channels=n_channels, out_channels=q_channels, 
                              kernel_size=1, stride=1, padding=0),
                    nn.Conv2d(in_channels=q_channels, out_channels=q_channels, 
                              kernel_size=5, stride=1, padding=2),
                    nn.Conv2d(in_channels=q_channels, out_channels=q_channels, 
                              kernel_size=3, stride=1, dilation=3, padding=3)
                    )

    def forward(self,x):
        x_0 = self.branch_0(x)
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)  
        out = torch.cat((x_0, x_1, x_2, x_3), 1)
        return out + x
    
if __name__=="__main__":
    device='cuda'
    # Define input parameters
    n_channels = 64
    kernel_size = 3

    # Create a sample input tensor
    batch_size = 1
    height, width = 32, 32  # Size of the input image
    input_tensor = torch.randn(batch_size, n_channels, height, width).to(device)

    block = RFBlock(n_channels=64).to(device)
    print(block(input_tensor).size())
    print(input_tensor.size())
