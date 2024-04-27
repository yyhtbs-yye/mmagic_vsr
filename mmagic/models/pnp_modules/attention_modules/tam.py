import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Misra, D., Nalamada, T., Arasanipalai, A.U. and Hou, Q., 2021. 
Rotate to attend: Convolutional triplet attention module. 
In Proceedings of the IEEE/CVF winter conference on applications of computer vision 
(pp. 3139-3148).
'''

class TAMBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.cw = CommonAttention()
        self.hc = CommonAttention()
        self.hw = CommonAttention()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        
        x_out = self.hw(x)
        x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        return x_out
    
class CommonAttention(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        u = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        u = self.conv(u)
        return x * F.sigmoid(u)

if __name__=="__main__":
    device='cuda'
    # Define input parameters
    n_channels = 64
    kernel_size = 3

    # Create a sample input tensor
    batch_size = 1
    height, width = 32, 32  # Size of the input image
    input_tensor = torch.randn(batch_size, n_channels, height, width).to(device)
    block = TAMBlock().to(device)
    print(block(input_tensor).size())
    print(input_tensor.size())
