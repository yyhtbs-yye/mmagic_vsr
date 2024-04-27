import torch
import torch.nn as nn
'''
Yang, L., Zhang, R.Y., Li, L. and Xie, X., 2021, July. 
Simam: A simple, parameter-free attention module for convolutional neural networks. 
In International conference on machine learning (pp. 11863-11874). PMLR.
'''
class SIMAMBlock(nn.Module):
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.epsilon = epsilon

    def forward(self, x):
        b, c, h, w = x.size()
        
        num_elements = w * h - 1

        mean_centered_squared = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        
        denominator = 4 * (mean_centered_squared.sum(dim=[2, 3], keepdim=True) / num_elements + self.epsilon) + 0.5

        response = mean_centered_squared / denominator

        return x * self.sigmoid(response)

if __name__=="__main__":
    device='cuda'
    # Define input parameters
    n_channels = 64

    # Create a sample input tensor
    batch_size = 1
    height, width = 32, 32  # Size of the input image
    input_tensor = torch.randn(batch_size, n_channels, height, width).to(device)
    block = SIMAMBlock(n_channels).to(device)
    print(block(input_tensor).size())
    print(input_tensor.size())