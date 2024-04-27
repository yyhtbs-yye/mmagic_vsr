import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Li, C., Zhou, A. and Yao, A., 2022, October. 
Omni-Dimensional Dynamic Convolution. 
In International Conference on Learning Representations (ICLR). Spotlight Paper
'''

class ODConv2d(nn.Module):
    def __init__(self, n_channels, kernel_size=3, 
                 stride=1, padding=1, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super().__init__()
        self.n_channels = n_channels
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = MultiAttention(
                            n_channels, kernel_size,
                            reduction=reduction, 
                            kernel_num=kernel_num)
        
        self.weight = nn.Parameter(torch.randn(kernel_num, n_channels, n_channels//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def forward(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, spatial_attention, kernel_attention = self.attention(x)
        b, c, h, w = x.size()
        
        x = x * channel_attention

        x = x.reshape(1, -1, h, w)
        
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)

        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.n_channels // self.groups, self.kernel_size, self.kernel_size])
        
        output = F.conv2d(x, weight=aggregate_weight, 
                          bias=None, 
                          stride=self.stride, 
                          padding=self.padding,
                          dilation=self.dilation, 
                          groups=self.groups * b)
        output = output.view(b, self.n_channels, output.size(-2), output.size(-1))
        return output


class MultiAttention(nn.Module): # Group is not enabled in Multi-Attention, it needs to be fixed. 

    def __init__(self, n_channels, kernel_size, 
                 reduction=0.0625, kernel_num=4, 
                 min_channel=16):
        super().__init__()
        attention_channel = max(int(n_channels * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(n_channels, attention_channel, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.cab = nn.Conv2d(attention_channel, n_channels, 1, bias=True)
        self.sab = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
        self.kab = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.relu(x)

        # Compute channel attention directly in the forward method
        channel_attention = torch.sigmoid(self.cab(x).view(x.size(0), -1, 1, 1) / self.temperature)

        # Compute spatial attention directly in the forward method
        spatial_attention = self.sab(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)

        # Compute kernel attention directly in the forward method
        kernel_attention = self.kab(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)

        return channel_attention, spatial_attention, kernel_attention

if __name__=="__main__":
    device='cuda'
    # Define input parameters
    n_channels = 64
    kernel_size = 3

    # Create a sample input tensor
    batch_size = 1
    height, width = 32, 32  # Size of the input image
    input_tensor = torch.randn(batch_size, n_channels, height, width).to(device)
    block = ODConv2d(n_channels).to(device)
    print(block(input_tensor).size())
    print(input_tensor.size())
