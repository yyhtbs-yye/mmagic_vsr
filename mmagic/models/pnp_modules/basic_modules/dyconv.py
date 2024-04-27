import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, ratios, temperature, init_weight=True):
        super().__init__()
        assert temperature % 3 == 1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_channels != 3:
            n_channels = int(in_channels * ratios) + 1
        else:
            n_channels = out_channels
        self.fc1 = nn.Conv2d(in_channels, n_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(n_channels, out_channels, 1, bias=True)
        self.temperature = temperature

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def t_decay(self):
        if self.temperature != 1:
            self.temperature -= 3

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


class DyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size=1, ratios=0.25, stride=1, 
                 padding=0, dilation=1, groups=1, 
                 bias=True, n_channels=4 ,temperature=34, 
                 init_weight=True):
        
        super().__init__()
        assert in_channels % groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.n_channels = n_channels
        self.bias = bias

        # 
        self.attention = ChannelAttention(in_channels, n_channels, ratios, temperature)

        self.weight = nn.Parameter(torch.randn(n_channels, out_channels, 
                                               in_channels//groups, 
                                               kernel_size, kernel_size), 
                                               requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(n_channels, out_channels))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    
    def _initialize_weights(self):
        for i in range(self.n_channels):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.t_decay()

    def forward(self, x):
        # Treat batch as a dimensional variable and perform group convolution, 
        # because the weights of group convolutions are different, and the 
        # weights of dynamic convolutions are also different.
        softmax_attention = self.attention(x)
        b, c, h, w = x.size()

        # Change into one dimension for group convolution
        x = x.view(1, -1, h, w) 
        weight = self.weight.view(self.n_channels, -1)

        # Generation of dynamic convolution weights, batch_size convolution 
        # parameters are generated (each parameter is different)
        aggregate_weight = torch.mm(softmax_attention, weight).view(b * self.out_channels, 
                                                                    self.in_channels // self.groups, 
                                                                    self.kernel_size, self.kernel_size)
        
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, 
                                      self.bias).view(-1)
            
            output = F.conv2d(x, weight=aggregate_weight, 
                              bias=aggregate_bias, 
                              stride=self.stride, 
                              padding=self.padding,
                              dilation=self.dilation, 
                              groups=self.groups * b)
        else:
            output = F.conv2d(x, weight=aggregate_weight, 
                              bias=None, 
                              stride=self.stride, 
                              padding=self.padding,
                              dilation=self.dilation, 
                              groups=self.groups * b)

        output = output.view(b, self.out_channels, output.size(-2), output.size(-1))
        return output

if __name__=="__main__":
    device='cuda'
    # Define input parameters
    in_channels = 64
    out_channels = 128
    kernel_size = 3

    # Create a sample input tensor
    batch_size = 1
    height, width = 32, 32  # Size of the input image
    input_tensor = torch.randn(batch_size, in_channels, height, width).to(device)
    block = DyConv2d(in_channels, out_channels).to(device)
    print(block(input_tensor).size())
    print(input_tensor.size())
