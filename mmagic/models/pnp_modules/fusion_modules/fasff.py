import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SpectralNorm
'''
Li, X., Li, W., Ren, D., Zhang, H., Wang, M. and Zuo, W., 2020. 
Enhanced blind face restoration with multi-exemplar images and adaptive spatial feature fusion. 
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 
(pp. 2706-2715).
'''
class FASFFBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.MaskModel = nn.Sequential(
            SpectralNorm(nn.Conv2d(n_channels // 2 * 3, n_channels, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_channels, n_channels, 3, 1, 1, bias=False),
        )
        self.MaskConcat = nn.Sequential(
            SpectralNorm(nn.Conv2d(n_channels, n_channels //2 , 1, 1)),
        )
        self.DegradedModel = nn.Sequential(
            SpectralNorm(nn.Conv2d(n_channels, n_channels, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(n_channels, n_channels, 3, 1, 1)),
        )
        self.DegradedConcat = nn.Sequential(
            SpectralNorm(nn.Conv2d(n_channels, n_channels //2 , 1, 1)),
        )
        self.RefModel = nn.Sequential(
            SpectralNorm(nn.Conv2d(n_channels,n_channels,3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(n_channels, n_channels, 3, 1, 1)),
        )
        self.RefConcat = nn.Sequential(
            SpectralNorm(nn.Conv2d(n_channels, n_channels //2 , 1, 1)),
        )

    def forward(self, x0, x1, x2):
        #x0 feature, x1: Ref Feature, x2: MaskFeature
        MaskC = x2
        DegradedF = self.DegradedModel(x0)
        RefF = self.RefModel(x1)

        DownMask = self.MaskConcat(MaskC)
        DownDegraded = self.DegradedConcat(DegradedF)
        DownRef = self.RefConcat(RefF)

        ConcatMask = torch.cat((DownMask,DownDegraded,DownRef),1)
        MaskF = self.MaskModel(ConcatMask)

        return DegradedF + (RefF - DegradedF) * MaskF

if __name__=="__main__":
    device='cuda'
    # Define input parameters
    n_channels = 64
    kernel_size = 3

    # Create a sample input tensor
    batch_size = 1
    height, width = 32, 32  # Size of the input image
    input_tensor = torch.randn(batch_size, n_channels, height, width).to(device)
    block = FASFFBlock(n_channels).to(device)
    print(block(input_tensor, input_tensor, input_tensor).size())
    print(input_tensor.size())
