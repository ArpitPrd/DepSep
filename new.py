import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class BandWiseConv(nn.Module):
    def __init__(self, dim):
        super(BandWiseConv, self).__init__()
        self.seq = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=5, padding="same", groups = dim),
                                 nn.GELU(),
                                 nn.BatchNorm2d(dim)
                                )

    def forward(self, x):
        ans = torch.zeros_like(x)
        for i in range(x.shape[1]):
            ans[:, i, :, :, :] = self.seq(x[:, i, :, :, :])
        return ans


class PixelWiseConv(nn.Module):
    def __init__(self, dim):
        super(PixelWiseConv, self).__init__()
        self.seq = nn.Sequential(nn.Conv2d(dim, dim, kernel_size= (1, 5), padding="same", groups = dim),
                                 nn.GELU(),
                                 nn.BatchNorm2d(dim)
                                )
        
    def forward(self, x):
        x = torch.transpose(x, 2, 4)
        ans = torch.zeros_like(x)
        for i in range(x.shape[1]):
            ans[:, i, :, :, :] = self.seq(x[:, i, :, :, :])
        ans = torch.transpose(ans, 2, 4)
        return ans

class ChannelWiseConv(nn.Module):
    def __init__(self, shape):
        super(ChannelWiseConv, self).__init__()
        self.shape = shape
        self.seq = nn.Sequential(
            BandWiseConv(self.shape[2]),
            PixelWiseConv(self.shape[-1]),
        )

    def forward(self, x):
        output = self.seq(x)
        return output

class PointWiseConv(nn.Module):
    def __init__(self, dim):
        super(PointWiseConv, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, padding="same"),
            nn.GELU(),
            nn.BatchNorm3d(dim)
        )

    def forward(self, x):
        output = self.seq(x)
        return output


class Conv3d_depsep(nn.Module):
    def __init__(self, shape, depth):
        super(Conv3d_depsep, self).__init__()
        self.shape = shape
        self.seq = nn.Sequential(
            *[
                nn.Sequential(
                    Residual(ChannelWiseConv(self.shape)),
                    PointWiseConv(self.shape[1])
                )
                for _ in range(depth)
            ]
        )
    
    def forward(self, x):
        inp = x.reshape(self.shape)
        output = self.seq(inp)
        output = output.reshape(x.shape)
        return output

## Enter the shape of grouped image with batch - also let all the default parmaters same
## batch size = 32 set
## kernel_size = 5
## patch_size not required
def AdapConvMixer(dim = 64, kernel_size = 5, shape = (32, 16, 4, 64, 64), depth = 10):
    return nn.Sequential(
        nn.Conv2d(31, dim, kernel_size=5, padding = "same"),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        Conv3d_depsep(shape, depth),
        nn.Conv2d(dim,31,kernel_size=3, padding="same")
    )

# def test():
#     t = torch.ones((32, 31, 64, 64))
#     conv = AdapConvMixer()
#     print(conv(t))

# test()