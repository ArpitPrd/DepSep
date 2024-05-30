import torch
import torch.nn as nn
import torch.nn.functional as F

class Grouper(nn.Module):
    def __init__(self, g):
        super(Grouper, self).__init__()
        self.g = g

    def forward(self, x):
        if x.dim() == 4:  # (b, c, h, w)
            b, c, h, w = x.shape
            x = x.reshape(b, self.g, c // self.g, h, w)
        elif x.dim() == 3:  # (c, h, w)
            c, h, w = x.shape
            x = x.reshape(self.g, c // self.g, h, w)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        return x

class Concater(nn.Module):
    def __init__(self):
        super(Concater, self).__init__()
        pass

    def forward(self, x):
        if x.dim() == 4:
            g, c, h, w = x.shape
            x = x.reshape(g*c, h, w)
        elif x.dim() == 5:
            b, g, c, h, w = x.shape
            x = x.reshape(b, g*c, h, w)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        return x

class bandWiseConv(nn.Module):
    def __init__(self):
        super(bandWiseConv, self).__init__()

    def forward(self, x):
        requires_grad = x.requires_grad
        if x.dim() == 5:  # (b, g, c//g, h, w)
            b, g, c, h, w = x.shape
            conv2dDepSep = nn.Sequential(nn.Conv2d(c, c, 3, groups=c, padding="same"), nn.GELU(), nn.BatchNorm2d(c)).to(x.device)
            output = torch.zeros_like(x)
            for i in range(b):
                for j in range(g):
                    output[i, j] = torch.squeeze(conv2dDepSep(torch.unsqueeze(x[i, j], 0)), 0)
        elif x.dim() == 4:  # (g, c//g, h, w)
            g, c, h, w = x.shape
            conv2dDepSep = nn.Sequential(nn.Conv2d(c, c, 3, groups=c, padding="same"), nn.GELU(), nn.BatchNorm2d(c)).to(x.device)
            output = torch.zeros_like(x)
            for i in range(g):
                output[i] = torch.squeeze(conv2dDepSep(torch.unsqueeze(x[i], 0)), 0)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        output.requires_grad_(requires_grad)
        return output

class pixelWiseConv(nn.Module):
    def __init__(self):
        super(pixelWiseConv, self).__init__()

    def forward(self, x):
        requires_grad = x.requires_grad
        if x.dim() == 5:
            b, g, c, h, w = x.shape
            conv2dDepSep =nn.Sequential(nn.Conv2d(c, c, 1, padding="same"), nn.GELU(), nn.BatchNorm2d(c)).to(x.device)
            output = torch.zeros_like(x)
            for i in range(b):
                for j in range(g):
                    output[i, j] = torch.squeeze(conv2dDepSep(torch.unsqueeze(x[i, j], 0)), 0)
        elif x.dim() == 4:
            g, c, h, w = x.shape
            conv2dDepSep =nn.Sequential(nn.Conv2d(c, c, 1, padding="same"), nn.GELU(), nn.BatchNorm2d(c)).to(x.device)
            output = torch.zeros_like(x)
            for i in range(g):
                output[i] = torch.squeeze(conv2dDepSep(torch.unsqueeze(x[i], 0)), 0)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        output.requires_grad_(requires_grad)
        return output

class pointWiseConv(nn.Module):
    def __init__(self):
        super(pointWiseConv, self).__init__()

    def forward(self, x):
        requires_grad = x.requires_grad
        if x.dim() == 5:
            b, g, c, h, w = x.shape
            conv3d = nn.Sequential(nn.Conv3d(g, g, 1, padding="same"), nn.GELU(), nn.BatchNorm3d(g)).to(x.device)
            output = conv3d(x)

        elif x.dim() == 4:
            g, c, h, w = x.shape
            conv3d = nn.Sequential(nn.Conv3d(g, g, 1, padding="same"), nn.GELU(), nn.BatchNorm3d(g)).to(x.device)
            output = conv3d(x)

        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        output.requires_grad_(requires_grad)
        return output

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        y = self.fn(x)
        print("x, y", x.shape, y.shape)
        return y + x

def depSep3d(dim=64, depth=10, kernel_size2d=3, kernel_size3d=3, patch_size=1, g=4):
    return nn.Sequential(
        nn.Conv2d(31, dim, kernel_size=kernel_size2d, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[
            nn.Sequential(
                Residual(
                    nn.Sequential(
                        Grouper(g),
                        bandWiseConv(),
                        pixelWiseConv(),
                        Concater()
                    )
                ),
                Grouper(g),
                pointWiseConv(),
                Concater()
            ) for i in range(depth)],
        nn.Conv2d(dim,31,kernel_size=3, padding="same")
    )

# f = nn.Sequential(
#     Grouper(3),
#     bandWiseConv(),
#     pixelWiseConv(),
#     pointWiseConv(),
#     Concater()
# )

x = torch.ones((1, 31, 64, 64), requires_grad=True)
f = depSep3d(64, 1, 3, 3, 1, 16)
y = f(x)
print(y.shape)