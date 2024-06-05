####################  Case-1 ###########################

# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn
#
#     def forward(self, x):
#         return self.fn(x) + x
#
#
# def AdapConvMixer(dim, depth, kernel_size=9, patch_size=7):
#     return nn.Sequential(
#         nn.Conv2d(31, dim, kernel_size=patch_size, stride=patch_size),
#         nn.GELU(),
#         nn.BatchNorm2d(dim),
#         *[nn.Sequential(
#                 Residual(nn.Sequential(
#                     #add groups=16 or groups=16(meaning dim=BN,(dim/groups),H,W)#dw conv
#                     nn.Conv2d(dim, dim, kernel_size, groups=16, padding="same"),
#                     nn.GELU(),
#                     nn.BatchNorm2d(dim)
#                 # )))],
#         #pointwise conv with groups=2 meaning [BN,dim/groups,h,w]    - 14/7/'22
#         nn.Conv2d(dim, dim, kernel_size=1,groups=1),
#         nn.GELU(),
#         nn.BatchNorm2d(dim),
#         ) for i in range(depth)],
#         #for change in arch from 32 to 31    - 14/07/'22
#         nn.Conv2d(dim,31,kernel_size=3, padding="same")
#     )
#
# model=AdapConvMixer(dim=64,depth=1,kernel_size=3,patch_size=1)
# print(summary(model,(31,64,64)))



import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Grouper(nn.Module):
    def __init__(self, g):
        super(Grouper, self).__init__()
        self.g = g

    def forward(self, x):
        if x.dim() == 4:  # (b, c, h, w)
            b, c, h, w = x.shape
            x = x.reshape(b, self.g, c // self.g, h, w)
        return x


class Concater(nn.Module):
    def __init__(self):
        super(Concater, self).__init__()

    def forward(self, x):
        if x.dim() == 5:
            b, g, c, h, w = x.shape
            x = x.reshape(b, g*c, h, w)
        return x

class bandWiseConv(nn.Module):
    def __init__(self,c):
        super(bandWiseConv, self).__init__()
        self.conv2dDepSep = nn.Sequential(nn.Conv2d(c, c, 3, groups=c, padding="same"), nn.GELU(), nn.BatchNorm2d(c))

    def forward(self, x):
        b, g, c, h, w = x.shape
        output = torch.zeros_like(x)
        for j in range(g):
            output[:, j] = self.conv2dDepSep(x[:, j])
        return output

class pixelWiseConv(nn.Module):
    def __init__(self,c):
        super(pixelWiseConv, self).__init__()
        self.conv2dDepSep = nn.Sequential(nn.Conv2d(c, c, 1, padding="same"), nn.GELU(), nn.BatchNorm2d(c))

    def forward(self, x):
        b, g, c, h, w = x.shape
        output = torch.zeros_like(x)
        for j in range(g):
            output[:, j] = self.conv2dDepSep(x[:, j])
        return output

class pointWiseConv(nn.Module):
    def __init__(self,g):
        super(pointWiseConv, self).__init__()
        self.conv3d = nn.Sequential(nn.Conv3d(g, g, 1, padding="same"), nn.GELU(), nn.BatchNorm3d(g))

    def forward(self, x):
        b, g, c, h, w = x.shape
        output = self.conv3d(x)
        return output

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

# def AdapConvMixer1(dim=64, depth=10, kernel_size2d=3,  patch_size=1, g=4):
#     return nn.Sequential(
#         nn.Conv2d(31, dim, kernel_size=patch_size, stride=patch_size),
#         nn.GELU(),
#         nn.BatchNorm2d(dim),
#
#         *[
#             nn.Sequential(
#                 Residual(
#                     nn.Sequential(
#                         Grouper(g),
#                         bandWiseConv(dim//g),
#                         pixelWiseConv(dim//g),
#                         Concater()
#                     )
#                 ),
#                 Grouper(g),
#                 pointWiseConv(g),
#                 Concater()
#             ) for i in range(depth)],
#         nn.Conv2d(dim,31,kernel_size=3, padding="same")
#     )

import numpy as np
#
##Changes i have made - i have included a extra residual, as the conv2D dep sep included the conv2D only in starting i have included conv 3D in the starting in the program
## using the conv 3-D has removed all the grouper and concaters
## i have reduced the dimension so that as per the previous code there are only 4 groups
## i have only included a residual on the bandwise conv as the conv 2-D we a residual on every depsep


# def AdapConvMixer2(dim=4, depth=10, kernel_size2d=3,  patch_size=1, g=4):
#     return nn.Sequential(
#         nn.Conv3d(1, dim, kernel_size=patch_size, stride=patch_size),
#         nn.GELU(),
#         nn.BatchNorm3d(dim),
#         *[
#             nn.Sequential(
#                 Residual(
#                     nn.Sequential(
#                          Residual(
#                           bandWiseConv(31)),
#                          pixelWiseConv(31),
#                      )
#                 ),
#                 pointWiseConv(4),
#             ) for i in range(depth)],
#         nn.Conv3d(dim,1,kernel_size=3, padding="same")
#     )
# model=AdapConvMixer2(4,1,kernel_size2d=3,patch_size=1,g=4)
# inputs=np.random.randn(1,1,31,64,64)
# inputs = torch.tensor(inputs, dtype=torch.float32)
#
# outputs=model(inputs)
# print(outputs.shape)
#
#
# print(summary(model,(1,31,64,64)))

## 60 thousand parameters


import numpy as np

## As the channels in the previous code was increased from the 31 to 16 , it has increased the parameters a lot then i have made
## Again the use of Grouper and Concater to keep the channels 16 but i have introduced the conv 3-D and extra residual in between.




def AdapConvMixer2(dim=64, depth=10, kernel_size2d=3,  patch_size=1, g=4):
    return nn.Sequential(
        nn.Conv2d(31,dim,patch_size,patch_size),
        Grouper(g),
        nn.Conv3d(g, g, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm3d(g),
        *[
            nn.Sequential(
                Residual(
                    nn.Sequential(
                         Residual(
                          bandWiseConv(dim//g)),
                         pixelWiseConv(dim//g),
                     )
                ),
                pointWiseConv(g),
            ) for i in range(depth)],
        nn.Conv3d(g,g,kernel_size=3, padding="same"),
        Concater(),
        nn.Conv2d(dim,31,patch_size,patch_size)
    )

model=AdapConvMixer2(64,10,kernel_size2d=3,patch_size=1,g=4)
inputs=np.random.randn(1,31,64,64)
inputs = torch.tensor(inputs, dtype=torch.float32)

outputs=model(inputs)
print(outputs.shape)

print(summary(model,(31,64,64)))

# 24 thousand parameters














