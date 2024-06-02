#################### Imports ###########################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

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
#                 )))],
#         #pointwise conv with groups=2 meaning [BN,dim/groups,h,w]    - 14/7/'22
#         nn.Conv2d(dim, dim, kernel_size=1,groups=1),
#         nn.GELU(),
#         nn.BatchNorm2d(dim),
#         # ) for i in range(depth)],
#         #for change in arch from 32 to 31    - 14/07/'22
#         nn.Conv2d(dim,31,kernel_size=3, padding="same")
#     )
#
# model=AdapConvMixer(dim=64,depth=1,kernel_size=3,patch_size=1)
# print(summary(model,(31,64,64)))


######## Result-1 and Explanation ##########

#           Conv2d - 1                     [-1, 64, 64, 64]           2,048        (31*64*1*1)+64=2,048
#             GELU - 2                     [-1, 64, 64, 64]               0        requires no paramtere
#      BatchNorm2d - 3                     [-1, 64, 64, 64]             128        it takes 2 parameters with c length =64*2=128
#           Conv2d - 4                     [-1, 64, 64, 64]          2, 368        (4*4*16*3*3)+64=2,368
#             GELU - 5                     [-1, 64, 64, 64]               0        requires no parameters
#      BatchNorm2d - 6                     [-1, 64, 64, 64]             128        it takes 2 parameters with c length =64*2=128
#         Residual - 7                     [-1, 64, 64, 64]               0        requires no parameters
#           Conv2d - 8                     [-1, 64, 64, 64]            4,160       (64*64)+64=4160
#             GELU - 9                     [-1, 64, 64, 64]                0       requires no parameters
#     BatchNorm2d - 10                     [-1, 64, 64, 64]              128       it takes 2 parameters with c length =64*2=128
#          Conv2d - 11                     [-1, 31, 64, 64]           17,887       64*31*3*3 + 31=17,887
#
# Total params: 26,847
# Trainable params: 26,847
# Non-trainable params: 0
# Total params with depth=10: 87,903


################# Case-2 #######################

# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn
#
#     def forward(self, x):
#         return self.fn(x) + x
#
#
# def AdapConvMixer(dim, depth, kernel_size=9, patch_size=7):#(1*31*64*64)
#     return nn.Sequential(
#         nn.Conv3d(1, dim, kernel_size=(1,patch_size,patch_size), stride=(1,patch_size,patch_size)),#(64,31,64,64)
#         nn.GELU(),#(64,31,64,64)
#         nn.BatchNorm3d(dim),#(64,31,64,64)
#
#         *[nn.Sequential(
#                 Residual(nn.Sequential(#Adding the F(x) with x
#                     #add groups=16 or groups=16(meaning dim=BN,(dim/groups),H,W)#dw conv
#                     nn.Conv3d(dim, dim, kernel_size, groups=16, padding="same"),# Padding same ensures the same size # (64,31,64,64)
#                     nn.GELU(),
#                     nn.BatchNorm3d(dim)
#                 )),
#         # )],
#
#         nn.Conv3d(dim, dim, kernel_size=1,groups=1),#(64,31,64,64)
#         nn.GELU(),
#         nn.BatchNorm3d(dim),
#
#         ) for i in range(depth)],# (64,31,64,64)
#         nn.Conv3d(dim,1,kernel_size=3, padding="same")#(1,31,64,64)
#     )
#
# model=AdapConvMixer(dim=64,depth=1,kernel_size=3,patch_size=1)
# print(summary(model,(1,31,64,64)))

################## Result-2 and Explanation #####################

#             Conv3d-1       [-1, 64, 31, 64, 64]             128          1*1*1*64+64=128
#               GELU-2       [-1, 64, 31, 64, 64]               0          requires no Parameter
#        BatchNorm3d-3       [-1, 64, 31, 64, 64]             128          it takes 2 parameters with c length =64*2=128
#             Conv3d-4       [-1, 64, 31, 64, 64]           6,976          3*3*3*4*4*16+64=6976
#               GELU-5       [-1, 64, 31, 64, 64]               0          requires no Parameter
#        BatchNorm3d-6       [-1, 64, 31, 64, 64]             128          it takes 2 parameters with c length =64*2=128
#           Residual-7       [-1, 64, 31, 64, 64]               0          requires no Parameter
#             Conv3d-8       [-1, 64, 31, 64, 64]           4,160          1*1*1*64*64+64=4160
#               GELU-9       [-1, 64, 31, 64, 64]               0          requires no Parameter
#       BatchNorm3d-10       [-1, 64, 31, 64, 64]             128          it takes 2 parameters with c length =64*2=128
#            Conv3d-11       [-1, 1, 31, 64, 64]            1,729          3*3*3*64+1=1729
#
# Total params: 13,377
# Trainable params: 13,377
# Non-trainable params: 0
# Total params with depth=10: 115,905


################ Case-3 ####################

# class Grouper(nn.Module):
#     def __init__(self, g):
#         super(Grouper, self).__init__()
#         self.g = g
#
#     def forward(self, x):
#         if x.dim() == 4:  # (b, c, h, w)
#             b, c, h, w = x.shape
#             x = x.reshape(b, self.g, c // self.g, h, w)
#         return x
#
#
# class Concater(nn.Module):
#     def __init__(self):
#         super(Concater, self).__init__()
#
#     def forward(self, x):
#         if x.dim() == 5:
#             b, g, c, h, w = x.shape
#             x = x.reshape(b, g*c, h, w)
#         return x
#
# class bandWiseConv(nn.Module):
#     def __init__(self,c):
#         super(bandWiseConv, self).__init__()
#         self.conv2dDepSep = nn.Sequential(nn.Conv2d(c, c, 3, groups=c, padding="same"), nn.GELU(), nn.BatchNorm2d(c))
#
#     def forward(self, x):
#         b, g, c, h, w = x.shape
#         output = torch.zeros_like(x)
#         for j in range(g):
#             output[:, j] = self.conv2dDepSep(x[:, j])
#         return output
#
# class pixelWiseConv(nn.Module):
#     def __init__(self,c):
#         super(pixelWiseConv, self).__init__()
#         self.conv2dDepSep = nn.Sequential(nn.Conv2d(c, c, 1, padding="same"), nn.GELU(), nn.BatchNorm2d(c))
#
#     def forward(self, x):
#         b, g, c, h, w = x.shape
#         output = torch.zeros_like(x)
#         for j in range(g):
#             output[:, j] = self.conv2dDepSep(x[:, j])
#         return output
#
# class pointWiseConv(nn.Module):
#     def __init__(self,g):
#         super(pointWiseConv, self).__init__()
#         self.conv3d = nn.Sequential(nn.Conv3d(g, g, 1, padding="same"), nn.GELU(), nn.BatchNorm3d(g))
#
#     def forward(self, x):
#         b, g, c, h, w = x.shape
#         output = self.conv3d(x)
#         return output
#
# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn
#
#     def forward(self, x):
#         print('hello')
#         return self.fn(x) + x
#
# def depSep3d(dim=64, depth=10, kernel_size2d=3,  patch_size=1, g=4):
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
#                         bandWiseConv(64//g),
#                         pixelWiseConv(64//g),
#                         Concater()
#                     )
#                 ),
#                 Grouper(g),
#                 pointWiseConv(g),
#                 Concater()
#             ) for i in range(depth)],
#         nn.Conv2d(dim,31,kernel_size=3, padding="same")
#     )
#
# model=depSep3d(64,1,kernel_size2d=3,patch_size=1,g=4)
# print(summary(model,(31,64,64)))

################### Result-3 and Explanation #############



#         Conv2d-1           [-1, 64, 64, 64]           2,048           31*64+64=2,048
#           GELU-2           [-1, 64, 64, 64]               0           requires no Parameter
#    BatchNorm2d-3           [-1, 64, 64, 64]             128           64*2=128
#        Grouper-4        [-1, 4, 16, 64, 64]               0           requires no Parameter
#         Conv2d-5           [-1, 16, 64, 64]             160           16*3*3+16=160
#           GELU-6           [-1, 16, 64, 64]               0           requires no Parameter
#    BatchNorm2d-7           [-1, 16, 64, 64]              32           16*2=32
#         Conv2d-8           [-1, 16, 64, 64]             160           //repeat 4 times
#           GELU-9           [-1, 16, 64, 64]               0
#   BatchNorm2d-10           [-1, 16, 64, 64]              32
#        Conv2d-11           [-1, 16, 64, 64]             160
#          GELU-12           [-1, 16, 64, 64]               0
#   BatchNorm2d-13           [-1, 16, 64, 64]              32
#        Conv2d-14           [-1, 16, 64, 64]             160
#          GELU-15           [-1, 16, 64, 64]               0
#   BatchNorm2d-16           [-1, 16, 64, 64]              32
#  bandWiseConv-17        [-1, 4, 16, 64, 64]               0
#        Conv2d-18           [-1, 16, 64, 64]             272           16*16+16=272
#          GELU-19           [-1, 16, 64, 64]               0           requires no Parameter
#   BatchNorm2d-20           [-1, 16, 64, 64]              32           16*2=32
#        Conv2d-21           [-1, 16, 64, 64]             272           //repeat 4 times
#          GELU-22           [-1, 16, 64, 64]               0
#   BatchNorm2d-23           [-1, 16, 64, 64]              32
#        Conv2d-24           [-1, 16, 64, 64]             272
#          GELU-25           [-1, 16, 64, 64]               0
#   BatchNorm2d-26           [-1, 16, 64, 64]              32
#        Conv2d-27           [-1, 16, 64, 64]             272
#          GELU-28           [-1, 16, 64, 64]               0
#   BatchNorm2d-29           [-1, 16, 64, 64]              32
# pixelWiseConv-30        [-1, 4, 16, 64, 64]               0
#      Concater-31           [-1, 64, 64, 64]               0
#      Residual-32           [-1, 64, 64, 64]               0
#       Grouper-33        [-1, 4, 16, 64, 64]               0
#        Conv3d-34        [-1, 4, 16, 64, 64]              20           1*1*1*4*4+4=20
#          GELU-35        [-1, 4, 16, 64, 64]               0           requires no Parameter
#   BatchNorm3d-36        [-1, 4, 16, 64, 64]               8           4*2=8
# pointWiseConv-37        [-1, 4, 16, 64, 64]               0           requires no Parameter
#      Concater-38           [-1, 64, 64, 64]               0           requires no Parameter
#        Conv2d-39           [-1, 31, 64, 64]          17,887           64*31*3*3+31=17887

# Total params: 22,075
# Trainable params: 22,075
# Non-trainable params: 0
# Total params with depth=10: 40,183
