import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x
    
    
#Convmixer with change in architecture(has group-size in dwc,groups in pwc,and conv2d at the last layer to change 32 dim to 31) (this now also has residual_x)     -    15/7/'22    

#this AdapConvMixer no residual(skip) 
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
#                 )),
#                 #pointwise conv with groups=2 meaning [BN,dim/groups,h,w]    - 14/7/'22
#                 nn.Conv2d(dim, dim, kernel_size=1,groups=1),
#                 nn.GELU(),
#                 nn.BatchNorm2d(dim)
#         ) for i in range(depth)],
#         #for change in arch from 32 to 31    - 14/07/'22
#         nn.Conv2d(dim,31,kernel_size=3, padding="same")
#     )


#Basic AdapConvMixer and with residual
# def AdapConvMixer(dim, depth, kernel_size=9, patch_size=7):
#     return Residual_x(nn.Sequential(
#         nn.Conv2d(31, dim, kernel_size=patch_size, stride=patch_size),
#         nn.GELU(),
#         nn.BatchNorm2d(dim),
#         *[nn.Sequential(
#                 Residual(nn.Sequential(
#                     #add groups=16 or groups=16(meaning dim=BN,(dim/groups),H,W)#dw conv
#                     nn.Conv2d(dim, dim, kernel_size, groups=32, padding="same"),
#                     nn.GELU(),
#                     nn.BatchNorm2d(dim)
#                 )),
#                 #pointwise conv with groups=2 meaning [BN,dim/groups,h,w]    - 14/7/'22
#                 nn.Conv2d(dim, dim, kernel_size=1,groups=1),
#                 nn.GELU(),
#                 nn.BatchNorm2d(dim)
#         ) for i in range(depth)],
#         #for change in arch from 32 to 31    - 14/07/'22
#         nn.Conv2d(dim,31,kernel_size=3, padding="same")),
#         patch_size
#     ) 


#   NO DWC  (with Inter-pixel sub-block)
def AdapConvMixer(dim, depth, kernel_size=9, patch_size=7):
    return nn.Sequential(
        nn.Conv2d(31, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    #add groups=16 or groups=16(meaning dim=BN,(dim/groups),H,W)#dw conv
                    nn.Conv2d(dim, dim, kernel_size, groups=16, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )))],
        #pointwise conv with groups=2 meaning [BN,dim/groups,h,w]    - 14/7/'22
        nn.Conv2d(dim, dim, kernel_size=1,groups=1),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        # ) for i in range(depth)],
        #for change in arch from 32 to 31    - 14/07/'22
        nn.Conv2d(dim,31,kernel_size=3, padding="same")
    )

############## 3d #################

# def AdapConvMixer(dim, depth, kernel_size=9, patch_size=7):#(1*31*64*64)
#     return nn.Sequential(
#         nn.Conv3d(1, dim, kernel_size=(1,patch_size,patch_size), stride=(1,patch_size,patch_size)),#(64,31,64,64)
#         nn.GELU(),#(64,31,64,64)
#         nn.BatchNorm3d(dim),#(64,31,64,64)

#         *[nn.Sequential(
#                 Residual(nn.Sequential(#Adding the F(x) with x
#                     #add groups=16 or groups=16(meaning dim=BN,(dim/groups),H,W)#dw conv
#                     nn.Conv3d(dim, dim, kernel_size, groups=16, padding="same"),# Padding same ensures the same size # (64,31,64,64)
#                     nn.GELU(),
#                     nn.BatchNorm3d(dim)
#                 )),
#         # )],
#         nn.Conv3d(dim, dim, kernel_size=1,groups=1),#(64,31,64,64)
#         nn.GELU(),
#         nn.BatchNorm3d(dim),
#         ) for i in range(depth)],# (64,31,64,64)
#         nn.Conv3d(dim,1,kernel_size=3, padding="same")#(1,31,64,64)
#     )

####################################

#AdaptiveMixer with residual removed from dwc 
# def AdapConvMixer(dim, depth, kernel_size=9, patch_size=7):
#     return Residual_x(nn.Sequential(
#         nn.Conv2d(31, dim, kernel_size=patch_size, stride=patch_size),
#         nn.GELU(),
#         nn.BatchNorm2d(dim),
#         *[nn.Sequential(
#                 nn.Sequential(
#                     #add groups=16 or groups=16(meaning dim=BN,(dim/groups),H,W)#dw conv
#                     nn.Conv2d(dim, dim, kernel_size, groups=32, padding="same"),
#                     nn.GELU(),
#                     nn.BatchNorm2d(dim)
#                 ),
#                 #pointwise conv with groups=2 meaning [BN,dim/groups,h,w]    - 14/7/'22
#                 nn.Conv2d(dim, dim, kernel_size=1,groups=1),
#                 nn.GELU(),
#                 nn.BatchNorm2d(dim)
#         ) for i in range(depth)],
#         #for change in arch from 32 to 31    - 14/07/'22
#         nn.Conv2d(dim,31,kernel_size=3, padding="same")),
#         patch_size
#     ) 

#AdaptiveMixer with Intra-channel sub-block
# def AdapConvMixer(dim, depth, kernel_size=9, patch_size=7):
#     return Residual_x(nn.Sequential(
#         nn.Conv2d(31, dim, kernel_size=patch_size, stride=patch_size),
#         nn.GELU(),
#         nn.BatchNorm2d(dim),
#         *[nn.Sequential(       #pointwise conv with groups=2 meaning [BN,dim/groups,h,w]    - 14/7/'22
#                 nn.Conv2d(dim, dim, kernel_size=1,groups=1),
#                 nn.GELU(),
#                 nn.BatchNorm2d(dim)
#            ) for i in range(depth)],
#         #for change in arch from 32 to 31    - 14/07/'22
#         nn.Conv2d(dim,31,kernel_size=3, padding="same")),
#         patch_size
#     ) 

# RELU instead of GELU
# def AdapConvMixer(dim, depth, kernel_size=9, patch_size=7):
#     return Residual_x(nn.Sequential(
#         nn.Conv2d(31, dim, kernel_size=patch_size, stride=patch_size),
#         nn.ReLU(),
#         nn.BatchNorm2d(dim),
#         *[nn.Sequential(
#                 Residual(nn.Sequential(
#                     #add groups=16 or groups=16(meaning dim=BN,(dim/groups),H,W)#dw conv
#                     nn.Conv2d(dim, dim, kernel_size, groups=16, padding="same"),
#                     nn.ReLU(),
#                     nn.BatchNorm2d(dim)
#                 )),
#                 #pointwise conv with groups=2 meaning [BN,dim/groups,h,w]    - 14/7/'22
#                 nn.Conv2d(dim, dim, kernel_size=1,groups=1),
#                 nn.ReLU(),
#                 nn.BatchNorm2d(dim)
#         ) for i in range(depth)],
#         #for change in arch from 32 to 31    - 14/07/'22
#         nn.Conv2d(dim,31,kernel_size=3, padding="same")),
#         patch_size
#     ) 

############use this for patch_size =1##################
class Residual_x(nn.Module):
    def __init__(self, fn, patch_size):
        super().__init__()
        self.fn = fn
        self.patch_size = patch_size
        #self.size = 64
        
    def forward(self, x):
#         out_w = x.size(dim=3)/self.patch_size
#         out_h = x.size(dim=2)/self.patch_size
#         print('x shape',x.shape)
#         intw = int(out_w)
#         inth = int(out_h)
#         #out = F.interpolate(x, size=ints)
#         out = F.interpolate(x, size=[intw,inth])
        #print(out.shape)
        out = x
        return self.fn(x) + out