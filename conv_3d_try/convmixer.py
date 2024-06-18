import torch
import torch.nn as nn
#from torchsummary import summary


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class BandwiseConv(nn.Module):
    def __init__(self, block_size, kernel_size, groups):
        super(BandwiseConv, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=block_size, out_channels=block_size, kernel_size=kernel_size, groups=groups, padding="same"),
            nn.BatchNorm2d(block_size),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv2d(x)

class PixelwiseConv(nn.Module):
    def __init__(self, block_size):
        super(PixelwiseConv, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=block_size, out_channels=block_size, kernel_size=1),
            nn.BatchNorm2d(block_size),
            nn.GELU()
            
        )

    def forward(self, x):
        return self.conv2d(x)

class PointwiseConv(nn.Module):
    def __init__(self, num_blocks):
        super(PointwiseConv, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels=num_blocks, out_channels=num_blocks, kernel_size=1),
            nn.BatchNorm3d(num_blocks),
            nn.GELU()
            
        )

    def forward(self, x):
        return self.conv3d(x)

class Core_Method(nn.Module):
    def __init__(self, num_blocks, block_size, kernel_size):
        super().__init__()
        self.channelwise = nn.ModuleList([
        	Residual(nn.Sequential(
        		BandwiseConv(block_size, kernel_size, groups=block_size),
        		PixelwiseConv(block_size)
        	))
        	for _ in range(num_blocks)
        ])
        
        self.pointwise = PointwiseConv(num_blocks)
        
    def forward(self, x):
        y_list = []
        for block in range(x.shape[1]):
            y = self.channelwise[block](x[:, block, :, : ,:])
            y = torch.unsqueeze(y, dim=1)
            y_list.append(y)
        y = torch.cat(y_list, dim = 1)
        y = self.pointwise(y)
        return y

class Method(nn.Module):
    def __init__(self, num_blocks, block_size, kernel_size,  depth):
        super(Method, self).__init__()
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.depth = depth
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels=num_blocks, out_channels=num_blocks, kernel_size=kernel_size, groups=num_blocks, padding="same"),
            nn.BatchNorm3d(num_blocks),
            nn.GELU()
            
        )

        self.core = nn.Sequential(*[
        	Core_Method(num_blocks, block_size, kernel_size)
        for _ in range(depth)])
        self.last_conv = nn.Conv2d(num_blocks * block_size, num_blocks * block_size, kernel_size = 1, padding = "same")


    def forward(self, x):
        batch, channels, height, width = x.shape
        x_org = x
        x = x.reshape(batch, self.num_blocks, self.block_size, height, width)
        x = self.conv3d(x)
        #print(x.shape)
        x = self.core(x)
        #x = self.conv(x)
        #print(x.shape)
        x = x.reshape(batch, channels, height, width)
        x = x + x_org
        x = self.last_conv(x)
        return x

if __name__ == '__main__':
    conv = Method(num_blocks=4, block_size=8, kernel_size=3,  depth=10)
    t = torch.ones((1, 32, 64, 64))
    print(conv(t).shape)
    #summary(conv, (32, 64, 64))
