import torch
import torch.nn as nn

class bandwise_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups):
        super(bandwise_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, groups = groups, padding = "same"),
            nn.GELU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # x.shape = batch, in_channels, height, width
        ans = self.conv2d(x)
        return ans

class pixelwise_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(pixelwise_conv, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, padding = "same"),
            nn.GELU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.conv2d(x)
        return x

class pointwise_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(pointwise_conv, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, padding = "same"),
            nn.GELU(),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        x = self.conv3d(x)
        return x

class Method(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, block_size, depth):
        super(Method, self).__init__()
        self.block_size = block_size
        self.in_channels = in_channels
        self.depth = depth
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, groups = in_channels, padding = "same"),
            nn.GELU(),
            nn.BatchNorm3d(out_channels)
        )
        # TODO : keep same params for blocks
        self.bandwise = [[bandwise_conv(block_size, block_size, kernel_size, groups = block_size).cuda() for _ in range(block_size)] for __ in range(depth)]
        self.pixelwise = [[pixelwise_conv(block_size, block_size).cuda() for ___ in range(block_size)] for ____ in range(depth)]
        self.pointwise = [pointwise_conv(in_channels, out_channels).cuda() for _____ in range(depth)]
        
    def forward(self, x):
        # x.shape = batch, channels, height, width
        batch, channels, height, width = x.shape
        out = x
        x = x.reshape(batch, self.in_channels, self.block_size, height, width)
        x = self.conv3d(x)
        for depth_no in range(self.depth):
            y_list = []
            for block in range(self.in_channels):
                y = self.bandwise[depth_no][block](x[:, block, :, :, :])
                y = self.pixelwise[depth_no][block](y)
                y = torch.unsqueeze(y, dim=1)
                y_list.append(y)
            y = torch.cat(y_list, dim=1)
            x = x + y
            x = self.pointwise[depth_no](x)
        x = x.reshape(batch, channels, height, width)
        #x = x + out
        return x
