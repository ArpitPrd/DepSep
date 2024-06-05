
def AdapConvMixer(dim=64, depth=10, kernel_size2d=3,  patch_size=1, g=4):
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
                    *[nn.Sequential(
                         Residual(
                          bandWiseConv(dim//g)),
                         pixelWiseConv(dim//g),
                     ) for j in range (4)],
                    )
                ),
                pointWiseConv(g),
            ) for i in range(depth)],
        nn.Conv3d(g,g,kernel_size=3, padding="same"),
        Concater(),
        nn.Conv2d(dim,31,patch_size,patch_size)
    )
