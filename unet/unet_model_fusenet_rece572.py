""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts_fusenet_rece572 import *
class UNet_fusenet_rece572(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_fusenet_rece572, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.down5 = Down(1024, 1024)
        self.up1 = Up(2048, 512, bilinear)
        self.up2 = Up(1024, 256, bilinear)
        self.up3 = Up(512, 128, bilinear)
        self.up4 = Up(256, 64, bilinear)
        self.up5 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        # self.up1 = Up(2048, 512, bilinear)
        # self.up2 = Up(1024, 256, bilinear)
        # self.up3 = Up(512, 128, bilinear)
        # self.up4 = Up(256, 128, bilinear)
        # self.outc = OutConv(128, n_classes)

        # Backup
        # self.up1 = Up(1024, 256, bilinear)
        # self.up2 = Up(512, 128, bilinear)
        # self.up3 = Up(256, 64, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)

    def forward(self, x, y):
        y1 = self.inc(y)
        y2, idy2 = self.down1(y1)
        y3, idy3 = self.down2(y2)
        y4, idy4 = self.down3(y3)
        y5, idy5 = self.down4(y4)
        y6, idy6 = self.down5(y5)

        x1 = self.inc(x)
        x1_a = torch.add(x1, y1)
        x1_a = torch.div(x1_a, 2)
        x2, idx2 = self.down1(x1_a)
        x2_a = torch.add(x2, y2)
        x2_a = torch.div(x2_a, 2)
        x3, idx3 = self.down2(x2_a)
        x3_a = torch.add(x3, y3)
        x3_a = torch.div(x3_a, 2)
        x4, idx4 = self.down3(x3_a)
        x4_a = torch.add(x4, y4)
        x4_a = torch.div(x4_a, 2)
        x5, idx5 = self.down4(x4_a)
        x5_a = torch.add(x5, y5)
        x5_a = torch.div(x5_a, 2)
        x6, idx6 = self.down5(x5_a)
        x6_a = torch.add(x6, y6)
        x6_a = torch.div(x6_a, 2)

        x = self.up1(x6_a, x5, idx6)
        x = self.up2(x, x4, idx5)
        x = self.up3(x, x3, idx4)
        x = self.up4(x, x2, idx3)
        x = self.up5(x, x1, idx2)
        # x5 = torch.cat([x5, y5], dim=1)
        # x = self.up1(x5, x4, y4)
        # x = self.up2(x, x3, y3)
        # x = self.up3(x, x2, y2)
        # x = self.up4(x, x1, y1)
        logits = self.outc(x)
        return logits

# Backup
# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         self.down4 = Down(512, 512)
#         self.up1 = Up(1024, 256, bilinear)
#         self.up2 = Up(512, 128, bilinear)
#         self.up3 = Up(256, 64, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits
