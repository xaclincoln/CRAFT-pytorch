"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

#from basenet.vgg16_bn import vgg16_bn, init_weights
from basenet.resnet_bn import resnet_bn
from basenet.weight_init import init_weights

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT(nn.Module):
    def __init__(self, pretrained=False, freeze=False, bn_arch='resnet18'):
        super(CRAFT, self).__init__()

        """ Base network """
        self.basenet = resnet_bn(pretrained=pretrained,freeze=freeze,arch=bn_arch)

        """ U network """
        if bn_arch=='resnet50':
            self.upconv1 = double_conv(2048, 1024, 512)
            self.upconv2 = double_conv(512, 512, 256)
            self.upconv3 = double_conv(256, 256, 64)
            self.upconv4 = double_conv(64, 64, 32)
        elif bn_arch=='resnet18':
            self.upconv1 = double_conv(512, 256, 128)
            self.upconv2 = double_conv(128, 128, 64)
            self.upconv3 = double_conv(64, 64, 32)
            self.upconv4 = None

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        if self.upconv4 is not None:
            init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())
        
    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)

        """ U network """
        y = F.interpolate(sources[0], size=sources[1].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        if self.upconv4 is not None:
            y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
            y = torch.cat([y, sources[4]], dim=1)
            feature = self.upconv4(y)
        else:
            feature = y

        y = self.conv_cls(feature)

        return y.permute(0,2,3,1), feature

if __name__ == '__main__':
    model = CRAFT(pretrained=False,bn_arch='resnet18')
    output, _ = model(torch.randn(1, 3, 768, 768))
    print(output.shape)