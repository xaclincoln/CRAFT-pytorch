from basenet.weight_init import init_weights
from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import model_urls


class resnet50_bn(torch.nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(resnet50_bn, self).__init__()
        model_urls['resnet50'] = model_urls['resnet50'].replace('https://', 'http://')
        self.bn = models.resnet50(pretrained=pretrained)
    
        if not pretrained:
            init_weights(self.bn.layer1.modules())
            init_weights(self.bn.layer2.modules())
            init_weights(self.bn.layer3.modules())
            init_weights(self.bn.layer4.modules())

        # init_weights(self.slice5.modules())        # no pretrained model for fc6 and fc7

    def forward(self, x):
        x = self.bn.conv1(x)
        x = self.bn.bn1(x)
        x = self.bn.relu(x)
        layer0 = self.bn.maxpool(x)

        layer1 = self.bn.layer1(layer0)
        layer2 = self.bn.layer2(layer1)
        layer3 = self.bn.layer3(layer2)
        layer4 = self.bn.layer4(layer3)

        return (layer4,layer3,layer2,layer1,layer0)


if __name__=="__main__":
    resnet = resnet50_bn(pretrained=False)
    x = torch.rand([2,3,800,600])
    y = resnet(x)
    for layer in y:
        print(layer.size())
# should output
# torch.Size([2, 2048, 25, 19])
# torch.Size([2, 1024, 50, 38])
# torch.Size([2, 512, 100, 75])
# torch.Size([2, 256, 200, 150])
# torch.Size([2, 64, 200, 150])