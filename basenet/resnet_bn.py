from basenet.weight_init import init_weights
import torch
from torchvision import models
from torchvision.models.resnet import model_urls


class resnet_bn(torch.nn.Module):
    def __init__(self, arch: str = 'resnet18', pretrained=True, freeze=True):
        super(resnet_bn, self).__init__()
        model_urls[arch] = model_urls[arch].replace('https://', 'http://')
        self.bn = getattr(models,arch)(pretrained=pretrained)

        if not pretrained:
            init_weights(self.bn.conv1.modules())
            init_weights(self.bn.layer1.modules())
            init_weights(self.bn.layer2.modules())
            init_weights(self.bn.layer3.modules())
            init_weights(self.bn.layer4.modules())

        # init_weights(self.slice5.modules())        # no pretrained model for fc6 and fc7

    def forward(self, x):
        x = self.bn.conv1(x)
        x = self.bn.bn1(x)
        layer0 = self.bn.relu(x)
        #x = self.bn.maxpool(x)

        layer1 = self.bn.layer1(layer0)
        layer2 = self.bn.layer2(layer1)
        layer3 = self.bn.layer3(layer2)
        layer4 = self.bn.layer4(layer3)

        return (layer4, layer3, layer2, layer1, layer0)


if __name__ == "__main__":
    resnet = resnet_bn(pretrained=False,arch='resnet18')
    x = torch.rand([2, 3, 800, 600])
    y = resnet(x)
    for layer in y:
        print(layer.size())
# should output
# torch.Size([2, 2048, 25, 19])
# torch.Size([2, 1024, 50, 38])
# torch.Size([2, 512, 100, 75])
# torch.Size([2, 256, 200, 150])
# torch.Size([2, 64, 200, 150])
