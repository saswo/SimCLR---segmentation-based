import torch.nn as nn
from torchvision.models import resnet50

class CIFAR10_ResNet50(nn.Module):
    def __init__(self):
        super(CIFAR10_ResNet50, self).__init__()
        
        self.resnet = resnet50(weights=None)

        self.resnet.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.resnet = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
            self.resnet.avgpool  
        )

    def forward(self, x):
        return self.resnet(x)
