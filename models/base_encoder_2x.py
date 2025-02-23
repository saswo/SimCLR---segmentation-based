import torch.nn as nn
from torchvision.models import wide_resnet50_2

class BaseEncoderResNet2x(nn.Module):
    def __init__(self):
        super(BaseEncoderResNet2x, self).__init__()
        self.resnet = wide_resnet50_2(weights=None)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        features = self.resnet(x)
        return features.view(features.size(0), -1)
