import torch.nn as nn
from models.resnet50_4 import wide_resnet50_4

class BaseEncoderResNet4x(nn.Module):
    def __init__(self):
        super(BaseEncoderResNet4x, self).__init__()
        self.resnet = wide_resnet50_4(weights=None)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        features = self.resnet(x)
        return features.view(features.size(0), -1)