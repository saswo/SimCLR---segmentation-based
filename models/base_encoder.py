import torch.nn as nn
from torchvision.models import resnet50

class BaseEncoder(nn.Module):
    def __init__(self):
        super(BaseEncoder, self).__init__()
        self.resnet = resnet50(weights=None)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1]) 

    def forward(self, x):
        features = self.resnet(x)
        return features.view(features.size(0), -1)