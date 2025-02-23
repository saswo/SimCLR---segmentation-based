import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, in_features=2048, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)