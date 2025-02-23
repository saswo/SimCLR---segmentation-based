from torchvision.models.resnet import Bottleneck, ResNet

def wide_resnet50_4(weights=None):
    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        width_per_group=256,  
    )
    return model
