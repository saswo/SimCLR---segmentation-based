import torchvision.models as models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

def load_deeplabv3():
    model = models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.eval()
    return model