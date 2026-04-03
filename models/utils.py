from models.ensemble import EnsembleModel
from models.resnet import ResNet18
from models.transformer import VisionTransformer


def get_model(model, device):
    
    if model == "resnet":
        return ResNet18().to(device)
        
    elif model == "vit":
        return VisionTransformer().to(device)
        
    else:
        return EnsembleModel().to(device)

