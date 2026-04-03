from typing import Literal

from .ensemble import EnsembleModel
from .resnet import ResNet18
from .transformer import VisionTransformer


def get_model(
    model: Literal["resnet", "vit", "ensemble"], 
    device
):
    
    if model == "resnet":
        return ResNet18().to(device)
        
    elif model == "vit":
        return VisionTransformer().to(device)
        
    else:
        return EnsembleModel().to(device)

