from typing import Literal
import torch
import torch.nn as nn
import torchvision

from .uni import UNIClassifier, UNI2Classifier
from .dinobloom import DinoBloomClassifier

from config import UNI_WEIGHTS, UNI2_H_WEIGHTS, DINOBLOOM_WEIGHTS


# ResNet18 image classifier
# This function returns the model adjusted for the number of classes
# The final fully-connected layer is adjusted
def resnet18(num_classes: int = 2):
    
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(512, num_classes)
    
    return model
    
    
# Vision transformer
# ViT large-16 is the architecture used by UNI, a foundation model
# UNI2-h which uses ViT huge-14 cannot be handled by DRAC
def vit_l_16(num_classes: int = 2):
    
    model = torchvision.models.vit_l_16()
    model.heads.head = nn.Linear(1024, num_classes)
    
    return model


class EnsembleModel(nn.Module):
    
    def __init__(
        self, 
        img_size = 64,
        num_classes = 2
    ):
        
        super().__init__()
        self.resnet = resnet18(num_classes = num_classes)
        self.vit = vit_l_16(num_classes = num_classes)
        
        self.w_r = nn.Parameter(torch.randn(1))
        self.w_v = nn.Parameter(torch.randn(1))

    def forward(self, x):
        # Forward pass of individual models
        return self.resnet(x) * self.w_r + self.vit(x) * self.w_v


def get_model(
    model: Literal[
        "resnet", "vit", "ensemble",    # Base trainable models
        "uni", "uni2", "dinobloom"                   # Foundation models
    ], 
    device
):
    
    if model == "resnet":
        return resnet18().to(device)
        
    elif model == "vit":
        return vit_l_16().to(device)
        
    elif model == "ensemble":
        return EnsembleModel().to(device)
        
    elif model == "uni":
        
        model = UNIClassifier(weights = UNI_WEIGHTS).to(device)
        # Freeze foundation model
        for param in model.fm.parameters():
            param.requires_grad = False
            
        return model

    elif model == "uni2":
        
        model = UNI2Classifier(weights = UNI2_H_WEIGHTS).to(device)
        # Freeze foundation model
        for param in model.fm.parameters():
            param.requires_grad = False
            
        return model

    elif model == "dinobloom":
        
        model = DinoBloomClassifier(
            weights = DINOBLOOM_WEIGHTS,
            modelname = "dinov2_vits14",
            num_classes = 2,
        ).to(device)
        
        return model
