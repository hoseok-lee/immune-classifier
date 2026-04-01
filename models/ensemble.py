import torch
import torch.nn as nn

from .resnet import ResNet
from .transformer import VisionTransformer


class EnsembleModel(nn.Module):
    
    def __init__(
        self, 
        img_size = 64,
        num_classes = 2
    ):
        
        super().__init__()
        self.resnet = ResNet(num_classes = num_classes)
        self.vit = VisionTransformer(
            img_size = img_size,
            num_classes = num_classes
        )
        
        self.w_resnet = nn.Parameter(torch.randn(1))
        self.w_vit = nn.Parameter(torch.randn(1))

    def forward(self, x):
        
        # Forward pass of individual models
        return self.resnet(x) * self.w_resnet + self.vit(x) * self.w_vit