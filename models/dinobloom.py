import torch
import torch.nn as nn


EMBED_SIZES = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


def dinobloom(weights: str, modelname: str = "dinov2_vits14"):
    
    model = torch.hub.load("facebookresearch/dinov2", modelname)
    pretrained = torch.load(weights, map_location = "cpu")

    new_state_dict = {}
    for key, value in pretrained["teacher"].items():
        
        if "dino_head" in key or "ibot_head" in key:
            continue
            
        new_key = key.replace("backbone.", "")
        new_state_dict[new_key] = value

    pos_embed = nn.Parameter(torch.zeros(1, 257, EMBED_SIZES[modelname]))
    model.pos_embed = pos_embed
    model.load_state_dict(new_state_dict, strict = True)
    
    return model


class DinoBloomClassifier(nn.Module):
    
    def __init__(
        self,
        weights: str,
        modelname: str = "dinov2_vits14",
        num_classes: int = 2,
    ):
        
        super().__init__()
        self.fm = dinobloom(weights, modelname)
        self.fc = nn.Linear(EMBED_SIZES[modelname], num_classes)

        # ImageNet normalization for DINOv2 / DinoBloom
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        
        # Normalization
        x = (x - self.mean) / self.std
        out = self.fm(x)
        out = self.fc(out)
        
        return out