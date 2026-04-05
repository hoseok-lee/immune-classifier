import torch
import torch.nn as nn
import timm


def uni(weights: str):
    
    model = timm.create_model(
        "vit_large_patch16_224", 
        img_size = 224, 
        patch_size = 16, 
        init_values = 1e-5, 
        num_classes = 0, 
        dynamic_img_size = True
    )
    
    model.load_state_dict(
        torch.load(weights),
        strict = True
    )
    
    return model


def uni2_h(weights: str):
    
    model = timm.create_model(
        **{
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
    )
    
    model.load_state_dict(
        torch.load(weights),
        strict = True
    )
    
    return model
    

class UNIClassifier(nn.Module):
    
    def __init__(self, weights, num_classes = 2):
        
        super().__init__()
        self.fm = uni(weights)
        self.fc = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        
        out = self.fm(x)
        out = self.fc(out)
        
        return out
    

class UNI2Classifier(nn.Module):
    
    def __init__(self, weights, num_classes = 2):
        
        super().__init__()
        self.fm = uni2_h(weights)
        self.fc = nn.Linear(1536, num_classes)
        
    def forward(self, x):
        
        out = self.fm(x)
        out = self.fc(out)
        
        return out
        