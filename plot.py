import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from models.models import get_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoints = Path("/project/6101831/shared/blood_vs_tissue/checkpoints")


iens = torch.load(checkpoints / "immunocto" / "ensemble.pt", weights_only = False, map_location = device)
ires = torch.load(checkpoints / "immunocto" / "resnet.pt", weights_only = False, map_location = device)
ivit = torch.load(checkpoints / "immunocto" / "vit.pt", weights_only = False, map_location = device)
ibl = torch.load(checkpoints / "immunocto" / "uni.pt", weights_only = False, map_location = device)

bens = torch.load(checkpoints / "blood_ensemble.pt", weights_only = False, map_location = device)
bres = torch.load(checkpoints / "blood_resnet.pt", weights_only = False, map_location = device)
bvit = torch.load(checkpoints / "blood_vit.pt", weights_only = False, map_location = device)
bbl = torch.load(checkpoints / "blood_dinobloom.pt", weights_only = False, map_location = device)

def plot_curves():
    
    x = range(1, len(bens['train_loss']) + 1)
    plt.clf()
    fig, axs = plt.subplots(3, 2, sharex = 'col', sharey = 'row')
    fig.set_size_inches(10, 6)
    
    axs.flat[0].plot(x, bens['train_loss'], label = "Ensemble")
    axs.flat[0].plot(x, bres['train_loss'], label = "ResNet18")
    axs.flat[0].plot(x, bvit['train_loss'], label = "ViT-l-16")
    axs.flat[0].plot(x, bbl['train_loss'], label = "DinoBloom/UNI")
    
    axs.flat[0].set_ylabel("BCE Loss")
    axs.flat[0].set_title("APL Dataset")
    
    axs.flat[2].plot(x, bens['train_acc'], label = "Ensemble")
    axs.flat[2].plot(x, bres['train_acc'], label = "ResNet18")
    axs.flat[2].plot(x, bvit['train_acc'], label = "ViT-l-16")
    axs.flat[2].plot(x, bbl['train_acc'], label = "DinoBloom/UNI")
    
    axs.flat[2].set_ylabel("Training Accuracy")
    
    axs.flat[4].plot(x, bens['valid_acc'], label = "Ensemble")
    axs.flat[4].plot(x, bres['valid_acc'], label = "ResNet18")
    axs.flat[4].plot(x, bvit['valid_acc'], label = "ViT-l-16")
    axs.flat[4].plot(x, bbl['valid_acc'], label = "DinoBloom/UNI")
    
    axs.flat[4].xaxis.set_major_locator(ticker.MaxNLocator(integer = True))
    axs.flat[4].set_ylabel("Validation Accuracy")
    
    
    
    axs.flat[1].plot(x, iens['train_loss'], label = "Ensemble")
    axs.flat[1].plot(x, ires['train_loss'], label = "ResNet18")
    axs.flat[1].plot(x, ivit['train_loss'], label = "ViT-l-16")
    axs.flat[1].plot(x, ibl['train_loss'], label = "DinoBloom/UNI")
    
    axs.flat[1].set_title("Immunocto Dataset")
    
    axs.flat[3].plot(x, iens['train_acc'], label = "Ensemble")
    axs.flat[3].plot(x, ires['train_acc'], label = "ResNet18")
    axs.flat[3].plot(x, ivit['train_acc'], label = "ViT-l-16")
    axs.flat[3].plot(x, ibl['train_acc'], label = "DinoBloom/UNI")
    
    axs.flat[5].plot(x, iens['valid_acc'], label = "Ensemble")
    axs.flat[5].plot(x, ires['valid_acc'], label = "ResNet18")
    axs.flat[5].plot(x, ivit['valid_acc'], label = "ViT-l-16")
    axs.flat[5].plot(x, ibl['valid_acc'], label = "DinoBloom/UNI")
    
    axs.flat[5].xaxis.set_major_locator(ticker.MaxNLocator(integer = True))
    
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.supxlabel("Epochs")
    fig.legend(handles, labels, loc='upper right')
    plt.savefig(
        "/project/6101831/shared/blood_vs_tissue/figures/curves.svg",
        bbox_inches = "tight"
    )
    
    
# plot_curves()

imodel = get_model("ensemble", device = device)
bmodel = get_model("ensemble", device = device)

imodel.load_state_dict(iens['model_state'])
imodel.eval()
bmodel.load_state_dict(bens['model_state'])
bmodel.eval()

print(imodel.w_r, imodel.w_v)
print(bmodel.w_r, bmodel.w_v)
