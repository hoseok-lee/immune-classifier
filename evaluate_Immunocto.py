import torch
from torch.utils.data import DataLoader

from datasets.immunocto import ImmunoctoDataset
from datasets.blood_dataset import BloodDataset
from models.models import get_model

import torch

from datasets.immunocto import get_immunocto_loader
from models.models import get_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

obj = torch.load(
    "/home/yyx01056/projects/aip-gregorys/shared/blood_vs_tissue/checkpoints/blood_ensemble.pt",
    map_location=device,
    weights_only=False
)

model = get_model("ensemble", device)
model.load_state_dict(obj['model_state'])
model.eval()

trainloader, validloader, testloader = get_immunocto_loader(
    root_dir="/datasets/schwartz-lab/hslee/CRC_Immunocto",
    n_samples=10,
    splits=[0.8, 0.1, 0.1],
    batch_size=100,
    n_jobs=4
)

correct, total = 0, 0

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_acc = 100.0 * correct / total

print(test_acc)
print(f"Ensemble: Immunocto validation accuracy: {test_acc:.2f}%")