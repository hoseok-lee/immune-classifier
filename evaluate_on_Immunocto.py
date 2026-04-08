import torch
from torch.utils.data import DataLoader

from datasets.immunocto import ImmunoctoDataset, get_immunocto_loader
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

dataloader = get_immunocto_loader(
    root_dir="/datasets/schwartz-lab/shared/CRC_Immunocto",
    n_samples=10,
    splits=None,
    batch_size=100,
    n_jobs=4
)

correct, total = 0, 0
TP, FP, FN, TN = 0, 0, 0, 0

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        TP += ((predicted == 1) & (labels == 1)).sum().item()
        FP += ((predicted == 1) & (labels == 0)).sum().item()
        FN += ((predicted == 0) & (labels == 1)).sum().item()
        TN += ((predicted == 0) & (labels == 0)).sum().item()

test_acc = 100.0 * correct / total

print(f"==========Evaluation of blood-trained ensemble model on Immunocto dataset:==========")

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

print(f"Accuracy:     {test_acc:.2f}%")
print(f"Precision:    {precision:.4f}")
print(f"Recall:       {recall:.4f}")
print(f"F1-score:     {f1:.4f}")
print(f"Specificity:  {specificity:.4f}")