import torch
from torch.utils.data import DataLoader

from datasets.blood_dataset import BloodDataset
from models.models import get_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

obj = torch.load(
    "checkpoints/immunocto/ensemble.pt",
    map_location = torch.device(device),
    weights_only = False
)

model = get_model("ensemble", device)
model.load_state_dict(obj["model_state"])
model.eval()

testset = BloodDataset(
    csv_path="/home/yyx01056/scratch/splits/test.csv",
    image_size=224,
)

testloader = DataLoader(
    testset,
    batch_size=100,
    shuffle=False,
    pin_memory=True,
)


correct, total = 0, 0
TP, FP, FN, TN = 0, 0, 0, 0

with torch.no_grad():
    for inputs, labels in testloader:
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

print("==========Evaluation of Immunocto-trained Ensemble model on blood dataset:==========")

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

print(f"Accuracy:     {test_acc:.2f}%")
print(f"Precision:    {precision:.4f}")
print(f"Recall:       {recall:.4f}")
print(f"F1-score:     {f1:.4f}")
print(f"Specificity:  {specificity:.4f}")

