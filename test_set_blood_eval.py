import argparse
import torch
from torch.utils.data import DataLoader

from datasets.apl import APLDataset
from models.models import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=["resnet", "vit", "ensemble", "dinobloom"]
)
parser.add_argument(
    "--ckpt_dir",
    type=str,
    default="/project/6101831/shared/blood_vs_tissue/checkpoints"
)
args = parser.parse_args()

ckpt_path = f"{args.ckpt_dir}/blood_{args.model}.pt"
print(f"Loading checkpoint: {ckpt_path}")

obj = torch.load(
    ckpt_path,
    map_location=device,
    weights_only=False
)

model = get_model(args.model, device)
model.load_state_dict(obj["model_state"])
model.eval()

testset = APLDataset(
    csv_path="/home/yyx01056/scratch/splits/test.csv",
    image_size=224,
)

testloader = DataLoader(
    testset,
    batch_size=100,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
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

if total == 0:
    raise ValueError("Empty testloader.")

test_acc = 100.0 * correct / total
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

print(f"==========Evaluation of blood-trained {args.model} model on blood test set:==========")
print(f"Accuracy:     {test_acc:.2f}%")
print(f"Precision:    {precision:.4f}")
print(f"Recall:       {recall:.4f}")
print(f"F1-score:     {f1:.4f}")
print(f"Specificity:  {specificity:.4f}")
