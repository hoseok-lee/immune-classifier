import torch
from torch.utils.data import DataLoader

from datasets.immunocto import ImmunoctoDataset
from datasets.blood_dataset import BloodDataset
from models.ensemble import EnsembleModel
from models.resnet import ResNet18
from models.transformer import VisionTransformer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

obj = torch.load(
    "checkpoints/immunocto/ensemble.pt",
    map_location = torch.device(device),
    weights_only = False
)

model = EnsembleModel().to(device)
model.load_state_dict(obj['model_state'])
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

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_acc = 100.0 * correct / total
print(test_acc)