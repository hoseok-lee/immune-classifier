import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from time import time

from datasets.apl import APLDataset
from models.models import get_model



BATCH_SIZE = 100
N_JOBS = 4
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def blood_loader(
    split_dir,
    batch_size=100,
    n_jobs=4,
    image_size=224,
    train_samples_per_epoch=None,
):
    trainset = APLDataset(
        csv_path=f"{split_dir}/train.csv",
        image_size=image_size,
    )
    validset = APLDataset(
        csv_path=f"{split_dir}/val.csv",
        image_size=image_size,
    )
    testset = APLDataset(
        csv_path=f"{split_dir}/test.csv",
        image_size=image_size,
    )

    # weighted sampling to address class imbalance
    train_labels = trainset.df["label"].values.astype(int)
    class_counts = np.bincount(train_labels, minlength=2)

    print(f"Train class counts: {class_counts.tolist()}")  # [non_immune, immune]

    # inverse-frequency sample weights
    class_sample_weights = 1.0 / class_counts
    sample_weights = class_sample_weights[train_labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    if train_samples_per_epoch is None:
        train_samples_per_epoch = len(trainset)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=train_samples_per_epoch,
        replacement=True,
    )
    
    # use the sampler in the trainloader, and shuffle=False since shuffling is handled by the sampler
    trainloader = DataLoader(
    trainset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=n_jobs,
    pin_memory=True,
    persistent_workers=(n_jobs > 0),
    )

    validloader = DataLoader(
        validset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_jobs,
        pin_memory=True,
        persistent_workers=(n_jobs > 0),
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_jobs,
        pin_memory=True,
        persistent_workers=(n_jobs > 0),
    )

    return trainloader, validloader, testloader



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="ensemble",
        choices=["resnet", "vit", "ensemble", "dinobloom"],
        help="model for training"
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default="/home/yyx01056/scratch/splits",
        help="directory containing train.csv, val.csv, test.csv"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=N_JOBS
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=IMAGE_SIZE
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="freeze DinoBloom backbone if model == dinobloom"
    )
    parser.add_argument(
        "--train_samples_per_epoch",
        type=int,
        default=30000,
        help="number of sampled training examples per epoch with WeightedRandomSampler"
    )

    args = parser.parse_args()

    model = get_model(args.model, device)
    
    if args.model == "dinobloom" and args.freeze_backbone:
        print("Freezing DinoBloom backbone")
        for p in model.fm.parameters():
            p.requires_grad = False
    

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=30,
        gamma=0.1
    )

    trainloader, validloader, testloader = blood_loader(
    split_dir=args.split_dir,
    batch_size=args.batch_size,
    n_jobs=args.n_jobs,
    image_size=args.image_size,
    train_samples_per_epoch=args.train_samples_per_epoch,
    )

    train_losses, train_acc_list, valid_acc_list = [], [], []

    for epoch in range(args.num_epochs):
        model.train()

        running_loss = 0.0
        correct, total = 0, 0
        start = time()

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(
                outputs,
                F.one_hot(labels, num_classes=2).float()
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        train_losses.append(train_loss)
        train_acc_list.append(train_acc)

        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        valid_acc = 100.0 * correct / total
        valid_acc_list.append(valid_acc)

        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{args.num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Validation Acc: {valid_acc:.2f}% | "
            f"{(time() - start) / 60:.2f} minutes"
        )

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100.0 * correct / total

    if args.save_path is None:
        save_path = f"/project/6101831/shared/blood_vs_tissue/checkpoints/blood_{args.model}.pt"
    else:
        save_path = args.save_path

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "criterion": criterion,
            "train_loss": train_losses,
            "train_acc": train_acc_list,
            "valid_acc": valid_acc_list,
            "test_acc": test_acc,
            "split_dir": args.split_dir,
            "image_size": args.image_size,
            "model_name": args.model,
            "input_mode": "grayscale_masked",
        },
        save_path
    )

    print(f"Test Acc: {test_acc:.2f}%")
    print(f"Saved checkpoint to: {save_path}")
