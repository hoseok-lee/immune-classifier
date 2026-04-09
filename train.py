import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from time import time

from datasets.apl import get_blood_loader, get_immunocto_loader
from models.models import get_model
from config import APL_SPLIT_PATH, IMMUNOCTO_PATH


# Instantiate CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model",
        help = "model for training",
        type = str,
        default = "ensemble",
        choices = ["resnet", "vit", "ensemble", "dinobloom", "uni", "uni2"],
        required = False
    )
    parser.add_argument(
        "-d", "--dataset",
        help = "dataset for training",
        type = str,
        default = "apl",
        choices = ["apl", "immunocto"],
    )
    parser.add_argument(
        "-b", "--batch_size",
        type = int,
        default = 100
    )
    parser.add_argument(
        "-j", "--n_jobs",
        type = int,
        default = 1
    )
    parser.add_argument(
        "-i", "--image_size",
        type = int,
        default = IMAGE_SIZE
    )
    parser.add_argument(
        "-e", "--num_epochs",
        type = int,
        default = 10
    )
    parser.add_argument(
        "-o", "--save_path",
        type = str,
        default = None
    )
    parser.add_argument(
        "-t", "--train_samples_per_epoch",
        help = "number of sampled training examples per epoch with WeightedRandomSampler",
        type = int,
        default = 30000
    )

    args = parser.parse_args()

    # Instantiate model
    model = get_model(args.model, device)
    
    # BCE loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size = 30,
        gamma = 0.1
    )

    # Dataset
    if args.dataset == "apl":
        trainloader, validloader, testloader = get_blood_loader(
            split_dir = APL_SPLIT_PATH,
            batch_size = args.batch_size,
            n_jobs = args.n_jobs,
            image_size = args.image_size,
            train_samples_per_epoch = args.train_samples_per_epoch
        )
        
    else:
        trainloader, validloader, testloader = get_immunocto_loader(
            IMMUNOCTO_PATH,
            n_samples = args.n_samples,
            batch_size = args.batch_size,
            n_jobs = args.n_jobs
        )

    train_losses, train_acc_list, valid_acc_list = [], [], []

    # Main training loop
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
                F.one_hot(labels, num_classes = 2).float()
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

    # Test accuracy
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100.0 * correct / total


    # Save model
    if args.save_path is not None:
        os.makedirs(Path("./checkpoints") / args.dataset, exist_ok = True)
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
            Path("./checkpoints") / args.dataset / f"{args.model}.pt"
        )
