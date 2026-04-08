import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from time import time

from datasets.immunocto import get_immunocto_loader
from models.models import get_model
from config import IMMUNOCTO_PATH


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
        choices = ["resnet", "vit", "ensemble", "uni", "uni2"],
        required = False
    )
    parser.add_argument(
        "-s", "--n_samples",
        type = int,
        default = 1e6
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
        "-e", "--n_epochs",
        type = int,
        default = 10
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
    
    trainloader, validloader, testloader = get_immunocto_loader(
        IMMUNOCTO_PATH,
        n_samples = args.n_samples,
        batch_size = args.batch_size,
        n_jobs = args.n_jobs
    )
    
    train_losses, train_acc_list, valid_acc_list = [], [], []
    
    # Main training loop
    for epoch in range(args.n_epochs):
        
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
                # Use one-hot encoding
                F.one_hot(labels, num_classes = 2).float()
            )
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(trainloader.dataset)
        train_acc = 100. * correct / total
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
                
        valid_acc = 100. * correct / total
        valid_acc_list.append(valid_acc)
        
        scheduler.step()
        print(f"Epoch [{epoch + 1}/{args.n_epochs}] \
            Train Loss: {train_loss:.4f} | \
            Train Acc: {train_acc:.2f}% | \
            Validation Acc: {valid_acc:.2f}% | \
            {(time() - start) / 60:.2f} minutes")
    
    
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
            
    test_acc = 100. * correct / total
    
    
    # Save model
    torch.save(
        {
            'model_state':      model.state_dict(),
            'optimizer_state':  optimizer.state_dict(),
            'criterion':        criterion,
            'train_loss':       train_losses,
            'train_acc':        train_acc_list,
            'valid_acc':        valid_acc_list,
            'test_acc':         test_acc,
            'n_samples':        args.n_samples
        }, 
        f"/project/6101831/shared/blood_vs_tissue/checkpoints/immunocto/{args.model}.pt"
    )