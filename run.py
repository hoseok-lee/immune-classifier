import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from datasets.immunocto import ImmunoctoDataset
from models.ensemble import EnsembleModel


BATCH_SIZE = 100
N_JOBS = 4
N_SAMPLES = 1


# Instantiate CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def immunocto_loader(
    root_dir,
    n_samples = 1,
    train_split = 0.8,
    batch_size = 100,
    n_jobs = 1
):
    
    dataset = ImmunoctoDataset(
        root_dir = root_dir,
        n_samples = n_samples
    )
    
    trainset, testset = random_split(dataset, [train_split, 1 - train_split])
    
    trainloader = DataLoader(
        trainset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = n_jobs,
    )
    testloader = DataLoader(
        testset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = n_jobs,
    )
    
    return trainloader, testloader


if __name__ == "__main__":
    
    # Instantiate model
    model = EnsembleModel().to(device)
    
    # BCE loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size = 30, 
        gamma = 0.1
    )
    
    trainloader, testloader = immunocto_loader(
        "/datasets/schwartz-lab/hslee/CRC_Immunocto",
        n_samples = N_SAMPLES,
        train_split = 0.8,
        batch_size = BATCH_SIZE,
        n_jobs = N_JOBS
    )
    
    num_epochs = 10
    train_losses, train_acc_list, test_acc_list = [], [], []
    
    # Main training loop
    for epoch in range(num_epochs):
        
        model.train()
        
        running_loss = 0.0
        correct, total = 0, 0
        
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
            print(correct, loss.item())
        
        train_loss = running_loss / len(trainloader.dataset)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_acc_list.append(train_acc)
    
        model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        test_acc = 100. * correct / total
        test_acc_list.append(test_acc)
        
        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), "/project/6101831/shared/blood_vs_tissue/checkpoints")