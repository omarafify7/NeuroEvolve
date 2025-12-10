import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional
from .genome import Genome

def get_data_loaders(batch_size: int = 128, num_workers: int = 2) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Downloads CIFAR-10 and returns train and val data loaders.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Download to ./data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader

def train_model(genome: Genome, train_loader, val_loader, epochs: int = 5, device: str = 'cpu', verbose: bool = False, return_model: bool = False):
    """
    Trains a genome's model and returns validation accuracy and model size.
    
    Args:
        genome: The Genome to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs: Number of epochs to train.
        device: Device to train on.
        verbose: If True, print training progress.
        return_model: If True, returns (val_accuracy, param_count, model).
        
    Returns:
        (val_accuracy, param_count) or (val_accuracy, param_count, model)
    """
    try:
        model = genome.decode(input_shape=(3, 32, 32)).to(device)
    except Exception as e:
        # This is expected for some random mutations (e.g. too many pooling layers)
        # We assign 0 fitness so it dies out.
        print(f"Genome invalid (assigning fitness 0): {e}")
        if return_model:
            return 0.0, 0, None
        return 0.0, 0
        
    try:
        param_count = sum(p.numel() for p in model.parameters())
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=genome.learning_rate)
        
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            train_acc = 100 * correct / total
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} - Acc: {train_acc:.2f}%")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        
        if return_model:
            return val_accuracy, param_count, model
            
        return val_accuracy, param_count

    finally:
        # Aggressive cleanup to prevent OOM/fragmentation
        if not return_model:
            del model
        if 'optimizer' in locals():
            del optimizer
        torch.cuda.empty_cache()
        import gc
        gc.collect()

import ray

import gc

@ray.remote(num_gpus=0.2, max_restarts=-1) # Allocate fractional GPU, auto-restart on crash
class TrainActor:
    def __init__(self, batch_size=256):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Actor initialized on {self.device}")
        
        # Load data once per actor
        # We use a smaller batch size for parallel workers to avoid OOM
        self.train_loader, self.val_loader = get_data_loaders(batch_size=batch_size, num_workers=0) # num_workers=0 for safety in Ray
        
    def train(self, genome: Genome, epochs: int = 1):
        return train_model(
            genome, 
            self.train_loader, 
            self.val_loader, 
            epochs=epochs, 
            device=self.device, 
            verbose=False
        )
