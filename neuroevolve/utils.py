import torch
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# Device Configuration
# ==============================================================================

def get_device(verbose=True):
    """
    Get the available device, prioritizing MPS (Apple Silicon) > CUDA > CPU.
    Performs a smoke test on CUDA devices to ensure compatibility.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        try:
            # Smoke test: Try to move a tensor to the GPU and do a simple operation
            x = torch.tensor([1.0]).to(device)
            y = x * 2
            _ = y.cpu()
            
            if verbose:
                if hasattr(torch.version, 'hip') and torch.version.hip:
                     print(f"Success: ROCm (HIP) is available! Using AMD GPU: {torch.cuda.get_device_name(0)}")
                else:
                    print(f"Success: CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"\nCRITICAL ERROR: CUDA is available but the GPU failed the smoke test.")
            print(f"Error details: {e}")
            print("This usually means your PyTorch version is not compatible with your GPU architecture.")
            print("User requested to ABORT if GPU is not working.")
            import sys
            sys.exit(1)
            
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("Success: MPS (Metal Performance Shaders) is available! Using Apple Silicon GPU.")
    else:
        try:
            import torch_directml
            if torch_directml.device_count() > 1:
                device = torch_directml.device(1)
                if verbose:
                    print(f"Success: DirectML is available! Using Device 1: {torch_directml.device_name(1)}")
            else:
                device = torch_directml.device()
                if verbose:
                    print(f"Success: DirectML is available! Using Default Device: {torch_directml.device_name(0)}")
        except ImportError:
            device = torch.device("cpu")
            if verbose:
                print("MPS/CUDA/DirectML not available. Using CPU.")
            
    return device

# ==============================================================================
# Checkpointing Functions
# ==============================================================================

def save_checkpoint(state, checkpoint_dir, is_best=False, filename='checkpoint.pth', best_filename='model_best.pth'):
    """
    Save a training checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, best_filename)
        shutil.copyfile(filepath, best_path)
        print(f"Saved new best model to {best_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """
    Load a checkpoint and resume the model/optimizer state.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")
        
    print(f"Loading checkpoint from '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    state_dict = None
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        try:
            state_dict = checkpoint
        except:
            print("Warning: Could not find 'state_dict' or 'model_state_dict' key.")

    if state_dict is not None:
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"Strict loading failed: {e}. Retrying with strict=False")
            model.load_state_dict(state_dict, strict=False)

    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    elif optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scheduler and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    elif scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint

# ==============================================================================
# Visualization Functions
# ==============================================================================

def plot_training_metrics(train_losses, val_losses, train_metrics=None, val_metrics=None, 
                          metric_label='Accuracy', save_dir='.', filename='training_plot.png'):
    """
    Plot training and validation loss and a secondary metric.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    has_metrics = train_metrics is not None and val_metrics is not None
    
    if has_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        loss_ax = axes[0]
        metric_ax = axes[1]
    else:
        fig, loss_ax = plt.subplots(1, 1, figsize=(8, 5))
        metric_ax = None

    epochs = range(1, len(train_losses) + 1)
    loss_ax.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3)
    loss_ax.plot(epochs, val_losses, label='Val Loss', marker='s', markersize=3)
    loss_ax.set_xlabel('Epoch')
    loss_ax.set_ylabel('Loss')
    loss_ax.set_title('Training and Validation Loss')
    loss_ax.legend()
    loss_ax.grid(True, alpha=0.3)

    if has_metrics:
        metric_ax.plot(epochs, train_metrics, label=f'Train {metric_label}', marker='o', markersize=3)
        metric_ax.plot(epochs, val_metrics, label=f'Val {metric_label}', marker='s', markersize=3)
        metric_ax.set_xlabel('Epoch')
        metric_ax.set_ylabel(metric_label)
        metric_ax.set_title(f'Training and Validation {metric_label}')
        metric_ax.legend()
        metric_ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training plot to {save_path}")
