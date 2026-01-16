"""
Comprehensive seeding utilities for reproducible EEG experiments
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True):
    """
    Set all random seeds for reproducibility
    
    Args:
        seed: Random seed value
        deterministic: If True, use deterministic CUDA operations (slower but reproducible)
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Set environment variables for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Make CUDA operations deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # For PyTorch >= 1.8
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
    else:
        # Faster but may have slight variations
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    print(f"✓ All random seeds set to {seed}")
    if deterministic:
        print("✓ Deterministic mode enabled (slower but fully reproducible)")
    else:
        print("⚠ Non-deterministic mode (faster but may have minor variations)")


def worker_init_fn(worker_id: int, seed: int = 42):
    """
    Initialize random seeds for DataLoader workers
    
    Args:
        worker_id: Worker ID (automatically passed by DataLoader)
        seed: Base seed value
    """
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_generator(seed: int = 42):
    """
    Create a PyTorch random generator with fixed seed
    
    Args:
        seed: Random seed
        
    Returns:
        torch.Generator with fixed seed
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# Example usage functions
def create_reproducible_dataloader(dataset, batch_size=32, num_workers=4, 
                                   seed=42, shuffle=True):
    """
    Create a DataLoader with proper seeding for reproducibility
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        num_workers: Number of workers
        seed: Random seed
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader with reproducible behavior
    """
    from torch.utils.data import DataLoader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, seed),
        generator=get_generator(seed) if shuffle else None
    )


if __name__ == "__main__":
    # Test seeding
    seed_everything(42, deterministic=True)
    
    # Verify reproducibility
    print("\nVerifying reproducibility:")
    print(f"Random: {random.random()}")
    print(f"NumPy: {np.random.rand()}")
    print(f"PyTorch: {torch.rand(1).item()}")
    
    # Reset and verify same values
    seed_everything(42, deterministic=True)
    print("\nAfter re-seeding:")
    print(f"Random: {random.random()}")
    print(f"NumPy: {np.random.rand()}")
    print(f"PyTorch: {torch.rand(1).item()}")