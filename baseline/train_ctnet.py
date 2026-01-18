"""
CTNet: A Convolutional Transformer Network for EEG-Based Motor Imagery Classification
Supports: SSVEP, P300, MI (Motor Imagery), Imagined Speech, Lee2019_MI, Lee2019_SSVEP, BNCI2014_P300

Reference: Zhao et al. (2024) - CTNet: a convolutional transformer network for EEG-based motor imagery classification
Paper: https://doi.org/10.1038/s41598-024-71118-7
GitHub: https://github.com/snailpt/CTNet
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import random
import sys
from typing import Optional, Dict, Tuple

# Add scale-net directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scale-net'))

from seed_utils import seed_everything, worker_init_fn, get_generator
from dataset import load_dataset, TASK_CONFIGS

# einops for tensor operations
try:
    from einops import rearrange
    from einops.layers.torch import Rearrange
    EINOPS_AVAILABLE = True
except ImportError:
    print("Warning: einops not installed. Install with: pip install einops")
    EINOPS_AVAILABLE = False


# ==================== CTNet Model Components ====================

class PatchEmbeddingCNN(nn.Module):
    """
    CNN-based patch embedding module (EEGNet-style)
    Extracts local and spatial features from EEG time series
    """
    def __init__(self, f1=8, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8, 
                 dropout_rate=0.3, n_channels=22, emb_size=40):
        super().__init__()
        
        if not EINOPS_AVAILABLE:
            raise ImportError("einops is required for CTNet. Install with: pip install einops")
        
        f2 = D * f1
        self.cnn_module = nn.Sequential(
            # Temporal convolution (kernel_size ~ 0.25 * fs)
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding='same', bias=False),
            nn.BatchNorm2d(f1),
            # Depthwise spatial convolution
            nn.Conv2d(f1, f2, (n_channels, 1), (1, 1), groups=f1, padding='valid', bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # Average pooling 1 (acts as patch slicing)
            nn.AvgPool2d((1, pooling_size1)),
            nn.Dropout(dropout_rate),
            # Separable convolution
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # Average pooling 2
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),
        )
        
        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_module(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention module"""
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.masked_fill_(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class FeedForwardBlock(nn.Sequential):
    """Feed-forward network in Transformer"""
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class ResidualAdd(nn.Module):
    """Residual connection with layer normalization"""
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)
        out = self.layernorm(self.drop(res) + x_input)
        return out


class TransformerEncoderBlock(nn.Sequential):
    """Single Transformer encoder block"""
    def __init__(self, emb_size, num_heads=4, drop_p=0.5, 
                 forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(
                MultiHeadAttention(emb_size, num_heads, drop_p),
                emb_size, drop_p
            ),
            ResidualAdd(
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                emb_size, drop_p
            )
        )


class TransformerEncoder(nn.Sequential):
    """Stack of Transformer encoder blocks"""
    def __init__(self, num_heads, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size, num_heads) for _ in range(depth)])


class PositionalEncoding(nn.Module):
    """Learnable positional encoding"""
    def __init__(self, emb_size, max_length=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, max_length, emb_size))
    
    def forward(self, x):
        # x: [batch, length, embedding]
        x = x + self.encoding[:, :x.shape[1], :]
        return self.dropout(x)


class ClassificationHead(nn.Module):
    """Classification head with dropout"""
    def __init__(self, flatten_size, n_classes, dropout=0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(flatten_size, n_classes)
        )

    def forward(self, x):
        return self.fc(x)


class CTNet(nn.Module):
    """
    CTNet: Convolutional Transformer Network for EEG Classification
    
    Architecture:
    1. PatchEmbeddingCNN: EEGNet-style CNN for local feature extraction
    2. Positional Encoding: Learnable positional embeddings
    3. Transformer Encoder: Multi-head self-attention for global dependencies
    4. Classification Head: Linear layer for final classification
    """
    def __init__(self, n_channels=22, n_samples=1000, n_classes=4,
                 # CNN parameters
                 f1=8, D=2, kernel_size=64, 
                 pooling_size1=8, pooling_size2=8, cnn_dropout=0.3,
                 # Transformer parameters
                 emb_size=16, num_heads=2, depth=6,
                 # Classification parameters
                 fc_dropout=0.5):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.emb_size = emb_size
        self.is_binary = (n_classes == 2)
        
        # Calculate sequence length after CNN pooling
        seq_len = n_samples // pooling_size1 // pooling_size2
        flatten_size = seq_len * emb_size
        
        # Adjust emb_size to be divisible by num_heads
        f2 = D * f1
        actual_emb_size = f2  # The embedding size is determined by CNN output channels
        
        # CNN backbone
        self.cnn = PatchEmbeddingCNN(
            f1=f1,
            kernel_size=kernel_size,
            D=D,
            pooling_size1=pooling_size1,
            pooling_size2=pooling_size2,
            dropout_rate=cnn_dropout,
            n_channels=n_channels,
            emb_size=actual_emb_size
        )
        
        # Positional encoding
        self.position = PositionalEncoding(actual_emb_size, max_length=seq_len + 10, dropout=0.1)
        
        # Transformer encoder
        self.transformer = TransformerEncoder(num_heads, depth, actual_emb_size)
        
        # Classification head
        flatten_size = seq_len * actual_emb_size
        self.flatten = nn.Flatten()
        self.classification = ClassificationHead(
            flatten_size, 
            1 if self.is_binary else n_classes, 
            dropout=fc_dropout
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, C, T) or (B, 1, C, T)
            
        Returns:
            Logits of shape (B, n_classes) or (B, 1) for binary
        """
        # Ensure input is 4D: (B, 1, C, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # CNN feature extraction
        cnn_out = self.cnn(x)  # (B, seq_len, emb_size)
        
        # Scale and add positional encoding
        cnn_out = cnn_out * math.sqrt(self.emb_size)
        cnn_out = self.position(cnn_out)
        
        # Transformer encoding
        trans_out = self.transformer(cnn_out)
        
        # Residual connection
        features = cnn_out + trans_out
        
        # Classification
        out = self.flatten(features)
        out = self.classification(out)
        
        return out


# ==================== Raw EEG Dataset ====================

class RawEEGDataset(Dataset):
    """Dataset for raw EEG data"""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, 
                 normalize: bool = True, augment: bool = False):
        self.data = data.astype(np.float32)
        self.labels = torch.LongTensor(labels)
        self.normalize = normalize
        self.augment = augment
    
    def __len__(self):
        return len(self.data)
    
    def _augment(self, x: np.ndarray) -> np.ndarray:
        """Apply augmentation on raw EEG"""
        # Gaussian noise injection
        if np.random.random() < 0.5:
            x = x + np.random.randn(*x.shape).astype(np.float32) * 0.05 * np.std(x)
        # Amplitude scaling
        if np.random.random() < 0.5:
            x = x * np.random.uniform(0.8, 1.2)
        # Time shift
        if np.random.random() < 0.3:
            shift = np.random.randint(-10, 11)
            x = np.roll(x, shift, axis=-1)
        return x
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx].copy()
        y = self.labels[idx]
        
        if self.augment:
            x = self._augment(x)
        
        # Normalize (per-sample, per-channel)
        if self.normalize:
            mean = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1, keepdims=True) + 1e-8
            x = (x - mean) / std
        
        return torch.FloatTensor(x), y


# ==================== Data Loader Creation ====================

def create_raw_dataloaders(datasets: Dict, batch_size: int = 32, 
                           num_workers: int = 4, augment_train: bool = True, 
                           seed: int = 44) -> Dict:
    """Create DataLoaders for raw EEG data"""
    loaders = {}
    
    for split, (X, y) in datasets.items():
        augment = augment_train if split == 'train' else False
        shuffle = (split == 'train')
        
        ds = RawEEGDataset(X, y, normalize=True, augment=augment)
        loaders[split] = DataLoader(
            ds, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers, 
            pin_memory=True,
            worker_init_fn=lambda worker_id: worker_init_fn(worker_id, seed),
            generator=get_generator(seed) if shuffle else None
        )
    
    return loaders


# ==================== Multi-GPU Setup ====================

def setup_device():
    """Setup device and return device info"""
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device = torch.device('cuda')
        print(f"CUDA available: {n_gpus} GPU(s) detected")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return device, n_gpus
    else:
        print("CUDA not available, using CPU")
        return torch.device('cpu'), 0


def wrap_model_multi_gpu(model, n_gpus):
    """Wrap model with DataParallel if multiple GPUs available"""
    if n_gpus > 1:
        print(f"Using DataParallel with {n_gpus} GPUs")
        model = nn.DataParallel(model)
    return model


def unwrap_model(model):
    """Get the underlying model from DataParallel wrapper"""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


# ==================== Training Functions ====================

def train_epoch(model, loader, criterion, optimizer, device, is_binary=False):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    pbar = tqdm(loader, desc='Train', ncols=100)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        if is_binary:
            labels_loss = labels.float().unsqueeze(1)
        else:
            labels_loss = labels
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels_loss)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if is_binary:
            pred = (torch.sigmoid(outputs) > 0.5).squeeze(1).long()
        else:
            _, pred = outputs.max(1)
        
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{total_loss/(pbar.n+1):.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, device, criterion=None, is_binary=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Eval', ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            
            if is_binary:
                labels_loss = labels.float().unsqueeze(1)
            else:
                labels_loss = labels
            
            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, labels_loss)
                total_loss += loss.item()
            
            if is_binary:
                pred = (torch.sigmoid(outputs) > 0.5).squeeze(1).long()
            else:
                _, pred = outputs.max(1)
            
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    
    avg_loss = total_loss / len(loader) if criterion is not None else None
    acc = 100. * correct / total
    return avg_loss, acc


# ==================== CTNet Configuration per Task ====================

def get_ctnet_config(task: str, n_channels: int, n_samples: int, sampling_rate: int) -> Dict:
    """
    Get CTNet hyperparameters optimized for each task
    
    Args:
        task: Task name
        n_channels: Number of EEG channels
        n_samples: Number of time samples
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary of CTNet configuration
    """
    # Default configuration (from CTNet paper for BCI IV-2a)
    config = {
        'f1': 8,
        'D': 2,
        'kernel_size': max(sampling_rate // 4, 32),  # ~0.25s
        'pooling_size1': 8,
        'pooling_size2': 8,
        'cnn_dropout': 0.3,
        'emb_size': 16,
        'num_heads': 2,
        'depth': 6,
        'fc_dropout': 0.5,
    }
    
    # Task-specific adjustments
    if task in ['SSVEP', 'Lee2019_SSVEP']:
        config['kernel_size'] = max(sampling_rate // 2, 64)
        config['f1'] = 8
        config['depth'] = 4
    elif task in ['P300', 'BNCI2014_P300']:
        config['kernel_size'] = sampling_rate // 4
        config['depth'] = 4
        config['num_heads'] = 2
    elif task in ['MI', 'Lee2019_MI']:
        config['kernel_size'] = 64
        config['f1'] = 8
        config['depth'] = 6
    elif task == 'Imagined_speech':
        config['kernel_size'] = min(256, n_samples // 8)
        config['pooling_size1'] = 16
        config['pooling_size2'] = 16
        config['depth'] = 4
    
    # Adjust pooling based on signal length to avoid size issues
    total_pool = config['pooling_size1'] * config['pooling_size2']
    while n_samples // total_pool < 2:
        if config['pooling_size2'] > 2:
            config['pooling_size2'] //= 2
        elif config['pooling_size1'] > 2:
            config['pooling_size1'] //= 2
        else:
            break
        total_pool = config['pooling_size1'] * config['pooling_size2']
    
    # Ensure emb_size is divisible by num_heads
    f2 = config['D'] * config['f1']
    while f2 % config['num_heads'] != 0:
        if config['num_heads'] > 1:
            config['num_heads'] -= 1
        else:
            break
    
    return config


# ==================== Main Training ====================

def train_task(task: str, config: Optional[Dict] = None, model_path: Optional[str] = None) -> Tuple:
    """
    Train CTNet for a specific EEG task
    
    Args:
        task: One of 'SSVEP', 'P300', 'MI', 'Imagined_speech', etc.
        config: Training configuration (uses defaults if None)
        model_path: Path to save best model
        
    Returns:
        (model, results_dict)
    """
    task_config = TASK_CONFIGS.get(task, {})
    
    if config is None:
        config = {
            'data_dir': task_config.get('data_dir', '/ocean/projects/cis250213p/shared/ssvep'),
            'num_seen': task_config.get('num_seen', 33),
            'seed': 44,
            'n_classes': task_config.get('num_classes', 26),
            'sampling_rate': task_config.get('sampling_rate', 250),
            'batch_size': 64,
            'num_epochs': 100,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'patience': 20,
            'scheduler': 'ReduceLROnPlateau',
        }
    else:
        config.setdefault('n_classes', task_config.get('num_classes', 26))
        config.setdefault('sampling_rate', task_config.get('sampling_rate', 250))
        config.setdefault('scheduler', 'ReduceLROnPlateau')
        config.setdefault('data_dir', task_config.get('data_dir'))
        config.setdefault('num_seen', task_config.get('num_seen'))

    seed = config.get('seed', 44)
    seed_everything(seed, deterministic=True)
    
    device, n_gpus = setup_device()
    print(f"\n{'='*70}")
    print(f"CTNet - {task} Classification")
    print(f"{'='*70}")
    print(f"Device: {device}, GPUs: {n_gpus}")
    
    # ====== Load Data ======
    datasets = load_dataset(
        task=task,
        data_dir=config.get('data_dir'),
        num_seen=config.get('num_seen'),
        seed=config.get('seed', 44)
    )
    
    if not datasets:
        raise ValueError(f"Failed to load data for task: {task}")
    
    # ====== Create Data Loaders ======
    loaders = create_raw_dataloaders(
        datasets, 
        batch_size=config['batch_size'],
        num_workers=4,
        augment_train=True,
        seed=seed
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    test1_loader = loaders.get('test1')
    test2_loader = loaders.get('test2')
    
    # Get dimensions
    sample_x, _ = next(iter(train_loader))
    n_channels, n_samples = sample_x.shape[1], sample_x.shape[2]
    print(f"Input shape: ({n_channels} channels, {n_samples} samples)")
    
    # ====== Get CTNet Configuration ======
    ctnet_config = get_ctnet_config(
        task, n_channels, n_samples, config['sampling_rate']
    )
    print(f"\nCTNet Configuration:")
    for k, v in ctnet_config.items():
        print(f"  {k}: {v}")
    
    # ====== Create Model ======
    n_classes = config['n_classes']
    model = CTNet(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
        **ctnet_config
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {n_params:,}")
    print(f"Classes: {n_classes}")
    
    model = wrap_model_multi_gpu(model, n_gpus)
    
    # ====== Loss & Optimizer ======
    is_binary = (n_classes == 2)
    if is_binary:
        criterion = nn.BCEWithLogitsLoss()
        print("Using BCEWithLogitsLoss for binary classification")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"Using CrossEntropyLoss for {n_classes}-class classification")
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    scheduler_type = config.get('scheduler', 'ReduceLROnPlateau')
    if scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['num_epochs'] // 2, eta_min=1e-6
        )
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    else:
        raise ValueError(f"Invalid scheduler: {scheduler_type}")
    
    # ====== Training Loop ======
    best_val_acc = 0
    patience_counter = 0
    
    if model_path is None:
        model_path = f'best_ctnet_{task.lower()}_model.pth'
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch [{epoch+1}/{config['num_epochs']}]")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, is_binary=is_binary)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion, is_binary=is_binary)
        
        if scheduler_type == 'CosineAnnealingLR':
            scheduler.step()
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrap_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'task': task,
                'config': config,
                'ctnet_config': ctnet_config,
                'n_channels': n_channels,
                'n_samples': n_samples,
            }, model_path)
            print(f"✓ Best model saved! ({val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{config['patience']}")
        
        # Early stopping
        if patience_counter >= config['patience']:
            print("\nEarly stopping triggered!")
            break
    
    # ====== Final Evaluation ======
    print(f"\n{'='*70}")
    print("Loading best model for final evaluation...")
    print(f"Best model path: {model_path}")
    checkpoint = torch.load(model_path)
    unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
    
    results = {'val': best_val_acc}
    
    if test1_loader:
        test1_loss, test1_acc = evaluate(model, test1_loader, device, criterion, is_binary=is_binary)
        results['test1'] = test1_acc
        results['test1_loss'] = test1_loss
    
    if test2_loader:
        test2_loss, test2_acc = evaluate(model, test2_loader, device, criterion, is_binary=is_binary)
        results['test2'] = test2_acc
        results['test2_loss'] = test2_loss
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS - {task} (CTNet)")
    print(f"{'='*70}")
    print(f"Best Val Acc:    {best_val_acc:.2f}%")
    if 'test1' in results:
        print(f"Test1 (Seen):    {results['test1']:.2f}% (loss {results['test1_loss']:.4f})")
    if 'test2' in results:
        print(f"Test2 (Unseen):  {results['test2']:.2f}% (loss {results['test2_loss']:.4f})")
    print(f"{'='*70}")
    
    return model, results


def train_all_tasks(tasks: Optional[list] = None, save_dir: str = './checkpoints'):
    """Train CTNet models for all specified tasks"""
    if tasks is None:
        tasks = ['SSVEP', 'P300', 'MI', 'Imagined_speech', 'Lee2019_MI', 'Lee2019_SSVEP', 'BNCI2014_P300']
    
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = {}
    
    print("=" * 80)
    print("CTNet - MULTI-TASK EEG CLASSIFICATION")
    print("=" * 80)
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"TASK: {task}")
        print(f"{'='*60}")
        
        try:
            model_path = os.path.join(save_dir, f'best_ctnet_{task.lower()}_model.pth')
            model, results = train_task(task, model_path=model_path)
            all_results[task] = results
            
            print(f"\n{task} completed!")
            print(f"  Best Val Acc: {results['val']:.2f}%")
            if 'test1' in results:
                print(f"  Test1 Acc: {results['test1']:.2f}%")
            if 'test2' in results:
                print(f"  Test2 Acc: {results['test2']:.2f}%")
                
        except Exception as e:
            print(f"Error training {task}: {e}")
            import traceback
            traceback.print_exc()
            all_results[task] = {'error': str(e)}
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY RESULTS (CTNet)")
    print(f"{'='*80}")
    
    for task, results in all_results.items():
        if 'error' in results:
            print(f"\n{task}: FAILED - {results['error']}")
        else:
            print(f"\n{task}:")
            print(f"  Best Val Acc: {results['val']:.2f}%")
            if 'test1' in results:
                print(f"  Test1 Acc:    {results['test1']:.2f}%")
            if 'test2' in results:
                print(f"  Test2 Acc:    {results['test2']:.2f}%")
    
    print(f"\n{'='*80}")
    print("CTNet MULTI-TASK TRAINING COMPLETED!")
    print(f"{'='*80}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CTNet on EEG tasks')
    parser.add_argument('--task', type=str, default='SSVEP',
                        choices=['SSVEP', 'P300', 'MI', 'Imagined_speech', 'Lee2019_MI', 'Lee2019_SSVEP', 'BNCI2014_P300', 'all'],
                        help='Task to train on (default: SSVEP)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=44,
                        help='Random seed')
    
    args = parser.parse_args()
    
    config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'patience': 20,
        'scheduler': 'ReduceLROnPlateau',
        'seed': args.seed,
    }
    
    if args.task == 'all':
        results = train_all_tasks(save_dir=args.save_dir)
    else:
        model_path = os.path.join(args.save_dir, f'best_ctnet_{args.task.lower()}_model.pth')
        os.makedirs(args.save_dir, exist_ok=True)
        model, results = train_task(args.task, config=config, model_path=model_path)
