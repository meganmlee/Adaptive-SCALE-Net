"""
Test script for ChannelWiseSpectralCLDNN + SE Block
Uses best_cldnn_model.pth checkpoint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import signal
from tqdm import tqdm
import os
import random
from typing import Optional
from sklearn.metrics import confusion_matrix


# ==================== SE Block ====================

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ==================== Data Loading ====================

def load_ssvep_data(data_dir, num_seen=33, seed=44):
    """Load SSVEP data with subject-wise split"""
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"Loading SSVEP data from: {data_dir}")
    
    all_subjects = list(range(1, 36))
    random.shuffle(all_subjects)
    
    seen_subjects = all_subjects[:num_seen]
    unseen_subjects = all_subjects[num_seen:]
    
    print(f"Seen subjects ({len(seen_subjects)}): {sorted(seen_subjects)[:5]}...")
    print(f"Unseen subjects ({len(unseen_subjects)}): {unseen_subjects}")
    
    train_sessions = [0, 1, 2, 3]
    val_session = 4
    test1_session = 5
    
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test1, y_test1 = [], []
    X_test2, y_test2 = [], []
    
    for sid in seen_subjects:
        filepath = os.path.join(data_dir, f"S{sid}_chars.npy")
        if not os.path.exists(filepath):
            continue
        data = np.load(filepath)
        
        for char_idx in range(26):
            for sess in train_sessions:
                X_train.append(data[char_idx, sess])
                y_train.append(char_idx)
            X_val.append(data[char_idx, val_session])
            y_val.append(char_idx)
            X_test1.append(data[char_idx, test1_session])
            y_test1.append(char_idx)
    
    for sid in unseen_subjects:
        filepath = os.path.join(data_dir, f"S{sid}_chars.npy")
        if not os.path.exists(filepath):
            continue
        data = np.load(filepath)
        
        for char_idx in range(26):
            for sess in range(6):
                X_test2.append(data[char_idx, sess])
                y_test2.append(char_idx)
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test1, y_test1 = np.array(X_test1), np.array(y_test1)
    X_test2, y_test2 = np.array(X_test2), np.array(y_test2)
    
    return X_train, y_train, X_val, y_val, X_test1, y_test1, X_test2, y_test2


# ==================== STFT Transform ====================

def apply_stft_transform(data, fs=250, nperseg=128, noverlap=112, nfft=512):
    """Apply STFT transform to EEG data"""
    single_sample = False
    if len(data.shape) == 2:
        data = data[np.newaxis, :, :]
        single_sample = True
    
    n_samples, n_channels, n_times = data.shape
    stft_data = []
    
    for sample_idx in range(n_samples):
        channels_stft = []
        for ch in range(n_channels):
            f, t, Zxx = signal.stft(
                data[sample_idx, ch, :],
                fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft
            )
            power = np.abs(Zxx) ** 2
            channels_stft.append(power)
        stft_data.append(np.stack(channels_stft, axis=0))
    
    stft_data = np.array(stft_data)
    
    if single_sample:
        return stft_data[0]
    return stft_data


# ==================== Dataset ====================

class EEGDataset(Dataset):
    def __init__(self, data, labels, stft_config=None, normalize=True):
        self.data = data
        self.labels = torch.LongTensor(labels)
        self.normalize = normalize
        self.stft_config = stft_config or {
            'fs': 250, 'nperseg': 128, 'noverlap': 112, 'nfft': 512
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx].copy()
        y = self.labels[idx]
        
        x = apply_stft_transform(x, **self.stft_config)
        
        if self.normalize:
            mean = x.mean(axis=(1, 2), keepdims=True)
            std = x.std(axis=(1, 2), keepdims=True) + 1e-8
            x = (x - mean) / std
        
        return torch.FloatTensor(x), y


# ==================== SE Block Model ====================

class ChannelWiseSpectralCLDNN_SE(nn.Module):
    """
    ChannelWiseSpectralCLDNN with SE Block (Option B)
    - Two CNN stages with SE blocks
    """
    
    def __init__(self, freq_bins, time_bins, n_channels, n_classes, 
                 cnn_filters=16, lstm_hidden=128, pos_dim=16):
        super().__init__()
        self.n_channels = n_channels
        self.freq_bins = freq_bins
        self.time_bins = time_bins
        
        # ====== Stage 1: Conv(1→16) + SE + Pool ======
        self.conv1 = nn.Conv2d(1, cnn_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cnn_filters)
        self.se1 = SqueezeExcitation(cnn_filters, reduction=4)
        self.pool1 = nn.MaxPool2d(2)
        
        # ====== Stage 2: Conv(16→32) + SE + Pool ======
        self.conv2 = nn.Conv2d(cnn_filters, cnn_filters * 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cnn_filters * 2)
        self.se2 = SqueezeExcitation(cnn_filters * 2, reduction=4)
        self.pool2 = nn.MaxPool2d(2)
        
        # CNN output dimension (2x pooling)
        self.cnn_out_dim = (freq_bins // 4) * (time_bins // 4) * (cnn_filters * 2)
        
        # ====== Channel Position Embedding ======
        self.chan_emb = nn.Embedding(n_channels, pos_dim)
        self.pos_projection = nn.Linear(pos_dim, self.cnn_out_dim)
        
        # ====== LSTM across channels ======
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=False
        )
        
        # ====== Classifier ======
        self.classifier = nn.Linear(lstm_hidden, n_classes)
    
    def forward(self, x, chan_ids: Optional[torch.Tensor] = None):
        B, C, Fr, T = x.shape
        
        if C != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {C}")
        
        # Per-channel CNN with SE Block
        x = x.view(B * C, 1, Fr, T)
        
        # Stage 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.se1(x)
        x = self.pool1(x)
        
        # Stage 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.se2(x)
        x = self.pool2(x)
        
        x = x.view(B, C, -1)
        
        # Channel position embedding
        if chan_ids is None:
            chan_ids = torch.arange(C, device=x.device).unsqueeze(0).expand(B, C)
        
        pos = self.chan_emb(chan_ids)
        pos = self.pos_projection(pos)
        x = x + pos
        
        # LSTM
        _, (h, _) = self.lstm(x)
        
        return self.classifier(h.squeeze(0))


# ==================== Evaluation ====================

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Evaluating', ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return 100. * correct / total, np.array(all_preds), np.array(all_labels)


# ==================== Confusion Matrix ====================

def print_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Print confusion matrix as text in terminal
    Shows counts and percentages for each class
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(cm)
    class_labels = [chr(65 + i) for i in range(n_classes)]
    
    # Normalize to percentages
    cm_percent = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8) * 100
    
    print(f"\n{title}")
    
    # Calculate column width (need space for 3-digit numbers)
    col_width = 4  # Each cell is 4 characters wide
    
    # Total width: "True\\Pred" (9) + n_classes * col_width + " Total" (6)
    total_width = 9 + n_classes * col_width + 6
    print("=" * total_width)
    
    # Header: Predicted labels
    header = "True\\Pred"
    for label in class_labels:
        header += f"{label:>{col_width}s}"
    header += " Total"
    print(header)
    print("-" * total_width)
    
    # Each row: True label, predicted counts, total
    for i in range(n_classes):
        row = f"   {class_labels[i]:>3s}  "
        total_true = cm[i].sum()
        for j in range(n_classes):
            count = cm[i, j]
            row += f"{count:>{col_width}d}"
        row += f"{total_true:>6d}"
        print(row)
    
    print("-" * total_width)
    
    # Column totals
    col_totals = cm.sum(axis=0)
    row = "  Total  "
    for total in col_totals:
        row += f"{total:>{col_width}d}"
    row += f"{len(y_true):>6d}"
    print(row)
    print("=" * total_width)
    
    # Summary statistics
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = 100. * correct / total
    
    print(f"\nSummary:")
    print(f"  Total samples: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Overall accuracy: {accuracy:.2f}%")
    
    # Per-class accuracy
    print(f"\nPer-class accuracy:")
    for i in range(n_classes):
        if cm[i, i] > 0 or cm[i].sum() > 0:
            cls_total = cm[i].sum()
            cls_correct = cm[i, i]
            cls_acc = 100. * cls_correct / cls_total if cls_total > 0 else 0.0
            print(f"  Class {i:2d} ({class_labels[i]}): {cls_acc:5.1f}% ({cls_correct}/{cls_total})")


# ==================== Main Test ====================

def main():
    # Config
    config = {
        'data_dir': '/ocean/projects/cis250213p/shared/ssvep',
        'checkpoint': 'best_cldnn_model.pth',
        'num_seen': 33,
        'seed': 44,
        'batch_size': 32,
        
        # STFT
        'stft_fs': 250,
        'stft_nperseg': 128,
        'stft_noverlap': 112,
        'stft_nfft': 512,
        
        # Model (Original)
        'cnn_filters': 16,
        'lstm_hidden': 128,
        'pos_dim': 16,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Testing ChannelWiseSpectralCLDNN + SE Block")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Checkpoint: {config['checkpoint']}")
    
    # ====== Load Data ======
    X_train, y_train, X_val, y_val, X_test1, y_test1, X_test2, y_test2 = \
        load_ssvep_data(config['data_dir'], config['num_seen'], config['seed'])
    
    stft_config = {
        'fs': config['stft_fs'],
        'nperseg': config['stft_nperseg'],
        'noverlap': config['stft_noverlap'],
        'nfft': config['stft_nfft']
    }
    
    # ====== Create Datasets ======
    val_ds = EEGDataset(X_val, y_val, stft_config, normalize=True)
    test1_ds = EEGDataset(X_test1, y_test1, stft_config, normalize=True)
    test2_ds = EEGDataset(X_test2, y_test2, stft_config, normalize=True)
    
    # Get dimensions
    sample, _ = val_ds[0]
    n_channels, freq_bins, time_bins = sample.shape
    print(f"STFT shape: ({n_channels}, {freq_bins}, {time_bins})")
    
    # ====== Data Loaders ======
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=4, pin_memory=True)
    test1_loader = DataLoader(test1_ds, batch_size=config['batch_size'], 
                              shuffle=False, num_workers=4, pin_memory=True)
    test2_loader = DataLoader(test2_ds, batch_size=config['batch_size'], 
                              shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Val: {len(val_ds)}, Test1 (Seen): {len(test1_ds)}, Test2 (Unseen): {len(test2_ds)}")
    
    # ====== Create Model (SE Block) ======
    model = ChannelWiseSpectralCLDNN_SE(
        freq_bins=freq_bins,
        time_bins=time_bins,
        n_channels=n_channels,
        n_classes=26,
        cnn_filters=config['cnn_filters'],
        lstm_hidden=config['lstm_hidden'],
        pos_dim=config['pos_dim']
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {n_params:,}")
    
    # ====== Load Checkpoint ======
    print(f"\nLoading checkpoint: {config['checkpoint']}")
    checkpoint = torch.load(config['checkpoint'], map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best Val Acc: {checkpoint.get('best_val_acc', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
    
    print("✓ Checkpoint loaded successfully!")
    
    # ====== Evaluate ======
    print(f"\n{'='*70}")
    print("Evaluating...")
    print(f"{'='*70}")
    
    val_acc, val_preds, val_labels = evaluate(model, val_loader, device)
    test1_acc, test1_preds, test1_labels = evaluate(model, test1_loader, device)
    test2_acc, test2_preds, test2_labels = evaluate(model, test2_loader, device)
    
    # ====== Results ======
    print(f"\n{'='*70}")
    print(f"RESULTS (CLDNN + SE Block)")
    print(f"{'='*70}")
    print(f"Validation Acc:  {val_acc:.2f}%")
    print(f"Test1 (Seen):    {test1_acc:.2f}%")
    print(f"Test2 (Unseen):  {test2_acc:.2f}%")
    print(f"{'='*70}")
    
    # ====== Confusion Matrix ======
    print_confusion_matrix(
        test1_labels, test1_preds, 
        title="Confusion Matrix - Test1 (Seen Subjects)"
    )
    
    return {
        'val_acc': val_acc,
        'test1_acc': test1_acc,
        'test2_acc': test2_acc
    }


if __name__ == "__main__":
    results = main()

