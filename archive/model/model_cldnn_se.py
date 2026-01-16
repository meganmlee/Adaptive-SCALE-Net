"""
ChannelWiseSpectralCLDNN + SE Block - for SSVEP
Based on the working implementation from spectral_cnn_lstm notebooks
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


# ==================== Data Loading (Subject-wise Split) ====================

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
    
    print(f"\nData shapes:")
    print(f"  Train: {X_train.shape}, labels: {y_train.min()}-{y_train.max()}")
    print(f"  Val: {X_val.shape}, labels: {y_val.min()}-{y_val.max()}")
    print(f"  Test1 (seen): {X_test1.shape}, labels: {y_test1.min()}-{y_test1.max()}")
    print(f"  Test2 (unseen): {X_test2.shape}, labels: {y_test2.min()}-{y_test2.max()}")
    
    return X_train, y_train, X_val, y_val, X_test1, y_test1, X_test2, y_test2


# ==================== STFT Transformation ====================

def apply_stft_transform(data, fs=250, nperseg=128, noverlap=112, nfft=512):
    """
    Apply STFT transform to EEG data
    
    Args:
        data: (samples, channels, time) or (channels, time)
        
    Returns:
        (samples, channels, freq_bins, time_bins) or (channels, freq_bins, time_bins)
    """
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
            power = np.abs(Zxx) ** 2  # No log compression (like spectral_cnn_lstm)
            channels_stft.append(power)
        stft_data.append(np.stack(channels_stft, axis=0))
    
    stft_data = np.array(stft_data)
    
    if single_sample:
        return stft_data[0]
    return stft_data


# ==================== Dataset ====================

class EEGDataset(Dataset):
    """EEG Dataset with STFT transform"""
    
    def __init__(self, data, labels, stft_config=None, normalize=True, augment=False):
        self.data = data
        self.labels = torch.LongTensor(labels)
        self.normalize = normalize
        self.augment = augment
        
        # STFT config (same as spectral_cnn_lstm)
        self.stft_config = stft_config or {
            'fs': 250,
            'nperseg': 128,
            'noverlap': 112,
            'nfft': 512
        }
    
    def __len__(self):
        return len(self.data)
    
    def _augment_raw(self, x):
        """Augmentation on raw EEG"""
        # Noise injection
        if np.random.random() < 0.5:
            x = x + np.random.randn(*x.shape) * 0.05 * np.std(x)
        # Amplitude scaling
        if np.random.random() < 0.5:
            x = x * np.random.uniform(0.8, 1.2)
        # Time shift
        if np.random.random() < 0.3:
            x = np.roll(x, np.random.randint(-5, 6), axis=-1)
        return x
    
    def _augment_stft(self, x):
        """Augmentation on STFT"""
        # Frequency masking
        if np.random.random() < 0.3:
            n_freq = x.shape[1]
            f_mask = np.random.randint(1, max(2, n_freq // 10))
            f_start = np.random.randint(0, n_freq - f_mask)
            x[:, f_start:f_start+f_mask, :] = 0
        # Time masking
        if np.random.random() < 0.3:
            n_time = x.shape[2]
            t_mask = np.random.randint(1, max(2, n_time // 5))
            t_start = np.random.randint(0, n_time - t_mask)
            x[:, :, t_start:t_start+t_mask] = 0
        return x
    
    def __getitem__(self, idx):
        x = self.data[idx].copy()
        y = self.labels[idx]
        
        # Raw augmentation
        if self.augment:
            x = self._augment_raw(x)
        
        # STFT transform
        x = apply_stft_transform(
            x, **self.stft_config
        )
        
        # STFT augmentation
        if self.augment:
            x = self._augment_stft(x)
        
        # Normalize (per-channel, freq×time)
        if self.normalize:
            mean = x.mean(axis=(1, 2), keepdims=True)
            std = x.std(axis=(1, 2), keepdims=True) + 1e-8
            x = (x - mean) / std
        
        return torch.FloatTensor(x), y


# ==================== SE Block ====================

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation Block
    """
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

class ChannelWiseSpectralCLDNN(nn.Module):
    """
    Channel-wise Spectral CNN-LSTM-DNN model
    """
    
    def __init__(self, freq_bins, time_bins, n_channels, n_classes, 
                 cnn_filters=16, lstm_hidden=128, pos_dim=16):
        super().__init__()
        self.n_channels = n_channels
        self.freq_bins = freq_bins
        self.time_bins = time_bins
        
        # ====== Per-channel CNN with SE Block (weight sharing) ======
        # Treat each channel's STFT (freq × time) as an "image"
        
        # Stage 1: Conv(1→16) + SE + Pool
        self.conv1 = nn.Conv2d(1, cnn_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cnn_filters)
        self.se1 = SqueezeExcitation(cnn_filters, reduction=4)
        self.pool1 = nn.MaxPool2d(2)  # (F, T) → (F/2, T/2)
        
        # Stage 2: Conv(16→32) + SE + Pool
        self.conv2 = nn.Conv2d(cnn_filters, cnn_filters * 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cnn_filters * 2)
        self.se2 = SqueezeExcitation(cnn_filters * 2, reduction=4)
        self.pool2 = nn.MaxPool2d(2)  # (F/2, T/2) → (F/4, T/4)
        
        # Calculate CNN output dimension (2 pooling operations applied)
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
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, chan_ids: Optional[torch.Tensor] = None):
        """
        Args:
            x: (B, C, F, T) - Batch, Channels, Freq, Time
            chan_ids: Optional channel indices (default: 0 to C-1)
            
        Returns:
            (B, n_classes) - Classification logits
        """
        B, C, Fr, T = x.shape
        
        if C != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {C}")
        
        # ====== Step 1: Per-channel CNN with SE Block ======
        # (B, C, Fr, T) → (B*C, 1, Fr, T) - treat each channel as an individual image
        x = x.view(B * C, 1, Fr, T)
        
        # Stage 1: Conv → BN → ReLU → SE → Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.se1(x)
        x = self.pool1(x)  # (B*C, 16, F/2, T/2)
        
        # Stage 2: Conv → BN → ReLU → SE → Pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.se2(x)
        x = self.pool2(x)  # (B*C, 32, F/4, T/4)
        
        # Flatten
        x = x.view(B, C, -1)  # (B, C, cnn_out_dim)
        
        # ====== Step 2: Add channel position embedding ======
        if chan_ids is None:
            chan_ids = torch.arange(C, device=x.device).unsqueeze(0).expand(B, C)
        
        pos = self.chan_emb(chan_ids)
        pos = self.pos_projection(pos)
        x = x + pos
        
        # ====== Step 3: LSTM across channels ======
        _, (h, _) = self.lstm(x)
        
        # ====== Step 4: Classify ======
        return self.classifier(h.squeeze(0))

# ==================== Training Functions ====================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    pbar = tqdm(loader, desc='Train', ncols=100)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{total_loss/(pbar.n+1):.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Eval', ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    
    return 100. * correct / total


# ==================== Main Training ====================

def train_model(config=None):
    """Main training function"""
    
    if config is None:
        config = {
            'data_dir': '/ocean/projects/cis250213p/shared/ssvep',
            'num_seen': 33,
            'seed': 44,
            
            # Model
            'cnn_filters': 16,
            'lstm_hidden': 128,
            'pos_dim': 16,
            
            # STFT (same as spectral_cnn_lstm baseline)
            'stft_fs': 250,
            'stft_nperseg': 128,
            'stft_noverlap': 112,
            'stft_nfft': 512,
            
            # Training
            'batch_size': 32,
            'num_epochs': 100,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'patience': 15,
            'scheduler': 'CosineAnnealingLR',
        }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"ChannelWiseSpectralCLDNN - SSVEP Classification")
    print(f"{'='*70}")
    print(f"Device: {device}")
    
    # ====== Load Data ======
    X_train, y_train, X_val, y_val, X_test1, y_test1, X_test2, y_test2 = \
        load_ssvep_data(config['data_dir'], config['num_seen'], config['seed'])
    
    # STFT config
    stft_config = {
        'fs': config['stft_fs'],
        'nperseg': config['stft_nperseg'],
        'noverlap': config['stft_noverlap'],
        'nfft': config['stft_nfft']
    }
    
    # ====== Create Datasets ======
    train_ds = EEGDataset(X_train, y_train, stft_config, normalize=True, augment=True)
    val_ds = EEGDataset(X_val, y_val, stft_config, normalize=True, augment=False)
    test1_ds = EEGDataset(X_test1, y_test1, stft_config, normalize=True, augment=False)
    test2_ds = EEGDataset(X_test2, y_test2, stft_config, normalize=True, augment=False)
    
    # Get dimensions from a sample
    sample, _ = train_ds[0]
    n_channels, freq_bins, time_bins = sample.shape
    print(f"STFT shape: ({n_channels}, {freq_bins}, {time_bins})")
    
    # ====== Data Loaders ======
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=4, pin_memory=True)
    test1_loader = DataLoader(test1_ds, batch_size=config['batch_size'], 
                              shuffle=False, num_workers=4, pin_memory=True)
    test2_loader = DataLoader(test2_ds, batch_size=config['batch_size'], 
                              shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test1: {len(test1_ds)}, Test2: {len(test2_ds)}")
    
    # ====== Create Model ======
    model = ChannelWiseSpectralCLDNN(
        freq_bins=freq_bins,
        time_bins=time_bins,
        n_channels=n_channels,
        n_classes=26,
        cnn_filters=config['cnn_filters'],
        lstm_hidden=config['lstm_hidden'],
        pos_dim=config['pos_dim']
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # ====== Loss & Optimizer ======
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['num_epochs'] // 2, eta_min=1e-6
        )
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
    else:
        raise ValueError(f"Invalid scheduler: {config['scheduler']}")
    
    # ====== Training Loop ======
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch [{epoch+1}/{config['num_epochs']}]")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_acc)
        else:
            raise ValueError(f"Invalid scheduler: {config['scheduler']}")
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, 'best_cldnn_se_model.pth')
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
    checkpoint = torch.load('best_cldnn_se_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test1_acc = evaluate(model, test1_loader, device)
    test2_acc = evaluate(model, test2_loader, device)
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Best Val Acc:    {best_val_acc:.2f}%")
    print(f"Test1 (Seen):    {test1_acc:.2f}%")
    print(f"Test2 (Unseen):  {test2_acc:.2f}%")
    print(f"Gap:             {test1_acc - test2_acc:.2f}%")
    print(f"{'='*70}")
    
    return model, {'val': best_val_acc, 'test1': test1_acc, 'test2': test2_acc}


if __name__ == "__main__":
    model, results = train_model()
