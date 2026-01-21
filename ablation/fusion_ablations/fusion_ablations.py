"""
Standalone Fusion Ablation Study for EEG Classification

Self-contained script to test 6 different fusion strategies:
1. static - Equal 0.5/0.5 weighting
2. global_attention - Trial-level dynamic weighting
3. spatial_attention - Channel-level dynamic weighting  
4. glu - Gated Linear Unit (noise suppression)
5. multiplicative - Element-wise feature interaction
6. bilinear - Full pairwise interaction

No dependencies on other model files - completely standalone!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import pandas as pd
from tqdm import tqdm
from typing import Dict, Optional
from sklearn.metrics import f1_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import mne

# Import only utilities (assumed available)
scale_net_path = os.path.join(os.path.dirname(__file__), '..', '..', 'scale_net')
sys.path.insert(0, scale_net_path)

# Import from scale-net
from seed_utils import seed_everything
from dataset import load_dataset, TASK_CONFIGS, create_dataloaders

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


# ==================== Fusion Strategies ====================

class FusionAblations(nn.Module):
    """
    Implements 6 strategies for multimodal fusion ablation:
    1. 'static': Linear equal weighting (0.5/0.5)
    2. 'global_attention': Trial-level dynamic weighting
    3. 'spatial_attention': Channel-level dynamic weighting
    4. 'glu': Gated Linear Unit for noise suppression
    5. 'multiplicative': Simple element-wise product interaction
    6. 'bilinear': Full pairwise feature interaction
    """
    def __init__(self, mode, dim, n_channels=None, temperature=2.0, rank=16):
        super().__init__()
        self.mode = mode
        self.dim = dim
        self.temperature = temperature
        
        if mode == 'global_attention' or mode == 'spatial_attention':
            self.attn = nn.Sequential(
                nn.Linear(dim*2, dim), 
                nn.ReLU(), 
                nn.Linear(dim, 2)
            )
        elif mode == 'glu':
            self.gate = nn.Linear(dim, dim)
        elif mode == 'multiplicative':
            self.proj_s = nn.Linear(dim, dim)
            self.proj_t = nn.Linear(dim, dim)
        elif mode == 'bilinear':
            self.rank = rank
            self.U_s = nn.Linear(dim, rank, bias=False)
            self.U_t = nn.Linear(dim, rank, bias=False)
            self.out_proj = nn.Linear(rank, dim)

    def forward(self, x_s, x_t):
        if self.mode == 'static':
            return 0.5 * x_s + 0.5 * x_t, None
            
        elif self.mode == 'global_attention':
            # Average across channels to get trial-level context
            ctx = torch.cat([x_s.mean(1), x_t.mean(1)], dim=-1)
            attn_logits = self.attn(ctx) / self.temperature
            w = torch.softmax(attn_logits, dim=-1)  # (B, 2)
            # Same weights for all channels in a trial
            return w[:, 0].view(-1, 1, 1) * x_s + w[:, 1].view(-1, 1, 1) * x_t, w
            
        elif self.mode == 'spatial_attention':
            # Keep channel dimension for channel-specific weighting
            ctx = torch.cat([x_s, x_t], dim=-1)  # (B, C, dim*2)
            attn_logits = self.attn(ctx) / self.temperature  # (B, C, 2)
            w = torch.softmax(attn_logits, dim=-1)  # (B, C, 2)
            # Different weights per channel
            return w[:, :, 0].unsqueeze(-1) * x_s + w[:, :, 1].unsqueeze(-1) * x_t, w
            
        elif self.mode == 'glu':
            combined = x_s + x_t
            gate_val = torch.sigmoid(self.gate(combined))
            return combined * gate_val, gate_val

        elif self.mode == 'multiplicative':
            return self.proj_s(x_s) * self.proj_t(x_t), None
            
        elif self.mode == 'bilinear':
            z_s = self.U_s(x_s)      # (B, C, rank)
            z_t = self.U_t(x_t)      # (B, C, rank)
            z = z_s * z_t            # Element-wise product
            fused = self.out_proj(z) # (B, C, dim)
            return fused, None


# ==================== Dual-Stream EEG Model ====================

class DualStreamEEGModel(nn.Module):
    """
    Dual-stream EEG model with configurable fusion strategy
    
    Architecture:
    - Spectral Stream: 2D CNN on STFT features
    - Temporal Stream: EEGNet-style 1D CNN on raw EEG
    - Fusion: One of 6 strategies (configurable)
    - LSTM: Temporal aggregation across channels
    - Classifier: Final prediction
    """
    
    def __init__(self, freq_bins, time_bins, n_channels, n_classes, T_raw,
                 cnn_filters=16, lstm_hidden=128, pos_dim=16,
                 dropout=0.3, cnn_dropout=0.2, use_hidden_layer=False, hidden_dim=64,
                 fusion_mode='global_attention', fusion_temperature=2.0, fusion_rank=16):
        
        super().__init__()
        self.n_channels = n_channels
        self.T_raw = T_raw
        self.fusion_mode = fusion_mode
        
        # ====== SPECTRAL STREAM (2D CNN) ======
        self.spec_cnn_filters = cnn_filters * 2
        
        # Stage 1: Conv(1→16) + BN + ReLU + SE + Dropout + Pool
        self.spec_conv1 = nn.Conv2d(1, cnn_filters, kernel_size=7, padding=3, bias=False)
        self.spec_bn1 = nn.BatchNorm2d(cnn_filters)
        self.spec_se1 = SqueezeExcitation(cnn_filters, reduction=4)
        self.spec_dropout_cnn1 = nn.Dropout2d(cnn_dropout)
        self.spec_pool1 = nn.MaxPool2d(2)
        
        # Stage 2: Conv(16→32) + BN + ReLU + SE + Dropout + Pool
        self.spec_conv2 = nn.Conv2d(cnn_filters, self.spec_cnn_filters, kernel_size=5, padding=2, bias=False)
        self.spec_bn2 = nn.BatchNorm2d(self.spec_cnn_filters)
        self.spec_se2 = SqueezeExcitation(self.spec_cnn_filters, reduction=4)
        self.spec_dropout_cnn2 = nn.Dropout2d(cnn_dropout)
        self.spec_pool2 = nn.MaxPool2d(2)
        
        # Spectral CNN Output Dimension
        self.spec_out_dim = (freq_bins // 4) * (time_bins // 4) * self.spec_cnn_filters
        
        # ====== TEMPORAL STREAM (EEGNet Inspired) ======
        F1 = 8
        D = 2
        F2 = F1 * D
        
        # Layer 1: Temporal Conv
        self.temp_conv = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn_temp = nn.BatchNorm2d(F1)
        
        # Layer 2: Spatial Conv (Depthwise)
        self.spatial_conv = nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False)
        self.bn_spatial = nn.BatchNorm2d(F2)
        self.pool_spatial = nn.AvgPool2d((1, 4))
        
        # Layer 3: Separable Conv
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(cnn_dropout)
        )
        
        # Calculate temporal stream output dimension
        final_time_dim = (T_raw // 4) // 8
        self.time_out_dim = F2 * final_time_dim

        # ====== FEATURE PROJECTION ======
        self.proj_spec = nn.Linear(self.spec_out_dim, lstm_hidden)
        self.proj_time = nn.Linear(self.time_out_dim, lstm_hidden)

        # ====== FUSION LAYER ======
        self.fusion_layer = FusionAblations(
            mode=fusion_mode,
            dim=lstm_hidden,
            n_channels=n_channels,
            temperature=fusion_temperature,
            rank=fusion_rank
        )

        # Channel Position Embedding
        self.chan_emb = nn.Embedding(n_channels, lstm_hidden)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=False,
            dropout=0
        )
        self.dropout_lstm = nn.Dropout(dropout)
        
        # Classifier
        self.use_hidden_layer = use_hidden_layer
        if use_hidden_layer:
            self.hidden_layer = nn.Sequential(
                nn.Linear(lstm_hidden, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            classifier_input = hidden_dim
        else:
            classifier_input = lstm_hidden
            
        self.classifier = nn.Linear(classifier_input, 1 if n_classes==2 else n_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_time, x_spec, chan_ids=None):
        B, C, _, _ = x_spec.shape

        # ====== 1. SPECTRAL STREAM ======
        x_spec_feat = x_spec.view(B * C, 1, x_spec.size(2), x_spec.size(3))
        
        # Stage 1
        x_spec_feat = self.spec_conv1(x_spec_feat)
        x_spec_feat = self.spec_bn1(x_spec_feat)
        x_spec_feat = F.relu(x_spec_feat, inplace=True)
        x_spec_feat = self.spec_se1(x_spec_feat)
        x_spec_feat = self.spec_dropout_cnn1(x_spec_feat)
        x_spec_feat = self.spec_pool1(x_spec_feat)
        
        # Stage 2
        x_spec_feat = self.spec_conv2(x_spec_feat)
        x_spec_feat = self.spec_bn2(x_spec_feat)
        x_spec_feat = F.relu(x_spec_feat, inplace=True)
        x_spec_feat = self.spec_se2(x_spec_feat)
        x_spec_feat = self.spec_dropout_cnn2(x_spec_feat)
        x_spec_feat = self.spec_pool2(x_spec_feat)
        
        x_spec_feat = x_spec_feat.view(B, C, -1)

        # ====== 2. TEMPORAL STREAM ======
        x_global = x_time.unsqueeze(1)
        
        x_global = self.temp_conv(x_global)
        x_global = self.bn_temp(x_global)
        
        x_global = self.spatial_conv(x_global)
        x_global = F.relu(self.bn_spatial(x_global))
        x_global = self.pool_spatial(x_global)
        
        x_global = self.separable_conv(x_global)
        x_global = x_global.view(B, -1)
        
        # ====== 3. FEATURE PROJECTION ======
        x_spec_proj = self.proj_spec(x_spec_feat)  # (B, C, lstm_hidden)
        x_global_proj = self.proj_time(x_global.unsqueeze(1).expand(-1, C, -1))  # (B, C, lstm_hidden)
        
        # ====== 4. FUSION ======
        features, weights = self.fusion_layer(x_spec_proj, x_global_proj)
        
        # ====== 5. LSTM AND CLASSIFICATION ======
        if chan_ids is None:
            chan_ids = torch.arange(C, device=features.device).unsqueeze(0).expand(B, C)
        features = features + self.chan_emb(chan_ids)
        
        _, (h, _) = self.lstm(features)
        h = h.squeeze(0)
        h = self.dropout_lstm(h)
        
        if self.use_hidden_layer:
            h = self.hidden_layer(h)
        
        return self.classifier(h), weights


# ==================== Training Functions ====================

def train_epoch(model, loader, criterion, optimizer, device, is_binary=False):
    """Train for one epoch"""
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    pbar = tqdm(loader, desc='Train', ncols=100, leave=False)
    for (x_time, x_spec), labels in pbar:
        x_time, x_spec, labels = x_time.to(device), x_spec.to(device), labels.to(device)
        
        if is_binary:
            labels_float = labels.float().unsqueeze(1)
        else:
            labels_float = labels
        
        optimizer.zero_grad()
        outputs, _ = model(x_time, x_spec)
        loss = criterion(outputs, labels_float)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if is_binary:
            preds = (torch.sigmoid(outputs) > 0.5).long().squeeze()
        else:
            preds = outputs.argmax(dim=1)
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
    
    return total_loss / len(loader), 100 * correct / total


def evaluate(model, loader, device, criterion, is_binary=False):
    """Evaluate model"""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for (x_time, x_spec), labels in loader:
            x_time, x_spec, labels = x_time.to(device), x_spec.to(device), labels.to(device)
            
            if is_binary:
                labels_float = labels.float().unsqueeze(1)
            else:
                labels_float = labels
            
            outputs, _ = model(x_time, x_spec)
            loss = criterion(outputs, labels_float)
            total_loss += loss.item()
            
            if is_binary:
                preds = (torch.sigmoid(outputs) > 0.5).long().squeeze()
            else:
                preds = outputs.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(loader), 100 * correct / total


def evaluate_comprehensive(model, loader, device, is_binary=False):
    """Evaluate with additional metrics (F1, Recall, AUC)"""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for (x_time, x_spec), labels in loader:
            x_time, x_spec = x_time.to(device), x_spec.to(device)
            outputs, _ = model(x_time, x_spec)
            
            if is_binary:
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
            else:
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
            
            all_preds.extend(preds.flatten())
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    avg_type = 'binary' if is_binary else 'macro'
    metrics = {
        'acc': np.mean(np.array(all_preds) == np.array(all_labels)) * 100,
        'f1': f1_score(all_labels, all_preds, average=avg_type) * 100,
        'recall': recall_score(all_labels, all_preds, average=avg_type) * 100,
    }
    
    try:
        if is_binary:
            metrics['auc'] = roc_auc_score(all_labels, all_probs) * 100
        else:
            metrics['auc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr') * 100
    except:
        metrics['auc'] = None
    
    return metrics


# ==================== Feature Statistics Collection ====================

def collect_feature_statistics(model, loader, device, fusion_mode):
    """Collect fusion statistics and attention weights"""
    model.eval()
    stats = {
        'spec_contribution': [], 
        'time_contribution': [],
        'labels': [], 
        'predictions': [], 
        'confidences': []
    }
    
    if fusion_mode in ['global_attention', 'spatial_attention']:
        stats['attn_weights'] = []
    if fusion_mode == 'spatial_attention':
        stats['spatial_weights'] = []
    elif fusion_mode == 'glu':
        stats['gate_sparsity'] = []
        stats['gate_values'] = []
    
    with torch.no_grad():
        for (x_time, x_spec), labels in tqdm(loader, desc='Collecting Stats', ncols=100, leave=False):
            x_time, x_spec = x_time.to(device), x_spec.to(device)
            B, C = x_spec.shape[0], x_spec.shape[1]
            
            # Get projected features manually to calculate contributions
            x_spec_f = x_spec.view(B * C, 1, x_spec.size(2), x_spec.size(3))
            x_spec_f = model.spec_conv1(x_spec_f)
            x_spec_f = model.spec_bn1(x_spec_f)
            x_spec_f = F.relu(x_spec_f)
            x_spec_f = model.spec_se1(x_spec_f)
            x_spec_f = model.spec_pool1(x_spec_f)
            x_spec_f = model.spec_conv2(x_spec_f)
            x_spec_f = model.spec_bn2(x_spec_f)
            x_spec_f = F.relu(x_spec_f)
            x_spec_f = model.spec_se2(x_spec_f)
            x_spec_f = model.spec_pool2(x_spec_f)
            x_spec_proj = model.proj_spec(x_spec_f.view(B, C, -1))
            
            x_g = x_time.unsqueeze(1)
            x_g = model.temp_conv(x_g)
            x_g = model.bn_temp(x_g)
            x_g = model.spatial_conv(x_g)
            x_g = F.relu(model.bn_spatial(x_g))
            x_g = model.pool_spatial(x_g)
            x_g = model.separable_conv(x_g)
            x_global_proj = model.proj_time(x_g.view(B, -1).unsqueeze(1).expand(-1, C, -1))
            
            _, weights = model.fusion_layer(x_spec_proj, x_global_proj)

            if fusion_mode == 'spatial_attention' and weights is not None:
                # weights shape is (B, C, 2) - take spectral weights
                stats['spatial_weights'].append(weights[:, :, 0].cpu().numpy())
            
            # Calculate contribution magnitudes
            spec_mag = torch.norm(x_spec_proj, dim=-1)
            time_mag = torch.norm(x_global_proj, dim=-1)
            total = spec_mag + time_mag + 1e-8
            
            stats['spec_contribution'].append((spec_mag / total).cpu().numpy())
            stats['time_contribution'].append((time_mag / total).cpu().numpy())
            
            # Get predictions
            outputs, _ = model(x_time, x_spec)
            if outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs)
            else:
                probs = torch.softmax(outputs, dim=1)
            stats['confidences'].append(probs.max(dim=1)[0].cpu().numpy())
            stats['labels'].append(labels.numpy())
            
            # Store weights
            if weights is not None:
                if 'attn_weights' in stats:
                    stats['attn_weights'].append(weights.cpu().numpy())
                elif 'gate_values' in stats:
                    stats['gate_values'].append(weights.cpu().numpy())
                    stats['gate_sparsity'].append((weights < 0.1).float().mean().cpu().item())

    # Concatenate all arrays
    return {
        k: np.concatenate(v, axis=0) if (len(v) > 0 and isinstance(v[0], np.ndarray))
        else np.array(v)
        for k, v in stats.items()
    }


def analyze_fusion_statistics(stats, fusion_mode, task):
    """Analyze and save fusion statistics"""
    save_dir = f'./ablation_{task}/analysis'
    os.makedirs(save_dir, exist_ok=True)
    
    avg_spec = stats['spec_contribution'].mean()
    avg_time = stats['time_contribution'].mean()
    
    analysis = {
        'fusion_mode': fusion_mode,
        'spectral_dominance': avg_spec,
        'temporal_dominance': avg_time,
        'primary_modality': 'Spectral' if avg_spec > avg_time else 'Temporal'
    }
    
    if 'attn_weights' in stats:
        analysis['attn_variance'] = stats['attn_weights'].var()
    if 'gate_sparsity' in stats:
        analysis['noise_suppression'] = np.mean(stats['gate_sparsity'])
    
    pd.DataFrame([analysis]).to_csv(f'{save_dir}/{task}_{fusion_mode}_usage.csv', index=False)
    return analysis


# ==================== Visualization ====================

def plot_modality_contributions(task, results_df):
    """Plot modality contribution balance across strategies"""
    save_dir = f'./ablation_{task}/plots'
    os.makedirs(save_dir, exist_ok=True)
    
    plot_data = []
    for _, row in results_df.iterrows():
        if 'spectral_dominance' in row and 'temporal_dominance' in row:
            plot_data.append({
                'Strategy': row['strategy'],
                'Contribution': row['spectral_dominance'],
                'Modality': 'Spectral'
            })
            plot_data.append({
                'Strategy': row['strategy'],
                'Contribution': row['temporal_dominance'],
                'Modality': 'Temporal'
            })
    
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    df = pd.DataFrame(plot_data)
    sns.barplot(x='Strategy', y='Contribution', hue='Modality', data=df, palette='viridis')
    plt.title(f'Modality Usage Balance - {task}', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.ylabel('Contribution', fontsize=12)
    plt.xlabel('Fusion Strategy', fontsize=12)
    
    # Add accuracy annotations
    for i, row in results_df.iterrows():
        if 'val_acc' in row:
            plt.text(i, 0.95, f"{row['val_acc']:.1f}%", 
                    ha='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{task}_modality_balance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved modality balance plot to {save_dir}/{task}_modality_balance.png")


def plot_spatial_attention_heatmap(task, stats, n_channels):
    """
    Plots both a topographical brain map and a named grid map of attention weights.
    Categorizes channel types to prevent overlapping position errors in MNE.
    """
    if 'spatial_weights' not in stats:
        print(f"No spatial weights found for {task}. Skipping heatmap.")
        return

    save_dir = f'./ablation_{task}/plots'
    os.makedirs(save_dir, exist_ok=True)
    
    # Average weights across all trials in the test set
    mean_weights = stats['spatial_weights'].mean(axis=0)

    if n_channels == 64 or n_channels == 62: 
        # Base 60 EEG channels common to both
        eeg_names = [
            'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 
            'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 
            'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 
            'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 
            'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 
            'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2'
        ]
        
        if n_channels == 64: # Wang2016 specific additions
            ch_names = eeg_names + ['CB1', 'CB2', 'VEO', 'HEO']
        else: # Lee2019 specific (62 channels total)
            ch_names = eeg_names + ['VEO', 'HEO']
            
        # Assign types: EEG for the first 60, EOG/Misc for the rest
        ch_types = ['eeg'] * 60 + ['eog'] * (n_channels - 60)
            
    elif n_channels == 22:  # MI: BNCI2014_001
        ch_names = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
            'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz',
            'P2', 'POz'
        ]
        ch_types = ['eeg'] * 22
        
    elif n_channels == 32:  # P300: BI2014B
        ch_names = [
            'Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 
            'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 
            'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO3', 'PO4', 'O1', 
            'Oz', 'O2'
        ]
        ch_types = ['eeg'] * 32

    elif n_channels == 16: # P300: BNCI2014_009
        ch_names = [
            'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'Oz', 'F3', 'F4', 
            'C3', 'C4', 'P3', 'P4', 'PO7', 'PO8', 'O1', 'O2'
        ]
        ch_types = ['eeg'] * 16

    else:
        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
        ch_types = ['eeg'] * n_channels

    # ==================== PART 1: MNE BRAIN TOPOMAP ====================
    # Create info with specific types to allow MNE to ignore eye/neck channels
    info = mne.create_info(ch_names=ch_names[:n_channels], sfreq=250, ch_types=ch_types)
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, on_missing='ignore')

    # Pick only 'eeg' types to avoid the "overlapping positions" error
    eeg_picks = mne.pick_types(info, eeg=True, eog=False, misc=False)
    filtered_weights = mean_weights[eeg_picks]
    filtered_info = mne.pick_info(info, sel=eeg_picks)

    fig, ax = plt.subplots(figsize=(7, 7))
    im, _ = mne.viz.plot_topomap(
        filtered_weights, 
        filtered_info,
        axes=ax, 
        show=False, 
        cmap='YlGnBu', 
        contours=4
    )
    plt.colorbar(im, ax=ax, label='Mean Spectral Attention Weight')
    ax.set_title(f'Spatial Attention Topomap: {task}\n({len(eeg_picks)} Scalp Channels)')
    plt.savefig(f'{save_dir}/{task}_brain_topomap.png', dpi=300)
    plt.close()

    # ==================== PART 2: NAMED GRID MAP ====================
    # Calculate grid size (e.g., 5x5 for 22ch, 6x6 for 32ch, 8x8 for 64ch)
    cols = int(np.ceil(np.sqrt(n_channels)))
    rows = int(np.ceil(n_channels / cols))
    
    # Pad weights and names to match rectangular grid
    padded_weights = np.zeros(rows * cols)
    padded_weights[:n_channels] = mean_weights
    
    padded_names = [""] * (rows * cols)
    for i in range(n_channels):
        padded_names[i] = ch_names[i]

    # Create annotation labels (Electrode Name + Weight Value)
    labels = np.array([f"{name}\n{val:.2f}" if name else "" 
                      for name, val in zip(padded_names, padded_weights)]).reshape(rows, cols)
    heatmap_data = padded_weights.reshape(rows, cols)

    plt.figure(figsize=(cols * 1.5, rows * 1.5))
    sns.heatmap(heatmap_data, annot=labels, fmt="", cmap='YlGnBu', 
                cbar_kws={'label': 'Mean Attention Weight'},
                xticklabels=False, yticklabels=False)
    
    plt.title(f'Spatial Attention Channel Grid: {task}', fontsize=14, pad=20)
    plt.savefig(f'{save_dir}/{task}_spatial_grid_named.png', bbox_inches='tight', dpi=300)
    print(f"✓ Saved brain topomap and named grid map to {save_dir}")
    plt.close()


# ==================== Main Ablation Study ====================

def run_ablation_study(task: str, config: Dict = None):
    """
    Run fusion strategy ablation study for a given task
    
    Args:
        task: Task name (e.g., 'SSVEP', 'P300', 'MI', etc.)
        config: Configuration dictionary with training/model parameters
    """
    strategies = ['static', 'global_attention', 'spatial_attention', 'glu', 'multiplicative', 'bilinear']
    
    if config is None:
        config = {
            'num_epochs': 40,
            'batch_size': 64,
            'lr': 1e-3,
            'seed': 44,
            'cnn_filters': 16,
            'lstm_hidden': 128,
            'pos_dim': 16,
            'dropout': 0.3,
            'cnn_dropout': 0.2,
            'use_hidden_layer': True,
            'hidden_dim': 64,
            'patience': 5
        }

    # Get task configuration
    task_config = TASK_CONFIGS.get(task, {})
    n_classes = task_config.get('num_classes', 26)
    is_binary = (n_classes == 2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"FUSION ABLATION STUDY: {task}")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Testing {len(strategies)} fusion strategies")
    print(f"{'='*70}\n")
    
    results_log = []
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"STRATEGY: {strategy.upper()}")
        print(f"{'='*60}")
        
        seed_everything(config['seed'], deterministic=True)
        
        # Load dataset
        datasets = load_dataset(
            task=task,
            data_dir=task_config.get('data_dir'),
            seed=config['seed']
        )

        # Auto-determine fusion rank based on dataset size
        n_train_samples = len(datasets['train'][0])
        if n_train_samples < 1000:
            fusion_rank = 8
        elif n_train_samples < 5000:
            fusion_rank = 16
        else:
            fusion_rank = 32
        
        # STFT config
        stft_config = {
            'fs': config.get('stft_fs', task_config.get('sampling_rate', 250)),
            'nperseg': config.get('stft_nperseg', task_config.get('stft_nperseg', 128)),
            'noverlap': config.get('stft_noverlap', task_config.get('stft_noverlap', 112)),
            'nfft': config.get('stft_nfft', task_config.get('stft_nfft', 512))
        }
        
        loaders = create_dataloaders(datasets, stft_config, batch_size=config['batch_size'])
        
        # Get dimensions
        sample_x, _ = next(iter(loaders['train']))
        _, n_channels, T_raw = sample_x[0].shape
        _, _, freq_bins, time_bins = sample_x[1].shape
        
        print(f"Data: {n_channels} channels, {n_classes} classes")
        print(f"Spectral: {freq_bins}x{time_bins}, Temporal: {T_raw} samples")
        
        # Create model
        model = DualStreamEEGModel(
            freq_bins=freq_bins,
            time_bins=time_bins,
            n_channels=n_channels,
            n_classes=n_classes,
            T_raw=T_raw,
            fusion_mode=strategy,
            cnn_filters=config['cnn_filters'],
            lstm_hidden=config['lstm_hidden'],
            pos_dim=config['pos_dim'],
            dropout=config['dropout'],
            cnn_dropout=config['cnn_dropout'],
            use_hidden_layer=config['use_hidden_layer'],
            hidden_dim=config['hidden_dim'],
            fusion_temperature=2.0,
            fusion_rank=fusion_rank
        ).to(device)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0
        epochs_no_improve = 0
        patience = config.get('patience', 5)
        
        model_save_path = f'./ablation_{task}/models/best_{task}_{strategy}.pth'
        os.makedirs(f'./ablation_{task}/models', exist_ok=True)
        
        for epoch in range(config['num_epochs']):
            train_loss, train_acc = train_epoch(model, loaders['train'], criterion, optimizer, device, is_binary)
            val_loss, val_acc = evaluate(model, loaders['val'], device, criterion, is_binary)
            
            print(f"Epoch {epoch+1}/{config['num_epochs']} | "
                  f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), model_save_path)
                print(f"  → Best model saved! Val Acc: {val_acc:.2f}%")
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f"  → Early stopping (patience={patience})")
                break
        
        # Load best model
        model.load_state_dict(torch.load(model_save_path))
        
        # Evaluate
        log_entry = {'strategy': strategy, 'val_acc': best_val_acc}
        
        if 'test2' in loaders:
            test2_metrics = evaluate_comprehensive(model, loaders['test2'], device, is_binary)
            log_entry.update({f"test2_{k}": v for k, v in test2_metrics.items()})
            
            # Collect statistics
            stats = collect_feature_statistics(model, loaders['test2'], device, strategy)
            analysis = analyze_fusion_statistics(stats, strategy, task)
            log_entry.update({
                'spectral_dominance': analysis['spectral_dominance'],
                'temporal_dominance': analysis['temporal_dominance']
            })
            
            # Plot spatial attention if applicable
            if strategy == 'spatial_attention':
                plot_spatial_attention_heatmap(task, stats, n_channels)
        
        results_log.append(log_entry)
        print(f"\n{strategy} Results:")
        print(f"  Val Acc: {log_entry['val_acc']:.2f}%")
        if 'test2_acc' in log_entry:
            print(f"  Test2 Acc: {log_entry['test2_acc']:.2f}%")
            print(f"  Spectral Dom: {log_entry.get('spectral_dominance', 0):.3f}")
            print(f"  Temporal Dom: {log_entry.get('temporal_dominance', 0):.3f}")

    # Save results
    df = pd.DataFrame(results_log)
    os.makedirs(f'./ablation_{task}/results', exist_ok=True)
    csv_path = f"./ablation_{task}/results/ablation_results_{task}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*70}")
    print(f"✓ Results saved to: {csv_path}")
    print(f"{'='*70}\n")
    
    # Print summary
    print("ABLATION SUMMARY:")
    print(df.to_string(index=False))
    
    # Find best strategy
    best_idx = df['test2_acc'].idxmax()
    best_strategy = df.loc[best_idx, 'strategy']
    best_acc = df.loc[best_idx, 'test2_acc']
    print(f"\n🏆 BEST STRATEGY: {best_strategy} ({best_acc:.2f}%)")
    
    # Plot modality contributions
    plot_modality_contributions(task, df)
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run fusion ablation study')
    parser.add_argument('--task', type=str, required=True,
                        choices=['SSVEP', 'P300', 'MI', 'Imagined_speech',
                                'Lee2019_MI', 'Lee2019_SSVEP', 'BNCI2014_P300', 'BI2014b_P300'])
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--patience', type=int, default=5)
    
    # STFT configuration (optional - uses task defaults if not specified)
    parser.add_argument('--stft_fs', type=int, default=None)
    parser.add_argument('--stft_nperseg', type=int, default=None)
    parser.add_argument('--stft_noverlap', type=int, default=None)
    parser.add_argument('--stft_nfft', type=int, default=None)
    
    args = parser.parse_args()
    
    config = {
        # Training
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'seed': args.seed,
        'patience': args.patience,
        
        # Model architecture
        'cnn_filters': 16,
        'lstm_hidden': 128,
        'pos_dim': 16,
        'dropout': 0.3,
        'cnn_dropout': 0.2,
        'use_hidden_layer': True,
        'hidden_dim': 64,
    }
    
    # Add STFT config if provided
    if args.stft_fs is not None:
        config['stft_fs'] = args.stft_fs
    if args.stft_nperseg is not None:
        config['stft_nperseg'] = args.stft_nperseg
    if args.stft_noverlap is not None:
        config['stft_noverlap'] = args.stft_noverlap
    if args.stft_nfft is not None:
        config['stft_nfft'] = args.stft_nfft
    
    results = run_ablation_study(args.task, config)