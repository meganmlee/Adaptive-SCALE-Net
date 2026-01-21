"""
AdaptiveSCALENet - Adaptive SCALE-Net Model for Multi-Task EEG Classification
Supports: SSVEP, P300, MI (Motor Imagery), Imagined Speech

Dual-stream architecture with adaptive fusion strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from typing import Optional, Dict, Tuple
from seed_utils import seed_everything
from sklearn.metrics import f1_score, recall_score, roc_auc_score

# Import from dataset.py
from dataset import (
    load_dataset, 
    TASK_CONFIGS, 
    EEGDataset, 
    create_dataloaders,
)

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
    def __init__(self, mode, dim, n_channels=None, temperature=2.0):
        super().__init__()
        self.mode = mode
        self.dim = dim
        self.temperature = temperature
        
        if mode == 'global_attention' or mode == 'spatial_attention':
            self.attn = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Linear(dim, 2))
        elif mode == 'glu':
            self.gate = nn.Linear(dim, dim)
        elif mode == 'multiplicative':
            self.proj_s = nn.Linear(dim, dim)
            self.proj_t = nn.Linear(dim, dim)
        elif mode == 'bilinear':
            self.bilinear = nn.Bilinear(dim, dim, dim)

    def forward(self, x_s, x_t):
        if self.mode == 'static':
            return 0.5 * x_s + 0.5 * x_t, None
            
        elif self.mode == 'global_attention':
            ctx = torch.cat([x_s.mean(1), x_t.mean(1)], dim=-1)
            attn_logits = self.attn(ctx) / self.temperature
            w = torch.softmax(attn_logits, dim=-1)
            return w[:, 0].view(-1, 1, 1) * x_s + w[:, 1].view(-1, 1, 1) * x_t, w
            
        elif self.mode == 'spatial_attention':
            ctx = torch.cat([x_s, x_t], dim=-1)
            attn_logits = self.attn(ctx) / self.temperature
            w = torch.softmax(attn_logits, dim=-1)
            return w[:, :, 0].unsqueeze(-1) * x_s + w[:, :, 1].unsqueeze(-1) * x_t, w
            
        elif self.mode == 'glu':
            combined = x_s + x_t
            gate_val = torch.sigmoid(self.gate(combined))
            return combined * gate_val, gate_val

        elif self.mode == 'multiplicative':
            return self.proj_s(x_s) * self.proj_t(x_t), None
            
        elif self.mode == 'bilinear':
            B, C, D = x_s.shape
            res = self.bilinear(x_s.view(-1, D), x_t.view(-1, D))
            return res.view(B, C, D), None


class AttentionFusion(nn.Module):
    """
    Legacy attention fusion (equivalent to 'global_attention' with projection).
    Kept for backward compatibility.
    """
    def __init__(self, spec_dim, time_dim, out_dim):
        super().__init__()
        self.spec_proj = nn.Linear(spec_dim, out_dim)
        self.time_proj = nn.Linear(time_dim, out_dim)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, 2), # Outputs weights for [Spec, Time]
            nn.Softmax(dim=1)
        )
        
    def forward(self, x_spec, x_time):
        # Project both to same dimension
        # x_spec: (B, C, spec_dim)
        # x_time: (B, C, time_dim) (We expanded time feats to C channels)
        
        feat_spec = self.spec_proj(x_spec)
        feat_time = self.time_proj(x_time)
        
        # Calculate attention weights per trial (Global Average Pooling context)
        # We average over Channels to get a general "trial state"
        ctx_spec = feat_spec.mean(dim=1) # (B, out_dim)
        ctx_time = feat_time.mean(dim=1) # (B, out_dim)
        ctx = torch.cat([ctx_spec, ctx_time], dim=1) # (B, out_dim*2)
        
        weights = self.attention(ctx) # (B, 2)
        w_spec = weights[:, 0].view(-1, 1, 1) # (B, 1, 1)
        w_time = weights[:, 1].view(-1, 1, 1) # (B, 1, 1)
        
        # Weighted sum (Fusion)
        fused = (w_spec * feat_spec) + (w_time * feat_time)
        return fused, weights

class AdaptiveSCALENet(nn.Module):
    """
    Adaptive SCALE-Net: Dual-stream architecture with adaptive fusion strategies
    
    Combines spectral (STFT) and temporal (raw) streams with configurable fusion methods.
    
    Inputs:
        x_time: (B, C, T_raw) - Raw temporal EEG data
        x_spec: (B, C, F, T) - STFT features
    """
    
    def __init__(self, freq_bins, time_bins, n_channels, n_classes, T_raw,
                 cnn_filters=16, lstm_hidden=128, pos_dim=16,
                 dropout=0.3, cnn_dropout=0.2, use_hidden_layer=False, hidden_dim=64,
                 fusion_mode='global_attention', fusion_temperature=2.0):
        
        super().__init__()
        self.n_channels = n_channels
        self.T_raw = T_raw
        
        # ====== SPECTRAL STREAM (2D CNN) - Same as original model ======
        self.spec_cnn_filters = cnn_filters * 2
        
        # Stage 1: Conv(1→16) + BN + ReLU + SE + Dropout + Pool
        self.spec_conv1 = nn.Conv2d(1, cnn_filters, kernel_size=7, padding=3, bias=False)
        self.spec_bn1 = nn.BatchNorm2d(cnn_filters)
        self.spec_se1 = SqueezeExcitation(cnn_filters, reduction=4)
        self.spec_dropout_cnn1 = nn.Dropout2d(cnn_dropout)
        self.spec_pool1 = nn.MaxPool2d(2)  # (F, T) → (F/2, T/2)
        
        # Stage 2: Conv(16→32) + BN + ReLU + SE + Dropout + Pool
        self.spec_conv2 = nn.Conv2d(cnn_filters, self.spec_cnn_filters, kernel_size=5, padding=2, bias=False)
        self.spec_bn2 = nn.BatchNorm2d(self.spec_cnn_filters)
        self.spec_se2 = SqueezeExcitation(self.spec_cnn_filters, reduction=4)
        self.spec_dropout_cnn2 = nn.Dropout2d(cnn_dropout)
        self.spec_pool2 = nn.MaxPool2d(2)  # (F/2, T/2) → (F/4, T/4)
        
        # Spectral CNN Output Dimension
        self.spec_out_dim = (freq_bins // 4) * (time_bins // 4) * self.spec_cnn_filters
        
        # ====== TEMPORAL STREAM (EEGNet Inspired) ======
        F1 = 8  # F1 filters per temporal kernel
        D = 2   # Depth multiplier (Spatial filters per temporal filter)
        F2 = F1 * D
        
        # Layer 1: Temporal Conv (Kernels along Time)
        # Input: (B, 1, C, T) -> Output: (B, F1, C, T)
        self.temp_conv = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn_temp = nn.BatchNorm2d(F1)
        
        # Layer 2: Spatial Conv (Kernels along Channels)
        # Input: (B, F1, C, T) -> Output: (B, F2, 1, T)
        # GROUPS=F1 makes this a Depthwise convolution (crucial for EEGNet)
        self.spatial_conv = nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False)
        self.bn_spatial = nn.BatchNorm2d(F2)
        self.pool_spatial = nn.AvgPool2d((1, 4)) # Pool time by 4
        
        # Layer 3: Separable Conv
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False), # Depthwise
            nn.Conv2d(F2, F2, (1, 1), bias=False), # Pointwise
            nn.BatchNorm2d(F2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1, 8)), # Pool time aggressively
            nn.Dropout(cnn_dropout)
        )
        
        # Calculate dimension after flattening the temporal stream
        # This vector represents the "Global Context" of the trial
        final_time_dim = (T_raw // 4) // 8 
        self.time_out_dim = F2 * final_time_dim

        # ====== FEATURE PROJECTION (for fusion) ======
        # Project both streams to same dimension before fusion
        self.proj_spec = nn.Linear(self.spec_out_dim, lstm_hidden)
        self.proj_time = nn.Linear(self.time_out_dim, lstm_hidden)

        # ====== FUSION LAYER ======
        # Support multiple fusion strategies: 'static', 'global_attention', 'spatial_attention', 
        # 'glu', 'multiplicative', 'bilinear'
        # Default: 'global_attention' (equivalent to original AttentionFusion)
        self.fusion_mode = fusion_mode
        if fusion_mode == 'legacy' or fusion_mode == 'original':
            # Use original AttentionFusion for backward compatibility
            self.fusion_layer = AttentionFusion(self.spec_out_dim, self.time_out_dim, lstm_hidden)
        else:
            # Use FusionAblations with projection applied before fusion
            self.fusion_layer = FusionAblations(
                mode=fusion_mode, 
                dim=lstm_hidden, 
                n_channels=n_channels,
                temperature=fusion_temperature
            )

        # Channel Position Embedding
        self.chan_emb = nn.Embedding(n_channels, lstm_hidden)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_hidden, # Projected dimension
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

    def forward(self, x_time, x_spec, chan_ids: Optional[torch.Tensor] = None):
        B, C, _, _ = x_spec.shape
        _, _, T_raw = x_time.shape

        # ====== 1. SPECTRAL STREAM (2D CNN) ======
        # (B, C, F, T) → (B*C, 1, F, T)
        x_spec = x_spec.view(B * C, 1, x_spec.size(2), x_spec.size(3))
        
        # Stage 1
        x_spec = self.spec_conv1(x_spec); x_spec = self.spec_bn1(x_spec); x_spec = F.relu(x_spec, inplace=True)
        x_spec = self.spec_se1(x_spec); x_spec = self.spec_dropout_cnn1(x_spec); x_spec = self.spec_pool1(x_spec)
        
        # Stage 2
        x_spec = self.spec_conv2(x_spec); x_spec = self.spec_bn2(x_spec); x_spec = F.relu(x_spec, inplace=True)
        x_spec = self.spec_se2(x_spec); x_spec = self.spec_dropout_cnn2(x_spec); x_spec = self.spec_pool2(x_spec)
        
        x_spec = x_spec.view(B, C, -1)  # (B, C, spec_out_dim)

        # ====== 2. TEMPORAL STREAM (1D CNN) ======
        # Input: (B, C, T) -> (B, 1, C, T) to use 2D Spatial Conv
        x_global = x_time.unsqueeze(1) 
        
        x_global = self.temp_conv(x_global)
        x_global = self.bn_temp(x_global)
        
        # Crucial Step: Spatial Conv reduces C dimension to 1
        x_global = self.spatial_conv(x_global) 
        x_global = F.relu(self.bn_spatial(x_global))
        x_global = self.pool_spatial(x_global)
        
        x_global = self.separable_conv(x_global)
        
        # Flatten: (B, F2, 1, T_small) -> (B, F2*T_small)
        x_global = x_global.view(B, -1)
        
        # ====== 3. FEATURE PROJECTION ======
        # Project both streams to same dimension
        x_spec_proj = self.proj_spec(x_spec)  # (B, C, lstm_hidden)
        x_global_expanded = x_global.unsqueeze(1).expand(-1, C, -1)  # (B, C, time_out_dim)
        x_time_proj = self.proj_time(x_global_expanded)  # (B, C, lstm_hidden)
        
        # ====== 4. FEATURE FUSION ======
        # Apply fusion strategy
        if self.fusion_mode == 'legacy' or self.fusion_mode == 'original':
            # Original AttentionFusion (handles projection internally)
            features, weights = self.fusion_layer(x_spec, x_global_expanded)
        else:
            # FusionAblations (projection already applied)
            features, weights = self.fusion_layer(x_spec_proj, x_time_proj)  # (B, C, lstm_hidden)
        
        # Add Position Embeddings (Transformer style addition)
        if chan_ids is None:
            chan_ids = torch.arange(C, device=features.device).unsqueeze(0).expand(B, C)
        pos = self.chan_emb(chan_ids) # (B, C, pos_dim)
        features = features + pos
        
        # LSTM Aggregation
        _, (h, _) = self.lstm(features)
        h = h.squeeze(0)
        h = self.dropout_lstm(h)

        if self.use_hidden_layer:
            h = self.hidden_layer(h)
        
        return self.classifier(h), weights

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
        x_time, x_spec = inputs
        x_time, x_spec, labels = x_time.to(device), x_spec.to(device), labels.to(device)
        
        # Convert labels for binary classification
        if is_binary:
            labels_float = labels.float().unsqueeze(1)  # (B,) → (B, 1) for BCEWithLogitsLoss
        else:
            labels_float = labels
        
        optimizer.zero_grad()
        outputs, _ = model(x_time, x_spec)
        loss = criterion(outputs, labels_float)
        
        loss.backward()
        # Handle DataParallel case for gradient clipping
        actual_model = unwrap_model(model)
        torch.nn.utils.clip_grad_norm_(actual_model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Prediction: binary uses sigmoid threshold, multi-class uses argmax
        if is_binary:
            pred = (torch.sigmoid(outputs) > 0.5).squeeze(1).long()  # (B, 1) → (B,)
        else:
            _, pred = outputs.max(1)
        
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{total_loss/(pbar.n+1):.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, device, criterion=None, is_binary=False, return_metrics=False):
    """
    Evaluate model on a dataset
    
    Args:
        model: Model to evaluate
        loader: DataLoader
        device: Device to run on
        criterion: Loss function (optional)
        is_binary: Whether this is binary classification
        return_metrics: If True, return additional metrics (f1, recall, auc)
    
    Returns:
        If return_metrics=False: (avg_loss, acc)
        If return_metrics=True: (avg_loss, acc, metrics_dict)
            where metrics_dict contains 'f1', 'recall', 'auc' (if applicable)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds = []
    all_labels = []
    all_probs = []  # For AUC calculation
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Eval', ncols=100):
            x_time, x_spec = inputs
            x_time, x_spec, labels = x_time.to(device), x_spec.to(device), labels.to(device)
            
            # Convert labels for binary classification
            if is_binary:
                labels_float = labels.float().unsqueeze(1)  # (B,) → (B, 1) for BCEWithLogitsLoss
            else:
                labels_float = labels
            
            outputs, _ = model(x_time, x_spec)
            if criterion is not None:
                loss = criterion(outputs, labels_float)
                total_loss += loss.item()
            
            # Prediction: binary uses sigmoid threshold, multi-class uses argmax
            if is_binary:
                probs = torch.sigmoid(outputs).squeeze(1)  # (B,)
                pred = (probs > 0.5).long()  # (B,)
                if return_metrics:
                    all_probs.append(probs.cpu().numpy())
            else:
                probs = F.softmax(outputs, dim=1)  # (B, n_classes)
                _, pred = outputs.max(1)
                if return_metrics:
                    all_probs.append(probs.cpu().numpy())
            
            if return_metrics:
                all_preds.append(pred.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    
    avg_loss = total_loss / len(loader) if criterion is not None else None
    acc = 100. * correct / total
    
    if not return_metrics:
        return avg_loss, acc
    
    # Calculate additional metrics
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    metrics = {}
    
    # F1 score (macro average for multi-class, binary for binary)
    if is_binary:
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    else:
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics['f1'] = f1 * 100  # Convert to percentage
    
    # Recall (macro average for multi-class, binary for binary)
    if is_binary:
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    else:
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics['recall'] = recall * 100  # Convert to percentage
    
    # AUC
    try:
        if is_binary:
            # Binary classification: use probabilities directly
            auc = roc_auc_score(all_labels, all_probs)
            metrics['auc'] = auc * 100  # Convert to percentage
        else:
            # Multi-class: use sklearn's built-in multi-class AUC
            n_classes = all_probs.shape[1]
            if n_classes == 2:
                # Binary case with 2 classes (use positive class probability)
                auc = roc_auc_score(all_labels, all_probs[:, 1])
                metrics['auc'] = auc * 100
            else:
                # Multi-class: use one-vs-rest (ovr) or one-vs-one (ovo) approach
                # 'ovr' computes AUC for each class against the rest, then averages
                # 'macro' averages the AUCs of each class
                try:
                    # Try one-vs-rest macro average
                    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
                    metrics['auc'] = auc * 100
                except ValueError:
                    # Fallback: if some classes are missing, try weighted average
                    try:
                        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
                        metrics['auc'] = auc * 100
                    except ValueError:
                        # If still fails, set to None
                        metrics['auc'] = None
    except (ValueError, IndexError) as e:
        # If AUC calculation fails (e.g., only one class present), set to None
        metrics['auc'] = None
    
    return avg_loss, acc, metrics


# ==================== Main Training ====================
def collect_weights(model, loader, device):
    """
    Runs an evaluation pass and collects the attention weights for every trial.
    
    Returns:
        weights_array: Shape (N, 2) where N is number of trials
            - For global_attention: (B, 2) -> (N, 2) after concatenation
            - For spatial_attention: (B, C, 2) -> average over channels -> (B, 2) -> (N, 2)
            - For other modes: may return None or different shapes
    """
    model.eval()
    all_weights = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc='Collecting Weights', ncols=100):
            x_time, x_spec = inputs
            x_time, x_spec = x_time.to(device), x_spec.to(device)
            
            # Outputs is a 2-tuple, we only care about the second element (weights)
            _, weights = model(x_time, x_spec)
            
            if weights is not None:
                weights_np = weights.cpu().numpy()
                
                # Handle different weight shapes based on fusion mode
                if weights_np.ndim == 2 and weights_np.shape[1] == 2:
                    # Global attention: (B, 2) - already in correct format
                    all_weights.append(weights_np)
                elif weights_np.ndim == 3 and weights_np.shape[2] == 2:
                    # Spatial attention: (B, C, 2) - average over channels to get (B, 2)
                    weights_avg = weights_np.mean(axis=1)  # (B, C, 2) -> (B, 2)
                    all_weights.append(weights_avg)
                else:
                    # Other modes or unexpected shapes - skip or handle appropriately
                    # For static, glu, multiplicative, bilinear, weights may be None
                    # Create dummy weights if needed
                    B = weights_np.shape[0] if weights_np.ndim > 0 else 1
                    dummy_weights = np.array([[0.5, 0.5]] * B)
                    all_weights.append(dummy_weights)
    
    if len(all_weights) == 0:
        return None
    
    return np.concatenate(all_weights, axis=0)

def train_task(task: str, config: Optional[Dict] = None, model_path: Optional[str] = None) -> Tuple:
    """
    Train model for a specific EEG task
    
    Args:
        task: One of 'SSVEP', 'P300', 'MI', 'Imagined_speech'
        config: Training configuration (uses defaults if None)
        model_path: Path to save best model
        
    Returns:
        (model, results_dict)
    """
    # Get task-specific defaults
    task_config = TASK_CONFIGS.get(task, {})
    
    if config is None:
        config = {
            'data_dir': task_config.get('data_dir', '/ocean/projects/cis250213p/shared/ssvep'),
            'num_seen': task_config.get('num_seen', 33),
            'seed': 44,
            'n_classes': task_config.get('num_classes', 26),
            
            # Model
            'cnn_filters': 16,
            'lstm_hidden': 128,
            'pos_dim': 16,
            'dropout': 0.3,  # Dropout for LSTM output and classifier
            'cnn_dropout': 0.2,  # Dropout for CNN layers (spatial dropout)
            'use_hidden_layer': False,  # Whether to add hidden dense layer after LSTM
            'hidden_dim': 64,  # Hidden layer dimension if use_hidden_layer=True
            'fusion_mode': 'global_attention',  # Fusion strategy: 'static', 'global_attention', 'spatial_attention', 'glu', 'multiplicative', 'bilinear', 'legacy'
            'fusion_temperature': 2.0,  # Temperature for attention-based fusion strategies
            
            # STFT - Use task-specific parameters from TASK_CONFIGS
            'stft_fs': task_config.get('sampling_rate', 250),
            'stft_nperseg': task_config.get('stft_nperseg', 128),
            'stft_noverlap': task_config.get('stft_noverlap', 112),
            'stft_nfft': task_config.get('stft_nfft', 512),
            
            # Training
            'batch_size': 64,
            'num_epochs': 100,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'patience': 20,
            'scheduler': 'ReduceLROnPlateau',
        }
    else:
        # Fill in missing keys with task-specific defaults
        config.setdefault('n_classes', task_config.get('num_classes', 26))
        config.setdefault('stft_fs', task_config.get('sampling_rate', 250))
        config.setdefault('stft_nperseg', task_config.get('stft_nperseg', 128))
        config.setdefault('stft_noverlap', task_config.get('stft_noverlap', 112))
        config.setdefault('stft_nfft', task_config.get('stft_nfft', 512))
        config.setdefault('dropout', 0.3)
        config.setdefault('cnn_dropout', 0.2)
        config.setdefault('use_hidden_layer', False)
        config.setdefault('hidden_dim', 64)
        config.setdefault('scheduler', 'ReduceLROnPlateau')  # Default scheduler
        config.setdefault('fusion_mode', 'global_attention')  # Default fusion strategy
        config.setdefault('fusion_temperature', 2.0)  # Default temperature for attention

    seed = config.get('seed', 44)
    seed_everything(seed, deterministic=True)
    
    # Setup device and multi-GPU
    device, n_gpus = setup_device()
    print(f"\n{'='*70}")
    print(f"{task} Classification")
    print(f"{'='*70}")
    print(f"Device: {device}, GPUs: {n_gpus}")
    print(f"Fusion Mode: {config.get('fusion_mode', 'global_attention')}")
    
    # ====== Load Data ======
    datasets = load_dataset(
        task=task,
        data_dir=config.get('data_dir'),
        num_seen=config.get('num_seen'),
        seed=config.get('seed', 44)
    )
    
    if not datasets:
        raise ValueError(f"Failed to load data for task: {task}")
    
    # STFT config
    stft_config = {
        'fs': config['stft_fs'],
        'nperseg': config['stft_nperseg'],
        'noverlap': config['stft_noverlap'],
        'nfft': config['stft_nfft']
    }
    
    # Print STFT parameters
    print(f"\nSTFT Parameters (task-specific):")
    print(f"  Sampling Rate: {stft_config['fs']} Hz")
    print(f"  nperseg: {stft_config['nperseg']} samples ({stft_config['nperseg']/stft_config['fs']:.3f} sec)")
    print(f"  noverlap: {stft_config['noverlap']} samples ({100*stft_config['noverlap']/stft_config['nperseg']:.1f}% overlap)")
    print(f"  nfft: {stft_config['nfft']}")
    print(f"  Frequency resolution: {stft_config['fs']/stft_config['nfft']:.2f} Hz/bin")
    
    # ====== Create Data Loaders ======
    loaders = create_dataloaders(
        datasets, 
        stft_config, 
        batch_size=config['batch_size'],
        num_workers=4,
        augment_train=config.get('augment_train', True),
        seed=seed,
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    test1_loader = loaders.get('test1')
    test2_loader = loaders.get('test2')
    
    # Get dimensions from a sample
    sample_x, _ = next(iter(train_loader)) # Now returns (x_time, x_spec), labels
    sample_x_time, sample_x_spec = sample_x
    _, n_channels, T_raw = sample_x_time.shape # Get T_raw from the first element
    _, _, freq_bins, time_bins = sample_x_spec.shape
    print(f"STFT shape: ({n_channels}, {freq_bins}, {time_bins})")
    
    # ====== Create Model ======
    n_classes = config['n_classes']
    model = AdaptiveSCALENet(
        freq_bins=freq_bins,
        time_bins=time_bins,
        n_channels=n_channels,
        n_classes=n_classes,
        T_raw=T_raw,
        cnn_filters=config['cnn_filters'],
        lstm_hidden=config['lstm_hidden'],
        pos_dim=config['pos_dim'],
        dropout=config.get('dropout', 0.3),
        cnn_dropout=config.get('cnn_dropout', 0.2),
        use_hidden_layer=config.get('use_hidden_layer', False),
        hidden_dim=config.get('hidden_dim', 64),
        fusion_mode=config.get('fusion_mode', 'global_attention'),
        fusion_temperature=config.get('fusion_temperature', 2.0)
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Classes: {n_classes}")
    
    # Wrap model for multi-GPU training
    model = wrap_model_multi_gpu(model, n_gpus)
    
    # ====== Loss & Optimizer ======
    # Use Binary Cross Entropy for binary classification (n_classes=2)
    # Use Cross Entropy for multi-class classification (n_classes > 2)
    train_labels = datasets['train'][1]
    if n_classes == 2:
        # Calculate class imbalance
        class_counts = np.bincount(train_labels)
        class_ratio = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0
        
        print(f"  Imbalance Ratio: {class_ratio:.2f}:1")
        
        # Only use pos_weight if imbalance ratio > 1.5
        if class_ratio > 1.5 or class_ratio < 0.67:
            pos_weight = torch.tensor([class_ratio], device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"Using BCEWithLogitsLoss with pos_weight={class_ratio:.2f} (imbalanced)")
        else:
            criterion = nn.BCEWithLogitsLoss()
            print(f"Using BCEWithLogitsLoss without pos_weight (balanced dataset)")
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
        seed = config.get('seed', 44)
        model_path = f'best_{task.lower()}_model_{seed}.pth'
    
    # Check if binary classification
    is_binary = (n_classes == 2)
    
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
        
        # Save best model (save unwrapped model state for portability)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrap_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'task': task,
                'config': config,
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
    # Handle DataParallel wrapper when loading
    unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate validation set with metrics
    val_loss, val_acc, val_metrics = evaluate(model, val_loader, device, criterion, is_binary=is_binary, return_metrics=True)
    results = {
        'val': best_val_acc,
        'val_f1': val_metrics.get('f1'),
        'val_recall': val_metrics.get('recall'),
        'val_auc': val_metrics.get('auc'),
    }
    
    if test1_loader:
        test1_loss, test1_acc, test1_metrics = evaluate(model, test1_loader, device, criterion, is_binary=is_binary, return_metrics=True)
        results['test1'] = test1_acc
        results['test1_loss'] = test1_loss
        results['test1_f1'] = test1_metrics.get('f1')
        results['test1_recall'] = test1_metrics.get('recall')
        results['test1_auc'] = test1_metrics.get('auc')
    
    if test2_loader:
        test2_loss, test2_acc, test2_metrics = evaluate(model, test2_loader, device, criterion, is_binary=is_binary, return_metrics=True)
        results['test2'] = test2_acc
        results['test2_loss'] = test2_loss
        results['test2_f1'] = test2_metrics.get('f1')
        results['test2_recall'] = test2_metrics.get('recall')
        results['test2_auc'] = test2_metrics.get('auc')
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS - {task}")
    print(f"{'='*70}")
    print(f"Best Val Acc:    {best_val_acc:.2f}%")
    if results.get('val_f1') is not None:
        print(f"Val F1:          {results['val_f1']:.2f}%")
        print(f"Val Recall:      {results['val_recall']:.2f}%")
        if results.get('val_auc') is not None:
            print(f"Val AUC:         {results['val_auc']:.2f}%")
    if 'test1' in results:
        print(f"Test1 (Seen):    {results['test1']:.2f}% (loss {results['test1_loss']:.4f})")
        if results.get('test1_f1') is not None:
            print(f"  Test1 F1:      {results['test1_f1']:.2f}%")
            print(f"  Test1 Recall:  {results['test1_recall']:.2f}%")
            if results.get('test1_auc') is not None:
                print(f"  Test1 AUC:     {results['test1_auc']:.2f}%")
    if 'test2' in results:
        print(f"Test2 (Unseen):  {results['test2']:.2f}% (loss {results['test2_loss']:.4f})")
        if results.get('test2_f1') is not None:
            print(f"  Test2 F1:      {results['test2_f1']:.2f}%")
            print(f"  Test2 Recall:  {results['test2_recall']:.2f}%")
            if results.get('test2_auc') is not None:
                print(f"  Test2 AUC:     {results['test2_auc']:.2f}%")
    print(f"{'='*70}")

    # ====== COLLECT AND SAVE ATTENTION WEIGHTS ======
    print(f"\n{'='*70}")
    print(f"Collecting attention weights for {task}...")
    
    # Use the Test 2 (Unseen) loader if available, otherwise use Val loader
    weights_loader = test2_loader if test2_loader else val_loader
    weights_type = 'unseen' if test2_loader else 'val'
    
    if weights_loader:
        # Note: You must ensure the model is on device and unwrapped for this call,
        # or handle DataParallel inside the collect_weights if necessary.
        # Since 'model' is already loaded and on the device, we proceed:
        
        all_weights_array = collect_weights(model, weights_loader, device)
        
        if all_weights_array is not None and len(all_weights_array) > 0:
            # Ensure 2D shape (N, 2)
            if all_weights_array.ndim == 3:
                # If somehow still 3D, average over middle dimension
                all_weights_array = all_weights_array.mean(axis=1)
            
            # Convert to DataFrame and save
            weights_df = pd.DataFrame(
                all_weights_array, 
                columns=['Spectral_Weight', 'Temporal_Weight']
            )
            
            # Generate filename based on model_path if available, otherwise use default
            if model_path:
                # Extract config name from model path if it follows the pattern
                # e.g., "ssvep_stft_nperseg128_overlap50pct_nfft512_model.pth" -> "nperseg128_overlap50pct_nfft512"
                import os
                base_name = os.path.basename(model_path)
                if '_stft_' in base_name and '_model.pth' in base_name:
                    config_part = base_name.split('_stft_')[1].replace('_model.pth', '')
                    csv_filename = f'{task.lower()}_attention_weights_{config_part}_{weights_type}.csv'
                else:
                    csv_filename = f'{task.lower()}_attention_weights_{weights_type}.csv'
            else:
                csv_filename = f'{task.lower()}_attention_weights_{weights_type}.csv'
            
            weights_df.to_csv(csv_filename, index=False)
            
            # Calculate and print mean for sanity check
            mean_spec = weights_df['Spectral_Weight'].mean()
            mean_time = weights_df['Temporal_Weight'].mean()
            
            print(f"✓ Attention weights (N={len(weights_df)}) saved to: {csv_filename}")
            print(f"  Mean Spectral Weight: {mean_spec:.4f}")
            print(f"  Mean Temporal Weight: {mean_time:.4f}")
        else:
            print(f"⚠ No attention weights collected (fusion mode may not support weights)")

    print(f"{'='*70}")
    
    return model, results


def train_all_tasks(tasks: Optional[list] = None, save_dir: str = './checkpoints'):
    """
    Train models for all specified tasks
    
    Args:
        tasks: List of task names (default: all tasks)
        save_dir: Directory to save model checkpoints
        
    Returns:
        Dictionary of results for each task
    """
    if tasks is None:
        tasks = ['SSVEP', 'P300', 'MI', 'Imagined_speech', 'Lee2019_MI', 'Lee2019_SSVEP', 'BNCI2014_P300', 'BI2014b_P300']
    
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = {}
    
    print("=" * 80)
    print("MULTI-TASK EEG CLASSIFICATION")
    print("=" * 80)
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"TASK: {task}")
        print(f"{'='*60}")
        
        try:
            # Use default seed 44 for train_all_tasks
            model_path = os.path.join(save_dir, f'best_{task.lower()}_model_44.pth')
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
    print("SUMMARY RESULTS")
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
    print("MULTI-TASK TRAINING COMPLETED!")
    print(f"{'='*80}")
    
    return all_results


# Legacy function for backward compatibility
def train_model(config=None, model_path=None):
    """Legacy training function for SSVEP (backward compatibility)"""
    return train_task('SSVEP', config, model_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train AdaptiveSCALENet on EEG tasks')
    parser.add_argument('--task', type=str, default='SSVEP',
                        choices=['SSVEP', 'P300', 'MI', 'Imagined_speech', 'Lee2019_MI', 'Lee2019_SSVEP', 'BNCI2014_P300', 'BI2014b_P300', 'all'],
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
                        help='Random seed for reproducibility')
    parser.add_argument('--fusion_mode', type=str, default='global_attention',
                        choices=['static', 'global_attention', 'spatial_attention', 'glu', 'multiplicative', 'bilinear', 'legacy', 'original'],
                        help='Fusion strategy: static, global_attention, spatial_attention, glu, multiplicative, bilinear, legacy (default: global_attention)')
    parser.add_argument('--fusion_temperature', type=float, default=2.0,
                        help='Temperature for attention-based fusion strategies (default: 2.0)')
    # STFT parameters (optional - uses task defaults if not specified)
    parser.add_argument('--stft_fs', type=int, default=None,
                        help='STFT sampling frequency (Hz). If not specified, uses task default.')
    parser.add_argument('--stft_nperseg', type=int, default=None,
                        help='STFT window length (nperseg). If not specified, uses task default.')
    parser.add_argument('--stft_noverlap', type=int, default=None,
                        help='STFT overlap length (noverlap). If not specified, uses task default.')
    parser.add_argument('--stft_nfft', type=int, default=None,
                        help='STFT FFT length (nfft). If not specified, uses task default.')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable train-time augmentation (default: augmentation ON).')
    
    args = parser.parse_args()
    
    # Config with task-specific STFT defaults (will be filled in train_task)
    config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        # STFT params will use task-specific defaults from TASK_CONFIGS if not provided
        'cnn_filters': 16,
        'lstm_hidden': 128,
        'pos_dim': 16,
        'dropout': 0.3,
        'cnn_dropout': 0.2,
        'use_hidden_layer': True,
        'hidden_dim': 64,
        'weight_decay': 1e-4,
        'patience': 15,
        'scheduler': 'ReduceLROnPlateau',
        'seed': args.seed,
        'fusion_mode': args.fusion_mode,
        'fusion_temperature': args.fusion_temperature,
    }

    # Train-time augmentation toggle
    config['augment_train'] = (not args.no_augment)
    
    # Add STFT config if provided (None values will use task defaults)
    if args.stft_fs is not None:
        config['stft_fs'] = args.stft_fs
    if args.stft_nperseg is not None:
        config['stft_nperseg'] = args.stft_nperseg
    if args.stft_noverlap is not None:
        config['stft_noverlap'] = args.stft_noverlap
    if args.stft_nfft is not None:
        config['stft_nfft'] = args.stft_nfft
    
    if args.task == 'all':
        results = train_all_tasks(save_dir=args.save_dir)
    else:
        seed = config.get('seed', 44)
        model_path = os.path.join(args.save_dir, f'best_{args.task.lower()}_model_{seed}.pth')
        os.makedirs(args.save_dir, exist_ok=True)
        model, results = train_task(args.task, config=config, model_path=model_path)