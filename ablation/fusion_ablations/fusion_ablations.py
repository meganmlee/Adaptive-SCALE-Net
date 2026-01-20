import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import pandas as pd
from tqdm import tqdm
from typing import Dict
from sklearn.metrics import f1_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import mne

# ==================== Path Setup ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
# Moves up from ablation/fusion_ablations/ to ablation/, then up to root, then into scale_net
scale_net_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'scale-net'))

if scale_net_path not in sys.path:
    sys.path.insert(0, scale_net_path)

# Import local modules from the scale_net folder
from seed_utils import seed_everything
from scale_net_adaptive import (
    AdaptiveSCALENet as BaseDualModel,
    train_epoch, 
    evaluate
)
from dataset import load_dataset, TASK_CONFIGS, create_dataloaders

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

# ==================== Ablation Model Wrapper ====================

class AblationModel(BaseDualModel):
    """
    Subclasses the base model to swap the fusion layer and ensures
    projections are applied before fusion to avoid dimension mismatches.
    """
    def __init__(self, *args, fusion_mode='global_attention', **kwargs):
        super().__init__(*args, **kwargs)
        dim = kwargs.get('lstm_hidden', 128)
        
        # Explicitly define projection layers missing in simple subclassing
        self.proj_spec = nn.Linear(self.spec_out_dim, dim)
        self.proj_time = nn.Linear(self.time_out_dim, dim)
        
        self.fusion_layer = FusionAblations(
            mode=fusion_mode, 
            dim=dim, 
            n_channels=self.n_channels,
            temperature=2.0
        )

    def forward(self, x_time, x_spec, chan_ids=None):
        B, C, _, _ = x_spec.shape

        # 1. Spectral Stream (2D CNN)
        x_spec_feat = x_spec.view(B * C, 1, x_spec.size(2), x_spec.size(3))
        x_spec_feat = self.spec_conv1(x_spec_feat); x_spec_feat = self.spec_bn1(x_spec_feat)
        x_spec_feat = F.relu(x_spec_feat, inplace=True); x_spec_feat = self.spec_se1(x_spec_feat)
        x_spec_feat = self.spec_dropout_cnn1(x_spec_feat); x_spec_feat = self.spec_pool1(x_spec_feat)
        x_spec_feat = self.spec_conv2(x_spec_feat); x_spec_feat = self.spec_bn2(x_spec_feat)
        x_spec_feat = F.relu(x_spec_feat, inplace=True); x_spec_feat = self.spec_se2(x_spec_feat)
        x_spec_feat = self.spec_dropout_cnn2(x_spec_feat); x_spec_feat = self.spec_pool2(x_spec_feat)
        x_spec_feat = x_spec_feat.view(B, C, -1) 

        # 2. Temporal Stream (EEGNet-style)
        x_global = x_time.unsqueeze(1)
        x_global = self.temp_conv(x_global); x_global = self.bn_temp(x_global)
        x_global = self.spatial_conv(x_global); x_global = F.relu(self.bn_spatial(x_global))
        x_global = self.pool_spatial(x_global); x_global = self.separable_conv(x_global)
        x_global = x_global.view(B, -1)
        
        # 3. Apply Projections (Aligns dimensions)
        x_spec_proj = self.proj_spec(x_spec_feat) 
        x_global_proj = self.proj_time(x_global.unsqueeze(1).expand(-1, C, -1))
        
        # 4. Fusion
        features, weights = self.fusion_layer(x_spec_proj, x_global_proj)
        
        # 5. LSTM and Classification
        if chan_ids is None:
            chan_ids = torch.arange(C, device=features.device).unsqueeze(0).expand(B, C)
        features = features + self.chan_emb(chan_ids)
        
        _, (h, _) = self.lstm(features)
        h = h.squeeze(0)
        h = self.dropout_lstm(h)
        if self.use_hidden_layer:
            h = self.hidden_layer(h)
        
        return self.classifier(h), weights

# ==================== Evaluation & Statistics ====================

def evaluate_comprehensive(model, loader, device, is_binary=False):
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
        'f1': f1_score(all_labels, all_preds, average=avg_type),
        'recall': recall_score(all_labels, all_preds, average=avg_type),
    }
    try:
        metrics['auc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr' if not is_binary else 'raise')
    except:
        metrics['auc'] = 0.5
    return metrics

def collect_feature_statistics(model, loader, device, fusion_mode):
    model.eval()
    stats = {
        'spec_contribution': [], 'time_contribution': [],
        'labels': [], 'predictions': [], 'confidences': []
    }
    if fusion_mode in ['global_attention', 'spatial_attention']: stats['attn_weights'] = []
    if fusion_mode == 'spatial_attention': stats['spatial_weights'] = []
    elif fusion_mode == 'glu': stats['gate_sparsity'] = []; stats['gate_values'] = []
    
    with torch.no_grad():
        for (x_time, x_spec), labels in tqdm(loader, desc='Analyzing Usage', ncols=100, leave=False):
            x_time, x_spec = x_time.to(device), x_spec.to(device)
            B, C = x_spec.shape[0], x_spec.shape[1]
            
            # Use model components to get projected features
            # Re-running logic here to get intermediates
            x_spec_f = x_spec.view(B * C, 1, x_spec.size(2), x_spec.size(3))
            x_spec_f = model.spec_conv1(x_spec_f); x_spec_f = model.spec_bn1(x_spec_f); x_spec_f = F.relu(x_spec_f)
            x_spec_f = model.spec_se1(x_spec_f); x_spec_f = model.spec_pool1(x_spec_f)
            x_spec_f = model.spec_conv2(x_spec_f); x_spec_f = model.spec_bn2(x_spec_f); x_spec_f = F.relu(x_spec_f)
            x_spec_f = model.spec_se2(x_spec_f); x_spec_f = model.spec_pool2(x_spec_f)
            x_spec_proj = model.proj_spec(x_spec_f.view(B, C, -1))
            
            x_g = x_time.unsqueeze(1)
            x_g = model.temp_conv(x_g); x_g = model.bn_temp(x_g); x_g = model.spatial_conv(x_g)
            x_g = F.relu(model.bn_spatial(x_g)); x_g = model.pool_spatial(x_g); x_g = model.separable_conv(x_g)
            x_global_proj = model.proj_time(x_g.view(B, -1).unsqueeze(1).expand(-1, C, -1))
            
            _, weights = model.fusion_layer(x_spec_proj, x_global_proj)

            if fusion_mode == 'spatial_attention' and weights is not None:
                # weights shape is (Batch, Channels, 2)
                # We take weights[:, :, 0] which represents the Spectral stream's importance
                stats['spatial_weights'].append(weights[:, :, 0].cpu().numpy())
            
            spec_mag = torch.norm(x_spec_proj, dim=-1)
            time_mag = torch.norm(x_global_proj, dim=-1)
            total = spec_mag + time_mag + 1e-8
            
            stats['spec_contribution'].append((spec_mag / total).cpu().numpy())
            stats['time_contribution'].append((time_mag / total).cpu().numpy())
            
            outputs, _ = model(x_time, x_spec)
            probs = torch.sigmoid(outputs) if outputs.shape[1] == 1 else torch.softmax(outputs, dim=1)
            stats['confidences'].append(probs.max(dim=1)[0].cpu().numpy())
            stats['labels'].append(labels.numpy())
            
            if weights is not None:
                if 'attn_weights' in stats: stats['attn_weights'].append(weights.cpu().numpy())
                elif 'gate_values' in stats: 
                    stats['gate_values'].append(weights.cpu().numpy())
                    stats['gate_sparsity'].append((weights < 0.1).float().mean().cpu().item())

    return {
        k: np.concatenate(v, axis=0) if (len(v) > 0 and isinstance(v[0], np.ndarray)) 
        else np.array(v) 
        for k, v in stats.items()
    }

def analyze_fusion_statistics(stats, fusion_mode, task):
    save_dir=f'./ablation_{task}/analysis'
    os.makedirs(save_dir, exist_ok=True)
    avg_spec = stats['spec_contribution'].mean()
    avg_time = stats['time_contribution'].mean()
    analysis = {
        'fusion_mode': fusion_mode, 'spectral_dominance': avg_spec, 'temporal_dominance': avg_time,
        'primary_modality': 'Spectral' if avg_spec > avg_time else 'Temporal'
    }
    if 'attn_weights' in stats: analysis['attn_variance'] = stats['attn_weights'].var()
    if 'gate_sparsity' in stats: analysis['noise_suppression'] = stats['gate_sparsity'].mean()
    
    pd.DataFrame([analysis]).to_csv(f'{save_dir}/{task}_{fusion_mode}_usage.csv', index=False)
    return analysis

def plot_modality_contributions(task, results_df):
    save_dir=f'./ablation_{task}/plots'
    os.makedirs(save_dir, exist_ok=True)
    plot_data = []
    for _, row in results_df.iterrows():
        plot_data.append({'Strategy': row['strategy'], 'Contribution': row['spectral_dominance'], 'Modality': 'Spectral'})
        plot_data.append({'Strategy': row['strategy'], 'Contribution': row['temporal_dominance'], 'Modality': 'Temporal'})
    
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    sns.barplot(x='Strategy', y='Contribution', hue='Modality', data=pd.DataFrame(plot_data), palette='viridis')
    plt.title(f'Modality Usage Balance - {task}')
    plt.ylim(0, 1.0)
    
    for i, row in results_df.iterrows():
        plt.text(i, 0.95, f"{row['val_acc']:.1f}%", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{task}_modality_balance.png', dpi=300)
    plt.close()

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

# ==================== Ablation Study ====================

def run_ablation_study(task: str, config: Dict = None):
    """
    Run fusion strategy ablation study for a given task.
    
    Args:
        task: Task name (e.g., 'SSVEP', 'P300', 'MI', etc.)
        config: Configuration dictionary. Can include:
            - Training params: 'num_epochs', 'batch_size', 'lr', 'seed', 'patience'
            - Model params: 'cnn_filters', 'lstm_hidden', 'pos_dim', 'dropout', etc.
            - STFT params (optional): 'stft_fs', 'stft_nperseg', 'stft_noverlap', 'stft_nfft'
                If not provided, uses default values from TASK_CONFIGS for the task.
    """
    strategies = ['spatial_attention', 'static', 'global_attention', 'glu', 'multiplicative', 'bilinear']
    if config is None:
        config = {
            'num_epochs': 30, 'batch_size': 16, 'lr': 1e-3, 'seed': 44,
            'cnn_filters': 16, 'lstm_hidden': 128, 'pos_dim': 16,
            'dropout': 0.3, 'cnn_dropout': 0.2, 'use_hidden_layer': True, 'hidden_dim': 64
        }

    train_keys = {'num_epochs', 'batch_size', 'lr', 'seed', 'weight_decay', 'patience'}
    stft_keys = {'stft_fs', 'stft_nperseg', 'stft_noverlap', 'stft_nfft', 'sampling_rate'}
    model_kwargs = {k: v for k, v in config.items() if k not in train_keys and k not in stft_keys}

    task_config = TASK_CONFIGS.get(task, {})
    patience = config.get('patience', 5)
    n_classes = task_config.get('num_classes', 26)
    is_binary = (n_classes == 2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results_log = []
    for strategy in strategies:
        print(f"\n>>> STRATEGY: {strategy.upper()}")
        seed_everything(config['seed'], deterministic=True)
        
        datasets = load_dataset(task=task, data_dir=task_config.get('data_dir'), seed=config['seed'])
        
        # STFT config: Use custom values from config if provided, otherwise use task_config defaults
        stft_config = {
            'fs': config.get('stft_fs') or config.get('sampling_rate') or task_config.get('sampling_rate', 250),
            'nperseg': config.get('stft_nperseg', task_config.get('stft_nperseg', 128)),
            'noverlap': config.get('stft_noverlap', task_config.get('stft_noverlap', 112)),
            'nfft': config.get('stft_nfft', task_config.get('stft_nfft', 512))
        }
        loaders = create_dataloaders(datasets, stft_config, batch_size=config['batch_size'])
        
        sample_x, _ = next(iter(loaders['train']))
        _, n_channels, T_raw = sample_x[0].shape
        _, _, freq_bins, time_bins = sample_x[1].shape
        
        model = AblationModel(freq_bins, time_bins, n_channels, n_classes, T_raw, fusion_mode=strategy, **model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
        
        best_val_acc = 0
        epochs_no_improve = 0
        model_save_path = f'./ablation_{task}/models/best_{task}_{strategy}.pth'
        os.makedirs(f'./ablation_{task}/models', exist_ok=True)
        
        for epoch in range(config['num_epochs']):
            train_loss, train_acc = train_epoch(model, loaders['train'], criterion, optimizer, device, is_binary)
            _, val_acc = evaluate(model, loaders['val'], device, criterion, is_binary)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), model_save_path)
                print(f"Epoch {epoch+1}: New Best Val Acc {val_acc:.2f}% - Model Saved")
            else:
                epochs_no_improve += 1
                print(f"Epoch {epoch+1}: Val Acc {val_acc:.2f}% - No improvement for {epochs_no_improve} epochs")

            # Trigger early stop
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered! No improvement for {patience} epochs.")
                break

        model.load_state_dict(torch.load(model_save_path))
        log_entry = {'strategy': strategy, 'val_acc': best_val_acc}
        
        if 'test2' in loaders:
            test2_metrics = evaluate_comprehensive(model, loaders['test2'], device, is_binary)
            log_entry.update({f"test2_{k}": v for k, v in test2_metrics.items()})
            stats = collect_feature_statistics(model, loaders['test2'], device, strategy)
            analysis = analyze_fusion_statistics(stats, strategy, task)
            log_entry.update({'spectral_dominance': analysis['spectral_dominance'], 'temporal_dominance': analysis['temporal_dominance']})
            if strategy == 'spatial_attention':
                n_channels = sample_x[0].shape[1]
                plot_spatial_attention_heatmap(task, stats, n_channels)

        results_log.append(log_entry)

    df = pd.DataFrame(results_log)
    os.makedirs(f'./ablation_{task}/results', exist_ok=True)
    df.to_csv(f"./ablation_{task}/results/ablation_results_{task}.csv", index=False)
    plot_modality_contributions(task, df)
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run fusion ablation study')
    parser.add_argument('--task', type=str, required=True,
                        choices=['SSVEP', 'P300', 'MI', 'Imagined_speech', 'Lee2019_MI', 'Lee2019_SSVEP', 'BNCI2014_P300', 'BI2014b_P300'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    # STFT configuration parameters (optional - uses task defaults if not specified)
    parser.add_argument('--stft_fs', type=int, default=None, 
                        help='STFT sampling frequency (Hz). If not specified, uses task default.')
    parser.add_argument('--stft_nperseg', type=int, default=None,
                        help='STFT window length (nperseg). If not specified, uses task default.')
    parser.add_argument('--stft_noverlap', type=int, default=None,
                        help='STFT overlap length (noverlap). If not specified, uses task default.')
    parser.add_argument('--stft_nfft', type=int, default=None,
                        help='STFT FFT length (nfft). If not specified, uses task default.')
    
    args = parser.parse_args()
    
    # Define parameters clearly
    experiment_config = {
        # Training Hyperparameters
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'seed': 44,
        # Architectural Parameters
        'cnn_filters': 16,
        'lstm_hidden': 128,
        'pos_dim': 16,
        'dropout': 0.3,
        'cnn_dropout': 0.2,
        'use_hidden_layer': True,
        'hidden_dim': 64,
        'patience': 5,
    }
    
    # Add STFT config if provided (None values will use task defaults)
    if args.stft_fs is not None:
        experiment_config['sampling_rate'] = args.stft_fs
    if args.stft_nperseg is not None:
        experiment_config['stft_nperseg'] = args.stft_nperseg
    if args.stft_noverlap is not None:
        experiment_config['stft_noverlap'] = args.stft_noverlap
    if args.stft_nfft is not None:
        experiment_config['stft_nfft'] = args.stft_nfft
    
    run_ablation_study(args.task, experiment_config)