"""
MultiDiffNet (DualDiff) Model for Multi-Task EEG Classification
Supports: SSVEP, P300, MI (Motor Imagery), Imagined Speech, Lee2019_MI, Lee2019_SSVEP, BNCI2014_P300

Reference: MultiDiffNet: A Multi-Objective Diffusion Framework for Generalizable Brain Decoding
Based on: https://github.com/eddieguo-1128/DualDiff
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
from functools import partial
from einops import reduce

# Add scale-net directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scale-net'))

from seed_utils import seed_everything, worker_init_fn, get_generator
from dataset import load_dataset, TASK_CONFIGS


# ==================== MultiDiffNet Model Components ====================

def get_padding(kernel_size, dilation=1):
    """Padding utility"""
    return int((kernel_size * dilation - dilation) / 2)


class Swish(nn.Module):
    """Swish activation"""
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return x * self.sigmoid(x)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal Positional Embedding for diffusion timesteps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class WeightStandardizedConv1d(nn.Conv1d):
    """Conv1d with weight standardization"""
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv1d(x, normalized_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ResidualConvBlock(nn.Module):
    """Residual convolutional block"""
    def __init__(self, inc: int, outc: int, kernel_size: int, stride=1, gn=8):
        super().__init__()
        self.same_channels = inc == outc
        self.ks = kernel_size
        # Ensure gn divides outc
        actual_gn = min(gn, outc)
        while outc % actual_gn != 0 and actual_gn > 1:
            actual_gn -= 1
        self.conv = nn.Sequential(
            WeightStandardizedConv1d(inc, outc, self.ks, stride, get_padding(self.ks)),
            nn.GroupNorm(actual_gn, outc),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
        if self.same_channels:
            out = (x + x1) / 2
        else:
            out = x1
        return out


class UnetDown(nn.Module):
    """UNet Downsampling Block"""
    def __init__(self, in_channels, out_channels, kernel_size, gn=8, factor=2):
        super(UnetDown, self).__init__()
        self.pool = nn.MaxPool1d(factor)
        self.layer = ResidualConvBlock(in_channels, out_channels, kernel_size, gn=gn)

    def forward(self, x):
        x = self.layer(x)
        x = self.pool(x)
        return x


class UnetUp(nn.Module):
    """UNet Upsampling Block"""
    def __init__(self, in_channels, out_channels, kernel_size, gn=8, factor=2):
        super(UnetUp, self).__init__()
        self.pool = nn.Upsample(scale_factor=factor, mode="nearest")
        self.layer = ResidualConvBlock(in_channels, out_channels, kernel_size, gn=gn)

    def forward(self, x):
        x = self.pool(x)
        x = self.layer(x)
        return x


class AttentionPool1d(nn.Module):
    """Attention pooling over temporal dimension"""
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(in_channels))

    def forward(self, x):  # x: [B, C, T]
        B, C, T = x.shape
        scores = torch.einsum('bct,c->bt', x, self.query)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(x * weights.unsqueeze(1), dim=-1)
        return pooled


class ConditionalUNet(nn.Module):
    """Conditional UNet for DDPM"""
    def __init__(self, in_channels, n_feat=256, gn_final=8):
        super(ConditionalUNet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat

        self.d1_out = n_feat * 1
        self.d2_out = n_feat * 2
        self.d3_out = n_feat * 3

        self.u1_out = n_feat
        self.u2_out = n_feat
        self.u3_out = n_feat
        self.u4_out = in_channels

        self.sin_emb = SinusoidalPosEmb(n_feat)

        self.down1 = UnetDown(in_channels, self.d1_out, 1, gn=8, factor=2)
        self.down2 = UnetDown(self.d1_out, self.d2_out, 1, gn=8, factor=2)
        self.down3 = UnetDown(self.d2_out, self.d3_out, 1, gn=8, factor=2)

        self.up2 = UnetUp(self.d3_out, self.u2_out, 1, gn=8, factor=2)
        self.up3 = UnetUp(self.u2_out + self.d2_out, self.u3_out, 1, gn=8, factor=2)
        self.up4 = UnetUp(self.u3_out + self.d1_out, self.u4_out, 1, gn=gn_final, factor=2)
        self.out = nn.Conv1d(self.u4_out + in_channels, in_channels, 1)

    def forward(self, x, t):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        temb = self.sin_emb(t).view(-1, self.n_feat, 1)

        up1 = self.up2(down3)
        
        if (up1 + temb).shape[-1] != down2.shape[-1]:
            target_len = min((up1 + temb).shape[-1], down2.shape[-1])
            up1 = F.interpolate(up1, size=target_len)
            down2 = F.interpolate(down2, size=target_len)

        up2 = self.up3(torch.cat([up1 + temb, down2], 1))

        if (up2 + temb).shape[-1] != down1.shape[-1]:
            target_len = min((up2 + temb).shape[-1], down1.shape[-1])
            up2 = F.interpolate(up2, size=target_len)
            down1 = F.interpolate(down1, size=target_len)

        up3 = self.up4(torch.cat([up2 + temb, down1], 1))

        if up3.shape[-1] != x.shape[-1]:
            target_len = min(up3.shape[-1], x.shape[-1])
            up3 = F.interpolate(up3, size=target_len)
            x = F.interpolate(x, size=target_len)

        out = self.out(torch.cat([up3, x], 1))

        down = (down1, down2, down3)
        up = (up1, up2, up3)
        return out, down, up


class MultiDiffNetEncoder(nn.Module):
    """EEGNet-style Encoder for MultiDiffNet"""
    def __init__(self, n_classes, n_channels=64, n_samples=250, dropout_rate=0.5,
                 kernel_length=64, F1=8, D=2, F2=16, F3=32, encoder_dim=256):
        super(MultiDiffNetEncoder, self).__init__()

        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.dropout3 = nn.Dropout(p=dropout_rate)

        # Block 1
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kernel_length), 
                               padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise_conv = nn.Conv2d(F1, F1, kernel_size=(n_channels, 1), 
                                        groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1)
        self.activation1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 2))

        # Block 2
        self.sep_conv = nn.Conv2d(F1, F2, kernel_size=(1, 16), padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 2))

        # Block 3
        self.conv3 = nn.Conv2d(F2, F3, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(F3)
        self.activation3 = nn.ELU()
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 2))

        # Projection layers
        self.proj1 = nn.Conv1d(F1, encoder_dim, kernel_size=1)
        self.proj2 = nn.Conv1d(F2, encoder_dim, kernel_size=1)
        self.proj3 = nn.Conv1d(F3, encoder_dim, kernel_size=1)

        self.att_pool = AttentionPool1d(encoder_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, C, T]

        # Block 1
        dn1 = self.conv1(x)
        dn1 = self.bn1(dn1)
        dn1 = self.depthwise_conv(dn1)
        dn1 = self.bn2(dn1)
        dn1 = self.activation1(dn1)
        dn1 = self.pool1(dn1)
        dn1 = self.dropout1(dn1)

        # Block 2
        dn2 = self.sep_conv(dn1)
        dn2 = self.bn3(dn2)
        dn2 = self.activation2(dn2)
        dn2 = self.pool2(dn2)
        dn2 = self.dropout2(dn2)

        # Block 3
        dn3 = self.conv3(dn2)
        dn3 = self.bn4(dn3)
        dn3 = self.activation3(dn3)
        dn3 = self.pool3(dn3)
        dn3 = self.dropout3(dn3)

        # Squeeze to [B, C, T]
        dn1_ = dn1.squeeze(2)
        dn2_ = dn2.squeeze(2)
        dn3_ = dn3.squeeze(2)

        # Project to encoder_dim
        dn1_out = self.proj1(dn1_)
        dn2_out = self.proj2(dn2_)
        dn3_out = self.proj3(dn3_)

        # z vector via attention pooling
        z = self.att_pool(dn3_out)

        down = (dn1_out, dn2_out, dn3_out)
        return (down, z)


class LinearClassifier(nn.Module):
    """Linear classifier head"""
    def __init__(self, in_dim, latent_dim, n_classes, n_channels=64):
        super().__init__()
        # Compute number of groups for GroupNorm (must divide latent_dim)
        num_groups = min(4, latent_dim)
        while latent_dim % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        
        self.linear_out = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=latent_dim),
            nn.GroupNorm(num_groups, latent_dim),
            nn.PReLU(),
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.GroupNorm(num_groups, latent_dim),
            nn.PReLU(),
            nn.Linear(in_features=latent_dim, out_features=n_classes))
        self.eeg_proj = nn.Conv1d(n_channels, 256, kernel_size=1)
        self.att_pool = AttentionPool1d(256)

    def forward(self, x): 
        if x.dim() == 2:
            return self.linear_out(x)
        elif x.dim() == 3:
            x = self.eeg_proj(x)
            x = self.att_pool(x)
            return self.linear_out(x)
        else:
            raise ValueError(f"Unexpected input shape to LinearClassifier: {x.shape}")


class DiffE(nn.Module):
    """DiffE combines encoder and classifier (no decoder for simplicity)"""
    def __init__(self, encoder, fc):
        super(DiffE, self).__init__()
        self.encoder = encoder
        self.fc = fc

    def forward(self, x0, ddpm_out=None):
        encoder_out = self.encoder(x0)
        z = encoder_out[1]
        fc_out = self.fc(z)
        return fc_out, z


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule for DDPM"""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def ddpm_schedules(beta1, beta2, T):
    """DDPM noise schedules"""
    beta_t = cosine_beta_schedule(T, s=0.008).float()
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()
    sqrtab = torch.sqrt(alphabar_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    return {"sqrtab": sqrtab, "sqrtmab": sqrtmab}


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model"""
    def __init__(self, nn_model, betas, n_T, device):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device

    def forward(self, x):
        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)
        x_t = self.sqrtab[_ts, None, None] * x + self.sqrtmab[_ts, None, None] * noise
        times = _ts / self.n_T
        output, down, up = self.nn_model(x_t, times)
        return output, down, up, noise, times


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss"""
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        sim = torch.matmul(features, features.T) / self.temperature

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        mask = mask * logits_mask
        sim = sim - 1e9 * (1 - logits_mask)

        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-6)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        loss = -mean_log_prob_pos.mean()
        return loss


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""
    def __init__(self, input_dim=256, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, z):
        return F.normalize(self.net(z), dim=1)


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
        if np.random.random() < 0.5:
            x = x + np.random.randn(*x.shape).astype(np.float32) * 0.05 * np.std(x)
        if np.random.random() < 0.5:
            x = x * np.random.uniform(0.8, 1.2)
        if np.random.random() < 0.3:
            shift = np.random.randint(-10, 11)
            x = np.roll(x, shift, axis=-1)
        return x
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx].copy()
        y = self.labels[idx]
        
        if self.augment:
            x = self._augment(x)
        
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

def train_epoch(diffe, ddpm, loader, criterion, optimizer_diffe, optimizer_ddpm,
                device, is_binary=False, use_ddpm=True, use_supcon=True,
                proj_head=None, supcon_loss=None, alpha=1.0, beta=0.1):
    """Train one epoch with both DDPM and DiffE"""
    diffe.train()
    if use_ddpm and ddpm is not None:
        ddpm.train()
    
    total_loss, correct, total = 0, 0, 0
    
    pbar = tqdm(loader, desc='Train', ncols=120)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Convert labels for binary classification
        if is_binary:
            labels_loss = labels.float().unsqueeze(1)
        else:
            labels_loss = labels
        
        # Train DDPM (denoising)
        if use_ddpm and ddpm is not None and optimizer_ddpm is not None:
            optimizer_ddpm.zero_grad()
            x_hat, down, up, noise, t = ddpm(inputs)
            
            # Align dimensions if needed
            if x_hat.shape[-1] != inputs.shape[-1]:
                target_len = min(x_hat.shape[-1], inputs.shape[-1])
                x_hat = F.interpolate(x_hat, size=target_len)
                inputs_aligned = F.interpolate(inputs, size=target_len)
            else:
                inputs_aligned = inputs
            
            loss_ddpm = F.l1_loss(x_hat, inputs_aligned)
            loss_ddpm.backward()
            optimizer_ddpm.step()
        
        # Train DiffE (classification)
        optimizer_diffe.zero_grad()
        outputs, z = diffe(inputs)
        loss_cls = criterion(outputs, labels_loss)
        
        # SupCon loss
        loss_supcon = torch.tensor(0.0, device=device)
        if use_supcon and proj_head is not None and supcon_loss is not None:
            z_proj = proj_head(z)
            loss_supcon = supcon_loss(z_proj, labels)
        
        # Combined loss
        loss = alpha * loss_cls + beta * loss_supcon
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unwrap_model(diffe).parameters(), 1.0)
        optimizer_diffe.step()
        
        total_loss += loss.item()
        
        # Prediction
        if is_binary:
            pred = (torch.sigmoid(outputs) > 0.5).squeeze(1).long()
        else:
            _, pred = outputs.max(1)
        
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{total_loss/(pbar.n+1):.4f}', 
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(loader), 100. * correct / total


def evaluate(diffe, loader, device, criterion=None, is_binary=False):
    """Evaluate model"""
    diffe.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Eval', ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            
            if is_binary:
                labels_loss = labels.float().unsqueeze(1)
            else:
                labels_loss = labels
            
            outputs, z = diffe(inputs)
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


# ==================== MultiDiffNet Configuration per Task ====================

def get_multidiffnet_config(task: str, n_channels: int, n_samples: int, sampling_rate: int) -> Dict:
    """Get MultiDiffNet hyperparameters optimized for each task"""
    
    # Determine appropriate GroupNorm groups based on channel count
    if n_channels == 1:
        gn_final = 1
    elif n_channels < 8:
        gn_final = 1
    else:
        gn_final = 8
    
    config = {
        'encoder_dim': 256,
        'ddpm_dim': 128,
        'n_T': 1000,
        'F1': 16,
        'D': 2,
        'F2': 32,
        'F3': 64,
        'kernel_length': max(sampling_rate // 4, 32),
        'dropout_rate': 0.3,
        'fc_dim': 512,
        'gn_final': gn_final,
        # Training params
        'use_ddpm': True,
        'use_supcon': True,
        'supcon_temp': 0.07,
        'alpha': 1.0,  # classification loss weight
        'beta': 0.1,   # supcon loss weight
    }
    
    # Task-specific adjustments
    if task in ['SSVEP', 'Lee2019_SSVEP']:
        config['kernel_length'] = max(sampling_rate // 2, 64)
        config['F1'] = 16
    elif task in ['P300', 'BNCI2014_P300']:
        config['kernel_length'] = sampling_rate // 4
        config['use_supcon'] = False  # Binary classification may not benefit
    elif task in ['MI', 'Lee2019_MI']:
        config['kernel_length'] = 64
        config['F1'] = 16
    elif task == 'Imagined_speech':
        config['kernel_length'] = min(256, n_samples // 8)
        config['F1'] = 16
        config['F2'] = 32
    
    return config


# ==================== Main Training ====================

def train_task(task: str, config: Optional[Dict] = None, model_path: Optional[str] = None) -> Tuple:
    """Train MultiDiffNet for a specific EEG task"""
    
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
    print(f"MultiDiffNet - {task} Classification")
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
    
    # ====== Get Model Configuration ======
    model_config = get_multidiffnet_config(
        task, n_channels, n_samples, config['sampling_rate']
    )
    print(f"\nMultiDiffNet Configuration:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")
    
    # ====== Create Models ======
    n_classes = config['n_classes']
    
    # Encoder
    encoder = MultiDiffNetEncoder(
        n_classes=n_classes,
        n_channels=n_channels,
        n_samples=n_samples,
        dropout_rate=model_config['dropout_rate'],
        kernel_length=model_config['kernel_length'],
        F1=model_config['F1'],
        D=model_config['D'],
        F2=model_config['F2'],
        F3=model_config['F3'],
        encoder_dim=model_config['encoder_dim']
    ).to(device)
    
    # Classifier
    fc = LinearClassifier(
        in_dim=model_config['encoder_dim'],
        latent_dim=model_config['fc_dim'],
        n_classes=1 if n_classes == 2 else n_classes,
        n_channels=n_channels
    ).to(device)
    
    # DiffE (encoder + classifier)
    diffe = DiffE(encoder, fc).to(device)
    
    # DDPM (optional)
    ddpm = None
    optimizer_ddpm = None
    if model_config['use_ddpm']:
        ddpm_model = ConditionalUNet(
            in_channels=n_channels, 
            n_feat=model_config['ddpm_dim'],
            gn_final=model_config['gn_final']
        ).to(device)
        ddpm = DDPM(
            nn_model=ddpm_model, 
            betas=(1e-6, 1e-2), 
            n_T=model_config['n_T'], 
            device=device
        ).to(device)
        ddpm = wrap_model_multi_gpu(ddpm, n_gpus)
        optimizer_ddpm = torch.optim.RMSprop(unwrap_model(ddpm).parameters(), lr=config['lr'])
    
    # Projection head and SupCon loss (optional)
    proj_head = None
    supcon_loss = None
    if model_config['use_supcon']:
        proj_head = ProjectionHead(
            input_dim=model_config['encoder_dim'], 
            proj_dim=128
        ).to(device)
        supcon_loss = SupConLoss(temperature=model_config['supcon_temp'])
    
    # Print model info
    n_params_diffe = sum(p.numel() for p in diffe.parameters())
    n_params_ddpm = sum(p.numel() for p in ddpm.parameters()) if ddpm else 0
    print(f"\nDiffE Parameters: {n_params_diffe:,}")
    print(f"DDPM Parameters: {n_params_ddpm:,}")
    print(f"Total Parameters: {n_params_diffe + n_params_ddpm:,}")
    print(f"Classes: {n_classes}")
    
    # Wrap DiffE for multi-GPU
    diffe = wrap_model_multi_gpu(diffe, n_gpus)
    
    # ====== Loss & Optimizer ======
    is_binary = (n_classes == 2)
    if is_binary:
        criterion = nn.BCEWithLogitsLoss()
        print("Using BCEWithLogitsLoss for binary classification")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"Using CrossEntropyLoss for {n_classes}-class classification")
    
    optimizer_diffe = torch.optim.Adam(
        diffe.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    # Add projection head params to optimizer if using SupCon
    if proj_head is not None:
        optimizer_diffe.add_param_group({'params': proj_head.parameters()})
    
    scheduler_type = config.get('scheduler', 'ReduceLROnPlateau')
    if scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_diffe, T_max=config['num_epochs'] // 2, eta_min=1e-6
        )
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_diffe, mode='min', factor=0.5, patience=5
        )
    else:
        raise ValueError(f"Invalid scheduler: {scheduler_type}")
    
    # ====== Training Loop ======
    best_val_acc = 0
    patience_counter = 0
    
    if model_path is None:
        model_path = f'best_multidiffnet_{task.lower()}_model.pth'
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch [{epoch+1}/{config['num_epochs']}]")
        
        # Adjust beta (SupCon weight) with scheduler
        beta_val = min(1.0, epoch / 50) * model_config['beta']
        
        train_loss, train_acc = train_epoch(
            diffe, ddpm, train_loader, criterion, 
            optimizer_diffe, optimizer_ddpm, device,
            is_binary=is_binary,
            use_ddpm=model_config['use_ddpm'],
            use_supcon=model_config['use_supcon'],
            proj_head=proj_head,
            supcon_loss=supcon_loss,
            alpha=model_config['alpha'],
            beta=beta_val
        )
        
        val_loss, val_acc = evaluate(diffe, val_loader, device, criterion, is_binary=is_binary)
        
        if scheduler_type == 'CosineAnnealingLR':
            scheduler.step()
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        
        current_lr = optimizer_diffe.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dict = {
                'epoch': epoch,
                'diffe_state_dict': unwrap_model(diffe).state_dict(),
                'optimizer_diffe_state_dict': optimizer_diffe.state_dict(),
                'best_val_acc': best_val_acc,
                'task': task,
                'config': config,
                'model_config': model_config,
                'n_channels': n_channels,
                'n_samples': n_samples,
            }
            if ddpm is not None:
                save_dict['ddpm_state_dict'] = unwrap_model(ddpm).state_dict()
            if proj_head is not None:
                save_dict['proj_head_state_dict'] = proj_head.state_dict()
            
            torch.save(save_dict, model_path)
            print(f"✓ Best model saved! ({val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{config['patience']}")
        
        if patience_counter >= config['patience']:
            print("\nEarly stopping triggered!")
            break
    
    # ====== Final Evaluation ======
    print(f"\n{'='*70}")
    print("Loading best model for final evaluation...")
    print(f"Best model path: {model_path}")
    checkpoint = torch.load(model_path)
    unwrap_model(diffe).load_state_dict(checkpoint['diffe_state_dict'])
    
    results = {'val': best_val_acc}
    
    if test1_loader:
        test1_loss, test1_acc = evaluate(diffe, test1_loader, device, criterion, is_binary=is_binary)
        results['test1'] = test1_acc
        results['test1_loss'] = test1_loss
    
    if test2_loader:
        test2_loss, test2_acc = evaluate(diffe, test2_loader, device, criterion, is_binary=is_binary)
        results['test2'] = test2_acc
        results['test2_loss'] = test2_loss
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS - {task} (MultiDiffNet)")
    print(f"{'='*70}")
    print(f"Best Val Acc:    {best_val_acc:.2f}%")
    if 'test1' in results:
        print(f"Test1 (Seen):    {results['test1']:.2f}% (loss {results['test1_loss']:.4f})")
    if 'test2' in results:
        print(f"Test2 (Unseen):  {results['test2']:.2f}% (loss {results['test2_loss']:.4f})")
    print(f"{'='*70}")
    
    return diffe, results


def train_all_tasks(tasks: Optional[list] = None, save_dir: str = './checkpoints'):
    """Train MultiDiffNet models for all specified tasks"""
    if tasks is None:
        tasks = ['SSVEP', 'P300', 'MI', 'Imagined_speech', 'Lee2019_MI', 'Lee2019_SSVEP', 'BNCI2014_P300']
    
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = {}
    
    print("=" * 80)
    print("MultiDiffNet - MULTI-TASK EEG CLASSIFICATION")
    print("=" * 80)
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"TASK: {task}")
        print(f"{'='*60}")
        
        try:
            model_path = os.path.join(save_dir, f'best_multidiffnet_{task.lower()}_model.pth')
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
    print("SUMMARY RESULTS (MultiDiffNet)")
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
    print("MultiDiffNet MULTI-TASK TRAINING COMPLETED!")
    print(f"{'='*80}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MultiDiffNet on EEG tasks')
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
        model_path = os.path.join(args.save_dir, f'best_multidiffnet_{args.task.lower()}_model.pth')
        os.makedirs(args.save_dir, exist_ok=True)
        model, results = train_task(args.task, config=config, model_path=model_path)
