import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Configurable Squeeze-and-Excitation Block.
    """
    def __init__(self, channels, mode='channel', reduction=16):
        super(SEBlock, self).__init__()
        self.mode = mode
        # Ensure reduction doesn't squash channels to 0
        reduced = max(channels // reduction, 2)
        
        if mode == 'channel':
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channels, reduced, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(reduced, channels, bias=False),
                nn.Sigmoid()
            )
        elif mode == 'time':
            self.conv = nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, *spatial = x.size()

        if self.mode == 'channel':
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, *([1] * len(spatial)))
            return x * y.expand_as(x)

        elif self.mode == 'time':
            if x.dim() == 4:
                x_flat = x.squeeze(2)
            else:
                x_flat = x
            
            avg_out = torch.mean(x_flat, dim=1, keepdim=True)
            max_out, _ = torch.max(x_flat, dim=1, keepdim=True)
            scale = torch.cat([avg_out, max_out], dim=1)
            
            scale = self.sigmoid(self.conv(scale))
            
            if x.dim() == 4:
                scale = scale.unsqueeze(2)
            
            return x * scale

class DualStreamExtractor(nn.Module):
    """
    Extracts features from a SINGLE channel's data.
    Added Dropout to prevent overfitting.
    """
    def __init__(self, cnn_filters=8, dropout_rate=0.4):
        super(DualStreamExtractor, self).__init__()
        
        # === SPECTRAL STREAM (2D) ===
        self.spec_conv = nn.Sequential(
            nn.Conv2d(1, cnn_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_filters), nn.ReLU(),
            SEBlock(cnn_filters, mode='channel'), 
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(cnn_filters, cnn_filters*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_filters*2), nn.ReLU(),
            SEBlock(cnn_filters*2, mode='channel'),
            nn.AdaptiveAvgPool2d((4, 8)),
            nn.Dropout2d(dropout_rate)
        )
        
        # === TIME STREAM (1D) ===
        self.time_conv = nn.Sequential(
            nn.Conv1d(1, cnn_filters, kernel_size=32, padding='same', bias=False),
            nn.BatchNorm1d(cnn_filters), nn.ReLU(),
            SEBlock(cnn_filters, mode='time'), 
            nn.MaxPool1d(4),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(cnn_filters, cnn_filters*2, kernel_size=8, padding='same', bias=False),
            nn.BatchNorm1d(cnn_filters*2), nn.ReLU(),
            SEBlock(cnn_filters*2, mode='time'),
            nn.AdaptiveAvgPool1d(32),
            nn.Dropout(dropout_rate)
        )
        
        self.output_dim = (cnn_filters * 2 * 4 * 8) + (cnn_filters * 2 * 32)

    def forward(self, x_time, x_spec):
        out_spec = self.spec_conv(x_spec)
        out_spec = out_spec.view(out_spec.size(0), -1) 
        
        out_time = self.time_conv(x_time)
        out_time = out_time.view(out_time.size(0), -1)
        
        return torch.cat([out_spec, out_time], dim=1)

class MultiExpertNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_fft=64, cnn_filters=16, lstm_hidden=128, pos_dim=16, dropout=0.4):
        super(MultiExpertNet, self).__init__()
        self.n_channels = n_channels
        self.n_fft = n_fft
        
        self.extractor = DualStreamExtractor(cnn_filters, dropout_rate=dropout)
        
        self.chan_emb = nn.Embedding(n_channels, pos_dim)
        self.pos_projection = nn.Linear(pos_dim, self.extractor.output_dim)
        
        self.lstm = nn.LSTM(
            input_size=self.extractor.output_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=False
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, n_classes)
        )

    def _compute_stft(self, x):
        b, c, t = x.size()
        x_flat = x.reshape(b * c, t)
        n_fft = min(self.n_fft, t)
        
        spec_complex = torch.stft(
            x_flat, 
            n_fft=n_fft, 
            hop_length=n_fft//2, 
            win_length=n_fft, 
            return_complex=True
        )
        return torch.abs(spec_complex).view(b, c, -1, spec_complex.size(-1))

    def forward(self, x):
        B, C, T = x.size()
        
        # Add Gaussian Noise during training to inputs
        if self.training:
            x = x + torch.randn_like(x) * 0.01

        x_spec = self._compute_stft(x)
        
        # Reshape for parallel processing
        x_time_flat = x.view(B*C, 1, T)
        x_spec_flat = x_spec.view(B*C, 1, x_spec.size(2), x_spec.size(3))
        
        features = self.extractor(x_time_flat, x_spec_flat)
        features = features.view(B, C, -1)
        
        # Position Embeddings
        chan_ids = torch.arange(C, device=x.device).unsqueeze(0).expand(B, C)
        pos = self.chan_emb(chan_ids)
        pos = self.pos_projection(pos)
        features = features + pos
        
        _, (h, _) = self.lstm(features)
        
        return self.classifier(h.squeeze(0))