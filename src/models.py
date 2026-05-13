"""
Model architectures for EEG classification.

Includes:
- EEGNet: compact convolutional network (primary model)
- EEGTransformer: transformer with convolutional tokenizer
- EEGGAT: graph attention network (experimental)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class EEGNet(nn.Module):
    """EEGNet from Lawhern et al. 2018, adapted for binary classification."""
    def __init__(self, in_channels=19, n_classes=2, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding=(0, 32)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, (in_channels, 1), groups=8),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, n_classes)
        )

    def forward(self, x):
        # x: (batch, channels, time)
        x = x.unsqueeze(1)          # (batch, 1, channels, time)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.classifier(x)


class ConvTokenizer(nn.Module):
    def __init__(self, in_channels=19, emb_dim=32, kernel_size=5, stride=2):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, groups=in_channels,
                                   padding=kernel_size//2)
        self.pointwise = nn.Conv1d(in_channels, emb_dim, 1)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = x.transpose(1, 2)
        return self.norm(x)


class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim=32, num_heads=4, hidden_dim=64, num_layers=1, dropout=0.5):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)


class EEGTransformer(nn.Module):
    def __init__(self, in_channels=19, emb_dim=32, num_heads=4,
                 hidden_dim=64, num_layers=1, num_classes=2, dropout=0.5):
        super().__init__()
        self.tokenizer = ConvTokenizer(in_channels, emb_dim)
        self.transformer = TransformerEncoder(emb_dim, num_heads, hidden_dim, num_layers, dropout)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        return self.classifier(x)


class EEGGAT(nn.Module):
    """Graph Attention Network for EEG."""
    def __init__(self, num_nodes=19, in_features=64, hidden=64, out_classes=2, heads=4, dropout=0.3):
        super().__init__()
        self.temporal_extract = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(32)
        )
        self.node_projection = nn.Linear(16 * 32, in_features)
        self.gat1 = GATConv(in_features, hidden, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden * heads, hidden, heads=1, concat=False, dropout=dropout)
        self.classifier = nn.Linear(hidden, out_classes)

    def forward(self, x, edge_index):
        batch_size = x.size(0)
        x = x.view(batch_size * 19, 1, -1)
        x = self.temporal_extract(x)
        x = x.view(x.size(0), -1)
        x = self.node_projection(x)
        edge_index = edge_index.to(x.device)
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        batch_vector = torch.arange(batch_size, device=x.device).repeat_interleave(19)
        x = global_mean_pool(x, batch_vector)
        return self.classifier(x)
