"""
Compact, faithful implementations of standard tabular neural-net baselines, so the Redox
Network is judged against other NEURAL networks (not only trees). ResNet and FT-Transformer
follow Gorishniy et al. (2021), "Revisiting Deep Learning Models for Tabular Data".
"""
import torch
import torch.nn as nn


class TabResNet(nn.Module):
    """Tabular ResNet: a stack of pre-norm residual MLP blocks."""
    def __init__(self, d_in, n_classes=2, d=128, n_blocks=3, d_hidden=256, p=0.1):
        super().__init__()
        self.first = nn.Linear(d_in, d)
        self.blocks = nn.ModuleList(nn.ModuleDict(dict(
            norm=nn.BatchNorm1d(d), lin1=nn.Linear(d, d_hidden),
            lin2=nn.Linear(d_hidden, d), drop=nn.Dropout(p))) for _ in range(n_blocks))
        self.head_norm = nn.BatchNorm1d(d)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        x = self.first(x)
        for b in self.blocks:
            z = b["norm"](x)
            z = torch.relu(b["lin1"](z))
            z = b["drop"](z)
            z = b["lin2"](z)
            x = x + z
        return self.head(torch.relu(self.head_norm(x)))


class FTTransformer(nn.Module):
    """Feature-Tokenizer Transformer: each scalar feature -> a token, plus a CLS token."""
    def __init__(self, d_in, n_classes=2, d_token=64, n_layers=3, n_heads=8, p=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_in, d_token) * 0.02)   # per-feature embed
        self.bias = nn.Parameter(torch.zeros(d_in, d_token))
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        layer = nn.TransformerEncoderLayer(d_token, n_heads, dim_feedforward=2 * d_token,
                                           dropout=p, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_classes)

    def forward(self, x):
        tok = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)  # (B,d_in,d_token)
        cls = self.cls.expand(x.shape[0], -1, -1)
        h = self.enc(torch.cat([cls, tok], dim=1))
        return self.head(self.norm(h[:, 0]))
