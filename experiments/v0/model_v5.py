"""
Redox Network — v5: REDOX HALF-CELL intake (axis-aligned threshold splits).

Why: trees win on tabular via AXIS-ALIGNED HARD SPLITS ("is feature 7 > 0.5?"). Our old
dense linear encoder BLENDS all features together at intake -> destroys that structure
before the dynamics ever see it. v5 fixes the front door: each input neuron is a HALF-CELL
tied to ONE feature with a learned threshold -> it "reacts" (injects field) as a soft
axis-aligned split, exactly like a tree node. Several half-cells per feature = several
thresholds = a tree's multi-split. Downstream (conserved settling, washboard) is unchanged,
so this isolates the effect of the tree-like intake.

(Differentiable axis-aligned splits are what make NODE / EBM rival GBDT on tabular; the
redox half-cell framing is our novel version.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import RedoxNetwork


class RedoxNetworkV5(RedoxNetwork):
    def __init__(self, n_features, stumps_per_feature=3, n_hidden=32, n_out=8,
                 n_classes=2, **kw):
        n_in = stumps_per_feature * n_features
        super().__init__(n_features, n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                         n_classes=n_classes, **kw)
        # each input neuron (half-cell) is tied to ONE feature (round-robin)
        self.register_buffer("feat_idx", torch.arange(n_in) % n_features)
        self.thresh = nn.Parameter(torch.randn(n_in) * 0.5)     # learned split point
        self.log_tau = nn.Parameter(torch.zeros(n_in))          # learned sharpness

    def forward(self, x, return_conv=False):
        tau = F.softplus(self.log_tau) + 0.1
        xf = x[:, self.feat_idx]                                 # (B, n_in) each cell's feature
        field = torch.tanh((xf - self.thresh) / tau)            # axis-aligned soft split
        q_out, conv = self.settle(field)
        logits = self.readout(q_out)
        return (logits, conv) if return_conv else logits
