"""
Redox Network — v6: "Redox Forest" = axis-aligned leaf-co-occurrence feature map fused
with our conserved-charge equilibrium combiner.

Deep-research finding (research/02): trees win because they ARE a learned, AXIS-ALIGNED,
NON-SMOOTH, feature-selective KERNEL (leaf co-occurrence). Our equilibrium settle IS a
kernel / electrical-network machine -- we just used the wrong (smooth/dense) affinity.
v6 fixes the FRONT DOOR with the mathematically-right object:

  phi(x) = soft leaf-membership indicators from M learned AXIS-ALIGNED soft oblivious
           trees (each split is a threshold on ONE feature -> breaks rotation invariance;
           leaf indicators are non-smooth + feature-selective).

Two readouts (ablation of our contribution):
  settle=False : phi -> Linear -> logits     (a jointly-fit soft forest; the "math only")
  settle=True  : phi -> conserved redox settling -> readout  (our equilibrium combiner)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import RedoxNetwork


class RedoxForest(nn.Module):
    def __init__(self, n_features, n_trees=32, depth=4, n_classes=2, settle=True,
                 seed=0, n_hidden=32, n_out=8, **redox):
        super().__init__()
        self.n_trees, self.depth, self.L = n_trees, depth, 2 ** depth
        self.settle_on = settle
        # fixed random feature assignment per (tree, split) -> axis-aligned, diverse trees
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("feat_idx", torch.randint(0, n_features, (n_trees, depth), generator=g))
        self.thresh = nn.Parameter(torch.randn(n_trees, depth) * 0.5)   # learned split points
        self.log_tau = nn.Parameter(torch.zeros(n_trees, depth))        # learned sharpness
        # binary codes of each leaf: (L, depth)
        code = torch.tensor([[(l >> k) & 1 for k in range(depth)] for l in range(self.L)],
                            dtype=torch.float32)
        self.register_buffer("code", code)

        n_in = n_trees * self.L
        if settle:
            self.core = RedoxNetwork(n_features=1, n_in=n_in, n_hidden=n_hidden,
                                     n_out=n_out, n_classes=n_classes, **redox)
        else:
            self.readout = nn.Linear(n_in, n_classes)

    def phi(self, x):
        # soft leaf memberships per tree (each tree's memberships sum to 1)
        xf = x[:, self.feat_idx]                                   # (B, n_trees, depth)
        tau = F.softplus(self.log_tau) + 0.1
        dec = torch.sigmoid((xf - self.thresh) / tau)              # (B, n_trees, depth)
        decs = dec.unsqueeze(2)                                    # (B, n_trees, 1, depth)
        code = self.code.view(1, 1, self.L, self.depth)           # (1,1,L,depth)
        memb = (code * decs + (1 - code) * (1 - decs)).prod(dim=3)  # (B, n_trees, L)
        return memb.reshape(x.shape[0], -1)                        # (B, n_trees*L)

    def forward(self, x, return_conv=False):
        f = self.phi(x)
        if self.settle_on:
            q_out, conv = self.core.settle(f)
            logits = self.core.readout(q_out)
        else:
            logits = self.readout(f)
            conv = f.new_zeros(())
        return (logits, conv) if return_conv else logits
