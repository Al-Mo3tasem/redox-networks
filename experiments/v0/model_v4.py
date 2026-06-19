"""
Redox Network — v4: TWO-STAGE intake = global hard-prune  ->  conditional gate.

Lesson from v2 vs v3:
  - v2 (global blind gate): killed junk to exactly 0 -> ROBUST, but too crude (no "it
    depends", hurt clean accuracy).
  - v3 (conditional gate-net reading the whole row): got "it depends" + clean accuracy,
    but the gate-net READS the junk -> junk sneaks in the back door -> robustness lost.

v4 puts two doormen in sequence:
  Stage 1 GLOBAL gate (input-INDEPENDENT, Hard-Concrete, per feature): slams obvious junk
          to EXACTLY 0. Robustness. Junk dies here, before anything else sees it.
  Stage 2 CONDITIONAL gate-net (reads only the survivors x' = global_gate * x): per-sample
          "it depends" fine-tuning. Conditional power + clean accuracy.
Junk is zeroed before the smart gate-net, so it can't poison the conditional decisions.
Everything downstream (conserved flow, washboard, settle) is identical to v0.
"""
import math
import torch
import torch.nn as nn

TWO_PI = 2.0 * math.pi


class RedoxNetworkV4(nn.Module):
    def __init__(self, n_features, n_in=16, n_hidden=32, n_out=8, n_classes=2,
                 charge_per_neuron=0.5, T=30, eta=0.1, beta=0.05,
                 coupling_budget=0.4, conserve=True, n_levels=4, conf_coef=1.0,
                 gate_hidden=64, global_sparse_mult=1.0, cond_sparse_mult=0.0):
        super().__init__()
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        self.N = n_in + n_hidden + n_out
        self.T, self.eta, self.beta = T, eta, beta
        self.coupling_budget = coupling_budget
        self.conserve = conserve
        self.n_levels, self.conf_coef = n_levels, conf_coef
        self.global_sparse_mult = global_sparse_mult
        self.cond_sparse_mult = cond_sparse_mult
        self.q0 = charge_per_neuron
        self.Q = charge_per_neuron * self.N

        self.chi_in = nn.Parameter(torch.zeros(n_in))
        self.chi_hidden = nn.Parameter(torch.zeros(n_hidden))
        self.chi_out = nn.Parameter(torch.zeros(n_out))
        self.W1 = nn.Parameter(torch.randn(n_in, n_hidden) * 0.1)
        self.W2 = nn.Parameter(torch.randn(n_hidden, n_out) * 0.1)

        self.beta_hc, self.gamma_hc, self.zeta_hc = 2.0 / 3.0, -0.1, 1.1
        # Stage 1: global per-feature hard-prune gate (input-independent)
        self.log_alpha_global = nn.Parameter(torch.full((n_features,), 2.0))
        # Stage 2: conditional gate-network (reads pruned survivors)
        self.gate_net = nn.Sequential(
            nn.Linear(n_features, gate_hidden), nn.ReLU(),
            nn.Linear(gate_hidden, n_features))
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, 2.0)

        self.encoder = nn.Linear(n_features, n_in)
        self.readout = nn.Linear(n_out, n_classes)
        self.last_sparsity = None
        # Staged training: Stage 1 holds the conditional gate OPEN so the GLOBAL gate is
        # the only feature-selector (forced to prune junk). Stage 2 turns it back on.
        self.use_conditional = True
        self.hard_global = True   # exact-zero global prune (straight-through)

    def _hc(self, la):
        if self.training:
            u = torch.rand(la.shape, device=la.device).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + la) / self.beta_hc)
        else:
            s = torch.sigmoid(la)
        return (s * (self.zeta_hc - self.gamma_hc) + self.gamma_hc).clamp(0.0, 1.0)

    def _hc_hard(self, la):
        # Straight-through HARD gate: forward = exact 0/1 (a pruned feature is EXACTLY 0,
        # so the encoder can't leak it back); backward = soft gradient (still trainable).
        soft = self._hc(la)
        hard = (soft > 0.5).float()
        return hard + (soft - soft.detach())

    def _frac(self, la):
        c = self.beta_hc * math.log(-self.gamma_hc / self.zeta_hc)
        return torch.sigmoid(la - c).mean()

    def _count(self, la):
        # expected NUMBER of active gates -> strong per-feature pruning pressure (~lambda),
        # unlike the fraction (~lambda/D) which is too weak to push junk to zero.
        c = self.beta_hc * math.log(-self.gamma_hc / self.zeta_hc)
        return torch.sigmoid(la - c).sum()

    def make_field(self, x):
        gam = (self._hc_hard(self.log_alpha_global) if self.hard_global
               else self._hc(self.log_alpha_global))      # (D,) global prune (junk -> exact 0)
        x1 = gam.unsqueeze(0) * x                          # survivors only
        if self.use_conditional:                           # Stage 2: conditional gate on
            la_c = self.gate_net(x1)                        # (B, D) conditional logits
            g = self._hc(la_c)                              # (B, D) per-sample gates
            cond_pen = self.cond_sparse_mult * self._frac(la_c)
        else:                                              # Stage 1: gate held fully open
            g = torch.ones_like(x1)
            cond_pen = x1.new_zeros(())
        # global gate is the junk-killer: COUNT penalty (strong per-feature pruning).
        self.last_sparsity = self.global_sparse_mult * self._count(self.log_alpha_global) + cond_pen
        return self.encoder(g * x1)

    @torch.no_grad()
    def global_gates(self):
        self.eval()
        return (self._hc_hard if self.hard_global else self._hc)(self.log_alpha_global)

    @torch.no_grad()
    def gate_values(self, x):
        self.eval()
        gam = self._hc(self.log_alpha_global)
        return self._hc(self.gate_net(gam.unsqueeze(0) * x))

    def _restoring(self, q):
        g = torch.zeros_like(q)
        if self.beta > 0:
            g = g + self.beta * TWO_PI * torch.sin(TWO_PI * q)
        if self.conf_coef > 0:
            g = g + self.conf_coef * (torch.relu(q - (self.n_levels - 1)) - torch.relu(-q))
        return g

    def _bounded_W(self):
        def norm(W):
            deg = torch.maximum(W.abs().sum(dim=1).max(), W.abs().sum(dim=0).max())
            scale = torch.clamp(self.eta * deg / self.coupling_budget, min=1.0)
            return W / scale
        return norm(self.W1), norm(self.W2)

    def settle(self, field, return_diag=False):
        B = field.shape[0]
        dev = field.device
        W1b, W2b = self._bounded_W()
        q_in = torch.full((B, self.n_in), self.q0, device=dev)
        q_hidden = torch.full((B, self.n_hidden), self.q0, device=dev)
        q_out = torch.full((B, self.n_out), self.q0, device=dev)
        gaps, totals = [], []
        conv_pen = field.new_zeros(())
        for _ in range(self.T):
            mu_in = (q_in - self.chi_in - field
                     - q_hidden @ W1b.t() + self._restoring(q_in))
            mu_hidden = (q_hidden - self.chi_hidden
                         - q_in @ W1b - q_out @ W2b.t() + self._restoring(q_hidden))
            mu_out = (q_out - self.chi_out
                      - q_hidden @ W2b + self._restoring(q_out))
            if self.conserve:
                flow1 = self.eta * W1b.unsqueeze(0) * (mu_in.unsqueeze(2) - mu_hidden.unsqueeze(1))
                flow2 = self.eta * W2b.unsqueeze(0) * (mu_hidden.unsqueeze(2) - mu_out.unsqueeze(1))
                dq_in = -flow1.sum(dim=2)
                dq_hidden = flow1.sum(dim=1) - flow2.sum(dim=2)
                dq_out = flow2.sum(dim=1)
            else:
                dq_in, dq_hidden, dq_out = -self.eta * mu_in, -self.eta * mu_hidden, -self.eta * mu_out
            q_in, q_hidden, q_out = q_in + dq_in, q_hidden + dq_hidden, q_out + dq_out
            conv_pen = (dq_in.pow(2).sum(1) + dq_hidden.pow(2).sum(1) + dq_out.pow(2).sum(1)).mean()
            if return_diag:
                with torch.no_grad():
                    gaps.append(max(dq_in.abs().max().item(), dq_hidden.abs().max().item(),
                                    dq_out.abs().max().item()))
                    totals.append((q_in.sum(1) + q_hidden.sum(1) + q_out.sum(1)).mean().item())
        if return_diag:
            return q_out, conv_pen, gaps, totals
        return q_out, conv_pen

    def forward(self, x, return_conv=False):
        field = self.make_field(x)
        q_out, conv_pen = self.settle(field)
        logits = self.readout(q_out)
        if return_conv:
            return logits, conv_pen
        return logits
