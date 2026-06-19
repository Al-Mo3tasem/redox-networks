"""
Redox Network — v0 prototype (stabilized, with ablation switches).
Spec: experiments/v0-spec.md (D1-D6) + Run-1 stability fix + Run-2 confirmed settling.

A layered RELAXING network. A conserved "charge" is poured in (fixed budget Q).
Each input row tilts the landscape (applied field). Charge flows donor->acceptor
between adjacent layers, conserved at every hand-off, self-limiting (so it settles),
biased toward DISCRETE MULTI-LEVELS via a smooth "washboard" + box confinement to
[0, n_levels-1]. Inference = settle to equilibrium. Training (v0) = backprop through
the unrolled settling against a task loss (+ convergence penalty).

Ablation switches (for the focused eval):
  conserve=False   -> free per-neuron descent (NO conservation law) = "fancy relaxing net"
  beta=0.0         -> NO washboard (no discrete-level preference)
  n_levels=2       -> binary levels (like prior art) vs >2 = our multi-level differentiator
"""
import math
import torch
import torch.nn as nn

TWO_PI = 2.0 * math.pi


class RedoxNetwork(nn.Module):
    def __init__(self, n_features, n_in=16, n_hidden=32, n_out=8, n_classes=2,
                 charge_per_neuron=0.5, T=30, eta=0.1, beta=0.05,
                 coupling_budget=0.4, conserve=True, n_levels=4, conf_coef=1.0):
        super().__init__()
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        self.N = n_in + n_hidden + n_out
        self.T, self.eta, self.beta = T, eta, beta
        self.coupling_budget = coupling_budget
        self.conserve = conserve
        self.n_levels = n_levels
        self.conf_coef = conf_coef
        self.q0 = charge_per_neuron
        self.Q = charge_per_neuron * self.N

        self.chi_in = nn.Parameter(torch.zeros(n_in))
        self.chi_hidden = nn.Parameter(torch.zeros(n_hidden))
        self.chi_out = nn.Parameter(torch.zeros(n_out))
        self.W1 = nn.Parameter(torch.randn(n_in, n_hidden) * 0.1)
        self.W2 = nn.Parameter(torch.randn(n_hidden, n_out) * 0.1)
        self.encoder = nn.Linear(n_features, n_in)
        self.readout = nn.Linear(n_out, n_classes)

    def _restoring(self, q):
        # washboard (discrete-level preference) + soft box confinement to [0, n_levels-1]
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
                # conservative donor->acceptor flows (total charge invariant)
                flow1 = self.eta * W1b.unsqueeze(0) * (mu_in.unsqueeze(2) - mu_hidden.unsqueeze(1))
                flow2 = self.eta * W2b.unsqueeze(0) * (mu_hidden.unsqueeze(2) - mu_out.unsqueeze(1))
                dq_in = -flow1.sum(dim=2)
                dq_hidden = flow1.sum(dim=1) - flow2.sum(dim=2)
                dq_out = flow2.sum(dim=1)
            else:
                # ABLATION: free per-neuron descent — NO conservation
                dq_in = -self.eta * mu_in
                dq_hidden = -self.eta * mu_hidden
                dq_out = -self.eta * mu_out

            q_in = q_in + dq_in
            q_hidden = q_hidden + dq_hidden
            q_out = q_out + dq_out
            conv_pen = (dq_in.pow(2).sum(1) + dq_hidden.pow(2).sum(1)
                        + dq_out.pow(2).sum(1)).mean()
            if return_diag:
                with torch.no_grad():
                    g = max(dq_in.abs().max().item(), dq_hidden.abs().max().item(),
                            dq_out.abs().max().item())
                    tot = (q_in.sum(1) + q_hidden.sum(1) + q_out.sum(1)).mean().item()
                    gaps.append(g)
                    totals.append(tot)
        if return_diag:
            return q_out, conv_pen, gaps, totals
        return q_out, conv_pen

    def forward(self, x, return_conv=False):
        field = self.encoder(x)
        q_out, conv_pen = self.settle(field)
        logits = self.readout(q_out)
        if return_conv:
            return logits, conv_pen
        return logits
