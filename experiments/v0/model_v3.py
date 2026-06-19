"""
Redox Network — v3: CONDITIONAL reactive intake (context-dependent feature gates).

v2 had ONE fixed gate per feature (same for every row) -> too crude: it can't express
"income matters only when age is in 20-30", and it dims useful features globally (hurt
clean accuracy). v3 fixes this: a small gate-network reads the WHOLE row and outputs a
fresh Hard-Concrete gate PER FEATURE, PER SAMPLE. So a feature can be reactive for one
row and inert for another -- the redox idea that reactivity depends on context (what else
is in the beaker). This is conditional computation; it's how trees do "if A then use B".

Everything downstream (conserved donor/acceptor flow, self-limiting greed, washboard
levels, settle) is identical to v0/v2.
"""
import math
import torch
import torch.nn as nn

TWO_PI = 2.0 * math.pi


class RedoxNetworkV3(nn.Module):
    def __init__(self, n_features, n_in=16, n_hidden=32, n_out=8, n_classes=2,
                 charge_per_neuron=0.5, T=30, eta=0.1, beta=0.05,
                 coupling_budget=0.4, conserve=True, n_levels=4, conf_coef=1.0,
                 gate_hidden=64):
        super().__init__()
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        self.N = n_in + n_hidden + n_out
        self.T, self.eta, self.beta = T, eta, beta
        self.coupling_budget = coupling_budget
        self.conserve = conserve
        self.n_levels, self.conf_coef = n_levels, conf_coef
        self.q0 = charge_per_neuron
        self.Q = charge_per_neuron * self.N

        self.chi_in = nn.Parameter(torch.zeros(n_in))
        self.chi_hidden = nn.Parameter(torch.zeros(n_hidden))
        self.chi_out = nn.Parameter(torch.zeros(n_out))
        self.W1 = nn.Parameter(torch.randn(n_in, n_hidden) * 0.1)
        self.W2 = nn.Parameter(torch.randn(n_hidden, n_out) * 0.1)

        # --- CONDITIONAL INTAKE: gate-network reads whole row -> per-feature gate ---
        self.beta_hc, self.gamma_hc, self.zeta_hc = 2.0 / 3.0, -0.1, 1.1
        self.gate_net = nn.Sequential(
            nn.Linear(n_features, gate_hidden), nn.ReLU(),
            nn.Linear(gate_hidden, n_features))
        nn.init.zeros_(self.gate_net[-1].weight)        # start: gates ~ independent of x
        nn.init.constant_(self.gate_net[-1].bias, 2.0)  # start mostly OPEN, then learn

        self.encoder = nn.Linear(n_features, n_in)
        self.readout = nn.Linear(n_out, n_classes)
        self.last_sparsity = None

    def _hard_concrete(self, la):
        if self.training:
            u = torch.rand_like(la).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + la) / self.beta_hc)
        else:
            s = torch.sigmoid(la)
        return (s * (self.zeta_hc - self.gamma_hc) + self.gamma_hc).clamp(0.0, 1.0)

    def make_field(self, x):
        la = self.gate_net(x)                       # (B, n_features) per-sample gate logits
        g = self._hard_concrete(la)                 # (B, n_features) per-sample gates
        c = self.beta_hc * math.log(-self.gamma_hc / self.zeta_hc)
        self.last_sparsity = torch.sigmoid(la - c).mean()
        return self.encoder(g * x)

    @torch.no_grad()
    def gate_values(self, x):                       # deterministic gates, for inspection
        self.eval()
        return self._hard_concrete(self.gate_net(x))

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
