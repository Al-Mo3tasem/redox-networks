"""
Redox Network — v1: DYNAMIC donor->acceptor matching (user's parked idea).

v0 used fixed pipes (static learned W) between layers. v1 instead COMPUTES, per input
row, an optimal-transport routing of who-donates-to-whom — "neurons decide which giver
gives to which receiver by mathematics."

Why optimal transport (Sinkhorn) is the principled choice:
  - It moves a CONSERVED mass from sources to sinks -> matches our charge conservation.
  - It minimizes a transport cost -> maps to redox favorability (favorable couples
    transfer more).
  - It is the conservation-respecting generalization of attention (attention routes but
    conserves nothing; OT routes AND conserves).
  - A learnable "DUSTBIN" lets neurons OPT OUT of transferring -> inert/irrelevant
    neurons don't participate (preserves the "ignore junk" property).
  - Sinkhorn marginals are bounded -> the conservative flow is automatically stable
    (no bounded-W hack needed).

Everything else (conserved donor/acceptor flow, self-limiting greed, washboard discrete
multi-levels, settle-to-equilibrium, backprop training) is identical to v0, so a v0-vs-v1
comparison isolates the effect of the dynamic matching.
"""
import math
import torch
import torch.nn as nn

TWO_PI = 2.0 * math.pi


class RedoxNetworkV1(nn.Module):
    def __init__(self, n_features, n_in=16, n_hidden=32, n_out=8, n_classes=2,
                 charge_per_neuron=0.5, T=25, eta=0.1, beta=0.05,
                 n_levels=4, conf_coef=1.0, d_match=8, sink_iters=10):
        super().__init__()
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        self.N = n_in + n_hidden + n_out
        self.T, self.eta, self.beta = T, eta, beta
        self.n_levels, self.conf_coef = n_levels, conf_coef
        self.d = d_match
        self.sink_iters = sink_iters
        self.q0 = charge_per_neuron
        self.Q = charge_per_neuron * self.N

        # greed, encoder, readout (same as v0)
        self.chi_in = nn.Parameter(torch.zeros(n_in))
        self.chi_hidden = nn.Parameter(torch.zeros(n_hidden))
        self.chi_out = nn.Parameter(torch.zeros(n_out))
        self.encoder = nn.Linear(n_features, n_in)
        self.readout = nn.Linear(n_out, n_classes)

        # --- dynamic-matching params: base node embeddings + input-conditioned part ---
        d = d_match
        self.base_K1 = nn.Parameter(torch.randn(n_in, d) * 0.1)      # input keys (coupling 1)
        self.base_Q1 = nn.Parameter(torch.randn(n_hidden, d) * 0.1)  # hidden queries (c1)
        self.base_K2 = nn.Parameter(torch.randn(n_hidden, d) * 0.1)  # hidden keys (coupling 2)
        self.base_Q2 = nn.Parameter(torch.randn(n_out, d) * 0.1)     # output queries (c2)
        self.cond_K1 = nn.Linear(n_features, n_in * d)
        self.cond_Q1 = nn.Linear(n_features, n_hidden * d)
        self.cond_K2 = nn.Linear(n_features, n_hidden * d)
        self.cond_Q2 = nn.Linear(n_features, n_out * d)
        self.dustbin = nn.Parameter(torch.zeros(1))                  # opt-out score

    # ---- discrete-level restoring force (washboard + box), same as v0 ----
    def _restoring(self, q):
        g = torch.zeros_like(q)
        if self.beta > 0:
            g = g + self.beta * TWO_PI * torch.sin(TWO_PI * q)
        if self.conf_coef > 0:
            g = g + self.conf_coef * (torch.relu(q - (self.n_levels - 1)) - torch.relu(-q))
        return g

    # ---- differentiable optimal transport (log-domain Sinkhorn) with dustbin ----
    def _sinkhorn(self, S):
        # S: (B, m, n) compatibility scores -> partial transport plan P (B, m, n),
        # each real row/col summing to <= 1 (rest absorbed by the dustbin).
        B, m, n = S.shape
        dev = S.device
        db = self.dustbin.view(1, 1, 1)
        Saug = torch.cat([S, db.expand(B, m, 1)], dim=2)             # (B, m, n+1)
        Saug = torch.cat([Saug, db.expand(B, 1, n + 1)], dim=1)      # (B, m+1, n+1)
        log_mu = torch.cat([torch.zeros(m, device=dev),
                            torch.full((1,), math.log(n), device=dev)])
        log_nu = torch.cat([torch.zeros(n, device=dev),
                            torch.full((1,), math.log(m), device=dev)])
        u = torch.zeros(B, m + 1, device=dev)
        v = torch.zeros(B, n + 1, device=dev)
        for _ in range(self.sink_iters):
            u = log_mu[None] - torch.logsumexp(Saug + v[:, None, :], dim=2)
            v = log_nu[None] - torch.logsumexp(Saug + u[:, :, None], dim=1)
        P = torch.exp(Saug + u[:, :, None] + v[:, None, :])
        return P[:, :m, :n]

    def _plans(self, x):
        B = x.shape[0]
        scale = 1.0 / math.sqrt(self.d)
        K1 = self.base_K1[None] + self.cond_K1(x).view(B, self.n_in, self.d)
        Q1 = self.base_Q1[None] + self.cond_Q1(x).view(B, self.n_hidden, self.d)
        P1 = self._sinkhorn(torch.bmm(K1, Q1.transpose(1, 2)) * scale)  # (B,n_in,n_hidden)
        K2 = self.base_K2[None] + self.cond_K2(x).view(B, self.n_hidden, self.d)
        Q2 = self.base_Q2[None] + self.cond_Q2(x).view(B, self.n_out, self.d)
        P2 = self._sinkhorn(torch.bmm(K2, Q2.transpose(1, 2)) * scale)  # (B,n_hidden,n_out)
        return P1, P2

    def settle(self, x, field, return_diag=False):
        B = field.shape[0]
        dev = field.device
        P1, P2 = self._plans(x)                       # per-sample dynamic routing
        q_in = torch.full((B, self.n_in), self.q0, device=dev)
        q_hidden = torch.full((B, self.n_hidden), self.q0, device=dev)
        q_out = torch.full((B, self.n_out), self.q0, device=dev)

        gaps, totals = [], []
        conv_pen = field.new_zeros(())
        for _ in range(self.T):
            # coupling enters mu via the per-sample plan (einsum over the bipartite edges)
            c_in = torch.einsum("bij,bj->bi", P1, q_hidden)
            c_hidden = (torch.einsum("bij,bi->bj", P1, q_in)
                        + torch.einsum("bjk,bk->bj", P2, q_out))
            c_out = torch.einsum("bjk,bj->bk", P2, q_hidden)
            mu_in = q_in - self.chi_in - field - c_in + self._restoring(q_in)
            mu_hidden = q_hidden - self.chi_hidden - c_hidden + self._restoring(q_hidden)
            mu_out = q_out - self.chi_out - c_out + self._restoring(q_out)

            flow1 = self.eta * P1 * (mu_in.unsqueeze(2) - mu_hidden.unsqueeze(1))
            flow2 = self.eta * P2 * (mu_hidden.unsqueeze(2) - mu_out.unsqueeze(1))
            dq_in = -flow1.sum(dim=2)
            dq_hidden = flow1.sum(dim=1) - flow2.sum(dim=2)
            dq_out = flow2.sum(dim=1)
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
        q_out, conv_pen = self.settle(x, field)
        logits = self.readout(q_out)
        if return_conv:
            return logits, conv_pen
        return logits
