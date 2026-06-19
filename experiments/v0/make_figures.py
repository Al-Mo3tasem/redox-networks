"""Generate illustration figures: (1) the settling process, (2) results comparison."""
import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import RedoxNetwork

os.makedirs("figures", exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)

# ---------- Figure 1: the settling process ("charge finding its level") ----------
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

m = RedoxNetwork(n_features=X.shape[1], n_in=10, n_hidden=12, n_out=4, n_classes=2)
opt = torch.optim.Adam(m.parameters(), lr=0.01)
lf = torch.nn.CrossEntropyLoss()
Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
ytr_t = torch.tensor(ytr, dtype=torch.long)
for _ in range(80):
    opt.zero_grad()
    logit, conv = m(Xtr_t, return_conv=True)
    (lf(logit, ytr_t) + 0.1 * conv).backward()
    opt.step()

xb = torch.tensor(Xte[:1], dtype=torch.float32)
with torch.no_grad():
    field = m.encoder(xb)
    W1b, W2b = m._bounded_W()
    qi = torch.full((1, m.n_in), m.q0)
    qh = torch.full((1, m.n_hidden), m.q0)
    qo = torch.full((1, m.n_out), m.q0)
    traj = [torch.cat([qi, qh, qo], 1).numpy()[0].copy()]
    for _ in range(m.T):
        mu_i = qi - m.chi_in - field - qh @ W1b.t() + m._restoring(qi)
        mu_h = qh - m.chi_hidden - qi @ W1b - qo @ W2b.t() + m._restoring(qh)
        mu_o = qo - m.chi_out - qh @ W2b + m._restoring(qo)
        f1 = m.eta * W1b.unsqueeze(0) * (mu_i.unsqueeze(2) - mu_h.unsqueeze(1))
        f2 = m.eta * W2b.unsqueeze(0) * (mu_h.unsqueeze(2) - mu_o.unsqueeze(1))
        qi = qi - f1.sum(2)
        qh = qh + f1.sum(1) - f2.sum(2)
        qo = qo + f2.sum(1)
        traj.append(torch.cat([qi, qh, qo], 1).numpy()[0].copy())
traj = np.array(traj)

plt.figure(figsize=(8, 5))
for n in range(traj.shape[1]):
    plt.plot(range(traj.shape[0]), traj[:, n], alpha=0.75)
for lvl in range(4):
    plt.axhline(lvl, ls="--", c="gray", lw=0.7)
plt.xlabel("settling step (time) -->")
plt.ylabel("charge held by each neuron")
plt.title("Charge 'finding its level': each line = one neuron settling to equilibrium")
plt.text(1, 3.05, "dashed lines = preferred discrete levels (0,1,2,3)", fontsize=8, color="gray")
plt.savefig("figures/settling.png", dpi=120, bbox_inches="tight")
print(f"settling.png saved | total charge start={traj[0].sum():.3f} end={traj[-1].sum():.3f}")

# ---------- Figure 2: results comparison ----------
if os.path.exists("results_v1.json"):
    res = json.load(open("results_v1.json"))
    datasets = list(res.keys())
    models = ["xgboost", "lightgbm", "mlp", "redox_v0_static", "redox_v1_dynamic"]
    means = {ds: {} for ds in datasets}
    for ds in datasets:
        rows = list(res[ds].values())
        for mn in models:
            vals = [r[mn] for r in rows if mn in r]
            means[ds][mn] = float(np.mean(vals)) if vals else np.nan
    x = np.arange(len(datasets))
    w = 0.16
    plt.figure(figsize=(9, 5))
    for i, mn in enumerate(models):
        ys = [means[ds][mn] for ds in datasets]
        plt.bar(x + (i - 2) * w, ys, w, label=mn)
    plt.xticks(x, datasets)
    plt.ylim(0.75, 0.92)
    plt.ylabel("test accuracy")
    plt.title("Results: our Redox network vs standard baselines")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.3)
    plt.savefig("figures/results.png", dpi=120, bbox_inches="tight")
    print("results.png saved")
