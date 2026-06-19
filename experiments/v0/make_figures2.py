"""Richer teaching figures: architecture diagram, washboard, and a SETTLING ANIMATION (gif)."""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import RedoxNetwork

os.makedirs("figures", exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)
BLUE, GREEN, ORANGE = "#4C9BE8", "#7BC47F", "#E8A33D"


# ---------- 1) architecture schematic ----------
def layer_y(n):
    return np.linspace(-(n - 1) / 2, (n - 1) / 2, n)


fig, ax = plt.subplots(figsize=(9, 5.5))
xin, xhid, xout = 0, 2, 4
yin, yhid, yout = layer_y(4), layer_y(5), layer_y(2)
for a in yin:
    for b in yhid:
        ax.plot([xin, xhid], [a, b], color="0.85", lw=1, zorder=1)
for a in yhid:
    for b in yout:
        ax.plot([xhid, xout], [a, b], color="0.85", lw=1, zorder=1)
ax.scatter([xin] * 4, yin, s=700, c=BLUE, zorder=3, edgecolors="k")
ax.scatter([xhid] * 5, yhid, s=700, c=GREEN, zorder=3, edgecolors="k")
ax.scatter([xout] * 2, yout, s=700, c=ORANGE, zorder=3, edgecolors="k")
for x, t in [(xin, "INPUT tanks"), (xhid, "HIDDEN tanks"), (xout, "OUTPUT tanks")]:
    ax.text(x, 3.1, t, ha="center", fontweight="bold")
ax.annotate("one row of data\ntilts these tanks", xy=(xin - 0.08, yin[-1]),
            xytext=(xin - 1.7, 1.6), arrowprops=dict(arrowstyle="->"),
            ha="center", fontsize=9)
ax.annotate("read the\nanswer here", xy=(xout + 0.08, 0), xytext=(xout + 1.5, 1.4),
            arrowprops=dict(arrowstyle="->"), ha="center", fontsize=9)
ax.text(2, -3.3, "gray lines = pipes; charge flows giver -> receiver along them\n"
        "TOTAL charge never changes  (our conservation law)",
        ha="center", fontsize=9, color="0.3")
ax.axis("off")
ax.set_ylim(-3.8, 3.6)
ax.set_title("The Redox Network = connected water tanks")
plt.savefig("figures/architecture.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------- 2) washboard floor ----------
q = np.linspace(-0.3, 3.3, 400)
V = 1 - np.cos(2 * np.pi * q)
plt.figure(figsize=(8, 4))
plt.plot(q, V, lw=2, color=BLUE)
plt.scatter(range(4), [0, 0, 0, 0], s=120, color="red", zorder=5)
plt.annotate("a ball (the charge) rolls\ninto a dip = a 'level'", xy=(1, 0.05),
             xytext=(1.6, 1.2), arrowprops=dict(arrowstyle="->"), fontsize=10)
plt.xlabel("charge a neuron holds")
plt.ylabel("'discomfort' (energy)")
plt.title("The bumpy floor: dips at 0,1,2,3 give sharp yes/no decisions")
plt.savefig("figures/washboard.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------- 3) settling animation ----------
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
m = RedoxNetwork(n_features=X.shape[1], n_in=6, n_hidden=8, n_out=4, n_classes=2)
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
N = traj.shape[1]
colors = [BLUE] * m.n_in + [GREEN] * m.n_hidden + [ORANGE] * m.n_out

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(range(N), traj[0], color=colors, edgecolor="k")
for lvl in range(4):
    ax.axhline(lvl, ls="--", c="gray", lw=0.7)
ax.set_ylim(-0.5, 3.5)
ax.set_xlabel("each bar = one neuron  (blue=input, green=hidden, orange=output)")
ax.set_ylabel("charge held")
title = ax.set_title("")


def update(f):
    for b, h in zip(bars, traj[f]):
        b.set_height(h)
    title.set_text(f"settling step {f}/{len(traj)-1}   |   "
                   f"total charge = {traj[f].sum():.1f}  (never changes!)")
    return list(bars) + [title]


ani = animation.FuncAnimation(fig, update, frames=len(traj), interval=250, blit=False)
try:
    ani.save("figures/settling_animation.gif", writer=animation.PillowWriter(fps=4))
    print("settling_animation.gif saved")
except Exception as e:  # noqa
    print("gif failed:", repr(e)[:150])
    idxs = np.linspace(0, len(traj) - 1, 6).astype(int)
    fig2, axs = plt.subplots(1, 6, figsize=(15, 3), sharey=True)
    for ax2, fi in zip(axs, idxs):
        ax2.bar(range(N), traj[fi], color=colors)
        for lvl in range(4):
            ax2.axhline(lvl, ls="--", c="gray", lw=0.5)
        ax2.set_title(f"step {fi}")
    plt.savefig("figures/settling_filmstrip.png", dpi=110, bbox_inches="tight")
    print("settling_filmstrip.png saved")
print("done")
