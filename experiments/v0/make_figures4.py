"""Teaching figures for the v4->v6 story: rotation invariance, and settling=smoothing."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
rng = np.random.RandomState(0)

# ---------- Figure A: rotation invariance ----------
X = rng.uniform(-1, 1, (400, 2))
y = (X[:, 0] > 0).astype(int)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
ax[0].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=12, edgecolors="none")
ax[0].axvline(0, color="k", lw=2.5)
ax[0].set_title("Original axes:\ntree splits ONCE ('feature1 > 0')")
ax[0].set_xlabel("feature 1"); ax[0].set_ylabel("feature 2")

th = np.pi / 4
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
Xr = X @ R.T
ax[1].scatter(Xr[:, 0], Xr[:, 1], c=y, cmap="coolwarm", s=12, edgecolors="none")
ax[1].plot([-1.4, 1.4], [1.4, -1.4], "k--", lw=1.5, label="true boundary (diagonal)")
gx = np.linspace(-1.4, 1.4, 10)
gy = -gx
ax[1].step(gx, gy, where="mid", color="green", lw=2.5, label="tree: staircase of MANY cuts")
ax[1].set_title("Rotated axes (columns scrambled):\ntree now needs many cuts")
ax[1].set_xlabel("mix of features"); ax[1].legend(fontsize=8)
fig.tight_layout()
plt.savefig("figures/rotation.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------- Figure B: settling = smoothing ----------
x = np.linspace(0, 1, 400)
step = (x >= 0.5).astype(float)
smoothed = 1 / (1 + np.exp(-(x - 0.5) * 6))
plt.figure(figsize=(7.5, 4.3))
plt.plot(x, step, "k", lw=2.5, label="what tabular needs / trees fit (sharp jump)")
plt.plot(x, smoothed, "r--", lw=2.5, label="what settling = diffusion produces (smeared)")
plt.fill_between(x, step, smoothed, color="red", alpha=0.12)
plt.xlabel("a feature (e.g. age)")
plt.ylabel("prediction")
plt.title("Settling to equilibrium is SMOOTHING: it rounds off the sharp edges tabular needs")
plt.legend()
plt.savefig("figures/smoothing.png", dpi=120, bbox_inches="tight")
plt.close()
print("saved figures/rotation.png and figures/smoothing.png")
