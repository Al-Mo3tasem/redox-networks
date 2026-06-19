"""Teaching figures for reactive intake: (1) the 'doorman', (2) the Goldilocks knob."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

os.makedirs("figures", exist_ok=True)
GREEN, RED, GRAY, WHITE, BLUE = "#7BC47F", "#E06C6C", "#AAAAAA", "#F2F2F2", "#4C9BE8"

# ---------- Figure 1: the doorman ----------
fig, ax = plt.subplots(figsize=(9, 5.5))
feats = [("age", True), ("income", True), ("test value", True),
         ("junk #1", False), ("junk #2", False), ("junk #3", False)]
ys = np.linspace(4, -4, len(feats))
for (name, real), yy in zip(feats, ys):
    ax.add_patch(Rectangle((-4.4, yy - 0.32), 1.9, 0.64,
                           color=GREEN if real else RED, ec="k"))
    ax.text(-3.45, yy, name, ha="center", va="center", fontsize=9)
    gx = -1.6
    if real:
        ax.add_patch(Rectangle((gx, yy - 0.2), 0.5, 0.4, color="white", ec=GREEN, lw=2))
        ax.text(gx + 0.25, yy + 0.5, "OPEN", ha="center", fontsize=7, color="green")
        ax.annotate("", xy=(1.0, 0), xytext=(gx + 0.55, yy),
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6))
    else:
        ax.add_patch(Rectangle((gx, yy - 0.2), 0.5, 0.4, color=GRAY, ec=RED, lw=2))
        ax.text(gx + 0.25, yy + 0.5, "SHUT", ha="center", fontsize=7, color="red")
        ax.plot(gx + 0.75, yy, "x", color=RED, ms=9, mew=2)
ax.add_patch(Circle((1.7, 0), 0.95, color=BLUE, ec="k"))
ax.text(1.7, 0, "the\ntanks", ha="center", va="center", color="white", fontweight="bold")
ax.text(-1.35, 4.8, "reactivity gate", ha="center", fontsize=9, style="italic")
ax.set_title("Reactive intake: a 'doorman' lets useful features in, blocks the junk")
ax.axis("off")
ax.set_xlim(-4.7, 3.0)
ax.set_ylim(-5, 5.3)
plt.savefig("figures/reactive_intake.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------- Figure 2: the Goldilocks knob ----------
# 6 features: first 3 real, last 3 junk. colors show what happens to each.
panels = [
    ("knob too LOW\n-> junk leaks in",      [GREEN, GREEN, GREEN, RED, RED, RED]),
    ("knob JUST RIGHT\n-> junk out, real kept", [GREEN, GREEN, GREEN, WHITE, WHITE, WHITE]),
    ("knob too HIGH\n-> good info lost too", [GRAY, GRAY, GRAY, WHITE, WHITE, WHITE]),
]
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, (title, colors) in zip(axs, panels):
    for i, c in enumerate(colors):
        ax.add_patch(Rectangle((i, 0), 0.9, 0.9, color=c, ec="k"))
        ax.text(i + 0.45, -0.35, "real" if i < 3 else "junk", ha="center", fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.3, 6.2)
    ax.set_ylim(-0.8, 1.2)
    ax.axis("off")
fig.suptitle("The sparsity knob: green=useful kept, white=junk blocked (good), "
             "red=junk leaked (bad), gray=useful lost (bad)", fontsize=10)
plt.savefig("figures/goldilocks.png", dpi=120, bbox_inches="tight")
plt.close()
print("saved figures/reactive_intake.png and figures/goldilocks.png")
