"""
Robustness-to-junk-features experiment (Grinsztajn challenge #1).
Add k useless random-noise features, see whose accuracy survives.
Hypothesis: conserved Redox (esp. v1 with dustbin opt-out) degrades less than an MLP.
Saves figures/robustness_<dataset>.png and robust_results.json.
"""
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import datasets as D
from eval import train_redox, baselines, REDOX_VARIANTS

JUNK = [0, 100]              # clean vs heavy-junk (focused lambda comparison)
DATASETS = ["MagicTelescope", "electricity"]
SEEDS = 2
EPOCHS = 40
SUB = 6000
BS = 512


def add_junk(X, k, seed):
    if k == 0:
        return X
    rng = np.random.RandomState(seed + 999)
    return np.concatenate([X, rng.randn(X.shape[0], k).astype(np.float32)], axis=1)


def plog(msg, f="robust_progress.txt"):
    print(msg, flush=True)
    with open(f, "a") as fh:
        fh.write(msg + "\n")


def main():
    open("robust_progress.txt", "w").close()
    results = {}
    for ds in DATASETS:
        X, y, nc = D.load(ds)
        plog(f"=== {ds} X={X.shape} classes={nc} ===")
        results[ds] = {}
        for k in JUNK:
            accs = {}
            for seed in range(SEEDS):
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.2, random_state=seed, stratify=y)
                if len(Xtr) > SUB:
                    rng = np.random.RandomState(seed)
                    idx = rng.choice(len(Xtr), SUB, replace=False)
                    Xtr, ytr = Xtr[idx], ytr[idx]
                Xtr_j, Xte_j = add_junk(Xtr, k, seed), add_junk(Xte, k, seed)
                sc = StandardScaler().fit(Xtr_j)
                Xtr_j, Xte_j = sc.transform(Xtr_j), sc.transform(Xte_j)

                row = baselines(Xtr_j, ytr, Xte_j, yte,
                                cache_key=f"{ds}|j{k}|s{seed}|n{len(Xtr_j)}|d{Xtr_j.shape[1]}")
                for name, (cls, cfg, lam) in REDOX_VARIANTS.items():
                    acc, _ = train_redox(cls, Xtr_j, ytr, Xte_j, yte, nc, EPOCHS, seed,
                                         bs=BS, lambda_sparse=lam, **cfg)
                    row[name] = acc
                for m, v in row.items():
                    accs.setdefault(m, []).append(v)
                plog(f"{ds} junk={k:3d} seed{seed}: "
                     + ", ".join(f"{m}={v:.4f}" for m, v in row.items()))
            results[ds][k] = {m: [float(np.mean(v)), float(np.std(v))]
                              for m, v in accs.items()}
            json.dump(results, open("robust_results.json", "w"), indent=2)

    # plot degradation curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs("figures", exist_ok=True)
        for ds, kd in results.items():
            ks = sorted(kd.keys())
            models = list(kd[ks[0]].keys())
            plt.figure(figsize=(8, 5))
            for m in models:
                ys = [kd[k][m][0] for k in ks]
                plt.plot(ks, ys, marker="o", linewidth=2, label=m)
            plt.xlabel("number of junk (noise) features added")
            plt.ylabel("test accuracy")
            plt.title(f"Robustness to useless features — {ds}")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(f"figures/robustness_{ds}.png", dpi=120, bbox_inches="tight")
        plog("saved figures/robustness_*.png")
    except Exception as e:  # noqa
        plog("plot failed: " + repr(e)[:200])
    plog("DONE")


if __name__ == "__main__":
    main()
