"""
Focused evaluation harness (v0).
Compares the Redox Network (full + ablations) against strong baselines on
Grinsztajn-style numerical-classification datasets, with multiple seeds.

Ablations test whether each ingredient earns its place (CHARTER rule 7):
  redox_no_conserve  : conservation OFF  -> is the conservation law doing anything?
  redox_no_washboard : washboard OFF     -> do discrete levels help?
  redox_binary       : 2 levels          -> is multi-level worth it vs binary prior art?

Usage:
  python eval.py                         # full: 3 datasets x 3 seeds x 60 epochs
  python eval.py --quick                 # tiny end-to-end validation
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import datasets as D
from model import RedoxNetwork
from model_v1 import RedoxNetworkV1
from model_v2 import RedoxNetworkV2
from model_v3 import RedoxNetworkV3
from model_v4 import RedoxNetworkV4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# name -> (model class, config, lambda_sparse). Each variant gets its own sparsity weight.
# v4 = hard straight-through global prune + staged training (auto in train_redox).
REDOX_VARIANTS = {
    "redox_v0_static": (RedoxNetwork,   dict(conserve=True, beta=0.05, n_levels=4), 0.0),
    "redox_v4_lam010": (RedoxNetworkV4, dict(conserve=True, beta=0.05, n_levels=4), 0.01),
    "redox_v4_lam020": (RedoxNetworkV4, dict(conserve=True, beta=0.05, n_levels=4), 0.02),
}


def train_redox(cls, Xtr, ytr, Xte, yte, n_classes, epochs, seed,
                bs=256, lr=0.01, lambda_conv=0.1, lambda_sparse=0.05, **cfg):
    torch.manual_seed(seed)
    model = cls(n_features=Xtr.shape[1], n_classes=n_classes, **cfg).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    n = len(Xtr_t)
    staged = isinstance(model, RedoxNetworkV4)   # v4: stage 1 global-only, stage 2 +conditional
    half = max(1, epochs // 2)
    for ep in range(epochs):
        if staged:
            model.use_conditional = ep >= half
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr_t[idx].to(DEVICE)
            yb = ytr_t[idx].to(DEVICE)
            opt.zero_grad()
            logits, conv = model(xb, return_conv=True)
            loss = lossfn(logits, yb) + lambda_conv * conv
            if getattr(model, "last_sparsity", None) is not None:
                loss = loss + lambda_sparse * model.last_sparsity
            if not torch.isfinite(loss):
                return float("nan"), float("nan")
            loss.backward()
            opt.step()

    model.eval()
    preds = []
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(Xte_t), 1024):
            preds.append(model(Xte_t[i:i + 1024].to(DEVICE)).argmax(1).cpu())
    acc = float((torch.cat(preds).numpy() == yte).mean())

    with torch.no_grad():  # settling ratio: final flow / first flow (small = settled)
        xb = Xte_t[:256].to(DEVICE)
        if isinstance(model, RedoxNetworkV1):
            field = model.encoder(xb)
            _, _, gaps, _ = model.settle(xb, field, return_diag=True)
        else:
            field = model.make_field(xb) if hasattr(model, "make_field") else model.encoder(xb)
            _, _, gaps, _ = model.settle(field, return_diag=True)
    settle_ratio = float(gaps[-1] / (gaps[0] + 1e-9))
    return acc, settle_ratio


BASELINE_CACHE = "baseline_cache.json"


def baselines(Xtr, ytr, Xte, yte, cache_key=None):
    # Baselines are deterministic given (data, seed) -> cache and reuse across experiments.
    if cache_key:
        try:
            cache = json.load(open(BASELINE_CACHE)) if os.path.exists(BASELINE_CACHE) else {}
        except Exception:  # noqa
            cache = {}
        if cache_key in cache:
            return cache[cache_key]
    res = {}
    try:
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                          eval_metric="logloss", verbosity=0)
        m.fit(Xtr, ytr)
        res["xgboost"] = float((m.predict(Xte) == yte).mean())
    except Exception as e:  # noqa
        print("  [warn] xgboost:", repr(e)[:120])
    try:
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, verbose=-1)
        m.fit(Xtr, ytr)
        res["lightgbm"] = float((m.predict(Xte) == yte).mean())
    except Exception as e:  # noqa
        print("  [warn] lightgbm:", repr(e)[:120])
    try:
        from sklearn.neural_network import MLPClassifier
        m = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=300, early_stopping=True)
        m.fit(Xtr, ytr)
        res["mlp"] = float((m.predict(Xte) == yte).mean())
    except Exception as e:  # noqa
        print("  [warn] mlp:", repr(e)[:120])
    if cache_key:
        try:
            cache = json.load(open(BASELINE_CACHE)) if os.path.exists(BASELINE_CACHE) else {}
            cache[cache_key] = res
            json.dump(cache, open(BASELINE_CACHE, "w"), indent=2)
        except Exception:  # noqa
            pass
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["MagicTelescope", "electricity", "bank-marketing"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--subsample", type=int, default=10000)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.datasets = ["MagicTelescope"]
        args.seeds = 1
        args.epochs = 3
        args.subsample = 1500

    def plog(msg):
        print(msg, flush=True)
        with open("progress.txt", "a") as f:
            f.write(msg + "\n")

    open("progress.txt", "w").close()  # reset progress file
    plog(f"device={DEVICE} datasets={args.datasets} seeds={args.seeds} "
         f"epochs={args.epochs} subsample={args.subsample} bs={args.bs}")
    total = len(args.datasets) * args.seeds
    done = 0
    results = {}
    t0 = time.time()
    for ds in args.datasets:
        try:
            X, y, nc = D.load(ds)
        except Exception as e:  # noqa
            plog(f"skip {ds}: {repr(e)[:200]}")
            continue
        plog(f"\n=== {ds}: X={X.shape} classes={nc} ===")
        results[ds] = {}
        for seed in range(args.seeds):
            tseed = time.time()
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=y)
            if len(Xtr) > args.subsample:
                rng = np.random.RandomState(seed)
                idx = rng.choice(len(Xtr), args.subsample, replace=False)
                Xtr, ytr = Xtr[idx], ytr[idx]
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

            row = baselines(Xtr, ytr, Xte, yte,
                            cache_key=f"{ds}|clean|s{seed}|n{len(Xtr)}|d{Xtr.shape[1]}")
            plog(f"[{done}/{total}] {ds} seed{seed} | baselines: "
                 + ", ".join(f"{k}={v:.4f}" for k, v in row.items()))
            for name, (cls, cfg, lam) in REDOX_VARIANTS.items():
                acc, settle = train_redox(cls, Xtr, ytr, Xte, yte, nc, args.epochs, seed,
                                          bs=args.bs, lambda_sparse=lam, **cfg)
                row[name] = acc
                row[name + "__settle"] = settle
                plog(f"      {name:20s} {acc:.4f}  (settle {settle:.2f})")
            results[ds][f"seed{seed}"] = row
            done += 1
            json.dump(results, open(args.out, "w"), indent=2)
            plog(f"[{done}/{total}] {ds} seed{seed} DONE in {time.time()-tseed:.0f}s "
                 f"(elapsed {time.time()-t0:.0f}s)")

    # ---- aggregate: mean +/- std across seeds ----
    print("\n========== SUMMARY (mean +/- std test accuracy) ==========")
    for ds, seeds in results.items():
        rows = list(seeds.values())
        keys = [k for k in rows[0] if not k.endswith("__settle")]
        print(f"\n{ds}:")
        stats = {}
        for k in keys:
            vals = np.array([r[k] for r in rows if k in r], dtype=float)
            stats[k] = (np.nanmean(vals), np.nanstd(vals))
        for k in sorted(stats, key=lambda z: -stats[z][0]):
            m, s = stats[k]
            tag = "  <-- redox" if k.startswith("redox") else ""
            print(f"  {k:20s} {m:.4f} +/- {s:.4f}{tag}")
        # settling health per redox variant
        for name in REDOX_VARIANTS:
            key = name + "__settle"
            vals = [r[key] for r in rows if key in r]
            if vals:
                print(f"  [{name} settle-ratio {np.nanmean(vals):.3f} (small=settled)]")
    print(f"\ntotal time {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
