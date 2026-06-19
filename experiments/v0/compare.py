"""
Main comparison for the paper: Redox Network vs trees AND other neural networks, on clean
data and with 100 junk features. Models: XGBoost, LightGBM, MLP, ResNet, FT-Transformer,
TabNet, and our Redox (hard-gated). Reports mean test accuracy over seeds.
"""
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import datasets as D
from robustness import add_junk
from eval import train_redox
from model_v4 import RedoxNetworkV4
from baselines_nn import TabResNet, FTTransformer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATASETS = ["MagicTelescope", "electricity"]
JUNK = [0, 100]
SEEDS = 2
SUB = 8000


def plog(m, f="compare_progress.txt"):
    print(m, flush=True)
    with open(f, "a") as fh:
        fh.write(m + "\n")


def train_torch(model, X, y, epochs=50, bs=512, lr=1e-3, wd=1e-5):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lf = nn.CrossEntropyLoss()
    Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.long)
    n = len(Xt)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(n).split(bs):
            opt.zero_grad()
            loss = lf(model(Xt[idx].to(DEV)), yt[idx].to(DEV))
            loss.backward(); opt.step()
    return model


def acc_torch(model, X, y):
    model.eval()
    with torch.no_grad():
        p = []
        Xt = torch.tensor(X, dtype=torch.float32)
        for i in range(0, len(Xt), 2048):
            p.append(model(Xt[i:i + 2048].to(DEV)).argmax(1).cpu())
    return float((torch.cat(p).numpy() == y).mean())


def fit_all(Xtr, ytr, Xte, yte, nc, seed):
    row = {}
    # ---- trees ----
    try:
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                          eval_metric="logloss", verbosity=0).fit(Xtr, ytr)
        row["xgboost"] = float((m.predict(Xte) == yte).mean())
    except Exception as e:  # noqa
        plog("  xgb fail " + repr(e)[:80])
    try:
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, verbose=-1).fit(Xtr, ytr)
        row["lightgbm"] = float((m.predict(Xte) == yte).mean())
    except Exception as e:  # noqa
        plog("  lgbm fail " + repr(e)[:80])
    # ---- simple NN ----
    try:
        from sklearn.neural_network import MLPClassifier
        m = MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=300, early_stopping=True).fit(Xtr, ytr)
        row["mlp"] = float((m.predict(Xte) == yte).mean())
    except Exception as e:  # noqa
        plog("  mlp fail " + repr(e)[:80])
    # ---- ResNet / FT-Transformer ----
    for name, ctor in [("resnet", lambda: TabResNet(Xtr.shape[1], nc)),
                       ("ft_transformer", lambda: FTTransformer(Xtr.shape[1], nc))]:
        try:
            torch.manual_seed(seed)
            row[name] = acc_torch(train_torch(ctor().to(DEV), Xtr, ytr), Xte, yte)
        except Exception as e:  # noqa
            plog(f"  {name} fail " + repr(e)[:80])
    # ---- TabNet ----
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        xa, xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.15, random_state=seed, stratify=ytr)
        clf = TabNetClassifier(verbose=0, seed=seed)
        clf.fit(xa, ya, eval_set=[(xv, yv)], max_epochs=80, patience=15,
                batch_size=1024, virtual_batch_size=128)
        row["tabnet"] = float((clf.predict(Xte) == yte).mean())
    except Exception as e:  # noqa
        plog("  tabnet fail " + repr(e)[:80])
    # ---- our Redox (hard-gated, staged) ----
    try:
        acc, _ = train_redox(RedoxNetworkV4, Xtr, ytr, Xte, yte, nc, epochs=40, seed=seed,
                             bs=512, lambda_sparse=0.01, conserve=True, beta=0.05, n_levels=4)
        row["redox_ours"] = acc
    except Exception as e:  # noqa
        plog("  redox fail " + repr(e)[:80])
    return row


def main():
    open("compare_progress.txt", "w").close()
    plog(f"device={DEV} datasets={DATASETS} junk={JUNK} seeds={SEEDS}")
    out = {}
    for ds in DATASETS:
        X, y, nc = D.load(ds)
        out[ds] = {}
        for k in JUNK:
            accs = {}
            for seed in range(SEEDS):
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
                if len(Xtr) > SUB:
                    rng = np.random.RandomState(seed); idx = rng.choice(len(Xtr), SUB, replace=False)
                    Xtr, ytr = Xtr[idx], ytr[idx]
                Xtr_j, Xte_j = add_junk(Xtr, k, seed), add_junk(Xte, k, seed)
                sc = StandardScaler().fit(Xtr_j)
                Xtr_j, Xte_j = sc.transform(Xtr_j), sc.transform(Xte_j)
                r = fit_all(Xtr_j, ytr, Xte_j, yte, nc, seed)
                for m, v in r.items():
                    accs.setdefault(m, []).append(v)
                plog(f"  {ds} junk={k} seed{seed}: " + ", ".join(f"{m}={v:.4f}" for m, v in r.items()))
            out[ds][k] = {m: [float(np.mean(v)), float(np.std(v))] for m, v in accs.items()}
            json.dump(out, open("compare_results.json", "w"), indent=2)
    plog("DONE")


if __name__ == "__main__":
    main()
