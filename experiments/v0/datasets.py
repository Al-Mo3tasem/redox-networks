"""
Dataset loader for the focused eval — Grinsztajn-style numerical classification
datasets (the regime where trees beat nets), fetched from OpenML and cached locally.
"""
import os
import numpy as np
from sklearn.datasets import fetch_openml

CACHE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

# Candidate numerical-classification datasets (trees-win regime).
CANDIDATES = {
    "MagicTelescope": dict(name="MagicTelescope", version=1),
    "electricity":    dict(name="electricity", version=1),
    "bank-marketing": dict(name="bank-marketing", version=1),
    "eye_movements":  dict(name="eye_movements", version=1),
    "phoneme":        dict(name="phoneme", version=1),
}


def load(key):
    """Return (X float32 [n,d], y int64 [n], n_classes). Numeric features only;
    NaNs imputed by column mean. Cached to data/<key>.npz."""
    os.makedirs(CACHE, exist_ok=True)
    cache_path = os.path.join(CACHE, key + ".npz")
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        return d["X"], d["y"], int(d["n_classes"])

    spec = CANDIDATES[key]
    ds = fetch_openml(spec["name"], version=spec["version"], as_frame=True, parser="auto")
    Xdf = ds.data.select_dtypes(include=["number"]).copy()
    # impute NaNs with column means
    Xdf = Xdf.fillna(Xdf.mean(numeric_only=True))
    X = Xdf.to_numpy(dtype=np.float32)

    yser = ds.target
    classes = sorted(map(str, yser.unique()))
    ymap = {c: i for i, c in enumerate(classes)}
    y = yser.astype(str).map(ymap).to_numpy().astype(np.int64)
    n_classes = len(classes)

    np.savez(cache_path, X=X, y=y, n_classes=n_classes)
    return X, y, n_classes
