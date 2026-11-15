"""Statistical check: can a classifier tell two tile sources apart from per-tile spectral statistics (CLEAR tiles only)?
AUC near 0.5 = indistinguishable (safe to pool).  AUC near 1.0 = strong distribution shift.
   python domain_check.py cloud_tiles cloud_tiles_l8biome
Folds are grouped by scene file so correlated tiles never sit on both sides."""
import glob, os, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

def load(folder, max_per_file=60, seed=0):
    rng = np.random.default_rng(seed); F, G = [], []
    for gi, p in enumerate(sorted(glob.glob(os.path.join(folder, "*.npz")))):
        z = np.load(p); X, f = z["X"], z["frac"]
        idx = np.where(f < 0.05)[0]                                  # clear tiles only
        if len(idx) == 0: continue
        idx = rng.choice(idx, min(max_per_file, len(idx)), replace=False)
        x = X[idx].astype(np.float32); n = len(idx)
        flat = x.reshape(n, 5, -1)
        q = np.percentile(flat, [10, 50, 90], axis=2).transpose(1, 2, 0).reshape(n, -1)
        red, nir = flat[:, 2], flat[:, 3]
        ndvi = ((nir - red) / (nir + red + 1e-6)).mean(1, keepdims=True)
        F.append(np.hstack([flat.mean(2), flat.std(2), q, ndvi])); G += [f"{folder}:{gi}"] * n
    return np.vstack(F), np.array(G)

if __name__ == "__main__":
    a, b = sys.argv[1:3]
    Xa, Ga = load(a); Xb, Gb = load(b)
    X = np.vstack([Xa, Xb]); y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]; g = np.r_[Ga, Gb]
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=150), X, y, groups=g,
                          cv=GroupKFold(n_splits=5), method="predict_proba")[:, 1]
    print(f"clear tiles: {a}={len(Xa)}, {b}={len(Xb)}  domain-classifier AUC = {roc_auc_score(y, p):.3f}")
    names = [f"{s}_b{i}" for s in ("mean", "std", "p10", "p50", "p90") for i in range(5)]
    print("per-band median (a vs b):", np.round(np.median(Xa[:, :5], 0), 3), "vs", np.round(np.median(Xb[:, :5], 0), 3))
