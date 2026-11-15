"""Steps 2, 3 (percentiles/shift/sharpness/hist), 4, 8 of the compatibility audit. Read-only on all tiles/weights.
   python compat_eval/analyze.py            (run from anywhere)"""
import glob, json, os, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "compat_eval"
sys.path.insert(0, str(ROOT))
import torch
from onboard_cloudscout import CloudScoutNet

SRC = {"hls": ROOT / "cloud_tiles", "l8biome": ROOT / "cloud_tiles_l8biome", "cs12": ROOT / "cloud_tiles_cs12"}
SRC = {k: v for k, v in SRC.items() if v.exists() and list(v.glob("*.npz"))}
BANDS = ["blue", "green", "red", "nir", "swir1"]
rng = np.random.default_rng(0)
MAX_TILES_FILE = 40
summary = {}


def load(folder, k=MAX_TILES_FILE):
    X, F, ID, ND = [], [], [], []
    for p in sorted(folder.glob("*.npz")):
        z = np.load(p)
        n = len(z["frac"]); idx = np.sort(rng.choice(n, min(k, n), replace=False))
        X.append(z["X"][idx]); F.append(z["frac"][idx]); ID += [(p.name, int(i)) for i in idx]
        ND.append(z["nodata_frac"][idx] if "nodata_frac" in z.files else np.full(len(idx), np.nan, np.float32))
    return np.concatenate(X), np.concatenate(F), ID, np.concatenate(ND)


data = {}
for name, folder in SRC.items():
    print(f"loading {name} ...", flush=True)
    data[name] = load(folder)
    print(f"  {name}: {len(data[name][1])} tiles sampled", flush=True)

# ---- Step 2: band semantics -------------------------------------------------------------
rows2 = {}
for name, (X, F, _, _) in data.items():
    clear = X[F < 0.05]
    sub = clear[:, :, ::4, ::4].astype(np.float32)
    nd = (sub[:, 2] + sub[:, 3]) > 0
    ndvi = ((sub[:, 3] - sub[:, 2]) / np.where(nd, sub[:, 3] + sub[:, 2], 1))[nd]
    allx = X[:, :, ::4, ::4].astype(np.float32)
    nz = (allx > 0).any(1)
    d = dict(n_clear=int(len(clear)), ndvi_median=float(np.median(ndvi)) if ndvi.size else None,
             ndvi_share_positive=float((ndvi > 0).mean()) if ndvi.size else None, bands={})
    for i, b in enumerate(BANDS):
        v = allx[:, i][nz]
        vc = sub[:, i]
        d["bands"][b] = dict(min=float(v.min()), median=float(np.median(v)), p99=float(np.percentile(v, 99)),
                             share_eq_1=float((v >= 0.9999).mean()), clear_median=float(np.median(vc)))
    # SWIR vs visible on clear tiles (snow signature: SWIR << visible)
    d["swir_over_green_clear_median"] = float(np.median(sub[:, 4][nd] / np.maximum(sub[:, 1][nd], 1e-4)))
    rows2[name] = d
summary["step2_band_semantics"] = rows2

# ---- Step 3: clear-tile percentiles, shift/IQR, sharpness, histograms ----------------------------
def pct_table(X, F):
    x = X[F < 0.05][:, :, ::4, ::4].astype(np.float32)
    valid = (x > 0).any(1)
    return {b: [float(np.percentile(x[:, i][valid], q)) for q in (10, 50, 90)] for i, b in enumerate(BANDS)}, \
           {b: x[:, i][valid][::37] for i, b in enumerate(BANDS)}

pt, samples = {}, {}
for name, (X, F, _, _) in data.items():
    pt[name], samples[name] = pct_table(X, F)
summary["step3_clear_percentiles_p10_p50_p90"] = pt
shift = {}
if "hls" in pt:
    for name in pt:
        if name == "hls": continue
        shift[name] = {}
        for b in BANDS:
            pooled = np.concatenate([samples["hls"][b], samples[name][b]])
            iqr = np.percentile(pooled, 75) - np.percentile(pooled, 25)
            shift[name][b] = float((pt[name][b][1] - pt["hls"][b][1]) / max(iqr, 1e-6))
summary["step3_median_shift_over_pooled_IQR"] = shift

def lap_abs(tile):  # mean abs Laplacian of the red band, full res
    r = tile[2].astype(np.float32)
    lap = -4 * r[1:-1, 1:-1] + r[:-2, 1:-1] + r[2:, 1:-1] + r[1:-1, :-2] + r[1:-1, 2:]
    return float(np.abs(lap).mean())

sharp = {}
for name, (X, F, _, _) in data.items():
    idx = np.where(F < 0.05)[0]; idx = idx[:: max(1, len(idx) // 150)][:150]
    ratios = [lap_abs(X[i]) / max(float(X[i][2].astype(np.float32).mean()), 1e-4) for i in idx]   # normalised by brightness
    sharp[name] = dict(mean_abs_laplacian_over_mean_red=float(np.mean(ratios)), n=len(idx))
summary["step3_sharpness"] = sharp

fig, axs = plt.subplots(1, 5, figsize=(18, 3.2))
for i, b in enumerate(BANDS):
    for name in samples:
        axs[i].hist(samples[name][b], bins=60, range=(0, 1), density=True, histtype="step", label=name)
    axs[i].set_title(b); axs[i].set_yscale("log")
axs[0].legend(); plt.tight_layout(); plt.savefig(OUT / "band_histograms_clear_tiles.png", dpi=110); plt.close()

# ---- Step 4: label definitions ------------------------------------------------------------------
s4 = {}
for name, (X, F, ID, ND) in data.items():
    # use ALL tiles' frac (cheap), not just the sample
    allf = np.concatenate([np.load(p)["frac"] for p in sorted(SRC[name].glob("*.npz"))])
    s4[name] = dict(n_tiles=int(len(allf)), cloudy_share_at_0p70=float((allf >= 0.70).mean()),
                    share_frac_lt_0p05=float((allf < 0.05).mean()), share_frac_gt_0p95=float((allf > 0.95).mean()),
                    share_30_90=float(((allf >= 0.3) & (allf < 0.9)).mean()))
    if name != "hls":
        sh = []
        for p in sorted(SRC[name].glob("*.npz")):
            z = np.load(p)
            if "mask" in z.files:
                sh.append((z["mask"] == 2).mean((1, 2)))
        if sh:
            sh = np.concatenate(sh); s4[name]["share_tiles_shadow_gt_5pct"] = float((sh > 0.05).mean())
    plt.hist(allf, bins=20, range=(0, 1), histtype="step", density=True, label=name)
plt.yscale("log"); plt.xlabel("cloud fraction per tile"); plt.legend(); plt.savefig(OUT / "frac_histograms.png", dpi=110); plt.close()
summary["step4_labels"] = s4

# ---- Step 5-lite for the contact sheets: model predictions on the sample ------------------------
net = CloudScoutNet(); net.load_state_dict(torch.load(ROOT / "trained_model/cloudscout.pt", map_location="cpu")); net.eval()
def predict(X):
    P = []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            P.append(torch.softmax(net(torch.from_numpy(X[i:i + 32]).float()), 1)[:, 1].numpy())
    return np.concatenate(P)

for name, (X, F, ID, ND) in data.items():
    sel = np.sort(rng.choice(len(F), min(500, len(F)), replace=False))
    p = predict(X[sel]); f = F[sel]
    rand = rng.choice(len(sel), min(12, len(sel)), replace=False)
    dis = np.argsort(-np.abs(p - (f >= 0.70).astype(float)))[:6]
    fig, axs = plt.subplots(3, 6, figsize=(16, 8.6))
    for ax, k in zip(axs.ravel(), list(rand) + list(dis)):
        t = X[sel[k]].astype(np.float32)
        ax.imshow(np.clip(t[[2, 1, 0]].transpose(1, 2, 0) / 0.3, 0, 1)); ax.axis("off")
        ax.set_title(f"{'D' if k in dis else 'R'} frac={f[k]:.2f} p={p[k]:.2f}", fontsize=8)
    fig.suptitle(f"{name}: rows 1-2 = 12 random tiles (R), last 6 = largest model/label disagreement (D)")
    plt.tight_layout(); plt.savefig(OUT / f"contact_{name}.png", dpi=90); plt.close()

json.dump(summary, open(OUT / "analysis_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
