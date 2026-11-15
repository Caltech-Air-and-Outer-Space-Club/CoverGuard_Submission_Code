"""Score the CURRENT CloudScout weights on every tile in a folder (no training, no split).
   python eval_folder.py --tiles cloud_tiles_l8biome --weights trained_model/cloudscout.pt [--out compat_eval/l8biome_metrics.json]
Reports: baseline accuracy, accuracy / clear-tile discard / cloud-miss at several thresholds, ROC AUC,
miss rate at a 2% clear-tile discard rate (the primary operating-point metric), and the by-cloud-fraction table."""
import argparse, glob, json, os, sys
import numpy as np, torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onboard_cloudscout import CloudScoutNet            # the repo's model definition

ap = argparse.ArgumentParser()
ap.add_argument("--tiles", required=True); ap.add_argument("--weights", required=True)
ap.add_argument("--cutoff", type=float, default=0.70); ap.add_argument("--out", default=None)
ap.add_argument("--max-files", type=int, default=0)
a = ap.parse_args()

net = CloudScoutNet(); net.load_state_dict(torch.load(a.weights, map_location="cpu")); net.eval()
files = sorted(glob.glob(os.path.join(a.tiles, "*.npz")))
files = files[:a.max_files] if a.max_files else files
P, F, S = [], [], []
for f in files:
    z = np.load(f); X = torch.from_numpy(z["X"])
    with torch.no_grad():
        for i in range(0, len(X), 32):
            P.append(torch.softmax(net(X[i:i + 32].float()), 1)[:, 1].numpy())
    F.append(z["frac"]); S += [os.path.basename(f)] * len(z["frac"])
p, frac = np.concatenate(P), np.concatenate(F); y = frac >= a.cutoff
res = {"folder": a.tiles, "files": len(files), "tiles": int(len(y)), "cloudy_share": float(y.mean()),
       "baseline_acc": float(max(y.mean(), 1 - y.mean()))}
def rates(t):
    pred = p >= t; tp, fp, fn, tn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum(), (~pred & ~y).sum()
    return dict(thr=t, acc=float((tp + tn) / len(y)), discard=float(fp / max(fp + tn, 1)), miss=float(fn / max(fn + tp, 1)))
res["sweep"] = [rates(t) for t in (0.3, 0.5, 0.7, 0.9)]
if 0 < y.sum() < len(y):
    res["roc_auc"] = float(roc_auc_score(y, p))
    thr = np.quantile(p[~y], 0.98)                       # cutoff giving ~2% clear-tile discard on this set
    res["miss_at_2pct_discard"] = float((p[y] < thr).mean())
bins = [0, .1, .3, .5, .7, .9, 1.0001]; tab = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (frac >= lo) & (frac < hi)
    if m.sum(): tab.append(dict(bin=f"{lo:.0%}-{min(hi, 1):.0%}", n=int(m.sum()), mean_p=float(p[m].mean()), pred_cloudy=float((p[m] >= 0.5).mean())))
res["by_cloud_fraction"] = tab
print(json.dumps(res, indent=2))
if a.out:
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"), indent=2)
