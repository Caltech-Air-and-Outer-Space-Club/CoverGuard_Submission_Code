"""Phase 0 diagnostics for the CloudScout gate. Reads data/weights only; tunes nothing.

  python diagnose.py predict   # score current weights on cloud_tiles_old (original 80 scenes) and cloud_tiles (221) -> diagnostics/preds_*.npz
  python diagnose.py report    # tables from the cached predictions -> diagnostics/phase0_report.json (+ printed summary)
  python diagnose.py overfit   # train a fresh net on 256 tiles; it must approach 100% training accuracy

All paths are anchored to this file. Progress is printed per file / per epoch.
"""
import csv, glob, json, os, re, subprocess, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn

HERE = Path(__file__).resolve().parent
DIAG = HERE / "diagnostics"; DIAG.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
from onboard_cloudscout import CloudScoutNet

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
WEIGHTS = HERE / "trained_model" / "cloudscout.pt"
SETS = {"old": HERE / "cloud_tiles_old", "new": HERE / "cloud_tiles"}
THRESH = 0.70


def nodata_frac(X):
    return (X.max(1) == 0).mean((1, 2)).astype(np.float32)        # pixels where all 5 bands are 0 (nodata is written as 0)


def predict():
    net = CloudScoutNet(); net.load_state_dict(torch.load(WEIGHTS, map_location="cpu")); net.eval()
    for tag, folder in SETS.items():
        files = sorted(glob.glob(str(folder / "*.npz"))); t0 = time.time()
        P, F, N, FI, TI, LOSS = [], [], [], [], [], []
        for k, f in enumerate(files, 1):
            z = np.load(f); X = z["X"]
            with torch.no_grad():
                for i in range(0, len(X), 32):
                    lg = net(torch.from_numpy(X[i:i + 32]).float())
                    P.append(torch.softmax(lg, 1)[:, 1].numpy())
                    y = torch.from_numpy((z["frac"][i:i + 32] >= THRESH).astype(np.int64))
                    LOSS.append(nn.functional.cross_entropy(lg, y, reduction="none").numpy())
            F.append(z["frac"]); N.append(nodata_frac(X)); FI += [k - 1] * len(z["frac"]); TI += list(range(len(z["frac"])))
            if k % 10 == 0 or k == len(files):
                el = time.time() - t0
                print(f"[{tag}] {k}/{len(files)} files | {el/60:.1f} min | ETA {el/k*(len(files)-k)/60:.1f} min", flush=True)
        np.savez(DIAG / f"preds_{tag}.npz", p=np.concatenate(P), frac=np.concatenate(F), nodata=np.concatenate(N),
                 loss=np.concatenate(LOSS), file_idx=np.array(FI), files=np.array([Path(f).name for f in files]))
    print("predict done")


def metrics(p, y, t=0.5):
    pred = p >= t
    tp, fp, fn, tn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum(), (~pred & ~y).sum()
    return dict(n=int(len(y)), cloudy=float(y.mean()), baseline=float(max(y.mean(), 1 - y.mean())),
                acc=float((tp + tn) / len(y)), discard=float(fp / max(fp + tn, 1)), miss=float(fn / max(fn + tp, 1)))


def auc(p, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None


def granule_table():
    """gNNNN.npz <-> sorted HLS granule index NNNN (this is how build_tiles.py numbered scenes)."""
    gr = sorted({re.sub(r"\.B02\.tif$", "", p) for p in glob.glob(str(HERE / "hls_raw" / "*.B02.tif"))})
    rmap = {}
    idx = HERE / "hls_raw" / "scene_index.csv"
    if idx.exists():
        for row in csv.reader(open(idx)):
            if len(row) >= 2: rmap[row[0]] = row[1]
    out = {}
    for i, g in enumerate(gr):
        name = os.path.basename(g)
        m = re.match(r"HLS\.L30\.(T\w+)\.(\d{4})(\d{3})T", name)
        out[i] = dict(granule=name, mgrs=m.group(1) if m else "?", year=int(m.group(2)) if m else 0,
                      doy=int(m.group(3)) if m else 0, region=rmap.get(name, "central_valley_ca (default: not in scene_index)"))
    return out


def report():
    R = {"git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
         "weights_size_bytes": WEIGHTS.stat().st_size}
    gt = granule_table()
    old, new = np.load(DIAG / "preds_old.npz"), np.load(DIAG / "preds_new.npz")

    def split(d, cut_frac=0.8):
        nf = len(d["files"]); cut = max(1, int(cut_frac * nf))
        return d["file_idx"] < cut, nf, cut

    # --- Provenance of the weights: trained Oct 13-14 on cloud_tiles_old; check val reproduces the earlier 0.984
    tr, nf, cut = split(old); y = old["frac"] >= THRESH
    R["old_set"] = {"files": nf, "train_files": cut, "val_files": nf - cut,
                    "train": metrics(old["p"][tr], y[tr]) | {"loss": float(old["loss"][tr].mean()), "auc": auc(old["p"][tr], y[tr])},
                    "val": metrics(old["p"][~tr], y[~tr]) | {"loss": float(old["loss"][~tr].mean()), "auc": auc(old["p"][~tr], y[~tr])},
                    "train_nodata_mean": float(old["nodata"][tr].mean()), "val_nodata_mean": float(old["nodata"][~tr].mean())}

    # --- New 221-scene set: same split rule as evaluate_gate.py, plus per-region view
    tr2, nf2, cut2 = split(new); y2 = new["frac"] >= THRESH
    R["new_set"] = {"files": nf2, "first80": metrics(new["p"][tr2], y2[tr2]) | {"auc": auc(new["p"][tr2], y2[tr2])},
                    "last20": metrics(new["p"][~tr2], y2[~tr2]) | {"auc": auc(new["p"][~tr2], y2[~tr2])}}
    regs, per_region = {}, {}
    for fi, name in enumerate(new["files"]):
        gi = int(re.search(r"g(\d{4})", str(name)).group(1)); regs[fi] = gt.get(gi, {"region": "unmapped", "mgrs": "?", "year": 0, "doy": 0})
    region_of_tile = np.array([regs[i]["region"] for i in new["file_idx"]])
    for r in sorted(set(region_of_tile)):
        k = region_of_tile == r
        per_region[r] = metrics(new["p"][k], y2[k]) | {"auc": auc(new["p"][k], y2[k]), "files": int(len({i for i in new["file_idx"][k]})),
                                                         "in_last20_files": int(len({i for i in new["file_idx"][k & ~tr2]}))}
    R["new_set_per_region"] = per_region
    R["new_set_last20_regions"] = {r: int(len({i for i in new["file_idx"][(region_of_tile == r) & ~tr2]}))
                                   for r in sorted(set(region_of_tile[~tr2]))}
    R["new_set_first80_regions"] = {r: int(len({i for i in new["file_idx"][(region_of_tile == r) & tr2]}))
                                    for r in sorted(set(region_of_tile[tr2]))}
    yrs = lambda m: sorted({regs[i]["year"] for i in set(new["file_idx"][m])})
    R["new_set_years"] = {"first80": yrs(tr2), "last20": yrs(~tr2)}
    R["new_set_mgrs_overlap_first80_last20"] = len({regs[i]["mgrs"] for i in set(new["file_idx"][tr2])} &
                                                   {regs[i]["mgrs"] for i in set(new["file_idx"][~tr2])})

    # --- Nodata cue: all-zero-pixel share in cloudy vs clear tiles
    for tag, d, yy in (("old", old, y), ("new", new, y2)):
        R[f"nodata_by_label_{tag}"] = {"cloudy_mean": float(d["nodata"][yy].mean()), "clear_mean": float(d["nodata"][~yy].mean()),
                                       "cloudy_share_tiles_with_nodata_gt_1pct": float((d["nodata"][yy] > 0.01).mean()),
                                       "clear_share_tiles_with_nodata_gt_1pct": float((d["nodata"][~yy] > 0.01).mean())}
    json.dump(R, open(DIAG / "phase0_report.json", "w"), indent=2)
    print(json.dumps(R, indent=2))


def overfit(n=256, epochs=40):
    """Capacity check: 256 tiles (128 cloudy / 128 clear from the OLD set), no augmentation, same optimizer/loss as train_cloudscout.py."""
    torch.manual_seed(0); rng = np.random.default_rng(0)
    files = sorted(glob.glob(str(SETS["old"] / "*.npz")))[:64]
    pool_c, pool_k = [], []
    for f in files:
        z = np.load(f); y = z["frac"] >= THRESH
        for lab, pool in ((1, pool_c), (0, pool_k)):
            idx = np.where(y == lab)[0]
            if len(idx): pool.append(z["X"][rng.choice(idx, min(4, len(idx)), replace=False)])
    Xc, Xk = np.concatenate(pool_c)[:n // 2], np.concatenate(pool_k)[:n // 2]
    X = torch.from_numpy(np.concatenate([Xc, Xk])); y = torch.cat([torch.ones(len(Xc)), torch.zeros(len(Xk))]).long()
    print(f"overfit set: {len(y)} tiles, {y.float().mean():.0%} cloudy", flush=True)
    net = CloudScoutNet(); opt = torch.optim.Adam(net.parameters(), 1e-3); lf = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.0]))
    log, t0 = [], time.time()
    for ep in range(1, epochs + 1):
        net.train(); perm = torch.randperm(len(y)); tot = 0.0
        for i in range(0, len(perm), 16):
            b = perm[i:i + 16]; loss = lf(net(X[b].float()), y[b]); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        net.eval()
        with torch.no_grad():
            pr = torch.cat([net(X[i:i + 32].float()).argmax(1) for i in range(0, len(y), 32)])
        acc = (pr == y).float().mean().item(); log.append(dict(epoch=ep, loss=tot / len(y), train_acc=acc, lr=opt.param_groups[0]["lr"]))
        print(f"epoch {ep}/{epochs} loss {tot/len(y):.4f} train-acc(eval mode) {acc:.3f} | {time.time()-t0:.0f}s | "
              f"ETA {(time.time()-t0)/ep*(epochs-ep)/60:.1f} min", flush=True)
    json.dump(log, open(DIAG / "overfit_log.json", "w"), indent=1)


if __name__ == "__main__":
    {"predict": predict, "report": report, "overfit": overfit}[sys.argv[1]]()
