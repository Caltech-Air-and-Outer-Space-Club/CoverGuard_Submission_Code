"""Config-driven cloud-gate run on the 128 px cache with a region-grouped split.

  python train_v2.py --variant A --seed 0 [--epochs 18 --tiles-per-epoch 4000 --threads 4]

Split (whole regions, never mixed): TEST = amazon_basin, interior_alaska, iowa_cropland;
VAL (checkpoint + threshold selection only) = great_basin_nv, sahel_west_africa; TRAIN = all other regions.
Checkpoint/early stopping by validation ROC AUC (threshold-free). Operating threshold = val score giving 2% clear-tile
discard; the TEST set is scored once, at the end. Writes weights, config, metrics JSON and a loss-curve PNG to runs/<name>/.
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from sklearn.metrics import roc_auc_score, average_precision_score
from models_v2 import build, score

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache128"
TEST_R = ["amazon_basin", "interior_alaska", "iowa_cropland"]
VAL_R = ["great_basin_nv", "sahel_west_africa"]
CUT = 0.70
BINS = [0, .1, .3, .5, .7, .9, 1.0001]

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True, choices=["baseline", "A", "B", "AB", "C"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=18)
ap.add_argument("--patience", type=int, default=4)
ap.add_argument("--tiles-per-epoch", type=int, default=4000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--class-weights", type=float, nargs=2, default=[2.0, 1.0])
ap.add_argument("--balance-regions", action="store_true", help="sample training tiles with equal probability per region")
ap.add_argument("--name", default=None)
ap.add_argument("--val-regions", nargs="+", default=None, help="override VAL_R (regions removed from train)")
ap.add_argument("--weight-c-bce", action="store_true", help="variant C: apply --class-weights (clear, cloud) to the per-pixel BCE")
ap.add_argument("--suffix", default="", help="appended to run name; also the aggregate group label")
a = ap.parse_args()
if a.val_regions: VAL_R = a.val_regions
torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed)
tag = "bal" if a.balance_regions else "std"
name = a.name or f"{a.variant}{'_bal' if a.balance_regions else ''}{a.suffix}_s{a.seed}"
RUN = HERE / "runs" / name; RUN.mkdir(parents=True, exist_ok=True)

Xall = np.load(CACHE / "X.npy", mmap_mode="r"); frac = np.load(CACHE / "frac.npy"); mask = np.load(CACHE / "mask.npy", mmap_mode="r")
region = np.load(CACHE / "region.npy"); scene = np.load(CACHE / "scene.npy")
is_test, is_val = np.isin(region, TEST_R), np.isin(region, VAL_R)
tr_idx, va_idx, te_idx = [np.where(m)[0] for m in (~(is_test | is_val), is_val, is_test)]
print(f"[{name}] train {len(tr_idx)} tiles / val {len(va_idx)} / test {len(te_idx)}; cloudy@{CUT}: "
      f"{(frac[tr_idx] >= CUT).mean():.2f} / {(frac[va_idx] >= CUT).mean():.2f} / {(frac[te_idx] >= CUT).mean():.2f}", flush=True)
Xva, Xte = np.asarray(Xall[va_idx]), np.asarray(Xall[te_idx])
rng_tr = np.random.default_rng(a.seed + 1000)
sub_tr = np.sort(rng_tr.choice(tr_idx, 3000, replace=False)); Xtr_sub = np.asarray(Xall[sub_tr])

net = build(a.variant)
n_params = sum(p.numel() for p in net.parameters())
opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=a.wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
cw = torch.tensor(a.class_weights)


def augment(x, m=None):
    B = len(x)
    for i in range(B):
        k, fl = np.random.randint(4), np.random.rand() < 0.5
        x[i] = torch.rot90(x[i], k, (1, 2)); m_i = None
        if fl: x[i] = x[i].flip(2)
        if m is not None:
            m[i] = torch.rot90(m[i], k, (0, 1))
            if fl: m[i] = m[i].flip(1)
    x = x * (1 + 0.05 * torch.randn(B, 5, 1, 1)) + 0.005 * torch.randn_like(x)
    return x.clamp_(0, 1), m


def loss_fn(out, xb_frac, mb):
    if a.variant in ("baseline", "A"):
        return Fn.cross_entropy(out, (xb_frac >= CUT).long(), weight=cw)
    if a.variant in ("B", "AB"):
        return Fn.huber_loss(torch.sigmoid(out[:, 0]), xb_frac, delta=0.1)
    valid = (mb != 255).float(); tgt = (mb == 1).float(); p = torch.sigmoid(out[:, 0])
    wpix = (cw[0] + (cw[1] - cw[0]) * tgt) if a.weight_c_bce else 1.0   # default C: unweighted BCE (class weights were never applied to C)
    bce = (Fn.binary_cross_entropy_with_logits(out[:, 0], tgt, reduction="none") * valid * wpix).sum() / (valid * wpix).sum().clamp(min=1)
    inter, den = (p * tgt * valid).sum(), ((p + tgt) * valid).sum()
    return bce + (1 - (2 * inter + 1) / (den + 1))


@torch.no_grad()
def scores_of(model, X, bs=128):
    model.eval(); S = []
    for i in range(0, len(X), bs):
        S.append(score(a.variant, model(torch.from_numpy(X[i:i + bs]).float())).numpy())
    return np.concatenate(S)


def op_metrics(s, f, t):
    y, pred = f >= CUT, s >= t
    tp, fp, fn, tn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum(), (~pred & ~y).sum()
    return dict(thr=float(t), acc=float((tp + tn) / len(y)), discard=float(fp / max(fp + tn, 1)), miss=float(fn / max(fn + tp, 1)))


def full_metrics(s, f, t_rule, t_val):
    y = f >= CUT
    d = dict(n=int(len(y)), cloudy=float(y.mean()), baseline=float(max(y.mean(), 1 - y.mean())),
             at_rule=op_metrics(s, f, t_rule), at_val_2pct=op_metrics(s, f, t_val))
    if 0 < y.sum() < len(y):
        d["roc_auc"], d["pr_auc"] = float(roc_auc_score(y, s)), float(average_precision_score(y, s))
        d["oracle_miss_at_2pct_discard"] = float((s[y] < np.quantile(s[~y], 0.98)).mean())
    d["mean_score_90_100"] = float(s[f >= 0.9].mean()) if (f >= 0.9).any() else None
    d["by_cloud_fraction"] = [dict(bin=f"{lo:.0%}-{min(hi, 1):.0%}", n=int(((f >= lo) & (f < hi)).sum()),
                                   mean_score=float(s[(f >= lo) & (f < hi)].mean()), pred_cloudy=float((s[(f >= lo) & (f < hi)] >= t_rule).mean()))
                              for lo, hi in zip(BINS[:-1], BINS[1:]) if ((f >= lo) & (f < hi)).any()]
    return d


t_rule = 0.5 if a.variant in ("baseline", "A") else CUT
yva = frac[va_idx] >= CUT
log, best, best_ep, bad_ep, t0 = [], -1.0, 0, 0, time.time()
for ep in range(1, a.epochs + 1):
    net.train(); te = time.time()
    if a.balance_regions:
        w = np.array([1.0 / (region[tr_idx] == r).sum() for r in region[tr_idx]]); w /= w.sum()
        pick = np.sort(rng_tr.choice(tr_idx, min(a.tiles_per_epoch, len(tr_idx)), replace=False, p=w))
    else:
        pick = np.sort(rng_tr.choice(tr_idx, min(a.tiles_per_epoch, len(tr_idx)), replace=False))
    perm = rng_tr.permutation(len(pick))
    Xe = np.asarray(Xall[pick]); fe = torch.from_numpy(frac[pick]); me = torch.from_numpy(np.asarray(mask[pick])) if a.variant == "C" else None
    run, seen = 0.0, 0
    for i in range(0, len(perm), a.batch):
        b = perm[i:i + a.batch]
        if len(b) < 4: continue
        xb = torch.from_numpy(Xe[b]).float(); mb = me[b].clone() if me is not None else None
        xb, mb = augment(xb, mb)
        loss = loss_fn(net(xb), fe[b], mb)
        opt.zero_grad(); loss.backward(); opt.step(); run += loss.item() * len(b); seen += len(b)
    sched.step()
    sva = scores_of(net, Xva); vauc = float(roc_auc_score(yva, sva))
    rec = dict(epoch=ep, train_loss=run / seen, val_auc=vauc, val_acc_rule=op_metrics(sva, frac[va_idx], t_rule)["acc"], lr=opt.param_groups[0]["lr"], sec=time.time() - te)
    log.append(rec); saved = ""
    if vauc > best:
        best, best_ep, bad_ep = vauc, ep, 0; torch.save(net.state_dict(), RUN / "weights.pt"); saved = "  <- saved"
    else:
        bad_ep += 1
    el = time.time() - t0
    print(f"[{name}] epoch {ep}/{a.epochs} loss {rec['train_loss']:.4f} | val AUC {vauc:.3f} acc@rule {rec['val_acc_rule']:.3f} | "
          f"{rec['sec']:.0f}s ETA {el/ep*(a.epochs-ep)/60:.1f} min{saved}", flush=True)
    if bad_ep >= a.patience:
        print(f"[{name}] early stop (patience {a.patience})", flush=True); break

net.load_state_dict(torch.load(RUN / "weights.pt"))
sva = scores_of(net, Xva); t_val = float(np.quantile(sva[~yva], 0.98))
ste, str_ = scores_of(net, Xte), scores_of(net, Xtr_sub)
R = dict(name=name, variant=a.variant, tag=tag, seed=a.seed, params=n_params, weights_bytes=os.path.getsize(RUN / "weights.pt"), best_epoch=best_ep,
         epochs_run=len(log), wall_min=(time.time() - t0) / 60, config=vars(a), split=dict(test=TEST_R, val=VAL_R, n=[len(tr_idx), len(va_idx), len(te_idx)]),
         threshold_rule=t_rule, threshold_val_2pct=t_val,
         train=full_metrics(str_, frac[sub_tr], t_rule, t_val), val=full_metrics(sva, frac[va_idx], t_rule, t_val),
         test=full_metrics(ste, frac[te_idx], t_rule, t_val))
R["test_per_region"] = {r: full_metrics(ste[region[te_idx] == r], frac[te_idx][region[te_idx] == r], t_rule, t_val) for r in TEST_R}
for r in R["test_per_region"].values(): r.pop("by_cloud_fraction", None)
# per-scene miss breakdown on test (worst 5)
sc = scene[te_idx]; yte = frac[te_idx] >= CUT
R["test_worst_scenes_by_miss"] = sorted([dict(scene=int(s), n_cloudy=int((yte & (sc == s)).sum()),
                                              miss=float((ste[(sc == s) & yte] < t_val).mean())) for s in np.unique(sc) if (yte & (sc == s)).sum() >= 5],
                                        key=lambda d: -d["miss"])[:5]
net16 = build(a.variant); net16.load_state_dict(net.state_dict()); net16 = net16.half().float()
s16 = scores_of(net16, Xte)
R["fp16_roundtrip"] = dict(test_miss_at_val2pct=op_metrics(s16, frac[te_idx], t_val)["miss"], delta_miss=op_metrics(s16, frac[te_idx], t_val)["miss"] - R["test"]["at_val_2pct"]["miss"],
                           test_auc=float(roc_auc_score(yte, s16)))
x1 = torch.from_numpy(Xte[:1]).float(); net.eval()
with torch.no_grad():
    net(x1); tl = time.time()
    for _ in range(20): net(x1)
R["cpu_latency_ms_per_tile_128px"] = (time.time() - tl) / 20 * 1000
json.dump(dict(R, log=log), open(RUN / "metrics.json", "w"), indent=2)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3)); e = [r["epoch"] for r in log]
    ax[0].plot(e, [r["train_loss"] for r in log]); ax[0].set_title("train loss"); ax[1].plot(e, [r["val_auc"] for r in log]); ax[1].set_title("val AUC")
    plt.tight_layout(); plt.savefig(RUN / "loss_curve.png", dpi=100); plt.close()
except Exception as ex:
    print("plot skipped:", ex)
T = R["test"]
print(f"[{name}] DONE {R['wall_min']:.1f} min | params {n_params:,} | test AUC {T.get('roc_auc', float('nan')):.3f} | "
      f"@val-2%disc: discard {T['at_val_2pct']['discard']:.3f} miss {T['at_val_2pct']['miss']:.3f} | acc@rule {T['at_rule']['acc']:.3f} (base {T['baseline']:.3f})", flush=True)
