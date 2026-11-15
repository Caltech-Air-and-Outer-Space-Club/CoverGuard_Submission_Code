"""Collect runs/*/metrics.json into one results table (runs/results.md). Mean +- sd over seeds when a variant has several."""
import glob, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
runs = [json.load(open(p)) for p in sorted(glob.glob(str(HERE / "runs" / "*" / "metrics.json")))]
by = {}
for r in runs:
    k = r["variant"] + ("+bal" if r.get("tag") == "bal" else "") + r["config"].get("suffix", "")
    by.setdefault(k, []).append(r)


def ms(vals, fmt="{:.3f}"):
    v = np.array(vals, float)
    return fmt.format(v.mean()) + (f" ± {fmt.format(v.std(ddof=1))}" if len(v) > 1 else "")


cols = [("seeds", lambda R: str(len(R))), ("params", lambda R: f"{R[0]['params']:,}"), ("size MB", lambda R: f"{R[0]['weights_bytes']/1e6:.2f}"),
        ("train AUC", lambda R: ms([r["train"]["roc_auc"] for r in R])), ("val AUC", lambda R: ms([r["val"]["roc_auc"] for r in R])),
        ("TEST AUC", lambda R: ms([r["test"]["roc_auc"] for r in R])), ("TEST PR-AUC", lambda R: ms([r["test"]["pr_auc"] for r in R])),
        ("test acc @rule (base)", lambda R: ms([r["test"]["at_rule"]["acc"] for r in R]) + f" ({R[0]['test']['baseline']:.3f})"),
        ("train acc @rule", lambda R: ms([r["train"]["at_rule"]["acc"] for r in R])),
        ("test discard @val-2%", lambda R: ms([r["test"]["at_val_2pct"]["discard"] for r in R])),
        ("TEST miss @val-2%", lambda R: ms([r["test"]["at_val_2pct"]["miss"] for r in R])),
        ("mean score 90-100%", lambda R: ms([r["test"]["mean_score_90_100"] for r in R], "{:.2f}")),
        ("FP16 Δmiss", lambda R: ms([r["fp16_roundtrip"]["delta_miss"] for r in R], "{:+.3f}")),
        ("latency ms (128px)", lambda R: ms([r["cpu_latency_ms_per_tile_128px"] for r in R], "{:.0f}")),
        ("best ep", lambda R: ms([r["best_epoch"] for r in R], "{:.0f}"))]
order = [v for v in ("baseline", "A", "B", "AB", "C", "baseline+bal", "AB+bal", "C+bal") if v in by] + sorted(v for v in by if v not in ("baseline", "A", "B", "AB", "C", "baseline+bal", "AB+bal", "C+bal"))
L = ["| variant | " + " | ".join(c[0] for c in cols) + " |", "|---|" + "---|" * len(cols)]
for v in order: L.append(f"| {v} | " + " | ".join(f(by[v]) for _, f in cols) + " |")
L += ["", "Per-region TEST miss @val-2% / AUC (mean over seeds):", "", "| variant | " + " | ".join(by[order[0]][0]["test_per_region"]) + " |", "|---|" + "---|" * len(by[order[0]][0]["test_per_region"])]
for v in order:
    L.append(f"| {v} | " + " | ".join(f"{np.mean([r['test_per_region'][g]['at_val_2pct']['miss'] for r in by[v]]):.2f} / {np.mean([r['test_per_region'][g]['roc_auc'] for r in by[v]]):.2f}"
                                    for g in by[v][0]["test_per_region"]) + " |")
(HERE / "runs" / "results.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
