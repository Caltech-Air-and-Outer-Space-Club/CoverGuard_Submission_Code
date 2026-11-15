"""Idle-CPU latency of the C+balanced U-Net and the baseline classifier on one 128 px tile (median of 50)."""
import time, torch, statistics as st
from models_v2 import build
for v in ("C", "baseline"):
    for th in (1, 4):
        torch.set_num_threads(th); n = build(v).eval(); x = torch.rand(1, 5, 128, 128)
        with torch.no_grad():
            for _ in range(5): n(x)
            t = []
            for _ in range(50): s = time.perf_counter(); n(x); t.append((time.perf_counter() - s) * 1000)
        print(f"{v:9s} threads={th}: median {st.median(t):.1f} ms/tile (128 px)")
