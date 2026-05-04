"""
CoverGuard-PFI  |  Section 4 — Downlink Simulation
====================================================
Paper section: §4 — Progressive Downlink Scheduling

Orchestrates four progressive downlink scheduler stages that build on each
other, from a simple FIFO baseline to a full adaptive multi-value-class
scheduler with stale suppression and constellation duplicate removal.

Inputs:
  - downlink_stages/stage1_basic_scheduler.py      — baseline FIFO vs knapsack
  - downlink_stages/stage2_fragmentation.py        — resumable packet fragmentation
  - downlink_stages/stage3_lookahead_deadlines.py  — deadline-aware lookahead
  - downlink_stages/stage4_adaptive_scheduler.py   — value classes + deduplication

Outputs:
  outputs/section4_downlink_simulation/
    stage1_baseline_fifo_vs_adaptive_knapsack.txt
    stage2_fragmented_raw_image_transmission.txt
    stage3_deadline_aware_lookahead_scheduling.txt
    stage4_adaptive_value_classes_stale_suppression.txt

Run:
    python downlink_simulation_section4.py

Dependencies: no third-party packages required; each stage script runs as a
              subprocess using the same Python interpreter as this orchestrator.
"""

import subprocess
import sys
import os
import time

BASE = os.path.dirname(os.path.abspath(__file__))
STAGES_DIR = os.path.join(BASE, "downlink_stages")
OUTPUT_DIR = os.path.join(BASE, "outputs", "section4_downlink_simulation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Progressive stage definitions ──────────────────────────────────────────
# Each tuple is (script_filename, display_label, output_filename).
# Stages are ordered from simplest to most realistic: each builds on the
# constraints of the previous, forming a fair progression for the paper's
# comparative analysis. Running them in sequence also lets Stage 4 benefit
# from any shared I/O state written by earlier stages.
STAGES = [
    ("stage1_basic_scheduler.py",     "Stage 1 — Basic scheduler (indivisible packets, FIFO vs adaptive)",
     "stage1_baseline_fifo_vs_adaptive_knapsack.txt"),
    ("stage2_fragmentation.py",       "Stage 2 — Fragmentation (resumable raw-image transmission)",
     "stage2_fragmented_raw_image_transmission.txt"),
    ("stage3_lookahead_deadlines.py", "Stage 3 — Lookahead + deadline-aware scheduling",
     "stage3_deadline_aware_lookahead_scheduling.txt"),
    ("stage4_adaptive_scheduler.py",  "Stage 4 — Adaptive: value classes, stale suppression, duplicate removal",
     "stage4_adaptive_value_classes_stale_suppression.txt"),
]


def run_stage(filename, label, out_name):
    """
    Execute one scheduler stage as a subprocess and capture its output.
    subprocess is used (rather than direct import) to isolate each stage's
    module-level state — global variables and class definitions in stage N
    cannot bleed into stage N+1, which would produce misleading results.
    Full stdout is written to out_name; only the first 40 lines print to
    the console to keep the orchestrator output readable.

    Args:   filename — script filename inside STAGES_DIR;
            label    — human-readable stage description for the console header;
            out_name — output text filename inside OUTPUT_DIR
    Returns: subprocess return code (0 = success, non-zero = failure)
    """
    path = os.path.join(STAGES_DIR, filename)
    out_file = os.path.join(OUTPUT_DIR, out_name)

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")

    start = time.time()
    result = subprocess.run(
        [sys.executable, path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.time() - start

    with open(out_file, "w") as f:
        f.write(result.stdout)

    lines = result.stdout.splitlines()
    for line in lines[:40]:
        print(line)
    if len(lines) > 40:
        print(f"  ... ({len(lines) - 40} more lines — see {out_file})")

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n  [{status}] in {elapsed:.1f}s — full output saved to {out_file}")
    return result.returncode


def main():
    print("CoverGuard — Section 4: Downlink Simulation")
    print("=" * 65)
    print(f"Running {len(STAGES)} stages. Outputs saved to: {OUTPUT_DIR}\n")

    overall_start = time.time()
    failures = []   # collect failed stage filenames for the final summary

    for filename, label, out_name in STAGES:
        code = run_stage(filename, label, out_name)
        if code != 0:
            # Non-zero exit means the stage script raised an exception or
            # asserted a failure condition; collect but continue so later
            # stages still run and produce output for the comparison table.
            failures.append(filename)

    total = time.time() - overall_start
    print(f"\n{'='*65}")
    print(f"All stages complete in {total:.1f}s")
    if failures:
        print(f"  WARNING — {len(failures)} stage(s) failed: {', '.join(failures)}")
    else:
        print("  All stages passed.")


if __name__ == "__main__":
    main()
