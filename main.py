"""
CoverGuard-PFI  |  NASA Space to Soil Challenge Submission
===========================================================
4-satellite SmallSat constellation for cover-crop failure diagnosis from orbit.

Runs the full pipeline in paper section order:

  Section 2  ground_pipeline_section2.py       NASA OPERA + HLS → per-parcel health signals
  Section 3  onboard_classifier_section3.py    LightGBM: 6-class cover-crop failure diagnosis
  Section 4  downlink_simulation_section4.py   Adaptive downlink scheduler, 4-stage progression

Run:
    python main.py                  full pipeline (sections 2 → 3 → 4)
    python main.py --section 2      ground pipeline only
    python main.py --section 3      onboard classifier only
    python main.py --section 4      downlink simulation only

Setup:
    pip install -r requirements.txt
    Section 2 requires a free NASA Earthdata account: https://urs.earthdata.nasa.gov
    Sections 3 and 4 are fully self-contained (no internet needed).

Outputs are saved to the outputs/ folder.
"""

import subprocess
import sys
import os
import time
import argparse

BASE = os.path.dirname(os.path.abspath(__file__))

SECTIONS = {
    2: {
        "label":  "Section 2 — Ground Pipeline",
        "script": os.path.join(BASE, "ground_pipeline_section2.py"),
        "note":   "NASA OPERA disturbance layers + HLS surface reflectance → per-parcel health scores (1–10).\n"
                  "  Outputs: GeoJSON, CSV, diagnostic figure.  Requires NASA Earthdata account + internet.",
    },
    3: {
        "label":  "Section 3 — Onboard Classifier",
        "script": os.path.join(BASE, "onboard_classifier_section3.py"),
        "note":   "LightGBM trained on 297,019 labeled pixels. Classifies: HEALTHY, MOISTURE_STRESS,\n"
                  "  EXCESS_WETNESS, NUTRIENT_DEFICIT, POOR_ESTABLISHMENT, PEST_OR_DISEASE.\n"
                  "  Macro-F1 = 0.96  |  <1 ms/pixel inference  |  5 MB model size.",
    },
    4: {
        "label":  "Section 4 — Downlink Simulation",
        "script": os.path.join(BASE, "downlink_simulation_section4.py"),
        "note":   "4-stage progressive scheduler simulation (Monte Carlo, 30 seeds).\n"
                  "  Adaptive scheduler improves median utility 20.8% over naive FIFO.\n"
                  "  99.96% data reduction: 128 MB raw imagery → ~50 bytes per parcel.",
    },
}


def run_section(sec_num):
    sec = SECTIONS[sec_num]
    print(f"\n{'='*70}")
    print(f"  {sec['label']}")
    print(f"  {sec['note']}")
    print(f"{'='*70}\n")

    start = time.time()
    result = subprocess.run([sys.executable, sec["script"]], text=True)
    elapsed = time.time() - start

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n  [Section {sec_num} {status}] in {elapsed:.1f}s")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="CoverGuard full pipeline runner")
    parser.add_argument("--section", type=int, choices=[2, 3, 4],
                        help="Run a single section only (2, 3, or 4)")
    args = parser.parse_args()

    print("=" * 70)
    print("  CoverGuard-PFI  |  NASA Space to Soil Challenge")
    print("  Full pipeline — paper section order: Ground → Classifier → Downlink")
    print("=" * 70)

    sections_to_run = [args.section] if args.section else [2, 3, 4]
    failures = []

    for sec_num in sections_to_run:
        code = run_section(sec_num)
        if code != 0:
            failures.append(sec_num)

    print(f"\n{'='*70}")
    if failures:
        print(f"  Pipeline complete — {len(failures)} section(s) failed: {failures}")
    else:
        print("  Pipeline complete — all sections passed.")


if __name__ == "__main__":
    main()
