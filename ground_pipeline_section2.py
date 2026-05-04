"""
CoverGuard-PFI  |  Section 2 — Ground Pipeline
================================================
Paper section: §2 — Ground Data Processing Pipeline

Downloads NASA OPERA disturbance layers and HLS Landsat surface reflectance
for a target AOI, fuses them into per-parcel health signals, and writes
a scored GeoJSON, CSV, and six-panel diagnostic figure.

Inputs:
  - NASA Earthdata account (free): https://urs.earthdata.nasa.gov
  - BBOX, DIST_DATES, HLS_DATES constants defined in STEP 1

Outputs:
  outputs/section2_ground_pipeline/
    field_health_scores_geospatial.geojson    — per-segment health scores + geometry
    field_health_scores_per_parcel.csv        — tabular version of the same data
    field_health_scores_critical_parcels.csv  — segments with health score ≤ 3
    coverguard_field_health_diagnostic_CentralValley_CA.png — six-panel figure

Run:
    python ground_pipeline_section2.py

Dependencies: earthaccess, rasterio, rasterstats, geopandas, scikit-image,
              matplotlib, numpy, pandas. Requires a NASA Earthdata account.
"""

# ══════════════════════════════════════════════════════════════════
# STEP 1 — IMPORTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════
# Load all dependencies and define AOI, date windows, alarm thresholds,
# and health-score weights used throughout the pipeline.
import warnings
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for subprocess/headless runs
import matplotlib.pyplot as plt

import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import shapes as rio_shapes
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union
from rasterstats import zonal_stats

import earthaccess

warnings.filterwarnings("ignore")
# --- VS CODE PATH FIX ---
# This ensures the output folder is created exactly where your script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
OUT = SCRIPT_DIR / "outputs" / "section2_ground_pipeline"
# Output file names
OUT_GEOJSON  = OUT / "field_health_scores_geospatial.geojson"
OUT_CSV      = OUT / "field_health_scores_per_parcel.csv"
OUT_WARNINGS = OUT / "field_health_scores_critical_parcels.csv"
OUT_FIGURE   = OUT / "coverguard_field_health_diagnostic_CentralValley_CA.png"
OUT.mkdir(exist_ok=True)

# Create sub-folders immediately to prevent Windows path errors
(OUT / "dist").mkdir(exist_ok=True)
(OUT / "hls").mkdir(exist_ok=True)

# --- AOI: ~30 km × 30 km, Central Valley CA (intensively farmed) ---
# Format: (xmin, ymin, xmax, ymax) in WGS84 decimal degrees
BBOX = (-120.75, 36.50, -120.45, 36.80)

# --- Temporal windows ---
# DIST-ANN first available year is 2023 (annual calendar summaries only)
DIST_DATES = ("2023-01-01", "2023-12-31")
# Peak summer growing season for NDVI acquisition
HLS_DATES  = ("2023-06-01", "2023-08-31")

# --- Disturbance alarm thresholds ---
# VEG/GEN_DIST_STATUS: 0=none 1=first-detection 2=provisional 3=confirmed 4=ongoing
# CONF layers: 0–100 (percent confidence in detected disturbance)
VEG_STATUS_THR = 2      # ≥ provisional
VEG_CONF_THR   = 50     # ≥ 50 % confidence
GEN_STATUS_THR = 2
GEN_CONF_THR   = 50

# --- NDVI health flags ---
ALPHA_WEAK    = 0.40    # below this → "weak cover" flag
ALPHA_SEVERE  = 0.20    # below this → "severe" flag
ANOMALY_THR   = 0.15    # |segment NDVI − scene mean| → anomaly flag

# --- Grid segmentation parameter ---
# Target number of grid cells; actual cell side length l is derived at runtime
# from sqrt(H * W / GRID_N_CELLS) so the grid adapts to any BBOX size.
GRID_N_CELLS = 300

# --- Health score weights (must sum to 1.0) ---
# Each weight governs one sub-score; see compute_health_score() for details.
WEIGHTS = {
    "ndvi":        0.35,   # absolute NDVI level
    "anomaly":     0.20,   # deviation below scene mean
    "disturbance": 0.25,   # OPERA disturbance severity
    "variability": 0.10,   # within-segment patchiness
    "coverage":    0.10,   # valid-pixel fraction
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# ══════════════════════════════════════════════════════════════════
# STEP 2 — NASA EARTHDATA LOGIN
# ══════════════════════════════════════════════════════════════════
# Authenticate with NASA Earthdata via earthaccess; credentials are cached
# to ~/.netrc after the first interactive login so subsequent runs are silent.
try:
    auth = earthaccess.login(strategy="interactive") 
except Exception as e:
    print(f"Login failed: {e}")

# ══════════════════════════════════════════════════════════════════
# STEP 3 — DOWNLOAD OPERA DIST-ANN-HLS DISTURBANCE LAYERS
# ══════════════════════════════════════════════════════════════════
# Query and download the annual vegetation and general disturbance COGs for
# the target BBOX and year; each MGRS tile produces up to 21 separate layer files.
# File suffix uses dashes: e.g. _VEG-DIST-STATUS.tif, _GEN-DIST-CONF.tif

print("Searching OPERA_L3_DIST-ANN-HLS_V1 …")
dist_results = earthaccess.search_data(
    short_name="OPERA_L3_DIST-ANN-HLS_V1",
    bounding_box=BBOX,
    temporal=DIST_DATES,
    count=30,  # one MGRS tile has ~21 layer files; request generously
)
if not dist_results:
    raise RuntimeError(
        "No OPERA DIST-ANN-HLS granules found.\n"
        "  • Verify NASA Earthdata credentials.\n"
        "  • Confirm DIST_DATES is within 2023–present.\n"
        "  • Try search.earthdata.nasa.gov to check coverage for your BBOX."
    )
print(f"  Found {len(dist_results)} asset(s).")

dist_files = earthaccess.download(dist_results, local_path=str(OUT / "dist"))
print(f"  Downloaded {len(dist_files)} file(s).")

# ══════════════════════════════════════════════════════════════════
# STEP 4 — LOAD & PARSE DISTURBANCE LAYERS
# ══════════════════════════════════════════════════════════════════
# Read the downloaded COGs into numpy arrays and build the alarming-pixel
# boolean mask by combining vegetation and general disturbance status/confidence.

def find_layer_file(files, keyword):
    """
    Locate a downloaded layer file by keyword match on its filename.
    Checks both dash and underscore variants because OPERA naming conventions
    are inconsistent across product versions.

    Args:   files — list of downloaded file paths; keyword — e.g. "VEG-DIST-STATUS"
    Returns: absolute path string of the first match, or None if not found
    """
    kd = keyword.replace("_", "-").upper()
    ku = keyword.replace("-", "_").upper()
    for f in files:
        n = Path(str(f)).name.upper()
        if kd in n or ku in n:
            return str(f)
    return None

def read_cog(fpath):
    """
    Read band 1 of a single-band Cloud-Optimized GeoTIFF into memory.

    Returns: (float32 numpy array, rasterio profile dict)
    """
    with rasterio.open(fpath) as src:
        return src.read(1).astype("float32"), src.profile.copy()

def load_required(files, keyword):
    """
    Load a mandatory disturbance layer and raise a descriptive error if missing.

    Returns: (float32 array, rasterio profile dict)
    """
    f = find_layer_file(files, keyword)
    if f is None:
        raise RuntimeError(
            f"Required layer '{keyword}' not found.\n"
            "Downloaded files:\n" + "\n".join(f"  {x}" for x in files)
        )
    return read_cog(f)

def load_optional(files, keyword, shape):
    """
    Load an optional ancillary layer, falling back to zeros if the file is absent.
    Used for layers like VEG-MAX-ANOM and VEG-DIST-DUR that may not be present
    in all OPERA granules.

    Args:   shape — fallback array shape (H, W) matching the required layers
    Returns: (float32 array, rasterio profile dict or None)
    """
    f = find_layer_file(files, keyword)
    if f is None:
        return np.zeros(shape, dtype="float32"), None
    return read_cog(f)

# --- Required layers ---
veg_status, d_prof = load_required(dist_files, "VEG-DIST-STATUS")
veg_conf,   _      = load_required(dist_files, "VEG-DIST-CONF")
gen_status, _      = load_required(dist_files, "GEN-DIST-STATUS")
gen_conf,   _      = load_required(dist_files, "GEN-DIST-CONF")

# --- Optional ancillary layers used in health scoring ---
# VEG-MAX-ANOM: peak spectral anomaly magnitude (0–100 scale) during disturbance
# VEG-DIST-DUR: days from first to last detected loss anomaly (duration)
veg_max_anom, _ = load_optional(dist_files, "VEG-MAX-ANOM",  veg_status.shape)
veg_dist_dur,  _ = load_optional(dist_files, "VEG-DIST-DUR",  veg_status.shape)

# ── Alarming mask ──────────────────────────────────────────────────────────
# A pixel is "alarming" if vegetation OR general disturbance exceeds both
# status and confidence thresholds. The OR union captures cases where the two
# disturbance detection algorithms disagree — either trigger is sufficient.
alarming = (
    ((veg_status >= VEG_STATUS_THR) & (veg_conf >= VEG_CONF_THR)) |
    ((gen_status >= GEN_STATUS_THR) & (gen_conf >= GEN_CONF_THR))
)
print(f"Alarming pixel fraction (OPERA grid): {alarming.mean():.3%}")
if alarming.mean() == 0:
    print(
        "  WARNING: No alarming pixels found at current thresholds.\n"
        "  Consider lowering VEG_STATUS_THR or VEG_CONF_THR.\n"
        "  Continuing with full AOI for demonstration."
    )

# ══════════════════════════════════════════════════════════════════
# STEP 5 — DOWNLOAD HLS LANDSAT (HLSL30) SURFACE REFLECTANCE
# ══════════════════════════════════════════════════════════════════
# Query and download Harmonized Landsat-Sentinel surface reflectance bands
# for the peak growing season; individual band COGs are needed for NDVI/EVI.
# HLSL30 band naming: .B02. = Blue, .B04. = Red, .B05. = NIR, .Fmask. = cloud mask

print("\nSearching HLSL30 granules …")
hls_results = earthaccess.search_data(
    short_name="HLSL30",
    bounding_box=BBOX,
    temporal=HLS_DATES,
    count=5,
)
if not hls_results:
    raise RuntimeError(
        "No HLSL30 granules found.\n"
        "  • Try widening HLS_DATES.\n"
        "  • For Sentinel-2 coverage, switch short_name to 'HLSS30' "
        "    and NIR band tag to '.B8A.'"
    )
print(f"  Found {len(hls_results)} granule(s).")

hls_files = earthaccess.download(hls_results, local_path=str(OUT / "hls"))
print(f"  Downloaded {len(hls_files)} file(s).")

def pick_band(files, tag):
    """
    Match a downloaded HLS file by its dot-delimited band tag.

    Args:   tag — e.g. '.B04.' or '.Fmask.'
    Returns: absolute path string of the matching file, or None
    """
    for f in files:
        if tag in Path(str(f)).name:
            return str(f)
    return None

blue_f  = pick_band(hls_files, ".B02.")
red_f   = pick_band(hls_files, ".B04.")
nir_f   = pick_band(hls_files, ".B05.")
fmask_f = pick_band(hls_files, ".Fmask.")

if not red_f or not nir_f:
    raise RuntimeError(
        "B04 (Red) or B05 (NIR) not found in HLSL30 download.\n"
        "Check that earthaccess downloaded individual band files, "
        "not just a granule metadata file."
    )

# ══════════════════════════════════════════════════════════════════
# STEP 6 — COMPUTE NDVI & EVI FROM SURFACE REFLECTANCE
# ══════════════════════════════════════════════════════════════════
# Apply Fmask cloud/shadow masking, convert scaled integers to physical
# reflectance, then compute NDVI (and EVI when the blue band is available).

def read_hls_band(fpath):
    """
    Read a single HLS band COG and replace fill values with NaN.
    HLS uses negative integers as fill; masking them before arithmetic
    prevents spurious NDVI values at scene edges.

    Returns: (float32 array, rasterio profile dict)
    """
    with rasterio.open(fpath) as src:
        arr = src.read(1).astype("float32")
        arr[arr < 0] = np.nan
        return arr, src.profile.copy()

red,  ndvi_profile = read_hls_band(red_f)
nir,  _            = read_hls_band(nir_f)

# HLS stores surface reflectance as scaled integers (×10 000).
# Convert to physical reflectance for EVI; NDVI ratio is scale-invariant.
scale = 10000.0 if np.nanmax(red) > 2.0 else 1.0
red_r = red / scale
nir_r = nir / scale

# --- Cloud and shadow masking using Fmask ---
# Fmask bit 1 (0x02) = cloud, bit 3 (0x08) = cloud shadow.
# A boolean cloud_mask is kept for downstream reuse even when fmask_f is absent.
cloud_mask = np.zeros_like(red, dtype=bool)
if fmask_f:
    with rasterio.open(fmask_f) as fm:
        fmask_raw = fm.read(1).astype("uint8")
    cloud_mask = ((fmask_raw & 0x02) | (fmask_raw & 0x08)) > 0
    red[cloud_mask]  = np.nan
    nir[cloud_mask]  = np.nan
    red_r[cloud_mask] = np.nan
    nir_r[cloud_mask] = np.nan

# --- NDVI ---
# np.where preserves NaN at cloud/fill pixels; np.clip alone can collapse NaN
# to the lower bound in some numpy builds.
raw_ndvi = (nir - red) / (nir + red + 1e-6)
ndvi = np.where(cloud_mask | np.isnan(raw_ndvi), np.nan,
                np.clip(raw_ndvi, -1.0, 1.0)).astype("float32")

# --- EVI (Enhanced Vegetation Index) — less saturated than NDVI at high biomass ---
# EVI = 2.5 × (NIR − Red) / (NIR + 6·Red − 7.5·Blue + 1)
evi = None
if blue_f:
    blue, _ = read_hls_band(blue_f)
    blue_r  = blue / scale
    blue_r[cloud_mask] = np.nan   # mirror the same cloud mask
    raw_evi = 2.5 * (nir_r - red_r) / (nir_r + 6 * red_r - 7.5 * blue_r + 1 + 1e-6)
    evi = np.where(cloud_mask | np.isnan(raw_evi), np.nan,
                   np.clip(raw_evi, -1.0, 2.0)).astype("float32")

# --- Save NDVI raster ---
ndvi_profile.update(dtype="float32", count=1, nodata=float("nan"))
ndvi_f = OUT / "ndvi.tif"
with rasterio.open(ndvi_f, "w", **ndvi_profile) as dst:
    dst.write(ndvi, 1)

print(
    f"NDVI  mean={np.nanmean(ndvi):.3f}  "
    f"p10={np.nanpercentile(ndvi, 10):.3f}  "
    f"p90={np.nanpercentile(ndvi, 90):.3f}"
)
if evi is not None:
    print(
        f"EVI   mean={np.nanmean(evi):.3f}  "
        f"p10={np.nanpercentile(evi, 10):.3f}  "
        f"p90={np.nanpercentile(evi, 90):.3f}"
    )

# ══════════════════════════════════════════════════════════════════
# STEP 7 — REPROJECT OPERA LAYERS ONTO HLS RASTER GRID
# ══════════════════════════════════════════════════════════════════
# OPERA and HLS tiles share 30 m resolution but use different MGRS-derived
# CRS projections; reprojecting to the HLS grid ensures pixel-perfect alignment
# before any zonal or per-pixel fusion.

with rasterio.open(ndvi_f) as src:
    ndvi_meta = src.meta.copy()
    ndvi_arr  = src.read(1)

H, W = ndvi_meta["height"], ndvi_meta["width"]

def reproj(arr, src_profile, resamp=Resampling.nearest):
    """
    Reproject a 2-D float32 source array into the NDVI raster's CRS and grid.
    Nearest-neighbour is used by default to preserve categorical status values;
    pass Resampling.bilinear for continuous layers (anomaly magnitude, duration).

    Args:   arr — source array; src_profile — rasterio profile of the source
    Returns: reprojected float32 array matching (H, W) of the NDVI grid
    """
    dst = np.zeros((H, W), dtype="float32")
    reproject(
        source=arr.astype("float32"),
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=ndvi_meta["transform"],
        dst_crs=ndvi_meta["crs"],
        resampling=resamp,
    )
    return dst

veg_status_r  = reproj(veg_status,                d_prof)
veg_conf_r    = reproj(veg_conf,                  d_prof)
gen_status_r  = reproj(gen_status,                d_prof)
gen_conf_r    = reproj(gen_conf,                  d_prof)
veg_max_anom_r = reproj(veg_max_anom,             d_prof, Resampling.bilinear)
veg_dist_dur_r = reproj(veg_dist_dur,             d_prof, Resampling.bilinear)
alarming_r     = reproj(alarming.astype("float32"), d_prof).astype(bool)

# ══════════════════════════════════════════════════════════════════
# STEP 8 — ADAPTIVE GRID SEGMENTATION
# ══════════════════════════════════════════════════════════════════
# Divide the scene into a regular l×l pixel grid that scales to any AOI size.
# Cell side length l is derived from GRID_N_CELLS so the segmentation adapts
# automatically rather than using a fixed pixel count. Fully cloud-covered cells
# receive label 0 and are excluded from all downstream statistics and scoring.

GRID_N_CELLS = 300   # approximate number of grid cells across the scene

# Derive cell side length l (pixels) from the raster area.
# l = sqrt(H * W / N); rounded to nearest integer, minimum 1 pixel.
l = max(1, int(round(np.sqrt(H * W / GRID_N_CELLS))))
print(f"\nGrid cell side: {l} px  →  "
      f"{(H // l)} rows × {(W // l)} cols "
      f"= {(H // l) * (W // l)} cells (excluding boundary remainder)")

# Allocate label array; 0 = invalid/cloud-only, labels start at 1.
segments = np.zeros((H, W), dtype="int32")
label = 1
valid_cell_count = 0
cloudy_cell_count = 0

for r0 in range(0, H - l + 1, l):      # row-major iteration
    for c0 in range(0, W - l + 1, l):
        cell_ndvi = ndvi_arr[r0:r0 + l, c0:c0 + l]
        # Keep cell only if it has at least one valid (non-NaN) NDVI pixel.
        if np.any(np.isfinite(cell_ndvi)):
            segments[r0:r0 + l, c0:c0 + l] = label
            label += 1
            valid_cell_count += 1
        else:
            # Label stays 0 — cell is entirely cloud/fill; excluded downstream.
            cloudy_cell_count += 1

print(f"  Valid cells  : {valid_cell_count}")
print(f"  Cloud-only cells (excluded): {cloudy_cell_count}")

# ══════════════════════════════════════════════════════════════════
# STEP 9 — BUILD PER-CELL GEODATAFRAME
# ══════════════════════════════════════════════════════════════════
# Convert each valid grid cell to a rectangular Shapely polygon and assemble
# a GeoDataFrame. Alarm-zone membership is computed with numpy bincount
# (O(H×W)) rather than geometry predicates, which is orders of magnitude faster.

t   = ndvi_meta["transform"]
crs = ndvi_meta["crs"]

# --- Alarm zone flag via numpy (O(H×W), zero geometry cost) ----------------
seg_flat   = segments.ravel()
alarm_flat = alarming_r.ravel().astype(np.uint8)
n_labels   = int(segments.max()) + 1   # labels 1..n; 0 = cloud-only, ignored

alarm_px_count = np.bincount(seg_flat, weights=alarm_flat.astype(float),
                              minlength=n_labels)
alarm_by_id = alarm_px_count > 0   # True if any alarming pixel touches this cell

# --- Build one rectangular polygon per valid label -------------------------
# Rasterio's transform maps (col, row) → (x, y).  For a rectangle defined
# by pixel corners (r0, c0) to (r0+l, c0+l) in row/col space, the four
# geographic corners follow directly from the affine transform.
from rasterio.transform import xy as rio_xy
from shapely.geometry import box as shapely_box

rows_list = []
for r0 in range(0, H - l + 1, l):
    for c0 in range(0, W - l + 1, l):
        sid = int(segments[r0, c0])
        if sid == 0:
            continue   # cloud-only cell
        # Upper-left and lower-right corners in the raster's native CRS.
        x0, y0 = rio_xy(t, r0,     c0,     offset="ul")
        x1, y1 = rio_xy(t, r0 + l, c0 + l, offset="ul")
        rows_list.append({
            "seg_id":        sid,
            "geometry":      shapely_box(min(x0, x1), min(y0, y1),
                                         max(x0, x1), max(y0, y1)),
            "in_alarm_zone": bool(alarm_by_id[sid]),
        })

seg_gdf = gpd.GeoDataFrame(rows_list, crs=crs)
print(f"\nGrid GeoDataFrame: {len(seg_gdf)} valid cells.")

# ══════════════════════════════════════════════════════════════════
# STEP 10 — ZONAL STATISTICS PER SEGMENT
# ══════════════════════════════════════════════════════════════════
# Aggregate NDVI, EVI, and all reprojected OPERA layers to per-cell summary
# statistics using rasterstats. These form the raw inputs for health scoring.

affine = ndvi_meta["transform"]

def zs(layer, stats):
    """
    Shorthand wrapper for rasterstats.zonal_stats operating on numpy arrays.

    Args:   layer — 2-D numpy array aligned to the NDVI grid; stats — list of stat names
    Returns: list of dicts, one per segment in seg_gdf
    """
    return zonal_stats(seg_gdf, layer, affine=affine, stats=stats, nodata=np.nan)

ndvi_zs   = zs(ndvi_arr,       ["mean", "median", "std", "min", "max",
                                  "count", "percentile_10", "percentile_90"])
vst_zs    = zs(veg_status_r,   ["mean"])
vcf_zs    = zs(veg_conf_r,     ["mean"])
gst_zs    = zs(gen_status_r,   ["mean"])
gcf_zs    = zs(gen_conf_r,     ["mean"])
vanom_zs  = zs(veg_max_anom_r, ["mean"])
vdur_zs   = zs(veg_dist_dur_r, ["mean"])

# Add EVI zonal mean if available
if evi is not None:
    evi_zs = zs(evi, ["mean"])
    seg_gdf["evi_mean"] = [r["mean"] or 0.0 for r in evi_zs]
else:
    seg_gdf["evi_mean"] = np.nan

seg_gdf = seg_gdf.join(pd.DataFrame(ndvi_zs))

seg_gdf["veg_status_mean"]   = [r["mean"] or 0.0 for r in vst_zs]
seg_gdf["veg_conf_mean"]     = [r["mean"] or 0.0 for r in vcf_zs]
seg_gdf["gen_status_mean"]   = [r["mean"] or 0.0 for r in gst_zs]
seg_gdf["gen_conf_mean"]     = [r["mean"] or 0.0 for r in gcf_zs]
seg_gdf["veg_max_anom_mean"] = [r["mean"] or 0.0 for r in vanom_zs]
seg_gdf["veg_dist_dur_mean"] = [r["mean"] or 0.0 for r in vdur_zs]

# NDVI anomaly: segment mean minus scene mean; negative = below-average vegetation.
scene_mean = float(np.nanmean(ndvi_arr))
seg_gdf["ndvi_anomaly"] = seg_gdf["mean"].fillna(0.0) - scene_mean

# Binary disturbance and health flags
seg_gdf["flag_weak"]    = seg_gdf["mean"] < ALPHA_WEAK
seg_gdf["flag_severe"]  = seg_gdf["mean"] < ALPHA_SEVERE
seg_gdf["flag_anomaly"] = seg_gdf["ndvi_anomaly"].abs() > ANOMALY_THR

# ══════════════════════════════════════════════════════════════════
# STEP 11 — FIELD HEALTH SCORE (1–10)
# ══════════════════════════════════════════════════════════════════
# Compute a weighted composite health score for each segment on a 1–10 integer
# scale (1 = critically stressed, 10 = healthy). Five sub-scores capture
# complementary dimensions of vegetation condition:
#
# Each segment receives five sub-scores in [0, 1]:
#
#   Sub-score     Weight  Description
#   ──────────────────────────────────────────────────────────────────────────
#   ndvi           0.35   Absolute NDVI level; linear from 0 (NDVI≤0) to 1
#                          (NDVI≥0.80). Captures current greenness.
#   anomaly        0.20   Negative departure from scene mean penalised
#                          linearly up to −0.30; positive anomalies not
#                          rewarded beyond 1.0. Captures relative stress.
#   disturbance    0.25   Disturbance status (0–4) × confidence (0–1) penalty
#                          with additional penalties for large spectral anomaly
#                          magnitude and long disturbance duration.
#   variability    0.10   High within-segment NDVI std → patchy / stressed
#                          vegetation; std ≥ 0.25 maps to 0.
#   coverage       0.10   Fraction of valid (non-NaN) pixels in the segment;
#                          cloud-covered or filled segments score low here.
#
# Composite = Σ(weight × sub-score); mapped to int ∈ [1, 10] via round(×10).

def sub_ndvi(v):
    """
    Map segment mean NDVI to a [0, 1] health sub-score.
    Linear from 0 (NDVI ≤ 0) to 1 (NDVI ≥ 0.80); captures absolute greenness.

    Returns: float in [0, 1]
    """
    return float(np.clip(v / 0.80, 0.0, 1.0))

def sub_anomaly(a):
    """
    Penalise segments whose NDVI falls below the scene mean.
    Negative anomaly is capped at −0.30; positive anomalies are not rewarded
    beyond 1.0 to avoid over-scoring edge pixels next to bare soil.

    Returns: float in [0, 1]
    """
    return float(np.clip(1.0 - max(-a, 0.0) / 0.30, 0.0, 1.0))

def sub_disturbance(vs, vc, gs, gc, anom_mag, dur_days):
    """
    Compute a disturbance penalty sub-score combining status, confidence,
    anomaly magnitude, and event duration from OPERA disturbance layers.
    Higher status × confidence means more severe detected disturbance → lower score.
    Additional penalties for large spectral anomaly magnitude (up to −15%) and
    long disturbance duration (up to −10%) reflect persistent or severe events.

    Args:   vs/gs — mean VEG/GEN status (0–4); vc/gc — mean confidence (0–100);
            anom_mag — mean VEG-MAX-ANOM (0–100); dur_days — mean VEG-DIST-DUR (days)
    Returns: float in [0, 1]; 1 = no disturbance, 0 = severe confirmed disturbance
    """
    vp   = np.clip(vs / 4.0, 0, 1) * np.clip(vc / 100.0, 0, 1)
    gp   = np.clip(gs / 4.0, 0, 1) * np.clip(gc / 100.0, 0, 1)
    base = float(np.clip(1.0 - max(vp, gp), 0.0, 1.0))
    anom_pen = float(np.clip(anom_mag / 100.0, 0.0, 0.15))  # up to 15 % extra penalty
    dur_pen  = float(np.clip(dur_days  / 365.0, 0.0, 0.10)) # up to 10 % extra penalty
    return float(np.clip(base - anom_pen - dur_pen, 0.0, 1.0))

def sub_variability(std):
    """
    Penalise high within-segment NDVI variability, which indicates a patchy
    or partially stressed canopy rather than uniform healthy cover.
    std ≥ 0.25 maps to 0 (maximally patchy); std = 0 maps to 1.

    Returns: float in [0, 1]
    """
    return float(np.clip(1.0 - std / 0.25, 0.0, 1.0))

def sub_coverage(count, total_px):
    """
    Score segment data completeness as the fraction of valid (non-NaN) pixels.
    Cloud-dominated or fill-heavy segments have lower certainty, so they are
    down-scored proportionally rather than excluded outright.

    Args:   count — rasterstats valid-pixel count; total_px — total grid cell pixels
    Returns: float in [0, 1]
    """
    return float(np.clip(count / max(total_px, 1), 0.0, 1.0))

# Precompute raster pixel count per segment label
seg_px = {int(v): int((segments == v).sum()) for v in np.unique(segments)}

def compute_health_score(row):
    """
    Aggregate the five sub-scores into a final integer health score for one segment.
    The weighted sum is clamped to [0, 1] before scaling to guard against floating-
    point accumulation errors, then mapped to [1, 10] via round(raw × 10).

    Returns: int in [1, 10]
    """
    ndvi_v = row["mean"]          if pd.notna(row["mean"])          else 0.0
    anom   = row["ndvi_anomaly"]  if pd.notna(row["ndvi_anomaly"])  else 0.0
    std    = row["std"]           if pd.notna(row["std"])           else 0.0
    cnt    = row["count"]         if pd.notna(row["count"])         else 0.0

    raw = (
          WEIGHTS["ndvi"]        * sub_ndvi(ndvi_v)
        + WEIGHTS["anomaly"]     * sub_anomaly(anom)
        + WEIGHTS["disturbance"] * sub_disturbance(
            row["veg_status_mean"], row["veg_conf_mean"],
            row["gen_status_mean"], row["gen_conf_mean"],
            row["veg_max_anom_mean"], row["veg_dist_dur_mean"],
          )
        + WEIGHTS["variability"] * sub_variability(std)
        + WEIGHTS["coverage"]    * sub_coverage(cnt, seg_px.get(int(row["seg_id"]), 1))
    )
    # Clamp raw to [0, 1] before scaling to guard against float accumulation,
    # then map to [1, 10] as an integer.
    return int(np.clip(round(min(raw, 1.0) * 10), 1, 10))

seg_gdf["health_score"] = seg_gdf.apply(compute_health_score, axis=1)

print("Health score distribution (1 = critically stressed, 10 = healthy):")
print(seg_gdf["health_score"].value_counts().sort_index().to_string())

# ══════════════════════════════════════════════════════════════════
# STEP 12 — SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════
# Write the scored GeoDataFrame to GeoJSON (for GIS tools), CSV (for tabular
# analysis), a critical-parcel warning CSV (health ≤ 3), and a diagnostic figure.

keep_cols = [
    "seg_id", "in_alarm_zone",
    "mean", "median", "std", "min", "max", "percentile_10", "percentile_90",
    "evi_mean", "ndvi_anomaly",
    "flag_weak", "flag_severe", "flag_anomaly",
    "veg_status_mean", "veg_conf_mean",
    "gen_status_mean", "gen_conf_mean",
    "veg_max_anom_mean", "veg_dist_dur_mean",
    "health_score",
]

seg_gdf[keep_cols + ["geometry"]].to_crs("EPSG:4326").to_file(
    OUT_GEOJSON, driver="GeoJSON"
)
seg_gdf[keep_cols].to_csv(OUT_CSV, index=False)
print(f"Outputs written to: {OUT}/")

seg_gdf[seg_gdf[keep_cols]['health_score'] <= 3].to_csv(OUT_WARNINGS, index=False)
print(f"Outputs written to: {OUT}/")

# ══════════════════════════════════════════════════════════════════
# STEP 13 — DIAGNOSTIC VISUALISATION
# ══════════════════════════════════════════════════════════════════
# Render a six-panel figure showing NDVI, the alarming disturbance mask, grid
# segmentation, VEG_DIST_STATUS, NDVI anomaly, and per-segment health scores.

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("CubeSat Disturbance & Field Health Pipeline", fontsize=15, fontweight="bold")

def add_cbar(ax, im, label):
    """Attach a compact colorbar to ax with a consistent font size."""
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(label, fontsize=8)

# Panel 1: NDVI
ax = axes[0, 0]
im = ax.imshow(ndvi_arr, cmap="RdYlGn", vmin=-0.1, vmax=0.8)
add_cbar(ax, im, "NDVI")
ax.set_title("NDVI — HLS L30 (cloud-masked)"); ax.axis("off")

# Panel 2: Alarming disturbance mask
ax = axes[0, 1]
ax.imshow(alarming_r, cmap="Reds")
ax.set_title(
    f"Alarming Mask\n"
    f"(VEG status ≥ {VEG_STATUS_THR}, conf ≥ {VEG_CONF_THR} %  |  "
    f"GEN status ≥ {GEN_STATUS_THR}, conf ≥ {GEN_CONF_THR} %)"
)
ax.axis("off")

# Panel 3: Grid cell boundaries overlaid on colour-mapped NDVI
ax = axes[0, 2]
ndvi_display = np.nan_to_num(ndvi_arr, nan=0.0)
ndvi_rgb = plt.cm.RdYlGn(plt.Normalize(-0.1, 0.8)(ndvi_display))[:, :, :3]
# Draw cell grid lines directly on the image array
grid_overlay = ndvi_rgb.copy()
for r0 in range(0, H, l):
    if r0 < H:
        grid_overlay[r0, :] = [0, 0, 0]
for c0 in range(0, W, l):
    if c0 < W:
        grid_overlay[:, c0] = [0, 0, 0]
ax.imshow(grid_overlay)
ax.set_title(f"Grid Segmentation ({l}×{l} px cells, N≈{GRID_N_CELLS})")
ax.axis("off")

# Panel 4: VEG_DIST_STATUS (reprojected to HLS grid)
ax = axes[1, 0]
im = ax.imshow(veg_status_r, cmap="OrRd", vmin=0, vmax=4)
add_cbar(ax, im, "Status (0–4)")
ax.set_title("VEG_DIST_STATUS"); ax.axis("off")

# Panel 5: Per-pixel NDVI anomaly relative to scene mean
ax = axes[1, 1]
anom_px = np.nan_to_num(ndvi_arr, nan=0.0) - scene_mean
im = ax.imshow(anom_px, cmap="RdBu", vmin=-0.4, vmax=0.4)
add_cbar(ax, im, "Δ NDVI")
ax.set_title(f"NDVI Anomaly (pixel vs scene mean = {scene_mean:.2f})"); ax.axis("off")

# Panel 6: Health score rasterised per segment
health_raster = np.full((H, W), fill_value=np.nan, dtype="float32")
for _, row in seg_gdf.iterrows():
    health_raster[segments == int(row["seg_id"])] = row["health_score"]

ax = axes[1, 2]
try:
    cmap10 = plt.colormaps["RdYlGn"].resampled(10)   # matplotlib ≥ 3.7
except AttributeError:
    cmap10 = plt.cm.get_cmap("RdYlGn", 10)           # matplotlib < 3.7
im = ax.imshow(health_raster, cmap=cmap10, vmin=0.5, vmax=10.5)
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=range(1, 11))
cb.set_label("Health score (1 = critically stressed, 10 = healthy)", fontsize=8)
ax.set_title("Field Health Score (per segment)"); ax.axis("off")

plt.tight_layout()
plt.savefig(OUT_FIGURE, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure saved: {OUT_FIGURE}")