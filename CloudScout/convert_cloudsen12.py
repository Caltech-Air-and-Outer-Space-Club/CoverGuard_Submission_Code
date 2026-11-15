"""Convert CloudSEN12+ (Sentinel-2, expert labels) into CloudScout tiles (30 m, 256x256, 5 bands).

Only the 2000x2000 patches (10 m, about 20 km) are used: resampled 3x3 to 30 m they are about 666 px,
enough for 256 px tiles. The 509x509 patches are too small and are skipped.

Band choice (rasterio bands are 1-based = HF "Index" + 1):
    L2A: B02 blue=2, B03 green=3, B04 red=4, B08 NIR=8, B11 SWIR1=11     (L2A has no B10)
    L1C: B02 blue=2, B03 green=3, B04 red=4, B08 NIR=8, B11 SWIR1=12     (L1C includes B10)
Labels: 0 clear, 1 thick cloud, 2 thin cloud, 3 cloud shadow, 99 no data. Scale factor 0.0001.

    pip install tacoreader rasterio
    python inspect_cloudsen12.py                      # first, to see what the metadata looks like
    python convert_cloudsen12.py --out cloud_tiles_cs12 --limit 200 [--product l1c] [--stride 205] [--no-thin]
Reads remotely, one patch at a time (the full repository is about 248 GB; you do not need to download it all).
"""
import argparse, os, sys
import numpy as np
from tile_utils import TILE, cut_tiles, merge, save_npz

BAND_IDX = {"l2a": [2, 3, 4, 8, 11], "l1c": [2, 3, 4, 8, 12]}
PATCH = 2000
F = 3                                   # 10 m -> 30 m
SIDE = (PATCH // F) * F                 # 1998 px used, 666 px at 30 m


def patch_to_arrays(refl10, lab10, thin=True):
    """refl10: (5,2000,2000) float32 reflectance at 10 m, lab10: (2000,2000) CloudSEN12 labels.
    Returns refl30 (5,666,666) and cls30 (666,666) uint8 [0 clear, 1 cloud, 2 shadow, 255 invalid]."""
    refl10, lab10 = refl10[:, :SIDE, :SIDE], lab10[:SIDE, :SIDE]
    n = SIDE // F
    refl30 = refl10.reshape(5, n, F, n, F).mean((2, 4))
    def block(b):                       # fraction of 3x3 block that satisfies b
        return b.astype(np.float32).reshape(n, F, n, F).mean((1, 3))
    cloud = block((lab10 == 1) | ((lab10 == 2) & thin))
    shadow = block(lab10 == 3)
    invalid = block(lab10 == 99)
    cls = np.zeros((n, n), np.uint8)
    cls[shadow >= 0.5] = 2
    cls[cloud >= 0.5] = 1               # cloud wins over shadow
    cls[invalid > 0.5] = 255
    return refl30.astype(np.float32), cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cloud_tiles_cs12")
    ap.add_argument("--product", choices=["l2a", "l1c"], default="l2a")
    ap.add_argument("--limit", type=int, default=200, help="max number of patches to convert")
    ap.add_argument("--stride", type=int, default=TILE, help="256 = no overlap (4 tiles/patch); 205 = 9 overlapping tiles")
    ap.add_argument("--no-thin", dest="thin", action="store_false", help="do NOT count thin cloud as obscured")
    ap.add_argument("--skip-filter", action="store_true", help="scan every sample instead of using metadata columns")
    args = ap.parse_args()
    import tacoreader, rasterio as rio
    from rasterio.windows import Window
    os.makedirs(args.out, exist_ok=True)
    ds = tacoreader.load(f"tacofoundation:cloudsen12-{args.product}")
    idxs = range(len(ds))
    if not args.skip_filter:
        try:
            idxs = [i for i, ok in enumerate((ds["real_proj_shape"] == PATCH) & (ds["label_type"] == "high")) if ok]
            print(f"{len(idxs)} patches with real_proj_shape == 2000 and label_type == 'high'")
        except Exception as e:
            print("Could not filter by metadata columns (", e, ") -> scanning samples; run inspect_cloudsen12.py")
    done = 0
    for i in idxs:
        if done >= args.limit:
            break
        out = os.path.join(args.out, f"cs12_{args.product}_{i:06d}.npz")
        if os.path.exists(out):
            done += 1; continue
        try:
            s2p, lbp = ds.read(i).read(0), ds.read(i).read(1)
            with rio.open(lbp) as dst:
                if dst.width < 2000 or dst.height < 2000:
                    continue                                  # 509 px patch
                lab = dst.read(1, window=Window(0, 0, PATCH, PATCH))
            if (lab == 99).mean() > 0.2:
                continue                                      # scribble / no-label patch
            with rio.open(s2p) as src:
                refl = src.read(BAND_IDX[args.product], window=Window(0, 0, PATCH, PATCH)).astype(np.float32) * 1e-4
            r30, c30 = patch_to_arrays(refl, lab, args.thin)
            t = cut_tiles(r30, c30, stride=args.stride)
            if t is None:
                continue
            save_npz(out, t, f"cloudsen12_{args.product}", f"sample{i}")
            done += 1
            print(f"[{done}/{args.limit}] sample {i}: {len(t['frac'])} tiles, mean cloud frac {t['frac'].mean():.2f}", flush=True)
        except Exception as e:
            print(f"sample {i} failed: {e}")
    print("done:", done, "patches ->", args.out)


if __name__ == "__main__":
    main()
