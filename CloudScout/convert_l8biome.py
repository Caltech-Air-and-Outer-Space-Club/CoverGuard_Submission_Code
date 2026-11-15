"""Convert the USGS L8 Biome cloud validation set (as packaged by TorchGeo/Hugging Face) into CloudScout tiles.

Expected layout after extracting the biome tar.gz files (one folder per scene):
    LC8PPPRRRYYYYDDDGSIVV.TIF            multiband GeoTIFF, bands B1..B11 (Level-1 digital numbers)
    LC8PPPRRRYYYYDDDGSIVV_fixedmask.TIF  manual mask: 0 fill, 64 shadow, 128 clear, 192 thin cloud, 255 cloud
    LC8PPPRRRYYYYDDDGSIVV_MTL.txt        Level-1 metadata (reflectance rescaling coefficients, sun elevation)

Download (also works with TorchGeo: L8Biome(root, download=True)):
    https://hf.co/datasets/torchgeo/l8biome/resolve/f76df19accce34d2acc1878d88b9491bc81f94c8/<biome>.tar.gz
    biomes: barren forest grass_crops shrubland snow_ice urban water wetlands

Usage:
    python convert_l8biome.py --root /path/to/l8biome --out cloud_tiles_l8biome [--max-tiles 150] [--no-thin]
"""
import argparse, glob, os, re, sys
import numpy as np
import rasterio
from rasterio.windows import Window
from tile_utils import TILE, cut_tiles, subsample, merge, save_npz

BANDS = [2, 3, 4, 5, 6]        # B2 blue, B3 green, B4 red, B5 NIR, B6 SWIR1 (1-based in the B1..B11 stack)
NAME = re.compile(r"^LC8\d{3}\d{3}\d{7}[A-Z]{3}\d{2}\.TIF$")
STRIP = 8 * TILE               # read the scene in strips so memory stays bounded


def read_mtl(path):
    d = {}
    for line in open(path, errors="ignore"):
        if "=" in line:
            k, v = [s.strip() for s in line.split("=", 1)]
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v.strip('"')
    return d


def to_toa(dn, mtl, band):
    """Landsat 8 Level-1 DN -> TOA reflectance with sun-angle correction (USGS formula)."""
    m = mtl[f"REFLECTANCE_MULT_BAND_{band}"]
    a = mtl[f"REFLECTANCE_ADD_BAND_{band}"]
    sun = np.sin(np.deg2rad(mtl["SUN_ELEVATION"]))
    return ((m * dn.astype(np.float32) + a) / sun).astype(np.float32)


def open_scene(base):
    """Return (read_bands(window)->(5,h,W) uint16, mask dataset, width, height, closers) for either layout."""
    mask_p = next((base + e for e in ("_fixedmask.TIF", "_fixedmask.img") if os.path.exists(base + e)), None)
    if mask_p is None:
        return None
    ms = rasterio.open(mask_p)
    if os.path.exists(base + ".TIF"):                                   # multiband layout
        ds = rasterio.open(base + ".TIF")
        read = lambda win: ds.read(BANDS, window=win)
        return read, ms, ds.width, ds.height, [ds, ms]
    paths = [base + f"_B{b}.TIF" for b in BANDS]                        # native USGS layout: one file per band
    if not all(os.path.exists(p) for p in paths):
        ms.close(); return None
    dss = [rasterio.open(p) for p in paths]
    read = lambda win: np.stack([d.read(1, window=win) for d in dss])
    return read, ms, dss[0].width, dss[0].height, dss + [ms]


def convert_scene(img, args, rng):
    base = img[:-4]
    mtl_p = base + "_MTL.txt"
    sc = open_scene(base) if os.path.exists(mtl_p) else None
    if sc is None:
        print("  skip (missing mask, bands or MTL):", os.path.basename(img)); return None
    read, ms, W, H, closers = sc
    mtl = read_mtl(mtl_p)
    parts = []
    try:
        if (ms.width, ms.height) != (W, H):
            print("  skip (image/mask size differ):", os.path.basename(img)); return None
        n_strips = (H + STRIP - 1) // STRIP
        per_strip = max(1, int(np.ceil(args.max_tiles / n_strips)))
        for r0 in range(0, H, STRIP):
            h = min(STRIP, H - r0)
            win = Window(0, r0, W, h)
            dn = read(win)                                   # (5,h,W) uint16
            raw = ms.read(1, window=win)                     # uint8
            # The TorchGeo/HF packaging stores 8-bit bands (assumed = 16-bit DN >> 8; plausibility checked via NDVI/snow SWIR,
            # not against ground truth). Rescale to 16-bit DN midpoints before the USGS formula. Native USGS files are uint16.
            dn_f = dn.astype(np.float32) * 256 + 128 if dn.dtype == np.uint8 else dn
            refl = np.stack([to_toa(dn_f[i], mtl, b) for i, b in enumerate(BANDS)])
            cls = np.zeros(raw.shape, np.uint8)
            cls[raw == 64] = 2                               # shadow
            cls[raw == 192] = 1 if args.thin else 0          # thin cloud
            cls[raw == 255] = 1                              # cloud
            cls[(raw == 0) | (dn == 0).any(0)] = 255         # fill / no data
            parts.append(subsample(cut_tiles(refl, cls), per_strip, rng))
    finally:
        for c in closers: c.close()
    return merge(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="cloud_tiles_l8biome")
    ap.add_argument("--max-tiles", type=int, default=150, help="cap per scene (a full scene gives ~900 tiles, ~600 MB)")
    ap.add_argument("--no-thin", dest="thin", action="store_false", help="do NOT count thin cloud as obscured")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    # scenes are located by their MTL file, so both the TorchGeo layout (multiband <scene>.TIF) and the native USGS
    # layout (<scene>_B2.TIF ... plus ENVI <scene>_fixedmask.img) are found.
    imgs = [p[:-8] + ".TIF" for p in sorted(glob.glob(os.path.join(args.root, "**", "LC8*_MTL.txt"), recursive=True))
            if NAME.match(os.path.basename(p)[:-8] + ".TIF")]
    print(f"Found {len(imgs)} L8 Biome scenes under {args.root}; thin cloud counted as obscured: {args.thin}")
    if not imgs:
        sys.exit("No scenes found. Extract the biome tar.gz files first.")
    for i, img in enumerate(imgs, 1):
        name = os.path.basename(img)[:-4]
        out = os.path.join(args.out, f"l8b_{name}.npz")
        if os.path.exists(out):
            print(f"[{i}/{len(imgs)}] exists, skipping {name}"); continue
        t = convert_scene(img, args, rng)
        if t is None:
            continue
        save_npz(out, t, "l8biome_toa", name)
        print(f"[{i}/{len(imgs)}] {name}: {len(t['frac'])} tiles, mean cloud frac {t['frac'].mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
