"""Build a compact training cache from the existing 256 px tiles (speed cut: 128 px inputs, <=48 tiles per scene).

  python build_cache.py

Re-reads Fmask for every scene to add per-pixel masks, reproducing build_tiles.py's tile order exactly (verified:
the recomputed cloud fraction must equal the stored `frac`). 2x2 mean-pools the reflectance to 128 px; the mask is a
2x2 vote (obscured if >= half the block is cloud/shadow, invalid if > half is nodata).
Output (cache128/): X (N,5,128,128) f16, frac (N,), mask (N,128,128) u8 [0 clear, 1 obscured, 255 invalid],
nodata (N,), region (N,) str, scene (N,) int.  Read-only on all inputs.
"""
import glob, os, re, sys, time
from pathlib import Path
import numpy as np, rasterio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diagnose import granule_table

TILE, PER_SCENE, SEED = 256, 48, 0
BANDS = ["B02", "B03", "B04", "B05", "B06"]
OUT = HERE / "cache128"; OUT.mkdir(exist_ok=True)


def rd(p):
    with rasterio.open(p) as ds:
        return ds.read(1)


def main():
    rng = np.random.default_rng(SEED)
    gt = granule_table()
    grs = sorted({re.sub(r"\.B02\.tif$", "", p) for p in glob.glob(str(HERE / "hls_raw" / "*.B02.tif"))})
    X, F, M, ND, RG, SC = [], [], [], [], [], []
    t0 = time.time(); done = 0
    files = sorted(glob.glob(str(HERE / "cloud_tiles" / "g*.npz")))
    for k, f in enumerate(files, 1):
        gi = int(re.search(r"g(\d{4})", f).group(1)); g = grs[gi]
        z = np.load(f); Xs, frac = z["X"], z["frac"]
        fm = rd(f"{g}.Fmask.tif")
        bad = (np.stack([rd(f"{g}.{b}.tif") for b in BANDS]) == -9999).any(0) | (fm == 255)
        obsc = (((fm & 0x02) | (fm & 0x08)) > 0) & ~bad
        H, W = bad.shape; keep = []
        for r in range(0, H - TILE + 1, TILE):
            for c in range(0, W - TILE + 1, TILE):
                if bad[r:r + TILE, c:c + TILE].mean() > 0.2: continue
                keep.append((r, c))
        assert len(keep) == len(frac), (f, len(keep), len(frac))
        re_frac = np.array([obsc[r:r + TILE, c:c + TILE].mean() for r, c in keep], np.float32)
        assert np.abs(re_frac - frac).max() < 1e-4, f"frac mismatch in {f}"
        sel = np.sort(rng.choice(len(keep), min(PER_SCENE, len(keep)), replace=False))
        for i in sel:
            r, c = keep[i]
            x = Xs[i].astype(np.float32).reshape(5, 128, 2, 128, 2).mean((2, 4))
            ob = obsc[r:r + TILE, c:c + TILE].reshape(128, 2, 128, 2).mean((1, 3))
            bd = bad[r:r + TILE, c:c + TILE].reshape(128, 2, 128, 2).mean((1, 3))
            m = (ob >= 0.5).astype(np.uint8); m[bd > 0.5] = 255
            X.append(x.astype(np.float16)); F.append(frac[i]); M.append(m); ND.append(bad[r:r + TILE, c:c + TILE].mean())
            reg = gt[gi]["region"].split(" ")[0]
            RG.append(reg); SC.append(gi)
        done += 1
        el = time.time() - t0
        print(f"[{k}/{len(files)}] {os.path.basename(f)} {len(sel)} tiles | {el/60:.1f} min | ETA {el/k*(len(files)-k)/60:.1f} min", flush=True)
    np.save(OUT / "X.npy", np.stack(X)); np.save(OUT / "frac.npy", np.array(F, np.float32)); np.save(OUT / "mask.npy", np.stack(M))
    np.save(OUT / "nodata.npy", np.array(ND, np.float32)); np.save(OUT / "region.npy", np.array(RG)); np.save(OUT / "scene.npy", np.array(SC))
    print("cache built:", len(F), "tiles;", {r: RG.count(r) for r in sorted(set(RG))})


if __name__ == "__main__":
    main()
