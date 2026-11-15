"""Shared helpers: cut scenes into CloudScout tiles in the same .npz format that build_tiles.py writes.

Tile format (matches CloudScout/build_tiles.py):
    X            (n, 5, 256, 256) float16   reflectance 0-1, bands = blue, green, red, NIR, SWIR1
    frac         (n,)             float32   fraction of tile pixels that are cloud or cloud shadow
Extra arrays (ignored by the current train/eval scripts, used by the planned mask variant):
    nodata_frac  (n,) float32,  mask (n,256,256) uint8 [0 clear, 1 cloud, 2 shadow, 255 invalid]
    source, granule  (strings)
"""
import numpy as np

TILE = 256


def cut_tiles(refl, cls, tile=TILE, stride=None, max_bad=0.20):
    """refl: (5,H,W) float32 reflectance (NaN allowed). cls: (H,W) uint8 with
    0 clear, 1 cloud, 2 shadow, 255 invalid. Returns a dict of arrays, or None if no tile qualifies."""
    stride = stride or tile
    C, H, W = refl.shape
    assert C == 5 and cls.shape == (H, W), (refl.shape, cls.shape)
    valid = (cls != 255) & np.isfinite(refl).all(0)
    refl = np.clip(np.nan_to_num(refl, nan=0.0), 0.0, 1.0)
    obscured = ((cls == 1) | (cls == 2)) & valid
    X, frac, nod, M = [], [], [], []
    for r in range(0, H - tile + 1, stride):
        for c in range(0, W - tile + 1, stride):
            v = valid[r:r + tile, c:c + tile]
            bad = 1.0 - v.mean()
            if bad > max_bad:
                continue
            X.append(refl[:, r:r + tile, c:c + tile].astype(np.float16))
            frac.append(obscured[r:r + tile, c:c + tile].mean())   # same convention as build_tiles.py
            nod.append(bad)
            m = cls[r:r + tile, c:c + tile].copy()
            m[~v] = 255
            M.append(m)
    if not X:
        return None
    return dict(X=np.stack(X), frac=np.array(frac, np.float32),
                nodata_frac=np.array(nod, np.float32), mask=np.stack(M))


def subsample(tiles, k, rng):
    """Keep at most k tiles, chosen at random."""
    if tiles is None:
        return None
    n = len(tiles["frac"])
    if n <= k:
        return tiles
    idx = np.sort(rng.choice(n, k, replace=False))
    return {key: v[idx] for key, v in tiles.items()}


def merge(parts):
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def save_npz(path, tiles, source, granule):
    np.savez(path, **tiles, source=np.array(source), granule=np.array(granule))
