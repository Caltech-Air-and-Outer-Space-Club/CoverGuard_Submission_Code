"""Format check: does every .npz in a folder match the CloudScout tile schema?  python check_schema.py <folder> [...]"""
import glob, os, sys
import numpy as np

def check(path):
    errs = []
    z = np.load(path)
    for k in ("X", "frac"):
        if k not in z.files:
            errs.append(f"missing array '{k}'")
    if errs:
        return errs
    X, f = z["X"], z["frac"]
    if X.ndim != 4 or X.shape[1:] != (5, 256, 256): errs.append(f"X shape {X.shape}, expected (n,5,256,256)")
    if X.dtype != np.float16: errs.append(f"X dtype {X.dtype}, expected float16")
    if f.dtype != np.float32 or f.shape != (len(X),): errs.append(f"frac dtype/shape {f.dtype} {f.shape}")
    Xf = X.astype(np.float32)
    if not np.isfinite(Xf).all(): errs.append("X has NaN/inf")
    if Xf.min() < 0 or Xf.max() > 1: errs.append(f"X outside [0,1]: {Xf.min():.3f}..{Xf.max():.3f}")
    if len(f) and (f.min() < 0 or f.max() > 1): errs.append("frac outside [0,1]")
    if "mask" in z.files:
        m = z["mask"]
        if m.shape != (len(X), 256, 256) or m.dtype != np.uint8: errs.append(f"mask shape/dtype {m.shape} {m.dtype}")
        else:
            implied = ((m == 1) | (m == 2)).mean((1, 2))
            if np.abs(implied - f).max() > 1e-3: errs.append("frac disagrees with mask")
    if "nodata_frac" in z.files and z["nodata_frac"].max() > 0.2 + 1e-6: errs.append("nodata_frac > 0.2")
    return errs

if __name__ == "__main__":
    for folder in sys.argv[1:]:
        files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        bad = 0
        for p in files:
            e = check(p)
            if e:
                bad += 1; print(f"FAIL {os.path.basename(p)}: {'; '.join(e)}")
        print(f"{folder}: {len(files)} files, {bad} failed")
