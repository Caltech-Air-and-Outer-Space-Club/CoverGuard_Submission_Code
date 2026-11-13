"""Step 1 for CloudSEN12+: look at the dataset metadata before converting (nothing is downloaded in bulk).

    pip install tacoreader rasterio
    python inspect_cloudsen12.py

Prints the metadata columns and how many samples have real_proj_shape 2000 and label_type 'high'.
Only 2000x2000 patches (about 20 km at 10 m) are large enough to give 256x256 tiles at 30 m;
509x509 patches (about 170 px at 30 m) are too small.
"""
import tacoreader

ds = tacoreader.load("tacofoundation:cloudsen12-l2a")
print("samples:", len(ds))
print("columns:", list(ds.columns))
for col in ("real_proj_shape", "label_type"):
    if col in ds.columns:
        print(f"\n{col}:\n{ds[col].value_counts()}")
if "real_proj_shape" in ds.columns and "label_type" in ds.columns:
    sel = ds[(ds["real_proj_shape"] == 2000) & (ds["label_type"] == "high")]
    print("\n2000 px + high-quality labels:", len(sel))
