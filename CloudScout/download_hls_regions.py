"""More HLSL30 scenes for CloudScout: several regions, seasons and cloud-cover buckets, resumable, with progress.

Output files match download_hls.py exactly (hls_raw/*.B02.tif ... *.Fmask.tif), so build_tiles.py works unchanged.
HLSL30 only: do NOT add HLSS30 (Sentinel-2) here. Its band numbers differ (B05 is red edge, NIR is B8A),
so build_tiles.py's B02..B06 selection would silently pick the wrong bands.

    pip install earthaccess
    python download_hls_regions.py --dry-run                       # count granules, download nothing
    python download_hls_regions.py --regions central_valley_winter midwest_snow --years 2023 2024

Bounding boxes are APPROXIMATE (lon_min, lat_min, lon_max, lat_max); check them on a map and adjust.
Each region/month is searched twice: a low-cloud bucket and a cloudy bucket, so the training set is not all clear scenes.
"""
import argparse, os, time
import earthaccess

REGIONS = {
    "central_valley_winter": dict(bbox=(-120.75, 36.50, -120.45, 36.80), months=[11, 12, 1, 2]),   # tule fog season
    "salinas_marine_layer":  dict(bbox=(-121.80, 36.55, -121.50, 36.85), months=[5, 6, 7, 8, 9]),     # coastal fog/stratus
    "willamette_pnw":        dict(bbox=(-123.20, 44.80, -122.90, 45.10), months=[10, 11, 12, 1, 2, 3]),  # persistently cloudy
    "midwest_snow":          dict(bbox=(-88.40, 40.00, -88.10, 40.30), months=[12, 1, 2, 3]),         # snow vs cloud
    "imperial_valley_clear": dict(bbox=(-115.65, 32.80, -115.35, 33.10), months=list(range(1, 13))),  # bright bare soil, few clouds
    "mississippi_delta":     dict(bbox=(-90.80, 33.40, -90.50, 33.70), months=list(range(1, 13))),
}
WANT = ("B02", "B03", "B04", "B05", "B06", "Fmask")      # blue, green, red, NIR, SWIR1 (HLSL30) + quality bits
SUFFIXES = tuple(f".{b}.tif" for b in WANT)
BUCKETS = [("low cloud", (0, 20)), ("cloudy", (30, 100))]


def month_range(y, m):
    return f"{y}-{m:02d}-01", f"{y + (m == 12)}-{(m % 12) + 1:02d}-01"


def search(bbox, start, end, per_bucket, cc):
    kw = dict(short_name="HLSL30", version="2.0", bounding_box=bbox, temporal=(start, end), count=per_bucket)
    try:
        return earthaccess.search_data(cloud_cover=cc, **kw)
    except Exception as e:                       # cloud_cover filter unavailable -> unfiltered search
        print(f"    (cloud_cover filter failed: {e}; searching without it)")
        return earthaccess.search_data(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=list(REGIONS), choices=list(REGIONS))
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    ap.add_argument("--per-bucket", type=int, default=1)
    ap.add_argument("--out", default="hls_raw")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    earthaccess.login()
    urls = []
    for reg in a.regions:
        cfg = REGIONS[reg]
        for y in a.years:
            for m in cfg["months"]:
                start, end = month_range(y, m)
                for label, cc in BUCKETS:
                    res = search(cfg["bbox"], start, end, a.per_bucket, cc)
                    u = [x for g in res for x in g.data_links() if x.endswith(SUFFIXES)]
                    urls += u
                    print(f"[search] {reg} {y}-{m:02d} {label}: {len(res)} granules, {len(u)} files")
    urls = sorted(set(urls))
    todo = [u for u in urls if not os.path.exists(os.path.join(a.out, os.path.basename(u)))]
    print(f"\nTotal files {len(urls)}; already on disk {len(urls) - len(todo)}; to download {len(todo)}")
    if a.dry_run:
        print("dry run: nothing downloaded"); return
    t0, got, failed = time.time(), 0, []
    for i, u in enumerate(todo, 1):
        t1 = time.time()
        try:
            paths = earthaccess.download([u], a.out)
            got += sum(os.path.getsize(p) for p in paths if os.path.exists(p))
        except Exception as e:
            print(f"[{i}/{len(todo)}] FAILED {os.path.basename(u)}: {e}"); failed.append(u); continue
        el = time.time() - t0
        print(f"[{i}/{len(todo)}] {os.path.basename(u)} {time.time() - t1:.1f}s | {got / 1e9:.2f} GB | "
              f"elapsed {el / 60:.1f} min | ETA {el / i * (len(todo) - i) / 60:.1f} min", flush=True)
    print(f"Done. {len(todo) - len(failed)} files, {got / 1e9:.2f} GB. Failed: {len(failed)} (re-run to retry)")


if __name__ == "__main__":
    main()
