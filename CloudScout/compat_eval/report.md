# Data compatibility audit: new sources vs. the CloudScout HLS training set

Branch `data-compat-eval`, 2026-11-11. Nothing was trained; existing tiles, weights and LightGBM files were not modified.
Reference weights: `trained_model/cloudscout.pt`. All numbers at threshold 0.5 unless stated.

## Verdicts

| Source | Verdict | Main reason |
|---|---|---|
| A. More HLSL30 scenes | **NOT TESTED** | No Earthdata credentials (`~/.netrc` missing), so the downloader could not run. See below. |
| B. USGS L8 Biome (25 scenes: 12 snow_ice, 12 grass_crops, 1 barren) | **NOT COMPATIBLE for pooling.** Usable only as a stress/validation set. | Domain-classifier AUC 0.999; top-of-atmosphere vs surface reflectance; thin-cloud handling swings labels and metrics strongly; the 8-bit packaging needs an unverified rescale assumption. |
| C. CloudSEN12+ (60 patches, 240 tiles) | **COMPATIBLE WITH CAVEATS** (validation / fine-tuning with a source flag; do not mix blindly into pretraining) | Domain AUC 0.893 (flag band 0.70–0.90, computed on only 88 clear tiles); model AUC 0.74; 35% of tiles contain >5% shadow. |

An important context point: the **reference itself is weak**. On held-out HLS scenes the current weights reach ROC AUC 0.840 (below the 0.85 pass mark) and miss 73% of cloudy tiles at a 2% clear-discard rate. "Within 2x of the HLS reference" is therefore an easy bar and says little. Conclusions about the new sources are confounded with the weakness of the model.

## Evidence table

| Check | HLS held-out (reference) | L8 Biome (thin = cloud) | L8 Biome (no thin) | CloudSEN12+ |
|---|---|---|---|---|
| Schema (Step 1) | 221/221 pass | 25/25 pass | 25/25 pass | 60/60 pass |
| Tiles / cloudy share at 0.70 | 5,865 / 35.7% | 3,571 / 39.1% | 3,571 / 22.5% | 240 / 32.9% |
| Majority baseline accuracy | 0.643 | 0.609 | 0.775 | 0.671 |
| Accuracy @0.5 | 0.778 | 0.657 | **0.502** (below baseline) | 0.708 |
| Clear-tile discard @0.5 / cloud miss @0.5 | 19.6% / 26.7% | 55.3% / 1.6% | 64.1% / 0.2% | 34.2% / 19.0% |
| ROC AUC | 0.840 | 0.787 | 0.694 | 0.739 |
| Miss at 2% discard | 72.8% | 79.5% | 93.3% | 87.3% |
| Domain-classifier AUC vs HLS (clear tiles, grouped by scene) | n/a | **0.999** | not run | **0.893** (88 clear tiles) |
| Clear-tile median shift / pooled IQR (blue, green, red, NIR, SWIR1) | n/a | 8.9, 5.7, 3.6, 2.7, -0.2 | n/a | 1.27, 0.96, 0.63, 0.39, -0.07 |
| Clear-tile NDVI median / share > 0 | 0.30 / 0.89 | 0.03 / 0.67 | n/a | 0.08 / 0.81 |
| Tiles with > 5% shadow | not stored | 11.6% | n/a | 34.6% |
| Share of tiles with frac in 30–90% | 15.6% | 14.3% | n/a | 27.9% |

Source files: `hls_heldout_metrics.json`, `l8biome_metrics.json`, `l8biome_nothin_metrics.json`, `cs12_metrics.json`, `analysis_summary.json`, `domain_*.txt`. Plots: `band_histograms_clear_tiles.png`, `frac_histograms.png`, `contact_{hls,l8biome,cs12}.png`.
The by-cloud-fraction tables are in each metrics JSON.

Decision-rule outcomes: Schema passes everywhere. Median-shift/IQR exceeds 0.5 in blue, green and red for both new sources, and in NIR for L8. Domain AUC is "do not pool" for L8 (>0.90) and "flag" for CloudSEN12+ (0.70–0.90). Model AUC is below 0.85 on every set, including the reference.

## Source B: L8 Biome

- **Radiometry.** Top-of-atmosphere reflectance, computed with the MTL coefficients. This differs from HLS surface reflectance. It is closer to what an onboard sensor measures, but it is not the distribution the model trained on.
- **The set is snow-heavy.** Half the scenes come from `snow_ice`, so "clear" tiles are often bright snow (clear median 0.6–0.7 in visible bands). The huge domain AUC and shift/IQR therefore mix a radiometric difference with scene content; they cannot be separated with this sample. The contact sheet shows tiles labeled frac = 0.00 that are entirely white and scored p ≈ 1.0 by the model: the current weights read snow as cloud (the known weakness listed in the handoff), which explains the 55% clear-tile discard rate.
- **Thin cloud changes everything.** Counting thin cloud as obscured gives 39% cloudy tiles and AUC 0.79; not counting it gives 22.5% cloudy and AUC 0.69 (accuracy 0.50 vs a 0.775 baseline). The label definition is not settled for this source.
- **Sharpness** (mean absolute Laplacian normalised by mean red): 0.078 vs 0.298 for HLS. The difference is probably scene content (smooth snow and cloud) and not a sensor property; I did not isolate it.
- **Label noise.** Several tiles with intermediate frac look hazy rather than clear or cloudy; one frac = 0.06 desert tile was scored p = 1.00 (haze). This is inherent to hand labels with a different standard from Fmask.

## Source C: CloudSEN12+

- **Small sample.** 60 patches, 240 tiles, only 88 clear tiles in the domain test. Every number here has a wide interval; I computed none.
- **Spectral shift is moderate** (shift/IQR 0.4–1.3), and the 3x3 averaging leaves the tile about as sharp as HLS by the Laplacian measure (0.266 vs 0.298).
- **Labels differ in kind.** 35% of tiles contain more than 5% shadow, against none counted separately in HLS. The sheet shows frac = 1.00 tiles that look like clear farmland (scored p ≈ 0.00) and white snow tiles with frac ≈ 0 scored p = 1.0, again the snow confusion.
- Thin cloud no-thin regeneration was **not run** for this source.

## Source A: not tested

`download_hls_regions.py` needs Earthdata credentials, which are not on this machine. The existing set already covers 12 regions (`hls_raw/scene_index.csv`, about 12 scenes each), so a good part of what this source is meant to add (varied regions, snow, cloud-heavy scenes) is already present. HLSL30 is the same product, so format and semantics should match by construction, but I did not verify that, nor compute any distribution check. A dry run (`--dry-run`) after you run `earthaccess.login()` once would show the granule count at no download cost.

## Step 6: loader compatibility

`eval_folder.py` reads `X` and `frac` exactly as `evaluate_gate.py` does and worked on all three folders with no exceptions or dtype problems. I did not run the actual training script (`train_cloudscout.py` was out of scope). RAM if everything were loaded at once (float16, 0.66 MB per tile): the existing 31,838 HLS tiles are already about **20.9 GB**, more than this machine's 15 GB, so the original all-in-memory loading cannot hold the current set. Adding L8 Biome (3,571 tiles) adds 2.3 GB and CloudSEN12+ (240 tiles, with masks) about 0.17 GB. Training must load in chunks (`train_cloudscout2.py` already does).

## Step 7: leakage and overlap

- L8 Biome scenes are from 2013–2014 and sit on WRS-2 paths/rows like 001/011 and 042/008; none is on path 43 / row 35 (my estimate of the Central Valley scene, not verified against the AOI), and the HLS set is from 2022 onward. I found no overlap.
- CloudSEN12+ tiles do not store coordinates and I did not check the dataset centroids against the HLS regions, so overlap is **unchecked** for that source.

## Bugs found in the tools

1. `convert_l8biome.py` assumed one layout (multiband `<scene>.TIF` plus `_fixedmask.TIF`). The USGS-native download in `~/downloads` (per-band TIFs plus an ENVI `.img` mask) failed. I added support for both layouts.
2. The Hugging Face/TorchGeo packaging stores **8-bit bands**, not 16-bit digital numbers. The original conversion produced all-zero tiles for 24 of 25 scenes, and `check_schema.py` still passed them (all zeros is "valid"). I rescale 8-bit values by ×256 + 128 before the USGS formula. This is an assumption: it gives plausible results (crop NDVI 0.31–0.43, snow SWIR well below visible), but I could not check it against ground truth, and 8-bit quantization costs roughly 0.01 reflectance resolution.
3. `tacoreader` 2.x cannot read `tacofoundation:` datasets; `pip install 'tacoreader<1.0'` was needed (works on Python 3.14).
4. `check_schema.py` has no check for all-zero tiles. I did not change it; adding one is recommended.

## What to do next (ranked by value)

1. Do not pool L8 Biome into pretraining. Use its 25 scenes as an out-of-domain stress test, and especially the snow tiles: that is a real, repeatable failure of the current model.
2. Fix the weak baseline before judging data. AUC 0.84 on HLS means the model is not separating the classes well; Phase 0 of the architecture plan comes first.
3. Decide the thin-cloud policy explicitly (it moves L8 AUC by 0.09) and check CloudSEN12+ the same way.
4. Scale CloudSEN12+ to several hundred patches (847 usable exist) so the domain test and AUC have a meaningful sample; use a source indicator or fine-tune stage.
5. Provide Earthdata credentials if you still want Source A; given the existing 12-region set, its marginal value is lower than 3–4.
6. Add an all-zero check to `check_schema.py`.

## Commands used

```
# run from CloudScout/
python check_schema.py cloud_tiles cloud_tiles_l8biome cloud_tiles_l8biome_nothin cloud_tiles_cs12
python convert_l8biome.py --root l8biome_raw/{snow_ice,grass_crops,barren_local} --out cloud_tiles_l8biome [--no-thin → cloud_tiles_l8biome_nothin]
python convert_cloudsen12.py --out cloud_tiles_cs12 --limit 60
python eval_folder.py --tiles <folder> --weights trained_model/cloudscout.pt --out compat_eval/<name>_metrics.json
python domain_check.py cloud_tiles <cloud_tiles_l8biome | cloud_tiles_cs12>
python compat_eval/analyze.py
```
HLS reference = the last 45 of 221 scene files (symlinks in `compat_eval/hls_heldout/`), the same split rule as `evaluate_gate.py`; I could not confirm the checkpoint was not trained on those files.
