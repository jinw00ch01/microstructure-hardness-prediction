# feature-engineer handoff (updated 2026-10-06)

Owner files: `src/features.py`, `src/train_gbm.py`. I don't commit. Best feature model:
**feat2_v23cal_lgbs_het, CV 12.991** (baseline feat_lgb 14.202). A 50/50 average with
feat2_v3cal_ridge_het gives CV 12.87 (error correlation 0.946).

## Feature files (data/ is git-ignored; back these up to the cache)
| file | rows x cols | content | regenerate (1 core) | time |
|---|---|---|---|---|
| `data/features.parquet` | 1500x104 | v1 (unchanged) | `OMP_NUM_THREADS=1 python -m src.features --n_jobs 1` | ~8 min |
| `data/features_v2.parquet` | 1500x179 | v2: quality (`q_*`), levels `lv_*`, pores `pore*`, pixel dark fractions `ph_*`, dark blobs `dk_*`, sato-ridge + watershed grains `seg_*` (count density, area stats, aspect, orientation R, per-grain dark fraction `seg_fd*`), `segdk_*`, intercepts `seg_L_*`, autocorrelation `ac*`, structure tensor `st*` | `OMP_NUM_THREADS=1 python -W ignore -m src.features --v2 --n_jobs 1` | ~13 min |
| `data/features_v3.parquet` | 1500x135 | v3 `ic_*`: image divided by local matrix level (70th-pct filter, ~80 px). Pores, dark fraction at many thresholds, 2-GMM, dark chords / autocorrelation lengths along and across the elongation axis, per-grain classification `ic_seg_fd*`, plus 6 quality columns `ic_noise`, `ic_spec_*` (added 2026-10-06 by `--patch_v3_quality`; the other 128 columns are unchanged) | `OMP_NUM_THREADS=1 python -W ignore -m src.features --v3 --n_jobs 1` (includes the quality columns) | ~13-15 min |
| `data/cal_pairs.parquet` | 1392x141 | v3 features of 174 clean train images (ic_ridge_snr > 0.75) x 8 synthetic degradations (blur 0-2.5 on grains with crisp pores, shading, contrast, noise 0-18) | `OMP_NUM_THREADS=1 python -W ignore -m src.features --cal_build --n_aug 8 --snr_min 0.75 --n_jobs 1` | ~8 min |
| `data/features_cal.parquet` | 1500x30 | `cal_*`: LightGBM maps (fitted on cal_pairs only; no labels, no test images) from degraded v3 features to clean-image measurements | `OMP_NUM_THREADS=1 python -W ignore -m src.features --cal_apply` (prints the held-out-source R2 = cal_eval report) | ~2 min |

Calibration held-out-source R2: per-grain dark fraction `ic_seg_fd91` 0.94 (raw degraded features r=0.88),
`ic_st_coh` 0.94, `ic_fdo_93` 0.92, `ic_acg_len50_par` 0.82, `seg_count_density` 0.80, `seg_ori_R` 0.81,
`seg_inv_sqrt_d` 0.77. Failed (R2 < 0.3; always pass the --drop below): `ic_seg_L_*`, `ic_seg_mx_area_mean`,
`seg_area_cv`, `segdk_area_cv`.
`DROP="cal_ic_seg_L_,cal_ic_seg_mx_area_mean,cal_seg_area_cv,cal_segdk_area_cv"`

## Experiments (5 shared folds; no early stopping on the validation fold)
| exp | CV RMSE | folds |
|---|---|---|
| **feat2_v23cal_lgbs_het** (v3+v2+cal, lgbs, 3 seeds, hetero) | **12.991** | 13.35 12.91 13.31 12.80 12.57 |
| feat2_v3cal_ridge_het (v3+cal, ridge, hetero) | 13.101 | 13.80 12.76 13.22 13.00 12.70 |
| feat2_v3cal_ridge (v3+cal) | 13.177 | 13.71 13.05 13.31 13.04 12.76 |
| feat2_v23_lgbs (v3+v2, 3 seeds) | 13.212 | 13.38 13.37 13.38 13.10 12.82 |
| feat2_v3_ridge | 13.219 | 13.61 12.92 13.46 13.14 12.94 |
| feat2_v3cal_lgbs | 13.346 | 13.45 13.34 13.56 13.13 13.25 |
| feat2_v23_lgb | 13.530 | 13.68 13.50 13.92 13.54 13.00 |
| feat2_v23_ridge | 13.546 | 13.81 12.88 14.98 13.22 12.72 |
| feat2_v3_lgb | 13.612 | 13.84 13.54 13.85 13.38 13.44 |
| feat2_v3_fwd / feat2_v123_fwd | 13.721 / 14.165 | in-fold forward selection overfits |
| feat2_ridge / feat2_lgb_v1v2 / feat2_lgb (v2) | 14.055 / 14.127 / 14.226 | |

Exact commands for the saved 2026-10-06 runs (core 0):
```
taskset -c 0 python -W ignore -m src.train_gbm --feat features_v3.parquet,features_cal.parquet --model ridge --drop $DROP --name feat2_v3cal_ridge
taskset -c 0 python -W ignore -m src.train_gbm --feat features_v3.parquet,features_cal.parquet --model lgbs --drop $DROP --name feat2_v3cal_lgbs
taskset -c 0 python -W ignore -m src.train_gbm --feat features_v3.parquet,features_v2.parquet --model lgbs --seeds 3 --name feat2_v23_lgbs
taskset -c 0 python -W ignore -m src.train_gbm --feat features_v3.parquet,features_cal.parquet --model ridge --drop $DROP --hetero ic_acg_len50_gm --name feat2_v3cal_ridge_het
taskset -c 0 python -W ignore -m src.train_gbm --feat features_v3.parquet,features_v2.parquet,features_cal.parquet --model lgbs --seeds 3 --drop $DROP --hetero ic_acg_len50_gm --name feat2_v23cal_lgbs_het
```
`--hetero COL[,COL]`: in each fold, inner-CV ridge residuals on training rows give log(r^2+1) ~ log(COL);
the sample weight is 1/fitted variance (mean 1, clipped 0.25-4). The fitted slope is about +1.07, so variance
is roughly proportional to the correlation length. `--no_save` = screening run that writes nothing.

Screens (not saved): monotone (+) constraints on fd features in LGB: no change (13.347 vs 13.346).
Hetero with noise as a 2nd variance column: worse (13.15 vs 13.10). CatBoost + hetero on v3+v2+cal: 13.52.
Degraded copies as extra hardness training rows (labels of their source; only in-fold sources): ridge
13.21 -> 13.15 at weight 0.125, worse at higher weights; lgbs worse. OOF calibration slope is 0.96-1.01, so
stretching predictions hurts (+0.06-0.08).

## What mattered
1. Dark-phase fraction dominates: per-grain fd on clean images Spearman 0.70-0.72. Illumination correction
   raised overall Spearman 0.52 -> 0.61; calibration -> 0.65 (`cal_ic_seg_fd93`; worst-quality quartile 0.46 -> 0.50).
2. Correlation length along the elongation axis: partial rho +0.35 after fd (coarser/longer = harder). The
   calibrated version correlates 0.44 with hardness vs 0.27 raw. Grain count density: partial -0.28.
3. Pores: v1 z-score pore features were confounded with dark fraction; proper pore measures add little.
4. Heteroscedastic target: noise variance grows about linearly with correlation length. Weighting by it gains
   0.08-0.13 for both ridge and LGB. Coarse-tercile RMSE stays about 16.8 (fine 9.0, mid 12.1).

## Next steps (priority order)
1. Re-blend with the new OOFs (`feat2_v23cal_lgbs_het`, `feat2_v3cal_ridge_het`). Feature models alone
   probably top out around 12.8-12.9.
2. Give the embedding/CNN heads hetero sample weights (same variance model on `ic_acg_len50_gm`) and
   `features_cal.parquet` as side features; the CNN loss can use the same weights.
3. A calibration v2 could help noisy images more: more degradations (`--n_aug 16`), matching the real noise/blur
   distribution (estimate blur on real images), and fitting calibration maps per noise band.
4. Coarse images carry most of the error and look irreducible from image statistics. A remaining idea is a
   multi-crop/patch-level CNN to check whether any local signal exists.
