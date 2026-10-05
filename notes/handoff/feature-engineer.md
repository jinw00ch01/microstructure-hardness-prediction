# feature-engineer handoff (paused 2026-10-05)

Owner files: `src/features.py`, `src/train_gbm.py`. Nothing committed by me. Best feature model:
**feat2_v3_ridge, CV 13.219** (baseline feat_lgb 14.202).

## Feature files
| file | rows x cols | content | regenerate (1 core) | time |
|---|---|---|---|---|
| `data/features.parquet` | 1500x104 | v1 (unchanged) | `OMP_NUM_THREADS=1 python -m src.features --n_jobs 1` | ~8 min |
| `data/features_v2.parquet` | 1500x179 | v2: quality (`q_*`: noise, spectral bands/slopes, ridge SNR), global levels `lv_*` rel. to matrix mode, pores at 0.70/0.60/0.50 x matrix level (`pore*`: n, round/elong, frac, size, depth), pixel dark fractions `ph_*`, dark blobs `dk_*`, ridge(sato)+h-minima watershed grain segmentation `seg_*` (count density, area stats, aspect, orientation R, per-grain dark fraction `seg_fd12/15/20`), dark-grain shape `segdk_*`, linear intercepts `seg_L_*`, autocorrelation ellipse `ac*`/`acp_*`, structure tensor `st*` | `OMP_NUM_THREADS=1 python -W ignore -m src.features --v2 --n_jobs 1` | ~13 min (0.5 s/img) |
| `data/features_v3.parquet` | 1500x129 | v3 `ic_*` = illumination-corrected: image divided by local matrix level (70th-pct filter, ~80 px window). Pores (`ic_pore68/60/52_*`), dark fraction at many thresholds (`ic_fd2_*`, opened `ic_fdo_*`, soft, 2-GMM `ic_gmm_*`), dark-mask chord lengths along/across elongation axis (`ic_dk_chord_*`), dark blob shapes, grey/binary autocorrelation lengths along/across axis normalised at lag 1 (`ic_acg_*`, `ic_acb_*`), per-grain classification on corrected image (`ic_seg_fd93/91/89/87`), intercepts along/across axis (`ic_seg_L_*`) | `OMP_NUM_THREADS=1 python -W ignore -m src.features --v3 --n_jobs 1` | ~13-15 min (0.5 s/img) |

All transforms are per-image; nothing is fitted on test images.

## Experiments (all in experiments/, 5 shared folds, no early stopping on val fold)
| exp | CV RMSE | folds |
|---|---|---|
| feat2_lgb (v2) | 14.226 | 14.84 13.54 14.41 14.46 13.84 |
| feat2_ridge (v2) | 14.055 | 14.29 13.10 15.81 13.57 13.34 |
| feat2_lgb_v1v2 | 14.127 | 14.66 13.49 14.55 14.03 13.88 |
| feat2_v3_lgb | 13.612 | 13.84 13.54 13.85 13.38 13.44 |
| **feat2_v3_ridge** | **13.219** | 13.61 12.92 13.46 13.14 12.94 |
| feat2_v23_lgb (v3+v2) | 13.530 | 13.68 13.50 13.92 13.54 13.00 |
| feat2_v23_ridge | 13.546 | 13.81 12.88 14.98 13.22 12.72 |
| feat2_v3_fwd (in-fold fwd selection) | 13.721 | overfits inner CV |
| feat2_v123_fwd | 14.165 | overfits inner CV |

Ad-hoc (not saved): `lgbs` params (leaves 4, lr 0.01, 1500 trees, colsample 0.3) v2+v3 13.21, v3 13.45;
CatBoost depth 4 v3 13.43 / v2+v3 13.40; log target 13.28; fd x length interactions <=0.04 gain.
Note: baseline feat_lgb (14.20) used early stopping on the validation fold; `--model lgb` now uses fixed
700 trees (`--es` restores the old behaviour).

## What mattered
1. Dark-phase fraction dominates. Per-grain fd on clean images Spearman 0.70-0.72. Illumination correction
   raised overall fd Spearman 0.52 -> 0.61 (`ic_fdo_93`, `ic_gmm_w`, `ic_fd2_93`, `ic_seg_fd93`).
2. Next: correlation length along the elongation axis (`ic_acg_len50_par`, `ac50_major`), partial rho +0.35
   after fd: coarser/longer structures are harder. Grain count density partial -0.28 (anti Hall-Petch
   globally; within fine-grained images Hall-Petch shows weakly, +0.2 partial).
3. v1 pore/z features were confounded with dark fraction (IQR normalisation; corr -0.80 with v1 phase frac on
   clean images). Proper pore measures only add a little (partial rho -0.1..-0.2). Pores stay crisp even on
   blurred images, so they are reliably measured everywhere.
4. Weaker: dark-grain size spread (`segdk_area_cv`), structure-tensor coherence, grain area CV, tiny specks.

## Key insights / limits
- Heteroscedastic target: hardness std is 21.5 in the coarse-structure tercile vs 13.5 in the fine one; OOF
  RMSE 16.7 vs 9.3 (clean few-grain images 15.6 vs many-grain 8.8). Nothing image-based explained the extra
  variance (center-window features, per-grain Hall-Petch functionals, count vs area fd, interactions).
  This looks like noise built into the target, which caps feature models at ~12-13.
- Noisy images (top half by noise) have OOF RMSE 14.2-15.4 vs 11.8-12.2. fd Spearman in the worst-quality
  quartile is 0.46 vs ~0.69 elsewhere. Denoiser choice (NLM 1.0/1.4 sigma, Gaussian, bilateral; scale 2/3)
  makes no difference (0.50-0.57). Noise is white additive (flat spectral floor = estimate_sigma).
- Shading (illumination non-uniformity of 13-19 grey levels) is comparable to the matrix/dark contrast
  (~30 levels), so global thresholds fail on noisy images. Always normalise by the local matrix level.

## In progress when paused
Label-free measurement calibration (code is in `src/features.py`: `_degrade`, `cal_build`, `cal_apply`).
Clean train images (ic_ridge_snr > 0.9, 131) get degraded 8x (blur 0-2.5 on the grain layer with pores
pasted back crisp, shading, contrast jitter, noise 0-18). v3 features are recomputed and LightGBM maps
degraded features to the clean-image measurements. No labels are used and no test images are used in fitting.
The build was killed before writing `data/cal_pairs.parquet`; nothing from it is on disk.

## Next steps (priority order)
1. Calibration (about 25 min on 1 loaded core, then about 2 min):
   `OMP_NUM_THREADS=1 nohup python -W ignore -m src.features --cal_build --n_aug 8 --n_jobs 1 > logs/cal_build.log 2>&1 &`
   `OMP_NUM_THREADS=1 python -W ignore -m src.features --cal_eval` (held-out-source R2 per target)
   `OMP_NUM_THREADS=1 python -W ignore -m src.features --cal_apply` -> `data/features_cal.parquet`
   `OMP_NUM_THREADS=1 python -W ignore -m src.train_gbm --feat features_v3.parquet,features_cal.parquet --model ridge --name feat2_v3cal_ridge`
   (also `--model lgbs --name feat2_v3cal_lgbs`)
2. Save the regularised GBM: `OMP_NUM_THREADS=1 python -W ignore -m src.train_gbm --feat features_v3.parquet,features_v2.parquet --model lgbs --seeds 3 --name feat2_v23_lgbs` (ad-hoc 13.21).
3. Heteroscedastic weighting (sample weight about 1/sigma^2(correlation length), estimated in-fold) for ridge/LGB.
   Also try LGB monotone constraints (+) on the fd features.
4. Give `features_v3.parquet` to the embedding/CNN heads as side features. Blend feat2_v3_ridge with
   feat2_v23_lgb in the ensembler (their errors differ by quality quartile).
5. CNN idea for the orchestrator: add an input channel normalised by local matrix level (den / 70th-pct background).
