# feature-engineer handoff (updated 2026-10-06, session 4: spatial heterogeneity + local grain-size map)

## Session 4 summary
Best feature members now: **feat4_v3cal_ridge_het_spat_v4 (ridge, CV 12.686)** and
**feat3_v23cal_lgbs_het_spat (lgbs, CV 12.789)**. Their 50/50 average is 12.549 with error correlation 0.941
(feat3 pair: 12.611; feat2 pair: 12.868). I recommend these, plus feat3_v3cal_ridge_het_spat, in place of the feat2
members in the blend.

### Step 1: EDA spatial-heterogeneity files on the saved setups (fold-paired, same flags + new files)
| exp | added files | CV | folds | T1_noisy / T2 / T3_clean |
|---|---|---|---|---|
| **feat3_v3cal_ridge_het_spat** | lledge + ecs | **12.785** | 13.496 12.494 12.877 12.666 12.359 | 14.935 / 12.077 / 11.015 |
| **feat3_v23cal_lgbs_het_spat** | lledge + ecs + lf | **12.789** | 13.004 13.034 13.216 12.435 12.227 | 14.962 / 11.862 / 11.222 |
| base feat2_v3cal_ridge_het | - | 13.101 | 13.80 12.76 13.22 13.00 12.70 | 15.119 / 12.179 / 11.741 |
| base feat2_v23cal_lgbs_het | - | 12.991 | 13.35 12.91 13.31 12.80 12.57 | 15.008 / 11.984 / 11.718 |

- Ridge: lledge+ecs beats base on 5/5 folds. All three files give 12.824, the same as EDA's screen; lf hurts the linear model.
- lgbs: all three vs lledge+ecs is a tie (12.789 vs 12.785). All three wins 4/5 folds head-to-head and has the
  best clean tercile, so I saved it. Against base it is better on 4/5 folds (EDA's screen: 12.771). Either lgbs
  version blends with the ridge member to 12.611.
- Commands (core 0; `D1` is the failed-calibration drop list below):
```
taskset -c 0 python -W ignore -m src.train_gbm --feat features_v3.parquet,features_cal.parquet,eda_feats_lledge.parquet,eda_feats_ecs.parquet --model ridge --drop $D1 --hetero ic_acg_len50_gm --name feat3_v3cal_ridge_het_spat
taskset -c 0 python -W ignore -m src.train_gbm --feat features_v3.parquet,features_v2.parquet,features_cal.parquet,eda_feats_lledge.parquet,eda_feats_ecs.parquet,eda_feats_lf.parquet --model lgbs --seeds 3 --drop $D1 --hetero ic_acg_len50_gm --name feat3_v23cal_lgbs_het_spat
```

### Step 2: v4 local grain-size map (EDA recommendation 3), `src/features.py` v4 block
- Each image is cut into 64 px blocks (7x7 grid, stride 32). Per block, on the v3 pipeline (NLM denoise -> local
  matrix level -> sato ridge -> h-minima watershed):
  - watershed boundary density `bd_ws` and area-weighted mean log grain area `la`
  - region-centroid count density
  - sato ridge mean / p90 and an 8-orientation line-averaged (9 px) ridge energy with area fractions
  - gradient energy, dark fraction, grey sd
  - masked block autocorrelation of the raw image, normalised at lag 1 so white noise drops out
    (`acr_*`), and of the denoised image (`acd_*`)
  - image context: `ic_noise`, `ic_spec_*`, ridge SNR, matrix level, image means of every block measure
- Calibration is label-free and uses train images only. 131 clean train sources (`ic_ridge_snr > 0.9`) x 8
  `_degrade_v2` copies give 51k block rows; identity rows are added.
  - Stage 1 (block level): LightGBM maps degraded block measurements + context to the clean block's log `bd_ws`,
    `la` and `acd_len50`.
  - Stage 2 (image level): regression calibration of the map statistics.
  - Both stages are cross-fitted over sources with GroupKFold(5). Source images are always predicted by the fold
    model that did not see them; every other image (train and test) gets the mean of the 5 fold models.
- Held-out-source R² per block (stage 1; the raw degraded measurement is strongly biased):

  | target | R² stage 1 | R² raw | R² at noise <8.5 / 8.5-13 / >13 |
  |---|---|---|---|
  | log bd | 0.63 | -16 | 0.73 / 0.63 / 0.56 |
  | la | 0.61 | -7 | 0.65 / 0.58 / 0.59 |
  | acd_len50 | 0.87 | -0.6 | 0.92 / 0.90 / 0.83 |

  Image-level correlation of the stage-1 map statistic with the clean-image statistic:
  - mean / q90 / Hall-Petch average: 0.80-0.87
  - spread (sd, max-mean): 0.27-0.47 for bd/la, 0.88-0.95 for acd
  - raw spread: 0.12-0.27
  - Stage 2 is no better than the stage-1 statistics (1179 image rows is too few).

  Full table: `data/features_v4_report.csv`.
- `data/features_v4.parquet` (1500 x 121), column groups:
  - `v4r_*` raw-map stats, `v4c_*` stage-1 calibrated-map stats, `v4i_*` stage-2 calibrated stats
  - per map (`bd` log boundary density, `la`, `acd`): mean, sd, q0/10/25/50/75/90/100, max_m_mean, mean_m_min,
    q90_m_q10, sd4 and rng4 (non-overlapping 4x4 blocks)
  - `*_bd_hp` = mean sqrt(bd) and `*_la_hp` = mean area^-1/4: local Hall-Petch averages
- Screens with `--no_save`, fold-paired against the step-1 members:

  | variant | CV | delta | folds better | T1 / T2 / T3 |
  |---|---|---|---|---|
  | R1 ridge + 14 compact v4c cols (bd/la/acd mean, sd, q90, max_m_mean + 2 hp) | 12.686 | -0.099 | 4/5 | 14.854 / 11.987 / 10.879 |
  | R2 ridge + the same 14 as v4i | 12.681 | -0.104 | 3/5 | 14.751 / 12.019 / 10.967 |
  | L1 lgbs + all 120 v4 cols | 12.729 | -0.059 | 2/5 | |
  | L2 lgbs + v4c only | 12.752 | -0.037 | 3/5 | |
  | L3 lgbs + the 14 compact v4c cols | 12.718 | -0.071 | 2/5 | |

  - Permutation null for R1 (v4 rows shuffled across images, 10x): delta -0.024 to +0.097, mean +0.029. R1's
    -0.099 lies outside it.
  - Ablations of R1:
    - spread-only (sd and max-mean of bd/la/acd, 6 cols): -0.096, 4/5 folds
    - level-only: -0.053
    - bd+la only: -0.073
    - acd only: -0.067
  - So the between-block heterogeneity carries the gain, as EDA predicted.
  - Gains by grain-size tercile (`cal_seg_count_density`), coarse / mid / fine / clean-coarse:
    15.74 / 12.16 / 9.74 / 13.49 -> 15.58 / 12.15 / 9.61 / 13.25.
- Saved **feat4_v3cal_ridge_het_spat_v4**: CV 12.686, folds 13.235 12.337 12.724 12.882 12.226,
  T1 14.854 / T2 11.987 / T3 10.879.
  - Fold 3 is worse (+0.216) and the other four improve.
  - No lgbs v4 member was saved: no lgbs variant is consistent fold-paired.
  - Command:
```
taskset -c 0 python -W ignore -m src.train_gbm --feat features_v3.parquet,features_cal.parquet,eda_feats_lledge.parquet,eda_feats_ecs.parquet,features_v4.parquet --model ridge --drop "$D1,^v4(?!c_(bd|la|acd)_(mean|sd|q90|max_m_mean)\$|c_(bd|la)_hp\$)" --hetero ic_acg_len50_gm --name feat4_v3cal_ridge_het_spat_v4
```
- Rebuild v4 (1 core, about 25 min in total):
  1. `taskset -c 0 python -W ignore -m src.features --v4_blocks --n_jobs 1` -> `data/v4_blocks_real.parquet`
     (73500 block rows, ~10 min)
  2. `taskset -c 0 python -W ignore -m src.features --v4_cal_build --n_aug 8 --snr_min 0.9 --n_jobs 1` ->
     `data/v4_blocks_cal.parquet` (seed0 500000, ~6 min)
  3. `taskset -c 0 python -W ignore -m src.features --v4_apply` -> `data/features_v4.parquet` + `_report.csv` (~3 min)
- `src/train_gbm.py` notes now record the `--drop` regexes, with `|` escaped for the markdown LEADERBOARD. I also
  escaped the pipes by hand in my own feat4 LEADERBOARD line.

### Next ideas
1. Re-blend with feat4_v3cal_ridge_het_spat_v4 + feat3_v23cal_lgbs_het_spat (+ feat3 ridge for diversity).
2. The v4 spread statistics are only weakly recoverable under noise (image-level r 0.3-0.5 for bd/la). Options:
   - larger or Gaussian-weighted windows
   - domain reweighting
   - more sources (snr > 0.75)
   Each needs one ~25 min rebuild plus a fixed pre-registered screen.
3. CNN / embedding: per-block statistics of the calibrated maps could go in as side inputs to a head
   (`features_v4.parquet`, `v4c_*`).


## Session 3 summary (calibration v2, 1/N variance weights)
- Real low-quality images differ from v1's synthetic degradations. Their dark side is compressed toward the
  matrix level, the bright-side grain spread is preserved, and blur grows with noise. Noise >12 gives a GMM dark
  weight of 0.09 vs 0.25 for noise <8, although mean hardness is equal across SNR terciles.
- Calibration v2 (`_degrade_v2`, `cal_apply_v2`):
  - latent quality q sets noise 4-21; blur up to 3.2 px and dark-side contrast down to 0.3 grow with q;
    pores stay crisp; optional mildly correlated noise
  - 12 degradations x 174 sources = 2088 pairs
  - LightGBM per noise band (<8.5 / 8.5-13 / >13, training ranges ±1.5)
  - domain weights from logistic regression of real TRAIN vs synthetic quality descriptors (ESS 1180/2262)
  - commands:
    `taskset -c 0 python -W ignore -m src.features --cal_build --cal_version 2 --n_aug 12 --snr_min 0.75 --n_jobs 1 --cal_out cal2_pairs.parquet --seed0 20000` (~20 min)
    `taskset -c 0 python -W ignore -m src.features --cal2_apply` -> `data/features_cal2.parquet` + `features_cal2_report.csv` (~8 min)
- Held-out-source R² on realistic v2 degradations (v1 maps on the same data in brackets):
  per-grain fd `ic_seg_fd93` 0.88 [0.61], `ic_fdo_93` 0.88 [0.52], `ic_gmm_w` 0.87 [0.55],
  `ic_acg_len50_par` 0.84 [0.75], `seg_count_density` 0.78 [0.70], pores 0.75-0.78 [same].
  At noise >=13, fd R² is 0.80.
- Hardness CV did not improve on the noisy tercile. Real noisy images are still harder than the synthetic ones;
  within-T1 Spearman of fd with hardness is 0.50 (cal2) vs 0.56 (cal v1), while cal2 removes the level bias.
- 1/N (+snr) additive variance weights (`--hetero_add cal_seg_count_density,ic_ridge_snr`): equal within noise
  (lgbs 13.008 vs 12.991; ridge 13.141 vs 13.101). Saved as an lgbs variant.

| exp (session 3) | CV | folds | T1_noisy / T2 / T3_clean |
|---|---|---|---|
| feat2_v3calc2_ridge_het (v3+cal+cal2, len weights) | 13.067 | 13.78 12.62 13.25 12.92 12.73 | 15.080 / 12.104 / 11.755 |
| feat2_v23cal_lgbs_hetN (v3+v2+cal, 1/N+1/snr weights, 3 seeds) | 13.008 | 13.23 13.05 13.36 12.87 12.52 | 15.051 / 11.878 / 11.825 |
| reference: feat2_v23cal_lgbs_het | 12.991 | | 15.008 / 11.984 / 11.718 |
| reference: feat2_v3cal_ridge_het | 13.101 | | 15.119 / 12.179 / 11.741 |
| 50/50 lgbs_het + v3calc2_ridge_het (not saved) | 12.847 | | 14.852 / 11.896 / 11.529 |

Screens (not saved): ridge v3+cal2 13.117 (T1 15.19); lgbs v3+v2+cal2 13.067 (T1 15.16);
lgbs v3+v2+cal+cal2 13.032 (T1 15.07). Saved runs also write `tercile_rmse.json`.

Next for the noisy third:
1. Make degradations match better. Real noisy images are always blurred (drop low-blur samples at high noise)
   and keep more matrix grey-level spread. Fit the degradation prior by simulation-based matching of per-image
   descriptors (train images only), not by band medians.
2. Use CNN/embedding features of synthetically degraded clean train images (same calibration idea in feature
   space), for the CNN agent.
3. Accept the floor. EDA says T1 has about 4 RMSE of excess over its 11.1 floor; my features recover about 0.3.

Owner files: `src/features.py`, `src/train_gbm.py`. I don't commit. At the end of session 3 the best feature model
was feat2_v23cal_lgbs_het, CV 12.991 (baseline feat_lgb 14.202). Session 4 supersedes it (see the top of this note).

## Feature files (data/ is git-ignored; back these up to the cache)
| file | rows x cols | content | regenerate (1 core) | time |
|---|---|---|---|---|
| `data/features.parquet` | 1500x104 | v1 (unchanged) | `OMP_NUM_THREADS=1 python -m src.features --n_jobs 1` | ~8 min |
| `data/features_v2.parquet` | 1500x179 | v2: quality (`q_*`), levels `lv_*`, pores `pore*`, pixel dark fractions `ph_*`, dark blobs `dk_*`, sato-ridge + watershed grains `seg_*` (count density, area stats, aspect, orientation R, per-grain dark fraction `seg_fd*`), `segdk_*`, intercepts `seg_L_*`, autocorrelation `ac*`, structure tensor `st*` | `OMP_NUM_THREADS=1 python -W ignore -m src.features --v2 --n_jobs 1` | ~13 min |
| `data/features_v3.parquet` | 1500x135 | v3 `ic_*`: image divided by local matrix level (70th-pct filter, ~80 px). Pores, dark fraction at many thresholds, 2-GMM, dark chords / autocorrelation lengths along and across the elongation axis, per-grain classification `ic_seg_fd*`, plus 6 quality columns `ic_noise`, `ic_spec_*` (added 2026-10-06 by `--patch_v3_quality`; the other 128 columns are unchanged) | `OMP_NUM_THREADS=1 python -W ignore -m src.features --v3 --n_jobs 1` (includes the quality columns) | ~13-15 min |
| `data/cal_pairs.parquet` | 1392x141 | v3 features of 174 clean train images (ic_ridge_snr > 0.75) x 8 synthetic degradations (blur 0-2.5 on grains with crisp pores, shading, contrast, noise 0-18) | `OMP_NUM_THREADS=1 python -W ignore -m src.features --cal_build --n_aug 8 --snr_min 0.75 --n_jobs 1` | ~8 min |
| `data/features_cal.parquet` | 1500x30 | `cal_*`: LightGBM maps (fitted on cal_pairs only; no labels, no test images) from degraded v3 features to clean-image measurements | `OMP_NUM_THREADS=1 python -W ignore -m src.features --cal_apply` (prints the held-out-source R2 = cal_eval report) | ~2 min |
| `data/v4_blocks_real.parquet` / `v4_blocks_cal.parquet` | 73500x33 / 51352x40 | v4 per-block measurements (real train+test / degraded copies of 131 clean train images) | `--v4_blocks`, `--v4_cal_build` (see session 4) | ~10 / ~6 min |
| `data/features_v4.parquet` | 1500x121 | v4 local grain-size map stats `v4r_/v4c_/v4i_` | `--v4_apply` | ~3 min |
| `data/eda_feats_{lledge,ecs,lf}.parquet` | 1500x7 / 11 / 20 | eda-analyst's spatial heterogeneity features (their scripts, see `experiments/eda/NOTES.md`) | eda-analyst | |

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
