# feature-engineer handoff (updated 2026-10-06, session 7: high-noise specialist restorer, data_restored_hn)

## Session 7 summary (restored features on `data_restored_hn/`)
1. Features. `data_restored_hn/` holds the same file set as `data_restored/`: `features_v3`, `features_v2`,
   `v5_blocks_real`, plus the transferred `features_cal`, `features_v4` and `mil_blocks` (maps fitted on the degraded
   originals in `data/`, as before).
   - Only the 500 specialist images (raw ic_noise > 11.92: 167 train, 333 test) were re-extracted. The other 1000 rows
     were copied from `data_restored/`.
   - Bit-for-bit checks:
     - Re-extracting 16 unchanged images gives v3, v2 and v5 rows identical to `data_restored`.
     - All 1000 unchanged rows of every transferred file are identical too.
     - So the files equal a full 1500-image rebuild.
   - Wall time 9.6 min. Extraction: v3 52 s (4 cores), v5 blocks 73 s (3 cores), v2 127 s (2 cores). Transfers on 1 core
     each: cal 4.5 min, v4 2.5 min, MIL 4.3 min.
   - Script: scratchpad `hn/build_hn.py`. Log: `logs/resthn_summary.log`.
   - Full rebuild (same files, about 3x longer):
     `DATA_DIR=$HN python -m src.features --v3|--v2|--v4_blocks --v4_extra --n_jobs 4`
   - Transfers (DATA_DIR unset):
     - `F.cal_apply(apply_v3='$HN/features_v3.parquet', out='$HN/features_cal.parquet')`
     - `F.v4_apply(apply_file='$HN/v5_blocks_real.parquet', out='$HN/features_v4.parquet')`
     - `F.mil_blocks(apply_file='$HN/v5_blocks_real.parquet', out='$HN/mil_blocks.parquet')`
2. Screen: the saved members rebuilt with `data_restored_hn` in place of `data_restored`, fold-paired against the saved
   OOFs. All three saved members reproduce exactly from `data_restored`. Terciles are by raw ic_noise (train cuts
   7.49 / 11.92); the train N3 tercile is exactly the 167 changed images.

   | member | saved CV | hn CV | delta | fold deltas | better | N1 / N2 / N3 saved -> hn | N3 dMSE ± se |
   |---|---|---|---|---|---|---|---|
   | ridge_all | 12.5304 | 12.7323 | +0.202 | -0.002 +0.125 +0.300 +0.334 +0.245 | 1/5 | 10.72/11.32/15.09 -> 10.73/11.56/15.41 | +9.6 ± 5.6 |
   | ridge_add | 12.5249 | 12.6274 | +0.103 | +0.030 +0.124 +0.214 +0.121 +0.023 | 0/5 | 10.75/11.63/14.82 -> 10.75/11.78/14.96 | +4.1 ± 2.8 |
   | lgbs_add (3 seeds) | 12.7685 | 12.7223 | -0.046 | -0.157 -0.059 +0.060 -0.084 +0.010 | 3/5 | 11.04/11.58/15.26 -> 11.03/11.55/15.18 | -2.4 ± 5.4 |
   | lgbs_add, 10 seeds paired | 12.7550 | 12.7185 | -0.036 | -0.159 +0.014 +0.012 -0.054 +0.008 | 2/5 | N3 15.24 -> 15.18 | -2.0 ± 5.3 |

   - Why it fails. On real images the specialist's measures rank hardness worse in N3, although they match synthetic
     clean sources better.
     - Train Spearman in N3 (n 167), raw / data_restored / hn:
       - cal_ic_seg_fd93: 0.611 / 0.607 / 0.561
       - cal_ic_fdo_93: 0.601 / 0.557 / 0.511
       - ic_seg_fd93: 0.513 / 0.560 / 0.520
     - The hn copies sit further outside the cal maps' training range: median ridge SNR 11.0, vs 6.5 for data_restored.
     - The mean residual moves little (+0.5 at most). The loss is in ranking: ridge_all N3 corr(pred, y) 0.629 -> 0.607.
3. Optional: own columns. ridge_all plus `n3_*` copies of v3+cal, filled only for the 500 high-noise images and NaN
   elsewhere (median-imputed in fold), so N3 images get their own slopes. Built with `python -m src.features --n3own`.
   All rows are vs feat6_rest_ridge_all (12.530, N3 15.09):

   | own columns filled with | cal only | v3 + cal |
   |---|---|---|
   | hn values | 12.4425 (-0.088, 4/5), N3 15.00 | **12.3746** (-0.156, 3/5: +0.141 +0.171 -0.456 -0.198 -0.421), N3 14.83 |
   | data_restored values (control) | 12.4357 (4/5), N3 14.92 | 12.4867 (3/5), N3 15.03 |
   | raw values (control) | 12.4366 (4/5), N3 14.92 | **12.2623** (-0.268, 4/5: -0.071 +0.620 -0.452 -0.665 -0.737), N3 14.44 |
   | random 500-image subset (null, 5 draws) | | hn +0.19 to +0.45 (0-1/5); raw +0.14 to +0.50 (0-2/5) |

   - The separate high-noise slopes are real: random subsets never gain.
   - The specialist adds nothing beyond them. Raw values do as well or better.
   - On the ridge_add base the own columns do nothing: -0.009 with hn values, -0.007 with data_restored values.
   - Read-only blend check: nested NNLS over blend_v11's 6 non-zero members (single-stage, 12.4233 for v11's set).
     Swapping in each member for feat6_rest_ridge_all gives:
     - ridge_all hn: 12.467
     - own hn cal: 12.347 (5/5)
     - own hn v3+cal: 12.249 (5/5, N3 14.92 -> 14.65)
     - own raw v3+cal: **12.162** (5/5, N3 14.92 -> 14.39)
     - lgbs hn as a 7th member: weight 0.
4. Saved. These pass the rule: CV lower, at least 3/5 folds better, and lower N3.
   - **feat7_resthn_lgbs_add**: CV 12.7223, folds 13.042 12.552 13.246 12.540 12.205, SNR T1/T2/T3 15.099 / 11.519 / 11.168.
     Noise-level.
   - **feat7_resthn_ridge_all_n3own**: CV 12.3746, folds 13.130 11.565 12.507 12.465 12.153, SNR T1/T2/T3
     14.629 / 11.215 / 10.930.
   - Not saved:
     - own hn cal only: error correlation 0.98 with n3own, and its gain equals its controls
     - own raw v3+cal: best, but it is not an hn variant
   - Commands. `R` and `HN` are the absolute paths of data_restored and data_restored_hn;
     `D1="cal_ic_seg_L_,cal_ic_seg_mx_area_mean,cal_seg_area_cv,cal_segdk_area_cv"`:
```
taskset -c 0 python -W ignore -m src.train_gbm --feat features_v3.parquet,features_v2.parquet,features_cal.parquet,eda_feats_lledge.parquet,eda_feats_ecs.parquet,eda_feats_lf.parquet,$HN/features_v3.parquet,$HN/features_cal.parquet --model lgbs --seeds 3 --drop $D1 --hetero ic_acg_len50_gm --name feat7_resthn_lgbs_add
python -W ignore -m src.features --n3own --n3own_src $HN/features_v3.parquet,$HN/features_cal.parquet --cal_out $HN/features_n3own_v3cal.parquet
taskset -c 1 python -W ignore -m src.train_gbm --mil splmean --mil_design --mil_blocks_file $R/mil_blocks.parquet --feat $R/features_v3.parquet,$R/features_cal.parquet,eda_feats_lledge.parquet,eda_feats_ecs.parquet,$R/features_v4.parquet,$HN/features_n3own_v3cal.parquet --model ridge --drop "$D1,^v4(?!c_(bd|la|acd)_(mean|sd|q90|max_m_mean)\$|c_(bd|la)_hp\$)" --hetero ic_acg_len50_gm --name feat7_resthn_ridge_all_n3own
# raw-own lead: --n3own --n3own_src features_v3.parquet,features_cal.parquet --cal_out features_n3own_raw_v3cal.parquet,
# then the ridge command above with features_n3own_raw_v3cal.parquet in place of $HN/features_n3own_v3cal.parquet
```
   - The orchestrator saved the raw-own lead as **feat7_rawn3own_ridge_all** (CV 12.2623). In blend_v12 it has weight 0.59:
     nested CV 12.407 -> 12.162, 5/5 folds better, N3 14.85 -> 14.37.
   - Next: check the recurring fold-1 swing and RidgeCV's alpha per fold. Fold 1 has the fewest N3 images (23).
     The same split for the lgbs is unlikely to help: the trees can already split on raw ic_noise.
   - Screen scripts: scratchpad `hn/` (`screen_hn.py`, `check_transfer.py`, `null_own.py`, `blend_swap.py`,
     `screen_r2.py`). Logs: `logs/resthn_*.log`.
5. Round 2 (orchestrator's pre-registered list, 2 cores). Every comparator reproduces its saved OOF exactly.
   - New builders in `src.features` (DATA_DIR unset; all constants from train raw ic_noise only):
     - `--n3own --n3own_q 1/3 --n3own_qhi 2/3 --n3own_prefix n2_` -> `data/features_n2own_raw_v3cal.parquet`.
       Mid band 7.49 < ic_noise <= 11.916: 166 train + 355 test images.
     - `--nz_interact ramp|z` -> `data/features_nz{ramp,z}_raw_v3cal.parquet`: raw v3+cal x t, plus t itself.
       - ramp: t = clip((ic_noise - 7.4912) / (11.9159 - 7.4912), 0, 1)
       - z: t = (ic_noise - 10.3647) / 4.5331
     - The defaults reproduce `features_n3own_raw_v3cal` exactly.
   - Results (fold-paired; N1/N2/N3 = raw ic_noise terciles):

     | variant | comparator (CV) | CV | delta | fold deltas | better | N1 / N2 / N3 (comparator) |
     |---|---|---|---|---|---|---|
     | 1. N3 + N2 own columns | feat7_rawn3own_ridge_all (12.2623) | 12.6993 | +0.437 | +1.046 +0.266 +0.117 +0.476 +0.224 | 0/5 | 10.98/12.32/14.54 (10.68/11.33/14.44) |
     | 2a. x * t ramp (instead of N3 own) | same | 12.4253 | +0.163 | +0.043 +0.098 +0.060 +0.331 +0.295 | 0/5 | 10.86/11.53/14.57 |
     | 2b. x * t z-score | same | 12.5299 | +0.268 | +0.118 -0.020 +0.329 +0.375 +0.541 | 1/5 | 10.79/11.34/15.03 |
     | 3a. feat5 splmean + N3 own | feat5 splmean (12.5952) | **12.3648** | -0.230 | -0.067 +0.169 -0.146 -0.652 -0.470 | 4/5 | 10.65/11.92/14.25 (10.62/11.88/14.90) |
     | 3b. feat5 milspl + N3 own | feat5 milspl (12.6162) | **12.3225** | -0.294 | -0.091 +0.164 -0.145 -0.849 -0.570 | 4/5 | 10.49/12.01/14.19 (10.56/11.94/14.95) |
     | 3c. feat6_rest_ridge_add + N3 own | feat6_rest_ridge_add (12.5249) | **12.3396** | -0.185 | -0.006 +0.245 -0.192 -0.530 -0.445 | 4/5 | 10.72/11.68/14.33 (10.75/11.63/14.82) |

   - Only the noisiest third gains from its own slopes. Own slopes for the mid third hurt N2 (11.33 -> 12.32). The
     smooth forms lose to the hard split.
   - Saved (CLI, OOF = screen to 6e-14):
     - **feat7_rawn3own_ridge_splmean** 12.3648
     - **feat7_rawn3own_ridge_milspl** 12.3225
     - **feat7_rawn3own_ridge_add** 12.3396
   - Error correlations:
     - splmean vs milspl 0.996: use one of them, as with the feat5 pair
     - add vs feat7_rawn3own_ridge_all 0.987
     - splmean / milspl vs ridge_all 0.97
   - Fold 1 loses in every N3-own member.
   - Commands (`F5=features_v3.parquet,features_cal.parquet,eda_feats_lledge.parquet,eda_feats_ecs.parquet,features_v4.parquet`,
     `N3=features_n3own_raw_v3cal.parquet`, `KEEP='^v4(?!c_(bd|la|acd)_(mean|sd|q90|max_m_mean)$|c_(bd|la)_hp$)'`):
```
python -W ignore -m src.train_gbm --mil splmean --mil_design --feat $F5,$N3 --model ridge --drop "$D1,$KEEP" --hetero ic_acg_len50_gm --name feat7_rawn3own_ridge_splmean
python -W ignore -m src.train_gbm --mil spl --mil_design --feat $F5,$N3 --model ridge --drop "$D1,$KEEP" --hetero ic_acg_len50_gm --name feat7_rawn3own_ridge_milspl
python -W ignore -m src.train_gbm --mil splmean --mil_design --feat $F5,$R/features_v3.parquet,$R/features_cal.parquet,$N3 --model ridge --drop "$D1,$KEEP" --hetero ic_acg_len50_gm --name feat7_rawn3own_ridge_add
```

## Session 6 summary
1. Calibrated block-mean phase/pore columns for the lgbs: `data/features_v5blk.parquet`
   (`python -m src.features --v5_blockmeans`).
   - Columns: `v5m_` valid-weighted mean, `v5s_` sd and `v5q_` q90 over blocks of c_sfd93, c_sfd91, c_fdo93,
     c_deficit, c_pore60.
   - Fold-paired against feat3 lgbs (12.789):
     - means only: 12.767 (-0.022, 3/5 folds)
     - all 15 columns: 12.774 (-0.015, 3/5)
   - Not saved.
2. Restored images (cnn-trainer's U-Net, `data_restored/`).
   - I rebuilt, with `DATA_DIR=data_restored`:
     - `features_v3`, `features_v2` and `v5_blocks_real` (`.parquet`)
     - the transferred calibrations `features_cal`, `features_v4` and `mil_blocks`: fitted on degraded originals
       and applied with `cal_apply(apply_v3=...)`, `v4_apply(apply_file=...)` and `mil_blocks(apply_file=...)`
     - gated `*_g.parquet` copies: images with raw ic_ridge_snr < 0.08 (65 images: 23 train, 42 test) take their
       raw-image values
   - Restoration bias test on the restorer's 11 held-out sources. These are the only sources it never trained on.
     I made 12 fresh `_degrade_v2` copies of each and measured R² against the clean source's v3 value:

     | measure | degraded | degraded + cal | restored | restored + cal |
     |---|---|---|---|---|
     | ic_seg_fd93 | 0.43 | 0.80 | 0.78 | **0.93** |
     | ic_seg_fd93, noise >= 13 | -0.14 | 0.59 | 0.57 | **0.89** |
     | ic_fdo_93 | 0.06 | 0.65 | 0.85 | **0.92** |
     | ic_gmm_w | 0.34 | 0.70 | 0.84 | **0.93** |
     | ic_acg_len50_perp | -0.98 | 0.86 | 0.75 | **0.93** |
     | ic_seg_nfrac91 | -2.18 | 0.76 | 0.30 | **0.93** |
     | ic_pore60_frac | 0.04 | **0.40** | -0.84 | 0.30 |

     - On synthetic data, restoration plus the existing cal_ maps is the most accurate for everything except pores.
     - The maps still help on restored images, although restored noisy images fall outside their training range
       (ic_noise 0.7 vs 1.2-17; ridge SNR 6.1 vs q95 3.5).
     - Refitting is not needed for transfer on synthetic data. An honest refit would anyway need a restorer that
       never saw the source images, i.e. one cross-fitted over sources.
     - Pores: keep the raw-image pore features.
   - Real-image screens (fold-paired, saved OOFs as bases) show no consistent gain:

     | variant | CV | delta | folds better | T1 / T2 / T3 |
     |---|---|---|---|---|
     | ridge base + restored v3+cal (add) | 12.525 | -0.070 | 3/5 | 14.647 / 11.587 / 11.032 |
     | ridge, restored v3+cal replace raw (raw v4/MIL) | 12.579 | -0.016 | 2/5 | |
     | ridge, all restored (v3, cal, v4, MIL design) | 12.530 | -0.065 | 3/5 | 14.774 / 11.445 / 11.0x |
     | ridge, all restored, no cal | 12.535 | -0.060 | 2/5 | |
     | lgbs base + restored v3+cal (add) | 12.769 | -0.020 | 3/5 | 15.171 / 11.516 / 11.231 |
     | lgbs, restored v3+cal replace raw (raw v2) | 12.884 | +0.096 | 1/5 | |
     | lgbs, all restored (v3, v2, cal) | 12.899 | +0.110 | 2/5 | 15.366 / 11.385 / 11.541 |
     | lgbs, all restored, no cal | 12.912 | +0.124 | 2/5 | |
     | gated (snr < 0.08 -> raw): ridge add / ridge all-restored / lgbs add | 12.634 / 12.803 / 12.826 | +0.04 / +0.21 / +0.04 | 2/5 / 1/5 / 2/5 | |

     Bases: ridge = feat5 splmean 12.595 (T1 14.748 / T2 11.881 / T3 10.823); lgbs = feat3 12.789.
   - Pattern: restored features help the mid-noise tercile T2 a lot (ridge 11.88 -> 11.45-11.59, lgbs
     11.86 -> 11.39-11.52). They hurt the clean tercile and do not help the noisiest. Gains concentrate in fold 1
     (-0.2 to -0.7) and folds 2/4 lose.
   - Nested NNLS on OOFs (my quick version: 12.504 for the blend_v5 members):
     - + ridge all-restored: 12.458 (3/5 folds, weight 0.43)
     - + ridge add: -0.022 (3/5)
     - + lgbs add: -0.011 (3/5)
     - + lgbs all-restored: worse
   - Nothing qualified fold-paired. At the coordinator's request (for a per-tercile blend and the submission test of
     whether restoration carries over), the all-restored ridge was saved anyway as **feat6_rest_ridge_all**:
     - CV 12.5304, folds 12.989 11.394 12.963 12.664 12.575
     - T1 14.774 / T2 11.445 / T3 11.028 (raw-SNR terciles, like every other member)
     - test predictions come from the restored test features
     - command (`R=data_restored` absolute path):
       `taskset -c 0 python -W ignore -m src.train_gbm --mil splmean --mil_design --mil_blocks_file $R/mil_blocks.parquet --feat $R/features_v3.parquet,$R/features_cal.parquet,eda_feats_lledge.parquet,eda_feats_ecs.parquet,$R/features_v4.parquet --model ridge --drop "$D1,^v4(?!c_(bd|la|acd)_(mean|sd|q90|max_m_mean)\$|c_(bd|la)_hp\$)" --hetero ic_acg_len50_gm --name feat6_rest_ridge_all`
   - After blend_v7 (feat6_rest_ridge_all at weight 0.43) improved public LB from 11.9248 to 11.8795, two more
     restored members were saved (both reproduce their screens exactly; test predictions use restored test features):
     - **feat6_rest_ridge_add**: feat5 splmean config + restored v3 + cal next to the raw ones (MIL design from raw
       blocks). CV 12.5249, folds 13.039 11.814 12.819 12.691 12.224, T1 14.647 / T2 11.587 / T3 11.032.
     - **feat6_rest_lgbs_add**: feat3 config (3 seeds) + restored v3 + cal next to the raw ones.
       CV 12.7685, folds 13.199 12.611 13.185 12.624 12.194, T1 15.171 / T2 11.516 / T3 11.231.
     - Commands: the saved feat5 / feat3 commands with `$R/features_v3.parquet,$R/features_cal.parquet` appended to
       `--feat` (`R` = absolute path of data_restored).
   - Restored features only for mid-SNR images (train tercile cuts 0.283-0.774; raw values elsewhere; `*_m.parquet`):
     12.854 (+0.259, 0/5 folds). Mixing raw and restored values in the same columns hurts. Use the tercile pattern
     at blend level instead.
   - Scripts are in the session scratchpad: `restored_par.sh`, `restore_bias.py`, `screen_restored.py`,
     `diag_restored.py`. Logs: `logs/restored_*.log`.

## Session 5 summary: block-level MIL (tests the "local nonlinearity / Jensen term" reading)
Result: the local (Jensen) term is not supported once the global and spread features are in the model. A nonlinear
term in the calibrated block-mean measures does help the ridge member. Two ridge members were saved, both 4/5
folds better than feat4. Use one of them, not both (error correlation 0.996).

| exp | CV | folds | T1_noisy / T2 / T3_clean | coarse / mid / fine / clean-coarse |
|---|---|---|---|---|
| **feat5_v3cal_ridge_het_spat_v4_splmean** (feat4 ridge + B-splines of the image-mean calibrated block measures) | **12.595** | 13.136 12.049 12.864 12.678 12.216 | 14.748 / 11.881 / 10.823 | 15.49 / 12.08 / 9.48 / 13.20 |
| feat5_v3cal_ridge_het_spat_v4_milspl (feat4 ridge + block means of the B-spline bases, the local MIL form) | 12.616 | 13.182 12.035 12.888 12.776 12.161 | 14.791 / 11.854 / 10.868 | 15.52 / 12.11 / 9.48 / 13.08 |
| base feat4_v3cal_ridge_het_spat_v4 | 12.686 | 13.235 12.337 12.724 12.882 12.226 | 14.854 / 11.987 / 10.879 | 15.58 / 12.15 / 9.61 / 13.25 |

Blends with feat3_v23cal_lgbs_het_spat (50/50):
- splmean ridge: 12.496 (fold deltas -0.047 -0.129 +0.052 -0.091 -0.053)
- milspl ridge: 12.492 (-0.047 -0.146 +0.043 -0.066 -0.076)
- feat4 ridge: 12.549

### Block table (`data/mil_blocks.parquet`, 73500 = 1500 images x 49 blocks)
- Raw v4 block measures, plus `V4_EXTRA`: per-block versions of the v3 phase/pore features, from the same ic_
  segmentation:
  - per-grain dark fraction `sfd93/91/89` (grain-interior median < t; area >= 8; not a pore)
  - pixel `fd2_93/87`, opened `fdo_93`, `deficit`
  - `pore68/60_frac`, `pore_deficit`, `pore60_n`
- Quality context.
- Stage-1 calibrated block values `c_*`: same label-free, source-cross-fitted scheme as v4, inputs now include the
  extra measures.

  | column | held-out-source R² calibrated | R² raw |
  |---|---|---|
  | c_sfd93 | 0.88 | 0.36 |
  | c_sfd91 | 0.89 | 0.35 |
  | c_fdo93 | 0.87 | 0.50 |
  | c_deficit | 0.85 | 0.27 |
  | c_pore60 | 0.82 | 0.51 |
  | c_bd | 0.65 | |
  | c_la | 0.62 | |
  | c_acd | 0.87 | |

  Report: `data/mil_blocks_report.csv`.
- The v4 files and `features_v4.parquet` are unchanged. The v5 files reproduce the V4_MEAS columns exactly.

### Models and screens (`src/train_gbm.py` MIL section; shared folds by image)
1. Block-level LightGBM, every block's target = its image's hardness, aggregated per image
   (mean / valid-weighted mean / q10 / q50 / q90 / max / min / sd).
   - Alone: 16.02 (15.78 after in-fold linear recalibration). It fails because the block-level fit is
     attenuated: within-image variation of local measures acts as errors-in-variables, while the image-constant
     context is not attenuated.
   - Its cross-fitted aggregates add nothing: ridge -0.0002, lgbs -0.0001.
2. Additive MIL fitted on the image-level loss ("ridge on splines"). Per block, B-splines (5 quantile knots, fitted
   in-fold) of c_la, c_sfd93, c_pore60, c_acd and acr_len50, plus c_sfd93 x spline(c_la) and c_sfd93 x spline(c_acd).
   These are averaged over blocks with valid-area weights, then RidgeCV at image level. A block prediction is
   phi(x_b).beta, so the image fit is their weighted mean.
   - Alone: 13.444. Its Jensen control (same splines of the image means) alone: 13.615. Local beats the control
     on 4/5 folds here, but only without the global features.
   - Cross-fitted aggregates (training rows get inner-OOF values; inner folds = the other shared folds):
     - ridge 12.635 (-0.051, 3/5)
     - lgbs 12.719 (-0.069, 3/5)
     - blend 12.549 -> 12.506 (3/5)
     - Not saved.
3. The spline design itself added to the members, fitted jointly with the global features (`--mil_design`):
   - ridge + local design: 12.616 (-0.070, 4/5). Permutation null with the 49-block sets shuffled across images
     (10x): -0.008 to +0.284, mean +0.156.
   - ridge + image-mean control: 12.595 (-0.091, 4/5). Local vs control: 2/5 folds. So inside the member the gain
     is global nonlinearity, not the Jensen term.
   - lgbs + local design: 12.817 (+0.028, 2/5). Not saved.
- Fitted local functions (stand-alone additive model; "if every block had value x"):
  - convex and increasing in local dark fraction: 189 -> 185 -> 193 -> 203 -> 226 HV at the q05..q95 values
    0.01 / 0.07 / 0.13 / 0.21 / 0.37
  - mildly increasing in local log grain area: +6 HV (fd 0) to +9 HV (fd 0.6) from q05 to q95, i.e. bigger = harder
    and no Hall-Petch, as in EDA's physics fit
  - The convex dark-fraction effect is what the linear ridge was missing. The lgbs already models it, which is why
    lgbs gains nothing.

### Commands (1 core)
```
taskset -c 0 python -W ignore -m src.features --v4_blocks --v4_extra --n_jobs 1           # v5_blocks_real.parquet, ~9 min
taskset -c 0 python -W ignore -m src.features --v4_cal_build --v4_extra --n_aug 8 --snr_min 0.9 --n_jobs 1   # v5_blocks_cal.parquet, ~9 min
taskset -c 0 python -W ignore -m src.features --mil_blocks                                # mil_blocks.parquet + _report.csv, ~5 min
R="--feat features_v3.parquet,features_cal.parquet,eda_feats_lledge.parquet,eda_feats_ecs.parquet,features_v4.parquet --model ridge --hetero ic_acg_len50_gm"
taskset -c 0 python -W ignore -m src.train_gbm --mil splmean --mil_design $R --drop "$D1,^v4(?!c_(bd|la|acd)_(mean|sd|q90|max_m_mean)\$|c_(bd|la)_hp\$)" --name feat5_v3cal_ridge_het_spat_v4_splmean
taskset -c 0 python -W ignore -m src.train_gbm --mil spl --mil_design $R --drop "$D1,^v4(?!c_(bd|la|acd)_(mean|sd|q90|max_m_mean)\$|c_(bd|la)_hp\$)" --name feat5_v3cal_ridge_het_spat_v4_milspl
# screens: --mil lgb|spl|splmean --no_save (member alone); --mil lgb|spl --mil_stack <member flags> --no_save
```
- `run()` now takes `fold_extra` (fold-specific columns, e.g. cross-fitted stacked features). The refactor
  reproduces the saved feat4 OOF to 6e-14.

### Next ideas
1. Grain-level MIL: one instance per watershed grain (phase level, area, aspect, neighbours), with area- and
   number-weighted means of spline bases. It only works on clean images unless the per-grain measures are calibrated.
2. Give the lgbs the calibrated block-mean measures as plain columns (`v4c_*_mean` exist; c_sfd93 / c_fdo93 /
   c_pore60 means are not yet in a feature file).


## Session 4 summary (spatial heterogeneity + local grain-size map)
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
| `data/v5_blocks_real.parquet` / `v5_blocks_cal.parquet` | 73500x45 / 51352x52 | v4 block measures + `V4_EXTRA` phase/pore block measures (V4_MEAS columns identical to the v4 files) | `--v4_blocks --v4_extra`, `--v4_cal_build --v4_extra` | ~9 / ~9 min |
| `data/mil_blocks.parquet` | 73500x53 | MIL block table: raw + calibrated `c_*` block values + context | `--mil_blocks` | ~5 min |
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
