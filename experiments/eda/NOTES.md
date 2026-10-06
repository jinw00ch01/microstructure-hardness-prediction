# EDA notes: is the coarse-image error irreducible? (eda-analyst, 2026-10-06)

Residuals are the nested-CV OOF of `blend_v2` (CV 12.963), recomputed the same way as `src.ensemble`
(`blend_v2_nested_oof.csv`). Train labels only. Test images were used only for unsupervised marginals
(N_cal, snr). No labels or fitting on test. Nothing in `src/` was edited. Only `--no_save` screens were run.

## Verdict
Almost certainly irreducible. The coarse-image error behaves like label noise whose variance is inversely
proportional to the number of grains in the image. It does not look like a missing visible signal.

* Residual variance model (Gaussian ML, 5-fold CV on the shared folds): `v_i = b/N_i + c/snr_i`, where
  `N_i = cal_seg_count_density*6.5536` (calibrated grain count) and `snr_i = ic_ridge_snr`.
  Fit on all 500: b = 28,200 (bootstrap 90% CI 21.9k-32.4k), c = 10.9, intercept -> 0. The standardized residuals are
  then exactly Gaussian (sd 1.000, skew 0.005, excess kurtosis 0.18, vs 0.66 for the raw residuals).
  Fit on the cleanest 110 images (snr > 1) alone: intercept -> 0, b = 28,700. The same b comes out on clean and
  noisy data.
* The 1/N form beats the alternatives for every model family (blend, feat ridge, feat LGB, emb gridge, CNN).
  CV-NLL gain over const+1/snr for the blend: 1/N 31.1; mean dark-grain area 31.2 (∝ 1/N); N^-1.5 28.7;
  L_par^2 27.8; L_gm^2 25.8; 1/sqrt(N) 24.1; L_gm 22.0; **p(1-p)/N 21.1**; saturating b/(N+N0) no better (N0 ≈ 10).
* It is **not** sampling of a latent phase fraction p. A p(1-p)/N term loses to 1/N, and phase-specific per-grain
  variances (b_m(1-p)+b_d p)/N do not improve CV. The magnitude also rules it out. b ≈ 28k means a per-grain hardness
  scatter of about 150-170 HV averaged over N grains. Sampling a visible latent parameter could give at most
  β²·p(1-p) ≈ 3k for the phase fraction (β ≈ 140-170 HV per unit fraction), and far less for grain size or aspect ratio.
  This looks like an invisible per-grain random hardness, or equivalently label noise with SD ∝ grain size.
* Label variance itself grows with coarseness (y std 20.5 / 16.1 / 13.3 over N terciles), while prediction std
  moves much less (12.5 / 10.3 / 9.0). The fitted noise term accounts for 90 / 62 / 64% of the unexplained label
  variance in the coarse / mid / fine terciles.
* **Clean images already sit at the noise floor.** Observed RMSE vs sqrt(mean b/N) in the cleanest snr third:
  coarse 14.5 vs 15.9, mid 9.2 vs 9.7, fine 7.8 vs 7.7. All the excess error is in degraded images:
  noisy third 14.95 vs floor 11.1 (excess MSE ≈ 100), mid third 12.0 vs 10.6 (≈ 32). See `fig_var_vs_invN.png`.

## Floor estimates (label-noise term only, i.e. a perfect model with no measurement error)
| | estimate | range |
|---|---|---|
| train CV floor | **11.2-11.6** | 90% CI 9.8-12.3 (b bootstrap). With a free clean intercept: 10.4 (intercept reducible) to 11.9 (intercept is noise) |
| full test (1000) | **10.7-11.1** | test is slightly finer: mean 1/N 0.00430 vs 0.00465 |
| public 300 subset, perfect model | median 10.7 | 5-95% [9.2, 12.2]; P(<=10) ≈ 0.04 at fixed b, ≈ 0.2 including b uncertainty |
| public 300, degradation error halved | median 11.75 | 5-95% [10.7, 13.0]; P(<=10) ≈ 0.003 |
| public 300, blend_v2 as is | median 12.6 | 5-95% [11.4, 13.8] (expected test RMSE 12.6; CV 12.96) |

Paired public-LB noise between two submissions (random 30% subsets of train OOF, scaled to 300 images):
SD of the difference is 0.13 (blend vs best feature model) to 0.26 (blend vs CNN). LB differences under about 0.3 are noise.
**Conclusion: LB <= 10 is out of reach on expectation. Even a perfect model would need a lucky public split.**

## Signal hunt (all negative; effect sizes)
| candidate | test | result |
|---|---|---|
| fixed-location "indent" | corr map of residual vs local dark/pore fraction (minus global), 16/32/64-px blocks, permutation null for max | dark: max\|r\| 0.10-0.14 (all) vs null95 0.13-0.17, p 0.18-0.80; coarse p 0.39-0.60; centre disk R=4..64 corr -0.01..+0.09 (coarse, n=166, SE 0.08) |
| any existing feature (453 cols, v1/v2/v3/cal + my grain stats) | Spearman with residual | max \|rho\| 0.104 (all; null95 0.157), 0.19 coarse (null95 0.264). Hints only: elongation (+0.16-0.18) and raw brightness (-0.17) in coarse |
| per-grain aggregates (a8, 43 cols: brightness histogram of grains, matrix/dark levels and spreads, raw levels, interior texture, tensor strain / log-aspect / alignment, largest grain, area CV, dark clusters / connectivity / dark-dark contacts, neighbours, pores touching dark vs matrix) | in-fold ridge on residual, clean images | RMSE 11.855 -> 11.98 (worse) for every group. Clean-coarse rho all \|.\| <= 0.17 (n=101) |
| grain-boundary depth / width | Spearman, clean images | depth rho(y) +0.28 but rho(resid) -0.02; width -0.04 |
| local residual structure | kNN mean of neighbours' OOF residuals (features PCA20, effv2s / r18 embeddings PCA50) | features: **-0.19 to -0.30** (negative = shrinkage of noisy neighbours, no positive structure); embeddings -0.14..+0.13, none significant |
| ID order / label lattice | lag autocorr, decile means, lattice tests, KDE | none (lag-1 autocorr 0.01; no rounding or clusters; 500 distinct continuous values) |
| pores as the 1/N source | pore count / size vs N and \|r\| | pores barely scale with grain size; rho(\|r\|) about -0.05 |
| visual (m_coarse_pairs_clean.png, m_zoom_clean_coarse.png) | matched-prediction pairs, clean coarse | grains are flat-shaded with pixel noise and no inner texture. Positive-residual images looked more elongated, but tensor strain / log-aspect have rho ≈ 0 with the residual |

## Standard EDA items
* Target: mean 194.17, sd 17.70, range 152-264, skew 0.51 (log y skew 0.20), excess kurtosis 0.67, continuous.
  The skew comes from coarse images (y skew +0.48 in the coarse tercile, about -0.1 elsewhere).
* Feature vs hardness: `feature_hardness_spearman.csv` (all v1/v2/v3/cal columns: rho with y, rho on clean images,
  rho with the residual). The top features are dark-fraction measures: cal_ic_seg_fd93 0.65 (clean 0.71), cal_ic_fdo_93 0.64,
  cal_ic_gmm_w 0.64. Residual correlations of all features are <= 0.10.
* Quality groups: coarse-by-L (`ic_acg_len50_gm`) is confounded with blur (137 of 166 coarse-by-L images have snr <= 0.5).
  Use the calibrated count `cal_seg_count_density` for grain size, since it is balanced across quality groups.
* Train vs test: no shift. KS p > 0.06 for noise, blur slope, snr, shading, N_cal, correlation lengths, aspect,
  dark fraction, pores, orientation. Tercile occupancy (cut on train) in test: noise 31/36/33%, snr 32/35/33%, N 31/36/33%.
* Variance-weight screens (ridge v3+cal, `--no_save`): hetero on ic_acg_len50_gm 13.101; cal_seg_count_density 13.121;
  cal_seg_count_density + ic_ridge_snr 13.138. These are all within noise, so the existing weighting is fine.

## Recommendations for the modeling agents (priority order)
1. **Reset the goal.** The expected best achievable public LB is about 10.7-11.1, with ±0.9 split noise. Aim for
   test RMSE about 11.8-12.3. Treat LB <= 10 as luck, not a plan. Don't overfit the LB: 5 submissions/day and paired
   noise of 0.13-0.26.
2. **Spend effort only on degraded images.** Clean images are at the floor. All reducible error (about 37 of 168 MSE,
   i.e. 12.96 -> about 11.4 at best) sits in the noisy/blurred two thirds. The worst cells are noisy-mid
   (14.7 vs floor 9.8) and noisy-fine (11.2 vs 7.8). Ideas: calibration v2 (more degradations matched to the real
   noise/blur distribution, per noise band), a CNN with an NLM / denoised channel, multi-crop TTA, and training
   on synthetic degradations of clean train images (train images only).
3. Keep inverse-variance weights and shrinkage. Optimal predictions are E[y|x], so don't stretch predictions or
   chase coarse outliers. Coarse-image residuals of about 15-20 are expected noise.
4. Don't build more grain-level / location / texture features for coarse images. The 43 grain aggregates,
   location maps, boundaries and kNN checks all came back null.
5. For blending, judge gains in CV with fold-paired comparisons. Differences under about 0.1 CV / 0.3 LB are noise.

## Files (experiments/eda/)
Scripts: `eda_common.py` (nested blend OOF + feature table), `a1_hetero.py` (terciles, ID, discreteness),
`a2_maps.py CACHE train|test` (NLM + illumination-corrected maps + watershed, same as v3), `a3_location.py CACHE`,
`a4_grains.py CACHE train`, `a5_varfit.py CACHE`, `a6_montage.py CACHE`, `a7_resid_scan.py CACHE`, `a8_grainfeat.py CACHE train`,
`a9_gf_test.py CACHE`, `a10_knn.py`, `a11_floor.py`, `a12_varcompare.py`, `a13_boundary.py CACHE`,
`a14_shift.py CACHE`, `a15_lb_sim.py`, `a16_summary_tables.py`. CACHE is any scratch directory. Run a2, then a4 and a8 first
(about 4 min on 1 core for train; a2 on test takes about 3 min).
Outputs: `blend_v2_nested_oof.csv`, `a5_varfit.csv`, `a7_resid_feature_corr.csv`, `a11_floor_fits.csv`,
`feature_hardness_spearman.csv`, `corrmap_*_b32.npy`, figures `fig_var_vs_invN.png`, `m_*.png`.
