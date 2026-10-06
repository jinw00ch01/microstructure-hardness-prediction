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

---
## 2026-10-06 (afternoon): coarse residual reopened after public LB 12.345 (LB #1 9.244)

### Corrections to the morning section
* "Clean images sit at the floor" was circular: b was fitted on the same residuals. The b/N fit cannot separate label
  noise from model error that also scales as 1/N. **The floor numbers above are upper bounds on label noise, not a floor.**
* Naive residual probes and my kNN test were **biased negative by stacking leakage**. The OOF residuals of the
  training folds come from models that saw the held-out fold. Naive ridge probes on blend residuals gave −0.11 to −0.20
  for every representation. Leakage-free probes (cross-fitted, below) show real positive structure.
* Recommendation 4 ("no more grain / texture features for coarse images") is withdrawn.

### Method used for every test below (a17b / a22)
Per outer fold k:
* base = RidgeCV on v3+cal (the feat2_v3cal_ridge setup) trained on the other 4 folds.
* Inner 4-fold cross-fitted base residuals are computed inside those training folds.
* probe = RidgeCV(candidate → inner residual), applied to fold k. No fold-k label touches it.
* The same probe is added to the blend_v2 nested OOF.

A constant-per-fold probe gives corr ≈ +0.08 and a blend change of 12.963 → 12.959. That is the null baseline. N terciles use
`cal_seg_count_density`. "Clean" means `ic_ridge_snr > 0.5` in a22 and `> 0.9` in a29–a31.

### 1. Linear probes (a17, a17b, a17c, a25)
* **Global embeddings are null.** effv2s, r18, r18_nlm and cnxn give corr 0.02–0.08 (at the baseline). The full v1/v2/a8
  feature tables give 0.08.
* **Grid-cell heterogeneity is a positive signal:** the std of the pooled features over the 2x2 quadrant cells of the g2a
  embeddings. effv2s cellstd: corr 0.17 with the base residual, blend 12.963 → 12.818, coarse 16.33 → 16.08.
  - Permutation null over 40 shuffles: mean gain −0.015, max +0.05 (p < 0.025). 4 of 5 folds improve.
  - Stages 0–1 carry it. `ecs_blk0/1` (4 numbers) gives blend 12.963 → 12.695; coarse 16.33 → 15.65 (corr 0.32);
    clean-coarse 15.35 → 14.36 (corr 0.40). Fine images don't improve.
  - It replicates on other backbones (stage 0–1, 4 numbers): r18 12.963 → 12.805 (coarse 15.89), cnxn → 12.843.
* What it tracks:
  - `ecs_blk0_std` is quadrant heterogeneity of low-level edge / texture energy. It correlates 0.68 (all) and 0.84 (clean)
    with `ll2_edge_sd`, and 0.52 with the grain-count CV across quadrants.
  - `ecs_blk0_mean` is mostly shading (0.66–0.75 with q_illum_range). Shading alone is null.

### 2. Physics forms on clean images (a18, a20)
* I fitted parametric forms by soft-L1 least squares on the shared folds, restricted to clean rows:
  - rule of mixtures on area or number fraction
  - per-phase Hall-Petch, both d^-1/2 and linear-d forms
  - porosity terms: (1−cφ), exp(−bφ), (1−φ)^n
  - aspect terms
* None beats the blend. On snr > 0.5 (n=261) the blend is 11.71, mix_area 12.50, and the best form 12.10.
  The coarse tercile is 15.30 against the blend's 15.32.
* Area fraction is far better than number fraction (12.50 vs 15.03). Fitted H_matrix ≈ 176 and H_dark ≈ 303 HV.
  The Hall-Petch terms are tiny and favour "bigger = harder". Porosity coefficients are about 0.
* My robust per-grain phase measure (global shading surface, a18) is no better than ic_seg_fd91 (Spearman 0.70 vs 0.70).
  So the local-background normalisation does not corrupt coarse images.

### 3. Spatial arrangement of the dark phase (a21) and 4. border / largest grains (a18): null
All of these sit at the fold-mean baseline (blend 12.959):
* connectivity, largest component, spanning x / y / along the axis, Euler number
* contiguity C_dd
* normalised two-point S2(r) for r = 4..96, isotropic / along / across the axis
* lineal path, banding index
* window dispersion vs the binomial expectation
* dark-grain alignment
* border-cut area shares by phase
* phase, share and aspect of the 3 largest grains

The only exception is contiguity (blend 12.940; clean-coarse 15.35 → 15.14, but fine images get worse). This is a weak hint, not a result.

### New: spatial heterogeneity of grain size (a23, a29, a30), seen on clean images
* Visual check (`m_clean_extreme_resid.png`, `m_ecs_b0std_extremes.png`): positive-residual clean images are duplex or zoned.
  Regions of large grains sit next to regions of small grains.
* Feature: a local mean log grain-area field from watershed grains (Gaussian σ = 32 px, area-weighted, 8x8 grid).
  - On snr > 0.9 (n=131), `lf_la_sd_rel` (field std / global log-area std) has Spearman with the blend residual +0.30
    overall and +0.48 on coarse images. On snr > 1.2 (n=77) it is +0.36 overall and +0.66 on coarse.
  - The coarsest local region (`loc_max`), controlled for the global number- and area-weighted mean log area, has partial
    ρ = +0.32 to +0.37. The finest region gives −0.09.
  - Moran's I of log area is weaker (+0.15 to +0.23).
* Same direction as the global "coarser = harder" finding: zones of large grains raise hardness beyond what mean-size features predict.

### CV gains: fold-paired, saved setups, `--no_save` (a31; `src.train_gbm.run`, save=False)
| added to | ridge het (v3+cal) | lgbs het (v3+v2+cal, 3 seeds) |
|---|---|---|
| nothing | 13.101 [13.80 12.76 13.22 13.00 12.70] | 12.991 [13.35 12.91 13.31 12.80 12.57] |
| `eda_feats_lledge` | **12.854** (5/5 folds better) | **12.796** (5/5) |
| `eda_feats_ecs` | 12.907 (5/5) | 12.919 (5/5) |
| `eda_feats_lf` | 13.160 (worse, linear model) | 12.877 (5/5) |
| all three | **12.824** (5/5) | **12.771** (4/5) |
| edge only (4 cols) / noise only (2) / ll2_edge_sd only | 12.881 / 13.066 / 13.012 | 12.892 / 12.975 / 12.939 |

Blend check: blend_v2 weights with both feature members swapped for the +all-three versions.

| | overall | coarse | mid | fine | clean-coarse |
|---|---|---|---|---|---|
| base (in-sample weights) | 12.815 | 16.15 | 11.95 | 9.44 | 14.94 |
| + all three | **12.634** | 15.79 | 11.92 | 9.35 | 14.21 |

All 5 folds improve (fold deltas −0.21, −0.01, −0.21, −0.18, −0.29). Most of the gain is in the coarse tercile.

Variance model on these residuals (a32): the b/N coefficient drops from 27.65k to 25.6k (−7%); c/snr is unchanged.
So at least about 7% of the "label-noise" term was model error, and it is visible in the images.

### Feature definitions (fixed per-image transforms; train + test in `data/`)
* `data/eda_feats_lledge.parquet` (a28; raw uint8 image as float):
  - g = |∇ Gauss_σ1(raw)| via np.gradient.
  - Block means of g over 2x2 blocks (128 px) and 4x4 blocks (64 px).
  - `ll{2,4}_edge_sd` = std over blocks / raw mean. `ll{2,4}_edge_rng` = (max−min) / raw mean.
  - `ll2_noise_{sd,rng}` = the same over 2x2 blocks of 1.4826·MAD(raw − box3x3(raw)).
* `data/eda_feats_ecs.parquet` (a27): from `data/emb/tf_efficientnetv2_s.in21k_ft_in1k_256_g2a.npy` (6 views × 5 cells; cell 0 = global).
  - Take views id / hflip / transpose / rot90.
  - Compute the std over the 4 quadrant cells, average over views, apply sign·log1p, then average over the channels
    of each stage/pool block → `ecs_blk{0..4}_{mean,std}`.
  - Stages 0–1 carry the signal.
* `data/eda_feats_lf.parquet` (a30; needs the a2 caches rn / ws): watershed grains (area ≥ 12, non-pore).
  - Global stats: `lf_la_n`, `lf_la_w`, `lf_la_sd`.
  - Local log-area field at σ = 32 / 48 px: sd, range, max, min, sd_rel, max − global area-weighted mean.
  - `lf_dk{s}_sd`, `lf_mi_la_48`, `lf_n`.
  - Only meaningful where the watershed works (snr ≳ 0.9). It hurts ridge and helps LGB, which can gate on snr.

### Recommendations (replace the morning list)
1. feature-engineer: add `eda_feats_lledge` + `eda_feats_ecs` (+ `eda_feats_lf` for GBMs) and re-blend.
   Expect about −0.15 to −0.2 CV at the blend level (fold-paired above).
2. embedding-modeler / CNN: the signal is **spatial heterogeneity of low-level texture / grain size between image regions**,
   and global average pooling throws it away.
   - Add cross-cell statistics: std / range over a 2x2 and 4x4 grid of the stage 0–2 feature maps, as head inputs.
   - For end-to-end CNNs, try heads that pool std and max over spatial cells, not only the mean.
   - For multi-crop models, add cross-crop variance.
   - This is the most direct route to the coarse residual. A GPU CNN that sees full-resolution spatial structure is the
     natural next step.
3. Measure local grain size robustly on noisy images: a boundary-density map per 64 px block from a denoised ridge filter,
   calibrated like `cal_` on synthetic degradations of clean train images. Then use max / quantiles / sd of that map. On
   clean images the coarsest-zone statistic (`lf32_max_minus_w`, `lf*_sd_rel`) is the strongest single cue.
4. Treat the morning floor (about 11) as invalid. The b/N term is at least partly model error, and LB 9.24 shows much more is
   recoverable.

New scripts: a17, a17b, a17c, a18, a18b, a20–a32. Figures: `m_cellstd_probe_coarse.png`, `m_graincount_heterogeneity.png`,
`m_clean_extreme_resid.png`, `m_ecs_b0std_extremes.png`, `fig_clean_scatter.png`.

### Train vs test shift of the new features (unsupervised; per-column KS, standardised mean difference)
* The 35 columns of the three files show no meaningful shift: max KS 0.092 (`ecs_blk1_mean`, p 0.007, the only p < 0.01)
  and |SMD| ≤ 0.14.
* Test images are slightly *more* heterogeneous: SMD +0.10 for `ll2_edge_sd`, +0.14 for `ecs_blk0_std`. If the effect
  is real, the gain should carry over to the LB, or slightly exceed CV.
* NaN rate of `lf_*`: 3.2% train, 3.3% test.

## 2026-10-06 (evening): second hunt on the blend_v4 residual (nested 12.619)
Method: as in the afternoon, with the **base switched to the feat4_v3cal_ridge_het_spat_v4 setup** (`eda_cf.py`).
* The base uses the same 187 columns, RidgeCV and in-fold hetero weights via `src.train_gbm.hetero_weights` on
  `ic_acg_len50_gm`. It reproduces feat4 exactly: CV 12.686 with the same fold RMSEs.
* Its residual correlates 0.983 with the blend_v4 residual.
* Each probe is fitted on the inner cross-fitted residuals and added to the blend_v4 nested OOF.
* Nulls: a constant probe gives 12.619 → 12.616, and 10 noise columns give 12.650. Each candidate also gets a
  permutation null (20–40 row shuffles).
* "Clean" means `ic_ridge_snr > 0.9` (n=131: coarse 56, mid 40, fine 35).

### 1. blend_v4 nested OOF and the variance model (a33)
* `blend_v4_nested_oof.csv` reproduces 12.619. Gaussian-ML fit of v = a + b/N + c/snr on nested residuals:

  | | RMSE | a | b | c | mean b/N | mean c/snr |
  |---|---|---|---|---|---|---|
  | blend_v2 | 12.963 | 0 | 28,227 | 10.9 | 131 | 39 |
  | blend_v4 | 12.619 | 0 | 25,305 | 11.9 | 118 | 43 |

  **b(v4)/b(v2) = 0.896, paired bootstrap 90% CI [0.86, 0.97].** About 10% of the 1/N term has been removed so far,
  and about 90% is left. The b/N term is still 118 of the 159 MSE units.
* Where the gain landed:
  - RMSE by N tercile, all images: 16.33 / 12.05 / 9.59 → 15.79 / 11.89 / 9.33
  - clean images: 15.18 / 9.74 / 7.36 → 13.88 / 9.33 / 6.91
  - So most of the gain is on clean-coarse.
* **1/N vs 1/N_eff.** N_eff = (Σa)²/Σa² is what an area-weighted mean of independent per-grain values would give.
  CV negative log-likelihood, lower is better:
  - All images: b/N_cal 1941.9 beats the area-weighted proxies b·exp(v4c_la_mean)/A (1945.2) and b·L²/A (1946.5).
  - Clean images: b/N_cal 495.2 ≈ b/N_ws 495.6 beat b/N_eff_ws 498.6. In a joint fit N_eff gets weight 0.
  - corr(log N, log N_eff) = 0.91.
  - **The residual variance scales with the number of grains, not the effective (area-weighted) number.** It
    behaves like independent, equally weighted per-grain terms (or model error with that form), not like an
    area-weighted per-grain random hardness.

### 2. Per-grain size x phase, area-weighted multiple-instance view, and N_eff (a34, a35): null
* Per-grain table: `grains2_*`, watershed grains with area ≥ 12 and pore share ≤ 0.5. Recorded per grain: moments,
  solidity, interior median and IQR of rn. Dark means median rn < 0.91.
* Per-image views, all null as cross-fitted probes against the feat4 base. Each was probed both clean-only and on all
  images, giving blend 12.615–12.643 with permutation p 0.24–0.57. RidgeCV takes the maximum alpha and the probe
  collapses to its intercept.
  - area fractions in phase × log-area bins (8 bins, 12–3200 px)
  - the same as number fractions
  - a smooth area-weighted basis (z^0..3 of log area, per phase)
  - grey-level bins per phase
  - elongation per phase
  - N_ws, N_eff, log(N_eff/N)
* Direct clean-image fits of y, for orientation:
  - area-weighted bins 13.56 vs number-weighted bins 14.17 (area weighting is better)
  - both worse than fd + porosity (12.61)
  - feat4 OOF on the same rows: 10.82
  So no nonlinear per-grain f(size, phase) is visible beyond what feat4 has.
* log(N_eff/N) has a consistent within-tercile Spearman with the residual (−0.20 / −0.28 / −0.33 on clean images).
  It is mostly N and size spread again (rho 0.81 with N, −0.78 with v4c_bd_sd), and its probe is null (p 0.17).

### 3. Clean-image extremes of the blend_v4 residual (a36; `m_v4_extremes_clean_{coarse,mid}.png`)
* Scan of 702 per-image features on the 96 clean coarse+mid images:
  - max |rho| is 0.30 (`lf32_sd_rel`); the permutation null of max |rho| has median 0.28 and 95% at 0.37
  - **nothing passes multiple testing**
  - the top 20 are nearly all one family, **local grain-size heterogeneity**: `lf*_sd_rel`, the raw-map spreads
    `v4r_la_{sd,sd4,rng4,max_m_mean}`, `sp_cellcv_count_128`, `seg_area_cv`, at +0.22 to +0.30
  - the calibrated `v4c_la_*` spreads that feat4 uses are weaker on clean images (+0.16 to +0.23)
* Visually, under-predicted clean images are zoned: fine bands through coarse grains, or coarse clusters. Over-predicted
  ones are more uniform, and several mid ones are banded (dark grains in horizontal layers).
  - Banding (`sp_band_32` −0.04) and dark-phase grey level (`gf_lvl_dk` +0.04, `ic_gmm_mu_lo` +0.09) are null.
* **Elongated lens/needle grains by phase: null.**
  - Area-weighted log aspect: dark −0.00, matrix −0.07.
  - Area share with aspect > 3: dark +0.01, matrix −0.06.
  - Solidity, v2/v3 aspect and alignment features: all |rho| ≤ 0.2 with mixed signs between coarse and mid.
  - Medians, under- vs over-predicted thirds: dark aspect > 3 share 0.026 vs 0.029; matrix 0.140 vs 0.175.
  - Strongly elongated, aligned dark grains appear in both groups (coarse #1 under-predicted and #1 over-predicted in
    the montage). The morning impression came from the blend_v2 residual and does not hold now.

### 4. Candidate and CV gain: the SNR-gated raw size-spread is a null at blend level (a37, a38)
* Definition (`a37_gated_spread.py`):
  - g = clip((ic_ridge_snr − 0.7)/0.4, 0, 1)
  - columns g and g·x for x in `v4r_la_{sd,sd4,rng4,max_m_mean,q90_m_q10}`, `lf32_sd_rel`, `lf48_sd_rel`,
    `lf32_max_minus_w`
* Cross-fitted probe: 12.619 → 12.600–12.605 (3–4/5 folds, p 0.05–0.10).
* Fold-paired screen with `--no_save` setups (a38):
  - feat4 ridge 12.686 → 12.675 (3/5 folds)
  - feat3 lgbs 12.789 → 12.780 (2/5)
  - blend: fixed blend_v4 weights −0.004 (2/5); **nested NNLS re-fit over all 37 blend_v4 experiments −0.017 (2/5)**
  - clean-coarse 13.88 → 13.74
  - The screen's base re-fit reproduces 12.619 exactly.
* Not exported. `data/x_gsp.parquet` was deleted; a37 regenerates it in seconds.
* Upper bound for this family: rho +0.44 on 56 clean-coarse images. Even if fully exploited that is worth at most
  about −0.15 overall, and in practice it has to be learned from about 45 training images.

### Other checks this round, all null against the feat4 base
* **4x4 cross-cell spread of CNN maps** (a39; effv2s `c24n_cells`, stages 0–2): exactly at the intercept line
  (p 0.24–0.33).
* **2x2 cross-cell spread from the same extraction**: 12.619 → 12.580–12.586 (2–3/5 folds, p 0.05), and the fine
  tercile gets worse (9.33 → 9.43). The cs2 member and ecs already carry this.
* a40, cross-fitted probes, permutation p 0.05–0.90; none gives a consistent gain:
  - Moran's I and size-field (a29): clean 12.609 (2/5)
  - size–dark-phase co-location
  - number-weighted per-grain grey level per phase, and number fractions in 8 grey bins (motivated by the 1/N
    variance)
  - within-phase size–grey correlation
  - absolute elongation direction
* Grain-boundary depth/width (morning a13): Spearman with the residual −0.02.

### Verdict and recommendations (evening)
* No new visible signal was found. About 90% of the 1/N term remains. Its variance signature (number-weighted, not
  area-weighted) is what independent, equal-weight per-grain terms would give.
* None of the visible per-grain attributes we can measure carries it: size, phase, grey level, shape, orientation,
  neighbours, boundaries.
* This does **not** prove the term is irreducible. A full-resolution model could still find something we did not
  measure. The only remaining measurable residual is the zoning family on clean-coarse images, and it is too small
  and too sparse to move the blend.
* For the ensembler: on coarse images the feat4 ridge member alone beats the blend.

  | | clean-coarse (n=56) | noisy-coarse (snr tercile 1, n=45) | all coarse |
  |---|---|---|---|
  | feat4 ridge | 13.25 | 18.57 | 15.58 |
  | blend_v4 | 13.88 | 18.94 | 15.79 |

  The blend wins on mid and fine. An N-tercile-specific weight might help where a linear covariate stacker did not.
  This is untested by me and needs a nested check.
* Scripts: `eda_cf.py`, a33–a40. Outputs: `blend_v4_nested_oof.csv`, `a35_mi_spearman_clean.csv`,
  `a36_scan_clean_coarse_mid.csv`, `m_v4_extremes_clean_{coarse,mid}.png`.

### Addendum (evening): per-grain interior texture and boundary contrast, null (a41, a42)
Raw image divided by the local matrix level; watershed grains with interiors eroded 3 px; number-weighted per phase.
Cross-fitted against feat4 on clean images (snr > 0.9) and on all images.
* **Interior texture ("etch roughness")**: within-grain std, the MAD of the 3x3 residual, and G0.7−G2 band energy.
  - Normalised by the wavelet noise σ, by noise measured inside pores, or by spread across grains, with a
    sampling-corrected excess.
  - Also the dark − matrix difference and the within-image relation to size and level.
  - Every probe group is at the null (permutation p 0.10–0.71).
  - Spearman with the blend_v4 residual: max |rho| on clean coarse+mid is 0.21 (null median 0.24); on all clean images
    it is 0.15 (null median 0.20).
* **Boundary contrast per grain** (mean darkness of the grain's own watershed-line pixels, against its interior and
  against the matrix level, from both the raw-based and the NLM image):
  - median, spread across grains, and the relation to size, per phase
  - null: the best group, all boundary features on all images, gives 12.603 (4/5 folds, p 0.24)
  - matrix boundary darkness tracks y (rho +0.36 on clean images) but not the residual (−0.10); feat4 already explains it
* Nothing exported.
