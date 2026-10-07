# Resume guide (updated 2026-10-07 ~17:50 KST)

Goal: public LB RMSE <= 10.

## Resumed 2026-10-07 afternoon - read this first
- The user resumed after the move; the laptop is back on the branch with Remote Control in this thread.
- **blend_v16 scored public 11.7830** (user), best so far but only -0.013 vs v14 (11.7960). The nested-OOF simulation of
  300-image subsets expected -0.28 (90% [-0.47, -0.09]); landing at -0.013 or worse has probability ~1%, so one of v16's
  two changes probably does not carry over to public: (a) the cell NNLS (CNN 0.2 -> 0.61 + LightGBM in the fine-noisy
  cell) or (b) the out-of-cell NNLS that drops the CNN (0.2 -> 0) and reweights the feature members. (b) runs against
  the public CNN trend (public has favoured the CNN more than CV since v7).
- **5th submission of 2026-10-07: blend_v17** (`--mode nnls --other share --share-other 0.2 --members <v16 members>`):
  v16 on the 284 cell images, v14 on the other 716; nested CV 12.291 (v14 12.425, v16 12.143). It replaces the
  pre-registered v15 because it splits v16 exactly: in-cell effect = v17 - 11.7960, out-of-cell effect = 11.7830 - v17
  (exact in public MSE). Conditional on v16's score the simulation expects v17 ~11.74 (sd 0.05), P(v17 < v16) 0.78;
  v15 ~11.74 too. One public score moves +-0.07 on differences this size, so read it as a direction:
  v17 <= 11.75 -> the out-of-cell change hurt, keep the CNN outside the cell in later blends;
  v17 >= 11.80 -> the in-cell change did not carry over, v16 stays the reference; in between -> both small, v16 stays.
- **Value of the section-8 GPU run, estimated 2026-10-07 ~14:30 KST** (3 analyses + an adversarial check; scripts and
  reports in hardness-cache `scripts/gpuq-1007/`, REPORTS.md): the best blend with all section-8 outputs is expected to
  beat v17 by only ~0.015 public (P > 0.02 ~30%, P > 0.05 ~10%; public SE of such a change ~0.02). Reasons: 3->6 seeds
  and ConvNeXt left the v17-style blend unchanged (+0.004 / +0.03); inside the fine-noisy cell restored-image features
  are worse than raw ones (ridge +0.33) and do not correct the CNN's errors, so g has ~15% chance of a 0.3 cell gain;
  i (2x, 112 crops) is weakest: cell grains are already ~14 px, the residual is noise, small crops lose context;
  h1/h2 full-data ~-0.005 to -0.01 (fold averaging already holds most of it) and cannot be validated by CV. Simulated
  new members were ~0.024 nested too optimistic (they leak y), calibrate any such simulation on real CNN additions.
  Told the user it is not worth 2-3.5 h of the shared GPU; if run, g only (~1 h), and decide after v17's score
  (v17 >= 11.80 means cell gains do not carry over to public, so a better in-cell CNN likely won't either).
- Still waiting on the user: the GPU line (exclusive or shared mode) for `notes/handoff/laptop-gpu.md` section 8
  (run order g, i, h1, h2; 2-3.5 h). The laptop saw a robot-project process on the GPU after resume. No GPU run is approved.

## Evening of 2026-10-07: gap to LB #1 (8.963) - read this first
- Full reports and scripts: hardness-cache `scripts/gap-1007/` (REPORTS.md; workflow of 9 agents).
- #1 cannot get 8.96 from noisy images alone: clean COARSE images (RMSE 13.7-14.8 even in our cleanest) must improve.
  At least 30-50% of the clean "b/N label noise" is predictable model error (our b ~25k; #1 needs <= 18.7k).
- Found: a within-image grain-size heterogeneity term. het4 = sd over the 16 blocks of a 4x4 grid (64 px) of
  -2 log(mean a^-1/2) of grain areas; q2_fcv (CV of quadrant grain counts) is the simple version. With it, Hall-Petch
  gets the normal sign (finer = harder): the anti-Hall-Petch fit came from omitting it. Gated to clean images (raw
  ic_noise < 7.49): v18 + cross-fitted offset 11.920 (p=0.25), member route (feat6 + unpenalised gate*het4,
  gate*N^0.25, p=0.5, pre-registered) nested 12.045 (-0.230, 5/5). Verification: the 64-px alignment is NOT special
  (non-aligned grids do about as well), full nested grid selection gives -0.232 [-0.51, +0.02], the gain is carried by
  ~10 coarse clean images (median gate error gets worse). Honest effect -0.2 to -0.25 CV. Next: a shift-invariant
  version (averaged over grid offsets/sizes, interior grains) before using it in a bold slot.
- **Follow-up (18:30 KST): the het gain is mostly selection.** A pre-registered shift-invariant version (mean block-sd
  over 249 grids, mirror-corrected border grains) gives only -0.017 nested (hardness-cache `scripts/het-1007/`, inv/).
  With VISIBLE (uncorrected) border-grain areas it returns to -0.18 (post hoc: the label may be computed on the canvas,
  border-cut grains counting as small). 80%+ of every variant's SSE gain sits in ~10 coarse clean images (like v12).
  No blend_v19 was made. Restored images do not measure het in mid images (fidelity corr 0.20; blur merges grains),
  but the clean pipeline (filter sigma 1.0) on RAW images measures het/N well up to ic_noise 9.5 ("lower mid", ~100
  train / ~207 test): those images were never used to choose het4, so they are the out-of-sample test (running).
- **Out-of-sample check (19:00 KST)** on the 53 lower-mid train images (raw ic_noise 8.0-9.5) never used to choose het4,
  coefficients fitted on clean images only: RMSE 11.385 -> 10.119 (-1.27, 90% CI [-2.97, +0.60], 4/5 folds, median
  |e| 6.76 -> 5.24, 33/53 better), but the top 5 images carry the gain: fails the pre-registered bar (scripts in
  hardness-cache `scripts/conf-1007/`). Still handed over as a BOLD slot: **blend_v19** = v18 + offset on images with
  raw ic_noise < 9.5 (`python -m src.het_blocks; python -m src.het_offset --base blend_v18 --out blend_v19`), nested
  11.754 (selection-inflated), test offset sd 5.0 on 519 images, public SE vs v18 ~0.21, so one score can only show
  a big effect: v19 <= v18 - 0.2 -> het is real, extend it (noisy images, member route); >= v18 + 0.2 -> drop it.
- **Grid-aware CNN screen (19:10 KST), null:** `src.train_cnn --pool grid4` (crop 256, mean+sd over the fixed 4x4 grid of
  64-px cells), resnet18 CPU folds 0-1: 13.665 vs 13.743 for avg at crop 256 (-0.08, CI [-0.68, +0.55]); residuals of
  both still correlate +0.45 with het4 on low-noise images, so the head does not learn the grid term (details in
  notes/handoff/cnn-trainer.md). A GPU effnetv2-s version is not expected to change that.
- **blend_v20** (optional third slot for 2026-10-08): v18 with the cell's effnetv2-s at a fixed share 0.8
  (`--cell-share 0.8`), nested 12.277, public SE vs v18 ~0.034, calibration expects ~-0.01. v20 <= v18 - 0.07 would mean
  the in-cell CNN premium is large (then a dedicated cell CNN is worth GPU time).
- Noisy images: noise/blur destroy little information; the noisy preset compresses dark-phase contrast by ~half
  (offset -8.7 vs -17.5 grey), which is what loses the signal. Noise-robust measurement families (ACF length, histogram
  deconvolution, dark-tail pores, structure tensor) gave no significant gain. A learned grid het estimator: null in
  noisy, small real signal in mid (-0.05 to -0.10, not grid-specific).

## Afternoon of 2026-10-07: v17 scored, file for 2026-10-08 00:00 KST
- **blend_v17 scored public 11.7285** (user ~15:04 KST), best so far. Exact decomposition: in-cell change (v16 cell)
  -0.0675 public (CV -0.134), out-of-cell change of v16 (no CNN, restored ridge + embedding) +0.0545 (CV -0.148).
  Leaderboard #1 at 15:05 KST: **8.9630**, so the old "perfect model ~10.5" estimate is wrong; a gap investigation is next.
- Public calibration model (scratchpad `v18/calib/predict_public.py`, backed up in hardness-cache `scripts/v18-1007/`):
  public MSE(p) = c + mean_test[(p - t)^2], t = B + kappa (C - B), B = OOF-NNLS cell/outside blend (= v16 on test),
  C = CNN mix; kappa 0.32 [0.10, 0.54]; leave-one-submission-out error 0.053 vs 0.136 for "CV - 0.6". A "feature
  selection optimism" model fits worse. Implied: the CNN mix alone would score ~12.08 public (OOF 13.16).
  Candidates that are linear mixes of v7/v14/v16/v17 (all v14-outside share variants, v16-base + CNN outside) have
  their public score already pinned by the existing scores (+-0.004-0.015): submitting them teaches little.
  Selected feature-side gains carry over to public at about 0 (-0.43 +- 0.35); CNN share earns a public premium.
- **File for 2026-10-08 00:00 KST: blend_v18** (`--cnn-nnls cnn_ev2s_rawnlm_degcons_gpu_s6:1`, nested CV 12.2752, -0.016
  vs v17 in 4/5 folds; only the 284 cell images change). Expected about -0.01 vs v17 (a tie). Plan for the other 4 slots:
  the user wants only 1-2 safe submissions per day (said 16:12 KST), so v18 is the safe one and the other 4 go to bold
  attempts whose public score is not already pinned: gap-investigation features/models first, in-cell CNN share 0.8-1.0,
  the restored-channel CNN if the user grants the GPU tonight. Fallbacks in
  scratchpad `v18/cands/` (`cmb_Iev2s_Oc` = cell as v18, outside 50/50 v14-outside and v16-base + CNN 0.2, nested
  12.230 but outside reweighting is the kind public punished; `in_Iev2s_sh80` = more CNN in the cell, CV flat).
- GPU section 8: judged not needed now (restored-channel CNN g ~+0.013 expected, P(>0.03) ~0.2; h1 full-data ~+0.009,
  not CV-checkable). The laptop finished the CPU prep (8b, banks match the cloud: 1953 pairs, rms 2.23); GPU steps wait
  for the user's mode (exclusive = user stops the robot job; shared = user allows gpu_turn.py in the PC's /permissions).

## Morning of 2026-10-07 (read this first)
- **blend_v11 scored public 11.7997** (user, ~04:30 KST), the new best. CNN share 0 / 0.07 / 0.131 gave public
  11.8795 / 11.8267 / 11.7997 (v7 / v10 / v11), each time about CV - 0.55 to 0.61. The CNN helps public more than CV
  says: the CV-optimal share is 0.131, a parabola through the three public points has its minimum near 0.19.
- **blend_v12 scored public 11.9094** (user, 04:25 KST), worse than v10 (11.8267) and about v7 (11.8795), although its
  nested CV was 0.26 better (public only CV - 0.25). Not a bug: own columns are applied to the 333 high-noise test
  images exactly as to train (feat7 - feat6 differences have sd 5.2 on both). On OOF the v12 - v10 direction correlates
  -0.21 with v10's errors; on public about -0.03. **Cause unknown.** The OOF gain sat in about 10 images (73% of it),
  but that does not explain the miss: v4 -> v6 (top-10 share 0.79) and v6 -> v7 were as concentrated and carried over.
  Candidates: overfitting that 5-fold CV cannot see (the member was picked among several own-slope variants on the
  same folds), or luck on the ~300 public images. Keep own-slope members out of submissions until one of these is
  settled. blend_v13 (also uses that member) stays shelved.
- **blend_v14 scored public 11.7960** (user, ~04:39 KST), a tie with v11: the global CNN share is at its public optimum
  (a parabola through v7/v10/v11/v14 bottoms at share 0.18, 11.794). Do not push the global share further.
- **Next file: blend_v15** (`python -m src.blend_cells ... --share-other 0.2 --share-cell 0.7 --out blend_v15`, nested CV
  12.327 vs v14 12.425, 4/5 folds): v14 with CNN share 0.7 only for fine-grained AND noisy images (log calibrated grain
  count density and raw ic_noise above their train medians; 132 train / 284 test images). On OOF the CNN mix beats the
  base there (11.33 vs 12.04) and loses everywhere else (13.76 vs 12.56; coarse or clean images get LS share <= 0). All
  25 cut combinations (train quantiles 0.33-0.67) improve on v14; the change correlates -0.23 with v12's failed change.
  Public SE of the v15 - v14 difference is about 0.06. Smooth gates (share = clip(a + b.grain + c.noise)) reach nested
  12.28 but switch steeply between 0 and 1 and move test predictions twice as much (sd 2.0): kept as a later option.
- **blend_v16** (`python -m src.blend_cells --cnn $C --mode nnls --members feat6_rest_ridge_all,feat5_v3cal_ridge_het_spat_v4_milspl,
  feat3_v23cal_lgbs_het_spat,feat2_v23cal_lgbs_hetN,emb_effv2s_256_gridge3_noise_cs24 --out blend_v16`, nested CV **12.143**):
  separate NNLS weights in the fine-noisy cell (CNN mix 0.61 + LightGBM 0.39) and elsewhere (restored ridge 0.59 +
  embedding head 0.38 + LightGBM 0.03, no CNN). Half of the gain is outside the cell (the global weights were a
  compromise: LightGBM is the best feature model in the cell, 11.57, but poor elsewhere). Weights are stable across
  folds; re-discovering the split inside each outer fold (median cuts, CNN-share rule) picks the same cell in all 5
  folds (12.145); random 132-image cells do not help (+0.03). Offered to the user as the 4th submission instead of
  v15 (or as the 5th if v15 was already in). Not worth it: 2x2 or tercile regimes (overfit), bigger member pools per
  regime (-0.02 to -0.07, noisy; regime 1 then picks old members such as feat2_v3_lgb).
  The ensembler's covariate stacker (null on 2026-10-06, before the GPU CNNs) gives only -0.06 to -0.09 on these
  members: its linear/tercile forms cannot express the fine AND noisy cell.
- Where blend_v16 still loses (nested OOF RMSE by calibrated grain-count tercile x raw ic_noise tercile): coarse-noisy
  18.2 (every member 18.2-20), mid-noisy 13.9, coarse-midnoise 14.2; clean cells 7.9-13.0. In the noisy third the
  predictions are too flat (slope of y on prediction 1.21) and dark-fraction measures correlate about 0.55 with
  hardness vs 0.71 on clean images: the loss is measurement under heavy blur + white Gaussian noise (sd 16-21).
- Null on top of v16 (2026-10-07 morning): degraded copies of clean train images as extra labelled rows (ridge/LGBM
  worse), multi-scale GMM/threshold dark fractions (cal_ features stay best in noisy images), a noise-dependent stretch
  (-0.035, 3/5 folds) or global stretch (-0.019), absolute elongation direction (|rho| < 0.04 with the residual).
- The train-vs-test adversarial AUC of about 0.6 comes only from the 174 calibration-source train images:
  `src.features.cal_apply` fits them in-sample (identity rows), so their cal_ values differ from comparable test images
  (AUC 0.72; other train images vs test 0.54). It does not matter for predictions: validating with cross-fitted cal_
  values changes member CV by -0.01 to +0.03, and retraining on a cross-fitted table is not better (scratchpad
  `calxf/`). v4/v5 calibrations were already cross-fitted.
- CNN prep committed (5e51ca1, cnn-trainer): `--input raw+nlm+rest` (restored channel), `--full` (all-train models,
  test predictions only), `--scale S`; `src.restore --device cuda`. Laptop plan: `notes/handoff/laptop-gpu.md`
  section 8, run order g (restored-channel effnetv2-s, 3 seeds), i (`--scale 2 --crop 112`, 3 seeds), then h1/h2
  (full-data effnetv2-s x6, ConvNeXt x3). Asked the user ~05:55 KST for the GPU mode; nothing runs until they write
  the line in the thread. Judge new CNNs by the fine_noisy cell RMSE (ev2s_s6 11.26) and by v16-style blends.
- Full refit on all 500 train images (one model instead of the mean of 5 fold models) is worth little for the ridge
  members: one model on 400 rows vs the mean of 5 inner models on 320 rows gives -0.035 (feat6_rest_ridge_all) and
  -0.001 (feat5_v3cal_ridge_het_spat_v4_milspl). LightGBM members and the embedding head are being checked
  (scratchpad `refit/`, copies in hardness-cache `scripts/refit/`).
- Averaging gain of the CNN test predictions (mean of 5 fold models vs one fold model on OOF): from the CPU resnet18
  caches only 2-5% of the OOF MSE, too small to explain why public prefers a CNN share of 0.18 vs the CV fit 0.10.
- Next GPU candidate: a CNN with a restored-image input channel (cnn-trainer prepares the code and laptop steps,
  `notes/handoff/laptop-gpu.md` section 8). A GPU run needs the user's say-so and mode in the thread first.

## Night of 2026-10-07
- **blend_v12** (nested CV 12.162 vs v11 12.407, 5/5 folds, paired bootstrap 90% of the difference [-0.41, -0.09]) is the
  next submission (delivered ~02:15 KST). New member `feat7_rawn3own_ridge_all` (weight 0.59): the feat6 restored ridge
  plus "own columns", raw v3+cal copied as n3_* only for images with raw ic_noise > 11.916 (train top-tercile cut; NaN
  elsewhere, median-imputed in fold), so the noisiest third gets its own slopes (`python -m src.features --n3own`).
  Member CV 12.53 -> 12.26; random 500-image "own" groups never help, so the gain is specific to the noise split.
  Gain is mostly in the high-noise tercile (blend 14.85 -> 14.37).
- Null that night: the high-noise specialist restorer (`src.restore --tag hn`, `data_restored_hn/`) is better on
  synthetic pairs but its features rank hardness worse on real noisy images; effnetv2-s 6 seeds and ConvNeXt-tiny add
  nothing to the blend (laptop-gpu.md section 7).
- Follow-ups that night (all fold-paired, then checked in the two-stage blend):
  - N3 own columns also help the other ridge members (feat7_rawn3own_ridge_splmean 12.365, _milspl 12.323, _add
    12.340), but next to feat7_rawn3own_ridge_all they add nothing to the blend (errors correlate 0.97-0.99).
  - Null: own slopes for the mid-noise tercile (+0.44), smooth noise interactions x*t (+0.16 / +0.27), a bigger own
    block (v4, v5), own block from an NLM view or either restorer (raw calibrated measures rank hardness best in the
    noisiest third), own slopes in the cs24 embedding ridge (1088 own columns on ~133 images per fold overfit).
  - Ties: ic_noise in the ridge's variance model (feat7_rawn3own_ridge_all_hetnz, blend 12.159) and a noise-weighted
    embedding fit (emb_..._cs24_hetnz): **blend_v13**, nested 12.136 (4/5 folds vs v12, 90% [-0.051, -0.003]) but its
    test predictions differ from v12 by sd 0.19, so it is a public-LB tie with v12. Keep it as a later option.
- Next ideas: wait for v12's public score first (does the own-slope gain carry to test?). The noisiest third is still
  14.37 vs 11.30 / 10.46. Untried: a dedicated model for the noisiest third that borrows strength from the rest
  (e.g. ridge on the shared blend prediction + own block), and noise-dependent CNN shares.
- Two-stage blend script: hardness-cache `scripts/twostage.py` (`--add` extra pool members, `--stage2` CNNs,
  `--save-nested`, `--out`). Backups of that night: `experiments-test-extra/`, `data/data_restored_hn.tar`,
  `data/restore_cache/hn/`, the new `data/features_*` files, `submissions/blend_v12-13`.

## Late 2026-10-06
- **blend_v10 scored public 11.8267** (user, 23:33 KST), the new best: 0.053 below v7 although the nested CV tied
  (12.4242 vs 12.4254). The CNN direction helps on test more than OOF shows: v10 - v7 on test has sd 0.30, so for a
  0.053 drop it must correlate about -0.19 with v7's public errors (if the public subset looks like the whole test),
  vs about zero on OOF where the CV tied. One public result, about 2-3 SE.
- Next file: **blend_v11** (`submissions/blend_v11.json`), two-stage: v7-pool NNLS, then the effnetv2-s CNN at a share
  fit by least squares on the inner nested base OOF (refit per outer fold). CNN share 0.131, nested CV 12.4074 (beats
  joint NNLS 12.4242). Handed over for the first submission after 00:00 KST 2026-10-07. On test it moves about 1.9x
  further along v10's direction (v11 - v7 sd 0.60, correlation 0.96 with v10 - v7): expect public about 11.79 if the
  CNN effect is linear in its share, about 11.86 (v7 minus the CV gain) if v10's drop was luck.
- Next GPU runs (user picks exclusive or shared GPU mode in the thread first): `notes/handoff/laptop-gpu.md` section 6.
- Where the noise error is (orchestrator, 23:50 KST): blend_v11 OOF RMSE by calibrated grain count
  (`cal_seg_count_density`, Spearman 0.02 with noise) x raw `ic_noise` tercile (train cuts 7.49 / 11.92):
  fine 7.75 / 7.89 / 11.36, mid 8.82 / 11.03 / 14.15, coarse 13.43 / 14.14 / 18.19. A fit r^2 ~ a + b/N + c*noise gives
  c = 9.0 (90% CI 5.3-13.0): about 56 of the ~151 MSE units are tied to noise, in every grain-size tercile.
  Do not stratify by `ic_acg_len50_gm` or `ic_seg_L_gm` for this: they are biased by noise (Spearman 0.43 with
  ic_noise) and make the noisy tercile look like a coarse-grain confound.
- Cloud CPU next (running): idea 1 below, a high-noise specialist restorer (cnn-trainer; outputs to
  `data_restored_hn/`, git-ignored), then restored features and a ridge screen (feature-engineer).

## End of day 2026-10-06
- Public LB so far: blend_v2 12.3454, v4 12.0329, v6 11.9248, **v7 11.8795** (best). In all four, public came in
  0.55-0.6 below the nested CV. LB #1 that day: 9.2442.
- File handed over for the remaining slot (today's last or 00:00 KST 2026-10-07): **blend_v10** = v7 pool + the two
  laptop-GPU CNNs. Nested CV 12.4242 vs v7 12.4254 (bootstrap 90% of the difference [-0.032, 0.031]), so it is a tie:
  weights cs24 0.244, feat6_rest_ridge_all 0.393, feat3 lgbs 0.166, feat5 milspl 0.089, feat2 lgbs hetN 0.038,
  effnetv2-s CNN 0.07. Test predictions differ from v7 by sd 0.30 (max 1.25), so expect public about 11.88.
  CSV in `/mnt/project-files/work/hardness-cache/submissions/`. When the user reports the score, add it to
  `experiments/LEADERBOARD.md` and memory.
- Laptop GPU run (RTX 5060, exclusive use 19:16-20:11 KST, returned with the done file; plan in
  `notes/handoff/laptop-gpu.md`), base flags `--input raw+nlm --deg-p 0.5 --cons 1.0 --pool avg --epochs 30`, lr 1e-3:
  | experiment | seeds | CV | time per fold-seed | blend weight |
  |---|---|---|---|---|
  | `cnn_r18_rawnlm_degcons_gpu` (resnet18) | 3 | 13.593 | ~31 s | 0 |
  | `cnn_ev2s_rawnlm_degcons_gpu_s3` (tf_efficientnetv2_s.in21k_ft_in1k) | 3 | 13.246 | ~76 s | 0.07 |
  CNN residuals correlate 0.93 with the blend; even the best in-sample mix of resnet18 into v7 is weight 0. More seeds or
  bigger backbones of the same CNN are unlikely to move the blend. lr 3e-4 was not screened.
- The laptop session returned results as messages (2-decimal values in ID order plus checksums); the cloud copies and
  checking script are in the hardness-cache (`laptop-gpu-2026-10-06/`, `scripts/ingest.py`, `scripts/paired.py`).
- No CPU candidate beat v7 either (restored-image embeddings, restored-feature GBM, add-alongside ridge, tercile
  stacker, stretch recalibration, curated subsets).
- Where the error is: the noisy SNR tercile (blend about 14.7 vs about 11 on the other two) and coarse-grain images,
  whose extra variance tested as per-grain label noise the image does not show (every probe null; list in memory).
- Untested ideas for the next session, best first:
  1. A restorer trained only on low-SNR degradations, then restored features + ridge for the noisy tercile (restoration
     so far helps the mid tercile, not the noisy one).
  2. A CNN on raw + restored channels on the laptop GPU (needs `data_restored/` and `restore_unet.pt` on the laptop).
  3. Only if 2 gets real blend weight: more GPU seeds/backbones.

## Repo history rewrite (2026-10-06 ~11:30 UTC, user's choice)
- The user asked for the work on `main`: main was fast-forwarded to `claude/lb-under-10-7080jr`.
- Before that, 16 early `experiments/*/test.csv`, two EDA tables that copied the train labels
  (`experiments/eda/blend_v*_nested_oof.csv`) and the restorer's per-image test statistics
  (`experiments/restore/apply_change.csv`) were still tracked. They were untracked, git-ignored, and on the user's
  choice removed from every commit; main and the branch were force-pushed. Local copies:
  hardness-cache `experiments-test.tar` and `local-only-files.tar`. Pre-rewrite history: `git/lb-under-10-prepurge.bundle`.
- Old clones (the laptop) must re-sync once with `git fetch origin && git reset --hard origin/claude/lb-under-10-7080jr`.

## Cloud container lesson (pause 2026-10-06 07:05-07:38 UTC)
- The container sleeps a few minutes after the session goes idle and kills every detached job (the restorer's training
  died at step 3500 that day). Checkpoint long jobs (`src/restore.py --ckpt-every`, `train_cnn` per fold-seed caches).
- In a fresh session, re-brief agents from `notes/handoff/*.md`; agent ids from an old session do not carry over.

## Where we are (2026-10-06 evening)
| model family | best saved experiment | CV RMSE |
|---|---|---|
| OOF blend (NNLS, nested CV) | `blend_v10` (ties `blend_v7`, public 11.8795) | **12.42** |
| handcrafted features | `feat6_rest_ridge_add` / `feat6_rest_ridge_all` (restored-image features) | 12.52 / 12.53 |
| frozen embeddings | `emb_effv2s_256_gridge3_noise_csbag_rawrest` | 12.84 |
| CNN fine-tune | `cnn_ev2s_rawnlm_degcons_gpu_s3` (laptop GPU) | 13.25 |
| baseline | `feat_lgb` | 14.20 |

Target std is 17.7 (mean predictor). Submission CSVs are git-ignored; copies are in the hardness-cache `submissions/`.

## Restore a fresh cloud session (about 5 min)
```bash
cd /home/claude/microstructure-hardness-prediction        # repo, branch claude/lb-under-10-7080jr
git fetch origin claude/lb-under-10-7080jr && git checkout claude/lb-under-10-7080jr && git pull
python3 -m pip install -q -r requirements.txt             # torch comes from PyPI; PyWavelets needed by skimage
mkdir -p data && (cd data && unzip -qo /mnt/project-files/open.zip)
cp -r /mnt/project-files/work/hardness-cache/data/. data/  # folds, features v1-v3, embeddings, CNN caches
tar -xf data/data_restored.tar && rm data/data_restored.tar  # restored images -> data_restored/ (repo root)
tar -xf /mnt/project-files/work/hardness-cache/experiments-test.tar   # experiments/*/test.csv (git-ignored)
tar -xf /mnt/project-files/work/hardness-cache/figures-local.tar      # optional: figures that show competition images
mkdir -p submissions && cp /mnt/project-files/work/hardness-cache/submissions/* submissions/
```
Pretrained weights download from GitHub releases on first use (`src.common.create_timm`).
huggingface.co is blocked in the cloud sandbox.

## Cross-agent findings (read before planning)
- All families capture the same signal: residual correlations 0.87-0.93; blends gain only about 0.3-0.6.
  Getting to 10 needs a new signal, not more of the same models.
- Dark-phase fraction dominates (per-grain fraction Spearman about 0.7 on clean images). Normalise by the
  local matrix level: shading (13-19 grey levels) is comparable to the phase contrast (about 30).
- Error rises with image noise: about 12 on the cleaner half vs about 15 on the noisier half.
  Denoising the input hurts embeddings but an extra NLM channel helps the CNN.
- The target is heteroscedastic: coarse-structure images have hardness std 21.5 vs 13.5 for fine ones,
  and OOF RMSE 16.7 vs 9.3. Nothing measured so far explains the extra variance in coarse images.
  This is the biggest open question; candidates: a location-dependent label (e.g. an indent-like local
  sample), a grain-level property not yet measured, or label noise built into the generator.
- Every model regresses to the mean (RMSE about 18 in the extreme target quintiles vs about 9 in the middle).
- With 500 images, differences under about 0.1 RMSE are noise; single-fold screens are about ±0.4.
- `src/train_gbm.py` `feat_lgb` / `feat_cat` used early stopping on the validation fold (optimistic). Newer runs use fixed trees.

## Spatial-heterogeneity signal (2026-10-06 afternoon, eda-analyst; experiments/eda/NOTES.md from line 96)
- Part of the coarse-image "label noise" is model error. Images with zones of large grains next to zones of small grains
  are harder than their mean grain size predicts, and global pooling or global mean-size features average this away.
- Three per-image feature files (train + test, fixed transforms): `data/eda_feats_lledge.parquet` (block heterogeneity of
  edge energy), `data/eda_feats_ecs.parquet` (cross-cell std of g2a embeddings), `data/eda_feats_lf.parquet` (local
  grain-size field; GBMs only). Copies are in the hardness-cache `data/`.
- Fold-paired gains: ridge het 13.101 -> 12.824 (5/5 folds); lgbs het 12.991 -> 12.771 (4/5); blend about -0.18.
- Cross-fitted probes are needed for residual tests: naive OOF-residual probes are biased negative by stacking leakage.

## Noise-floor finding (2026-10-06 morning, eda-analyst) - SUPERSEDED by the section above; the floor is only an upper bound
- Residual variance follows label noise ~ b/N (N = calibrated grain count `cal_seg_count_density*6.5536`,
  b about 28k, 90% CI 21.9k-32.4k) plus a measurement term ~ c/snr. Clean images already sit at that floor
  in every grain-size tercile (orchestrator re-check: clean coarse 14.5 vs floor 14.5, mid 9.2 vs 9.7, fine 7.8 vs 7.8).
- Floor for a perfect model: train CV about 11.2-11.6, full test about 10.7-11.1; public 300 subset median 10.7
  [9.2, 12.2]. LB <= 10 is out of reach on expectation; blend_v2 is expected around public 12.6 [11.4, 13.8].
- All reducible error is in degraded (noisy/blurred) images: focus on measurement under noise/blur
  (calibration v2, NLM channel for CNNs, training on synthetic degradations). Stop adding coarse-image grain,
  location or texture features (all null). Public-LB differences under about 0.3 are noise.

## Compute notes
- Cloud sandbox: 4 CPU (Xeon with AMX), no GPU. bf16 + channels_last makes CNN training about 3x faster.
  Budget threads per agent (e.g. CNN 2, features 1, embeddings 1) when running agents in parallel.
- Laptop GPU (RTX 5060 8GB): this project's device folder is `C:\Daker\microstructure-hardness-prediction`
  (branch checked out, data copied to `data\`, `.venv` with torch 2.11+cu128). The GPU is shared with the user's
  robot project. Ask the user which mode applies before running anything on it:
  - shared: wrap every GPU command as
    `python C:\Dacon\RobotWorldModel_ActionVideo\wm_ops\gpu_turn.py --who hardness -- <command>`; on CUDA OOM lower `--bs`;
  - exclusive hand-over (as on 2026-10-06 19:16 KST): `--device cuda` directly, then return the GPU with PowerShell
    `New-Item C:\Dacon\WM_Runtime\hardness_gpu_done` when all GPU work is done.
  Never touch the robot project otherwise.
- "Jev" (TypeSafe AI) is a remote, text-only API model and is banned by competition rule 2, so it is not used.
