# Resume guide (updated 2026-10-09 ~15:45 KST)

Goal: public LB RMSE <= 10.

## 2026-10-09 15:45 KST: blend_v24 scored 10.5130 (best); section 12 running - read this first
- Public 10/09 (user): blend_v24 **10.5130** (-0.332 vs v23, train-expected -0.22 +- 0.24). Public-optimal scale along
  v23 -> v24 about 1.24 (worth about 0.01): no rescale (hardness-cache scripts/d1009/I/pub_analysis). Estimated private
  gain vs v23 about -0.27 +- 0.16, P(better) about 0.95. Final picks: v24 first; v23 second unless v25 passes.
- GPU: the robot side's hand-over tool (C:\Dacon\WM_Runtime\handover_after_jobs.py <ledger job>) handed the GPU over at
  15:25 KST when robot submit4 ended; the laptop started section 12 at 15:27 (screens -> pick -> 5-fold), return with
  hardness_gpu_done expected 16:30-16:50 KST. User line 14:43: "인계도구로 두 세션이 서로 점유를 주고받는 활동 시작할것".
- Section 12 next steps: 12b pick output; stage 1b (pooled P >= 0.54); then once `python H/h_test.py --het
  data/het_cnn/ev2s_loc12 --name loc12` (vs v24); if PASS `G/g_build.py blend_v25 data/het_cnn/ev2s_loc12`, verify,
  hand over as slot 4. Pre-set reading vs v24 in H/PREREG.txt addendum (<= 10.463 in line, > 10.613 demote).
- Slot 5 of 10/09 stays unused: the design review (scripts/d1009/I) found no CPU candidate worth a label test.
- Reaching <= 10 needs about 10.5 more public MSE, as much as the whole het route gave from v21 to v24; it needs new
  information for the G1 rows with ic_noise >= 12 (nested RMSE 14.1 vs 9.7 clean).

## 2026-10-09 afternoon: blend_v24 handed over (3rd slot)
- Public 10/09 (user): blend_v23 **10.8445** (-0.130 vs v22), blend_v23m15 **10.8393** (v23 with the 9.5-12 line-free part
  of the offset x1.5; public-only hedge, -0.005). Public-scale analysis (hardness-cache scripts/d1009/Q): public alone
  wanted that part x2.5-3, train says x0.9, combined private optimum x1.2-1.6; after v23m15 the public optimum is ~x1.3.
  No more rescaling.
- Laptop section 11 (GPU exclusive 12:57-13:47 KST, returned with the done file, liaison confirmed): localized het CNN
  ev2s_loc_e32r4 + _s1. Stage 1: render partial corr 0.49 / 0.50 (old 0.25), within 0.71 (old 0.43), real clean
  held-out 0.72 mean (old 0.29). Stage 2 (G/PREREG.txt, run once on the seed average): G1 13.770 -> 13.385, 4/5 folds,
  perm p 0.0005, drop-top 0.298: PASS. Dumps + parquets: hardness-cache laptop-gpu-2026-10-09/.
- **blend_v24.csv** (md5 f4953098..., hardness-cache/submissions; builder scripts/d1009/G/g_build.py): nested CV 11.465
  (v23 11.653). Independent verification (scripts/d1009/V24: reproduce, audit, skeptic, judge): exact, no leakage;
  all the gain is in raw ic_noise >= 12 (5/5 folds), the 9.5-12 band is neutral; concentrated in a few coarse-grain
  rows; bootstrap P(no gain vs v23) ~7%. Expected public 10.64-10.69 +- 0.2. Reading: <= 10.744 carried over;
  10.744-11.108 not shown (keep v24); >= 11.108 harmful (back to v23). Final picks: v24 + v23 (safe).
- Next lever (asked the user for GPU, decision card 14:15 KST): laptop-gpu.md section 12 screens 4 stronger localized
  designs (lambda-within 2 / 4, 64 epochs, 8 renders) on folds 0-1, picks by src/het_cnn_pick.py, trains the pick on 5
  folds; stage 2 vs v24 pre-registered in scripts/d1009/H (PREREG.txt, h_test.py sha 9cc5fc76...). If PASS:
  blend_v25 via G/g_build.py on the new dir. The laptop needs the user's own line in the thread to start.

## 2026-10-08 20:30 KST: paused for the user's move (laptop off) - read this first
- **On "재개":** the laptop runs `git pull origin claude/lb-under-10-7080jr`. If the user has written a GPU grant in the
  thread ("GPU 사용 허가", exclusive or shared), the laptop runs `notes/handoff/laptop-gpu.md` section 11 (a -> b -> c ->
  d -> e). Section 10 is superseded (dark-phase fraction is not the noisy-image problem). No laptop job was running
  at the pause; nothing on the laptop needed backing up (only CPU smoke outputs).
- **10/09 files so far:** `blend_v23.csv` is the safe file (md5 15fbe5e8...; in hardness-cache/submissions). Verified
  2026-10-08 evening (hardness-cache scripts/d1009/V1, V2, P23): exact reproduction, no leakage, but its gain over
  v22b comes from one train image (TRAIN_000239); expected public about -0.02 vs v22, a tie. d1009 X1 (whole-image
  noise-robust size-spread statistics) failed its pre-registered test.
- **Main lever in progress:** localized het CNN for line-free noisy images. `src/het_cnn.py` (92d6ebc) gained `--head
  fpn`, `--mosaic/--mosaic-dlogn`, `--zoom`, `--lambda-within`, `--screen` (defaults bit-identical). Pre-registration:
  hardness-cache `scripts/d1009/G/PREREG.txt` (stage-1 design pick from a CPU screen, GPU stage-1 gate, stage-2 bar on
  G1 vs blend_v23: >= 0.10 better, 4/5 folds, perm p < 0.05, survives dropping the top row).
- CPU screen (cloud, resnet18, fold 0, 16 epochs; hardness-cache `scripts/d1009/G/screen/`): the within-image
  deviation loss lifts render partial corr 0.20 -> 0.47-0.49; the combined design was picked. Its 5-fold CPU version
  (`cpu_loc_r18_e16`, render pc 0.35) FAILED stage 2 at 00:15 KST 10/09: G1 13.770 (v23) -> 13.848, 2/5 folds, perm p
  0.021 (signal, but weaker than the old GPU estimator). Render gains did not carry to the real labels. The GPU
  effnetv2-s run of the same design (section 11) is the last pre-registered test of this route (p bar 0.025).
- Told the user at 00:20 KST 10/09: submit blend_v23 first on 10/09 (safe); GPU grant needed on return.

## 2026-10-08 17:33 KST: v21 11.0620, v22 10.9742 (best) - read this first
- Public (user, 17:33 KST): blend_v21 11.0620 (-0.013 vs v19), **blend_v22 10.9742** (-0.088 vs v21). By the pre-set
  reading (<= -0.04) the GPU het-CNN estimate carries real signal on the noisiest images (raw ic_noise >= 12); v22's
  public gain beat its CV gain (-0.068). CV-to-public offset for v22: 0.714. All 5 submissions of 10-08 used.
- LB (user, 17:33 KST): #1 8.8662, #2 8.9630, #3 9.6509. Task now: improved files for 2026-10-09, "아주 신중한 판단으로".

## 2026-10-08 evening: files for the last 2 slots of the day
- **Submit in this order** (files in `/mnt/project-files/work/hardness-cache/submissions/`, also attached in the thread):
  4. `blend_v21.csv` (safe, nested CV 11.756, md5 a57dc832...): v19 + v20's in-cell CNN share, cell CNN test averaged
     half-and-half with the laptop full-data runs. Expected ~11.03-11.04.
  5. `blend_v22.csv` (exploratory probe, nested CV 11.688, md5 0ab58a1a...): v21 + beta * (het_cnn - g(logN_cnn)) on
     the 326 test images with raw ic_noise >= 12 (165 train), g linear and beta = 103.5 fitted on those train rows
     against the v21 nested OOF residual. FAILS its bar: gated 14.658 -> 14.492 (-0.165), perm p 0.02, but 3/5 folds
     (fold deltas -0.41 / +0.002 / -0.45 / +0.54 / -0.55; need 4/5), and the design (gate >= 12, residualised form)
     was chosen after seeing the primary test. Read v22 - v21 on public: <= -0.04 -> the CNN het estimate carries
     signal on noisy images, worth more GPU seeds / a better renderer; >= +0.04 -> drop the route; between -> undecided.
- GPU het CNN (`src/het_cnn.py`, laptop RTX 5060, exclusive hand-over 16:30-17:22 KST, returned with hardness_gpu_done):
  effnetv2-s, 32 epochs, 4 renders per clean train image in the noisy-preset style, targets = the 16 block b values and
  log N_eff measured on the clean originals (y never read). Seed 0: render corr 0.80, partial corr | logN 0.25; seed 1
  0.74 / 0.14; test het_cnn of the two seeds correlate 0.988. Dumps: hardness-cache
  `laptop-gpu-2026-10-08/het_ev2s_e32r4/{train,test}.csv` (seed 1 stays on the laptop in data\het_cnn\ev2s_e32r4_s1).
- Pre-registered primary stage 2 (`src/het_cnn_offset.py`, gate 9.5, OLS on [1, het_cnn, exp(0.25 logN_cnn)] on
  blend_v19): noisy 13.721 -> 13.799, 2/5 folds, FAIL (9.5-12 band 11.25 -> 11.77, 12-15 16.08 -> 15.95, 15-21 13.98 ->
  13.94). Secondary stage 2b and the v22 builder: hardness-cache `scripts/het-1008/stage2b/{stage2b.py,build_v22.py}`
  (`python build_v22.py --train <seed csv...> --test <seed csv...>`). Averaging seed 1 would flip the fold count to 4/5
  in ~40% of simulated draws (fold 1 is a tie), but moves the file itself by far less than its public noise.
- Also null today under pre-registered bars: clean-het refinement (`src/het_refine.py`, 9.717 -> 9.694, 3/5 folds) and
  the member route (het columns inside the five members, blend_v22m 11.794 vs v21 11.756).
- Next (after the user's v21 / v22 scores): noisy images still hold ~63% of the squared error. If v22 helps, train more
  het-CNN seeds and widen the renderer's noise range; if not, the noisy-image het estimate needs a different input
  (e.g. denoised or restored views) rather than more seeds.

## 2026-10-08 afternoon: blend_v19 scored 11.0751 (best)
- Public scores for 2026-10-08 (user, 14:45 KST): v18 11.7304 (tie with v17), **v19 11.0751** (-0.655 vs v18, so the
  het term is real by the pre-set rule), v20 11.6924 (-0.038 vs v18, below the 0.07 bar for a dedicated cell CNN).
  3 of 5 used; 2 left for the day.
- `src.het_blocks` het4 breaks right above its gate. Partial corr with the v18 residual given log N_eff: 0.52 / 0.48 /
  0.46 for raw ic_noise <6 / 6-8 / 8-9.5, but -0.13 (9.5-10.5), -0.20 (10.5-12), +0.15 (12-15), -0.02 (15-21). Applying
  the clean coefficients or a band-specific cross-fitted OLS makes every band above 9.5 worse, so the gate stays at 9.5.
- Laptop: full-data CNNs finished 00:35 KST (effnetv2-s 6 seeds, test sum 193318.54; ConvNeXt-tiny 3 seeds, test sum
  194022.17); test files in hardness-cache `laptop-gpu-2026-10-08/full6`, `full3` (no OOF). The GPU is held by
  RobotWorldModel again; the user will say in the thread when it is free (mode not chosen yet).
- Plan for the last 2 slots (user: "남은 두 파일은 GPU가속과 그 외 가능한 최대 심사숙고를 통해서 v19 v20만들것"):
  v21 = v19 + v20's in-cell CNN share; v22 = v21 + het for the noisy images from a CNN trained on clean train images
  re-rendered in the noisy-preset style (targets measured on the clean originals, never y), under a pre-registered bar.

## Paused 2026-10-07 20:10 KST for a PC restart - read this first
- The user paused all activity ("이 세션의 모든 활동 임시중단. pc 재시작 후 재개할수 있게 모든 스레드에 준비할것").
  At the pause: no cloud job running, git clean (branch = main = this commit's parent chain), every file below backed up.
  Laptop: no GPU job running (RobotWorldModel held the GPU, 99% / 6 GB with gpu.lock); section 8 g (3 seeds) and h
  (full data) are not started, all inputs cached on the laptop (restorer retrain PSNR 32.01, 1500 restored images,
  restored deg bank, fold-0 screen 13.724 vs 13.635). They need the user's GPU go-ahead (~1-1.5 h) and are worth ~0.015.
- **Submissions for 2026-10-08 (all 5 used on 10-07; count resets 00:00 KST), in this order, each scored before the next:**
  1. `blend_v18.csv` (safe, md5 3e095ae7...): v17 with effnetv2-s alone as the cell CNN; expected ~11.72 (tie with v17).
  2. `blend_v19.csv` (bold, md5 4ce52e76...): v18 + grain-size heterogeneity offset on the 519 low-noise test images.
     Read vs v18: <= v18 - 0.2 -> het is real, extend it to noisier images; >= v18 + 0.2 -> drop it; between -> undecided.
  3. `blend_v20.csv` (optional probe, md5 073ca7d8...): v18 with the cell CNN at a fixed share 0.8; expected ~-0.01,
     public SE ~0.034; <= v18 - 0.07 -> the in-cell CNN premium is large, a dedicated cell CNN is worth GPU time.
  Slots 4-5: no candidate; do not fill them with near-duplicates. Files: `/mnt/project-files/work/hardness-cache/submissions/`
  (also attached in the thread). They are git-ignored; rebuild in the repo with the commands in the v18-v20 bullets below.
- **How to resume:** the user powers on the laptop, opens Claude Code in C:\Daker\microstructure-hardness-prediction,
  re-enables Remote Control, and writes "재개" in the main thread; the laptop runs `git pull` first (branch
  claude/lb-under-10-7080jr). The cloud thread then: restore `data/` from hardness-cache if the container is new
  (open.zip + hardness-cache/data/*.parquet, cnn_cache, experiments-test.tar), check the user's scores against the
  reading rules above, and continue the gap work (het in noisy images is the open problem; null so far: restored-image
  het, learned per-block estimator, grid4 CNN head, extra clean-formula terms).
- **Resumed 21:04 KST; noisy-image het, three more nulls (workflow, pre-registered, train-only choices; hardness-cache
  `scripts/gap2-1007/`):** on the 233 train images with raw ic_noise >= 9.5, on top of blend_v19 nested OOF:
  block ACF length L50 +0.153 [+0.03, +0.29] (perm p 0.98, 2/5 folds), block power-spectrum mean frequency +0.125
  [+0.07, +0.18] (perm p 0.90), smoothed region segmentation for coarse images stopped at fidelity (corr with clean het4
  -0.08 on degraded copies). Lesson: raw fidelity with het4 comes from grain size (het4 vs log N_eff -0.83); judge any
  future estimator by partial corr with het4 given log N (all three ~0), and block ACF length follows block dark-phase
  fraction (0.50) more than block grain size (0.21). Test-image features (acf, spec) are in that folder, extraction only.

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
