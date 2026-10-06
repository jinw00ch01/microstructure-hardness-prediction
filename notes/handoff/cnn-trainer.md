# cnn-trainer handoff (updated 2026-10-06 ~20:35 UTC; nothing of mine is running)

## Saved experiments (shared folds; experiments/<name>/ + LEADERBOARD line; fold RMSE order 0..4)
SNR terciles: OOF RMSE by train `ic_ridge_snr` tercile (cuts 0.283 / 0.774), noisy / mid / clean.
| name | setup (resnet18.a1_in1k, 30 ep, lr 1e-3, crop 224, D4 TTA unless noted) | CV | folds | SNR terciles |
|---|---|---|---|---|
| **cnn_r18_rawnlm_degcons_e30** | raw+nlm, degradation aug p0.5 + consistency 1.0, 2 seeds | **13.516** | 13.861 12.604 13.985 14.081 12.982 | 15.70 / 11.75 / 12.79 |
| cnn_r18_rawnlm_deg_e30 | raw+nlm, degradation aug p0.5, 2 seeds | 13.606 | 13.600 12.763 14.343 14.258 12.988 | 15.30 / 11.86 / 13.43 |
| cnn_r18_c224_e30 | raw input, 1 seed | 13.787 | 14.118 13.346 14.593 13.264 13.569 | 15.52 / 12.17 / 13.46 |
| cnn_r18_rawnlm_e30_s3 | raw+nlm, 3 seeds | 13.823 | 13.760 12.637 14.725 14.069 13.841 | 15.81 / 12.24 / 13.17 |
| cnn_r18_rawnlm_e30 | raw+nlm, 2 seeds | 13.837 | 13.880 12.634 14.795 14.150 13.633 | 15.73 / 12.35 / 13.20 |
| cnn_r34d_rawnlm_e30 | resnet34d.ra2_in1k, raw+nlm, 1 seed | 14.194 | 13.609 13.212 15.162 15.226 13.633 | 15.94 / 12.57 / 13.85 |
(feat_lgb for reference: 14.202, terciles 16.34 / 13.35 / 12.64.)

Fold-paired against raw+nlm with 2 seeds (same seeds; image-bootstrap 90% CI, which ignores seed noise):
- degcons: fold deltas -0.02 / -0.03 / -0.81 / -0.07 / -0.65 (5/5 better); pooled -0.32 (-0.68, +0.02).
  By tercile: noisy -0.04, mid -0.60, clean -0.41. Residual corr with the baseline 0.94.
- deg: fold deltas -0.28 / +0.13 / -0.45 / +0.11 / -0.65 (3/5 better); pooled -0.23 (-0.49, +0.04).
  By tercile: noisy -0.43 (-0.83, -0.03), mid -0.48, clean +0.23. Residual corr with the baseline 0.97.
- The average of the deg and degcons OOF (4 models) is 13.445. Per-seed 5-fold CVs of one config vary by about 0.7
  (raw+nlm seeds: 13.670 / 14.349 / 14.103), so only multi-seed comparisons are meaningful.

## Screens (`--no-save --no-test`, cache only; folds 0-1, seed 0, paired with the same-seed raw+nlm avg-pool run 14.004 / 12.664)
- `--pool avgstd` (concat spatial mean+std of the final map): 14.428 / 12.694, pooled +0.24 (90% CI -0.33, +0.79).
  View-to-view prediction sd rises (2.9-3.1 vs about 2.6). No gain.
- `--pool avgstd2` (plus the std of the stride-8 stage map, block-averaged to the final grid): 13.783 on fold 1 and
  14.723 on fold 0, pooled +0.91 (+0.34, +1.48). Clearly worse. Not recommended as a default.
  The baseline seeds 0/1/2 pooled over folds 0-1 are 13.35 / 13.58 / 13.34.
- Brightness/contrast jitter off (dropped by the orchestrator after one seed): fold 0 15.077 vs 14.004, fold 1 12.043 vs 12.664.
- Earlier (2026-10-05): lr 1e-3 beats 3e-4; crop 224 beats full-256 training; strong jitter (c0.2/b0.06) is worse than mild.

## Recommended config for the laptop GPU runs (gpu_turn.py wrapper)
`--input raw+nlm --deg-p 0.5 --cons 1.0 --pool avg --epochs 30 --lr 1e-3 --seeds 3` (+ a bigger backbone).
- Needs `data/cnn_cache/_pre/nlm.npz` and `data/cnn_cache/_pre/deg2_k8_snr0.5.npz`. Both are rebuilt automatically or with
  `python -m src.train_cnn --build-deg --deg-k 8 --deg-snr-min 0.5`, which takes about 4.5 min on 1 CPU core and needs
  data/features_v3.parquet and src.features._degrade_v2.
- Bigger backbones need a fresh lr check. resnet34d at lr 1e-3 was not better than resnet18 on CPU (1 seed).
- `--cons` costs about 1.3x per step (degraded twins are added to the batch).

## State of `src/train_cnn.py` (owned by cnn-trainer)
- Honest CV: fixed OneCycle schedule, final weights predict the val fold (no checkpoint selection). Val RMSE is logged per
  epoch for monitoring only. Random 224 crops for training; full 256 image + D4 TTA for inference. Target standardized
  per fold; input mean/std from the training fold only.
- CPU: bf16 autocast + channels_last (about 3x faster than fp32 on this AMX Xeon). `--threads` capped at 2 on CPU,
  no DataLoader workers. `--device auto|cpu|cuda` (orchestrator's addition).
- Inputs/aug: `--input raw|nlm|raw+nlm|raw+rest|raw+nlm+rest` (`--rest-dir`); `--bright 0.03 --contrast 0.1 --noise 0.03
  --noise-p 0.5` defaults; `--norm global|image`; `--scale S` (bilinear input upsampling, 1.0 = off).
- Degradation: `--deg-p P` replaces (or with `--cons`, pairs) a training image from the cleaner half of train
  (ic_ridge_snr > `--deg-snr-min` 0.5) with one of `--deg-k` 8 cached degradations made by the feature-engineer's
  calibration-v2 model `src.features._degrade_v2` (imported read-only). Train images only. Validation-fold copies are
  never used in that fold.
- `--cons L`: each sampled twin gets the label loss plus L x MSE to the stop-grad prediction of its original under
  identical crop/D4/jitter (`Aug.sample/apply`).
- Pooling: `--pool avg|gem|avgstd|avgstd2`.
- Runs: `--seeds N`, `--train-seeds`, `--folds`, `--no-save`, `--no-test`. Per fold/seed results are cached in
  `data/cnn_cache/<name>/fold{f}_seed{s}.npz` and runs resume. Assembly prints and appends the SNR-tercile RMSE to the notes.
  It also prints the OOF RMSE in the blend's fine_noisy cell (and adds it to the notes) when `src.blend_cells` and
  data/features_cal.parquet are available. `--full`: all-train models, test predictions only (section below).
  Training is deterministic for a given seed and thread count.

## Insights
- Seed variance dominates the CNN: about 0.7 CV between single seeds of one config, and per-view prediction sd about 2.6 HV.
  Always compare with 2+ seeds and fold-paired.
- Degradation augmentation helps the degraded images (noisy/mid terciles) as eda-analyst predicted. The consistency
  term adds gains on mid/clean images and produces a more diverse model (residual corr 0.94).
- CNNs beat the feature models on noisy/mid images but not on clean ones (raw+nlm clean 13.2 vs feat_lgb 12.6), so an
  SNR-aware blend could help.
- No location effect: dark fraction in centre/quadrant windows adds nothing beyond the global fraction (partial |r| <= 0.09).
- Mild brightness/contrast jitter is a strong regularizer (off overfits on fold 0); the second fold disagreed (single seed).

## data/cnn_cache (git-ignored; all regenerable)
- `_pre/nlm.npz` (98 MB, all 1500 IDs) and `_pre/deg2_k8_snr0.5.npz` (276 MB, 263 train sources x 8 x (raw, nlm)).
- One directory per experiment above with fold/seed npz files (val + test predictions) and `_locks/` from the worker
  queue (harmless).
- `s0*`/`s01*` directories are screens (val only). No model weights are saved.
- Queue scripts and logs are in logs/ (cnn_worker.sh, cnn_tasks.txt, cnn_screen.sh, cnnq_core*.sh).

## Next steps (priority order)
1. GPU: the recommended config above with 3 seeds and a bigger backbone (screen lr on 2 folds first).
2. CPU, if no GPU: a third seed of degcons (about 75 core-min):
   `OMP_NUM_THREADS=1 taskset -c 1 python -W ignore -m src.train_cnn --epochs 30 --lr 1e-3 --input raw+nlm --deg-p 0.5 --cons 1.0 --threads 1 --seeds 3 --train-seeds 2 --no-save --name cnn_r18_rawnlm_degcons_e30_s3`
   after `mkdir -p data/cnn_cache/cnn_r18_rawnlm_degcons_e30_s3 && cp data/cnn_cache/cnn_r18_rawnlm_degcons_e30/fold?_seed?.npz data/cnn_cache/cnn_r18_rawnlm_degcons_e30_s3/`,
   then assemble with the same args and `--seeds 3` (no `--train-seeds`/`--no-save`).
3. Tune the consistency setup with 2 seeds on 2 folds: `--cons 0.5` / `2.0`, `--deg-p 1.0` (more twins), deg sources snr > 0.75.
4. Ensembler: blend degcons + deg + raw + raw+nlm CNNs (residual corr 0.94-0.97) and try SNR-gated weights.

## Restoration network `src/restore.py` (2026-10-06, cnn-trainer)
- What it does: a compact U-Net (0.90M params, residual output, inputs = image + estimate_sigma map) is trained on
  `_degrade_v2(clean) -> clean` pairs. Sources are the 131 train images with ic_ridge_snr > 0.9: 120 train sources x 32
  degradations, plus 11 held-out sources x 6 degradations. It is then applied as a fixed per-image transform (D4 x8 TTA).
  No labels and no test images are used in training.
- Photometric target: the clean image in the degraded image's photometry. The deterministic part of `_degrade_v2` is
  reconstructed, then a cubic gain surface + offset is fitted to the degraded image (median fit rms 2.2 grey levels).
  The network denoises, deblurs and restores dark contrast, but does not re-normalise grey levels.
- Training: patch 96, bs16, 5000 steps, L1 + 0.5 gradient-L1, 15% identity pairs. About 25 min on 2 threads.
  Checkpoint every 500 steps to `data/restore_cache/restore_ckpt.pt`; a rerun of `train` resumes bit-exactly (tested).
  `apply` writes atomically and accepts `--resume`.
- Held-out synthetic validation, PSNR in dB, degraded -> restored (`experiments/restore/val_metrics.json`):
  | noise band | PSNR | boundary PSNR | ridge corr | dark contrast |
  |---|---|---|---|---|
  | low (<9) | 37.1 -> 37.3 | 36.7 -> 36.8 | 0.974 -> 0.976 | 0.98 -> 0.98 |
  | mid (9-15) | 26.6 -> 30.6 | 25.6 -> 28.5 | 0.738 -> 0.846 | 0.88 -> 0.94 |
  | high (>15) | 22.0 -> 28.1 | 20.8 -> 25.2 | 0.371 -> 0.621 | 0.77 -> 0.88 |
  | all | 28.6 -> 32.0 | | | |
  With noise >= 13 and blur, dark-phase zones are recovered but thin bright-bright boundaries mostly are not.
- Real images (`experiments/restore/apply_change.csv`):
  - Clean images (snr > 0.9) pass through: 67.5% are pixel-identical, p90 MAD 0.045, worst PSNR 33.9.
  - Median MAD by band: mid 5.2, noisy 11.6.
  - Montage of noisy images (snr < 0.3): grain networks are clearly visible after restoration from snr ~0.08 upward.
    At snr 0.04 the output is a faint cellular texture that may be partly hallucinated.
  - Method noise looks structureless.
- Outputs: `data_restored/{train,test}/*.png` plus copied CSVs (58 MB). Use with `DATA_DIR=data_restored`.
  `.gitignore` does not cover `data_restored/` yet; the orchestrator should add it.
- Rebuild: `bank --part 0/1 --nparts 2` (about 3.5 min each, 1 core), then `train --threads 2 --patch 96 --bs 16 --steps 5000`,
  then `apply --part 0/1 --nparts 2` (about 7 min each), then `check`.

## High-noise specialist restorer `--tag hn` (2026-10-06 15:09-16:15 UTC, cnn-trainer) -> `data_restored_hn/`
- Why: blend_v11 error tied to image noise (orchestrator: c = 9.0 in r^2 ~ a + b/N + c*noise); the base restorer only
  helped the features in the mid tercile.
- What: the same 0.90M U-Net fine-tuned from `data/restore_cache/restore_unet.pt` on high-noise pairs only:
  q >= 0.4 (noise_t >= 10.8). That is 2335 pairs of the original bank plus a new bank of 1920 pairs (120 train sources x 16),
  drawn by exact rejection sampling on `_degrade_v2`'s latent q, which is the first draw of `default_rng([9, src, k])`.
  The same 11 held-out sources get 24 fresh high-noise pairs each (prefix 10). Training: 3000 steps, patch 128, bs16,
  OneCycle lr 5e-4 (5% warm-up), L1 + 0.5 gradient-L1, no identity pairs, fp32.
- `src/restore.py` changes (untagged commands behave as before; checked that the original bank pairs reproduce
  bit-exactly; the default checkpoint cfg is unchanged):
  - `bank --bank-name --q-min --k-train --k-val --seed-base`
  - `train --tag --banks --train-q-min --init --width --fp32 --pct-start --mon-n --no-validate`. `--width` > 1 with
    `--init` is a zero-expanded init that keeps the function exactly (tested; not used in this run).
  - new `compare`: paired held-out comparison by noise band with source-bootstrap 90% intervals; `--features` adds v3
    feature fidelity.
  - `apply --apply-noise-min --fill-from` and `check --tag`.
  - The thread cap is now the CPU count (it was 2).
- This cloud host has no AMX (avx512f only): bf16 autocast is about 2.2x SLOWER than fp32 here (patch 128 bs16 step:
  1.87 s vs 0.81 s on 4 threads). So the specialist trains and infers in fp32 (`data/restore_cache/hn/meta.json`).
  The base model stays bf16; on this host it reproduces `data_restored/` 99.8% bit-exact (max diff 1 grey level).
- Held-out comparison (`experiments/restore/hn/compare_metrics.json`, `compare_pairs.csv`): 330 pairs of the 11
  held-out sources (66 original + 264 fresh), 8-view TTA, means for degraded / base / hn:
  | band (noise_t) | n | PSNR | boundary PSNR | ridge corr | dark contrast |
  |---|---|---|---|---|---|
  | low (<=9) | 20 | 37.13 / 37.31 / 34.41 | 36.69 / 36.84 / 33.58 | 0.974 / 0.976 / 0.972 | 0.983 / 0.984 / 1.006 |
  | mid (9-15] | 142 | 26.12 / 30.62 / 31.13 | 25.05 / 28.41 / 29.37 | 0.697 / 0.837 / 0.866 | 0.884 / 0.945 / 0.960 |
  | high (>15) | 168 | 22.27 / 28.25 / 28.65 | 20.87 / 25.07 / 25.59 | 0.347 / 0.603 / 0.641 | 0.772 / 0.876 / 0.901 |
  | rule: sigma_est >= 11.92 | 228 | 23.02 / 28.85 / 29.26 | 21.73 / 25.88 / 26.50 | 0.418 / 0.654 / 0.690 | 0.797 / 0.894 / 0.914 |
  Paired hn - base, with source-bootstrap 90% intervals:
  - high band: PSNR +0.39 (0.34, 0.44), boundary PSNR +0.53 (0.46, 0.59), ridge corr +0.037 (0.032, 0.042),
    |1 - dark contrast| -0.028 (-0.037, -0.018).
  - rule domain: +0.42 (0.37, 0.47), +0.62 (0.55, 0.69), +0.036, -0.023.
  - mid band: +0.50, +0.96, +0.029.
  - Near the cut (sigma_est 11.92-13): +0.51 dB, with 97% of pairs better. In the rule domain, 2 of 228 pairs are
    worse by more than 0.1 dB.
  - **Low band: -2.90 (-4.83, -1.24)**. The specialist never saw clean inputs and over-smooths them, so apply it
    above the cut only.
  - The base's high-band numbers differ from its own report (28.1 / 25.2 / 0.621) because that band now has 168
    pairs instead of 18.
- Feature fidelity: R^2 of the v3 measure on the restored copy against the clean source's value, over the 228
  rule-domain pairs (degraded / base / hn); in the high band (168 pairs) hn beats base on all 5.
  | feature | R^2 deg / base / hn | bias base / hn | hn - base 90% CI |
  |---|---|---|---|
  | ic_seg_fd93 | -0.30 / 0.65 / 0.82 | -0.030 / -0.014 | (0.09, 0.46) |
  | ic_fdo_93 | -1.04 / 0.70 / 0.75 | -0.018 / -0.019 | (0.00, 0.24) |
  | ic_gmm_w | -0.35 / 0.73 / 0.74 | -0.023 / -0.026 | (-0.18, 0.22) |
  | ic_acg_len50_perp | -3.04 / 0.41 / 0.52 | +0.83 / +0.62 | (0.05, 0.63) |
  | ic_seg_nfrac91 | -7.67 / -0.80 / -0.33 | +0.089 / +0.065 | (0.08, 1.51) |
  R^2 is relative to the spread of only 11 sources. ic_seg_nfrac91 stays below 0 (it is still worse than the mean of
  the sources).
- Visual check: `experiments/restore/hn/montage_compare_synthetic.png` (synthetic: degraded | base | hn | target) and
  `data/restore_cache/hn/montage_real_raw_base_spec.png` (real train images, git-ignored). Dark-phase regions are cleaner,
  but thin bright-bright boundaries are still mostly not recovered at noise_t >= 18 with blur. On real images it draws a
  few more boundaries in bright matrix areas than the base. The synthetic ridge-corr gain says these are mostly real,
  but this cannot be verified on real images.
- Applied (fixed rule, nothing fitted on test): raw `ic_noise` (data/features_v3.parquet) >= 11.92 gets the specialist
  with 8-view D4 TTA. That is **500 images: 167 train, 333 test**.
  - The other 1000 images are byte-identical copies of `data_restored/`, so their features in `data_restored/*.parquet`
    can be reused; only the 500 need recomputing.
  - Layout as `data_restored/`: train/, test/, train.csv, sample_submission.csv, folds.csv (58 MB). Use with
    `DATA_DIR=data_restored_hn`.
  - Per-image list: `data/restore_cache/hn/applied_part*.csv`.
  - Real-image change vs base (median MAD): 1.5 / 1.7 / 1.8 grey levels for ic_noise < 15 / 15-18 / > 18, no mean
    shift. Per-image table in `data/restore_cache/hn/check_specialist.csv` (git-ignored, has test images).
- Commands and timings (4 cores, about 65 min wall in total):
  ```
  for P in 0 1 2 3; do OMP_NUM_THREADS=1 python -W ignore -m src.restore bank --bank-name hn --q-min 0.4 --k-train 16 --k-val 24 --seed-base 9 --part $P --nparts 4 & done; wait   # 71 s
  OMP_NUM_THREADS=4 python -W ignore -m src.restore train --tag hn --banks main,hn --train-q-min 0.4 --init data/restore_cache/restore_unet.pt --fp32 --threads 4 --patch 128 --bs 16 --steps 3000 --lr 5e-4 --identity 0 --mon-n 48 --ckpt-every 250 --eval-every 500 --no-validate   # 2695 s
  OMP_NUM_THREADS=4 python -W ignore -m src.restore compare --tags base,hn --banks main,hn --threads 4 --jobs 4 --features --report-tag hn   # 12.6 min (base bf16 416 s, hn 169 s, features 121 s)
  for P in 0 1 2 3; do OMP_NUM_THREADS=1 python -W ignore -m src.restore apply --tag hn --apply-noise-min 11.92 --fill-from data_restored --part $P --nparts 4 --threads 1 & done; wait   # 220 s
  python -W ignore -m src.restore check --tag hn --fill-from data_restored   # 10 s
  ```
  - Training resumes from `data/restore_cache/hn/restore_ckpt.pt` (every 250 steps). `apply` takes `--resume`.
  - `compare` caches restorations and deg/base/clean features in `data/restore_cache/compare/` (166 MB, regenerable).
- Files (all git-ignored except `experiments/restore/hn/*` minus the montage):
  - `data/restore_cache/hn/{restore_unet.pt, meta.json, restore_ckpt.pt}`
  - `data/restore_cache/bank_hn/` (137 MB)
  - `experiments/restore/hn/{compare_metrics.json, compare_pairs.csv, compare_features.csv}` (synthetic pairs of
    train images only, no labels)
  - logs in `logs/restore_hn_*.log`
- Next: the feature-engineer's restored features + ridge screen on `data_restored_hn/`, especially the noisy tercile.
  If the restored features help, a wider model (`--width 1.5`, zero-expanded from hn) or more steps is the next lever.
  At fp32 on this host it costs about 1.6 s/step at patch 128 (2x).

## Restored-image channel, full-data mode, --scale (2026-10-07 KST, cnn-trainer; code and CPU smoke tests only)
- `src/train_cnn.py`:
  - `--input raw+rest | raw+nlm+rest`: the original restorer's output read from `--rest-dir` (default `data_restored/`).
    It is a fixed per-image transform of train and test, and a denoised channel like nlm (no blur/noise jitter).
    Normalisation stats come from the training fold only, as before.
  - Degraded bank copies get their own restored channel. The same restorer (`data/restore_cache/restore_unet.pt`) is
    applied to each degraded raw copy exactly as `src.restore apply` does (8-view TTA, uint8).
    Build: `--build-deg --input raw+nlm+rest --deg-k 8 --deg-snr-min 0.5 [--device cuda]` writes
    `data/cnn_cache/_pre/deg2_k8_snr0.5_rest.npz` (138 MB).
  - That file stores the restorer's output on 8 noisy train images. They must match `--rest-dir` (mean |diff| < 0.05,
    < 0.1% of pixels off by > 1), checked at build time and at every load. A bank and a rest-dir from different
    restorers or precisions are refused, e.g. data_restored_hn: mean |diff| 0.82.
  - `--full`: one model per seed on all 500 train images (same epochs, so 31 instead of 25 steps per epoch). No
    validation, nothing selected, test predictions only. Bank sources are all train images passing the SNR rule.
    - Output: `experiments/<name>/test.csv` + `full.json` + a LEADERBOARD line ("full-data, no CV"). There is no oof.csv
      and deliberately no score.json, because `src.ensemble` globs `*/score.json` and reads oof.csv.
    - Seeds are cached as `data/cnn_cache/<name>/full_seed{s}.npz` (seed = SEED + 1000 s + 5). Fold and full runs
      cannot share a name (guarded).
  - `--scale S`: bilinear upsampling (on the device) of each augmented crop and of every TTA view; `--crop` stays in
    native px. 1.0 is a no-op.
  - Fine_noisy cell RMSE at assembly (see the State section).
  - The new options appear in args.json and the notes only when used.
- `src/restore.py`:
  - `--device auto|cpu|cuda` for train/apply/validate/compare. CUDA runs in strict fp32 (no autocast, TF32 off); the
    CPU path is unchanged.
  - `--report-dir`.
  - Untagged non-default runs (GPU or --fp32) write `data/restore_cache/meta.json`, which `load_model` reads, so apply
    uses the training precision. The cloud has no such file, so the original restorer stays bf16.
- Smoke tests (cloud CPU, fp32, 2 threads, scratch copies of data/ and an isolated repo root; nothing under data/,
  data_restored/ or experiments/ was written):
  - Old (HEAD) vs new code with identical args are bit-identical: every saved array (val, val_other, 1000 test
    predictions), args.json and the logged epoch lines. Runs: raw; raw+nlm + deg + cons; raw+nlm + deg with test.
  - Restorer old vs new: identical weights, checkpoint cfg and meta.json (bf16 and --fp32 runs), identical held-out
    report and identical applied PNGs.
  - New paths:
    - rest bank build (39 sources x 2 copies, 184 s);
    - a unit test of the tensors that reach `train_one`: channel order raw/nlm/rest in both train and bank tensors,
      fold runs use only training-fold bank sources, rest-bank entries recompute exactly;
    - raw+rest and raw+nlm+rest fold runs, with and without deg/cons and with test;
    - a 5-fold run with assembly and save;
    - `--full` with the exact effnetv2-s recipe flags (2 seeds seed by seed, then assembly), with raw+nlm+rest, and
      with ConvNeXt-tiny (`--scratch`: its weights are not reachable from the cloud);
    - all guards (missing bank, mismatched rest-dir at load and at build time, which leaves the bank untouched, name
      clashes, `--full --no-test`).
- Reference (blend cell, from the saved OOF):
  | experiment | fine_noisy cell RMSE | cell by fold |
  |---|---|---|
  | ev2s_s3 | 11.258 | 10.50 / 10.16 / 11.46 / 10.63 / 13.16 |
  | ev2s_s6 | 11.261 | |
  | cnxt_s3 | 11.685 | |
  | r18 gpu | 12.778 | |
  132 cell images, 32 / 20 / 26 / 27 / 27 per fold.
- Single-seed fold scores of one recipe scatter with sd about 0.4 (cloud resnet18 caches), so a one-fold screen only
  catches gross failures.
- Laptop plan: `notes/handoff/laptop-gpu.md` section 8 (restorer rebuild on the GPU, rest bank, screen + 3 seeds,
  full-data effnetv2-s x6 and ConvNeXt x3, optional `--scale 2 --crop 112`).
