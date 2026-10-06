# cnn-trainer handoff (updated 2026-10-06 06:05 UTC; nothing of mine is running)

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
- Inputs/aug: `--input raw|nlm|raw+nlm`; `--bright 0.03 --contrast 0.1 --noise 0.03 --noise-p 0.5` defaults; `--norm global|image`.
- Degradation: `--deg-p P` replaces (or with `--cons`, pairs) a training image from the cleaner half of train
  (ic_ridge_snr > `--deg-snr-min` 0.5) with one of `--deg-k` 8 cached degradations made by the feature-engineer's
  calibration-v2 model `src.features._degrade_v2` (imported read-only). Train images only. Validation-fold copies are
  never used in that fold.
- `--cons L`: each sampled twin gets the label loss plus L x MSE to the stop-grad prediction of its original under
  identical crop/D4/jitter (`Aug.sample/apply`).
- Pooling: `--pool avg|gem|avgstd|avgstd2`.
- Runs: `--seeds N`, `--train-seeds`, `--folds`, `--no-save`, `--no-test`. Per fold/seed results are cached in
  `data/cnn_cache/<name>/fold{f}_seed{s}.npz` and runs resume. Assembly prints and appends the SNR-tercile RMSE to the notes.
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
