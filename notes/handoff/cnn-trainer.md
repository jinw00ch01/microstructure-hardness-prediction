# cnn-trainer handoff (paused 2026-10-05 ~12:40 UTC)

## State of `src/train_cnn.py` (rewritten; owned by cnn-trainer)
- Honest CV: fixed OneCycle schedule, **final weights** predict the val fold (no per-fold checkpoint
  selection). Val RMSE is logged per epoch for monitoring only. EMA option exists (`--ema 0.99 --use ema`)
  but final == EMA at the end of cosine annealing (14.115 vs 14.111 on fold 0), so it is off by default.
- Train on random crops (`--crop 224`, no resizing), infer on the full 256 image with D4 TTA (`--tta 8`).
  Target standardized per fold; input mean/std computed from the training fold only.
- CPU speed: bf16 autocast + channels_last (AMX on this Xeon) is about 3x faster than fp32
  (resnet18, 2 threads: 24 ms/img train, 6.6 ms/img inference). `--threads` is capped at 2, no DataLoader workers.
  Two 1-thread processes give about 20% more total throughput than one 2-thread process.
- Flags that matter:
  - `--input raw|nlm|raw+nlm`: `raw+nlm` adds a non-local-means denoised channel (a fixed per-image
    transform, same as src/features.py: h = 1.2 x estimate_sigma). It is cached in `data/cnn_cache/_pre/nlm.npz`
    and computed on demand if missing (about 1.5 min for all 1500 images, 1 thread).
  - Augmentation: `--bright 0.03 --contrast 0.1` (defaults), `--noise 0.03 --noise-p 0.5`, `--blur-p 0` (off).
    D4 is always on.
  - `--norm global|image` (image = per-image standardization; implemented, never tested).
  - `--seeds N` = models per fold, averaged. `--train-seeds i ...` trains only those seed indices, so seeds can
    run in parallel processes. `--no-save` skips writing experiments/. `--folds ...` trains a subset of folds.
    `--no-test` is for sanity screens.
  - Per fold/seed results are cached in `data/cnn_cache/<name>/fold{f}_seed{s}.npz` (val idx, val pred,
    test pred). Re-running the same command resumes and skips cached fold/seeds. experiments/<name>/ is
    written only when all 5 folds x all seeds have test predictions.
  - Training is deterministic for a given seed and thread count (fold 0 seed 0 reproduced exactly).
- Also available: `--loss mse|huber`, `--pool avg|gem`, `--drop`, `--drop-path`, `--head-lr-mult`, `--wd`, `--epochs`, `--lr`.

## Experiments (shared folds; fold RMSEs in order 0..4)
| name | input / aug | CV RMSE | folds | status |
|---|---|---|---|---|
| **cnn_r18_c224_e30** | raw, crop224, lr1e-3, 30 ep, mild b/c jitter | **13.787** | 14.118, 13.346, 14.593, 13.264, 13.569 | complete in experiments/ (62 min, 1 thread) |
| cnn_r18_rawnlm_e30 (seeds 2) | raw+nlm, otherwise same | partial | f0: s0 14.004, s1 14.162, 2-seed mean **13.880**; f1: s0 12.664 | **paused**, 3 of 10 fold-seeds cached |

Fold screens (`--no-test`, cache only, not in experiments/; resnet18, lr1e-3, crop224 unless noted):
- fold 0, 20 ep: lr1e-3 14.115 vs lr3e-4 14.617
- fold 0, 30 ep: crop256 (full image) 14.416 vs crop224 14.118. Crop 256 fits training data much better
  (loss 0.22 vs 0.37) but generalizes worse.
- folds 0+1, raw+nlm: 14.004 / 12.664 (pooled 13.35) vs raw 14.118 / 13.346 (pooled 13.74). **NLM channel helps.**
- fold 0, raw+nlm, no brightness/contrast jitter: 15.077 (clear overfitting: train loss 0.28 vs 0.36)
- folds 0+1, raw+nlm, strong jitter (c0.2, b0.06): 14.508 / 13.612. Mild jitter (c0.1, b0.03) is best.

## What was running when paused (exact resume commands; cached fold/seeds are skipped)
```
cd /home/claude/microstructure-hardness-prediction
OMP_NUM_THREADS=1 nohup python -W ignore -m src.train_cnn --epochs 30 --lr 1e-3 --input raw+nlm --threads 1 --seeds 2 --train-seeds 0 --no-save --name cnn_r18_rawnlm_e30 > logs/cnn_r18_rawnlm_e30_s0.log 2>&1 &
OMP_NUM_THREADS=1 nohup python -W ignore -m src.train_cnn --epochs 30 --lr 1e-3 --input raw+nlm --threads 1 --seeds 2 --train-seeds 1 --no-save --name cnn_r18_rawnlm_e30 > logs/cnn_r18_rawnlm_e30_s1.log 2>&1 &
# after both finish (about 12.5 min per fold-seed at 1 thread; 7 fold-seeds remain, so about 50 min), assemble + write experiments/:
python -W ignore -m src.train_cnn --epochs 30 --lr 1e-3 --input raw+nlm --threads 2 --seeds 2 --name cnn_r18_rawnlm_e30
```
(The `>` redirects overwrite the paused runs' logs. Use `>>` to keep the epoch history of folds already done.)

## data/cnn_cache contents (git-ignored; everything can be regenerated)
- `_pre/nlm.npz`: NLM-denoised uint8 images for all 1500 IDs (98 MB). Regenerated automatically on demand.
- `cnn_r18_c224_e30/`: 5 fold npz files (the experiment is already written to experiments/).
- `cnn_r18_rawnlm_e30/`: fold0_seed0, fold0_seed1, fold1_seed0 (needed to resume; regenerable by retraining
  deterministically).
- `s0_*`, `s01_*`: screen outputs (val predictions only, no test). Safe to delete.
- No model weights are saved.

## Insights
- The CNN beats the feature models: 13.79 vs best feature model about 14.17. Residual correlation with LGB OOF is 0.87.
  A 50/50 blend of cnn_r18_c224_e30 and feat_lgb gives OOF 13.53, so it is worth stacking.
- Both CNN and LGB regress to the mean: RMSE about 18 in the lowest and highest target quintiles vs about 9 in
  the middle. Error rises with image noise (CNN by noise quartile: 13.5 / 12.4 / 13.9 / 15.1).
- noise_sigma has about zero correlation with hardness, so noise augmentation is safe. Global intensity
  statistics, however, are easy to memorize: removing brightness/contrast jitter costs about 1 RMSE (overfitting),
  but strong jitter also hurts. Keep it mild.
- The model is far from D4-invariant: per-view predictions differ by about 2.7 HV (sd) and single-view RMSEs
  spread about ±0.6. So D4 TTA (about -0.3) and seed averaging (fold 0: 14.00/14.16 to 13.88) are reliable gains.
- Single-fold, single-seed screens have about ±0.4 noise. Use 2+ folds or 5-fold runs to compare configs.

## Next steps (priority order)
1. Finish cnn_r18_rawnlm_e30 (2 seeds), using the resume commands above. Expected CV about 13.3-13.5.
2. Third seed for variance reduction. Copy the cache, train seed 2, then assemble:
   `cp -r data/cnn_cache/cnn_r18_rawnlm_e30 data/cnn_cache/cnn_r18_rawnlm_e30_s3 && OMP_NUM_THREADS=1 python -W ignore -m src.train_cnn --epochs 30 --lr 1e-3 --input raw+nlm --threads 1 --seeds 3 --train-seeds 2 --no-save --name cnn_r18_rawnlm_e30_s3`
   then run `python -W ignore -m src.train_cnn --epochs 30 --lr 1e-3 --input raw+nlm --threads 2 --seeds 3 --name cnn_r18_rawnlm_e30_s3`.
3. Second architecture for ensemble diversity: resnet34d.ra2_in1k (about 2x resnet18 cost; weights cached).
   Split folds across two 1-thread processes, then assemble:
   `OMP_NUM_THREADS=1 python -W ignore -m src.train_cnn --backbone resnet34d.ra2_in1k --input raw+nlm --epochs 30 --lr 1e-3 --threads 1 --folds 0 1 2 --no-save --name cnn_r34d_rawnlm_e30 &`
   `OMP_NUM_THREADS=1 python -W ignore -m src.train_cnn --backbone resnet34d.ra2_in1k --input raw+nlm --epochs 30 --lr 1e-3 --threads 1 --folds 3 4 --no-save --name cnn_r34d_rawnlm_e30`
   then run `python -W ignore -m src.train_cnn --backbone resnet34d.ra2_in1k --input raw+nlm --epochs 30 --lr 1e-3 --name cnn_r34d_rawnlm_e30`.
4. Untested ideas, screen on folds 0+1 with `--no-test`: `--drop-path 0.1` / `--drop 0.2` (overfitting is
   the main issue); `--crop 192` (faster, more regularization); `--norm image`; stronger noise aug for noisy images.
