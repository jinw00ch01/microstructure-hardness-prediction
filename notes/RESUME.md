# Resume guide (paused 2026-10-05 ~12:45 UTC)

Goal: public LB RMSE <= 10. Nothing has been submitted to the leaderboard yet.

## Where we are
| model family | best saved experiment | CV RMSE |
|---|---|---|
| OOF blend (NNLS, nested CV) | `submissions/blend_v2.json` (2026-10-06) | **12.96** |
| handcrafted features | `feat2_v3_ridge` | 13.22 |
| frozen embeddings | `emb_effv2s_256_ridge_aug` (unsaved gridge variant: 13.55) | 13.68 |
| CNN fine-tune | `cnn_r18_c224_e30` (raw+nlm run 3/10 done, folds 0-1 pooled 13.35) | 13.79 |
| baseline | `feat_lgb` | 14.20 |

Target std is 17.7 (mean predictor). `submissions/blend_v1.csv` (CV 13.15) is a first LB-calibration
candidate; it is git-ignored and backed up in the project folder (below).

## Restore a fresh cloud session (about 5 min)
```bash
cd /home/claude/microstructure-hardness-prediction        # repo, branch claude/lb-under-10-7080jr
git fetch origin claude/lb-under-10-7080jr && git checkout claude/lb-under-10-7080jr && git pull
python3 -m pip install -q -r requirements.txt             # torch comes from PyPI; PyWavelets needed by skimage
mkdir -p data && (cd data && unzip -qo /mnt/project-files/open.zip)
cp -r /mnt/project-files/work/hardness-cache/data/. data/  # folds, features v1-v3, embeddings, CNN caches
mkdir -p submissions && cp /mnt/project-files/work/hardness-cache/submissions/* submissions/
```
Pretrained weights download from GitHub releases on first use (`src.common.create_timm`).
huggingface.co is blocked in the cloud sandbox.

## Next steps (priority order; exact commands in notes/handoff/*.md)
1. **Finish the CNN raw+nlm run** (about 50 min, two 1-thread processes, resumes from cache):
   see "What was running when paused" in `notes/handoff/cnn-trainer.md`.
2. **Embedding heads**: evaluate the cached effv2s `_g2a` embeddings with `gridge --cells all --use-augs`,
   then save the 13.55 / 13.58 gridge models (`notes/handoff/embedding-modeler.md` steps 1-2).
3. **Feature calibration** (label-free degradation to clean-measurement mapping) and the regularised GBM
   (`notes/handoff/feature-engineer.md` steps 1-2).
4. Re-blend: `python -m src.ensemble --out blend_v2`, then hand the best CSV to the user to submit
   (the user submits; 5/day limit) and record the public score in `experiments/LEADERBOARD.md`.
5. Second CNN architecture (resnet34d raw+nlm) and a third seed for variance reduction.

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

## Noise-floor finding (2026-10-06, eda-analyst; details experiments/eda/NOTES.md)
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
- The user's PC has a GPU but it was busy with another project on 2026-10-05; ask before using it.
- "Jev" (TypeSafe AI) is a remote, text-only API model and is banned by competition rule 2, so it is not used.
