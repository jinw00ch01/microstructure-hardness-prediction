# Resume guide (paused 2026-10-05 ~12:45 UTC)

Goal: public LB RMSE <= 10.
First LB result (2026-10-06): `blend_v2` (nested CV 12.963) scored **public 12.3454**, so this public split runs about
0.6 below CV. LB #1 at that time was **9.2442**, which shows the morning "label-noise floor" verdict below was wrong.

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

## Next steps (priority order as of 2026-10-06 afternoon; exact commands in notes/handoff/*.md)
1. **feature-engineer**: save `feat3_v3cal_ridge_het_spat` and `feat3_v23cal_lgbs_het_spat` (members plus the
   eda_feats_* files), then a noise-robust local grain-size map per 64 px block (v4).
2. **embedding-modeler**: heads on global mean + cross-cell std/range of the cached g2a embeddings (stages 0-2);
   then a 4x4 grid extraction if it pays.
3. **cnn-trainer**: `--pool avgstd` (spatial mean + std) screen; degradation-bank augmentation results; then bigger
   backbones and 3 seeds on the laptop GPU through the wrapper.
4. Nested re-blend (`python -m src.ensemble --out blend_v4`), hand the CSV to the user to submit, record the public
   score in `experiments/LEADERBOARD.md`.

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
  robot project: wrap every GPU command as
  `python C:\Dacon\RobotWorldModel_ActionVideo\wm_ops\gpu_turn.py --who hardness -- <command>` (user's rule,
  2026-10-06), one CNN fold per wrapped command; on CUDA OOM lower `--bs`. Never touch the robot project otherwise.
- "Jev" (TypeSafe AI) is a remote, text-only API model and is banned by competition rule 2, so it is not used.
