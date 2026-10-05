# embedding-modeler handoff (paused 2026-10-05 ~12:35 UTC)

Owner files: `src/extract_embeddings.py`, `src/train_head.py`. No shared files edited. Nothing committed.
All runs used 1 thread pinned to one core: prefix commands with `taskset -c 3` and keep `--threads 1`.
Every backbone used so far is timm weights from GitHub releases, licensed Apache-2.0.

## Results so far (CV = shared 5 folds, `data/folds.csv`)

Saved experiments (each has oof.csv, test.csv, score.json, and a LEADERBOARD line):

| exp | CV RMSE | fold RMSE | setup |
|---|---|---|---|
| emb_effv2s_256_ridge_aug | 13.675 | 14.767 12.314 13.619 14.233 13.317 | tf_efficientnetv2_s.in21k_ft_in1k, stages 2,3, mean+std, ridge, view-mode aug |
| emb_r18_256_krr_feats | 13.902 | 15.004 12.775 14.212 14.006 13.412 | resnet18.a1_in1k, stages 1,2,3, mean+std + features.parquet, RBF KRR (*) |
| emb_r18_256_ridge_aug | 13.928 | 14.994 12.560 14.514 14.098 13.342 | resnet18.a1_in1k, stages 1,2,3, mean+std, ridge, view-mode aug |

(*) This was saved with the older, narrower KRR gamma grid. The current code gives 13.925 for the same command.

Best unsaved results. Re-run with the "save" commands under Next steps:

| CV RMSE | command (add `--no-save` to only print) |
|---|---|
| 13.551 | `--emb tf_efficientnetv2_s.in21k_ft_in1k_256,resnet18.a1_in1k_256 --pools mean,std --stages 1,2,3 --head gridge --block-norm` (~12 min) |
| 13.578 | `--emb tf_efficientnetv2_s.in21k_ft_in1k_256 --pools mean,std --stages 1,2,3 --head gridge --transform sqrt` (~40 s) |
| 13.622 | `--emb tf_efficientnetv2_s.in21k_ft_in1k_256 --pools mean,std --stages 2,3 --head gridge` |
| 13.681 | effv2s+r18 stages 1,2,3 ridge --block-norm (mean views) |
| 13.840 | resnet18 stages 1,2,3 gridge |

Things that did not help, all measured with the honest inner-CV protocol:
- NL-means denoised input (`--prep nlm`): resnet18 got worse, 14.29-14.46 vs 13.99. Concatenating it with the raw embeddings also hurt (14.16-14.41).
- Quantile pooling (q10/q50/q90): 14.40-14.88 vs 13.99. ReLU maps are sparse, so the low quantiles degenerate.
- max pooling: 15.00 alone, 14.38 together with mean, std and gem. GeM: 14.16.
- Stage 0 and the last stage alone: about 15.0. Stages 2-3 carry the signal for every backbone.
- view-std features: 14.39. Block-norm alone is neutral.
- Adding features.parquet inside gridge for effv2s: 13.75-13.96 vs 13.62. Keep it as a separate model for the blend.
- LightGBM head on PCA-32 plus features: 14.12. SVR: 14.45.

## CLI

`python -m src.extract_embeddings` writes `data/emb/<tag>.npy` (N, R, D) float32, `<tag>_ids.csv` and `<tag>_meta.json`. This is pure inference: nothing is fitted, and test images are only embedded.
- `--backbone NAME` (via common.create_timm, features_only), `--size 256`, `--views 4` (id, hflip, transpose, rot90; up to 8 = D4)
- `--pools mean,std[,max,gem,q10,q50,q90]` per stage, `--prep raw|imgnorm|blur<s>|nlm|median<k>` (numpy steps are cached in `data/emb/_img_<step>_1500.npy`)
- `--grid G`: also pools each cell of a GxG grid. The row axis is [view][cell], with cell 0 = global.
- `--augs noise0.03,blur1.0[,gamma<g>,contrast<c>]`: deterministic per-image nuisance copies (identity orientation), stored as extra `aug:*` views.
- `--bs 16 --threads 1 --limit N --tag TAG`

`python -m src.train_head` runs outer 5-fold CV. Inside each outer training split, everything (transform, scaler, block weights, PCA, head hyperparameters via inner leave-one-fold-out CV on the other 4 folds) is fitted on training rows only. The test prediction is the mean of the 5 fold models. It prints CV, fold RMSEs, and the hyperparameter chosen per fold.
- `--emb tag1[,tag2]` (concatenated by ID), `--stages all|1,2,3`, `--pools mean,std`, `--views N`
- `--view-mode mean|aug`, `--cells global|cells|all`, `--use-augs`, `--view-std`
- `--transform none|sqrt|log`, `--block-norm`, `--with-feats --feat-weight W`, `--pca K [--whiten]`
- `--head ridge|gridge|krr|svr|lgb`, `--name EXP` (saves via common.save_experiment), `--no-save`, `--threads 1`
- `gridge` is a generalized ridge. It fits on the view-averaged embedding and adds lam * (penalty on the head's response to within-image deviations: other views, grid cells, aug copies). Both lam and alpha are chosen by inner CV. Inner CV picks lam 4-32, which beats plain view augmentation.

## Cached files in data/emb/ (regenerate with `taskset -c 3 python -m src.extract_embeddings ... --threads 1`)

| file | args | time at 1 thread |
|---|---|---|
| resnet18.a1_in1k_256 | `--backbone resnet18.a1_in1k --size 256 --views 4` (pools mean,std,max,gem) | 467 s |
| resnet18.a1_in1k_256_nlm | same + `--prep nlm` | 486 s, plus ~100 s to build `_img_nlm_1500.npy` |
| _img_nlm_1500.npy/.ids.csv | NL-means image cache, built automatically by any `--prep nlm` run | ~100 s |
| resnet18.a1_in1k_256_q | `--backbone resnet18.a1_in1k --size 256 --views 4 --pools mean,std,q10,q50,q90 --tag resnet18.a1_in1k_256_q` | 1597 s (sorting is slow) |
| tf_efficientnetv2_s.in21k_ft_in1k_256 | `--backbone tf_efficientnetv2_s.in21k_ft_in1k --size 256 --views 4` | 920 s |
| tf_efficientnetv2_s.in21k_ft_in1k_256_g2a | `--backbone tf_efficientnetv2_s.in21k_ft_in1k --size 256 --views 4 --pools mean,std --grid 2 --augs noise0.03,blur1.0 --tag tf_efficientnetv2_s.in21k_ft_in1k_256_g2a` | 1425 s |

The timings were measured while the core was partly shared with head runs. Unshared speeds at 1 thread per image-view: resnet18 ~80 ms, effv2s ~130-270 ms, convnext_nano ~105-210 ms, resnet50 ~155-310 ms. At 512px, multiply by about 4.

## What was running when paused

`logs/queue4.sh`:
1. effv2s g2a. Finished and cached, but not evaluated yet.
2. convnext_nano g2a. Killed about 30 s in; no files written.
3. resnet50 g2a. Never started.

To resume, re-run the two remaining extractions:
```
taskset -c 3 python -m src.extract_embeddings --backbone convnext_nano.d1h_in1k --size 256 --views 4 --pools mean,std --grid 2 --augs noise0.03,blur1.0 --tag convnext_nano.d1h_in1k_256_g2a --threads 1   # ~20-25 min
taskset -c 3 python -m src.extract_embeddings --backbone resnet50.a1_in1k --size 256 --views 4 --pools mean,std --grid 2 --augs noise0.03,blur1.0 --tag resnet50.a1_in1k_256_g2a --threads 1               # ~30-40 min
```

## Insights
- Frozen embeddings, handcrafted LGB and ridge all plateau near 14. Residuals of the resnet18 head and feat_lgb correlate 0.93, and a 50/50 blend only reaches 13.80. The models capture the same signal: dark-phase fraction (+), pores (-), anisotropy (+).
- Error is driven by image noise. OOF RMSE is ~12.5 on the cleanest half and ~15.2 on the noisiest half. Denoising the input removes useful texture, so the next lever is nuisance-invariance in the head (gridge with noise/blur aug rows), not preprocessing.
- Mid stages (stride 8-16) with mean+std pooling are best. in21k EfficientNetV2-S beats resnet18 by about 0.3. Penalizing within-image variation (gridge) gives another 0.05-0.15.
- With 500 samples, config differences under ~0.1 RMSE are noise. Fold RMSE ranges from 12.3 to 15.0, and the "best-fixed-hp" diagnostic is about 0.1 optimistic versus inner-CV selection.

## Next steps (prioritized; prefix commands with `taskset -c 3 python -m src.train_head`)
1. Evaluate the cached effv2s g2a, which tests nuisance augmentations plus grid cells:
   `--emb tf_efficientnetv2_s.in21k_ft_in1k_256_g2a --pools mean,std --stages 1,2,3 --head gridge --cells all --use-augs --no-save`
   then ablate `--cells global` with and without `--use-augs`, and add `--transform sqrt`.
2. Save the best current combos:
   `--emb tf_efficientnetv2_s.in21k_ft_in1k_256,resnet18.a1_in1k_256 --pools mean,std --stages 1,2,3 --head gridge --block-norm --name emb_effv2s_r18_256_gridge`
   `--emb tf_efficientnetv2_s.in21k_ft_in1k_256 --pools mean,std --stages 1,2,3 --head gridge --transform sqrt --name emb_effv2s_256_gridge_sqrt`
3. Extract the convnext_nano and resnet50 g2a embeddings (commands above), and resnet18 g2a (`--backbone resnet18.a1_in1k ... --tag resnet18.a1_in1k_256_g2a`, ~12 min). Evaluate each with gridge, then concatenate the best 2-3 with `--block-norm`.
4. If gridge with `--use-augs` helps, try stronger or more nuisance augmentations (e.g. `noise0.05,blur1.5`), and effv2s at `--size 384`.
5. Hand the saved OOFs to the ensembler. They are complementary to the feat_* models in a blend.
