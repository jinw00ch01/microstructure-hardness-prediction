# embedding-modeler handoff (updated 2026-10-06)

Owner files: `src/extract_embeddings.py`, `src/train_head.py`. No shared files edited. Nothing committed by me.
CPU: since 2026-10-06 every run uses 1 thread pinned to core 3: prefix with `taskset -c 3` and keep `--threads 1`.
Every backbone is timm weights from GitHub releases, licensed Apache-2.0.

## Best saved experiments (CV = shared 5 folds; all hyperparameters chosen by inner CV inside each outer fold)
| exp | CV RMSE | fold RMSE | what |
|---|---|---|---|
| emb_effv2s_256_gridge3_aug_featv3 | **13.254** | 13.924 13.023 13.226 13.388 12.677 | effv2s noise/blur-penalised + features_v3 side features |
| emb_effv2s_r18_256_gridge3_noise | **13.403** | 14.307 11.908 13.949 13.737 12.979 | effv2s + resnet18, 4 views, noise-invariance penalty (0.03 + 0.06) |
| emb_effv2s_r18_256_gridge3_aug | 13.414 | 14.435 11.834 13.710 14.045 12.886 | same, noise0.03 + blur1.0 rows |
| emb_effv2s_256_gridge3_noise | 13.442 | 14.553 11.915 13.820 13.767 13.007 | effv2s only, noise 0.03 + 0.06 rows |
| emb_effv2s_256_gridge3_aug | 13.514 | 14.898 11.960 13.748 13.845 12.938 | effv2s only, noise0.03 + blur1.0 rows |
| emb_effv2s_256_ridge_aug | 13.675 | 14.767 12.314 13.619 14.233 13.317 | (2026-10-05) ridge, views as rows |
| emb_r18_256_krr_feats | 13.902 | 15.004 12.775 14.212 14.006 13.412 | (2026-10-05) resnet18 + features.parquet, KRR |
| emb_r18_256_ridge_aug | 13.928 | 14.994 12.560 14.514 14.098 13.342 | (2026-10-05) resnet18 ridge, views as rows |

Exact commands to reproduce (each 15 s to 3 min at 1 thread; add `--name <exp>` to save, `--no-save` to only print):
```
H="taskset -c 3 python -m src.train_head --threads 1"
E=tf_efficientnetv2_s.in21k_ft_in1k_256_g2a; R=resnet18.a1_in1k_256_g2a
EX=tf_efficientnetv2_s.in21k_ft_in1k_256_augs2; RX=resnet18.a1_in1k_256_augs2
$H --emb $E,$R --extra-rows $EX,$RX --pools mean,std --stages 1,2,3 --head gridge3 --aug-filter noise --grid-json '{"lam_view": [0, 4], "lam_aug": [4, 16, 64]}' --name emb_effv2s_r18_256_gridge3_noise
$H --emb $E,$R --pools mean,std --stages 1,2,3 --head gridge3 --use-augs --grid-json '{"lam_view": [0, 4], "lam_aug": [4, 16, 64]}' --name emb_effv2s_r18_256_gridge3_aug
$H --emb $E --extra-rows $EX --pools mean,std --stages 1,2,3 --head gridge3 --aug-filter noise --name emb_effv2s_256_gridge3_noise
$H --emb $E --pools mean,std --stages 1,2,3 --head gridge3 --use-augs --name emb_effv2s_256_gridge3_aug
$H --emb $E --pools mean,std --stages 1,2,3 --head gridge3 --use-augs --with-feats --feats features_v3.parquet --name emb_effv2s_256_gridge3_aug_featv3
```

## Key finding: penalise the head's sensitivity to image noise
`--head gridge3` is a generalized ridge on the view-averaged embedding:
min |y - Zbar w|^2 + lam_view*mean|w.(Z_view - Zbar)|^2 + lam_cell*mean|w.(Z_cell - mean_cell)|^2 + lam_aug*mean|w.(Z_aug - Z_id)|^2 + alpha|w|^2.
Here `Z_aug` is the embedding of a deterministically noised (or blurred, shaded, ...) copy of the same image.
- Noise rows are what help. effv2s goes from 13.636 (4 views, no aug rows) to 13.535 (noise0.03) to 13.442 (noise0.03 + 0.06). resnet18 goes from 13.84 to 13.68. Inner CV picks lam_aug 4-16 in every fold.
- No help from the other nuisance rows: blur1.0 13.688, shade0.12 13.636, contrast0.75 and blur2.0 in combination with noise make it slightly worse. Grid-cell (spatial) rows: inner CV always picks lam_cell = 0.
- Orientation views matter: effv2s with 2 views + noise rows gives 13.693, vs 13.442 with 4 views. 8 views is being tested (see below).
- Concatenation: effv2s + resnet18 13.403. Adding convnext_nano does not help (13.490). convnext_nano alone: 13.80.
- Joint model with features_v3: 13.254, about the same as feat2_v3_ridge alone (13.219). Residual correlation of the embedding model with feat2_v3_ridge is 0.92 and with the CNN 0.90. In a nested-CV NNLS blend, feat2_v3_ridge + emb gives 13.18. feat2_v3_ridge + feat2_v23_lgb + emb gives 13.13, vs 13.07 for feat + feat + CNN. The embedding family adds little to the blend.

Earlier negatives (2026-10-05): NL-means input, quantile/max/GeM pooling, stage 0 and the last stage, view-std, LightGBM/SVR heads, features.parquet (v1) inside the embedding head.

## CLI
`python -m src.extract_embeddings` writes `data/emb/<tag>.npy` (N, rows, D) float32, plus `_ids.csv` and `_meta.json`. This is pure per-image inference: nothing is fitted.
- `--backbone` (common.create_timm, features_only), `--size 256`, `--views N` (first N of id, hflip, transpose, rot90, vflip, rot180, rot270, antitranspose) or `--view-names a,b`
- `--pools mean,std[,max,gem,qXX]`, `--grid G` (global + GxG cell pooling; row axis [view][cell])
- `--augs noise<s>,blur<s>,shade<s>,gamma<g>,contrast<c>`: deterministic per-image nuisance copies of the identity view, stored as `aug:*` rows
- `--prep raw|imgnorm|blur<s>|nlm|median<k>|ic` (numpy steps cached as `data/emb/_img_<step>_1500.npy`; `ic` = divide by the local 70th-pct matrix level)
- `--bs --threads --limit --tag`

`python -m src.train_head` runs outer 5-fold CV on `data/folds.csv`. Inside each outer training split, the transform, scaler, PCA and head hyperparameters (inner leave-one-fold-out CV) are fitted on training images only. Test embeddings are only transformed; the test prediction is the mean of the 5 fold models.
- `--emb t1,t2` (concatenated by ID), `--stages 1,2,3` or per tag `"1,2,3;1,2"`, `--pools`, `--views N`
- `--cells global|cells|all`, `--use-augs`, `--aug-filter noise[,blur,...]`, `--extra-rows x1,x2` (one per tag, join several with '+'; appends new orientation views or aug rows from extra extractions of the same backbone)
- `--view-mode mean|aug`, `--view-std`, `--transform none|sqrt|log`, `--block-norm`, `--pca K [--whiten]`
- `--with-feats --feats features_v3.parquet[,...] --feat-weight W`
- `--head ridge|gridge|gridge3|krr|svr|lgb`, `--grid-json '{...}'` (override the head grid), `--name`, `--no-save`

## Cached files in data/emb/ (command prefix: `taskset -c 3 python -m src.extract_embeddings --threads 1`)
| tag | args | seconds at 1 thread (shared core) |
|---|---|---|
| tf_efficientnetv2_s.in21k_ft_in1k_256_g2a | `--backbone tf_efficientnetv2_s.in21k_ft_in1k --views 4 --pools mean,std --grid 2 --augs noise0.03,blur1.0 --tag <tag>` | 1425 |
| tf_efficientnetv2_s.in21k_ft_in1k_256_augs2 | `--backbone tf_efficientnetv2_s.in21k_ft_in1k --views 1 --pools mean,std --augs shade0.12,noise0.06,blur2.0,contrast0.75 --tag <tag>` | 1478 |
| resnet18.a1_in1k_256_g2a | `--backbone resnet18.a1_in1k --views 4 --pools mean,std --grid 2 --augs noise0.03,blur1.0 --tag <tag>` | 1531 |
| resnet18.a1_in1k_256_augs2 | `--backbone resnet18.a1_in1k --views 1 --pools mean,std --augs noise0.06 --tag <tag>` | 191 |
| convnext_nano.d1h_in1k_256_g2a | `--backbone convnext_nano.d1h_in1k --views 4 --pools mean,std --grid 2 --augs noise0.03,blur1.0 --tag <tag>` | 1171 |
| tf_efficientnetv2_s.in21k_ft_in1k_256 | `--backbone tf_efficientnetv2_s.in21k_ft_in1k --views 4` (mean,std,max,gem) | 920 |
| resnet18.a1_in1k_256 | `--backbone resnet18.a1_in1k --views 4` | 467 |
| resnet18.a1_in1k_256_nlm, _q | `--prep nlm` / `--pools mean,std,q10,q50,q90 --tag resnet18.a1_in1k_256_q` | 486 / 1597 |
| _img_nlm_1500, _img_ic_1500 | image caches, built automatically by `--prep nlm` / `--prep ic` | ~100 / ~150 |
Unshared speed per image-view at 1 thread: resnet18 ~0.08 s, effv2s ~0.15-0.2 s, convnext_nano ~0.15 s.
