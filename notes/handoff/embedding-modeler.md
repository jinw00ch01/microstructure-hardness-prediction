# embedding-modeler handoff (updated 2026-10-06, night: own-slope screen)

Owner files: `src/extract_embeddings.py`, `src/train_head.py`. No shared files edited. Nothing committed by me.
CPU: every run uses 1 thread. Latest assignment (night screen) was at most 2 cores; I used cores 2 and 3 with `taskset -c 2` / `taskset -c 3` and `--threads 1`, while the feature-engineer ran on core 0.
Every backbone is timm weights from GitHub releases, licensed Apache-2.0.

## Best saved experiments (CV = shared 5 folds; all hyperparameters chosen by inner CV inside each outer fold)
| exp | CV RMSE | fold RMSE | what |
|---|---|---|---|
| emb_effv2s_256_gridge3_noise_csbag_rawrest | **12.842** | 13.986 11.419 13.110 13.356 12.181 | fixed 0.5/0.5 average of csbag (raw) and csbag_rest (restored); not fitted |
| emb_effv2s_256_gridge3_noise_csbag_rest | 13.038 | 14.566 11.617 13.288 13.339 12.180 | same 7-spec csbag on c248n_rest (cnn-trainer UNet-restored images, `DATA_DIR=data_restored`) |
| emb_effv2s_256_gridge3_noise_csbag | **12.909** | 13.763 11.544 13.123 13.590 12.397 | equal-weight bag of 7 full CV runs, one per cellstat grid spec (2;4;8;2,4;2,8;4,8;2,4,8) on c248n |
| emb_effv2s_256_gridge3_noise_cs24 | 13.020 | 13.860 11.924 13.010 13.776 12.422 | effv2s noise model + cross-cell std on grids 2 and 4 (c24n, `--cs-rows orig`) |
| emb_effv2s_256_gridge3_noise_cs24_hetnz | 12.979 | 13.830 11.878 12.982 13.652 12.448 | cs24 + in-fold inverse-variance weights on log raw ic_noise (night screen below) |
| emb_effv2s_256_gridge3_noise_cs24_n3own_hpavg5 | 12.976 | 14.277 11.444 13.324 13.464 12.178 | cs24 + own columns for the high-noise third + hp-avg 5; like-for-like vs cs24 hp-avg 5 a tie (below) |
| emb_effv2s_256_gridge3_noise_cs2 | 13.134 | 14.172 11.780 13.631 13.503 12.442 | effv2s noise model + cross-cell std on grid 2 (g2a + augs2 noise rows) |
| emb_effv2s_256_gridge3_aug_featv3 | 13.254 | 13.924 13.023 13.226 13.388 12.677 | effv2s noise/blur-penalised + features_v3 side features |
| emb_effv2s_r18_256_gridge3_noise | 13.403 | 14.307 11.908 13.949 13.737 12.979 | effv2s + resnet18, 4 views, noise-invariance penalty (0.03 + 0.06) |
| emb_effv2s_r18_256_gridge3_aug | 13.414 | 14.435 11.834 13.710 14.045 12.886 | same, noise0.03 + blur1.0 rows |
| emb_effv2s_256_gridge3_noise | 13.442 | 14.553 11.915 13.820 13.767 13.007 | effv2s only, noise 0.03 + 0.06 rows |
| emb_effv2s_256_gridge3_aug | 13.514 | 14.898 11.960 13.748 13.845 12.938 | effv2s only, noise0.03 + blur1.0 rows |
| emb_effv2s_256_ridge_aug | 13.675 | 14.767 12.314 13.619 14.233 13.317 | (2026-10-05) ridge, views as rows |
| emb_r18_256_krr_feats | 13.902 | 15.004 12.775 14.212 14.006 13.412 | (2026-10-05) resnet18 + features.parquet, KRR |
| emb_r18_256_ridge_aug | 13.928 | 14.994 12.560 14.514 14.098 13.342 | (2026-10-05) resnet18 ridge, views as rows |

Paired tests of the bag (2000 bootstrap resamples, ΔRMSE = bag minus other model):
- vs cs2: ΔRMSE -0.228 (95% CI [-0.451, -0.013]); bag better on 4/5 folds.
- vs cs24: ΔRMSE -0.108 (CI [-0.285, +0.059]); bag better on 4/5 folds.
- vs effv2s_r18_noise, the best model before cellstats: ΔRMSE -0.496 (CI [-0.789, -0.206]); bag better on 5/5 folds.
- Residual correlation of the bag with cs24 is 0.988 and with cs2 0.982. The bag replaces them rather than adding a new signal.

Exact commands to reproduce. Single runs take 15 s to 3 min at 1 thread; the 7-member bag takes about 8.5 min (500 s). Add `--name <exp>` to save or `--no-save` to only print.
```
H="taskset -c 3 python -m src.train_head --threads 1"
E=tf_efficientnetv2_s.in21k_ft_in1k_256_g2a; R=resnet18.a1_in1k_256_g2a
EX=tf_efficientnetv2_s.in21k_ft_in1k_256_augs2; RX=resnet18.a1_in1k_256_augs2
C8=tf_efficientnetv2_s.in21k_ft_in1k_256_c248n; C4=tf_efficientnetv2_s.in21k_ft_in1k_256_c24n
CS="--pools mean,std --stages 1,2,3 --head gridge3 --use-augs --cellstats std --cs-stages 0,1,2 --cs-rows orig"
$H --emb $C8 $CS --cs-grid-bag "2;4;8;2,4;2,8;4,8;2,4,8" --name emb_effv2s_256_gridge3_noise_csbag
$H --emb $C4 $CS --cs-grid 2,4 --name emb_effv2s_256_gridge3_noise_cs24     # same result from $C8
$H --emb $E --extra-rows $EX --pools mean,std --stages 1,2,3 --head gridge3 --aug-filter noise --cellstats std --cs-stages 0,1,2 --name emb_effv2s_256_gridge3_noise_cs2
$H --emb $E,$R --extra-rows $EX,$RX --pools mean,std --stages 1,2,3 --head gridge3 --aug-filter noise --grid-json '{"lam_view": [0, 4], "lam_aug": [4, 16, 64]}' --name emb_effv2s_r18_256_gridge3_noise
$H --emb $E,$R --pools mean,std --stages 1,2,3 --head gridge3 --use-augs --grid-json '{"lam_view": [0, 4], "lam_aug": [4, 16, 64]}' --name emb_effv2s_r18_256_gridge3_aug
$H --emb $E --extra-rows $EX --pools mean,std --stages 1,2,3 --head gridge3 --aug-filter noise --name emb_effv2s_256_gridge3_noise
$H --emb $E --pools mean,std --stages 1,2,3 --head gridge3 --use-augs --name emb_effv2s_256_gridge3_aug
$H --emb $E --pools mean,std --stages 1,2,3 --head gridge3 --use-augs --with-feats --feats features_v3.parquet --name emb_effv2s_256_gridge3_aug_featv3
```

## Own slopes for the high-noise third (2026-10-06 night): no gain for the embedding head
Assignment: the feature-engineer's own-column trick (`src.features.n3own`; feature ridge 12.53 -> 12.26) on cs24.
`--own-col ic_noise --own-q 2/3` duplicates the whole head input (global + cellstats, every view and noise row) for the images with raw ic_noise > 11.9159. That is the train-only top-tercile cut; 167 train / 333 test images. Elsewhere the copy is NaN. Prep imputes the in-fold group mean, so after scaling the copy is exactly 0 for other images, in every row.
`--own-q 1/3,2/3` gives the mid tercile (7.4912 < ic_noise <= 11.9159) its own copy as well, i.e. 3 slope sets. The gridge3 grid key `own_w` is the own blocks' column weight: their own ridge group, with penalty alpha/own_w^2.
cs24 was rebuilt first: CV 13.0204, OOF and test identical to the saved files (max diff 1.7e-12 / 5.4e-13).

Fold-paired vs cs24 (13.020; folds 13.860 11.924 13.010 13.776 12.422):
- N1/N2/N3 = RMSE by raw ic_noise tercile (train cuts 7.49 / 11.92).
- boot95 = paired bootstrap of the RMSE difference (2000 resamples).
- blend = cross-validated 2-member non-negative blend with `feat7_rawn3own_ridge_all` (diagnostic only).
- Times are on 1 core.

| variant | CV | dCV | better | fold deltas | N1 | N2 | N3 | boot95 | blend | s |
|---|---|---|---|---|---|---|---|---|---|---|
| cs24 (parent) | 13.020 | | | | 10.878 | 12.386 | 15.387 | | 12.171 | 113 |
| own top third, own_w=1 | 13.150 | +0.130 | 3/5 | +.515 -.512 +.631 -.018 -.080 | 11.071 | 12.127 | 15.780 | [-.199, +.465] | 12.188 | 635 |
| own top third, own_w {0.5,1,2} (0.5 in every fold) | 13.033 | +0.013 | 3/5 | +.326 -.299 +.400 -.208 -.215 | 10.916 | 12.181 | 15.555 | [-.230, +.252] | 12.157 | 1847 |
| own mid + top (3 slope sets), own_w=1 | 13.050 | +0.029 | 3/5 | +.460 -.535 +.391 -.017 -.266 | 10.832 | 12.273 | 15.582 | [-.347, +.398] | 12.156 | 2160 |
| cs24 + hp-avg 5 (control) | 12.988 | -0.032 | 3/5 | +.036 -.259 +.189 -.128 -.022 | 10.745 | 12.277 | 15.485 | [-.143, +.079] | 12.176 | 97 |
| own top third, own_w grid + hp-avg 5: **saved `_n3own_hpavg5`** | 12.976 | -0.044 | 3/5 | +.416 -.481 +.314 -.312 -.244 | 10.957 | 12.042 | 15.489 | [-.289, +.200] | 12.174 | 1815 |
| hetero log ic_noise: **saved `_hetnz`** | 12.979 | -0.042 | 4/5 | -.030 -.046 -.028 -.124 +.026 | 10.740 | 12.394 | 15.372 | [-.145, +.059] | 12.138 | 169 |
| hetero ic_noise + ic_acg_len50_gm | 13.032 | +0.012 | 3/5 | -.001 -.268 +.410 -.191 +.091 | 10.662 | 12.551 | 15.435 | [-.163, +.189] | 12.161 | 168 |
| hetero ic_acg_len50_gm | 13.038 | +0.018 | 2/5 | +.025 -.249 +.399 -.199 +.098 | 10.674 | 12.565 | 15.431 | [-.154, +.191] | 12.159 | 165 |

- With the same protocol, own slopes do not beat cs24. In every own variant the high-noise tercile gets worse (15.39 -> 15.56-15.78) and only the mid tercile improves.
  - The 1088-column own block is fitted on about 133 high-noise images per outer fold and overfits.
  - Inner CV also switches to lam_aug=4 in folds 0 and 2, where cs24 always picks 0. Those folds lose 0.3-0.6.
  - Best-fixed-hp diagnostics are 12.82-12.90 vs cs24's 12.92, so the ceiling is at most about 0.1.
- `_n3own_hpavg5` passes the save rule (CV and 3/5 folds) only through hp averaging. Against cs24 + hp-avg 5 it is -0.012 (3/5 folds), a tie, with N3 unchanged and the blend diagnostic unchanged.
- The usable part of the split is down-weighting noisy images. `--hetero ic_noise` does this without extra capacity: log-variance slope 0.3-0.6, weights 0.6-1.6. It gives 12.979, 4/5 folds better, with the gain in the clean tercile. It is the only variant that moves the blend diagnostic (12.171 -> 12.138); the feature-engineer saw a tie when adding noise to their already-own-slope ridge.
- A random-group control (`--own-random 1`) was started but dropped once the noise split itself showed no gain.

Commands (cores 2-3 at 1 thread each). `_n3own_hpavg5` was saved from its screen outputs with the scratchpad `save_scratch.py`, which writes the same notes format via `common.save_experiment`; the command below with `--name` reproduces it.
```
H="taskset -c 2 python -m src.train_head --threads 1"; C4=tf_efficientnetv2_s.in21k_ft_in1k_256_c24n
CS="--pools mean,std --stages 1,2,3 --head gridge3 --use-augs --cellstats std --cs-stages 0,1,2 --cs-rows orig --cs-grid 2,4"
$H --emb $C4 $CS --no-save --oof-out <p> --test-out <p>                                          # cs24 rebuild
$H --emb $C4 $CS --own-col ic_noise --own-q 2/3 --no-save                                         # own_w=1
$H --emb $C4 $CS --own-col ic_noise --own-q 2/3 --grid-json '{"own_w": [0.5, 1, 2]}' --no-save    # own_w grid
$H --emb $C4 $CS --own-col ic_noise --own-q 1/3,2/3 --no-save                                     # mid + top
$H --emb $C4 $CS --hp-avg 5 --no-save
$H --emb $C4 $CS --own-col ic_noise --own-q 2/3 --grid-json '{"own_w": [0.5, 1, 2]}' --hp-avg 5 --name emb_effv2s_256_gridge3_noise_cs24_n3own_hpavg5
$H --emb $C4 $CS --hetero ic_noise --name emb_effv2s_256_gridge3_noise_cs24_hetnz
$H --emb $C4 $CS --hetero ic_noise,ic_acg_len50_gm --no-save; $H --emb $C4 $CS --hetero ic_acg_len50_gm --no-save
```

## Restored images (2026-10-06 evening)
`data_restored/` holds all 1500 images passed through cnn-trainer's restoration UNet. That UNet was trained on degraded copies of clean train images only. The c248n cache was rebuilt from these images with the same flags. Run the extractor and the head with `DATA_DIR=/home/claude/microstructure-hardness-prediction/data_restored`; the cache goes to `data_restored/emb/`. Extraction took 2 shards x ~640 s on cores 0-1. The full queue is `logs/queue_rest.sh`.

The rawrest average is saved by a small script, `save_avg.py` in this agent's scratchpad: it loads both experiments' oof/test CSVs, averages them, and calls `common.save_experiment`.

Comparisons:
- **rest vs raw:** 13.038 vs 12.909. ΔRMSE +0.129 (CI [-0.210, +0.481]); rest is better on only 2/5 folds. Residual correlation is 0.960, which is more diverse than any two raw variants (0.98-0.99).
- **rawrest vs the singles:** 12.842 beats both.
  - vs raw: ΔRMSE -0.067 (CI [-0.238, +0.111]); better on 4/5 folds.
  - vs rest: ΔRMSE -0.196 (CI [-0.370, -0.024]).
- **RMSE by SNR tercile** (`ic_ridge_snr` from `data/features_v3.parquet`, which is computed on raw images; noisy / mid / clean):

  | model | noisy | mid | clean |
  |---|---|---|---|
  | raw | 15.092 | 12.028 | 11.286 |
  | rest | 14.965 | 12.325 | 11.577 |
  | rawrest | **14.902** | 12.046 | 11.290 |

  Restoration helps the noisy third but costs on mid and clean images. Averaging keeps most of both.

## Key finding 2 (2026-10-06 afternoon): keep the spatial-heterogeneity signal (cross-cell std, "cellstats")
For each view row, the extractor stores stage 0-2 feature maps pooled (mean and std) over GxG grid cells. `--cellstats std` appends, for each channel, the std across cells of each grid. These values get a sign*log1p transform and go into the same gridge3 head. The cross-cell std is the second-order (Jensen) term of a per-cell nonlinearity, i.e. the zoned grain-size heterogeneity that eda-analyst found.

Grid 2, std, stages 0-2, all rows. These runs are on the effv2s noise model (13.442):
| variant | CV RMSE |
|---|---|
| baseline: cellstats added | **13.134** (better on 5/5 folds, ΔRMSE -0.308, CI [-0.543, -0.065]) |
| stages 0-1 | 13.39 |
| stages 0-3 | 13.33 |
| channel-mean (block) reduced | 13.24-13.37 |
| mean pool only | 13.26 |
| std pool only | 13.34 |
| no log transform | 13.21 |
| no noise penalty | 13.26 |
| std,range,max | 13.15 |
| + lledge side features | 13.22 |
| effv2s + resnet18, both with cellstats | 13.256 |
| effv2s globals + resnet18 cellstats | 13.396 |

Finer grids on c24n/c248n with `--cs-rows orig`, std, stages 0-2:

| grid spec | CV RMSE |
|---|---|
| 2 | 13.232 |
| 4 | 13.327 |
| 8 | 13.128 |
| 2,4 | 13.020 |
| 2,8 | **12.967** |
| 4,8 | 13.031 |
| 2,4,8 | 13.010 |
| 2,4,8 with std,max | 13.096 |
| 2,4,8 with stages 0,1 | 13.337 |
| 2,4 with `--cs-rows mean` | 13.349 |

Notes on these runs:
- Equal-weight bag of all 7 grid specs: **12.909**. The bag was fixed in advance, so this CV is honest.
- `--cs-rows orig` matters. Aug (noise) rows reuse the identity view's cellstats, so the noise penalty does not act on the cellstat columns. With `all` rows on c24n, grid 2 scores 13.325 instead of 13.232: the 0.06 noise strongly perturbs cell-level std. The g2a extraction (cs2) has no noise0.06 cells, so its `all` rows were harmless.
- With cellstats, inner CV mostly picks lam_aug=0 and a larger alpha (cs24: lam_aug=0 and alpha 100-316 in all folds). The noise penalty adds little once cellstats are present.
- Single-model selection noise is large: the "best-fixed-hp" diagnostic is 0.1-0.3 below the inner-CV result. Bagging over grid specs recovers part of that gap.
- Screens on grid spec 2,4,8 (13.010), not saved:
  - `--hetero ic_acg_len50_gm`: 12.984 (13.952 11.824 13.082 13.539 12.411). This uses in-fold inverse-variance sample weights, the feature-engineer's recipe; the log-var slope is about 1.1.
  - `--hp-avg 5`: 12.984 (13.832 11.690 13.259 13.610 12.407).
  - `--hp-avg 3`: 12.997.
  - All are within noise, about -0.03.

## Key finding 1 (2026-10-06 morning): penalise the head's sensitivity to image noise
`--head gridge3` is a generalized ridge on the view-averaged embedding:
min sum_i sw_i (y_i - Zbar_i w)^2 + lam_view*mean|w.(Z_view - Zbar)|^2 + lam_cell*mean|w.(Z_cell - mean_cell)|^2 + lam_aug*mean|w.(Z_aug - Z_id)|^2 + alpha|w|^2.
Here `Z_aug` is the embedding of a deterministically noised (or blurred, shaded, ...) copy of the same image, and sw = 1 unless `--hetero` is given.
- Noise rows are what help. Before cellstats, effv2s went from 13.636 (4 views, no aug rows) to 13.535 (noise0.03) to 13.442 (noise0.03 + 0.06). resnet18 went from 13.84 to 13.68.
- No help from blur/shade/contrast rows, from grid-cell rows (inner CV always picks lam_cell = 0) or from 8 dihedral views. Orientation views matter up to 4.
- Illumination correction (`--prep ic`), NL-means input, convnext_nano and quantile/max/GeM pooling do not help.
- Joint model with features_v3 inside the head: 13.254, about the same as the feature ridge alone. Residual correlation with the feature and CNN families is 0.90-0.98.

## CLI
`python -m src.extract_embeddings` writes `data/emb/<tag>.npy` (N, rows, D) float32, plus `_ids.csv` and `_meta.json`. This is pure per-image inference: nothing is fitted.
- `--backbone` (common.create_timm, features_only), `--size 256`, `--views N` (first N of id, hflip, transpose, rot90, vflip, rot180, rot270, antitranspose) or `--view-names a,b`
- `--pools mean,std[,max,gem,qXX]`, `--grid G` (global + GxG cell pooling inside the main array; row axis [view][cell])
- `--cell-grids 2,4,8 --cell-stages 0,1,2`: compact per-cell store `<tag>_cells.npy` (N, rows, sum of G^2 cells, Dc), described by `meta["cell_layout"]`. Used by `--cellstats`.
- `--augs noise<s>,blur<s>,shade<s>,gamma<g>,contrast<c>`: deterministic per-image nuisance copies of the identity view (seeded by image ID), stored as `aug:*` rows
- `--prep raw|imgnorm|blur<s>|nlm|median<k>|ic` (numpy steps cached as `data/emb/_img_<step>_1500.npy`)
- `--shard k/n` (run on separate cores) then `--merge-shards n` (same args): concatenates the shards, checks the ID order, and deletes the shard files
- `--bs --threads --limit --tag`

`python -m src.train_head` runs outer 5-fold CV on `data/folds.csv`. Inside each outer training split, the transform, scaler, PCA, sample weights and head hyperparameters (inner leave-one-fold-out CV) are fitted on training images only. Test embeddings are only transformed; the test prediction is the mean of the 5 fold models.

Inputs and features:
- `--emb t1,t2` (concatenated by ID), `--stages 1,2,3` or per tag `"1,2,3;1,2"`, `--pools`, `--views N`
- `--cells global|cells|all`, `--use-augs`, `--aug-filter noise[,blur,...]`, `--extra-rows x1,x2` (one per tag; join several with '+')
- `--cellstats std[,range,max,min]`:
  - `--cs-grid 2|4|8|2,4|...` and `--cs-grid-bag "2;4;8;..."` (a fixed equal-weight bag of full CV runs)
  - `--cs-stages 0,1,2 --cs-pools mean,std --cs-rows all|orig|mean --cs-transform log|none --cs-reduce none|block --cs-weight W`
- `--view-mode mean|aug`, `--view-std`, `--transform none|sqrt|log`, `--block-norm`, `--pca K [--whiten]`
- `--with-feats --feats features_v3.parquet[,...] --feat-weight W`

Head and outputs:
- `--head ridge|gridge|gridge3|krr|svr|lgb`, `--grid-json '{...}'` (override the head grid)
- `--hetero COL[,COL] --hetero-file features_v3.parquet` (gridge3 only): two-pass in-fold inverse-variance weights
- `--hp-avg K`: average the K best grid points
- `--own-col COL --own-file features_v3.parquet --own-q 2/3|1/3,2/3`: own columns per group of a label-free per-image column.
  - Cuts are quantiles over the train images only; group = number of cuts the value exceeds. Every group above 0 gets a NaN-elsewhere copy of the whole input.
  - Grid key `own_w` (default [1]) is their column weight (own ridge group, gridge3).
  - `--own-random SEED` permutes the groups as a control.
- `--name`, `--no-save`, `--oof-out path.csv` (writes a scratch OOF without registering an experiment), `--test-out path.csv` (scratch test predictions), `--note TEXT` (appended to the saved notes)

Defaults are unchanged: with no `--hetero`, `--hp-avg 1` and no `--own-col`, the outputs are bit-identical to before. Checked on c248n grid 2 (13.2318 both ways) and on cs24 (OOF and test identical, 2026-10-06 night).

## Cached files in data/emb/ (command prefix: `taskset -c 3 python -m src.extract_embeddings --threads 1`)
| tag | args | seconds at 1 thread |
|---|---|---|
| tf_efficientnetv2_s.in21k_ft_in1k_256_c248n | `--backbone tf_efficientnetv2_s.in21k_ft_in1k --size 256 --views 4 --pools mean,std --augs noise0.03,noise0.06 --cell-grids 2,4,8 --cell-stages 0,1,2 --tag <tag>`. Run as `--shard 0/2` and `--shard 1/2` on two cores, then `--merge-shards 2`. The cells file is 823 MB. | 2 x ~583 (≈1200 on one core) |
| tf_efficientnetv2_s.in21k_ft_in1k_256_c24n | same with `--cell-grids 2,4` (cells 196 MB; c248n contains it) | 1384 |
| tf_efficientnetv2_s.in21k_ft_in1k_256_g2a | `--backbone tf_efficientnetv2_s.in21k_ft_in1k --views 4 --pools mean,std --grid 2 --augs noise0.03,blur1.0 --tag <tag>` | 1425 |
| tf_efficientnetv2_s.in21k_ft_in1k_256_augs2 | `--backbone tf_efficientnetv2_s.in21k_ft_in1k --views 1 --pools mean,std --augs shade0.12,noise0.06,blur2.0,contrast0.75 --tag <tag>` | 1478 |
| resnet18.a1_in1k_256_g2a | `--backbone resnet18.a1_in1k --views 4 --pools mean,std --grid 2 --augs noise0.03,blur1.0 --tag <tag>` | 1531 |
| resnet18.a1_in1k_256_augs2 | `--backbone resnet18.a1_in1k --views 1 --pools mean,std --augs noise0.06 --tag <tag>` | 191 |
| convnext_nano.d1h_in1k_256_g2a | `--backbone convnext_nano.d1h_in1k --views 4 --pools mean,std --grid 2 --augs noise0.03,blur1.0 --tag <tag>` | 1171 |
| tf_efficientnetv2_s.in21k_ft_in1k_256 | `--backbone tf_efficientnetv2_s.in21k_ft_in1k --views 4` (mean,std,max,gem) | 920 |
| resnet18.a1_in1k_256 | `--backbone resnet18.a1_in1k --views 4` | 467 |
| resnet18.a1_in1k_256_nlm, _q | `--prep nlm` / `--pools mean,std,q10,q50,q90 --tag resnet18.a1_in1k_256_q` | 486 / 1597 |
| tf_efficientnetv2_s.in21k_ft_in1k_256_v8x / resnet18.a1_in1k_256_v8x | `--backbone <bb> --view-names vflip,rot180,rot270,antitranspose --pools mean,std --tag <tag>` | 682 / 386 |
| tf_efficientnetv2_s.in21k_ft_in1k_256_ic_v2n | `--backbone tf_efficientnetv2_s.in21k_ft_in1k --views 2 --pools mean,std --prep ic --augs noise0.03,noise0.06 --tag <tag>` | 651 |
| _img_nlm_1500, _img_ic_1500 | image caches, built automatically by `--prep nlm` / `--prep ic` | ~100 / ~150 |

## Status at hand-back (2026-10-06 night, after the own-slope screen)
Nothing of mine is running. Score.json notes record `emb=<tags>`, plus `[DATA_DIR=...]` when it is not `data`. I added this by hand to `emb_effv2s_256_gridge3_noise_csbag_rest/score.json`. That run's LEADERBOARD line was written before the change, so it lacks the tag; its name carries `_rest`.

## Next steps (prioritized)
0. Blend (night): blend_v12 uses cs24 (weight 0.24) next to the own-slope feature ridge. Try `emb_effv2s_256_gridge3_noise_cs24_hetnz` in place of cs24 or alongside it: the 2-member CV-blend diagnostic gives 12.138 vs 12.171. Don't spend more time on own slopes in the embedding head.
1. Blend: give the ensembler the two singles, `emb_effv2s_256_gridge3_noise_csbag` (12.909, raw) and `emb_effv2s_256_gridge3_noise_csbag_rest` (13.038, restored). Their residual correlation is 0.960. Alternatively use the fixed average `emb_effv2s_256_gridge3_noise_csbag_rawrest` (12.842). These supersede cs2. Next to the feature ridge, cs24 blends better than csbag (diagnostic 12.171 vs 12.195).
2. Cheap follow-up (~10-17 min on 1 core): the same bag with `--hetero ic_acg_len50_gm` or `--hp-avg 5`. Each was about -0.03 on a single spec, which is within noise.
3. Bigger ideas, uncertain:
   - Per-cell nonlinear MIL: the mean over cells of random Fourier or quadratic features of an in-fold PCA of the cell vectors. This generalises the cross-cell std, which is only the 2nd-order Jensen term.
   - effv2s at 512 px input with cells. Extraction costs about 4x; a 1-view pilot is ~20 min.
4. Don't spend more time on:
   - resnet18 cellstats, stage-3 cellstats, max/range cell stats, channel-reduced cellstats, side features inside the head;
   - grid-cell rows, blur/shade/contrast rows, ic/nlm prep, 8 views, convnext_nano, quantile/max pooling, tree heads.
