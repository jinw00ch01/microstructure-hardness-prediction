# Laptop GPU run (orchestrator's plan for the Remote Control session on DESKTOP-C78VQOR)

Folder: `C:\Daker\microstructure-hardness-prediction`, branch `claude/lb-under-10-7080jr`.
The repo is public (user's decision, 2026-10-06). Never commit or push test-set predictions (`experiments/*/test.csv`,
git-ignored) or figures that show competition images. This session does not commit or push at all.

## 1. Pull
`git pull origin claude/lb-under-10-7080jr`

The branch history was rewritten on 2026-10-06 (about 20:30 KST) to drop test predictions and train-label copies from
every commit, and `main` now matches the branch. A clone made before that cannot pull; re-sync it once with
`git fetch origin` then `git reset --hard origin/claude/lb-under-10-7080jr`. Ignored files (`data\`, the GPU runs'
`experiments\*\test.csv`, caches) stay; the 16 early tracked `test.csv` files are removed from the working tree (the
cloud keeps copies).

## 2. CPU prep (no GPU, no wrapper)
- If `data/features_v3.parquet` is missing: `python -W ignore -m src.features --v3 --n_jobs 4`
  (about 13-15 min on 1 core).
- `python -m src.train_cnn --build-deg --deg-k 8 --deg-snr-min 0.5` builds `data/cnn_cache/_pre/nlm.npz` and
  `deg2_k8_snr0.5.npz`. If another cached file is missing, `notes/handoff/feature-engineer.md` has its build command.

## 3. GPU (exclusive use granted by the user, 2026-10-06 19:16 KST)
The user stopped the robot project's GPU work and gave this project the GPU to itself: run directly with
`--device cuda`, **without** gpu_turn.py. All 8 GB of VRAM are free. When ALL GPU work is finished (including the 3-seed
run), return the GPU by running in PowerShell: `New-Item C:\Dacon\WM_Runtime\hardness_gpu_done`.
Never touch the robot project in any other way.

Base flags (best CPU config, `notes/handoff/cnn-trainer.md`):
`--input raw+nlm --deg-p 0.5 --cons 1.0 --pool avg --epochs 30`

a. CUDA check.
b. lr screen, folds 0 and 1, 1 seed, `--no-save`:
   - `--backbone tf_efficientnetv2_s.in21k_ft_in1k` (Apache-2.0) at `--lr 1e-3` and `--lr 3e-4`
   - `--backbone resnet18 --lr 1e-3` as the reference (CPU fold scores are in the cnn-trainer handoff)
c. Best setting, 3 seeds on all 5 folds, `--name cnn_<backbone>_rawnlm_degcons_gpu`. Run it seed by seed so a cut
   still leaves complete 5-fold seeds (`train_cnn` loops folds outside seeds, so one `--seeds 3` call cut short leaves
   the last folds empty and nothing can be blended):
   `--seeds 3 --no-save --train-seeds 0`, then `--train-seeds 1`, then `--train-seeds 2`.
   Then assemble and save with `--seeds N` (same name and flags, N = seeds finished on all 5 folds; everything is
   cached in `data/cnn_cache/<name>/`, so this call trains nothing).
   Measured on the laptop (2026-10-06), base flags: resnet18 about 31 s per fold-seed, effnetv2-s about 76 s (the first
   fold-seed of a run is much slower: 75 s and 301 s, warm-up included).

## 4. Return the results without pushing
Send the cloud orchestrator session (the one you already message) the contents of that experiment's `oof.csv`,
`test.csv` and `score.json`, one file per message, plus the step 3b screen results.
The cloud session blends them for the next submission.

## 5. Result of the 2026-10-06 run (GPU returned 20:11 KST)
- lr screen (1 seed): resnet18 fold 0 13.582, fold 1 12.943; effnetv2-s fold 0 13.635. 3e-4 not screened.
- `cnn_r18_rawnlm_degcons_gpu` (3 seeds) CV 13.593; `cnn_ev2s_rawnlm_degcons_gpu_s3` (3 seeds) CV 13.246.
- In the blend only effnetv2-s gets weight (0.07), and `blend_v10` ties `blend_v7`. Details and next ideas:
  `notes/RESUME.md`.
- The laptop keeps the per-fold-seed caches in `data\cnn_cache\<name>\` and the experiments in `experiments\`.

## 6. Next GPU runs (planned 2026-10-07 00:00 KST)
Why: `blend_v10` (effnetv2-s CNN at 7%) scored public 11.8267 vs v7 11.8795, and a nested two-stage blend gives the
CNN a 13% share (`blend_v11`, CV 12.407). Stronger CNN members are now the main lever.
GPU mode: only as the user says in the thread (exclusive: `--device cuda`, then the done file; shared: wrap each command
with gpu_turn.py). On 2026-10-06 23:59 KST the user granted **exclusive** use in the thread: run with `--device cuda`
without gpu_turn.py, and when all GPU work below is finished run `New-Item C:\Dacon\WM_Runtime\hardness_gpu_done` in
PowerShell to return the GPU. Base flags as in section 3, plus `--device cuda`. PowerShell, repo root.

a. effnetv2-s, 3 more seeds (keeps the 3-seed experiment as it is):
   `Copy-Item data\cnn_cache\cnn_ev2s_rawnlm_degcons_gpu_s3 data\cnn_cache\cnn_ev2s_rawnlm_degcons_gpu_s6 -Recurse`
   then with `--backbone tf_efficientnetv2_s.in21k_ft_in1k --lr 1e-3 --name cnn_ev2s_rawnlm_degcons_gpu_s6 --seeds 6`:
   `--no-save --train-seeds 3`, then `4`, then `5`, then one call without `--no-save`/`--train-seeds` to assemble and save.
b. ConvNeXt-tiny (MIT) for diversity: `--backbone convnext_tiny.fb_in22k_ft_in1k`.
   - Screen fold 0, 1 seed, `--no-save`, at `--lr 1e-4` and `--lr 3e-4`.
   - If the better one is under about 14.0 on fold 0, run it with 3 seeds seed by seed (`--seeds 3 --no-save --train-seeds 0/1/2`,
     then assemble), `--name cnn_cnxt_rawnlm_degcons_gpu_s3`. Otherwise stop after the screen.
c. Send each finished experiment as before (score.json, then oof and test as 2-decimal values in ID order with checksums).
d. When everything is done, return the GPU the way the user's mode requires.

## 7. Result of the 2026-10-07 run (exclusive GPU 23:59-01:23 KST, returned with the done file)
- `cnn_ev2s_rawnlm_degcons_gpu_s6` (6 seeds, seeds 0-2 shared with `_s3`): CV 13.191; seeds 3-5 alone 13.242. About
  3 min per fold-seed this time (vs 76 s on 2026-10-06).
- ConvNeXt-tiny lr screen, fold 0, 1 seed: 1e-4 14.550, 3e-4 13.631. `cnn_cnxt_rawnlm_degcons_gpu_s3` (lr 3e-4, 3 seeds):
  CV 13.286, about 270 s per fold-seed.
- Blend (two-stage, nested): `_s6` share 0.117, nested 12.413 vs 12.407 with `_s3` (blend_v11); ConvNeXt share 0 next to
  effnetv2-s. Neither moves the blend, so blend_v11 stays the candidate. More CNN seeds or ImageNet backbones of this
  recipe are not worth more GPU time; a change of input (e.g. the high-noise restored images) would be.

## 8. Next GPU run: restored-image channel, then full-data CNNs (prepared 2026-10-07 KST by cnn-trainer)
Why:
- The raw+nlm effnetv2-s CNN is the member that helps the public LB (blend_v10 11.8795 -> 11.8267; blend_v14 with CNN
  share 0.2 scored 11.7960). Since blend_v16 the blend uses the CNN only in the "fine_noisy" cell (fine-grained AND
  noisy images, `src.blend_cells.cell_masks`: 132 train / 284 test images). There the CNN mix beats every feature
  model (OOF 11.33; ev2s_s6 11.26). Judge new CNN inputs by the cell RMSE first.
- Steps c-g: a CNN that also sees the restored image (`--input raw+nlm+rest`); restored-image features helped the
  mid-noise tercile, and the CNN is weakest on noisy images.
- Step h: full-data models. Today the CNN test predictions are the mean of 5 fold models trained on 400 images each;
  `--full` trains on all 500 train images and writes only test predictions.
- Step i (optional): `--scale 2` (2x bilinear input upsampling, for the 1-2 px boundaries).

Rules for this run:
- GPU mode only as the user says in the thread for this run. Exclusive: run the [GPU] steps with `--device cuda` as
  written, and when all GPU work is finished (also when stopping early) return the GPU with
  `New-Item C:\Dacon\WM_Runtime\hardness_gpu_done`. Shared: prefix every [GPU] command with
  `python C:\Dacon\RobotWorldModel_ActionVideo\wm_ops\gpu_turn.py --who hardness -- `.
  Never touch the robot project in any other way.
- No commits or pushes. Restorer files (`data\restore_cache\`), `data_restored\`, montages and every `test.csv` stay
  local (all git-ignored).
- The laptop rebuilds the restorer from the repo code: the same bank and settings as the cloud's restorer, trained on
  the GPU in strict fp32. The weights are a new training run, not the cloud's file. That is fine because train and test
  images pass through the same laptop restorer. Never mix laptop and cloud restorer files or `data_restored` folders.
- PowerShell at the repo root with the `.venv` active. [CPU] steps do not use the GPU.

a. [CPU] Pull and check (1-2 min)
   ```powershell
   git status --short
   git pull origin claude/lb-under-10-7080jr
   ```
   The pull may refuse because of local copies from the earlier GPU runs: ` M experiments/LEADERBOARD.md`, or untracked
   `experiments/cnn_*_gpu*/` files that the cloud has committed since. In that case back them up and re-sync as in
   section 1. If any other tracked file shows as modified, stop and ask the cloud session.
   ```powershell
   New-Item -ItemType Directory -Force data\laptop_backup | Out-Null
   Copy-Item experiments\LEADERBOARD.md data\laptop_backup\
   Copy-Item experiments\cnn_*_gpu* data\laptop_backup\ -Recurse -Force
   git fetch origin
   git reset --hard origin/claude/lb-under-10-7080jr
   ```
   Then:
   ```powershell
   Test-Path data\features_v3.parquet, data\cnn_cache\_pre\nlm.npz, data\cnn_cache\_pre\deg2_k8_snr0.5.npz   # True x3
   Test-Path data\features_cal.parquet       # True: train_cnn also prints the fine_noisy cell RMSE
   Test-Path data\restore_cache, data_restored                                                                # False x2
   ```
   - If one of the first three is False, build it as in section 2.
   - If `features_cal.parquet` is missing, the cell lines say n/a. Do not build it for this: the cloud computes the cell
     numbers from the OOF you send.
   - If `data\restore_cache` or `data_restored` exist from an interrupted attempt of this plan, just continue: train
     resumes from its checkpoint and apply overwrites.

b. [CPU] Restorer training pairs (about 7 min)
   ```powershell
   python -W ignore -m src.restore bank --part 0 --nparts 2
   python -W ignore -m src.restore bank --part 1 --nparts 2
   ```
   - Each part ends with `part P: 1953 pairs in ...s; median fit rms 2.2x` and writes `data\restore_cache\bank_partP.npz`
     (about 128 MB). Cloud: 1953 pairs per part, 2.23, about 200 s each on one core.
   - Sources are the 131 cleanest train images; no labels, no test images.

c. [GPU] CUDA check, then train the restorer in strict fp32 (estimate 3-10 min; the cloud took 23 min on 2 CPU threads)
   ```powershell
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   python -W ignore -m src.restore train --device cuda --threads 4 --patch 96 --bs 16 --steps 5000 --report-dir data\restore_cache\report
   ```
   - Same settings as the cloud's restorer: steps 5000, bs 16, patch 96, lr 1e-3, grad-w 0.5, identity 0.15.
   - It logs every 100 steps and checkpoints every 500; rerunning the same command resumes.
   - Writes `data\restore_cache\restore_unet.pt` (3.6 MB) and `meta.json` (`"amp": false`, `"device": "cuda"`).
   - The log ends with the held-out table: 66 synthetic pairs of 11 held-out train sources.
   - Cloud restorer, for comparison: all-band psnr_rest 31.96 (degraded 28.56); low / mid / high band 37.31 / 30.62 /
     28.10. About 31.6 or more is fine. If it is clearly lower, stop and send the table.

d. [GPU] Restore all 1500 images (1-3 min), then [CPU] check them (1-2 min)
   ```powershell
   python -W ignore -m src.restore apply --device cuda --threads 4
   python -W ignore -m src.restore check --report-dir data\restore_cache\report
   (Get-ChildItem data_restored\train -Filter *.png).Count    # 500
   (Get-ChildItem data_restored\test -Filter *.png).Count     # 1000
   Get-ChildItem data_restored -File -Name                    # folds.csv, sample_submission.csv, train.csv
   ```
   - check prints `... all 1500 restored images present (train 500, test 1000)` and the median change by SNR band.
   - Cloud values: MAD noisy 11.62 / mid 5.22 / 0.77-0.9 1.02 / clean 0.00 (n 491 / 517 / 110 / 382). Expect similar
     values.

e. [GPU] Restored channel of the degradation bank (2-4 min)
   ```powershell
   python -W ignore -m src.train_cnn --build-deg --deg-k 8 --deg-snr-min 0.5 --input raw+nlm+rest --device cuda
   ```
   - It first restores 8 noisy train images and compares them with `data_restored\`. The line
     `restorer restore_unet.pt (<sha>, cuda fp32) vs data_restored/ on 8 noisy train images: mean |diff| ...` needs a
     mean |diff| below 0.05; the cloud test gave 0.0012, max 1.
   - If the check fails, the command stops. Run d's apply again, then e, both with `--device cuda`.
   - Then it restores 263 train sources x 8 degraded copies -> `data\cnn_cache\_pre\deg2_k8_snr0.5_rest.npz` (about
     138 MB). Every training start re-checks that file against `data_restored\`.

f. [CPU] Reference, then [GPU] screen (fold 0, seed 0)
   ```powershell
   python -W ignore -m src.train_cnn --backbone tf_efficientnetv2_s.in21k_ft_in1k --input raw+nlm --deg-p 0.5 --cons 1.0 --pool avg --epochs 30 --lr 1e-3 --device cpu --name cnn_ev2s_rawnlm_degcons_gpu_s3 --folds 0 --seeds 1 --no-save
   python -W ignore -m src.train_cnn --backbone tf_efficientnetv2_s.in21k_ft_in1k --input raw+nlm+rest --deg-p 0.5 --cons 1.0 --pool avg --epochs 30 --lr 1e-3 --device cuda --name cnn_ev2s_rawnlmrest_degcons_gpu_s3 --folds 0 --seeds 1 --no-save
   ```
   - The first call trains nothing. It reads the cached raw+nlm seed-0 models and prints their fold RMSEs (fold 0 =
     13.635) and, with features_cal, the cell RMSE by fold. Use fold 0 as the paired reference.
   - The second call is the screen (3-6 min including warm-up). It keeps its test predictions, so g reuses this
     fold-seed.
   - Fold 0 holds only 32 cell images, and single-seed fold scores of one recipe scatter with sd about 0.4. So the
     screen only catches a gross failure.
   - Go on to g unless fold 0 is more than 1.0 worse than the reference overall and also in the cell (or overall
     only, if the cell is n/a). If it is, send the numbers, continue with h, and run g only if the cloud session says so.

g. [GPU] 3 seeds of the restored-channel CNN, seed by seed (20-50 min plus warm-ups)
   ```powershell
   python -W ignore -m src.train_cnn --backbone tf_efficientnetv2_s.in21k_ft_in1k --input raw+nlm+rest --deg-p 0.5 --cons 1.0 --pool avg --epochs 30 --lr 1e-3 --device cuda --name cnn_ev2s_rawnlmrest_degcons_gpu_s3 --seeds 3 --no-save --train-seeds 0
   ```
   - Then the same command with `--train-seeds 1`, then with `--train-seeds 2`.
   - Then run it once more without `--no-save --train-seeds 0` to assemble and save. That call trains nothing.
   - The run writes `experiments\cnn_ev2s_rawnlmrest_degcons_gpu_s3\` and prints the fold RMSEs, the CV, the SNR
     terciles and the cell RMSE.
   - Raw+nlm 3 seeds for comparison: CV 13.246, terciles 15.08 / 12.27 / 12.18, cell 11.26 (cell by fold 10.50 /
     10.16 / 11.46 / 10.63 / 13.16).
   - Per fold-seed about 80 s - 3.5 min; the third input channel costs a little.

h. [GPU] Full-data models: all 500 train images, test predictions only
   - `--full` trains one model per seed on all train images with the same recipe and epochs (31 instead of 25 steps per
     epoch). There is no validation and nothing is selected.
   - The degradation bank uses all of its train sources.
   - It writes `experiments\<name>\test.csv` and `full.json` (no oof.csv, no score.json) plus a LEADERBOARD line.
   - Each seed is cached in `data\cnn_cache\<name>\full_seed<s>.npz` as soon as it is done. So one call runs all seeds,
     and if a run is cut, rerunning the same command skips the finished seeds and then saves.
   - Seed-by-seed calls (`--no-save --train-seeds s`, then one call to assemble) also work, but each call pays the CUDA
     warm-up again (up to about 4 min).
   - A full-data model costs about 1.25x a fold-seed: effnetv2-s about 1.5-4 min, ConvNeXt-tiny about 6 min.
   - A `--full` run needs its own name. Names of fold runs are refused.

   h1. effnetv2-s raw+nlm, 6 seeds (15-30 min):
   ```powershell
   python -W ignore -m src.train_cnn --backbone tf_efficientnetv2_s.in21k_ft_in1k --input raw+nlm --deg-p 0.5 --cons 1.0 --pool avg --epochs 30 --lr 1e-3 --device cuda --full --seeds 6 --name cnn_ev2s_rawnlm_degcons_gpu_full6
   ```
   h2. ConvNeXt-tiny raw+nlm, 3 seeds (about 20 min):
   ```powershell
   python -W ignore -m src.train_cnn --backbone convnext_tiny.fb_in22k_ft_in1k --input raw+nlm --deg-p 0.5 --cons 1.0 --pool avg --epochs 30 --lr 3e-4 --device cuda --full --seeds 3 --name cnn_cnxt_rawnlm_degcons_gpu_full3
   ```
   h3. Only if the cloud session asks for it after g: the same for raw+nlm+rest (`--input raw+nlm+rest --lr 1e-3
   --full --seeds 3 --name cnn_ev2s_rawnlmrest_degcons_gpu_full3`).

i. [GPU] Second fold experiment, run it right after g and before h (orchestrator's order, 2026-10-07): 2x input
   (`--scale 2 --crop 112`)
   - `--scale S` upsamples each augmented training crop and every TTA view bilinearly by S on the GPU. `--crop` stays
     in native pixels.
   - With `--crop 112` the network sees 224 px crops, so training costs as now; the 512 px TTA views make inference
     about 4x. Estimate about 1.7-4 min per fold-seed.
   - The literal `--scale 2` with crop 224 (448 px crops) costs about 4x per step (5-12 min per fold-seed). With the
     consistency twins at bs 16 it probably does not fit in 8 GB, so it is not planned.
   ```powershell
   python -W ignore -m src.train_cnn --backbone tf_efficientnetv2_s.in21k_ft_in1k --input raw+nlm --deg-p 0.5 --cons 1.0 --pool avg --epochs 30 --lr 1e-3 --scale 2 --crop 112 --device cuda --name cnn_ev2s_rawnlm_degcons_x2c112_gpu_s3 --folds 0 --seeds 1 --no-save
   ```
   - Screen and go rule as in f, against the same raw+nlm reference.
   - If it goes on: 3 seeds seed by seed and assemble, as in g (same flags with `--seeds 3 --no-save --train-seeds 0/1/2`,
     then without `--no-save --train-seeds`).

j. Send the results back without pushing, as in sections 4 and 6c, one file per message:
   - restorer: the held-out table of c, the check table of d and the check line of e (text);
   - screens: the fold-0 lines of f (reference and screen: overall, and the cell if printed) and of i;
   - fold runs: `score.json`, then `oof.csv` and `test.csv` as 2-decimal values in ID order with checksums;
   - full-data runs: `full.json`, then `test.csv` the same way (there is no oof.csv).

k. Return the GPU the way the user's mode requires (exclusive: the done file above).

Order and time (exclusive mode, estimates):
| steps | time |
|---|---|
| a-b (CPU) | about 10 min |
| c-e | 10-20 min |
| f | about 5 min |
| g | 25-55 min |
| h1 | 15-30 min |
| h2 | about 20 min |
| i, optional | screen about 5 min; 3 seeds 30-60 min |

About 2-3.5 h in all (1.5-2.5 h without i). Run order g, i, h1, h2 (orchestrator, 2026-10-07): g and i are fold runs
whose OOF the cloud can check in the fine_noisy cell; h1 and h2 only change test predictions. If time runs short,
stop after any finished experiment.

## 9. Next GPU run: grid-aligned het CNN for the noisy images (prepared 2026-10-08 KST by cnn-trainer)
Why:
- blend_v19 (the het4 offset on the 519 low-noise test images) scored 11.0751 against 11.7304 for v18, so het4 is a real
  label term. The noisy images (raw ic_noise >= 9.5: 233 train / 481 test) hold 63.5% of the squared error. The
  segmentation behind het4 (`src/het_blocks.py`) fails on them because they show no grain-boundary lines.
- `src/het_cnn.py` learns, from one image, the 16 block values b_k behind het4 (fixed 4x4 grid of 64-px blocks; het4 =
  their sd) and log N_eff. It sees the full 256 px image, with no crop or resize, so the grid stays aligned.
- It trains only on the 267 clean train images. Each one is re-rendered on the fly in the noisy-preset style: no
  boundary lines, compressed dark phase, blur, noise. The renderer is calibrated against the real noisy train images.
  - Targets come from the clean originals by the het_blocks method. The hardness label is never read.
  - Test images are only passed through the trained models.
- **This supersedes the laptop's own scratch het CNN** (image-level het4 regression on the team's degradation bank).
  That one may still be run as a second variant, but only if the user agrees in the thread and before the GPU is
  returned (step f); send its numbers separately and label them.

Rules: as in section 8.
- GPU mode only as the user says in the thread for this run. Exclusive: run the [GPU] step with `--device cuda` as
  written, and when all GPU work is finished (also when stopping early) return the GPU with
  `New-Item C:\Dacon\WM_Runtime\hardness_gpu_done`. Shared: prefix the [GPU] command with
  `python C:\Dacon\RobotWorldModel_ActionVideo\wm_ops\gpu_turn.py --who hardness -- `. Never touch the robot project in
  any other way.
- No commits or pushes. Everything under `data\het_cnn\` stays local (git-ignored).
- PowerShell at the repo root with the `.venv` active.

a. [CPU] Pull (1-2 min)
   ```powershell
   git status --short
   git pull origin claude/lb-under-10-7080jr
   Test-Path src\het_cnn.py      # True
   ```
   If the pull refuses, back up and re-sync as in 8a. If another tracked file shows as modified, stop and ask the cloud
   session.

b. [CPU] Inputs, prep, unit tests (about 3-5 min)
   ```powershell
   Test-Path data\features_v3.parquet, data\folds.csv, data\het_blocks_train.parquet   # True x3
   ```
   - If `het_blocks_train.parquet` is missing, build it: `python -W ignore -m src.het_blocks` (1-3 min, writes the train
     and test files).
   - Prep (cloud: 31 s on 3 cores; writes `data\het_cnn\prep.npz`, about 160 MB):
     ```powershell
     python -W ignore -m src.het_cnn --prep --threads 4
     ```
     It must print `prep reproduces het_blocks_train.parquet on 267 images (max |dhet4| 0.0e+00, |dN_eff| 0.0e+00)`
     and `... NaN blocks 0; het4 sum 48.727850, N_eff sum 76272.5 ...` (the cloud values). If either sum differs,
     stop and send both lines.
   - Unit tests (about 10 s): `python -W ignore -m src.het_cnn --selftest`. It must end with `selftest passed`.
   - Independent grid/TTA check (about 15 s): `python -W ignore -m src.het_cnn_gridcheck`. It must end with
     `gridcheck passed`.

c. [GPU] CUDA check, then 5 fold models (estimate 30-40 min; 15-20 min if the GPU runs at the 2026-10-06 speed)
   ```powershell
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   Exclusive mode:
   ```powershell
   python -W ignore -m src.het_cnn --device cuda --arch tf_efficientnetv2_s.in21k_ft_in1k --epochs 32 --renders 4 --batch 12 --lr 1e-3 --workers 4 --threads 2 --seed 0 --out ev2s_e32r4
   ```
   Shared mode (the same command behind the wrapper):
   ```powershell
   python C:\Dacon\RobotWorldModel_ActionVideo\wm_ops\gpu_turn.py --who hardness -- python -W ignore -m src.het_cnn --device cuda --arch tf_efficientnetv2_s.in21k_ft_in1k --epochs 32 --renders 4 --batch 12 --lr 1e-3 --workers 4 --threads 2 --seed 0 --out ev2s_e32r4
   ```
   - What it does per fold:
     - trains on the clean images outside the fold: about 213 images x 4 fresh renders x 32 epochs, 71 steps of 12
       per epoch;
     - scores the held-out clean images, each rendered with 4 fixed seeds;
     - predicts all 1500 real images with 8-view TTA.
   - Per epoch it logs `loss (block .. het .. logN ..)` and the seconds per epoch.
   - Per fold it logs a `fold f stage-1 (held-out renders ...)` line: corr(sd b-hat, het4), partial corr given log N,
     block corr, logN corr. A second line gives the same metrics on the real held-out clean images.
   - Each finished fold is kept in `data\het_cnn\ev2s_e32r4\fold<f>.{pt,npz,json}`. Rerunning the same command skips
     finished folds; a fold cut mid-way starts again.
   - At the end it writes `het_cnn_train.parquet`, `het_cnn_test.parquet` and `score.json`.
   - Time check: one fold takes about 32 x (seconds per epoch) plus 1-3 min of prediction.
     - If the first fold is projected above 12 min, stop after the fold that is running when 40 min are used up.
     - Then aggregate the finished folds without training: rerun the command with `--folds 0 1` (the finished
       ones).
   - VRAM: batch 12 at 256 px should need about 2.5-3 GB (cloud estimate, not measured on a GPU). On CUDA out of
     memory, rerun with `--batch 8`.
   - Render workers: 4 DataLoader workers each load the 160 MB prep once. A render costs about 10 ms of CPU, so 4
     workers keep up with the GPU.

d. [CPU] Send the results back without pushing, one message each:
   1. the contents of `data\het_cnn\ev2s_e32r4\score.json` (stage-1 metrics per fold and the exact command);
   2. `python -W ignore -m src.het_cnn --dump ev2s_e32r4 --chunk 0`: the 500 train images;
   3. `... --chunk 1`: test images 1-500;
   4. `... --chunk 2`: test images 501-1000.

   Each dump is ID-ordered `ID,het_cnn,logN_cnn,bmean_cnn` with 4 decimals and ends with a
   `# sums chunk k: het_cnn .. logN_cnn .. bmean_cnn .. nan 0` line (after a partial run, chunk 0 has NaN for the clean
   train images of the unfinished folds; that is expected). The cloud rebuilds the files with
   `python -m src.het_cnn --ingest ev2s_e32r4 c0.txt c1.txt c2.txt`, which checks the IDs and sums.
   - Clean train rows come from the one fold model that did not train on them.
   - Noisy train rows and all test rows are the mean over the 5 fold models.

e. Optional, only if the user agrees in the thread, in the same GPU mode and before f: the scratch het CNN as a second
   variant (see the first bullet).

f. Return the GPU the way the user's mode requires (exclusive: the done file above), also when stopping early.

| step | time |
|---|---|
| a-b (CPU) | about 5 min |
| c (GPU) | 30-40 min (15-20 min at the 2026-10-06 speed) |
| d | a few minutes |
