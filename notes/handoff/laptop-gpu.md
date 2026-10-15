# Laptop GPU run (orchestrator's plan for the Remote Control session on DESKTOP-C78VQOR)

Folder: `C:\Daker\microstructure-hardness-prediction`, branch `claude/lb-under-10-7080jr`.
The repo is public (user's decision, 2026-10-06). Never commit or push test-set predictions (`experiments/*/test.csv`,
git-ignored) or figures that show competition images. This session does not commit or push at all.

**GPU rule from 2026-10-11 18:58 KST until the deadline 2026-10-19 10:00 KST (user, project chat):** this project owns the
GPU unconditionally; the RobotWorldModel session is not working for now. Run GPU steps with `--device cuda` directly (no
gpu_turn.py), no per-run GPU grant and no hand-back file (`hardness_gpu_done`) are needed. The older per-section mode
lines below (shared / exclusive / return the GPU) are superseded for this period. Still never touch the robot project's
files or jobs.

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

## 10. Next GPU run: het CNN with extra targets, and a mid-noise renderer (prepared 2026-10-09 KST by cnn-trainer)
Why:
- blend_v22 (v21 plus the het-CNN offset on the images with raw ic_noise >= 12) scored 10.9742, against 11.0620 for
  v21. So estimators trained on renders of clean train images do transfer to the real noisy images. That band still
  holds about 51% of the squared error of the new base, blend_v22b.
- 10a: the same CNN also learns three label-free scalars, measured on the clean originals from the prep arrays (the
  hardness label is never read):
  - `fd`: dark-grain area fraction;
  - `pore`: area fraction of the het_blocks pore grains;
  - `asp`: area-weighted log aspect ratio of the interior grains.
  On clean images the label follows roughly HV ~ 132 f_dark - 317 phi_pore + 5.4 ln(aspect) + ...
  The noisy preset compresses the dark phase, so the feature models measure fd poorly there. A CNN estimate of fd is
  the next candidate offset term for the noisy band.
  - Head: global mean of the last feature map, then one linear output per target. Loss term: 0.5 x the standardised
    squared error of each target (`--lambda-x 0.5`), with standardisation from the training fold only.
- 10b (optional, after 10a): `--render-preset mid`, for the 68 train images with raw ic_noise 9.5-12. Renders keep
  strong boundary lines (p 0.9, strength 0.3-1.0), lose little dark contrast (c_d 0.8-1.0), get matrix spread x1.0-1.3,
  blur 0.3-1.0 and noise 8-13.
  - Cloud calibration (`hardness-cache/scripts/d1009/D/calib/`): the 9.5-12 band is a mix of two kinds of image.
    - 21 are line-visible, clean-preset-like (ic_ridge_snr >= 0.5; dark offset -17.8, d' 3.35).
    - 47 are noisy-preset-like (offset -8.7, d' 1.53; the existing noisy renderer already covers them).
  - Mid renders: offset -14.3, d' 2.82, noise 10.8 (band median 10.3). Their lines are weaker than in the real
    line-visible images: ic_ridge_snr median 0.32 against 0.62.
  - So 10b is exploratory and worth less than 10a.
- With none of the new flags, `src/het_cnn.py` reproduces the earlier runs bit for bit. Checked on CPU against the
  previous code with 1-epoch resnet18 runs: every fold array, weight, parquet column and log number identical.
  `--dump`/`--ingest` of the old 3-column format are unchanged too.

Rules: as in section 9.
- GPU mode only as the user says in the thread for this run. Exclusive: run the [GPU] steps with `--device cuda` as
  written, and when all GPU work is finished (also when stopping early) return the GPU with
  `New-Item C:\Dacon\WM_Runtime\hardness_gpu_done`. Shared: prefix each [GPU] command with
  `python C:\Dacon\RobotWorldModel_ActionVideo\wm_ops\gpu_turn.py --who hardness -- `. Never touch the robot project in
  any other way.
- No commits or pushes. Everything under `data\het_cnn\` stays local (git-ignored).
- PowerShell at the repo root with the `.venv` active.

a. [CPU] Pull, unit tests, CPU smoke of the new options (3-5 min)
   ```powershell
   Set-Location C:\Daker\microstructure-hardness-prediction
   git status --short
   git pull origin claude/lb-under-10-7080jr
   Test-Path data\het_cnn\prep.npz, data\folds.csv, data\features_v3.parquet        # True x3
   python -W ignore -m src.het_cnn --selftest        # 7 checks; must end with "selftest passed"
   python -W ignore -m src.het_cnn_gridcheck         # must end with "gridcheck passed"
   ```
   - If the pull refuses, back up and re-sync as in 8a. If another tracked file shows as modified, stop and ask the
     cloud session.
   - `prep.npz` is the file built in 9b; do not rebuild it. If it is missing, build it as in 9b.
   - Then the CPU smoke runs. They take about 1 min each, use no GPU, and test the Windows spawn workers:
     ```powershell
     python -W ignore -m src.het_cnn --device cpu --smoke --folds 0 --epochs 1 --renders 1 --batch 8 --threads 2 --workers 2 --extra-targets fd pore asp --out smoke_x
     python -W ignore -m src.het_cnn --device cpu --smoke --folds 0 --epochs 1 --renders 1 --batch 8 --threads 2 --workers 2 --render-preset mid --out smoke_mid
     ```
   - Each run must end with `total wall ...`.
   - The first run starts with this line (cloud values):
     `extra targets (clean originals, from the prep arrays): fd mean 0.1727 sd 0.0881, pore mean 0.0027 sd 0.0051, asp mean 0.7023 sd 0.1969`
     If a value differs by more than 0.001, send the line, but go on.

b. [GPU] 10a: extra targets, seed 0 (about 25-30 min; seeds 0 and 1 of section 9 took about 50 min together)
   ```powershell
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   python -W ignore -m src.het_cnn --device cuda --arch tf_efficientnetv2_s.in21k_ft_in1k --epochs 32 --renders 4 --batch 12 --lr 1e-3 --workers 4 --threads 2 --seed 0 --extra-targets fd pore asp --out ev2s_e32r4_x
   ```
   - These are the ev2s_e32r4 settings plus `--extra-targets`.
   - The header line ends with `| extra targets ['fd', 'pore', 'asp'] lambda_x 0.5`.
   - Each epoch line adds `fd .. pore .. asp ..` after `logN ..`.
   - Each fold adds a line `fold f extra targets, held-out renders: fd corr .. R2 .., pore ..., asp ... | K-avg: ... |
     real held-out clean: ...`.
   - The trunk is shared with the extra heads, so this run's het_cnn / logN_cnn differ from ev2s_e32r4 seed 0. Keep
     both runs.
   - Cut-off rule, resume and out-of-memory handling as in 9c. A rerun skips the finished folds. It stops with a
     message if `--out` holds folds trained with other `--extra-targets` or another `--render-preset`.

c. [GPU] 10b: mid-noise renderer (about 25-30 min). Run it only after 10a has finished and only if the GPU is still
   yours.
   ```powershell
   python -W ignore -m src.het_cnn --device cuda --arch tf_efficientnetv2_s.in21k_ft_in1k --epochs 32 --renders 4 --batch 12 --lr 1e-3 --workers 4 --threads 2 --seed 0 --render-preset mid --out ev2s_mid_e32r4
   ```
   - The header line ends with `| render preset mid`.
   - The stage-1 lines score mid-style renders of the held-out clean images. Expect higher numbers than in 9c, because
     these renders keep the boundary lines.

d. [CPU] Send the results back without pushing, one message each:
   1. the contents of `data\het_cnn\ev2s_e32r4_x\score.json`;
   2. `python -W ignore -m src.het_cnn --dump ev2s_e32r4_x --chunk 0` (the 500 train images);
   3. `python -W ignore -m src.het_cnn --dump ev2s_e32r4_x --chunk 1` (test images 1-500);
   4. `python -W ignore -m src.het_cnn --dump ev2s_e32r4_x --chunk 2` (test images 501-1000);
   5. if 10b ran: the contents of `data\het_cnn\ev2s_mid_e32r4\score.json`;
   6. `python -W ignore -m src.het_cnn --dump ev2s_mid_e32r4 --chunk 0`;
   7. `python -W ignore -m src.het_cnn --dump ev2s_mid_e32r4 --chunk 1`;
   8. `python -W ignore -m src.het_cnn --dump ev2s_mid_e32r4 --chunk 2`.

   Format of the dumps:
   - 10a dumps have the columns `ID,het_cnn,logN_cnn,bmean_cnn,fd_cnn,pore_cnn,asp_cnn` with 4 decimals. Their
     `# sums chunk k:` line covers all six columns.
   - 10b dumps keep the old 3-column format.
   - To save a dump to a file instead of copying the console, use
     `... | Out-File -Encoding utf8 c0.txt`, not `>`, which writes UTF-16 in Windows PowerShell.

   The cloud rebuilds the parquets with
   `python -m src.het_cnn --ingest ev2s_e32r4_x c0.txt c1.txt c2.txt` (and the same for ev2s_mid_e32r4). It checks the
   IDs and every column's sum.

e. Return the GPU the way the user's mode requires, also when stopping early. Exclusive mode:
   ```powershell
   New-Item C:\Dacon\WM_Runtime\hardness_gpu_done
   ```

| step | time |
|---|---|
| a (CPU) | 3-5 min |
| b, 10a (GPU) | 25-30 min |
| c, 10b (GPU, optional) | 25-30 min |
| d | a few minutes |

## 11. Next GPU run: het CNN that localizes grain size within the image (prepared 2026-10-08 evening by cnn-trainer)
This replaces section 10 as the next GPU run (10a/10b are not needed now: on 2026-10-08 the cloud found that the noisy
images' remaining error is grain-size non-uniformity, not the dark-phase fraction).

Why:
- The het CNN of section 9 predicts each image's average grain size well (corr 0.97) but barely tells the 16 blocks of
  one image apart (within-image block corr 0.36; partial corr of het with the truth given grain count 0.25 / 0.14 for
  seeds 0 / 1). The hardness signal we need is exactly that within-image spread.
- New options in `src/het_cnn.py` (commit 92d6ebc; defaults reproduce the earlier runs bit for bit, selftest check 12):
  - `--head fpn`: merges stride-8/16/32 features into a 16x16 map, with a block head and a 16x16 cell head (extra loss
    on 16-px cells, `--lambda-cell` 1.0).
  - `--mosaic 0.5 --mosaic-dlogn 0.5`: half of the cartoon renders join two training images of similar grain count
    along a smooth random region boundary (real images also mix coarse and fine regions).
  - `--zoom 1.0 1.4`: half of the cartoon renders zoom the label map in by 1.0-1.4 (more grain-size variety).
  - `--lambda-within 1`: an extra loss on each block's deviation from its image's mean (fights shrinkage).
- Label-free throughout: targets come from the clean originals' label maps; the hardness label is never read; test
  images are only predicted.

Rules: as in section 9 (GPU mode only as the user says in the thread for this run; exclusive mode returns the GPU with
`New-Item C:\Dacon\WM_Runtime\hardness_gpu_done`, also when stopping early; shared mode wraps each [GPU] command with
`python C:\Dacon\RobotWorldModel_ActionVideo\wm_ops\gpu_turn.py --who hardness -- `; never touch the robot project in any
other way; no commits or pushes; PowerShell at the repo root with the `.venv` active).

a. [CPU] Pull, unit tests, CPU smoke (about 5 min)
   ```powershell
   Set-Location C:\Daker\microstructure-hardness-prediction
   git status --short
   git pull origin claude/lb-under-10-7080jr
   python -W ignore -m src.het_cnn --selftest        # 12 checks; must end with "selftest passed"
   python -W ignore -m src.het_cnn --device cpu --smoke --folds 0 --epochs 1 --renders 1 --batch 8 --threads 2 --workers 2 --head fpn --mosaic 0.5 --mosaic-dlogn 0.5 --zoom 1.0 1.4 --lambda-within 1 --out smoke_loc
   ```
   - The smoke run must end with `total wall ...`; it tests the Windows spawn workers with the new renders.
   - Check 12 compares the default flags against commit 5d99d75 via `git show`; it needs the full git history (a normal
     clone has it). If only check 12 fails with a git error, send the message and go on.

b. [GPU] 11a: combined design, seed 0 (about 30-40 min)
   ```powershell
   python -W ignore -m src.het_cnn --device cuda --arch tf_efficientnetv2_s.in21k_ft_in1k --epochs 32 --renders 4 --batch 12 --lr 1e-3 --workers 4 --threads 2 --seed 0 --head fpn --mosaic 0.5 --mosaic-dlogn 0.5 --zoom 1.0 1.4 --lambda-within 1 --out ev2s_loc_e32r4
   ```
   - Same settings as ev2s_e32r4 plus the four options. Memory per batch equals the section 9 run (checked on CPU), so
     batch 12 fits; on out-of-memory use `--batch 8` and say so.
   - Each fold prints a `stage-1` line; the numbers to watch are `partial | logN` and `within-image` (old run: 0.25 and
     about 0.36).
   - Cut-off rule and resume as in 9c (a rerun skips finished folds).

c. [GPU] 11c: second run. FIRST run `git pull origin claude/lb-under-10-7080jr` and use the command in this step as it
   reads after the pull (the cloud may replace it after its CPU comparison). Default if unchanged: seed 1 of 11a.
   ```powershell
   python -W ignore -m src.het_cnn --device cuda --arch tf_efficientnetv2_s.in21k_ft_in1k --epochs 32 --renders 4 --batch 12 --lr 1e-3 --workers 4 --threads 2 --seed 1 --head fpn --mosaic 0.5 --mosaic-dlogn 0.5 --zoom 1.0 1.4 --lambda-within 1 --out ev2s_loc_e32r4_s1
   ```

d. [CPU] Send the results back without pushing, one message each (use `| Out-File -Encoding utf8 c0.txt` to save a
   dump to a file, not `>`):
   1. the contents of `data\het_cnn\ev2s_loc_e32r4\score.json`;
   2. `python -W ignore -m src.het_cnn --dump ev2s_loc_e32r4 --chunk 0`;
   3. `python -W ignore -m src.het_cnn --dump ev2s_loc_e32r4 --chunk 1`;
   4. `python -W ignore -m src.het_cnn --dump ev2s_loc_e32r4 --chunk 2`;
   5. the contents of `data\het_cnn\ev2s_e32r4\score.json` (the old seed-0 run, for the stage-1 comparison);
   6-9. the same as 1-4 for the 11c run's `--out` directory, if it ran.
   Send 1-5 as soon as 11a is done, before 11c finishes.

e. Return the GPU the way the user's mode requires, also when stopping early. Exclusive mode:
   ```powershell
   New-Item C:\Dacon\WM_Runtime\hardness_gpu_done
   ```

| step | time |
|---|---|
| a (CPU) | about 5 min |
| b, 11a (GPU) | 30-40 min |
| c, 11c (GPU) | 30-40 min |
| d | a few minutes |

## 12. Next GPU run: a stronger localized het CNN (prepared 2026-10-09 14:10 KST)
Why:
- Section 11 worked: the localized het CNN (ev2s_loc_e32r4 + _s1) passed its pre-registered label test (noisy line-free
  images 13.77 -> 13.38, nested CV 11.653 -> 11.465) and became blend_v24. The gain grew with the render partial corr
  (old 0.25 -> 0.49), so a design that localizes better should gain more. Pre-registration: hardness-cache
  `scripts/d1009/H/PREREG.txt`.
- 12a screens four one-change variants on folds 0 and 1 only (`--screen`: no parquets, label-free); 12b picks one with a
  fixed rule; 12c trains the pick on all 5 folds and dumps it.

Rules: as in section 11 (GPU mode only as the user says in the thread for this run; exclusive mode returns the GPU with
`New-Item C:\Dacon\WM_Runtime\hardness_gpu_done`, also when stopping early; shared mode wraps each [GPU] command with
`python C:\Dacon\RobotWorldModel_ActionVideo\wm_ops\gpu_turn.py --who hardness -- `; never touch the robot project in any
other way; no commits or pushes; PowerShell at the repo root with the `.venv` active).

a. [CPU] Pull and selftest (2 min)
   ```powershell
   Set-Location C:\Daker\microstructure-hardness-prediction
   git status --short
   git pull origin claude/lb-under-10-7080jr
   python -W ignore -m src.het_cnn --selftest        # must end with "selftest passed"
   Test-Path data\het_cnn\ev2s_loc_e32r4\score.json, data\het_cnn\ev2s_loc_e32r4_s1\score.json   # True x2
   ```

b. [GPU] 12a: four screens, seed 0, folds 0 and 1 (about 50 min in total; fastest first)
   ```powershell
   python -W ignore -m src.het_cnn --device cuda --arch tf_efficientnetv2_s.in21k_ft_in1k --batch 12 --lr 1e-3 --workers 4 --threads 2 --seed 0 --head fpn --mosaic 0.5 --mosaic-dlogn 0.5 --zoom 1.0 1.4 --folds 0 1 --screen --epochs 32 --renders 4 --lambda-within 2 --out scr12_lw2
   python -W ignore -m src.het_cnn --device cuda --arch tf_efficientnetv2_s.in21k_ft_in1k --batch 12 --lr 1e-3 --workers 4 --threads 2 --seed 0 --head fpn --mosaic 0.5 --mosaic-dlogn 0.5 --zoom 1.0 1.4 --folds 0 1 --screen --epochs 32 --renders 4 --lambda-within 4 --out scr12_lw4
   python -W ignore -m src.het_cnn --device cuda --arch tf_efficientnetv2_s.in21k_ft_in1k --batch 12 --lr 1e-3 --workers 4 --threads 2 --seed 0 --head fpn --mosaic 0.5 --mosaic-dlogn 0.5 --zoom 1.0 1.4 --folds 0 1 --screen --epochs 64 --renders 4 --lambda-within 1 --out scr12_e64
   python -W ignore -m src.het_cnn --device cuda --arch tf_efficientnetv2_s.in21k_ft_in1k --batch 12 --lr 1e-3 --workers 4 --threads 2 --seed 0 --head fpn --mosaic 0.5 --mosaic-dlogn 0.5 --zoom 1.0 1.4 --folds 0 1 --screen --epochs 32 --renders 8 --lambda-within 1 --out scr12_r8
   ```
   - Each is ev2s_loc_e32r4 with one change. Each run ends with `wrote ...\score.json (folds [0, 1])`.
   - If the GPU must go back early, stop after the current run; 12b uses whatever finished.

c. [CPU] 12b: the pick (seconds)
   ```powershell
   python -m src.het_cnn_pick --base ev2s_loc_e32r4 ev2s_loc_e32r4_s1 --runs scr12_lw2 scr12_lw4 scr12_e64 scr12_r8
   ```
   - The last line is `PICK <run>` or `PICK none`. Send the whole output to the cloud session right away.
   - `PICK none`: skip 12c, go to e.

d. [GPU] 12c: the pick on all 5 folds, seed 0 (lw2/lw4 about 20 min; e64/r8 about 40 min)
   ```powershell
   # PICK scr12_lw2:
   python -W ignore -m src.het_cnn --device cuda --arch tf_efficientnetv2_s.in21k_ft_in1k --batch 12 --lr 1e-3 --workers 4 --threads 2 --seed 0 --head fpn --mosaic 0.5 --mosaic-dlogn 0.5 --zoom 1.0 1.4 --epochs 32 --renders 4 --lambda-within 2 --out ev2s_loc12
   # PICK scr12_lw4:
   python -W ignore -m src.het_cnn --device cuda --arch tf_efficientnetv2_s.in21k_ft_in1k --batch 12 --lr 1e-3 --workers 4 --threads 2 --seed 0 --head fpn --mosaic 0.5 --mosaic-dlogn 0.5 --zoom 1.0 1.4 --epochs 32 --renders 4 --lambda-within 4 --out ev2s_loc12
   # PICK scr12_e64:
   python -W ignore -m src.het_cnn --device cuda --arch tf_efficientnetv2_s.in21k_ft_in1k --batch 12 --lr 1e-3 --workers 4 --threads 2 --seed 0 --head fpn --mosaic 0.5 --mosaic-dlogn 0.5 --zoom 1.0 1.4 --epochs 64 --renders 4 --lambda-within 1 --out ev2s_loc12
   # PICK scr12_r8:
   python -W ignore -m src.het_cnn --device cuda --arch tf_efficientnetv2_s.in21k_ft_in1k --batch 12 --lr 1e-3 --workers 4 --threads 2 --seed 0 --head fpn --mosaic 0.5 --mosaic-dlogn 0.5 --zoom 1.0 1.4 --epochs 32 --renders 8 --lambda-within 1 --out ev2s_loc12
   ```
   - Run only the line of the pick. Cut-off rule and resume as in 9c (a rerun skips finished folds).

e. [CPU] Send back without pushing, one message each: the 12b output (if not sent yet); the stage-1 summary of
   `data\het_cnn\ev2s_loc12\score.json` (pooled and per-fold P, W, R as in section 11); then
   `python -W ignore -m src.het_cnn --dump ev2s_loc12 --chunk 0`, `--chunk 1`, `--chunk 2`.

f. Return the GPU the way the user's mode requires, also when stopping early. Exclusive mode:
   ```powershell
   New-Item C:\Dacon\WM_Runtime\hardness_gpu_done
   ```

| step | time |
|---|---|
| a (CPU) | 2 min |
| b, 12a (GPU) | about 50 min |
| c, 12b (CPU) | seconds |
| d, 12c (GPU) | 20-40 min |
| e | a few minutes |

## 13. Next GPU run: route K, a realistic render prior (D1), else an ev2m backbone (D2) (prepared 2026-10-09 night)
Why:
- The localized het CNN of blend_v24 (ev2s_loc_e32r4 + _s1) learns from renders that are more blurred and carry fewer
  boundary lines than the real line-free noisy images (G1). D1 trains the same design on a widened, more realistic
  render prior (`--render-preset real`). Its label-free gates passed on the cloud (400 renders of clean train images
  with the pre-registered seeds): fast-statistic domain AUC vs the 208 real G1 train images 0.792 (current prior 0.902;
  bar <= 0.82), 98% of the renders keep ic_ridge_snr < 0.5 (bar >= 90%), and the inferred render parameters of 94% of
  real G1 train fall inside its box (bar >= 85%; the earlier P7r box held 50%).
- D2, only if D1 does not qualify: the same design with the tf_efficientnetv2_m backbone (Apache-2.0, 52.2M backbone
  parameters vs 19.9M for ev2s) on the current prior. Its FPN picks the stride 8/16/32 maps (80/176/512 channels).
- New options in `src/het_cnn.py` (defaults are bit-identical to before; selftest checks 13 and 13b):
  - `--render-preset real`: the D1 prior.
  - `--eval-presets noisy real`: after each fold, scores the fold model on fixed held-out renders of both priors. These
    renders are identical in every run. It also predicts the 208 real G1 train images (prediction only, no label).
  - `--eval-only RUN`: the same scores for an existing run's fold models, written to `score_eval.json`. It never
    touches `score.json` or the parquets.
  - The pick rule is in `src/het_cnn_pick_k.py`. Pre-registration: hardness-cache `scripts/d1010/K/PREREG.txt`.

Rules: as in section 12 (GPU mode only as the user says in the thread for this run; exclusive mode returns the GPU with
`New-Item C:\Dacon\WM_Runtime\hardness_gpu_done`, also when stopping early; shared mode wraps each [GPU] command with
`python C:\Dacon\RobotWorldModel_ActionVideo\wm_ops\gpu_turn.py --who hardness -- `; never touch the robot project in any
other way; no commits or pushes; PowerShell at the repo root with the `.venv` active).

a. [CPU] Pull, selftest, check the base checkpoints (3 min)
   ```powershell
   Set-Location C:\Daker\microstructure-hardness-prediction
   git status --short
   git pull origin claude/lb-under-10-7080jr
   python -W ignore -m src.het_cnn --selftest        # must end with "selftest passed" (checks 1-13b)
   git hash-object src\het_cnn.py src\het_cnn_pick_k.py   # git blob ids, the same with or without CRLF checkout
   Test-Path data\het_cnn\ev2s_loc_e32r4\fold0.pt, data\het_cnn\ev2s_loc_e32r4\fold1.pt, data\het_cnn\ev2s_loc_e32r4_s1\fold0.pt, data\het_cnn\ev2s_loc_e32r4_s1\fold1.pt   # True x4
   ```
   - Check 13 prints `208 real G1 train images`; another count means the laptop's `features_v3.parquet` differs from
     the cloud's: send the message and stop.
   - The two blob ids must be the pre-registered ones (hardness-cache `scripts/d1010/K/PREREG.txt`, addendum 2):
     `src\het_cnn.py` bc683a5218f713bb0a019f74b0c0fd5a6927a901,
     `src\het_cnn_pick_k.py` e44c34d2dce20a1d673f94ba9143c846d785483e. Otherwise send them and stop.

b. [GPU] Base scores on the fixed renders (about 3 min)
   ```powershell
   python -W ignore -m src.het_cnn --eval-only ev2s_loc_e32r4 --folds 0 1 --eval-presets noisy real --device cuda --threads 2
   python -W ignore -m src.het_cnn --eval-only ev2s_loc_e32r4_s1 --folds 0 1 --eval-presets noisy real --device cuda --threads 2
   ```
   - Each ends with `wrote ...\score_eval.json`. The `noisy: P` of each fold should match that run's section-11
     stage-1 values within about 0.005 (seed 0: 0.6416 / 0.3807; seed 1: 0.6471 / 0.4157). A larger gap: say so in the
     message and go on; step d checks the mean against 0.5213 +- 0.01 and stops if it is off.

c. [GPU] D1 screen, seed 0, folds 0 and 1 (about 15 min)
   ```powershell
   python -W ignore -m src.het_cnn --device cuda --arch tf_efficientnetv2_s.in21k_ft_in1k --epochs 32 --renders 4 --batch 12 --lr 1e-3 --workers 4 --threads 2 --seed 0 --head fpn --mosaic 0.5 --mosaic-dlogn 0.5 --zoom 1.0 1.4 --lambda-within 1 --render-preset real --folds 0 1 --screen --eval-presets noisy real --out scrK_d1
   ```
   - Same as ev2s_loc_e32r4 except `--render-preset real`, plus the screen and evaluation switches. Near the end it
     prints `wrote ...\score.json (folds [0, 1])`. A fold is finished only after its `fold k eval presets:` line.

d. [CPU] The pick (seconds)
   ```powershell
   python -m src.het_cnn_pick_k --base ev2s_loc_e32r4 ev2s_loc_e32r4_s1 --d1 scrK_d1
   ```
   - Send the whole output to the cloud session right away.
   - First it checks step b: the base fold models' P_cur on the fixed current-prior renders must reproduce 0.5213
     within 0.01. Otherwise it prints `BASE CHECK FAIL` and decides nothing: send the output and stop the GPU work.
   - What it compares (folds 0-1 means):
     - D1 R against 0.7358: real held-out clean images.
     - D1 W against 0.6865: within-image block corr on the fixed current-prior renders.
     - D1 P_cur against 0.5013 (fixed).
     - D1 P_real against the base P_real + 0.05: the base fold models scored on the identical 'real' renders in step b.
     - The real G1 logN-hat mean and sd must be within 0.10 of the base fold models' on the same 208 images.
     - It also prints the agreement with the v24 estimator on real G1 train (bar 0.75). This is a sanity check only
       and never decides.
   - Last line `PICK scrK_d1`: skip e and go to f with the D1 line. `NEXT d2`: go to e.
   - A line starting `ERROR:` (a fold without its eval, the wrong preset or backbone) means no decision. If it names an
     `--eval-only` command, run that command [GPU] and then d again; otherwise send the output and stop.

e. [GPU] Only after `NEXT d2`: the D2 screen, then the pick again (about 30-40 min)
   ```powershell
   python -c "from src.common import create_timm; create_timm('tf_efficientnetv2_m.in21k_ft_in1k', pretrained=True, features_only=True, in_chans=1); print('ev2m weights ok')"
   python -W ignore -m src.het_cnn --device cuda --arch tf_efficientnetv2_m.in21k_ft_in1k --epochs 32 --renders 4 --batch 12 --lr 1e-3 --workers 4 --threads 2 --seed 0 --head fpn --mosaic 0.5 --mosaic-dlogn 0.5 --zoom 1.0 1.4 --lambda-within 1 --folds 0 1 --screen --eval-presets noisy real --out scrK_m
   python -m src.het_cnn_pick_k --base ev2s_loc_e32r4 ev2s_loc_e32r4_s1 --d1 scrK_d1 --d2 scrK_m
   ```
   - The first line downloads the ev2m weights once (about 210 MB, GitHub release, Apache-2.0).
   - On CUDA out of memory, rerun the screen with `--batch 8` and say so. The cloud measured about 1.6x the
     activation memory of ev2s at the same batch.
   - Send the pick output. `PICK scrK_m`: go to f with the D2 line. `PICK none`: stop the GPU work and go to g. Route
     K ends without a file. `ERROR: ... folds ... not done` or `ERROR: ...scrK_m\score.json missing` after a cut: rerun
     the screen line (it skips finished folds), then the pick again. Any other `ERROR:` line follows the rule in d.

f. [GPU] The pick on all 5 folds, seed 0 (D1 about 25 min; D2 about 60-80 min)
   ```powershell
   # PICK scrK_d1:
   python -W ignore -m src.het_cnn --device cuda --arch tf_efficientnetv2_s.in21k_ft_in1k --epochs 32 --renders 4 --batch 12 --lr 1e-3 --workers 4 --threads 2 --seed 0 --head fpn --mosaic 0.5 --mosaic-dlogn 0.5 --zoom 1.0 1.4 --lambda-within 1 --render-preset real --eval-presets noisy real --out ev2sK_d1
   python -m src.het_cnn_pick_k --stage1b ev2sK_d1 --design d1
   # PICK scrK_m:
   python -W ignore -m src.het_cnn --device cuda --arch tf_efficientnetv2_m.in21k_ft_in1k --epochs 32 --renders 4 --batch 12 --lr 1e-3 --workers 4 --threads 2 --seed 0 --head fpn --mosaic 0.5 --mosaic-dlogn 0.5 --zoom 1.0 1.4 --lambda-within 1 --eval-presets noisy real --out ev2mK
   python -m src.het_cnn_pick_k --stage1b ev2mK --design d2
   ```
   - Run only the pair of lines for the pick. If the screen needed `--batch 8`, use it here too.
   - Cut-off rule and resume as in 9c: a rerun skips finished folds.
   - The `--stage1b` line prints per-fold and pooled stage-1 numbers and ends with `STAGE1B PASS`, `STAGE1B FAIL` or
     `STAGE1B INCOMPLETE` (fewer than 5 folds: rerun the training line, it skips finished folds; if the GPU must go
     back first, return it and resume later). The bars are pooled P_cur >= 0.48 and R (5-fold mean) >= 0.70 for D1,
     and pooled P >= 0.54 for D2. Make the dumps in g after PASS or FAIL; the cloud runs the label test only after a
     PASS.
   - A line starting `ERROR:` decides nothing. If it names an `--eval-only` command (a fold was cut between training
     and its eval), run that command [GPU] and then the `--stage1b` line again. A missing `score.json` after a cut:
     rerun the training line. Any other `ERROR:`: send the output and stop.

g. [CPU] Send back without pushing, one message each. Save a dump with `| Out-File -Encoding utf8 c0.txt`, not `>`.
   After `PICK none`, `BASE CHECK FAIL` or a stopping `ERROR:`, send item 1 only (there is no 5-fold run to dump).
   1. The output of d (and e, if it ran), if not sent yet.
   2. The `--stage1b` output of f.
   3-5. The dumps of the 5-fold run (`ev2sK_d1`, or `ev2mK` for D2):
   ```powershell
   python -W ignore -m src.het_cnn --dump ev2sK_d1 --chunk 0 | Out-File -Encoding utf8 c0.txt
   python -W ignore -m src.het_cnn --dump ev2sK_d1 --chunk 1 | Out-File -Encoding utf8 c1.txt
   python -W ignore -m src.het_cnn --dump ev2sK_d1 --chunk 2 | Out-File -Encoding utf8 c2.txt
   ```

h. Return the GPU the way the user's mode requires, also when stopping early. Exclusive mode:
   ```powershell
   New-Item C:\Dacon\WM_Runtime\hardness_gpu_done
   ```

| step | time |
|---|---|
| a (CPU) | 3 min |
| b (GPU) | about 3 min |
| c (GPU) | about 15 min |
| d (CPU) | seconds |
| e (GPU, only after `NEXT d2`) | 30-40 min |
| f (GPU) | D1 about 25 min; D2 about 60-80 min |
| g | a few minutes |

## 14. Next GPU run: het reader for the noisy line-free images (GPU-A), then a 1/N-weighted y-CNN (GPU-B), optional GPU-C (prepared 2026-10-11 06:00 KST)
Why:
- blend_v29g (the 10-11 file) improves het where it is measured on the image. The 320 noisy line-free test rows (G1)
  still read het with the localized het CNN of blend_v24 (ev2s_loc_e32r4 + _s1), trained on renders cleaner than the
  real G1 images. GPU-A tries three label-free ways to read het better there; the cloud tests each once against a
  bar fixed beforehand (hardness-cache `scripts/d1011/bold/PLAN_BOLD.txt` section 4 and its 2026-10-11 05:55 addendum):
  - A1: AdaBN. The same fold models, with batch-norm statistics re-estimated on the 208 real G1 TRAIN images (no
    label; test images are only predicted).
  - A2: retrain with half of the renders passed through the noise world of `tools/fsim` (degrade_sim).
  - A3: route K's realistic render prior (`--render-preset real`) on all 5 folds.
- GPU-B: the y-CNN recipe of `cnn_ev2s_rawnlm_degcons_gpu_s3` with each train image's loss weighted by its label-free
  grain count N (`--wN 1`; images with few grains carry noisier labels). New option, off by default and bit-identical
  at 0. It reads `data\het_cnn\ev2s_loc_e32r4{,_s1}\het_cnn_train.parquet` (no label).
- GPU-C (optional, only after B in the same session): the same with full 256-px crops and a 4x4 grid pool, 45 epochs.

Rules:
- GPU mode only as the user says in the thread for this run. The expected mode is the hand-over tool: if a robot job
  holds the GPU, arm `C:\Dacon\WM_Runtime\handover_after_jobs.py <ledger job id>` as on 10-09 and start the [GPU]
  steps once `handover_started.json` appears. Keep watching until the GPU is actually ours (a waiter that exits early
  misses the hand-over). Then run in exclusive mode and return the GPU with
  `New-Item C:\Dacon\WM_Runtime\hardness_gpu_done` after the last [GPU] step, also when stopping early.
- Never pause, stop or edit the robot project's jobs or files in any other way. If the robot side asks for the GPU
  back, stop after the current command, return it, and say which step was next (every training line resumes).
- Shared mode only if the user says so: wrap each [GPU] command with
  `python C:\Dacon\RobotWorldModel_ActionVideo\wm_ops\gpu_turn.py --who hardness -- `, and start B only if
  `nvidia-smi` shows memory.used <= 3 GB (C: <= 2.5 GB).
- CUDA out of memory (this replaces "lower the batch size" for this section):
  - First rerun the IDENTICAL command; it resumes from the finished folds / fold-seeds.
  - A2 / A3 after a second OOM: add `--batch 8` and append `_b8` to the `--out` name; say so in the message.
  - GPU-B: never change `--bs` or `--lr` (its bar is paired with s3). After a second OOM, skip B and say so.
  - GPU-C after a second OOM: the `_bs8` line in step i only.
- No commits or pushes. PowerShell at the repo root with the `.venv` active. Save each dump with
  `| Out-File -Encoding utf8 <file>`, not `>`.

a. [CPU] Pull and checks (about 5 min)
   ```powershell
   Set-Location C:\Daker\microstructure-hardness-prediction
   git status --short
   git pull origin claude/lb-under-10-7080jr
   git log -1 --format=%h
   git hash-object src\het_cnn.py tools\fsim\fsim_set.py tools\fsim\het_fsim.py tools\fsim\het_adabn.py tools\fsim\degrade_sim.py tools\fsim\fx_sim.py src\train_cnn.py
   Test-Path data\het_cnn\prep.npz, data\features_v3.parquet, data\folds.csv, data\het_cnn\ev2s_loc_e32r4\het_cnn_train.parquet, data\het_cnn\ev2s_loc_e32r4_s1\het_cnn_train.parquet, data\cnn_cache\_pre\nlm.npz, data\cnn_cache\_pre\deg2_k8_snr0.5.npz
   (Get-ChildItem data\het_cnn\ev2s_loc_e32r4\fold?.pt, data\het_cnn\ev2s_loc_e32r4_s1\fold?.pt).Count
   (Get-ChildItem data\het_cnn\ev2s_loc_e32r4\fold?.json).Count
   [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB)
   python -c "import numpy as np, pandas as pd; ids=pd.read_csv('data/train.csv', usecols=['ID']).ID; ls=[pd.read_parquet('data/het_cnn/'+d+'/het_cnn_train.parquet').set_index('ID').loc[ids, 'logN_cnn'].values for d in ('ev2s_loc_e32r4', 'ev2s_loc_e32r4_s1')]; N=np.exp(np.mean(ls, 0)); print('logN sums', [round(float(np.round(l, 4).sum()), 2) for l in ls], 'nan', int(np.isnan(N).sum()), 'N', round(float(N.min()), 1), round(float(N.max()), 1))"
   python -W ignore tools\fsim\het_fsim.py --selfcheck
   python -W ignore tools\fsim\het_fsim.py --p-fsim 0.5 --device cpu --smoke --folds 0 --epochs 1 --renders 1 --batch 8 --threads 2 --workers 2 --head fpn --mosaic 0.5 --mosaic-dlogn 0.5 --zoom 1.0 1.4 --lambda-within 1 --overwrite --out smoke_fsim
   python -W ignore tools\fsim\het_fsim.py --eval-fsim smoke_fsim --eval-smoke --device cpu
   ```
   - `git status` must not list `tools\` as untracked; the log line is 678f0d3 or a later commit.
   - Blob ids, in order (prefixes): bc683a52 14826b37 60d93eac 5f13a8ac 6c3d26b5 8366f904 fbcb1826.
   - Test-Path True x7; counts 10 and 5; the RAM line decides `--workers` (keep 4; never more than 4 below 32 GB).
   - The fingerprint must print `logN sums [2722.08, 2729.81] nan 0 N 41.3 698.2`. Otherwise skip GPU-B (steps g-i)
     and say so; never overwrite those parquets.
   - selfcheck ends `selfcheck passed: ... noise sd 15.3` (about 15 if the laptop prep differs slightly); the smoke
     ends `total wall ...` (1-2 min); the eval smoke prints 2 `EVAL` lines.
   - SEND one message: the log line, the blob ids, Test-Path / counts / RAM, the fingerprint, the selfcheck line, the
     smoke's last 4 lines, the 2 EVAL lines. A Traceback: send it and stop.

b. [GPU] A1: AdaBN, with a control first (about 10-15 min)
   ```powershell
   python -W ignore tools\fsim\het_adabn.py --run data\het_cnn\ev2s_loc_e32r4 --out data\het_cnn\ev2s_loc_e32r4_ctrlg1 --control --tta 8 --g1-only --device cuda --threads 2
   python -c "import pandas as pd, numpy as np; a=pd.read_parquet('data/het_cnn/ev2s_loc_e32r4_ctrlg1/het_cnn_train.parquet').set_index('ID').dropna(subset=['het_cnn']); b=pd.read_parquet('data/het_cnn/ev2s_loc_e32r4/het_cnn_train.parquet').set_index('ID').loc[a.index]; print('CTRL n %d max_d_het %.4f max_d_logN %.4f corr %.5f' % (len(a), np.abs(a.het_cnn-b.het_cnn).max(), np.abs(a.logN_cnn-b.logN_cnn).max(), np.corrcoef(a.het_cnn, b.het_cnn)[0,1]))"
   python -W ignore tools\fsim\het_adabn.py --run data\het_cnn\ev2s_loc_e32r4 --out data\het_cnn\ev2s_loc_e32r4_adabn --adapt-views 8 --tta 8 --with-test --device cuda --threads 2
   python -W ignore tools\fsim\het_adabn.py --run data\het_cnn\ev2s_loc_e32r4_s1 --out data\het_cnn\ev2s_loc_e32r4_s1_adabn --adapt-views 8 --tta 8 --with-test --device cuda --threads 2
   python -W ignore -m src.het_cnn --dump ev2s_loc_e32r4_adabn --chunk 0 | Out-File -Encoding utf8 data\het_cnn\ev2s_loc_e32r4_adabn\c0.txt
   python -W ignore -m src.het_cnn --dump ev2s_loc_e32r4_s1_adabn --chunk 0 | Out-File -Encoding utf8 data\het_cnn\ev2s_loc_e32r4_s1_adabn\c0.txt
   ```
   - Each het_adabn line first prints `het_adabn: device cuda ...`; it stops with an AssertionError if CUDA is not
     available (it never falls back to CPU).
   - The CTRL line must show n 208, max_d_het <= 0.005 and corr >= 0.999. Otherwise send the CTRL line, skip the
     other lines of b and go to c.
   - SEND now, one message each: the CTRL line; `ev2s_loc_e32r4_adabn\c0.txt`; `ev2s_loc_e32r4_s1_adabn\c0.txt`.
     The cloud tests A1 while c and d train.

c. [GPU] A2: retrain with 50% noise-world renders, seed 0, 5 folds (35-50 min)
   ```powershell
   python -W ignore tools\fsim\het_fsim.py --p-fsim 0.5 --device cuda --arch tf_efficientnetv2_s.in21k_ft_in1k --epochs 32 --renders 4 --batch 12 --lr 1e-3 --workers 4 --threads 2 --seed 0 --head fpn --mosaic 0.5 --mosaic-dlogn 0.5 --zoom 1.0 1.4 --lambda-within 1 --out ev2s_loc_fsim
   ```
   - First line `het_fsim: p_fsim 0.5 p_g1hi 0.77 -> ...`; last line `total wall ...`. A rerun skips finished folds.

d. [GPU] A3: route K's realistic render prior, seed 0, 5 folds (30-40 min)
   ```powershell
   python -W ignore -m src.het_cnn --device cuda --arch tf_efficientnetv2_s.in21k_ft_in1k --epochs 32 --renders 4 --batch 12 --lr 1e-3 --workers 4 --threads 2 --seed 0 --head fpn --mosaic 0.5 --mosaic-dlogn 0.5 --zoom 1.0 1.4 --lambda-within 1 --render-preset real --out ev2s_locK_real5
   ```
   - The header shows `render preset real`; last line `total wall ...`.

e. [GPU] Label-free noise-world numbers, report only (3-6 min)
   ```powershell
   python -W ignore tools\fsim\het_fsim.py --eval-fsim ev2s_loc_e32r4 ev2s_loc_fsim ev2s_locK_real5 --device cuda --threads 2
   ```
   - Six `EVAL` lines. They decide nothing.

f. [CPU] Dumps and summaries of A2 / A3 (seconds)
   ```powershell
   python -W ignore -m src.het_cnn --dump ev2s_loc_fsim --chunk 0 | Out-File -Encoding utf8 data\het_cnn\ev2s_loc_fsim\c0.txt
   python -W ignore -m src.het_cnn --dump ev2s_locK_real5 --chunk 0 | Out-File -Encoding utf8 data\het_cnn\ev2s_locK_real5\c0.txt
   python -c "import json; d=json.load(open('data/het_cnn/ev2s_loc_fsim/score.json')); p=d['stage1_pooled']; print('POOLED P %.4f W %.4f corr_logN %.4f' % (p['pcorr_het_given_logN'], p['corr_block_within'], p['corr_logN'])); [print('fold', k, 'P %.4f W %.4f R %.4f' % (v['stage1_renders']['pcorr_het_given_logN'], v['stage1_renders']['corr_block_within'], (v.get('heldout_real_clean') or {}).get('pcorr_het_given_logN', float('nan')))) for k, v in d['stage1_per_fold'].items()]"
   python -c "import json; d=json.load(open('data/het_cnn/ev2s_locK_real5/score.json')); p=d['stage1_pooled']; print('POOLED P %.4f W %.4f corr_logN %.4f' % (p['pcorr_het_given_logN'], p['corr_block_within'], p['corr_logN'])); [print('fold', k, 'P %.4f W %.4f R %.4f' % (v['stage1_renders']['pcorr_het_given_logN'], v['stage1_renders']['corr_block_within'], (v.get('heldout_real_clean') or {}).get('pcorr_het_given_logN', float('nan')))) for k, v in d['stage1_per_fold'].items()]"
   ```
   - SEND, one message each: the six EVAL lines; the two summaries; `ev2s_loc_fsim\c0.txt`;
     `ev2s_locK_real5\c0.txt`. Test chunks (1-2) only when the thread asks for them after a pass.

g. [GPU] GPU-B stage 1: 1/N-weighted y-CNN, 3 seeds x 5 folds (20-55 min)
   ```powershell
   python -W ignore -m src.train_cnn --backbone tf_efficientnetv2_s.in21k_ft_in1k --input raw+nlm --deg-p 0.5 --cons 1.0 --pool avg --epochs 30 --lr 1e-3 --wN 1 --device cuda --seeds 3 --name cnn_ev2s_rawnlm_degcons_wN1_gpu_s3
   ```
   - Early lines: `--wN: N of 500 train images from ...: sum logN 2725.94, range 41.3-698.2` and
     `degradation bank: 263 train sources x 8 copies, p=0.5`; 15 fold-seeds; the last line
     `[cnn_ev2s_rawnlm_degcons_wN1_gpu_s3] CV RMSE ...`.
   - A cut or OOM: the OOM rule above (never another `--bs` / `--lr`).

h. [CPU] GPU-B back to the cloud, one message each (score first; the 2 oof chunks are all the gate needs)
   ```powershell
   Get-Content experiments\cnn_ev2s_rawnlm_degcons_wN1_gpu_s3\score.json
   python -c "import pandas as pd; n='cnn_ev2s_rawnlm_degcons_wN1_gpu_s3'; f='oof'; k=0; d=pd.read_csv('experiments/'+n+'/'+f+'.csv'); v=d.hardness.round(2).values[250*k:250*k+250]; print('#', n, f, 'chunk', k, 'IDs', d.ID.iloc[250*k], d.ID.iloc[250*k+len(v)-1], 'n', len(v), 'sum %.2f' % v.sum()); print(' '.join('%.2f' % x for x in v))"
   ```
   - Run the chunk line for f='oof' k=0, 1, then f='test' k=0, 1, 2, 3 (7 messages with the score).

i. [GPU] Optional GPU-C, only if the GPU is still ours after h (40-105 min)
   ```powershell
   python -W ignore -m src.train_cnn --backbone tf_efficientnetv2_s.in21k_ft_in1k --input raw+nlm --deg-p 0.5 --cons 1.0 --crop 256 --pool grid4 --epochs 45 --lr 1e-3 --wN 1 --device cuda --seeds 3 --name cnn_ev2s_rawnlm_degcons_c256g4_e45_wN1_gpu_s3
   # only after a second CUDA OOM on the line above:
   python -W ignore -m src.train_cnn --backbone tf_efficientnetv2_s.in21k_ft_in1k --input raw+nlm --deg-p 0.5 --cons 1.0 --crop 256 --pool grid4 --epochs 45 --bs 8 --lr 5e-4 --wN 1 --device cuda --seeds 3 --name cnn_ev2s_rawnlm_degcons_c256g4_e45_wN1_bs8_gpu_s3
   ```
   - Send back as in h, with `n` set to the name that finished.

j. Return the GPU (exclusive mode) after the session's last [GPU] step, also when stopping early:
   ```powershell
   New-Item C:\Dacon\WM_Runtime\hardness_gpu_done
   ```

Later, only when the thread says a candidate passed (a new user line is needed then): GPU-B stage 2 (6 seeds plus a
--full run, 1-1.5 h) or A2 / A3 seed-1 runs and test chunks. Those commands go in the thread at that point.

| step | time |
|---|---|
| a (CPU) | about 5 min |
| b (GPU) | 10-15 min |
| c (GPU) | 35-50 min |
| d (GPU) | 30-40 min |
| e (GPU) | 3-6 min |
| f (CPU) | seconds |
| g (GPU) | 20-55 min |
| h (CPU) | a few minutes |
| i (GPU, optional) | 40-105 min |
| total GPU | about 2-2.8 h without i; up to about 4.5 h with i |

## 15. CPU only: send the test half of the A3 het reader (prepared 2026-10-11 13:35 KST, trimmed 14:35 KST)

No GPU. The cloud needs the test-image het of A3 (ev2s_locK_real5) to build the G1-row files of the 10-11 afternoon
plan (criterion N4, hardness-cache scripts/d1011/N4/slate/PLAN_N4.txt). The run already wrote
data\het_cnn\ev2s_locK_real5\het_cnn_test.parquet in section 14; these lines only print it (seconds, no training).
Chunk 0 (train) is dumped again so the cloud can check that the test half comes from the same fold models.

```powershell
Set-Location C:\Daker\microstructure-hardness-prediction
Test-Path data\het_cnn\ev2s_locK_real5\het_cnn_test.parquet   # True
python -W ignore -m src.het_cnn --dump ev2s_locK_real5 --chunk 0 | Out-File -Encoding utf8 data\het_cnn\ev2s_locK_real5\c0_again.txt
python -W ignore -m src.het_cnn --dump ev2s_locK_real5 --chunk 1 | Out-File -Encoding utf8 data\het_cnn\ev2s_locK_real5\c1.txt
python -W ignore -m src.het_cnn --dump ev2s_locK_real5 --chunk 2 | Out-File -Encoding utf8 data\het_cnn\ev2s_locK_real5\c2.txt
Select-String -Path data\het_cnn\ev2s_locK_real5\c0_again.txt -Pattern '# sums'
#   must print: # sums chunk 0: het_cnn 85.3587 logN_cnn 2747.2417 bmean_cnn 2776.8410 nan 0   (else send that line and stop)
```
- If Test-Path prints False, send that line and stop (the run's test half is missing).
- SEND, one message each: the '# sums' line of c0_again.txt; then the full text of c1.txt; then the full text of c2.txt.
  Each chunk starts with `# het_cnn ev2s_locK_real5 chunk <k> (500 rows)` and ends with a `# sums chunk <k>:` line.

## 16. Next GPU run: A3 reader on 6-map (V4) block targets, then A3 seed 2 (prepared 2026-10-11 18:00 KST)

Runs only on JINWOO's own word in the main thread ("섹션 16 실행"); mode as JINWOO says, else the same hand-over as the
17:22 KST seed-1 run; return the GPU with `New-Item C:\Dacon\WM_Runtime\hardness_gpu_done` when done or when stopping.
Purpose: two more readers for the G1 rows of the 10-11 slots 4-5 (cloud: hardness-cache scripts/d1011/N4/r2seed, r3).
- 16-1 = A3's recipe (route K, `--render-preset real`) trained on block targets averaged over the 6 segmentation maps
  that blend_v29g uses on the clean rows (V4), via the A4 `--target-file` option (bold/A4, verified 10-11 09:00 KST;
  code committed with this section). Renders are unchanged; only the clean images' block targets change.
- 16-2 = A3 with `--seed 2` (third seed, for a 3-seed reader mean).

```powershell
Set-Location C:\Daker\microstructure-hardness-prediction
git pull origin claude/lb-under-10-7080jr
python -W ignore -m src.het_cnn --selftest
#   must print "ok 14: --target-file ..." and end with "selftest passed" (else send the output and stop)
python -W ignore tools\a4\v4_block_targets.py --jobs 4
#   must print: reference map (1.0, 0.6) vs prep b: max |diff| 0.00e+00, NaN pattern equal True  (else it writes
#   nothing: send the output and stop). Send its checksum line (cloud: sum 23191.471116 sumsq 127600.524605).
python -W ignore -m src.het_cnn --device cpu --smoke --folds 0 --epochs 1 --renders 1 --batch 8 --threads 2 --workers 2 --head fpn --mosaic 0.5 --mosaic-dlogn 0.5 --zoom 1.0 1.4 --lambda-within 1 --render-preset real --target-file data\het_cnn\a4_v4_block_targets.npz --out smoke_s16
#   first line "--target-file ... block targets of the 267 clean images replaced; correction b_new - b_prep mean
#   -0.1349 sd 0.1586 ..."; last line "total wall". Any error: send it and stop.

# 16-1 [GPU] (about 25 min)
python -W ignore -m src.het_cnn --device cuda --arch tf_efficientnetv2_s.in21k_ft_in1k --epochs 32 --renders 4 --batch 12 --lr 1e-3 --workers 4 --threads 2 --seed 0 --head fpn --mosaic 0.5 --mosaic-dlogn 0.5 --zoom 1.0 1.4 --lambda-within 1 --render-preset real --target-file data\het_cnn\a4_v4_block_targets.npz --out ev2s_locK_real5_v4t
foreach ($k in 0,1,2) { python -W ignore -m src.het_cnn --dump ev2s_locK_real5_v4t --chunk $k | Out-File -Encoding utf8 data\het_cnn\ev2s_locK_real5_v4t\c$k.txt }

# 16-2 [GPU] (about 25 min)
python -W ignore -m src.het_cnn --device cuda --arch tf_efficientnetv2_s.in21k_ft_in1k --epochs 32 --renders 4 --batch 12 --lr 1e-3 --workers 4 --threads 2 --seed 2 --head fpn --mosaic 0.5 --mosaic-dlogn 0.5 --zoom 1.0 1.4 --lambda-within 1 --render-preset real --out ev2s_locK_real5_s2
foreach ($k in 0,1,2) { python -W ignore -m src.het_cnn --dump ev2s_locK_real5_s2 --chunk $k | Out-File -Encoding utf8 data\het_cnn\ev2s_locK_real5_s2\c$k.txt }
New-Item C:\Dacon\WM_Runtime\hardness_gpu_done
```
- CUDA OOM: rerun the identical command once; after a second OOM use `--batch 8` with `_b8` appended to `--out` and say so.
- SEND after each run, one message per chunk (c0, c1, c2), full text: each starts with `# het_cnn <run> chunk <k>` and
  ends with its `# sums chunk <k>:` line. Send 16-1's chunks before starting 16-2 if the GPU is busy with 16-2.
- The cloud decides everything from train labels only (pre-declared in N4/r3/options.txt); the laptop sends no numbers
  of its own beyond the sums and the stage-1 summary line of each score.json.

## 17. Next GPU run: N6-G1, the y-CNN started from the A3 het reader (prepared 2026-10-11 21:30 KST)

Runs on JINWOO's own run line in the main thread (10-11 21:33 KST: "섹션17 올라오면 바로 실행", which covers 17-0 to 17-3).
The laptop session cannot see the cloud's thread replies, so every cloud verdict comes through git instead: the cloud
commits one line per verdict to `notes/handoff/gate17_verdicts.txt` and pushes. After sending a step's messages, poll
`git pull origin claude/lb-under-10-7080jr` every 2 minutes until the line that step waits for is there. Each later
step (17-2, 17-3, and T0 after a cloud (c) FAIL) runs only on its verdict line. Verdict lines (exact text):
  `17-1 c T1: PASS` / `17-1 c T1: FAIL` / `17-1 c T0: PASS` / `17-1 c T0: FAIL`
  `17-1: PASS T1` / `17-1: PASS T0` / `17-1: CLOSED`
  `17-2 stage-1: PASS` / `17-2 stage-1: CLOSED`
  `17-3: received` (end of section)
GPU rule (header of this file): this project owns the GPU until 2026-10-19 10:00 KST. Run with `--device cuda`
directly: no gpu_turn.py wrapper, no hand-back file. Never touch the robot project's files or jobs.

Why (hardness-cache scripts/d1011/N6/PLAN_N6.txt sections 1-2, pick N6-G1):
- Every y-CNN so far starts from ImageNet weights. The A3 het reader (ev2s_locK_real5, section 14d) was trained
  without labels on 'real'-preset renders and already reads logN and het. Here the y-CNN starts from A3's fold models
  (trunk, FPN, block / cell / logN heads) and adds a y head on [GAP stride-32, mean b-hat, sd b-hat, logN-hat].
- `--anchor 0.5` keeps the block and logN outputs close to a frozen copy of the same A3 fold model on every training
  crop (label-free). The y labels, folds and augmentation are the incumbent s3 recipe's (raw+nlm, deg 0.5, cons 1.0,
  30 epochs, bs 16, OneCycle, TTA 8). The other changes are `--lr 3e-4` (the incumbent used 1e-3; the trunk starts
  from a trained reader, PLAN_N6) and `--crop 256`, forced by the 4x4 block head.
- NLM stays as a second input channel: the A3 stem gets a zero-initialised nlm channel (`--nlm-zero-stem`), so with 0
  training steps the model reads exactly what A3 reads.
- Fold model f / seed s starts from `data\het_cnn\<RUN[s mod 3]>\fold<f>.pt` (A3 seeds 0, 1, 2). A `--full` model s
  starts from fold s mod 5 of that run.

Code (committed with this section; all new options default off; without them every path is bit-identical, which
`--selftest-het` check 5 tests against the file before the options):
- `src\train_cnn.py`: `--init-het RUN [RUN ...]`, `--head hetfpn`, `--anchor LAMBDA`, `--nlm-zero-stem`, and
  `--selftest-het [RUN]`. Fold-seed caches of a hetfpn run also hold val_b / val_logn / test_b / test_logn.
- `src\het_cnn.py --eval-only RUN --dump-blocks`: inference only. Per image it keeps b00..b15 plus the 16x16 cell map's
  c_ring0 (outer 16-px ring), c_inner and c_sd. Clean train rows come from their own fold model, other rows from the
  5-fold mean. The result is printed by `--dump RUN --dump-blocks --chunk 0 1` (250 rows per chunk). Selftest check 15.
- `tools\n6\gate17.py`: the label-free gate lines. It reads no label (only the ID column of train.csv).

Gates, fixed before any run:
- (a) G1 train rows held out in the gate folds: corr(sd b-hat, het_A3) >= 0.85 and corr(logN-hat, logN_A3) >= 0.97.
  het_A3 / logN_A3 come from `data\het_cnn\ev2s_locK_real5\het_cnn_train.parquet`. Computed on the laptop.
- (b) partial corr(y-hat, het_A3 | logN_A3) on G1 train, folds 0-1. Report only. s3 reference -0.2799 (n 71).
- (c) held-out real clean train, folds 0-1: partial corr(y-hat, V4 het | N_eff^0.25) >= s3 + 0.10. s3 reference
  -0.0561 (n 120), so the bar is >= +0.0439. Computed on the cloud, because V4 exists only there. The laptop sends
  the y-hat block.
- (d) KS of y-hat, G1 test vs G1 train held-out, p > 0.01. Stop only. Computed on the laptop.
- The (b)/(c) references come from the s3 OOF, computed 10-11 before any run. File:
  hardness-cache `scripts/d1011/N6/run/gate17/ref_s3.json`, sha256
  9eeded73c3a7a9696d0aefe92b794352e5e4ac7829e2b7ec3fb69516f0870471. The cloud checks a T1 / T0 message with
  `gate17_bc.py` in that folder.

Rules:
- No commits or pushes. PowerShell at the repo root with the `.venv` active. Save every dump with
  `| Out-File -Encoding utf8 <file>`.
- Never send a line that holds a validation number. That is any training-log line with `val` or `rmse` in it:
  `val(tta1)`, `final val`, `per-view val rmse`, `fold f: rmse`, `OOF rmse ...` (tercile / fine_noisy cell),
  `partial run ... rmse on done folds`, and the `CV RMSE` line. Label-based numbers first reach the cloud in 17-2
  (score.json and the chunk lines), and only after its gate17_2 PASS.
- CUDA out of memory: rerun the IDENTICAL command once; finished fold-seeds are cached and resume. After a second OOM,
  stop and send the traceback. Never change `--bs`, `--lr` or `--crop`: the stage-1 bars are paired with s3.
- A cut, or a closed window: rerun the identical command. Fold-seeds are cached in `data\cnn_cache\<name>\`.

17-0 [CPU] Pull and selftests (about 10 min)
```powershell
Set-Location C:\Daker\microstructure-hardness-prediction
git status --short
git pull origin claude/lb-under-10-7080jr
git log -1 --format=%h
(Get-ChildItem data\het_cnn\ev2s_locK_real5\fold?.pt, data\het_cnn\ev2s_locK_real5_s1\fold?.pt, data\het_cnn\ev2s_locK_real5_s2\fold?.pt).Count
Test-Path data\het_cnn\ev2s_locK_real5\het_cnn_train.parquet, data\het_cnn\ev2s_locK_real5\fold0.npz, data\cnn_cache\_pre\nlm.npz, data\cnn_cache\_pre\deg2_k8_snr0.5.npz, data\features_v3.parquet
python -W ignore -m src.train_cnn --selftest-het --device cpu
python -W ignore -m src.train_cnn --selftest-het ev2s_locK_real5 --device cpu
python -W ignore -m src.het_cnn --selftest
```
- If the pull refuses because of a local ` M experiments/LEADERBOARD.md` (lines appended by earlier GPU runs) or
  untracked `experiments\cnn_*_gpu*` files, back them up and re-sync exactly as in section 8 a. If any other tracked
  file shows as modified, stop and ask. The `git log -1` line is sent with the selftest message (the cloud checks it
  against the section 17 commit).
- The count must be 15 and Test-Path True x5. Otherwise send those lines and stop.
- `--selftest-het` prints ok 1 .. ok 6 and ends `selftest-het passed (synthetic checkpoint, ...)`.
- The second line is the real-checkpoint check. ok 1-3 show the max |d| of b-hat / logN-hat vs
  `het_cnn.predict` of `ev2s_locK_real5\fold0.pt` (must be <= 1e-5; the code asserts it). It ends
  `selftest-het passed (ev2s_locK_real5, folds [0], device cpu, ...)`. Check 5 compares the old recipe with the
  file before these options (it needs the git history). It takes a few minutes on the CPU.
- The het_cnn selftest must print `ok 15: --dump-blocks ...` and end with `selftest passed`.
- SEND one message: the log line, the count / Test-Path lines, and every `ok` line plus the last line of the three
  selftests. A Traceback: send it and stop.

17-1 [GPU] T1 gate run (folds 0-1, 1 seed) and the A3 block re-dump (about 15-20 min)
```powershell
python -W ignore -m src.train_cnn --backbone tf_efficientnetv2_s.in21k_ft_in1k --init-het ev2s_locK_real5 ev2s_locK_real5_s1 ev2s_locK_real5_s2 --head hetfpn --anchor 0.5 --lr 3e-4 --input raw+nlm --nlm-zero-stem --crop 256 --deg-p 0.5 --cons 1.0 --pool avg --epochs 30 --bs 16 --tta 8 --device cuda --folds 0 1 --seeds 1 --no-save --name cnn_ev2s_A3init_anc05_s3
python -W ignore tools\n6\gate17.py --name cnn_ev2s_A3init_anc05_s3 --folds 0 1 --yhat | Out-File -Encoding utf8 data\cnn_cache\cnn_ev2s_A3init_anc05_s3\gate17_1.txt
python -W ignore -m src.het_cnn --eval-only ev2s_locK_real5 --dump-blocks --device cuda --threads 2 | Out-File -Encoding utf8 data\het_cnn\ev2s_locK_real5\blocks_log.txt
foreach ($k in 0,1) { python -W ignore -m src.het_cnn --dump ev2s_locK_real5 --dump-blocks --chunk $k | Out-File -Encoding utf8 data\het_cnn\ev2s_locK_real5\blocks_c$k.txt }
```
- First training lines: `degradation bank: 263 train sources x 8 copies, p=0.5`, then per fold
  `--init-het: fold 0 seed 0 <- ev2s_locK_real5/fold0.pt (sha256 ..., tf_efficientnetv2_s.in21k_ft_in1k, fpn_ch 128);
  raw channel mu ... (checkpoint consts); nlm stem channel zero-initialised; anchor 0.5 to the frozen checkpoint`.
  Every epoch line shows `anchor <value>` after the lr. The last line is `total wall time ...`. Because of --no-save, nothing is
  written to experiments\.
- gate17_1.txt holds the (a) and (d) lines (PASS / FAIL), a summary line, and a `# yhat` block with 200 values.
  It is about 2 KB.
- blocks_log.txt: 5 lines `fold f: vs its fold<f>.npz on 500 images: max |d b| ...` (bf16 autocast, so up to about
  1e-2 is fine), then `wrote ...blocks_train.parquet: 500 images, sources {...}; sums ...; vs het_cnn_train.parquet
  ...`. blocks_c0.txt / blocks_c1.txt start with `# het_cnn blocks ev2s_locK_real5 chunk <k> (250 rows)` and end with
  their `# sums chunk <k>:` line. Each is about 37 KB, under the 64 KiB limit.
- SEND, one message each:
  1. gate17_1.txt in full, plus the `total wall time` line of the training.
  2. The text of blocks_log.txt, then the full text of blocks_c0.txt.
  3. The full text of blocks_c1.txt.
  The cloud answers with the (b) and (c) lines from `gate17_bc.py`. It rebuilds the block dump with
  `python -m src.het_cnn --ingest ev2s_locK_real5 blocks_c0.txt blocks_c1.txt --dump-blocks`, which checks the IDs
  and sums (message 2 can be saved as one file: text before the chunk header and after its sums line is skipped).
- The block dump (messages 2 and 3) does not depend on the gate: send it in every case (N6-G2 waits for it).
- Stop rules:
  - (d) FAIL closes the line: send messages 1-3 and stop. No T0, no 17-2.
  - (a) FAIL, or the cloud's (c) FAIL (verdict line `17-1 c T1: FAIL`): run T0 once (below). Send gate17_1_T0.txt
    plus its `total wall time` line as in message 1, then wait for `17-1 c T0: ...` and `17-1: ...`. Do not wait for
    the cloud before T0 when (a) failed. The block dump is not repeated.
    ```powershell
    python -W ignore -m src.train_cnn --backbone tf_efficientnetv2_s.in21k_ft_in1k --init-het ev2s_locK_real5 ev2s_locK_real5_s1 ev2s_locK_real5_s2 --head hetfpn --anchor 0 --lr 3e-4 --input raw+nlm --nlm-zero-stem --crop 256 --deg-p 0.5 --cons 1.0 --pool avg --epochs 30 --bs 16 --tta 8 --device cuda --folds 0 1 --seeds 1 --no-save --name cnn_ev2s_A3init_anc0_s3
    python -W ignore tools\n6\gate17.py --name cnn_ev2s_A3init_anc0_s3 --folds 0 1 --yhat | Out-File -Encoding utf8 data\cnn_cache\cnn_ev2s_A3init_anc0_s3\gate17_1_T0.txt
    ```
  - A T0 miss on (a), (c) or (d) closes the line.
  - Arm M (ImageNet init plus aux heads) never runs.
- 17-1 PASS = (a) PASS and (d) PASS here, and (c) PASS from the cloud, for T1, or else for T0. The cloud writes the
  verdict line `17-1: PASS T1`, `17-1: PASS T0` or `17-1: CLOSED`.
- Amendment 2026-10-15 11:31 KST (cloud, before the T1 (c) number was computed): (c) is report only. 17-1 PASS = (a) PASS
  and (d) PASS for T1. If the cloud writes `17-1 c T1: FAIL` and then `17-1: PASS T1`, go to 17-2 with the T1 arm (a T0
  run already started may finish; its result is report only).

17-2 [GPU] Stage 1: 5 folds x 3 seeds of the arm that passed 17-1 (about 30-45 min). Run only on the verdict line
`17-1: PASS T1` or `17-1: PASS T0`. The lines below are the T1 arm. If T0 passed instead, replace `--anchor 0.5` with `--anchor 0` and
`anc05` with `anc0` in both lines (name `cnn_ev2s_A3init_anc0_s3`). The same name reuses the cached fold 0-1 seed-0
models of the gate run.
```powershell
python -W ignore -m src.train_cnn --backbone tf_efficientnetv2_s.in21k_ft_in1k --init-het ev2s_locK_real5 ev2s_locK_real5_s1 ev2s_locK_real5_s2 --head hetfpn --anchor 0.5 --lr 3e-4 --input raw+nlm --nlm-zero-stem --crop 256 --deg-p 0.5 --cons 1.0 --pool avg --epochs 30 --bs 16 --tta 8 --device cuda --folds 0 1 2 3 4 --seeds 3 --name cnn_ev2s_A3init_anc05_s3
python -W ignore tools\n6\gate17.py --name cnn_ev2s_A3init_anc05_s3 --folds 0 1 2 3 4 | Out-File -Encoding utf8 data\cnn_cache\cnn_ev2s_A3init_anc05_s3\gate17_2.txt
```
- 13 new fold-seeds; 2 are `cached`. It writes experiments\<name>\ (oof.csv, test.csv, score.json) and a
  LEADERBOARD.md line. Keep those local and uncommitted.
- gate17_2.txt holds (a) and (d) on the full OOF (208 G1 train rows), about 0.6 KB.
- SEND gate17_2.txt first, as its own message.
  - (a) or (d) FAIL: stop there and send nothing else. No label number ever leaves the laptop for this arm.
  - Both PASS: send score.json, then the GPU-B chunk lines (section 14 h) with `n` = the arm's name, for f='oof'
    k=0, 1 and f='test' k=0, 1, 2, 3. That is 7 messages of about 1-2 KB each.
- The cloud then computes the stage-1 bars with hardness-cache `scripts/d1011/bold/L6_leaders/l6_cnn_gate.py <oof>
  --ref s3`: all <= 13.12, >= 4/5 folds better than s3, fine-noisy cell <= 11.20. Any bar missed closes the line:
  no 17-3 and no N4 look.
- Amendment 2026-10-15 11:35 KST (cloud, before any 17-2 number): the cloud also writes `17-2 stage-1: PASS` on the
  diversity route (all-500 OOF RMSE <= 13.30 and OOF-residual corr with s3 <= 0.97). Nothing changes on the laptop side.

17-3 [GPU] Stage 3: 6 seeds plus 6 full-data models of the same arm (about 1.5 h). Run only on the verdict line
`17-2 stage-1: PASS`. Replace `anc05` with `anc0` and `--anchor 0.5` with `--anchor 0` if the arm is T0.
```powershell
python -W ignore -m src.train_cnn --backbone tf_efficientnetv2_s.in21k_ft_in1k --init-het ev2s_locK_real5 ev2s_locK_real5_s1 ev2s_locK_real5_s2 --head hetfpn --anchor 0.5 --lr 3e-4 --input raw+nlm --nlm-zero-stem --crop 256 --deg-p 0.5 --cons 1.0 --pool avg --epochs 30 --bs 16 --tta 8 --device cuda --seeds 6 --name cnn_ev2s_A3init_anc05_gpu_s6
python -W ignore -m src.train_cnn --backbone tf_efficientnetv2_s.in21k_ft_in1k --init-het ev2s_locK_real5 ev2s_locK_real5_s1 ev2s_locK_real5_s2 --head hetfpn --anchor 0.5 --lr 3e-4 --input raw+nlm --nlm-zero-stem --crop 256 --deg-p 0.5 --cons 1.0 --pool avg --epochs 30 --bs 16 --tta 8 --device cuda --full --seeds 6 --name cnn_ev2s_A3init_anc05_gpu_full6
python -W ignore tools\n6\gate17.py --stage3 cnn_ev2s_A3init_anc05_gpu_s6 cnn_ev2s_A3init_anc05_s3 cnn_ev2s_A3init_anc05_gpu_full6 | Out-File -Encoding utf8 data\cnn_cache\cnn_ev2s_A3init_anc05_gpu_s6\gate17_3.txt
python -W ignore tools\n6\gate17.py --stage3 cnn_ev2s_A3init_anc05_gpu_s6 cnn_ev2s_A3init_anc05_s3 cnn_ev2s_A3init_anc05_gpu_full6 --kf-chunk 0 1 2 3 | Out-File -Encoding utf8 data\cnn_cache\cnn_ev2s_A3init_anc05_gpu_s6\kf_chunks.txt
```
- The fold run trains 30 fold-seeds (its own name: nothing is shared with the s3 cache). The `--full` run trains 6
  models on all 500 images. Full model s starts from fold s mod 5 of run s mod 3. It writes test.csv + full.json only.
- gate17_3.txt has three checks:
  - corr(6-seed OOF, 3-seed OOF) >= 0.98.
  - rms(full6 test - 5-fold-mean test) <= 1.5. The incumbent s6 vs full6 reads 1.298; the cloud checked the tool
    against it.
  - KS of the kf test predictions on the G1 test rows vs the G1 train OOF, p > 0.01. Stop only.
  A sums line follows. kf = 0.5 x (5-fold mean + full6), the s6kf rule. kf_chunks.txt holds 4 chunks of 250 values,
  each with a `# <name>kf test chunk k IDs ... sum ...` header, about 1.8 KB each.
- SEND gate17_3.txt first.
  - Any FAIL: stop, send nothing else. There is no N4 look.
  - All PASS: send score.json of `..._gpu_s6`, then the GPU-B chunk lines of `..._gpu_s6` for f='oof' k=0, 1,
    then the 4 chunks of kf_chunks.txt, one per message. That is 7 messages.
- The cloud builds the kf member and runs P2 (the N4 look) on the cloud.

PLAN_N6's 17-4 (N6-G3 DANN reader screen) is not part of this section: its code does not exist yet, and it will come
as a section of its own. When the G1 line closes (at any step), stop and wait for the thread.

| step | time | send back |
|---|---|---|
| 17-0 (CPU) | about 10 min | 1 message: log line, checks, selftest lines (< 3 KB) |
| 17-1 (GPU) | 15-20 min (+ about 6 min if T0 runs) | gate17_1.txt (~2 KB); blocks_log + blocks_c0 (~38 KB); blocks_c1 (~37 KB); a T0 gate file if T0 runs |
| 17-2 (GPU) | 30-45 min | gate17_2.txt (< 1 KB); if PASS: score.json + oof k=0,1 + test k=0..3 (7 x 1-2 KB) |
| 17-3 (GPU) | about 1.5 h | gate17_3.txt (< 1 KB); if PASS: score.json + oof k=0,1 + kf chunks 0..3 (7 x 1-2 KB) |
| total GPU | about 2.3 h (2.5 h with T0) | |

## 18. More G1 readers: send B4 and A3 seed 4, then train B1 seeds 1-2 (prepared 2026-10-15 11:36 KST)

Runs on JINWOO's own run line in the main thread ("섹션18 실행"). GPU rule as in the header (this project owns the GPU until
2026-10-19 10:00 KST; `--device cuda` directly; never touch the robot project's files or jobs).
Why: the 10-15 files (hardness-cache scripts/d1015/g1mix) average het readers on the 433 G1 test rows: the A3 seeds
(ev2s_locK_real5, _s1, _s2, _s3) and B1 (ev2s_locK_fsim). More readers give more such pieces: B4 (A3 recipe on the
effnetv2-m backbone, ev2m_locK_real5) and A3 seed 4 already exist on the laptop only; B1 has one seed.

18-0 [CPU, about 1 min; may run while section 17 uses the GPU]
```powershell
Set-Location C:\Daker\microstructure-hardness-prediction
foreach ($r in 'ev2m_locK_real5','ev2s_locK_real5_s4') {
  if (Test-Path data\het_cnn\$r\het_cnn_test.parquet) {
    foreach ($k in 0,1,2) { python -W ignore -m src.het_cnn --dump $r --chunk $k | Out-File -Encoding utf8 data\het_cnn\$r\c$k.txt }
  } else { "missing data\het_cnn\$r\het_cnn_test.parquet" }
}
```
- If a run folder has another name (for example the B4 run was saved under a different `--out`), use that name and say so.
  If a test parquet is missing, send the `missing` line and the folder listing (`Get-ChildItem data\het_cnn -Directory`).
- SEND, one message per chunk, full text (c0, c1, c2 of each run; 6 messages). Each starts with
  `# het_cnn <run> chunk <k>` and ends with its `# sums chunk <k>:` line.

18-1 [GPU, about 25 min each; only after section 17 has finished or closed]
- B1 seeds 1 and 2: take the exact training command of B1 from `data\het_cnn\ev2s_locK_fsim\score.json` (keys `command`
  / `commands` / `args`) and run it twice, changing only `--seed 1 --out ev2s_locK_fsim_s1`, then
  `--seed 2 --out ev2s_locK_fsim_s2`. Then dump each:
```powershell
foreach ($r in 'ev2s_locK_fsim_s1','ev2s_locK_fsim_s2') { foreach ($k in 0,1,2) { python -W ignore -m src.het_cnn --dump $r --chunk $k | Out-File -Encoding utf8 data\het_cnn\$r\c$k.txt } }
```
- Before the first run, send the B1 command line you are going to use (one message), so the cloud can check it.
- CUDA OOM: rerun the identical command once; after a second OOM use `--batch 8` with `_b8` appended to `--out` and say so.
- SEND after each run: its 3 chunks, one message each (full text).
- The laptop sends no label numbers; the cloud scores every reader on train labels only.
