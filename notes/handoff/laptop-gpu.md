# Laptop GPU run (orchestrator's plan for the Remote Control session on DESKTOP-C78VQOR)

Folder: `C:\Daker\microstructure-hardness-prediction`, branch `claude/lb-under-10-7080jr`.
The repo is public (user's decision, 2026-10-06). Never commit or push test-set predictions (`experiments/*/test.csv`,
git-ignored) or figures that show competition images. This session does not commit or push at all.

## 1. Pull
`git pull origin claude/lb-under-10-7080jr`

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
c. Best setting: `--seeds 3` on all 5 folds, saved with `--name cnn_<backbone>_rawnlm_degcons_gpu`.
   One command for the run, or one per fold if `train_cnn` merges per-fold outputs.

## 4. Return the results without pushing
Send the cloud orchestrator session (the one you already message) the contents of that experiment's `oof.csv`,
`test.csv` and `score.json`, one file per message, plus the step 3b screen results.
The cloud session blends them for the next submission.
