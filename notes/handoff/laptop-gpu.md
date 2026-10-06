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

## 3. GPU (only after the user's approval is written in the thread)
Wrap every GPU command:
`python C:\Dacon\RobotWorldModel_ActionVideo\wm_ops\gpu_turn.py --who hardness -- <command>`
Add `--device cuda`. The paused robot job may hold about 4 GB of VRAM, so on CUDA out of memory lower `--bs`.
Never touch the robot project in any other way.

Base flags (best CPU config, `notes/handoff/cnn-trainer.md`):
`--input raw+nlm --deg-p 0.5 --cons 1.0 --pool avg --epochs 30`

a. Wrapped CUDA check.
b. lr screen, folds 0 and 1, 1 seed, `--no-save`:
   - `--backbone tf_efficientnetv2_s.in21k_ft_in1k` (Apache-2.0) at `--lr 1e-3` and `--lr 3e-4`
   - `--backbone resnet18 --lr 1e-3` as the reference (CPU fold scores are in the cnn-trainer handoff)
c. Best setting: `--seeds 3` on all 5 folds, saved with `--name cnn_<backbone>_rawnlm_degcons_gpu`.
   One wrapped command per fold if `train_cnn` merges per-fold outputs, otherwise one wrapped command for the run.

## 4. Return the results without pushing
Send the cloud orchestrator session (the one you already message) the contents of that experiment's `oof.csv`,
`test.csv` and `score.json`, one file per message, plus the step 3b screen results.
The cloud session blends them for the next submission.
