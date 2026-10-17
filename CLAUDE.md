# Microstructure Hardness Prediction (DACON)

> **Status (2026-10-17 21:40 KST):** best public blend_v37n2 9.9595 (10-16, g1mix6 N2 = the two Noisy Student G1 readers NSf + B8f; automatic final while it stays best); blend_v37x78 9.9599, v38q8 (B8f alone, 10-17 bold #1) 9.9649, v35xn 9.9722, v36l7 9.9779; target public in the 8s (user, 10-15 12:56 KST); blend_v38xn8 (XN8 = B1f + B4f + NSf + B8f, kriging21 9.9632, P(beat N2) 0.29) handed over 21:39 KST as 10-17 bold #2, reading pending; after it 8 slots left (3 SAFE-only on 10-17, 5 on 10-18); public model pub/kriging21.py; laptop section 24 on 24-4..6 (B5-AP, A3 recipe, done about 22:45 KST), then 24-10..12 (B4-NS), then sections 25 and 26 (user run lines 20:37-20:38 KST; all done about 09:15 KST 10-18); design round N7 found no lever for the 8s (P(<9.0) about 0.2%). Current status, restore steps and next actions: [notes/RESUME.md](notes/RESUME.md).
> Per-agent details: `notes/handoff/*.md`. Cached intermediates: `/mnt/project-files/work/hardness-cache/`.

256×256 8-bit grayscale synthetic microstructure image → `hardness` (HV proxy). Metric: RMSE.
Goal: **public LB in the 8s (below 9.0)** (user, 2026-10-15 12:56 KST, after blend_v33xb read 9.9868; earlier goals: ≤ 10, then ≤ 9 on 2026-10-10). Final = the submission with the best
public score, picked automatically at the deadline **2026-10-19 10:00 KST** (user, 2026-10-10). Train 500 images, test 1,000 images. Rules: no external data, test images
must never be used for training (no pseudo-labels, no test-time fitting of scalers), only openly
licensed pretrained weights (MIT/Apache/CC BY/CC BY-NC), no remote APIs, max 5 submissions/day.

## Data layout
Unzip `open.zip` into `data/` (git-ignored) or point `DATA_DIR` at it:
```
data/train/TRAIN_xxxxxx.png  data/test/TEST_xxxxxx.png  data/train.csv  data/sample_submission.csv
```

## Environment notes
- Cloud sandbox: 4 CPU, no GPU; github.com and PyPI reachable, huggingface.co / dl.fbaipublicfiles.com
  / download.pytorch.org blocked. Use `src.common.create_timm` for pretrained weights (GitHub releases).
- Remote-API models (e.g. hosted LLM/decision APIs) are banned by competition rule 2.

## Experiment protocol (every agent follows this)
- Folds: `python -m src.folds` writes `data/folds.csv` (5-fold, target-binned stratified, seed 42).
  Every model uses these same folds so OOF predictions are comparable and stackable.
- Each experiment writes to `experiments/<exp_name>/`:
  - `oof.csv` (ID, hardness) — out-of-fold prediction for all 500 train images
  - `test.csv` (ID, hardness) — mean of fold models on test (git-ignored: the repo is public, so test predictions stay local and are backed up to `/mnt/project-files/work/hardness-cache/experiments-test.tar`)
  - `score.json` — `{"cv_rmse": ..., "fold_rmse": [...], "notes": "..."}`
- Append one line per experiment to `experiments/LEADERBOARD.md` (CV RMSE, public LB if submitted).
- Final blend: `python -m src.ensemble` fits non-negative weights on OOF only, writes `submissions/`.

## Multi-agent layout
Agents in `.claude/agents/` own one model family each and run in parallel; the orchestrator
(main session) assigns experiments, reads `score.json`s, and runs the ensembler.
| agent | owns | entry point |
|---|---|---|
| `eda-analyst` | target/feature analysis, leakage & distribution checks | notebooks / `src/features.py` ideas |
| `feature-engineer` | handcrafted microstructure features + GBM | `src/features.py`, `src/train_gbm.py` |
| `embedding-modeler` | frozen pretrained backbones → ridge/SVR/GBM heads | `src/extract_embeddings.py`, `src/train_head.py` |
| `cnn-trainer` | end-to-end CNN fine-tuning with augmentation + TTA | `src/train_cnn.py` |
| `ensembler` | OOF-based blending, submission files | `src/ensemble.py` |

Agents must not edit another agent's files; shared code changes (`src/common.py`, `src/folds.py`)
go through the orchestrator.
