# Microstructure Hardness Prediction (DACON)

> **Status (2026-10-17 23:45 KST):** best public blend_v38q8a 9.9496 (10-17 extra #2, Q8A = G1 readers B8f + APf; automatic final while it stays best); v38xn8 9.9565, v37n2 9.9595, v37x78 9.9599, v38q8 9.9649, v35xn 9.9722, v38q86 9.9810; target public in the 8s (user, 10-15 12:56 KST); 10-17 closed (4 files); 5 slots on 10-18 and 5 on 10-19 until the 10:00 KST deadline, 2-bold cap per day (d1017/NOTE_slate1018.txt); public model pub/kriging24.py, env d1017/run_env_1018.sh; laptop 24-12 (B4-NS seed 2) then sections 25 and 26 overnight (done about 09:15 KST 10-18); design round N7 found no lever for the 8s (P(<9.0) about 0.2%). Current status, restore steps and next actions: [notes/RESUME.md](notes/RESUME.md).
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
