# Microstructure Hardness Prediction (DACON)

256×256 8-bit grayscale synthetic microstructure image → `hardness` (HV proxy). Metric: RMSE.
Goal: **public LB ≤ 10**. Train 500 images, test 1,000 images. Rules: no external data, test images
must never be used for training (no pseudo-labels, no test-time fitting of scalers), only openly
licensed pretrained weights (MIT/Apache/CC BY/CC BY-NC), no remote APIs, max 5 submissions/day.

## Data layout
Unzip `open.zip` into `data/` (git-ignored) or point `DATA_DIR` at it:
```
data/train/TRAIN_xxxxxx.png  data/test/TEST_xxxxxx.png  data/train.csv  data/sample_submission.csv
```

## Experiment protocol (every agent follows this)
- Folds: `python -m src.folds` writes `data/folds.csv` (5-fold, target-binned stratified, seed 42).
  Every model uses these same folds so OOF predictions are comparable and stackable.
- Each experiment writes to `experiments/<exp_name>/`:
  - `oof.csv` (ID, hardness) — out-of-fold prediction for all 500 train images
  - `test.csv` (ID, hardness) — mean of fold models on test
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
