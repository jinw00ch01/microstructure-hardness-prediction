import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
EXP_DIR = ROOT / "experiments"
SUB_DIR = ROOT / "submissions"
SEED = 42


def load_train():
    df = pd.read_csv(DATA_DIR / "train.csv")
    folds = pd.read_csv(DATA_DIR / "folds.csv")
    return df.merge(folds, on="ID")


def load_test():
    return pd.read_csv(DATA_DIR / "sample_submission.csv")[["ID"]]


def img_path(i):
    return DATA_DIR / ("train" if i.startswith("TRAIN") else "test") / f"{i}.png"


def read_img(i):
    return np.asarray(Image.open(img_path(i)).convert("L"), dtype=np.float32) / 255.0


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def save_experiment(name, train_df, oof, test_df, test_pred, notes=""):
    out = EXP_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": train_df.ID, "hardness": oof}).to_csv(out / "oof.csv", index=False)
    pd.DataFrame({"ID": test_df.ID, "hardness": test_pred}).to_csv(out / "test.csv", index=False)
    fold_rmse = [rmse(oof[train_df.fold == f], train_df.hardness[train_df.fold == f])
                 for f in sorted(train_df.fold.unique())]
    score = {"cv_rmse": rmse(oof, train_df.hardness), "fold_rmse": fold_rmse, "notes": notes}
    (out / "score.json").write_text(json.dumps(score, indent=2))
    with open(EXP_DIR / "LEADERBOARD.md", "a") as f:
        f.write(f"| {name} | {score['cv_rmse']:.3f} |  | {notes} |\n")
    print(f"[{name}] CV RMSE {score['cv_rmse']:.4f} folds {np.round(fold_rmse, 3).tolist()}")
    return score
