"""Shared helpers for the eda-analyst scripts (train labels only; test images only for unsupervised stats)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common import DATA_DIR, EXP_DIR, SUB_DIR, load_train, rmse  # noqa: E402

OUT = ROOT / "experiments" / "eda"


def _fit_w(P, y):
    w, _ = nnls(P, y)
    return w / w.sum() if w.sum() > 0 else np.full(P.shape[1], 1 / P.shape[1])


def blend_oof(blend="blend_v2"):
    """Nested-CV OOF of the saved blend (same procedure as src.ensemble). Returns train df with columns
    hardness, fold, pred, resid and the per-experiment OOF columns."""
    cfg = json.loads((SUB_DIR / f"{blend}.json").read_text())
    tr = load_train()
    exps = cfg["exps"]
    P = np.column_stack([tr[["ID"]].merge(pd.read_csv(EXP_DIR / e / "oof.csv"), on="ID").hardness for e in exps])
    y = tr.hardness.values
    nested = np.zeros(len(y))
    for f in range(5):
        m = (tr.fold == f).values
        nested[m] = P[m] @ _fit_w(P[~m], y[~m])
    tr = tr.copy()
    tr["pred"] = nested
    tr["resid"] = y - nested
    for e, col in zip(exps, P.T):
        tr["oof_" + e] = col
    return tr


def load_feats(names=("features_v3.parquet", "features_v2.parquet", "features_cal.parquet")):
    df = None
    for n in names:
        d = pd.read_parquet(DATA_DIR / n)
        df = d if df is None else df.merge(d, on="ID")
    return df


def train_table():
    tr = blend_oof()
    f = load_feats()
    return tr.merge(f, on="ID", how="left")
