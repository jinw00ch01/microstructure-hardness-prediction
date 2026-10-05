"""Non-negative OOF blend of experiments. python -m src.ensemble [exp1 exp2 ...] --out blend_v1"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from .common import EXP_DIR, SUB_DIR, load_test, load_train, rmse


def fit_w(P, y):
    w, _ = nnls(P, y)
    return w / w.sum() if w.sum() > 0 else np.full(P.shape[1], 1 / P.shape[1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="*")
    ap.add_argument("--out", default="blend")
    a = ap.parse_args()
    tr, te = load_train(), load_test()
    exps = a.exps or sorted(p.parent.name for p in EXP_DIR.glob("*/score.json"))
    P = np.column_stack([tr[["ID"]].merge(pd.read_csv(EXP_DIR / e / "oof.csv"), on="ID").hardness for e in exps])
    T = np.column_stack([te[["ID"]].merge(pd.read_csv(EXP_DIR / e / "test.csv"), on="ID").hardness for e in exps])
    y = tr.hardness.values
    for e, col in zip(exps, P.T):
        print(f"{e:50s} {rmse(col, y):.4f}")
    # honest estimate: weights fit on 4 folds, applied to the 5th
    nested = np.zeros(len(y))
    for f in range(5):
        m = (tr.fold == f).values
        nested[m] = P[m] @ fit_w(P[~m], y[~m])
    w = fit_w(P, y)
    print("weights", dict(zip(exps, np.round(w, 3))))
    print(f"blend in-sample {rmse(P @ w, y):.4f}  nested-CV {rmse(nested, y):.4f}")
    SUB_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"ID": te.ID, "hardness": T @ w}).to_csv(SUB_DIR / f"{a.out}.csv", index=False)
    (SUB_DIR / f"{a.out}.json").write_text(json.dumps({"exps": exps, "weights": w.tolist(),
                                                       "nested_cv_rmse": rmse(nested, y)}, indent=2))
    with open(EXP_DIR / "LEADERBOARD.md", "a") as fh:
        fh.write(f"| {a.out} (blend) | {rmse(nested, y):.3f} |  | nested-CV, {len(exps)} models |\n")
