"""Two-stage blend with a cell-dependent CNN share (orchestrator; reproduces blend_v14 and blend_v15).

Stage 1: NNLS over a member pool (src.ensemble.fit_w), the pool taken from an earlier blend json.
Stage 2: a fixed CNN mix at share s(x): s_cell for images in the cell, s_other elsewhere.
Cell "fine_noisy": log(cal_seg_count_density) > train median AND raw ic_noise > train median (no labels, no test
statistics). Nested CV refits the stage-1 weights per outer fold; the shares are fixed by the caller.

  python -m src.blend_cells --pool-from blend_v7 --cnn cnn_ev2s_rawnlm_degcons_gpu_s6:0.7,cnn_cnxt_rawnlm_degcons_gpu_s3:0.3 \
      --share-other 0.2 --share-cell 0.7 --out blend_v15
(--share-cell 0.2 gives blend_v14.)
"""
import argparse
import json

import numpy as np
import pandas as pd

from .common import DATA_DIR, EXP_DIR, SUB_DIR, load_test, load_train, rmse
from .ensemble import fit_w


def cell_masks(tr_ids, te_ids, cell="fine_noisy"):
    assert cell == "fine_noisy", cell
    cal = pd.read_parquet(DATA_DIR / "features_cal.parquet").set_index("ID")
    v3 = pd.read_parquet(DATA_DIR / "features_v3.parquet").set_index("ID")
    g = lambda ids: np.log(cal.loc[ids, "cal_seg_count_density"].values)
    n = lambda ids: v3.loc[ids, "ic_noise"].values
    mg, mn = float(np.median(g(tr_ids))), float(np.median(n(tr_ids)))
    rule = {"log_cal_seg_count_density_gt": mg, "ic_noise_gt": mn}
    return (g(tr_ids) > mg) & (n(tr_ids) > mn), (g(te_ids) > mg) & (n(te_ids) > mn), rule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-from", default="blend_v7")
    ap.add_argument("--cnn", required=True, help="name:weight,... (weights of the fixed CNN mix)")
    ap.add_argument("--share-other", type=float, default=0.2)
    ap.add_argument("--share-cell", type=float, default=0.2)
    ap.add_argument("--cell", default="fine_noisy")
    ap.add_argument("--out", default=None)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    tr, te = load_train(), load_test()
    y, folds = tr.hardness.values, tr.fold.values
    cnn = [(s.split(":")[0], float(s.split(":")[1])) for s in a.cnn.split(",")]
    pool = [e for e in json.load(open(SUB_DIR / f"{a.pool_from}.json"))["exps"] if e not in dict(cnn)]
    ld = lambda e, k: (tr if k == "oof" else te)[["ID"]].merge(pd.read_csv(EXP_DIR / e / f"{k}.csv"), on="ID").hardness.values
    P = np.column_stack([ld(e, "oof") for e in pool])
    T = np.column_stack([ld(e, "test") for e in pool])
    c = sum(w * ld(e, "oof") for e, w in cnn)
    ct = sum(w * ld(e, "test") for e, w in cnn)
    in_tr, in_te, rule = cell_masks(tr.ID, te.ID, a.cell)
    nested = np.zeros(len(y))
    for f in range(5):
        m = folds == f
        b = P[m] @ fit_w(P[~m], y[~m])
        nested[m] = b + np.where(in_tr[m], a.share_cell, a.share_other) * (c[m] - b)
    w = fit_w(P, y)
    bt = T @ w
    pred = bt + np.where(in_te, a.share_cell, a.share_other) * (ct - bt)
    fr = [rmse(nested[folds == f], y[folds == f]) for f in range(5)]
    print(f"nested CV {rmse(nested, y):.4f} folds {np.round(fr, 3).tolist()} | cell train {in_tr.sum()} test {in_te.sum()}")
    if a.out:
        sub = pd.DataFrame({"ID": te.ID, "hardness": pred})
        assert len(sub) == 1000 and np.isfinite(sub.hardness).all()
        sub.to_csv(SUB_DIR / f"{a.out}.csv", index=False)
        rule.update(train_n=int(in_tr.sum()), test_n=int(in_te.sum()))
        (SUB_DIR / f"{a.out}.json").write_text(json.dumps({
            "exps": pool + [e for e, _ in cnn], "base_weights": w.tolist(), "cnn_mix": dict(cnn),
            "cnn_share": {"cell": a.share_cell, "other": a.share_other}, "cell": a.cell, "cell_rule": rule,
            "nested_cv_rmse": rmse(nested, y), "fold_rmse": fr, "notes": a.note}, indent=2))
        print("written", SUB_DIR / f"{a.out}.csv")


if __name__ == "__main__":
    main()
