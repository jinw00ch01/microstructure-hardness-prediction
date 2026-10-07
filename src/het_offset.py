"""Grain-size heterogeneity offset on top of a blend, for images with raw ic_noise < 9.5 (src/het_blocks.py).

The residual y - (nested OOF of the base blend) is regressed by OLS on [1, het4, N_eff^0.25] over the gated train
images; nested CV fits the offset on the other four folds; the test offset uses all gated train images. Images
above the gate keep the base prediction.

  python -m src.blend_cells ... --out blend_v18          # also writes submissions/blend_v18_oof.csv (nested OOF)
  python -m src.het_blocks
  python -m src.het_offset --base blend_v18 --out blend_v19
"""
import argparse
import json

import numpy as np
import pandas as pd

from .common import DATA_DIR, SUB_DIR, load_test, load_train, rmse


def design(F, ids):
    F = F.set_index("ID").loc[ids]
    return np.column_stack([np.ones(len(ids)), F.het4.values, F.N_eff.values ** 0.25])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    tr, te = load_train(), load_test()
    y, folds = tr.hardness.values, tr.fold.values
    oof = tr[["ID"]].merge(pd.read_csv(SUB_DIR / f"{a.base}_oof.csv"), on="ID").hardness.values
    base = te[["ID"]].merge(pd.read_csv(SUB_DIR / f"{a.base}.csv"), on="ID").hardness.values
    Ftr, Fte = pd.read_parquet(DATA_DIR / "het_blocks_train.parquet"), pd.read_parquet(DATA_DIR / "het_blocks_test.parquet")
    g_tr, g_te = tr.ID.isin(Ftr.ID).values, te.ID.isin(Fte.ID).values
    X, Xt = np.zeros((len(tr), 3)), np.zeros((len(te), 3))
    X[g_tr], Xt[g_te] = design(Ftr, tr.ID[g_tr]), design(Fte, te.ID[g_te])
    r, off = y - oof, np.zeros(len(tr))
    for f in range(5):
        fit, app = g_tr & (folds != f), g_tr & (folds == f)
        off[app] = X[app] @ np.linalg.lstsq(X[fit], r[fit], rcond=None)[0]
    coef = np.linalg.lstsq(X[g_tr], r[g_tr], rcond=None)[0]
    nested, pred = oof + off, base + Xt @ coef
    fr = [rmse(nested[folds == f], y[folds == f]) for f in range(5)]
    print(f"base nested {rmse(oof, y):.4f} -> {rmse(nested, y):.4f} folds {np.round(fr, 3).tolist()} | gated train "
          f"{g_tr.sum()} test {g_te.sum()} | coef [1, het4, N_eff^0.25] {np.round(coef, 2).tolist()} | test offset sd "
          f"{(Xt @ coef)[g_te].std():.2f}")
    if a.out:
        sub = pd.DataFrame({"ID": te.ID, "hardness": pred})
        assert len(sub) == 1000 and np.isfinite(sub.hardness).all()
        sub.to_csv(SUB_DIR / f"{a.out}.csv", index=False)
        pd.DataFrame({"ID": tr.ID, "hardness": nested}).to_csv(SUB_DIR / f"{a.out}_oof.csv", index=False)
        (SUB_DIR / f"{a.out}.json").write_text(json.dumps({
            "base": a.base, "mode": "het_offset", "gate": "raw ic_noise < 9.5 (src.het_blocks.NOISE_MAX)",
            "coef_1_het4_hp": coef.tolist(), "train_n": int(g_tr.sum()), "test_n": int(g_te.sum()),
            "nested_cv_rmse": rmse(nested, y), "fold_rmse": fr, "notes": a.note}, indent=2))
        print("written", SUB_DIR / f"{a.out}.csv")


if __name__ == "__main__":
    main()
