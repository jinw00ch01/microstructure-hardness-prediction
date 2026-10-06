"""A24: multiple-instance view of grain size on truly clean images: hardness ~ sum_j w_j phi(phase_j, log A_j), i.e. area
fractions of grains in log-area bins per phase (sums to 1), vs simpler image-level summaries.  Ridge CV on the shared folds
restricted to clean images, compared with the blend OOF on the same images, overall and by N tercile.
Usage: python a24_sizebins.py CACHE_DIR [snr_min]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from eda_common import OUT, rmse, train_table

cache = Path(sys.argv[1])
snr_min = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
g = pd.read_parquet(cache / "grains_train.parquet")
g = g[g.area >= 12].copy()
g["dk"] = g.med < 0.91
edges = np.log([12, 50, 100, 200, 400, 800, 1600, 3200, 1e6])
g["bin"] = np.digitize(np.log(g.area), edges[1:-1])
rows = []
for i, d in g.groupby("ID"):
    A = d.area.values.astype(float)
    tot = A.sum()
    f = {"ID": i}
    for ph, m in (("dk", d.dk.values), ("mx", ~d.dk.values)):
        for b in range(len(edges) - 1):
            f[f"sb_{ph}_{b}"] = float(A[m & (d.bin.values == b)].sum() / tot)
    f["sb_fd"] = float(A[d.dk.values].sum() / tot)
    f["sb_logAn"] = float(np.log(A.mean()))
    f["sb_logAw"] = float(np.log((A ** 2).sum() / tot))
    f["sb_logAw_dk"] = float(np.log((A[d.dk.values] ** 2).sum() / max(A[d.dk.values].sum(), 1))) if d.dk.any() else np.nan
    f["sb_logAw_mx"] = float(np.log((A[~d.dk.values] ** 2).sum() / max(A[~d.dk.values].sum(), 1)))
    rows.append(f)
SB = pd.DataFrame(rows)
SB.to_parquet(cache / "sb_train.parquet")
t = train_table().merge(SB, on="ID", how="left")
N = t.cal_seg_count_density.values * 6.5536
t["terc"] = np.asarray(pd.qcut(N, 3, labels=["coarse", "mid", "fine"]))
c = t[t.ic_ridge_snr > snr_min].reset_index(drop=True)
c["sb_logAw_dk"] = c.sb_logAw_dk.fillna(c.sb_logAw_mx)
y = c.hardness.values
fold = c.fold.values
bins = [x for x in SB.columns if x.startswith("sb_dk_") or x.startswith("sb_mx_")]
sets = {
    "fd": ["sb_fd"],
    "fd + logAn": ["sb_fd", "sb_logAn"],
    "fd + logAw": ["sb_fd", "sb_logAw"],
    "fd + logAn + logAw": ["sb_fd", "sb_logAn", "sb_logAw"],
    "fd + logAw per phase": ["sb_fd", "sb_logAw_dk", "sb_logAw_mx"],
    "size bins per phase (16)": bins,
    "size bins + logAn + logAw": bins + ["sb_logAn", "sb_logAw"],
    "fd + cal_seg_count_density + ic_acg_len50_par": ["sb_fd", "cal_seg_count_density", "ic_acg_len50_par"],
}
print(f"clean images snr>{snr_min}: n={len(c)}; blend RMSE {rmse(c.pred, y):.3f} | " +
      " ".join(f"{g_} {rmse(c.pred[c.terc == g_], y[c.terc == g_]):.2f}" for g_ in ("coarse", "mid", "fine")))
coefs = None
for nm, cols in sets.items():
    X = c[cols].astype(float).values
    oof = np.zeros(len(y))
    for k in range(5):
        tr_, te_ = fold != k, fold == k
        mu, sd = X[tr_].mean(0), X[tr_].std(0) + 1e-9
        m = RidgeCV(alphas=np.logspace(-3, 3, 31)).fit((X[tr_] - mu) / sd, y[tr_])
        oof[te_] = m.predict((X[te_] - mu) / sd)
    blend_corr = np.corrcoef(oof - c.pred, c.resid)[0, 1]
    print(f"{nm:46s} CV {rmse(oof, y):.2f} | " + " ".join(f"{g_} {rmse(oof[c.terc == g_], y[c.terc == g_]):.2f}" for g_ in ("coarse", "mid", "fine")) +
          f" | 50/50 with blend {rmse(0.5 * oof + 0.5 * c.pred, y):.2f} | corr(oof-pred, resid) {blend_corr:+.2f}")
    if nm.startswith("size bins per phase"):
        mu, sd = X.mean(0), X.std(0) + 1e-9
        m = RidgeCV(alphas=np.logspace(-3, 3, 31)).fit((X - mu) / sd, y)
        coefs = pd.Series(m.coef_ / sd, index=cols)  # HV per unit area fraction
print("per-grain hardness by size bin (HV per unit area fraction, relative; bin edges in px:", np.round(np.exp(edges)).astype(int).tolist(), ")")
print(coefs.round(1).to_string())
