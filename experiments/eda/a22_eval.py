"""A22: evaluate candidate feature groups against the residual (leakage-free).
Per outer fold k: base = RidgeCV(v3+cal) on the training folds; inner 4-fold cross-fitted base residuals; probe =
RidgeCV(group -> inner residual) applied to fold k.  Reports corr(probe, base residual), base gain, and the change of the
blend_v2 nested OOF RMSE when the same probe is added (overall / N terciles / clean), plus per-feature Spearman with the
blend residual by tercile.  Usage: python a22_eval.py CACHE_DIR parquet[,parquet] 'group=regex;group=regex' [clean_only]"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

from eda_common import DATA_DIR, rmse, train_table

cache = Path(sys.argv[1])
files = sys.argv[2].split(",")
groups = dict(g.split("=", 1) for g in sys.argv[3].split(";"))
clean_only = len(sys.argv) > 4 and sys.argv[4] == "clean_only"
t = train_table()
for fn in files:
    p = Path(fn)
    t = t.merge(pd.read_parquet(p if p.exists() else cache / fn), on="ID", how="left")
y = t.hardness.values
fold = t.fold.values
N = t.cal_seg_count_density.values * 6.5536
terc = np.asarray(pd.qcut(N, 3, labels=["coarse", "mid", "fine"]))
clean = t.ic_ridge_snr.values > 0.5
DROP = ["cal_ic_seg_L_", "cal_ic_seg_mx_area_mean", "cal_seg_area_cv", "cal_segdk_area_cv"]
v3 = pd.read_parquet(DATA_DIR / "features_v3.parquet")
cal = pd.read_parquet(DATA_DIR / "features_cal.parquet")
base_cols = [c for c in list(v3.columns[1:]) + list(cal.columns[1:]) if not any(re.search(p_, c) for p_ in DROP)]
XB = t[base_cols].astype(float).replace([np.inf, -np.inf], np.nan)
XB = XB.fillna(XB.median()).values


def rfp(Xtr, ytr, Xte, alphas=np.logspace(-2, 4, 40)):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    return RidgeCV(alphas=alphas).fit((Xtr - mu) / sd, ytr).predict((Xte - mu) / sd)


base_oof = np.zeros(len(y))
inner = {}
for k in range(5):
    tr_, te_ = np.where(fold != k)[0], np.where(fold == k)[0]
    base_oof[te_] = rfp(XB[tr_], y[tr_], XB[te_])
    ir = np.zeros(len(tr_))
    for j in sorted(set(fold[tr_])):
        a = tr_[fold[tr_] != j]
        b = fold[tr_] == j
        ir[b] = y[tr_][b] - rfp(XB[a], y[a], XB[tr_][b])
    inner[k] = (tr_, te_, ir)
base_res = y - base_oof
br = t.resid.values
print(f"base ridge CV {rmse(base_oof, y):.3f}; blend {rmse(t.pred, y):.3f}")
for gname, rx in groups.items():
    cols = [c for c in t.columns if re.search(rx, c)]
    X = t[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().mean() > 0.5]
    X = X.fillna(X.median()).values
    pr = np.zeros(len(y))
    for k, (tr_, te_, ir) in inner.items():
        if clean_only:
            msk = clean[tr_]
            pr[te_] = rfp(X[tr_][msk], ir[msk], X[te_], alphas=np.logspace(-1, 6, 29))
        else:
            pr[te_] = rfp(X[tr_], ir, X[te_], alphas=np.logspace(-1, 6, 29))
    if clean_only:
        pr = np.where(clean, pr, 0.0)
    out = (f"[{gname}] {len(cols)} cols | corr(base res) {np.corrcoef(pr, base_res)[0, 1]:+.3f} base gain {rmse(base_res, 0) - rmse(base_res - pr, 0):+.3f}"
           f" | blend {rmse(br, 0):.3f}->{rmse(br - pr, 0):.3f} corr {np.corrcoef(pr, br)[0, 1]:+.3f} |")
    for g in ("coarse", "mid", "fine"):
        m = terc == g
        out += f" {g} {rmse(br[m], 0):.2f}->{rmse(br[m] - pr[m], 0):.2f} ({np.corrcoef(pr[m], br[m])[0, 1]:+.2f})"
    m = clean
    out += f" | clean {rmse(br[m], 0):.2f}->{rmse(br[m] - pr[m], 0):.2f} ({np.corrcoef(pr[m], br[m])[0, 1]:+.2f})"
    m = clean & (terc == "coarse")
    out += f" clean-coarse {rmse(br[m], 0):.2f}->{rmse(br[m] - pr[m], 0):.2f} ({np.corrcoef(pr[m], br[m])[0, 1]:+.2f})"
    print(out, flush=True)
# per-feature Spearman with blend residual (top by clean-coarse |rho|)
allc = sorted({c for rx in groups.values() for c in t.columns if re.search(rx, c)})
rows = []
for c in allc:
    x = t[c].astype(float).values
    r_ = {"feat": c, "y_all": spearmanr(x, y, nan_policy="omit")[0], "r_all": spearmanr(x, br, nan_policy="omit")[0]}
    for g in ("coarse", "mid", "fine"):
        m = terc == g
        r_[f"r_{g}"] = spearmanr(x[m], br[m], nan_policy="omit")[0]
        r_[f"r_clean_{g}"] = spearmanr(x[m & clean], br[m & clean], nan_policy="omit")[0]
    rows.append(r_)
R = pd.DataFrame(rows).set_index("feat")
pd.set_option("display.width", 250)
print(R.reindex(R.r_clean_coarse.abs().sort_values(ascending=False).index).head(20).round(3).to_string())
print(f"(null SE: all 0.045, tercile 0.077, clean-coarse {1 / np.sqrt((clean & (terc == 'coarse')).sum()):.3f})")
