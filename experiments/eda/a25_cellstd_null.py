"""A25: robustness of the effv2s grid-cell-std residual signal: per-fold gains on the blend, and a permutation null
(rows of the representation shuffled across images) for the cross-fitted probe gain on blend_v2."""
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from eda_common import DATA_DIR, OUT, rmse, train_table

t = train_table()
y = t.hardness.values; fold = t.fold.values; br = t.resid.values
N = t.cal_seg_count_density.values * 6.5536
terc = np.asarray(pd.qcut(N, 3, labels=["coarse", "mid", "fine"]))
DROP = ["cal_ic_seg_L_", "cal_ic_seg_mx_area_mean", "cal_seg_area_cv", "cal_segdk_area_cv"]
v3 = pd.read_parquet(DATA_DIR / "features_v3.parquet"); cal = pd.read_parquet(DATA_DIR / "features_cal.parquet")
base_cols = [c for c in list(v3.columns[1:]) + list(cal.columns[1:]) if not any(re.search(p, c) for p in DROP)]
XB = t[base_cols].astype(float).replace([np.inf, -np.inf], np.nan); XB = XB.fillna(XB.median()).values

def rfp(Xtr, ytr, Xte, alphas=np.logspace(-2, 4, 40)):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    return RidgeCV(alphas=alphas).fit((Xtr - mu) / sd, ytr).predict((Xte - mu) / sd)

inner = {}
for k in range(5):
    tr_, te_ = np.where(fold != k)[0], np.where(fold == k)[0]
    ir = np.zeros(len(tr_))
    for j in sorted(set(fold[tr_])):
        a = tr_[fold[tr_] != j]; b = fold[tr_] == j
        ir[b] = y[tr_][b] - rfp(XB[a], y[a], XB[tr_][b])
    inner[k] = (tr_, te_, ir)

def probe(X):
    pr = np.zeros(len(y))
    for k, (tr_, te_, ir) in inner.items():
        pr[te_] = rfp(X[tr_], ir, X[te_], alphas=np.logspace(-1, 6, 29))
    return pr

name = "tf_efficientnetv2_s.in21k_ft_in1k_256_g2a"
E = np.load(DATA_DIR / "emb" / f"{name}.npy", mmap_mode="r")
ids = pd.read_csv(DATA_DIR / "emb" / f"{name}_ids.csv").iloc[:, 0].values
pos = pd.Series(np.arange(len(ids)), index=ids)[t.ID].values
E = np.asarray(E[pos], np.float32).reshape(len(t), 6, 5, -1)
sl = lambda X: np.sign(X) * np.log1p(np.abs(X))
variants = {"cellstd(view-avg)": sl(E[:, :4, 1:].mean(1).std(1)), "cellstd(per-view, avg)": sl(E[:, :4, 1:].std(2).mean(1))}
rng = np.random.default_rng(0)
for nm, X in variants.items():
    pr = probe(X)
    g = rmse(br, 0) - rmse(br - pr, 0)
    pf = [f"{rmse(br[fold == k], 0):.2f}->{rmse((br - pr)[fold == k], 0):.2f}" for k in range(5)]
    null = []
    for _ in range(40):
        perm = rng.permutation(len(y))
        pn = probe(X[perm])
        null.append(rmse(br, 0) - rmse(br - pn, 0))
    null = np.array(null)
    print(f"{nm:24s} blend gain {g:+.3f} (12.963->{rmse(br - pr, 0):.3f}); per fold {pf}; null gain mean {null.mean():+.3f} "
          f"95% {np.quantile(null, .95):+.3f} max {null.max():+.3f}; p={np.mean(null >= g):.3f}", flush=True)
    np.save(OUT / f"a25_probe_{'pv' if 'per-view' in nm else 'va'}.npy", pr)
