"""A15: public-LB (random 300 of 1000 test images) distribution under the fitted variance model
v_i = b/N_i + c*s/snr_i (+a), with b, c from bootstrap draws (train residuals). Test images contribute only their
unsupervised N_cal / snr.  Also: paired LB noise between two submissions, from train OOFs (random 30% subsets)."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from eda_common import DATA_DIR, rmse, train_table

t = train_table()
r = t.resid.values
N = t.cal_seg_count_density.values * 6.5536
snr = np.clip(t.ic_ridge_snr.values, 0.05, None)
X = np.column_stack([np.ones(len(r)), 1 / N, 1 / snr])


def nll(v, rr):
    return 0.5 * np.sum(np.log(2 * np.pi * v) + rr ** 2 / v)


def fit(X, rr):
    obj = lambda th: nll(X @ np.exp(th), rr)
    res = minimize(obj, np.log([20, 2e4, 10]), method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-8, "fatol": 1e-10})
    return np.exp(minimize(obj, res.x, method="BFGS").x)


rng = np.random.default_rng(2)
bs = np.array([fit(X[ii], r[ii]) for ii in (rng.integers(0, len(r), len(r)) for _ in range(200))])
f = pd.read_parquet(DATA_DIR / "features_cal.parquet").merge(pd.read_parquet(DATA_DIR / "features_v3.parquet")[["ID", "ic_ridge_snr"]], on="ID")
te = f[f.ID.str.startswith("TEST")]
Nte = te.cal_seg_count_density.values * 6.5536
ste = np.clip(te.ic_ridge_snr.values, 0.05, None)
for name, deg_scale, use_a in (("perfect model (label-noise term only)", 0.0, False), ("degradation error halved", 0.5, True),
                               ("current blend (fitted model)", 1.0, True)):
    sims, full = [], []
    for k in range(20000):
        a, b, c = bs[k % len(bs)]
        v = b / Nte + deg_scale * c / ste + (a if use_a else 0.0)
        full.append(np.sqrt(v.mean()))
        idx = rng.choice(len(Nte), 300, replace=False)
        sims.append(np.sqrt(np.mean(rng.normal(0, np.sqrt(v[idx])) ** 2)))
    sims = np.array(sims)
    print(f"{name:40s}: expected test RMSE {np.mean(full):.2f}; public-300 median {np.median(sims):.2f}, 5-95% [{np.quantile(sims, .05):.2f}, {np.quantile(sims, .95):.2f}], P(<=10) {np.mean(sims <= 10):.3f}")
# train-side analogue (expected CV RMSE floor)
fl = np.sqrt(bs[:, 1] * np.mean(1 / N))
print(f"train CV floor (label-noise term only): median {np.median(fl):.2f}, 90% [{np.quantile(fl, .05):.2f}, {np.quantile(fl, .95):.2f}]")
# paired LB noise between blend and a single model, from train OOF subsets of 30%
y = t.hardness.values
for e in ("feat2_v23cal_lgbs_het", "feat2_v3cal_ridge_het", "cnn_r18_c224_e30"):
    d = []
    for _ in range(5000):
        idx = rng.choice(len(y), 150, replace=False)
        d.append(rmse(t.pred.values[idx], y[idx]) - rmse(t["oof_" + e].values[idx], y[idx]))
    d = np.array(d)
    print(f"LB(blend) - LB({e}) on random 30% subsets: full-data diff {rmse(t.pred, y) - rmse(t['oof_' + e], y):+.3f}, subset sd {d.std():.3f} "
          f"(scaled to 300 images: {d.std() * np.sqrt(150 / 300):.3f})")
