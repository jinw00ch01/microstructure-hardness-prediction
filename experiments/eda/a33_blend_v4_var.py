"""A33: nested OOF of blend_v4 (same nested NNLS as src.ensemble's default path) and the variance model refit.
v_i = a + b/N + c/snr fitted by Gaussian ML on nested residuals of blend_v2 and blend_v4 (paired bootstrap for b4/b2).
Also compares the 1/N term with effective-number forms: 1/N_eff from the watershed (clean images only),
exp(v4c_la_mean)/A (calibrated area-weighted grain area, all images), L^2/A (autocorrelation length).
Writes experiments/eda/blend_v4_nested_oof.csv.  Usage: python a33_blend_v4_var.py CACHE_DIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from eda_common import DATA_DIR, OUT, blend_oof, load_feats, rmse

cache = Path(sys.argv[1])
A = 256 * 256
b2, b4 = blend_oof("blend_v2"), blend_oof("blend_v4")
assert (b2.ID.values == b4.ID.values).all()
print(f"nested OOF RMSE blend_v2 {rmse(b2.pred, b2.hardness):.4f}  blend_v4 {rmse(b4.pred, b4.hardness):.4f}")
b4[["ID", "hardness", "fold", "pred", "resid"]].to_csv(OUT / "blend_v4_nested_oof.csv", index=False)
f = load_feats(("features_v3.parquet", "features_cal.parquet", "features_v4.parquet"))
t = b4[["ID", "hardness", "fold", "pred", "resid"]].merge(f, on="ID", how="left")
t["resid2"] = b2.resid.values
gs = pd.read_parquet(cache / "grainstats_train.parquet")
t = t.merge(gs, on="ID", how="left")
y = t.hardness.values
fold = t.fold.values
N = t.cal_seg_count_density.values * A / 1e4
snr = np.clip(t.ic_ridge_snr.values, 0.05, None)
terc = np.asarray(pd.qcut(N, 3, labels=["coarse", "mid", "fine"]))
clean = t.ic_ridge_snr.values > 0.9


def nll(v, rr):
    return 0.5 * np.sum(np.log(2 * np.pi * v) + rr ** 2 / v)


def fit_lin(X, rr):
    k = X.shape[1]
    th0 = np.log(np.full(k, np.mean(rr ** 2) / k) / np.maximum(X.mean(0), 1e-9))
    obj = lambda th: nll(X @ np.exp(th), rr)  # noqa: E731
    r0 = minimize(obj, th0, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-8, "fatol": 1e-10})
    return np.exp(minimize(obj, r0.x, method="BFGS").x)


def cv_nll(X, rr, ff):
    s = 0.0
    for k in range(5):
        w = fit_lin(X[ff != k], rr[ff != k])
        s += nll(X[ff == k] @ w, rr[ff == k])
    return s


one = np.ones(len(y))
X3 = np.column_stack([one, 1 / N, 1 / snr])
print("\n== variance model v = a + b/N + c/snr (nested residuals) ==")
fits = {}
for lab, rr in (("blend_v2", t.resid2.values), ("blend_v4", t.resid.values)):
    a_, bb, cc = fit_lin(X3, rr)
    fits[lab] = (a_, bb, cc)
    print(f"{lab}: RMSE {rmse(rr, 0):.3f} | a={a_:.2f} b={bb:.0f} c={cc:.2f} | mean parts: b/N {np.mean(bb / N):.1f} "
          f"c/snr {np.mean(cc / snr):.1f} a {a_:.1f} (total {np.mean(X3 @ np.array([a_, bb, cc])):.1f}; MSE {np.mean(rr ** 2):.1f})")
rng = np.random.default_rng(0)
ratios = []
for _ in range(200):
    i = rng.integers(0, len(y), len(y))
    ratios.append(fit_lin(X3[i], t.resid.values[i])[1] / fit_lin(X3[i], t.resid2.values[i])[1])
print(f"b(v4)/b(v2) = {fits['blend_v4'][1] / fits['blend_v2'][1]:.3f}; paired bootstrap 90% CI "
      f"[{np.quantile(ratios, 0.05):.3f}, {np.quantile(ratios, 0.95):.3f}]")
print("\nRMSE by N tercile (all | clean snr>0.9):")
for lab, rr in (("blend_v2", t.resid2.values), ("blend_v4", t.resid.values)):
    print(f"  {lab}: " + " ".join(f"{g} {rmse(rr[terc == g], 0):.2f}" for g in ("coarse", "mid", "fine")) + " | " +
          " ".join(f"{g} {rmse(rr[(terc == g) & clean], 0):.2f}" for g in ("coarse", "mid", "fine")))
# what the b/N term would imply if it were all label noise (upper bound on label noise, not a floor)
a_, bb, cc = fits["blend_v4"]
print(f"if all of b/N + a were label noise: RMSE of the noise alone {np.sqrt(np.mean(a_ + bb / N)):.2f} "
      f"(c/snr part {np.sqrt(np.mean(cc / snr)):.2f})")

print("\n== which size measure carries the variance? CV neg-log-lik (lower = better), blend_v4 residuals ==")
la_c = t.v4c_la_mean.values
L = t.ic_acg_len50_gm.values
forms = {
    "a + c/snr": np.column_stack([one, 1 / snr]),
    "a + b/N_cal + c/snr": X3,
    "a + b*exp(v4c_la_mean)/A + c/snr": np.column_stack([one, np.exp(la_c) / A, 1 / snr]),
    "a + b*L^2/A + c/snr": np.column_stack([one, L ** 2 / A, 1 / snr]),
    "a + b/N_cal + b2*exp(la)/A + c/snr": np.column_stack([one, 1 / N, np.exp(la_c) / A, 1 / snr]),
}
for nm, X in forms.items():
    ok = np.isfinite(X).all(1)
    print(f"  all ({ok.sum()}): {nm:38s} CV nll {cv_nll(X[ok], t.resid.values[ok], fold[ok]):9.2f}  coefs "
          f"{np.round(fit_lin(X[ok], t.resid.values[ok]), 3).tolist()}")
Nws, Neff = t.g_n.values.astype(float), t.g_neff.values
cm = clean & np.isfinite(Neff) & (Nws > 5)
print(f"  clean images (snr>0.9, n={cm.sum()}), watershed counts:")
cforms = {
    "a": one[:, None],
    "a + b/N_cal": np.column_stack([one, 1 / N]),
    "a + b/N_ws": np.column_stack([one, 1 / Nws]),
    "a + b/N_eff_ws": np.column_stack([one, 1 / Neff]),
    "a + b*exp(v4c_la_mean)/A": np.column_stack([one, np.exp(la_c) / A]),
    "a + b/N_cal + b2/N_eff_ws": np.column_stack([one, 1 / N, 1 / Neff]),
}
for nm, X in cforms.items():
    print(f"    {nm:30s} CV nll {cv_nll(X[cm], t.resid.values[cm], fold[cm]):8.2f}  coefs {np.round(fit_lin(X[cm], t.resid.values[cm]), 3).tolist()}")
print(f"  corr(log N_cal, log N_eff_ws) on clean {np.corrcoef(np.log(N[cm]), np.log(Neff[cm]))[0, 1]:.3f}; "
      f"median N_eff/N_ws {np.median(Neff[cm] / Nws[cm]):.3f}")
