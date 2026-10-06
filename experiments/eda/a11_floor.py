"""A11: floor RMSE from the fitted variance model.  v_i = a + b/N_i + c*deg_i  (N_i = calibrated grain count,
deg_i = degradation term).  Floor = sqrt(mean(b/N_i [+ a_clean])) on train; public-LB spread by simulation
on random 300-image subsets of the 1000 test images (only the unsupervised N_i of test images is used)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from eda_common import DATA_DIR, OUT, train_table

t = train_table()
r = t.resid.values
fold = t.fold.values
A = 65536 / 1e4
N = t.cal_seg_count_density.values * A
snr = t.ic_ridge_snr.values
clean = snr > 0.5
deg = {"none": np.zeros(len(r)), "1/snr": 1 / np.clip(snr, 0.05, None), "noise^2": t.ic_noise.values ** 2,
       "logsnr-": np.clip(-np.log(snr), 0, None)}


def nll(v, rr):
    return 0.5 * np.sum(np.log(2 * np.pi * v) + rr ** 2 / v)


def fit(X, rr):
    k = X.shape[1]
    th0 = np.log(np.full(k, np.mean(rr ** 2) / k) / np.maximum(X.mean(0), 1e-9))
    obj = lambda th: nll(X @ np.exp(th), rr)
    res = minimize(obj, th0, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-7, "fatol": 1e-9})
    res = minimize(obj, res.x, method="BFGS")
    return np.exp(res.x), res.fun


rows = []
for dn, d in deg.items():
    for sub, m in (("all", np.ones(len(r), bool)), ("clean", clean)):
        if dn != "none" and sub == "clean":
            continue
        X = np.column_stack([np.ones(len(r)), 1 / N] + ([d] if dn != "none" else []))[m]
        w, f = fit(X, r[m])
        cv = 0
        for k in range(5):
            tr_, te_ = fold[m] != k, fold[m] == k
            wk, _ = fit(X[tr_], r[m][tr_])
            cv += nll(X[te_] @ wk, r[m][te_])
        rows.append({"deg": dn, "subset": sub, "a": w[0], "b": w[1], "c": w[2] if len(w) > 2 else 0, "nll": f, "cv_nll": cv})
        print(rows[-1])
# bootstrap the clean fit for uncertainty on a, b
rng = np.random.default_rng(0)
Xc = np.column_stack([np.ones(clean.sum()), 1 / N[clean]])
bs = []
for _ in range(300):
    ii = rng.integers(0, clean.sum(), clean.sum())
    w, _ = fit(Xc[ii], r[clean][ii])
    bs.append(w)
bs = np.array(bs)
print("clean fit bootstrap a: median %.1f 90%% [%.1f, %.1f]; b: median %.0f 90%% [%.0f, %.0f]" % (
    np.median(bs[:, 0]), *np.quantile(bs[:, 0], [.05, .95]), np.median(bs[:, 1]), *np.quantile(bs[:, 1], [.05, .95])))
a_c, b_c = fit(Xc, r[clean])[0]
te = pd.read_parquet(DATA_DIR / "features_cal.parquet")
te = te[te.ID.str.startswith("TEST")]
Nte = te.cal_seg_count_density.values * A
print("N_cal train median %.0f, test median %.0f; mean 1/N train %.5f test %.5f" % (np.median(N), np.median(Nte), np.mean(1 / N), np.mean(1 / Nte)))
for nm, aa in (("per-grain term only (a=0)", 0.0), ("+ clean intercept", a_c)):
    v_tr = aa + b_c / N
    v_te = aa + b_c / Nte
    print(f"floor [{nm}]: train RMSE {np.sqrt(v_tr.mean()):.2f}, test(1000) {np.sqrt(v_te.mean()):.2f}")
    # public subset simulation: 300 random test images, Gaussian errors with variance v_i
    sims = []
    for _ in range(20000):
        idx = rng.choice(len(Nte), 300, replace=False)
        e = rng.normal(0, np.sqrt(v_te[idx]))
        sims.append(np.sqrt(np.mean(e ** 2)))
    sims = np.array(sims)
    print(f"   public-300 RMSE: mean {sims.mean():.2f}, 5-95% [{np.quantile(sims, .05):.2f}, {np.quantile(sims, .95):.2f}], P(<=10) {np.mean(sims <= 10):.3f}")
# bootstrap-propagated floor uncertainty
fl = [np.sqrt(np.mean(bb[1] / N)) for bb in bs]
fl2 = [np.sqrt(np.mean(bb[0] + bb[1] / N)) for bb in bs]
print("floor (b/N only) train 90%% CI [%.2f, %.2f]; with intercept [%.2f, %.2f]" % (*np.quantile(fl, [.05, .95]), *np.quantile(fl2, [.05, .95])))
pd.DataFrame(rows).to_csv(OUT / "a11_floor_fits.csv", index=False)
