"""A5: which variance model explains the blend OOF residuals?  Gaussian likelihood of r_i ~ N(0, v_i), variance
models fitted by ML; compared with 5-fold CV log-likelihood (shared folds) on all images and on clean images
(watershed reliable).  Sampling-noise hypothesis predicts v = s0^2 + b * p(1-p)/N  (or b/N)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from eda_common import OUT, train_table

cache = Path(sys.argv[1])
t = train_table().merge(pd.read_parquet(cache / "grainstats_train.parquet"), on="ID")
r = t.resid.values
fold = t.fold.values
A = 256 * 256

p_cal = np.clip(t.cal_ic_seg_fd91.values, 0.005, 0.995)
N_cal = t.cal_seg_count_density.values * A / 1e4
N_seg = t.g_n.values.astype(float)
Neff_seg = t.g_neff.values
L = t.ic_acg_len50_gm.values
Lpar = t.ic_acg_len50_par.values
Lpar_cal = t.cal_ic_acg_len50_par.values
N_ac = A / (t.ic_acg_len50_par.values * t.ic_acg_len50_perp.values)
noise = t.ic_noise.values
snr = t.ic_ridge_snr.values


def nll(v, rr):
    return 0.5 * np.sum(np.log(2 * np.pi * v) + rr ** 2 / v)


def fit_lin(X, rr):
    """v = sum_j exp(theta_j) * X_j  (X includes a constant column) -> non-negative coefficients."""
    k = X.shape[1]
    th0 = np.log(np.full(k, np.mean(rr ** 2) / k) / np.maximum(X.mean(0), 1e-9))
    obj = lambda th: nll(X @ np.exp(th), rr)
    res = minimize(obj, th0, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-8})
    res = minimize(obj, res.x, method="BFGS")
    return np.exp(res.x)


def fit_pow(x, rr):
    """v = exp(a) * x^c"""
    obj = lambda th: nll(np.exp(th[0]) * x ** th[1], rr)
    res = minimize(obj, [np.log(np.mean(rr ** 2)), 0.0], method="Nelder-Mead")
    return res.x


def evaluate(name, Xfun, mask, kind="lin"):
    rr = r[mask]
    X = Xfun()[mask] if kind == "lin" else Xfun()[mask]
    ff = fold[mask]
    cv = 0.0
    for f in range(5):
        tr_, te_ = ff != f, ff == f
        if kind == "lin":
            w = fit_lin(X[tr_], rr[tr_])
            v = X[te_] @ w
        else:
            a, c = fit_pow(X[tr_], rr[tr_])
            v = np.exp(a) * X[te_] ** c
        cv += nll(v, rr[te_])
    full = fit_lin(X, rr) if kind == "lin" else fit_pow(X, rr)
    return cv, full


one = lambda: np.ones(len(r))
models = {
    "const": (lambda: one()[:, None], "lin"),
    "a+b*L": (lambda: np.column_stack([one(), L]), "lin"),
    "a+b*L^2": (lambda: np.column_stack([one(), L ** 2]), "lin"),
    "a+b*Lpar": (lambda: np.column_stack([one(), Lpar]), "lin"),
    "a+b*Lpar^2": (lambda: np.column_stack([one(), Lpar ** 2]), "lin"),
    "a+b*Lpar_cal^2": (lambda: np.column_stack([one(), Lpar_cal ** 2]), "lin"),
    "pow(L)": (lambda: L, "pow"),
    "a+b/N_cal": (lambda: np.column_stack([one(), 1 / N_cal]), "lin"),
    "a+b*p(1-p)/N_cal": (lambda: np.column_stack([one(), p_cal * (1 - p_cal) / N_cal]), "lin"),
    "a+b/N_cal+c*p(1-p)/N_cal": (lambda: np.column_stack([one(), 1 / N_cal, p_cal * (1 - p_cal) / N_cal]), "lin"),
    "a+b/N_ac": (lambda: np.column_stack([one(), 1 / N_ac]), "lin"),
    "a+b*p(1-p)/N_ac": (lambda: np.column_stack([one(), p_cal * (1 - p_cal) / N_ac]), "lin"),
    "a+b/N_ac+c*p(1-p)/N_ac": (lambda: np.column_stack([one(), 1 / N_ac, p_cal * (1 - p_cal) / N_ac]), "lin"),
    "a+b/N_seg": (lambda: np.column_stack([one(), 1 / N_seg]), "lin"),
    "a+b/Neff_seg": (lambda: np.column_stack([one(), 1 / Neff_seg]), "lin"),
    "a+b*p(1-p)/Neff_seg": (lambda: np.column_stack([one(), p_cal * (1 - p_cal) / Neff_seg]), "lin"),
    "a+b/N_ac+c*noise^2": (lambda: np.column_stack([one(), 1 / N_ac, noise ** 2]), "lin"),
    "a+b*Lpar^2+c*noise^2": (lambda: np.column_stack([one(), Lpar ** 2, noise ** 2]), "lin"),
}
out = []
for sub, mask in (("all", np.ones(len(r), bool)), ("clean(snr>0.5)", snr > 0.5)):
    base = None
    print(f"\n=== {sub}: n={mask.sum()}  RMSE={np.sqrt(np.mean(r[mask] ** 2)):.3f}")
    for name, (fn, kind) in models.items():
        cv, full = evaluate(name, fn, mask, kind)
        base = cv if base is None else base
        print(f"{name:28s} CV-NLL {cv:9.2f}  gain vs const {base - cv:7.2f}   params {np.round(full, 4)}")
        out.append({"subset": sub, "model": name, "cv_nll": cv, "gain": base - cv, "params": np.round(full, 5).tolist()})
pd.DataFrame(out).to_csv(OUT / "a5_varfit.csv", index=False)

# binned view: mean r^2 vs mean 1/N_ac and vs p(1-p)/N_ac
for nm, x in (("1/N_ac", 1 / N_ac), ("p(1-p)/N_ac", p_cal * (1 - p_cal) / N_ac), ("1/N_cal", 1 / N_cal),
              ("Lpar^2", Lpar ** 2)):
    q = pd.qcut(x, 10, labels=False, duplicates="drop")
    g = pd.DataFrame({"x": x, "r2": r ** 2}).groupby(q).mean()
    print(f"\n{nm} deciles: x_mean / RMSE")
    print(" ".join(f"{a:.4g}/{np.sqrt(b):.1f}" for a, b in zip(g.x, g.r2)))
