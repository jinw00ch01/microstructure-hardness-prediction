"""A47: does the label follow the generator's dark-phase probability p (binomial realisation p_hat) or the realised image?
(a) Variance test on blend_v5 nested residuals: if the 1/N residual came from label = f(p) with p_hat ~ p + binomial
    noise, its variance would scale as b^2 p(1-p)/N, i.e. vanish for p ~ 0.  Fit v = a + b1/N + b2 p(1-p)/N + c/snr
    (Gaussian ML, CV neg-log-lik) and compare coarse-tercile residual variance for p_hat < 0.05 vs >= 0.15.
(b) Attenuation test: slope of y on p_hat by N tercile (OLS with controls: porosity, log N, aspect, alignment) vs the
    binomial-model reliability lambda = V_p / (V_p + E[p_hat(1-p_hat)/N_e]),  V_p = var(p_hat) - E[p_hat(1-p_hat)/N_e]
    (N_e = N for number fractions, N_eff for area fractions).  Paired bootstrap CI of slope ratios coarse/fine.
Clean images (snr > 0.9) with watershed fractions, then all images with calibrated fractions.
Usage: python a47_attenuation.py CACHE_DIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from eda_cf import table
from eda_common import rmse

cache = Path(sys.argv[1])
t = table("blend_v5")
mg = pd.read_parquet(cache / "mig_train.parquet")
mg["p_num"] = mg[[c for c in mg.columns if c.startswith("mi_n_dk_")]].sum(1)
t = t.merge(mg[["ID", "p_num", "mi_fd", "mi_N", "mi_Neff"]], on="ID", how="left")
t = t.merge(pd.read_parquet(cache / "ga_train.parquet")[["ID", "al_S_e_all", "al_asp_a_all"]], on="ID", how="left")
y, r, fold = t.hardness.values, t.resid.values, t.fold.values
Ncal = t.cal_seg_count_density.values * 6.5536
snr = np.clip(t.ic_ridge_snr.values, 0.05, None)
terc = np.asarray(pd.qcut(Ncal, 3, labels=["coarse", "mid", "fine"]))
clean = t.ic_ridge_snr.values > 0.9


def nll(v, rr):
    return 0.5 * np.sum(np.log(2 * np.pi * v) + rr ** 2 / v)


def fit_lin(X, rr):
    th0 = np.log(np.full(X.shape[1], np.mean(rr ** 2) / X.shape[1]) / np.maximum(X.mean(0), 1e-9))
    obj = lambda th: nll(X @ np.exp(th), rr)  # noqa: E731
    r0 = minimize(obj, th0, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-8, "fatol": 1e-10})
    return np.exp(minimize(obj, r0.x, method="BFGS").x)


def cvnll(X, rr, ff):
    return sum(nll(X[ff == k] @ fit_lin(X[ff != k], rr[ff != k]), rr[ff == k]) for k in range(5))


print("== (a) variance: is the 1/N residual binomial in the dark fraction?")
p_all = np.clip(t.cal_ic_seg_fd91.values, 0, 1)
one = np.ones(len(y))
for nm, m, p, N in (("all images, cal fd91 / N_cal", np.ones(len(y), bool), p_all, Ncal),
                    ("clean, watershed number fraction / N_ws", clean & t.p_num.notna().values, t.p_num.values, t.mi_N.values)):
    pq = p[m] * (1 - p[m])
    forms = {"a+b/N+c/snr": np.column_stack([one[m], 1 / N[m], 1 / snr[m]]),
             "a+b p(1-p)/N+c/snr": np.column_stack([one[m], pq / N[m], 1 / snr[m]]),
             "a+b1/N+b2 p(1-p)/N+c/snr": np.column_stack([one[m], 1 / N[m], pq / N[m], 1 / snr[m]])}
    s = f"  {nm} (n={m.sum()}):"
    for fn, X in forms.items():
        s += f" | {fn} CVnll {cvnll(X, r[m], fold[m]):.1f}"
    w = fit_lin(forms["a+b1/N+b2 p(1-p)/N+c/snr"], r[m])
    print(s + f" | joint coefs b1={w[1]:.0f} b2={w[2]:.0f}")
cm = terc == "coarse"
lo, hi = cm & (p_all < 0.05), cm & (p_all >= 0.15)
print(f"  coarse tercile residual RMSE: p_hat < 0.05 {rmse(r[lo], 0):.2f} (n={lo.sum()}) vs p_hat >= 0.15 {rmse(r[hi], 0):.2f} (n={hi.sum()}); "
      f"fine tercile: {rmse(r[(terc == 'fine') & (p_all < 0.05)], 0):.2f} vs {rmse(r[(terc == 'fine') & (p_all >= 0.15)], 0):.2f}")
for nm, m in (("coarse clean", cm & clean),):
    a_, b_ = m & (p_all < 0.05), m & (p_all >= 0.15)
    print(f"  {nm}: p_hat < 0.05 {rmse(r[a_], 0):.2f} (n={a_.sum()}) vs >= 0.15 {rmse(r[b_], 0):.2f} (n={b_.sum()})")

print("== (b) attenuation of the slope of y on p_hat by N tercile")


def slopes(m, p, ctrl, B=1000):
    rng = np.random.default_rng(0)
    out = {}
    idx = {g: np.where(m & (terc == g))[0] for g in ("coarse", "mid", "fine")}

    def sl(ii):
        X = np.column_stack([np.ones(len(ii)), p[ii], ctrl[ii]])
        return np.linalg.lstsq(X, y[ii], rcond=None)[0][1]
    est = {g: sl(ii) for g, ii in idx.items()}
    bs = {g: np.array([sl(rng.choice(ii, len(ii))) for _ in range(B)]) for g, ii in idx.items()}
    for g in idx:
        out[g] = (est[g], *np.quantile(bs[g], [0.05, 0.95]))
    ratio = bs["coarse"] / bs["fine"]
    out["ratio"] = (est["coarse"] / est["fine"], *np.quantile(ratio, [0.05, 0.95]))
    return out


def predicted(m, p, Ne):
    pq = p[m] * (1 - p[m]) / Ne[m]
    Vp = np.var(p[m]) - np.mean(pq)
    lam = {g: Vp / (Vp + np.mean((p * (1 - p) / Ne)[m & (terc == g)])) for g in ("coarse", "mid", "fine")}
    return Vp, lam


ctrl_cols = ["ic_pore68_frac", "al_asp_a_all", "al_S_e_all"]
for nm, m, p, Ne in (
        ("clean, watershed number fraction (N_e = N_ws)", clean & t.p_num.notna().values, t.p_num.values, t.mi_N.values.astype(float)),
        ("clean, watershed area fraction (N_e = N_eff)", clean & t.mi_fd.notna().values, t.mi_fd.values, t.mi_Neff.values),
        ("clean, ic_seg_fd91 area fraction (N_e = N_eff)", clean & t.mi_Neff.notna().values, t.ic_seg_fd91.values, t.mi_Neff.values),
        ("all, cal_ic_seg_fd91 (N_e = 0.72 N_cal)", np.isfinite(t.al_asp_a_all.values), p_all, 0.72 * Ncal)):
    C = t[ctrl_cols].astype(float).values
    C = np.column_stack([np.where(np.isfinite(C), C, np.nanmedian(C, 0)), np.log(Ncal)])
    s = slopes(m, p, C)
    Vp, lam = predicted(m, p, Ne)
    print(f"  {nm}: n={m.sum()} sd(p_hat) {np.std(p[m]):.3f} V_p {Vp:.4f}")
    print("    slope [90% CI] coarse {:.0f} [{:.0f},{:.0f}] mid {:.0f} [{:.0f},{:.0f}] fine {:.0f} [{:.0f},{:.0f}]".format(
        *s["coarse"], *s["mid"], *s["fine"]) + f" | observed coarse/fine {s['ratio'][0]:.2f} [{s['ratio'][1]:.2f},{s['ratio'][2]:.2f}]"
          f" | binomial-predicted lambda coarse/mid/fine {lam['coarse']:.3f}/{lam['mid']:.3f}/{lam['fine']:.3f} ratio {lam['coarse'] / lam['fine']:.3f}")

print("== (c) does the 1/N excess live in the dark phase?  variance forms in f = cal_ic_seg_fd91 (CV neg-log-lik, all images)")
f_ = p_all
for fn, X in {"a+b/N+c/snr": np.column_stack([one, 1 / Ncal, 1 / snr]),
              "a+b f/N+c/snr": np.column_stack([one, f_ / Ncal, 1 / snr]),
              "a+b1/N+b2 f/N+c/snr": np.column_stack([one, 1 / Ncal, f_ / Ncal, 1 / snr]),
              "a+b1/N+b3 f^2+c/snr": np.column_stack([one, 1 / Ncal, f_ ** 2, 1 / snr]),
              "a+b1/N+b2 f/N+b3 f^2+c/snr": np.column_stack([one, 1 / Ncal, f_ / Ncal, f_ ** 2, 1 / snr])}.items():
    print(f"  {fn:28s} CVnll {cvnll(X, r, fold):.1f} coefs {np.round(fit_lin(X, r), 1).tolist()}")
