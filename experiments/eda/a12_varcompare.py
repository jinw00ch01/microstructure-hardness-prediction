"""A12: variance-model comparison with a degradation term, for the blend and single-family OOFs.
v = c0 + b*size_term + c*(1/snr); CV NLL on shared folds (lower is better)."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from eda_common import train_table

t = train_table()
fold = t.fold.values
A = 6.5536
snr = np.clip(t.ic_ridge_snr.values, 0.05, None)
p = np.clip(t.cal_ic_seg_fd91.values, 0.005, 0.995)
Ncal = t.cal_seg_count_density.values * A
size_terms = {
    "none": None,
    "1/N_cal": 1 / Ncal,
    "p(1-p)/N_cal": p * (1 - p) / Ncal,
    "L_gm^2": t.ic_acg_len50_gm.values ** 2,
    "L_par^2": t.ic_acg_len50_par.values ** 2,
    "Lcal_par^2": t.cal_ic_acg_len50_par.values ** 2,
    "Lcal_par*Lcal_perp": t.cal_ic_acg_len50_par.values * t.cal_ic_acg_len50_perp.values,
    "L_gm": t.ic_acg_len50_gm.values,
    "1/sqrt(N_cal)": 1 / np.sqrt(Ncal),
    "1/N_cal^1.5": Ncal ** -1.5,
    "cal_dk_area_mean": t.cal_ic_seg_dk_area_mean.values,
    "cal_mx_area_mean": t.cal_ic_seg_mx_area_mean.values,
}


def nll(v, rr):
    return 0.5 * np.sum(np.log(2 * np.pi * v) + rr ** 2 / v)


def fit(X, rr):
    k = X.shape[1]
    th0 = np.log(np.full(k, np.mean(rr ** 2) / k) / np.maximum(X.mean(0), 1e-9))
    obj = lambda th: nll(X @ np.exp(th), rr)
    res = minimize(obj, th0, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-7, "fatol": 1e-9})
    return np.exp(minimize(obj, res.x, method="BFGS").x)


def cvnll(X, rr):
    s = 0
    for k in range(5):
        w = fit(X[fold != k], rr[fold != k])
        s += nll(X[fold == k] @ w, rr[fold == k])
    return s


resids = {"blend_v2": t.resid.values}
for e in ("feat2_v3_ridge", "emb_effv2s_256_gridge3_aug", "cnn_r18_c224_e30", "feat2_v23cal_lgbs_het"):
    resids[e] = t.hardness.values - t["oof_" + e].values
out = {}
for rn_, rr in resids.items():
    base = cvnll(np.column_stack([np.ones(len(rr)), 1 / snr]), rr)
    row = {}
    for nm, s in size_terms.items():
        X = np.column_stack([np.ones(len(rr)), 1 / snr] + ([s] if s is not None else []))
        row[nm] = base - cvnll(X, rr)
    out[rn_] = row
    print(rn_, "RMSE %.3f" % np.sqrt(np.mean(rr ** 2)), flush=True)
print("CV-NLL gain over [const + 1/snr] (higher = better):")
print(pd.DataFrame(out).round(2).to_string())
