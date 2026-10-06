"""A20: parametric physics-form fits on clean images (shared folds, fitted on the clean training rows of each fold).
Inputs: robust per-grain measurements (a18 rp_*: area / number dark fraction, per-phase grain size, pores, aspect) and
v3/cal measurements.  Forms: rule of mixtures (area or number fraction) x per-phase Hall-Petch, x porosity (linear / exp /
power), + aspect terms.  Compared per N tercile with the blend_v2 OOF on the same images.
Usage: python a20_physics_fit.py CACHE_DIR [snr_min]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import RidgeCV

from eda_common import OUT, rmse, train_table

cache = Path(sys.argv[1])
snr_min = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
t = train_table().merge(pd.read_parquet(cache / "rp_train.parquet"), on="ID")
N = t.cal_seg_count_density.values * 6.5536
t["terc"] = np.asarray(pd.qcut(N, 3, labels=["coarse", "mid", "fine"]))
c = t[(t.ic_ridge_snr > snr_min)].copy().reset_index(drop=True)
c = c.dropna(subset=["rp_fd90_area", "rp_d_mx", "rp_pore_frac", "rp_logasp_mx"]).reset_index(drop=True)
c["rp_d_dk"] = c.rp_d_dk.fillna(c.rp_d_mx)
y = c.hardness.values
fold = c.fold.values
fa, fn = c.rp_fd90_area.values, c.rp_fd90_num.values
dm, dd = c.rp_d_mx.values, c.rp_d_dk.values
dall = c.rp_d_all.values
phi = c.rp_pore_frac.values
la = np.nan_to_num(np.where(np.isfinite(c.rp_logasp_mx), c.rp_logasp_mx, 0.3))
lad = np.nan_to_num(np.where(np.isfinite(c.rp_logasp_dk), c.rp_logasp_dk, 0.3))
print(f"clean (snr>{snr_min}) n={len(c)}; blend RMSE on these {rmse(c.pred, y):.3f}")


def forms():
    F = {}
    F["mix_area"] = (lambda p: (1 - fa) * p[0] + fa * p[1], [180, 260])
    F["mix_num"] = (lambda p: (1 - fn) * p[0] + fn * p[1], [180, 260])
    F["mix_area+HP_common"] = (lambda p: (1 - fa) * p[0] + fa * p[1] + p[2] / np.sqrt(dall), [180, 260, 0])
    F["mix_area x HP_phase"] = (lambda p: (1 - fa) * (p[0] + p[2] / np.sqrt(dm)) + fa * (p[1] + p[3] / np.sqrt(dd)), [180, 260, 0, 0])
    F["mix_area x HP_phase(lin d)"] = (lambda p: (1 - fa) * (p[0] + p[2] * dm) + fa * (p[1] + p[3] * dd), [180, 260, 0, 0])
    F["mix_area x HP_phase x (1-c phi)"] = (lambda p: ((1 - fa) * (p[0] + p[2] / np.sqrt(dm)) + fa * (p[1] + p[3] / np.sqrt(dd))) * (1 - p[4] * phi), [180, 260, 0, 0, 0])
    F["mix_area x HP_phase x exp(-b phi)"] = (lambda p: ((1 - fa) * (p[0] + p[2] / np.sqrt(dm)) + fa * (p[1] + p[3] / np.sqrt(dd))) * np.exp(-p[4] * phi), [180, 260, 0, 0, 0])
    F["mix_area x (1-phi)^n"] = (lambda p: ((1 - fa) * p[0] + fa * p[1]) * np.clip(1 - phi, 1e-3, 1) ** p[2], [180, 260, 1])
    F["full: mix x HP_phase(lin d) x exp + aspect"] = (
        lambda p: ((1 - fa) * (p[0] + p[2] * dm + p[5] * la) + fa * (p[1] + p[3] * dd + p[6] * lad)) * np.exp(-p[4] * phi),
        [180, 260, 0, 0, 0, 0, 0])
    F["full num: mix_num x HP_phase(lin d) x exp + aspect"] = (
        lambda p: ((1 - fn) * (p[0] + p[2] * dm + p[5] * la) + fn * (p[1] + p[3] * dd + p[6] * lad)) * np.exp(-p[4] * phi),
        [180, 260, 0, 0, 0, 0, 0])
    return F


rows = []
for name, (fun, p0) in forms().items():
    oof = np.zeros(len(y))
    for k in range(5):
        tr_, te_ = fold != k, fold == k
        res = least_squares(lambda p: (fun(p) - y)[tr_], p0, loss="soft_l1", f_scale=20.0, max_nfev=20000)
        oof[te_] = fun(res.x)[te_]
    full = least_squares(lambda p: fun(p) - y, p0, loss="soft_l1", f_scale=20.0, max_nfev=20000).x
    row = {"form": name, "cv_rmse": rmse(oof, y)}
    for g in ("coarse", "mid", "fine"):
        m = c.terc.values == g
        row[f"rmse_{g}"] = rmse(oof[m], y[m])
        row[f"blend_{g}"] = rmse(c.pred.values[m], y[m])
        row[f"corr_resid_{g}"] = np.corrcoef(oof[m] - c.pred.values[m], c.resid.values[m])[0, 1]
    row["params"] = np.round(full, 2).tolist()
    rows.append(row)
    print(f"{name:52s} CV {row['cv_rmse']:.2f} | coarse {row['rmse_coarse']:.2f} (blend {row['blend_coarse']:.2f}) mid {row['rmse_mid']:.2f} "
          f"({row['blend_mid']:.2f}) fine {row['rmse_fine']:.2f} ({row['blend_fine']:.2f}) | p={row['params']}", flush=True)
# non-parametric reference on the same compact inputs + blend correction
X = np.column_stack([fa, fn, np.log(dm), np.log(dd), phi, la, lad, np.log(N[t.ic_ridge_snr > snr_min][:len(c)]) if False else np.log(c.cal_seg_count_density)])
oof = np.zeros(len(y))
for k in range(5):
    tr_, te_ = fold != k, fold == k
    mu, sd = X[tr_].mean(0), X[tr_].std(0)
    oof[te_] = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit((X[tr_] - mu) / sd, y[tr_]).predict((X[te_] - mu) / sd)
print(f"ridge on the same compact inputs: CV {rmse(oof, y):.2f}; by tercile " + " ".join(f"{g} {rmse(oof[c.terc.values == g], y[c.terc.values == g]):.2f}" for g in ("coarse", "mid", "fine")))
pd.DataFrame(rows).to_csv(OUT / f"a20_physics_fit_snr{snr_min}.csv", index=False)
