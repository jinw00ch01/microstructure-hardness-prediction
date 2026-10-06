"""A44: which grain-size estimator fits hardness best on clean images, and does the Hall-Petch sign come back?
Physics-style OLS on clean images (ic_ridge_snr > 0.9; shared folds restricted to those rows):
  y ~ 1 + g(d) + fd + porosity + alignment (+ aspect),  g(d) = d^-1/2 = exp(-logd/2)  or  log d
fd = ic_seg_fd91, porosity = ic_pore68_frac, alignment = al_S_e_all (elongation-weighted order parameter),
aspect = al_asp_a_all.  For each estimator (a43 gs_*, plus calibrated / v4 / autocorrelation references) reports the
CV RMSE, the size coefficient (full fit) with a 90% bootstrap CI, and the direction: 'finer=harder' (Hall-Petch) or
'coarser=harder'.  Then joint fits with a count-based (nominal) estimator + the area-weighted estimator.
Usage: python a44_hallpetch.py CACHE_DIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from eda_cf import table
from eda_common import load_feats, rmse

cache = Path(sys.argv[1])
t = table("blend_v5").merge(pd.read_parquet(cache / "ga_train.parquet"), on="ID", how="left")
t = t.merge(load_feats(("features_v2.parquet",))[["ID", "seg_count_density"]], on="ID", how="left")
t["ref_Ncal"] = 0.5 * np.log(65536.0 / (t.cal_seg_count_density * 6.5536))
t["ref_v4c_la"] = 0.5 * t.v4c_la_mean
t["ref_acg_gm"] = np.log(t.ic_acg_len50_gm)
c = (t.ic_ridge_snr > 0.9) & t.gs_cnt_jeff.notna()
d = t[c].reset_index(drop=True)
y, fold = d.hardness.values, d.fold.values
N = d.cal_seg_count_density.values * 6.5536
terc = np.asarray(pd.qcut(t.cal_seg_count_density * 6.5536, 3, labels=["coarse", "mid", "fine"]))[c.values]
print(f"clean images n={len(d)}; y sd {y.std():.2f}; blend_v5 nested RMSE on these {rmse(d.pred, y):.2f}")
COV = ["ic_seg_fd91", "ic_pore68_frac", "al_S_e_all", "al_asp_a_all"]
EST = ["gs_cnt_jeff", "gs_cnt_raw", "gs_cnt_ex10", "gs_cnt_ex20", "gs_med", "gs_mode", "gs_trim10", "gs_trim20",
       "gs_trimA10", "gs_num_mean", "gs_icpt", "gs_icpt_par", "gs_icpt_perp", "gs_area_w", "ref_Ncal", "ref_v4c_la", "ref_acg_gm"]


def ols(X, yy):
    A = np.column_stack([np.ones(len(yy)), X])
    return np.linalg.lstsq(A, yy, rcond=None)[0]


def cv(X):
    o = np.zeros(len(y))
    for k in range(5):
        a, b = fold != k, fold == k
        beta = ols(X[a], y[a])
        o[b] = np.column_stack([np.ones(b.sum()), X[b]]) @ beta
    return o


def boot_ci(X, j, n=500):
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        bs.append(ols(X[i], y[i])[1 + j])
    return np.quantile(bs, [0.05, 0.95])


Xc = d[COV].astype(float).fillna(d[COV].median()).values
o0 = cv(Xc)
print(f"no size term (fd + porosity + S + aspect): CV {rmse(o0, y):.2f}")
print(f"\n{'estimator':14s} {'rho(y,logd)':>11s} {'form':>6s} {'CV':>6s}  coef [90% CI]  direction   | coarse / mid / fine CV")
rows = []
for e in EST:
    ld = d[e].astype(float).values
    r0 = spearmanr(ld, y)[0]
    for form, g in (("d^-1/2", np.exp(-0.5 * ld)), ("log d", ld)):
        X = np.column_stack([g, Xc])
        o = cv(X)
        beta = ols(X, y)[1]
        lo, hi = boot_ci(X, 0)
        finer_harder = beta > 0 if form == "d^-1/2" else beta < 0
        sig = "" if lo * hi > 0 else " (CI spans 0)"
        rows.append({"est": e, "form": form, "cv": rmse(o, y), "coef": beta, "lo": lo, "hi": hi, "dir": "finer=harder" if finer_harder else "coarser=harder"})
        print(f"{e:14s} {r0:+11.2f} {form:>6s} {rmse(o, y):6.2f}  {beta:+8.2f} [{lo:+.2f}, {hi:+.2f}] {rows[-1]['dir']}{sig} | " +
              " / ".join(f"{rmse(o[terc == g_], y[terc == g_]):.2f}" for g_ in ("coarse", "mid", "fine")))
pd.DataFrame(rows).to_csv(Path(__file__).parent / "a44_hallpetch_clean.csv", index=False)

print("\nJoint fits: nominal (count-based) estimator + area-weighted estimator (log d forms) + covariates:")
for nom in ("gs_cnt_jeff", "gs_cnt_ex10", "gs_med", "ref_Ncal"):
    for big in ("gs_area_w", "sw_sd", "sw_top10_share"):
        X = np.column_stack([d[nom].values, d[big].values, Xc])
        o = cv(X)
        b = ols(X, y)
        ci1, ci2 = boot_ci(X, 0), boot_ci(X, 1)
        print(f"  {nom:12s} + {big:15s} CV {rmse(o, y):.2f} | coef log d_nominal {b[1]:+7.2f} [{ci1[0]:+.2f},{ci1[1]:+.2f}] "
              f"({'finer=harder' if b[1] < 0 else 'coarser=harder'}) | coef {big} {b[2]:+7.2f} [{ci2[0]:+.2f},{ci2[1]:+.2f}]")

print("\nRaw Spearman with y on clean images (size: positive = coarser is harder):")
for e in ["gs_cnt_jeff", "gs_med", "gs_area_w", "ref_Ncal", "ic_seg_fd91", "ic_pore68_frac", "al_S_e_all", "al_asp_a_all", "sw_sd"]:
    print(f"  {e:16s} rho(y) {spearmanr(d[e], y, nan_policy='omit')[0]:+.2f} | rho with ic_seg_fd91 {spearmanr(d[e], d.ic_seg_fd91, nan_policy='omit')[0]:+.2f}")
allr = t.cal_seg_count_density.notna()
print(f"all images: rho(y, log N_cal) {spearmanr(np.log(t.cal_seg_count_density[allr]), t.hardness[allr])[0]:+.2f} "
      f"(negative = coarser is harder); rho(y, cal_ic_seg_fd91) {spearmanr(t.cal_ic_seg_fd91, t.hardness)[0]:+.2f}")
