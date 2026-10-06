"""A45: cross-fitted probes (eda_cf, feat5 splmean ridge base; blend_v5 nested residual) for the official factor audit:
(1) robust grain-size estimators, (2) alignment degree / aspect per phase (no absolute angle), (3) size-distribution
width, (4) porosity nonlinearity (B-splines of pore fraction / count / mean size + interactions with phase fraction and
grain size).  Clean-only (snr > 0.9) and all images with an SNR gate g = clip((snr - 0.7)/0.4, 0, 1) for the
watershed-based groups (columns g and g*x); porosity on all images ungated.  20-shuffle permutation nulls.
Also the all-image sign check of the size term: OLS y ~ log d(N_cal) + cal fd + porosity + alignment + aspect.
Usage: python a45_audit_probes.py CACHE_DIR"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer

from eda_cf import Reporter, base, probe, table

cache = Path(sys.argv[1])
t = table("blend_v5")
oof, inner = base(t, kind="feat5s")
R = Reporter(t, oof)
GA = pd.read_parquet(cache / "ga_train.parquet")
t = t.merge(GA, on="ID", how="left")
c = R.clean
y = t.hardness.values
g = np.clip((t.ic_ridge_snr.values - 0.7) / 0.4, 0, 1)


def gated(X):
    X = np.asarray(X, float)
    med = np.nanmedian(X[g > 0], 0)
    X = np.where(np.isfinite(X), X, med)
    return np.column_stack([g[:, None] * X, g])


def run(name, X, modes=("clean", "gated")):
    X = np.asarray(X, float)
    for mode in modes:
        if mode == "clean":
            Xu, kw = X, {"fit_mask": c, "apply_mask": c}
        elif mode == "gated":
            Xu, kw = gated(X), {}
        else:
            Xu, kw = X, {}
        pr = probe(Xu, inner, **kw)
        nul = R.perm_null(Xu, inner, n=20, **kw)
        print(R.line(f"{mode}: {name}", pr, Xu.shape[1]) +
              f" | perm null mean {nul.mean():+.3f} max {nul.max():+.3f} p~{(np.sum(nul >= R.gain(pr)) + 1) / 21:.2f}", flush=True)


cols = lambda rx: [x for x in GA.columns if re.search(rx, x)]  # noqa: E731
print("== (1) robust grain-size estimators")
run("gs robust set (med, trim10, mode, cnt_jeff, cnt_ex10, icpt)", t[["gs_med", "gs_trim10", "gs_mode", "gs_cnt_jeff", "gs_cnt_ex10", "gs_icpt"]])
run("gs_med (log d)", t[["gs_med"]])
run("gs_cnt_ex10 (log d)", t[["gs_cnt_ex10"]])
run("gs robust set, d^-1/2 form", np.exp(-0.5 * t[["gs_med", "gs_trim10", "gs_cnt_jeff", "gs_cnt_ex10"]].values))
run("nominal + area-weighted (cnt_jeff, area_w)", t[["gs_cnt_jeff", "gs_area_w"]])
print("== (2) alignment degree and aspect (no absolute angle)")
run("S number/area/elongation-weighted per phase (9)", t[cols(r"^al_S_")])
run("S + aspect per phase (15)", t[cols(r"^al_")])
run("aspect only per phase (6)", t[cols(r"^al_asp_")])
print("== (3) size-distribution width")
run("width, all 13", t[cols(r"^sw_")])
run("width robust (sd, iqr, gini, top10, bimod, ashD, sd_aw, iqr_aw)",
    t[["sw_sd", "sw_iqr", "sw_gini", "sw_top10_share", "sw_bimod", "sw_ashD", "sw_sd_aw", "sw_iqr_aw"]])
print("== (4) porosity nonlinearity (v3/cal pore columns; all images ungated, and clean)")
pf = t.ic_pore60_frac.values
pn = np.log1p(t.ic_pore60_n.values)
ps = np.log(np.where(t.ic_pore60_n.values > 0, t.ic_pore60_area_mean.fillna(0).values, 0) + 1.0)
pfc = t.cal_ic_pore60_frac.values
fd = t.cal_ic_seg_fd91.values
lN = np.log(t.cal_seg_count_density.values)


def spl(x, k=5):
    x = np.where(np.isfinite(x), x, np.nanmedian(x))[:, None]
    return SplineTransformer(n_knots=k, degree=3, knots="quantile", extrapolation="linear", include_bias=False).fit(x).transform(x)


Sp = np.column_stack([spl(pf), spl(pn), spl(ps), spl(pfc)])
run("pore splines (frac, count, size, cal frac)", Sp, modes=("all", "clean"))
I1 = np.column_stack([pf * fd, pf * lN, pn * fd, pn * lN, pfc * fd, pfc * lN])
run("pore x phase fraction / grain size interactions (6)", I1, modes=("all", "clean"))
run("pore splines + interactions + spline(pf) x fd", np.column_stack([Sp, I1, spl(pf) * fd[:, None]]), modes=("all", "clean"))

print("\n== all-image sign check of the size term (OLS, n=500, bootstrap 90% CI)")
X = np.column_stack([0.5 * np.log(65536.0 / (t.cal_seg_count_density.values * 6.5536)), t.cal_ic_seg_fd91, t.ic_pore68_frac,
                     t.cal_seg_ori_R, t.cal_seg_asp_wmean])
A = np.column_stack([np.ones(len(y)), X])
beta = np.linalg.lstsq(A, y, rcond=None)[0]
rng = np.random.default_rng(0)
bs = np.array([np.linalg.lstsq(A[i], y[i], rcond=None)[0] for i in (rng.integers(0, len(y), len(y)) for _ in range(500))])
for j, nm in enumerate(["log d (N_cal)", "cal fd91", "pore68 frac", "cal ori_R", "cal asp_wmean"]):
    lo, hi = np.quantile(bs[:, j + 1], [0.05, 0.95])
    print(f"  {nm:14s} {beta[j + 1]:+8.2f} [{lo:+.2f}, {hi:+.2f}]")
