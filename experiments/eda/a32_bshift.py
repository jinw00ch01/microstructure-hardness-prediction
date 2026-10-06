"""A32: does the new signal shrink the 1/N variance term?  Refit v = a + b/N + c/snr on the residuals of the blend_v2-weighted
combination with the two feature members swapped for their +candidate versions (same as a31)."""
import contextlib, io, json, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from eda_common import ROOT, SUB_DIR, rmse, train_table
sys.path.insert(0, str(ROOT))
from src.train_gbm import run  # noqa: E402

DROP = "cal_ic_seg_L_,cal_ic_seg_mx_area_mean,cal_seg_area_cv,cal_segdk_area_cv"
t = train_table(); y = t.hardness.values
N = t.cal_seg_count_density.values * 6.5536; snr = np.clip(t.ic_ridge_snr.values, 0.05, None)
cfg = json.loads((SUB_DIR / "blend_v2.json").read_text()); W = dict(zip(cfg["exps"], cfg["weights"]))
others = sum(W[e] * t["oof_" + e].values for e in cfg["exps"] if e not in ("feat2_v3cal_ridge_het", "feat2_v23cal_lgbs_het"))
X = np.column_stack([np.ones(len(y)), 1 / N, 1 / snr])
def nll(v, rr): return 0.5 * np.sum(np.log(2 * np.pi * v) + rr ** 2 / v)
def fit(rr):
    obj = lambda th: nll(X @ np.exp(th), rr)
    r0 = minimize(obj, np.log([20, 2e4, 10]), method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-8, "fatol": 1e-10})
    return np.exp(minimize(obj, r0.x, method="BFGS").x)
for lab, extra in (("base", ""), ("ecs+lf+lledge", ",eda_feats_ecs.parquet,eda_feats_lf.parquet,eda_feats_lledge.parquet")):
    with contextlib.redirect_stdout(io.StringIO()):
        r = run("ridge", "x", feat_file="features_v3.parquet,features_cal.parquet" + extra, drop=DROP, hetero="ic_acg_len50_gm", save=False)
        l = run("lgbs", "x", feat_file="features_v3.parquet,features_v2.parquet,features_cal.parquet" + extra, seeds=3, drop=DROP, hetero="ic_acg_len50_gm", save=False)
    b = others + W["feat2_v3cal_ridge_het"] * r + W["feat2_v23cal_lgbs_het"] * l
    res = y - b
    a_, b_, c_ = fit(res)
    print(f"[{lab}] RMSE {rmse(b, y):.3f}: a={a_:.1f} b={b_:.0f} c={c_:.2f} | mean parts b/N {np.mean(b_ / N):.1f} c/snr {np.mean(c_ / snr):.1f} a {a_:.1f}")
