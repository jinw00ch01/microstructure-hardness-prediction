"""A31: fold-paired screens of candidate feature files with the saved feature-model setups (src.train_gbm.run, save=False):
  ridge het = feat2_v3cal_ridge_het setup, lgbs het = feat2_v23cal_lgbs_het setup (3 seeds).
Reports CV / folds / N terciles / clean (snr > 0.9) per tercile, and the blend_v2 effect when the two feature members are
swapped for their +candidate versions (blend_v2 weights kept, so this is a lower bound on a re-fitted blend).
Usage: python a31_screen.py "label=file1,file2;label=file3" """
import contextlib
import io
import json
import sys

import numpy as np
import pandas as pd

from eda_common import ROOT, SUB_DIR, rmse, train_table

sys.path.insert(0, str(ROOT))
from src.train_gbm import run  # noqa: E402

DROP = "cal_ic_seg_L_,cal_ic_seg_mx_area_mean,cal_seg_area_cv,cal_segdk_area_cv"
B3 = "features_v3.parquet,features_cal.parquet"
B23 = "features_v3.parquet,features_v2.parquet,features_cal.parquet"
t = train_table()
y = t.hardness.values
fold = t.fold.values
N = t.cal_seg_count_density.values * 6.5536
terc = np.asarray(pd.qcut(N, 3, labels=["coarse", "mid", "fine"]))
clean = t.ic_ridge_snr.values > 0.9
cfg = json.loads((SUB_DIR / "blend_v2.json").read_text())
W = dict(zip(cfg["exps"], cfg["weights"]))


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def summary(oof):
    s = f"CV {rmse(oof, y):.3f} folds {[round(rmse(oof[fold == k], y[fold == k]), 2) for k in range(5)]} |"
    for g in ("coarse", "mid", "fine"):
        m = terc == g
        s += f" {g} {rmse(oof[m], y[m]):.2f}"
    s += " | clean:"
    for g in ("coarse", "mid", "fine"):
        m = (terc == g) & clean
        s += f" {g} {rmse(oof[m], y[m]):.2f}"
    return s


cands = {"base": ""}
for part in sys.argv[1].split(";"):
    k, v = part.split("=", 1)
    cands[k] = "," + v
res = {}
for lab, extra in cands.items():
    r = quiet(run, "ridge", "x", feat_file=B3 + extra, drop=DROP, hetero="ic_acg_len50_gm", save=False)
    l = quiet(run, "lgbs", "x", feat_file=B23 + extra, seeds=3, drop=DROP, hetero="ic_acg_len50_gm", save=False)
    res[lab] = (r, l)
    print(f"[{lab}] ridge het {summary(r)}", flush=True)
    print(f"[{lab}] lgbs  het {summary(l)}", flush=True)
# blend effect: swap the two feature members (weights of blend_v2 kept)
others = sum(W[e] * t["oof_" + e].values for e in cfg["exps"] if e not in ("feat2_v3cal_ridge_het", "feat2_v23cal_lgbs_het"))
for lab, (r, l) in res.items():
    b = others + W["feat2_v3cal_ridge_het"] * r + W["feat2_v23cal_lgbs_het"] * l
    print(f"[{lab}] blend_v2 weights with swapped members: {summary(b)}")
    if lab != "base":
        rb, lb = res["base"]
        b0 = others + W["feat2_v3cal_ridge_het"] * rb + W["feat2_v23cal_lgbs_het"] * lb
        d = [rmse(b[fold == k], y[fold == k]) - rmse(b0[fold == k], y[fold == k]) for k in range(5)]
        print(f"      fold-paired blend delta {np.round(d, 3).tolist()} mean {np.mean(d):+.3f}")
