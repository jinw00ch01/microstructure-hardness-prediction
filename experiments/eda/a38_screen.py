"""A38: fold-paired screens on the blend_v4 feature members (src.train_gbm.run, save=False):
  feat4 ridge = feat4_v3cal_ridge_het_spat_v4 setup;  feat3 lgbs = feat3_v23cal_lgbs_het_spat setup (3 seeds).
Reports CV / folds / N terciles / clean (snr > 0.9) terciles, and the blend effect when both members are swapped for
their +candidate versions: (i) blend_v4 weights kept, (ii) honest nested NNLS re-fit over all blend_v4 experiments.
Usage: python a38_screen.py "label=file1,file2;label=file3"   (files in data/)"""
import contextlib
import io
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from eda_cf import F3L_FILES, F4_DROP, F4_FILES, D1
from eda_common import ROOT, SUB_DIR, rmse, train_table

sys.path.insert(0, str(ROOT))
from src.train_gbm import run  # noqa: E402

M4, M3 = "feat4_v3cal_ridge_het_spat_v4", "feat3_v23cal_lgbs_het_spat"
cfg = json.loads((SUB_DIR / "blend_v4.json").read_text())
EXPS, W = cfg["exps"], dict(zip(cfg["exps"], cfg["weights"]))
from eda_common import blend_oof  # noqa: E402

t = blend_oof("blend_v4").merge(train_table()[["ID", "cal_seg_count_density", "ic_ridge_snr"]], on="ID")
y, fold = t.hardness.values, t.fold.values
N = t.cal_seg_count_density.values * 6.5536
terc = np.asarray(pd.qcut(N, 3, labels=["coarse", "mid", "fine"]))
clean = t.ic_ridge_snr.values > 0.9


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def summary(oof):
    s = f"CV {rmse(oof, y):.3f} folds {[round(rmse(oof[fold == k], y[fold == k]), 2) for k in range(5)]} |"
    s += "".join(f" {g} {rmse(oof[terc == g], y[terc == g]):.2f}" for g in ("coarse", "mid", "fine"))
    s += " | clean:" + "".join(f" {g} {rmse(oof[(terc == g) & clean], y[(terc == g) & clean]):.2f}" for g in ("coarse", "mid", "fine"))
    return s


def fit_w(P, yy):
    w, _ = nnls(P, yy)
    return w / w.sum()


def nested_blend(P):
    out = np.zeros(len(y))
    for k in range(5):
        m = fold == k
        out[m] = P[m] @ fit_w(P[~m], y[~m])
    return out


cands = {"base": ""}
for part in sys.argv[1].split(";"):
    k, v = part.split("=", 1)
    cands[k] = "," + v
res = {}
for lab, extra in cands.items():
    r = quiet(run, "ridge", "x", feat_file=F4_FILES + extra, drop=F4_DROP, hetero="ic_acg_len50_gm", save=False)
    l = quiet(run, "lgbs", "x", feat_file=F3L_FILES + extra, seeds=3, drop=D1, hetero="ic_acg_len50_gm", save=False)
    res[lab] = (r, l)
    print(f"[{lab}] feat4 ridge {summary(r)}", flush=True)
    print(f"[{lab}] feat3 lgbs  {summary(l)}", flush=True)
P0 = np.column_stack([t["oof_" + e].values for e in EXPS])
others = sum(W[e] * t["oof_" + e].values for e in EXPS if e not in (M4, M3))
i4, i3 = EXPS.index(M4), EXPS.index(M3)
for lab, (r, l) in res.items():
    bw = others + W[M4] * r + W[M3] * l
    P = P0.copy()
    P[:, i4], P[:, i3] = r, l
    bn = nested_blend(P)
    print(f"[{lab}] blend_v4 weights, swapped members: {summary(bw)}")
    print(f"[{lab}] nested NNLS re-fit, swapped members: {summary(bn)}")
    if lab != "base":
        rb, lb = res["base"]
        b0w = others + W[M4] * rb + W[M3] * lb
        P = P0.copy()
        P[:, i4], P[:, i3] = rb, lb
        b0n = nested_blend(P)
        for nm, b, b0 in (("fixed weights", bw, b0w), ("nested re-fit", bn, b0n)):
            d = [rmse(b[fold == k], y[fold == k]) - rmse(b0[fold == k], y[fold == k]) for k in range(5)]
            print(f"      {nm}: fold-paired blend delta {np.round(d, 3).tolist()} mean {np.mean(d):+.3f} ({sum(x < 0 for x in d)}/5 better)")
