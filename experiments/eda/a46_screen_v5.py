"""A46: fold-paired screen on the blend_v5 feature members (src.train_gbm.run, save=False):
  feat5 splmean ridge (feat4 setup + MIL splmean design), feat5 milspl ridge (+ MIL spl design), feat3 lgbs (3 seeds).
Blend: nested NNLS re-fit over all blend_v5 experiments with the three members swapped for their +candidate versions,
and blend_v5 weights kept.  Usage: python a46_screen_v5.py "label=file[,file]"   (files in data/)"""
import contextlib
import io
import json
import sys

import numpy as np
from scipy.optimize import nnls

from eda_cf import D1, F3L_FILES, F4_DROP, F4_FILES
from eda_common import ROOT, SUB_DIR, blend_oof, rmse

sys.path.insert(0, str(ROOT))
from src.train_gbm import mil_design_fold_extra, run  # noqa: E402

cfg = json.loads((SUB_DIR / "blend_v5.json").read_text())
EXPS, W = cfg["exps"], dict(zip(cfg["exps"], cfg["weights"]))
t = blend_oof("blend_v5")
y, fold = t.hardness.values, t.fold.values
FE = {"splmean": mil_design_fold_extra(kind="splmean"), "spl": mil_design_fold_extra(kind="spl")}
MEM = {"feat5_v3cal_ridge_het_spat_v4_splmean": lambda x: run("ridge", "x", feat_file=F4_FILES + x, drop=F4_DROP, hetero="ic_acg_len50_gm", save=False, fold_extra=FE["splmean"]),
       "feat5_v3cal_ridge_het_spat_v4_milspl": lambda x: run("ridge", "x", feat_file=F4_FILES + x, drop=F4_DROP, hetero="ic_acg_len50_gm", save=False, fold_extra=FE["spl"]),
       "feat3_v23cal_lgbs_het_spat": lambda x: run("lgbs", "x", feat_file=F3L_FILES + x, seeds=3, drop=D1, hetero="ic_acg_len50_gm", save=False)}


def fr(p):
    return np.array([rmse(p[fold == k], y[fold == k]) for k in range(5)])


def nested(P):
    o = np.zeros(len(y))
    for k in range(5):
        m = fold == k
        w, _ = nnls(P[~m], y[~m])
        o[m] = P[m] @ (w / w.sum())
    return o


cands = {"base": ""}
for part in sys.argv[1].split(";"):
    k, v = part.split("=", 1)
    cands[k] = "," + v
res = {}
for lab, extra in cands.items():
    res[lab] = {}
    for e, f in MEM.items():
        with contextlib.redirect_stdout(io.StringIO()):
            res[lab][e] = f(extra)
        p = res[lab][e]
        s = f"[{lab}] {e[:40]:40s} CV {rmse(p, y):.3f}"
        if lab != "base":
            d = fr(p) - fr(res["base"][e])
            s += f" | fold deltas {np.round(d, 3).tolist()} ({int((d < 0).sum())}/5 better)"
        print(s, flush=True)
P0 = np.column_stack([t["oof_" + e].values for e in EXPS])
for lab in cands:
    P = P0.copy()
    for e, p in res[lab].items():
        P[:, EXPS.index(e)] = p
    res[lab]["_nested"] = nested(P)
    res[lab]["_fixed"] = P @ np.array([W[e] for e in EXPS])
for lab in cands:
    if lab == "base":
        print(f"[base] blend nested {rmse(res['base']['_nested'], y):.3f} fixed-weights {rmse(res['base']['_fixed'], y):.3f}")
        continue
    for kind in ("_nested", "_fixed"):
        d = fr(res[lab][kind]) - fr(res["base"][kind])
        print(f"[{lab}] blend{kind} {rmse(res['base'][kind], y):.3f} -> {rmse(res[lab][kind], y):.3f} | fold deltas "
              f"{np.round(d, 3).tolist()} ({int((d < 0).sum())}/5 better)")
