"""A29: spatial heterogeneity of grain size from the watershed grains (meaningful on clean images only).
Per image: Moran's I of log grain area over grain neighbours within radius R (R = 24, 48 px), the std of the local mean
log-area (Gaussian-weighted, sigma 32 px) across the image relative to the global log-area std, and the same for the
dark-phase indicator.  Usage: python a29_size_moran.py CACHE_DIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from eda_common import train_table

cache = Path(sys.argv[1])
g = pd.read_parquet(cache / "grains_train.parquet")
g = g[g.area >= 12]
rows = []
for i, d in g.groupby("ID"):
    if len(d) < 15:
        rows.append({"ID": i})
        continue
    P = d[["cy", "cx"]].values
    la = np.log(d.area.values)
    dk = (d.med.values < 0.91).astype(float)
    tree = cKDTree(P)
    f = {"ID": i}
    for R in (24, 48):
        pairs = tree.query_pairs(R, output_type="ndarray")
        for nm, v in (("la", la), ("dk", dk)):
            z = v - v.mean()
            if len(pairs) == 0 or z.std() == 0:
                f[f"mi_{nm}_{R}"] = np.nan
                continue
            num = 2 * np.sum(z[pairs[:, 0]] * z[pairs[:, 1]])
            f[f"mi_{nm}_{R}"] = float(len(v) / (2 * len(pairs)) * num / np.sum(z ** 2))
    # local mean log-area field
    yy, xx = np.mgrid[16:256:32, 16:256:32]
    G = np.column_stack([yy.ravel(), xx.ravel()])
    D2 = ((G[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    W = np.exp(-D2 / (2 * 32.0 ** 2)) * d.area.values[None, :]
    loc = (W * la[None, :]).sum(1) / W.sum(1)
    locd = (W * dk[None, :]).sum(1) / W.sum(1)
    f["lf_la_sd"] = float(loc.std())
    f["lf_la_sd_rel"] = float(loc.std() / max(la.std(), 1e-6))
    f["lf_dk_sd"] = float(locd.std())
    f["lf_la_dk_corr"] = float(np.corrcoef(loc, locd)[0, 1]) if loc.std() > 0 and locd.std() > 0 else 0.0
    f["lf_la_range"] = float(loc.max() - loc.min())
    rows.append(f)
M = pd.DataFrame(rows)
M.to_parquet(cache / "mi_train.parquet")
t = train_table().merge(M, on="ID", how="left")
N = t.cal_seg_count_density * 6.5536
t["terc"] = pd.qcut(N, 3, labels=["coarse", "mid", "fine"])
for nm, m0 in (("snr>0.9", t.ic_ridge_snr > 0.9), ("snr>1.2", t.ic_ridge_snr > 1.2)):
    print("==", nm, int(m0.sum()))
    for c in [c for c in M.columns if c != "ID"]:
        out = [f"{g_} y{spearmanr(t[c][m0 & (t.terc == g_)], t.hardness[m0 & (t.terc == g_)], nan_policy='omit')[0]:+.2f} r{spearmanr(t[c][m0 & (t.terc == g_)], t.resid[m0 & (t.terc == g_)], nan_policy='omit')[0]:+.2f}" for g_ in ("coarse", "mid", "fine")]
        print(f"{c:14s} all y{spearmanr(t[c][m0], t.hardness[m0], nan_policy='omit')[0]:+.3f} r{spearmanr(t[c][m0], t.resid[m0], nan_policy='omit')[0]:+.3f} | " + " | ".join(out))
