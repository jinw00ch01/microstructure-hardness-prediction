"""A30: spatial grain-size field features (fixed per-image transform; train or test).
Grains = cached sato-ridge watershed regions (same as v3 / a2), interior median of rn, non-pore (pore share <= 0.5),
area >= 12 px.  Local mean log grain area on an 8x8 grid of centres (every 32 px, offset 16): Gaussian weights
exp(-r^2 / 2 s^2) times grain area, s = 32 px (and 48).  Exports
  lf_la_n, lf_la_w, lf_la_sd           global number-mean / area-weighted mean / std of log area
  lf{s}_sd, lf{s}_rng, lf{s}_max, lf{s}_min, lf{s}_sd_rel, lf{s}_max_minus_w   local-field heterogeneity / extremes
  lf_dk{s}_sd                          same field for the dark-grain indicator
  lf_mi_la_48                          Moran's I of log area over grain pairs closer than 48 px
  lf_n                                 number of grains used (segmentation sanity)
Only meaningful where the watershed works (ic_ridge_snr >~ 0.9); GBMs can gate on ic_ridge_snr.
Usage: python a30_lf_feats.py CACHE_DIR split OUT_PARQUET"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from eda_common import DATA_DIR

cache, split, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
ids = pd.read_csv(DATA_DIR / ("train.csv" if split == "train" else "sample_submission.csv")).ID.tolist()
rn_all = np.load(cache / f"rn_{split}.npy", mmap_mode="r")
ws_all = np.load(cache / f"ws_{split}.npy", mmap_mode="r")
yy, xx = np.mgrid[16:256:32, 16:256:32]
GRID = np.column_stack([yy.ravel(), xx.ravel()]).astype(float)
rows = []
for k, i in enumerate(ids):
    rn = rn_all[k].astype(np.float32)
    w = np.asarray(ws_all[k]).astype(np.int32)
    f = {"ID": i}
    nl = int(w.max())
    if nl < 3:
        f["lf_n"] = nl
        rows.append(f)
        continue
    lab = np.arange(1, nl + 1)
    s1 = cv2.GaussianBlur(rn, (0, 0), 1.0)
    pm = ndi.binary_opening(s1 < 0.60, structure=np.ones((3, 3)))
    inner = w.copy()
    inner[ndi.binary_dilation(w == 0)] = 0
    area = np.asarray(ndi.sum(np.ones_like(rn), w, lab))
    a_in = np.asarray(ndi.sum(np.ones_like(rn), inner, lab))
    med = np.asarray(ndi.median(rn, inner, lab))
    pf = np.asarray(ndi.mean(pm.astype(np.float32), w, lab))
    com = np.array(ndi.center_of_mass(np.ones_like(rn), w, lab))
    ok = (area >= 12) & (a_in >= 8) & (pf <= 0.5) & np.isfinite(med)
    f["lf_n"] = int(ok.sum())
    if ok.sum() < 8:
        rows.append(f)
        continue
    A, la, dk, P = area[ok], np.log(area[ok]), (med[ok] < 0.91).astype(float), com[ok]
    f["lf_la_n"] = float(la.mean())
    f["lf_la_w"] = float(np.sum(A * la) / A.sum())
    f["lf_la_sd"] = float(la.std())
    D2 = ((GRID[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    for s in (32, 48):
        Wt = np.exp(-D2 / (2.0 * s * s)) * A[None, :]
        loc = (Wt * la[None, :]).sum(1) / Wt.sum(1)
        locd = (Wt * dk[None, :]).sum(1) / Wt.sum(1)
        f[f"lf{s}_sd"] = float(loc.std())
        f[f"lf{s}_rng"] = float(loc.max() - loc.min())
        f[f"lf{s}_max"] = float(loc.max())
        f[f"lf{s}_min"] = float(loc.min())
        f[f"lf{s}_sd_rel"] = float(loc.std() / max(la.std(), 1e-6))
        f[f"lf{s}_max_minus_w"] = float(loc.max() - f["lf_la_w"])
        f[f"lf_dk{s}_sd"] = float(locd.std())
    tree = cKDTree(P)
    pairs = tree.query_pairs(48, output_type="ndarray")
    z = la - la.mean()
    f["lf_mi_la_48"] = float(len(la) / (2 * len(pairs)) * 2 * np.sum(z[pairs[:, 0]] * z[pairs[:, 1]]) / np.sum(z ** 2)) if len(pairs) and z.std() > 0 else np.nan
    rows.append(f)
pd.DataFrame(rows).to_parquet(out)
print("done", split, len(rows))
