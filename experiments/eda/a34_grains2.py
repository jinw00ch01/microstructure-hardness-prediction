"""A34: extended per-grain table + per-image multiple-instance (MI) features.  Fixed per-image transform (train or test).
Grains = cached watershed regions (a2: NLM, local matrix level, sato ridge, h-minima watershed), same rules as a4:
interior (boundary-dilated) area >= 8, pore share <= 0.5; here additionally area >= 12.
Per grain: area, centroid, moment axes (maj, min), orientation, solidity, perimeter, interior median / IQR of rn, border.
Phase: dark = interior median rn < 0.91 (as a24/a29/a30).  Per image:
  mi_a_{dk,mx}_{b}   area fraction of the image's grain area in phase x log-area bin b (8 bins, edges 12..3200 px)
                     -> a linear model on these = area-weighted mean of a piecewise-constant per-grain f(size, phase)
  mi_n_{dk,mx}_{b}   the same, number fractions
  mi_s_{dk,mx}_{k}   area-weighted mean of z^k (k = 0..3), z = (log area - 5); smooth version of the bins
  mi_g_{mx,dk}_{b}   area fraction in per-grain grey-level bins (interior median rn), per phase
  mi_e_{dk,mx}_*     elongation: area-weighted mean log aspect (maj/min), area share with aspect > 2 / > 3, solidity
  mi_N, mi_Neff, mi_log_neff_over_n, mi_Neff_{dk,mx}   watershed grain number and effective number (sum a)^2/sum a^2
Writes CACHE/grains2_{split}.parquet and CACHE/mig_{split}.parquet.  Usage: python a34_grains2.py CACHE_DIR split"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure

from eda_common import DATA_DIR

cache, split = Path(sys.argv[1]), sys.argv[2]
ids = pd.read_csv(DATA_DIR / ("train.csv" if split == "train" else "sample_submission.csv")).ID.tolist()
rn_all = np.load(cache / f"rn_{split}.npy", mmap_mode="r")
ws_all = np.load(cache / f"ws_{split}.npy", mmap_mode="r")
SEDGES = np.log([12, 50, 100, 200, 400, 800, 1600, 3200, 1e7])
GEDGES = {"mx": [0.91, 0.95, 0.98, 1.01, 1.04, 9.0], "dk": [0.0, 0.80, 0.86, 0.91]}
grains, rows = [], []
for k, i in enumerate(ids):
    rn = rn_all[k].astype(np.float32)
    w = np.asarray(ws_all[k]).astype(np.int32)
    s1 = cv2.GaussianBlur(rn, (0, 0), 1.0)
    pm = ndi.binary_opening(s1 < 0.60, structure=np.ones((3, 3)))
    inner = w.copy()
    inner[ndi.binary_dilation(w == 0)] = 0
    nl = int(w.max())
    f = {"ID": i}
    if nl < 3:
        rows.append(f)
        continue
    idx = np.arange(1, nl + 1)
    med = np.asarray(ndi.median(rn, inner, idx))

    a_in = np.asarray(ndi.sum(np.ones_like(rn), inner, idx))
    pf = np.asarray(ndi.mean(pm.astype(np.float32), w, idx))
    border = np.zeros(nl + 1, bool)
    border[np.unique(np.concatenate([w[0], w[-1], w[:, 0], w[:, -1]]))] = True
    props = measure.regionprops(w, intensity_image=np.where(inner > 0, rn, np.nan).astype(np.float32))
    G = []
    for p_ in props:
        j = p_.label
        if a_in[j - 1] < 8 or pf[j - 1] > 0.5 or p_.area < 12 or not np.isfinite(med[j - 1]):
            continue
        vals = p_.image_intensity[p_.image & np.isfinite(p_.image_intensity)]
        iqr = float(np.subtract(*np.percentile(vals, [75, 25]))) if len(vals) >= 4 else np.nan
        G.append((j, p_.area, p_.centroid[0], p_.centroid[1], p_.axis_major_length, max(p_.axis_minor_length, 1.0),
                  p_.orientation, p_.solidity, p_.perimeter, med[j - 1], iqr, border[j]))
    G = pd.DataFrame(G, columns=["label", "area", "cy", "cx", "maj", "min", "ori", "solidity", "perim", "med", "iqr", "border"])
    G.insert(0, "ID", i)
    grains.append(G)
    if len(G) < 8:
        f["mi_N"] = len(G)
        rows.append(f)
        continue
    A = G.area.values.astype(float)
    la = np.log(A)
    dk = G.med.values < 0.91
    tot = A.sum()
    sb = np.digitize(la, SEDGES[1:-1])
    asp = np.log(G["maj"].values / G["min"].values)
    for ph, m in (("dk", dk), ("mx", ~dk)):
        for b in range(len(SEDGES) - 1):
            mm = m & (sb == b)
            f[f"mi_a_{ph}_{b}"] = float(A[mm].sum() / tot)
            f[f"mi_n_{ph}_{b}"] = float(mm.sum() / len(A))
        z = la - 5.0
        for p in range(4):
            f[f"mi_s_{ph}_{p}"] = float((A * m * z ** p).sum() / tot)
        e = GEDGES[ph]
        gb = np.digitize(G.med.values, e[1:-1])
        for b in range(len(e) - 1):
            f[f"mi_g_{ph}_{b}"] = float(A[m & (gb == b)].sum() / tot)
        if m.sum() >= 2:
            Am = A[m]
            f[f"mi_e_{ph}_logasp"] = float((Am * asp[m]).sum() / Am.sum())
            f[f"mi_e_{ph}_asp2"] = float(Am[asp[m] > np.log(2)].sum() / tot)
            f[f"mi_e_{ph}_asp3"] = float(Am[asp[m] > np.log(3)].sum() / tot)
            f[f"mi_e_{ph}_sol"] = float((Am * G.solidity.values[m]).sum() / Am.sum())
            f[f"mi_Neff_{ph}"] = float(Am.sum() ** 2 / (Am ** 2).sum())
            f[f"mi_iqr_{ph}"] = float(np.nansum(Am * G.iqr.values[m]) / Am[np.isfinite(G.iqr.values[m])].sum()) if np.isfinite(G.iqr.values[m]).any() else np.nan
        else:
            for nm in ("logasp", "sol"):
                f[f"mi_e_{ph}_{nm}"] = np.nan
            f[f"mi_e_{ph}_asp2"] = f[f"mi_e_{ph}_asp3"] = 0.0
            f[f"mi_Neff_{ph}"] = 0.0
            f[f"mi_iqr_{ph}"] = np.nan
    f["mi_N"] = len(A)
    f["mi_Neff"] = float(tot ** 2 / (A ** 2).sum())
    f["mi_log_neff_over_n"] = float(np.log(f["mi_Neff"] / len(A)))
    f["mi_fd"] = float(A[dk].sum() / tot)
    f["mi_mx_med_sd"] = float(np.sqrt(np.cov(G.med.values[~dk], aweights=A[~dk]))) if (~dk).sum() > 2 else np.nan
    rows.append(f)
    if k % 100 == 0:
        print(k, flush=True)
pd.concat(grains, ignore_index=True).to_parquet(cache / f"grains2_{split}.parquet")
pd.DataFrame(rows).to_parquet(cache / f"mig_{split}.parquet")
print("done", split, len(rows))
