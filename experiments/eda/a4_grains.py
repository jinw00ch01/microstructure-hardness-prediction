"""A4: per-grain table from the cached watershed (train + test): area, median rn, centroid, axes, pore overlap.
Per-image grain statistics used by the variance model: N (grain count), N_eff=(sum A)^2/sum A^2,
dark-grain count / effective count, largest grain, etc.  Usage: python a4_grains.py CACHE_DIR split"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure

from eda_common import DATA_DIR

cache = Path(sys.argv[1])
split = sys.argv[2]
ids = pd.read_csv(DATA_DIR / ("train.csv" if split == "train" else "sample_submission.csv")).ID.tolist()
rn = np.load(cache / f"rn_{split}.npy", mmap_mode="r")
ws = np.load(cache / f"ws_{split}.npy", mmap_mode="r")
rows, grains = [], []
for k, i in enumerate(ids):
    x = rn[k].astype(np.float32)
    s1 = cv2.GaussianBlur(x, (0, 0), 1.0)
    pm = ndi.binary_opening(s1 < 0.60, structure=np.ones((3, 3)))
    w = np.asarray(ws[k])
    line = w == 0
    inner = w.copy()
    inner[ndi.binary_dilation(line)] = 0
    idx = np.arange(1, w.max() + 1)
    med = np.asarray(ndi.median(x, inner, idx))
    area_in = np.asarray(ndi.sum(np.ones_like(x), inner, idx))
    area = np.asarray(ndi.sum(np.ones_like(x), w, idx))
    pf = np.asarray(ndi.mean(pm.astype(np.float32), w, idx))
    # touches border?
    border = np.zeros(w.max() + 1, bool)
    border[np.unique(np.concatenate([w[0], w[-1], w[:, 0], w[:, -1]]))] = True
    border = border[1:]
    ok = (area_in >= 8) & (pf <= 0.5)
    A, M, B = area[ok], med[ok], border[ok]
    dk = M < 0.91
    tot = A.sum()
    f = {"ID": i, "g_n": int(ok.sum()), "g_n_int": int((~B).sum()),
         "g_neff": float(tot ** 2 / (A ** 2).sum()) if tot > 0 else np.nan,
         "g_fd": float(A[dk].sum() / tot) if tot > 0 else np.nan,
         "g_nd": int(dk.sum()),
         "g_neff_dk": float(A[dk].sum() ** 2 / (A[dk] ** 2).sum()) if dk.any() else 0.0,
         "g_amax": float(A.max()) if len(A) else np.nan,
         "g_amax_frac": float(A.max() / tot) if tot > 0 else np.nan,
         "g_amax_dk_frac": float(A[dk].max() / tot) if dk.any() else 0.0,
         "g_amean": float(A.mean()) if len(A) else np.nan}
    rows.append(f)
    if split == "train":
        props = measure.regionprops(w)
        for p_, o in zip(props, ok):
            if not o:
                continue
            grains.append({"ID": i, "label": p_.label, "area": p_.area, "cy": p_.centroid[0], "cx": p_.centroid[1],
                           "maj": p_.axis_major_length, "min": p_.axis_minor_length, "ori": p_.orientation,
                           "med": float(med[p_.label - 1]), "border": bool(border[p_.label - 1])})
pd.DataFrame(rows).to_parquet(cache / f"grainstats_{split}.parquet")
if grains:
    pd.DataFrame(grains).to_parquet(cache / f"grains_{split}.parquet")
print("done", split)
