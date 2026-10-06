"""A26: local structure size maps (heterogeneity of grain size across the image).  Fixed per-image transform (train/test).
rn (cached NLM / local-matrix image), pores filled with 1, high-passed (sigma 16).  For each block (2x2 quadrants of 128 px,
4x4 blocks of 64 px) the normalised autocorrelation (lag-1 normalised, removes white noise) gives the lag where it falls to
0.5 along / across the image principal axis.  Heterogeneity = CV / max-min / area-weighted vs plain means across blocks;
also the watershed grain count per block for clean images.  Usage: python a26_localsize.py CACHE_DIR split"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from eda_common import DATA_DIR

cache = Path(sys.argv[1])
split = sys.argv[2]
ids = pd.read_csv(DATA_DIR / ("train.csv" if split == "train" else "sample_submission.csv")).ID.tolist()
rn_all = np.load(cache / f"rn_{split}.npy", mmap_mode="r")
ws_all = np.load(cache / f"ws_{split}.npy", mmap_mode="r")
T = np.arange(0, 40, 0.5)


def ac_len(x, ang, lv=0.5):
    x = x - x.mean()
    h, w = x.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    F = np.fft.fft2(x * win, s=(2 * h, 2 * w))
    ac = np.fft.fftshift(np.real(np.fft.ifft2(F * np.conj(F))))
    cy, cx = h, w
    c = 0.25 * (ac[cy - 1, cx] + ac[cy + 1, cx] + ac[cy, cx - 1] + ac[cy, cx + 1])
    ac = ac / (c + 1e-12)
    out = []
    for a in (ang, ang + np.pi / 2):
        prof = 0.5 * (ndi.map_coordinates(ac, [cy - T * np.sin(a), cx + T * np.cos(a)], order=1) +
                      ndi.map_coordinates(ac, [cy + T * np.sin(a), cx - T * np.cos(a)], order=1))
        below = np.where(prof[2:] < lv)[0]
        out.append(float(T[2:][below[0]]) if len(below) else 40.0)
    return out


rows = []
for k, i in enumerate(ids):
    rn = rn_all[k].astype(np.float32)
    s1 = cv2.GaussianBlur(rn, (0, 0), 1.0)
    pm = ndi.binary_dilation(ndi.binary_opening(s1 < 0.68, structure=np.ones((3, 3))), iterations=3)
    x = np.where(pm, 1.0, rn).astype(np.float32)
    x = x - cv2.GaussianBlur(x, (0, 0), 16.0)
    gy, gx = np.gradient(cv2.GaussianBlur(x, (0, 0), 2.0))
    Jxx, Jyy, Jxy = (gx * gx).mean(), (gy * gy).mean(), (gx * gy).mean()
    ang = -(0.5 * np.arctan2(2 * Jxy, Jxx - Jyy) + np.pi / 2)
    f = {"ID": i}
    w = np.asarray(ws_all[k])
    cen = np.array([c for c in ndi.center_of_mass(np.ones_like(w), w, np.arange(1, w.max() + 1))]) if w.max() else np.zeros((0, 2))
    for nb in (2, 4):
        bs = 256 // nb
        Lp, Lq, dk = [], [], []
        for by in range(nb):
            for bx in range(nb):
                blk = x[by * bs:(by + 1) * bs, bx * bs:(bx + 1) * bs]
                a_, b_ = ac_len(blk, ang)
                Lp.append(a_)
                Lq.append(b_)
                dk.append(float((cv2.GaussianBlur(rn, (0, 0), 2.0)[by * bs:(by + 1) * bs, bx * bs:(bx + 1) * bs] < 0.91).mean()))
        Lp, Lq = np.array(Lp), np.array(Lq)
        g = np.sqrt(Lp * Lq)
        f[f"ls{nb}_gm_mean"] = float(g.mean())
        f[f"ls{nb}_gm_cv"] = float(g.std() / max(g.mean(), 1e-6))
        f[f"ls{nb}_gm_maxmin"] = float(g.max() / max(g.min(), 0.5))
        f[f"ls{nb}_area_w_over_n"] = float((g ** 4).mean() / max((g ** 2).mean() ** 2, 1e-9))  # area-weighted heterogeneity
        f[f"ls{nb}_par_cv"] = float(Lp.std() / max(Lp.mean(), 1e-6))
        f[f"ls{nb}_perp_cv"] = float(Lq.std() / max(Lq.mean(), 1e-6))
        f[f"ls{nb}_dk_sd"] = float(np.std(dk))
        f[f"ls{nb}_corr_size_dk"] = float(np.corrcoef(g, dk)[0, 1]) if np.std(g) > 0 and np.std(dk) > 0 else 0.0
        if len(cen):
            gi = (cen[:, 0] // bs).astype(int) * nb + (cen[:, 1] // bs).astype(int)
            cnt = np.bincount(gi, minlength=nb * nb).astype(float)
            f[f"ls{nb}_count_cv"] = float(cnt.std() / max(cnt.mean(), 1e-6))
    # whole-image gradient of local size (plane fit on 4x4 log sizes)
    rows.append(f)
    if k % 100 == 0:
        print(k, flush=True)
pd.DataFrame(rows).to_parquet(cache / f"ls_{split}.parquet")
print("done", split, len(rows))
