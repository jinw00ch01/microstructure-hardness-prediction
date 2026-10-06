"""A48: extended per-grain table for the multi-dimensional per-grain test (fixed per-image transform; train or test).
Grains: cached watershed (a2), area >= 12, pore share <= 0.5, >= 8 px 1-px-eroded interior.  Per grain:
  la (log area), asp (log maj/min), rel_ang (cos 2(theta - theta0), theta0 = image area-weighted doubled-angle mean),
  grey (interior median of rn = NLM image / local matrix level), gstd (log std of raw/bg over the 3-px interior, NaN if
  < 15 px), bc (1 - mean G1(raw/bg) on the grain's own watershed-line pixels / interior median), dark (grey < 0.91),
  nb_n, nb_la, nb_dk (neighbour count, mean log area, dark share; neighbours = grains sharing a line pixel),
  d_pore (log1p distance from the centroid to the nearest pore pixel; 256 if none), zone (Gaussian sigma 32 px,
  area-weighted local mean log area around the centroid minus the image area-weighted mean), border.
Writes CACHE/g3_{split}.parquet.  Usage: python a48_grain_table3.py CACHE_DIR split"""
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


def _bg_matrix(den, pct=70, win=20, sub=4):
    small = cv2.resize(den, (den.shape[1] // sub, den.shape[0] // sub), interpolation=cv2.INTER_AREA)
    b = ndi.percentile_filter(small, pct, size=win, mode="reflect")
    b = cv2.GaussianBlur(b, (0, 0), win / 3)
    return cv2.resize(b, den.shape[::-1], interpolation=cv2.INTER_CUBIC)


out = []
for k, i in enumerate(ids):
    raw = cv2.imread(str(DATA_DIR / split / f"{i}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    x = raw / np.maximum(_bg_matrix(cv2.GaussianBlur(raw, (0, 0), 1.0)), 1.0)
    rn = np.asarray(rn_all[k], np.float32)
    w = np.asarray(ws_all[k]).astype(np.int32)
    nl = int(w.max())
    if nl < 5:
        continue
    idx = np.arange(1, nl + 1)
    pm = ndi.binary_opening(cv2.GaussianBlur(rn, (0, 0), 1.0) < 0.60, structure=np.ones((3, 3)))
    line = w == 0
    in1 = w.copy()
    in1[ndi.binary_dilation(line) | pm] = 0
    in3 = w.copy()
    in3[ndi.binary_dilation(line, iterations=3) | ndi.binary_dilation(pm, iterations=2)] = 0
    one = np.ones_like(x)
    a_in1 = np.asarray(ndi.sum(one, in1, idx))
    a_in3 = np.asarray(ndi.sum(one, in3, idx))
    pf = np.asarray(ndi.mean(pm.astype(np.float32), w, idx))
    grey = np.asarray(ndi.median(rn, in1, idx))
    gstd = np.sqrt(np.asarray(ndi.variance(x, in3, idx)))
    g1 = cv2.GaussianBlur(x, (0, 0), 1.0)
    int_g1 = np.asarray(ndi.median(g1, in1, idx))
    big = nl + 10
    lmax = ndi.maximum_filter(w, 3)
    lmin = ndi.minimum_filter(np.where(w == 0, big, w), 3)
    yy, xx = np.nonzero(line)
    L1, L2 = lmax[yy, xx], lmin[yy, xx]
    s_b, s_n = np.zeros(nl + 1), np.zeros(nl + 1)
    for L in (L1, L2):
        ok = (L > 0) & (L <= nl)
        np.add.at(s_b, L[ok], g1[yy[ok], xx[ok]])
        np.add.at(s_n, L[ok], 1.0)
    same = L1 == L2
    np.add.at(s_b, L1[same], -g1[yy[same], xx[same]])
    np.add.at(s_n, L1[same], -1.0)
    bc = 1.0 - np.where(s_n[1:] > 0, s_b[1:] / np.maximum(s_n[1:], 1), np.nan) / np.maximum(int_g1, 1e-3)
    pair = (L1 > 0) & (L2 <= nl) & (L1 != L2)
    E = np.unique(np.column_stack([L2[pair], L1[pair]]), axis=0)
    dpore = ndi.distance_transform_edt(~pm) if pm.any() else None
    props = measure.regionprops(w)
    rows = []
    for p_ in props:
        j = p_.label
        if p_.area < 12 or pf[j - 1] > 0.5 or a_in1[j - 1] < 8 or not np.isfinite(grey[j - 1]):
            continue
        rows.append((j, p_.area, p_.centroid[0], p_.centroid[1], p_.axis_major_length, max(p_.axis_minor_length, 1.0),
                     p_.orientation, grey[j - 1], gstd[j - 1] if a_in3[j - 1] >= 15 else np.nan, bc[j - 1],
                     any(v in (0, 255) for v in p_.bbox[:2]) or p_.bbox[2] >= 256 or p_.bbox[3] >= 256))
    if len(rows) < 8:
        continue
    G = pd.DataFrame(rows, columns=["label", "area", "cy", "cx", "maj", "min", "ori", "grey", "gstd", "bc", "border"])
    keep = set(G.label)
    la = np.log(G.area.values)
    dk = G.grey.values < 0.91
    pos = {lab: n for n, lab in enumerate(G.label)}
    nb = [[] for _ in range(len(G))]
    for a_, b_ in E:
        if a_ in keep and b_ in keep:
            nb[pos[a_]].append(pos[b_])
            nb[pos[b_]].append(pos[a_])
    G["nb_n"] = [len(v) for v in nb]
    G["nb_la"] = [la[v].mean() if v else np.nan for v in nb]
    G["nb_dk"] = [dk[v].mean() if v else np.nan for v in nb]
    A = G.area.values
    th0 = 0.5 * np.arctan2(np.sum(A * np.sin(2 * G.ori.values)), np.sum(A * np.cos(2 * G.ori.values)))
    G["rel_ang"] = np.cos(2 * (G.ori.values - th0))
    G["asp"] = np.log(G["maj"].values / G["min"].values)
    G["la"] = la
    G["dark"] = dk.astype(float)
    cyx = G[["cy", "cx"]].values
    if dpore is not None:
        G["d_pore"] = np.log1p(dpore[np.clip(cyx[:, 0].round().astype(int), 0, 255), np.clip(cyx[:, 1].round().astype(int), 0, 255)])
    else:
        G["d_pore"] = np.log1p(256.0)
    D2 = ((cyx[:, None, :] - cyx[None, :, :]) ** 2).sum(-1)
    Wt = np.exp(-D2 / (2 * 32.0 ** 2)) * A[None, :]
    G["zone"] = (Wt @ la) / Wt.sum(1) - np.sum(A * la) / A.sum()
    G["gstd"] = np.log(G.gstd.values)
    G.insert(0, "ID", i)
    out.append(G.drop(columns=["maj", "min", "ori", "cy", "cx"]))
    if k % 100 == 0:
        print(k, flush=True)
pd.concat(out, ignore_index=True).to_parquet(cache / f"g3_{split}.parquet")
print("done", split, len(out))
