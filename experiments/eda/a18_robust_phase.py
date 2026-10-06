"""A18: robust per-grain phase measurement (fixed per-image transform; train or test).
Concern: the v3 'ic_' normalisation divides by a 70th-percentile filter over ~80 px.  In coarse images a large dark grain
can fill >30% of that window, which pulls the local 'matrix level' down and hides the dark grain (and vice versa).
Here: watershed grains (cached), per-grain median of the NLM-denoised raw image, shading = robust quadratic surface fitted
to matrix-grain medians only (iterated), grains classified on the shading-corrected ratio with a per-image 2-cluster
split (valley of the area-weighted log-ratio histogram, fallback 0.90).  Outputs area / number fractions, per-phase sizes,
per-phase shape, pores, border shares, largest-grain descriptors.
Usage: python a18_robust_phase.py CACHE_DIR split"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure, restoration

from eda_common import DATA_DIR

cache = Path(sys.argv[1])
split = sys.argv[2]
ids = pd.read_csv(DATA_DIR / ("train.csv" if split == "train" else "sample_submission.csv")).ID.tolist()
ws_all = np.load(cache / f"ws_{split}.npy", mmap_mode="r")
H = W = 256
yy, xx = np.mgrid[:H, :W] / 255.0 - 0.5


def quad_design(cy, cx):
    cy, cx = np.asarray(cy) / 255.0 - 0.5, np.asarray(cx) / 255.0 - 0.5
    return np.column_stack([np.ones_like(cy), cy, cx, cy * cy, cx * cx, cy * cx])


def wm(v, w):
    return float(np.sum(v * w) / np.sum(w)) if np.sum(w) > 0 else np.nan


rows = []
for k, i in enumerate(ids):
    raw8 = cv2.imread(str(DATA_DIR / split / f"{i}.png"), cv2.IMREAD_GRAYSCALE)
    sig = float(restoration.estimate_sigma(raw8.astype(np.float32)))
    den = cv2.fastNlMeansDenoising(raw8, None, h=float(np.clip(sig, 2.0, 30.0)), templateWindowSize=5,
                                   searchWindowSize=21).astype(np.float32)
    w = np.asarray(ws_all[k]).astype(np.int32)
    line = w == 0
    inner = w.copy()
    inner[ndi.binary_dilation(line, iterations=1)] = 0
    nl = int(w.max())
    lab = np.arange(1, nl + 1)
    area = np.asarray(ndi.sum(np.ones_like(den), w, lab))
    a_in = np.asarray(ndi.sum(np.ones_like(den), inner, lab))
    med = np.asarray(ndi.median(den, inner, lab))
    q10 = np.asarray(ndi.labeled_comprehension(den, inner, lab, lambda v: np.percentile(v, 10) if len(v) else np.nan, float, np.nan))
    com = ndi.center_of_mass(np.ones_like(den), w, lab)
    cy = np.array([c[0] for c in com])
    cx = np.array([c[1] for c in com])
    border = np.zeros(nl + 1, bool)
    border[np.unique(np.concatenate([w[0], w[-1], w[:, 0], w[:, -1]]))] = True
    border = border[1:]
    f = {"ID": i, "rp_sigma": sig}
    # pores: compact very dark blobs relative to the grain they sit in -> simple global rule on the ratio image below
    ok = (a_in >= 8) & np.isfinite(med)
    if ok.sum() < 5:
        rows.append(f)
        continue
    # shading surface from matrix grains (start: brightest 60% of area)
    D = quad_design(cy[ok], cx[ok])
    m_ok, A_ok = med[ok], area[ok]
    ref = np.percentile(m_ok, 40)
    mx = m_ok >= ref
    for _ in range(4):
        wts = np.sqrt(A_ok[mx])
        coef = np.linalg.lstsq(D[mx] * wts[:, None], m_ok[mx] * wts, rcond=None)[0]
        surf = D @ coef
        ratio = m_ok / np.maximum(surf, 1.0)
        # 2-cluster split on log ratio: valley of area-weighted histogram between 0.70 and 0.97
        hist, edges = np.histogram(np.log(ratio), bins=np.linspace(np.log(0.6), np.log(1.15), 56), weights=A_ok)
        hs = np.convolve(hist, np.ones(3) / 3, mode="same")
        ctr = np.exp(0.5 * (edges[1:] + edges[:-1]))
        cand = (ctr > 0.72) & (ctr < 0.96)
        thr = 0.90
        if cand.any() and (ratio < 0.85).sum() >= 2:
            j = np.argmin(np.where(cand, hs, np.inf))
            thr = float(ctr[j])
        mx = ratio >= thr
    dk = ~mx
    # pores inside grains: pixels far darker than the matrix surface
    full_surf = (np.column_stack([np.ones(H * W), yy.ravel(), xx.ravel(), yy.ravel() ** 2, xx.ravel() ** 2, (yy * xx).ravel()]) @ coef).reshape(H, W)
    rr = cv2.GaussianBlur(den, (0, 0), 1.0) / np.maximum(full_surf, 1.0)
    pm = ndi.binary_opening(rr < 0.62, structure=np.ones((3, 3)))
    plab = measure.label(pm)
    pareas = np.bincount(plab.ravel())[1:]
    pareas = pareas[pareas >= 6]
    A = A_ok
    Aint = A[~border[ok]]
    dk_int = dk[~border[ok]]
    f.update({
        "rp_thr": thr, "rp_shade_range": float((surf.max() - surf.min()) / np.median(surf)),
        "rp_contrast": float(np.median(ratio[dk])) if dk.any() else np.nan,
        "rp_fd_area": float(A[dk].sum() / A.sum()),
        "rp_fd_num": float(dk.mean()),
        "rp_fd_area_int": float(Aint[dk_int].sum() / Aint.sum()) if Aint.sum() > 0 else np.nan,
        "rp_fd_num_int": float(dk_int.mean()) if len(dk_int) else np.nan,
        "rp_n": int(ok.sum()), "rp_n_int": int((~border[ok]).sum()), "rp_nd": int(dk.sum()),
        "rp_d_all": float(np.sqrt(Aint.mean())) if len(Aint) else np.nan,
        "rp_d_dk": float(np.sqrt(Aint[dk_int].mean())) if dk_int.any() else np.nan,
        "rp_d_mx": float(np.sqrt(Aint[~dk_int].mean())) if (~dk_int).any() else np.nan,
        "rp_border_area_share": float(A[border[ok]].sum() / A.sum()),
        "rp_border_dk_share": float(A[border[ok] & dk].sum() / max(A[dk].sum(), 1)),
        "rp_pore_frac": float(pareas.sum() / (H * W)), "rp_pore_n": int(len(pareas)),
        "rp_pore_area_mean": float(pareas.mean()) if len(pareas) else 0.0,
        "rp_ratio_mx_sd": float(np.sqrt(wm((ratio[mx] - wm(ratio[mx], A[mx])) ** 2, A[mx]))) if mx.sum() > 1 else np.nan,
        "rp_ratio_dk_sd": float(np.sqrt(wm((ratio[dk] - wm(ratio[dk], A[dk])) ** 2, A[dk]))) if dk.sum() > 1 else np.nan,
        "rp_q10_dk": float(wm(q10[ok][dk] / np.maximum(surf[dk], 1), A[dk])) if dk.any() else np.nan,
    })
    # largest grains: phase / shape / share
    order = np.argsort(-A)
    props = {p.label: p for p in measure.regionprops(w)}
    labs_ok = lab[ok]
    for j in range(3):
        if j < len(order):
            p = props.get(labs_ok[order[j]])
            f[f"rp_big{j}_share"] = float(A[order[j]] / A.sum())
            f[f"rp_big{j}_dark"] = int(dk[order[j]])
            f[f"rp_big{j}_border"] = int(border[ok][order[j]])
            f[f"rp_big{j}_asp"] = float(p.axis_major_length / max(p.axis_minor_length, 1)) if p is not None else np.nan
    # phase-specific aspect
    asp = np.array([props[l].axis_major_length / max(props[l].axis_minor_length, 1) if l in props else np.nan for l in labs_ok])
    la = np.log(np.clip(asp, 1, 50))
    f["rp_logasp_dk"] = wm(la[dk], A[dk]) if dk.any() else np.nan
    f["rp_logasp_mx"] = wm(la[mx], A[mx]) if mx.any() else np.nan
    rows.append(f)
    if k % 100 == 0:
        print(k, flush=True)
pd.DataFrame(rows).to_parquet(cache / f"rp_{split}.parquet")
print("done", split, len(rows))
