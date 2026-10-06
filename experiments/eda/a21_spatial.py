"""A21: spatial arrangement of the dark phase + border / largest-grain descriptors (fixed per-image transform; train or test).
Dark mask = (Gauss_2(rn) < 0.91) minus dilated pores (rn = cached NLM / local-matrix image, same as v3), specks < 15 px removed.
  connectivity: components (8-conn), largest component share, spanning along x / y / elongation axis, Euler number
  contiguity:   dark-dark vs dark-matrix boundary length (watershed lines, labels on both sides)
  two-point:    normalised S2(r) = (S2 - p^2)/(p - p^2) at r = 4..96 (isotropic, along / across the elongation axis)
  lineal path:  fraction of segments of length r fully inside the dark phase, along / across the axis
  dispersion:   variance of window dark fraction (w = 32/64/128) / binomial expectation from the window grain count
  alignment:    orientation order of dark components, dark-vs-matrix grain axis agreement, banding = S2_par / S2_perp
  grid:         2x2 / 4x4 cell std of dark fraction, grain count and mean rn
Usage: python a21_spatial.py CACHE_DIR split"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure, morphology

from eda_common import DATA_DIR

cache = Path(sys.argv[1])
split = sys.argv[2]
ids = pd.read_csv(DATA_DIR / ("train.csv" if split == "train" else "sample_submission.csv")).ID.tolist()
rn_all = np.load(cache / f"rn_{split}.npy", mmap_mode="r")
ws_all = np.load(cache / f"ws_{split}.npy", mmap_mode="r")
H = W = 256
R_LIST = (4, 8, 16, 32, 64, 96)


def s2_profile(m):
    x = m.astype(np.float32)
    F = np.fft.fft2(x, s=(2 * H, 2 * W))
    ac = np.real(np.fft.ifft2(F * np.conj(F)))
    ones = np.fft.fft2(np.ones_like(x), s=(2 * H, 2 * W))
    cnt = np.real(np.fft.ifft2(ones * np.conj(ones)))
    S = np.fft.fftshift(ac / np.maximum(cnt, 1))
    return S  # centre at (H, W)


def sample(S, r, ang):
    cy, cx = H, W
    vals = []
    for s in (1, -1):
        vals.append(ndi.map_coordinates(S, [[cy - s * r * np.sin(ang)], [cx + s * r * np.cos(ang)]], order=1)[0])
    return float(np.mean(vals))


def lineal(m, r, ang):
    """fraction of length-r segments (direction ang) fully inside m: erosion with a line structuring element."""
    L = int(r)
    k = np.zeros((2 * L + 1, 2 * L + 1), np.uint8)
    x1, y1 = int(round(L + 0.5 * L * np.cos(ang))), int(round(L - 0.5 * L * np.sin(ang)))
    x0, y0 = int(round(L - 0.5 * L * np.cos(ang))), int(round(L + 0.5 * L * np.sin(ang)))
    cv2.line(k, (x0, y0), (x1, y1), 1, 1)
    e = cv2.erode(m.astype(np.uint8), k, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    inner = np.zeros_like(m, bool)
    inner[L:-L, L:-L] = True
    return float(e[inner].mean())


rows = []
for k, i in enumerate(ids):
    rn = rn_all[k].astype(np.float32)
    w = np.asarray(ws_all[k]).astype(np.int32)
    s1 = cv2.GaussianBlur(rn, (0, 0), 1.0)
    s2 = cv2.GaussianBlur(rn, (0, 0), 2.0)
    pm = ndi.binary_opening(s1 < 0.68, structure=np.ones((3, 3)))
    pm_ex = ndi.binary_dilation(pm, iterations=3)
    dark = morphology.remove_small_objects((s2 < 0.91) & ~pm_ex, 15)
    p = float(dark.mean())
    f = {"ID": i, "sp_p": p}
    # principal axis (structure tensor on rn with pores filled, as v3)
    gy, gx = np.gradient(cv2.GaussianBlur(np.where(pm_ex, 1.0, rn).astype(np.float32), (0, 0), 2.0))
    Jxx, Jyy, Jxy = (gx * gx).mean(), (gy * gy).mean(), (gx * gy).mean()
    ang = -(0.5 * np.arctan2(2 * Jxy, Jxx - Jyy) + np.pi / 2)  # elongation axis, row/col convention as v3
    # ---- connectivity
    lab = measure.label(dark, connectivity=2)
    nc = int(lab.max())
    f["sp_ncomp_density"] = nc / (H * W) * 1e4
    if nc:
        areas = np.bincount(lab.ravel())[1:]
        big = int(np.argmax(areas)) + 1
        f["sp_maxcomp_img"] = float(areas.max() / (H * W))
        f["sp_maxcomp_share"] = float(areas.max() / areas.sum())
        f["sp_comp_wmean_img"] = float((areas ** 2).sum() / areas.sum() / (H * W))
        sx = sy = 0
        ext_par = 0.0
        for cl in range(1, nc + 1):
            if areas[cl - 1] < 50:
                continue
            ys, xs = np.nonzero(lab == cl) if areas[cl - 1] > 0.002 * H * W else (None, None)
            if ys is None:
                continue
            sx = max(sx, int(xs.min() <= 2 and xs.max() >= W - 3))
            sy = max(sy, int(ys.min() <= 2 and ys.max() >= H - 3))
            proj = xs * np.cos(ang) - ys * np.sin(ang)
            ext_par = max(ext_par, float(proj.max() - proj.min()))
        f["sp_span_x"], f["sp_span_y"] = sx, sy
        f["sp_span_any"] = int(sx or sy)
        f["sp_maxext_par"] = ext_par / 256.0
        f["sp_euler_density"] = float(measure.euler_number(dark, connectivity=2)) / (H * W) * 1e4
    else:
        for kk in ("sp_maxcomp_img", "sp_maxcomp_share", "sp_comp_wmean_img", "sp_span_x", "sp_span_y", "sp_span_any", "sp_maxext_par", "sp_euler_density"):
            f[kk] = 0.0
    # ---- contiguity from watershed lines: phase of the grains on both sides of each line pixel
    idx = np.arange(1, w.max() + 1)
    inner = w.copy()
    inner[ndi.binary_dilation(w == 0)] = 0
    gmed = np.asarray(ndi.median(rn, inner, idx)) if len(idx) else np.array([])
    gdark = np.zeros(w.max() + 1, bool)
    gdark[1:] = gmed < 0.91
    line = w == 0
    mx_ = ndi.maximum_filter(w, size=3)
    mn_ = ndi.minimum_filter(np.where(w == 0, 10 ** 9, w), size=3)
    ok = line & (mx_ > 0) & (mn_ < 10 ** 9) & (mx_ != mn_)
    a, b = gdark[mx_[ok]], gdark[np.clip(mn_[ok], 0, w.max())]
    ndd, ndm = int((a & b).sum()), int((a ^ b).sum())
    f["sp_contiguity_dd"] = 2 * ndd / max(2 * ndd + ndm, 1)
    f["sp_dd_line_frac"] = ndd / max(ok.sum(), 1)
    # ---- two-point correlation and lineal path
    S = s2_profile(dark)
    den = max(p - p * p, 1e-6)
    for r in R_LIST:
        sp_par, sp_perp = sample(S, r, ang), sample(S, r, ang + np.pi / 2)
        iso = np.mean([sample(S, r, a_) for a_ in np.linspace(0, np.pi, 8, endpoint=False)])
        f[f"sp_s2n_iso_{r}"] = (iso - p * p) / den
        f[f"sp_s2n_par_{r}"] = (sp_par - p * p) / den
        f[f"sp_s2n_perp_{r}"] = (sp_perp - p * p) / den
    for r in (8, 16, 32):
        f[f"sp_band_{r}"] = f[f"sp_s2n_par_{r}"] - f[f"sp_s2n_perp_{r}"]
    for r in (4, 8, 16, 32):
        f[f"sp_lp_par_{r}"] = lineal(dark, r, ang) / max(p, 1e-3)
        f[f"sp_lp_perp_{r}"] = lineal(dark, r, ang + np.pi / 2) / max(p, 1e-3)
    # ---- dispersion of the dark fraction in windows vs binomial expectation given grain counts
    centers = np.array([[ (pp.centroid[0]), (pp.centroid[1]), pp.area] for pp in measure.regionprops(w)]) if w.max() else np.zeros((0, 3))
    for ws_ in (32, 64, 128):
        nb = H // ws_
        fr = dark.reshape(nb, ws_, nb, ws_).mean(axis=(1, 3)).ravel()
        if len(centers):
            gi = (centers[:, 0] // ws_).astype(int) * nb + (centers[:, 1] // ws_).astype(int)
            cnt = np.bincount(gi, minlength=nb * nb).astype(float)
        else:
            cnt = np.ones(nb * nb)
        expv = np.mean(p * (1 - p) / np.maximum(cnt, 1))
        f[f"sp_disp_{ws_}"] = float(fr.var() / max(expv, 1e-6))
        f[f"sp_cellstd_dark_{ws_}"] = float(fr.std())
        f[f"sp_cellstd_rn_{ws_}"] = float(rn.reshape(nb, ws_, nb, ws_).mean(axis=(1, 3)).std())
        f[f"sp_cellcv_count_{ws_}"] = float(cnt.std() / max(cnt.mean(), 1e-6))
    # ---- alignment of dark components vs matrix grains
    dprops = [pp for pp in measure.regionprops(lab) if pp.area >= 30]
    if dprops:
        A = np.array([pp.area for pp in dprops], float)
        th = np.array([pp.orientation for pp in dprops])
        asp = np.array([pp.axis_major_length / max(pp.axis_minor_length, 1) for pp in dprops])
        Rd = np.sum(A * (asp - 1) * np.exp(2j * th)) / max(np.sum(A * (asp - 1)), 1e-6)
        f["sp_dk_align_R"] = float(np.abs(Rd))
        f["sp_dk_asp_wmean"] = float(np.sum(A * asp) / A.sum())
        gprops = [pp for pp in measure.regionprops(w) if pp.area >= 30]
        if gprops:
            Ag = np.array([pp.area for pp in gprops], float)
            thg = np.array([pp.orientation for pp in gprops])
            Rg = np.sum(Ag * np.exp(2j * thg)) / Ag.sum()
            f["sp_dk_vs_grain_axis"] = float(np.cos(np.angle(Rd) - np.angle(Rg)))
        else:
            f["sp_dk_vs_grain_axis"] = np.nan
    else:
        f["sp_dk_align_R"] = f["sp_dk_asp_wmean"] = f["sp_dk_vs_grain_axis"] = np.nan
    rows.append(f)
    if k % 100 == 0:
        print(k, flush=True)
pd.DataFrame(rows).to_parquet(cache / f"sp_{split}.parquet")
print("done", split, len(rows))
