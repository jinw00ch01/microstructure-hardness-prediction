"""Handcrafted microstructure features.

v1: python -m src.features            -> data/features.parquet     (103 generic texture/quality features)
v2: python -m src.features --v2       -> data/features_v2.parquet  (physics-oriented measurements)

All transforms are strictly per-image (nothing is fitted across images), so computing them on test
images is allowed by the competition rules.
"""
import argparse
import os
import warnings

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology, restoration, segmentation

import lightgbm as lgb

from .common import DATA_DIR, img_path, load_test, read_img

warnings.filterwarnings("ignore", category=FutureWarning)


def _radial_profile(power):
    h, w = power.shape
    y, x = np.indices(power.shape)
    r = np.hypot(y - h // 2, x - w // 2).astype(int)
    prof = np.bincount(r.ravel(), power.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    return prof[: h // 2]


def _autocorr_len(img, axis_vec):
    f = np.fft.fft2(img - img.mean())
    ac = np.real(np.fft.ifft2(f * np.conj(f)))
    ac = np.fft.fftshift(ac) / ac.max()
    c = np.array(ac.shape) // 2
    vals = [ac[c[0] + int(round(t * axis_vec[0])), c[1] + int(round(t * axis_vec[1]))] for t in range(60)]
    below = np.where(np.array(vals) < 1 / np.e)[0]
    return float(below[0]) if len(below) else 60.0


def extract(i):
    raw = read_img(i)
    f = {"ID": i}
    # ---- image quality
    f["raw_mean"], f["raw_std"] = raw.mean(), raw.std()
    for p in (1, 5, 50, 95, 99):
        f[f"raw_p{p}"] = np.percentile(raw, p)
    f["noise_sigma"] = float(restoration.estimate_sigma(raw))
    f["lap_var"] = float(cv2.Laplacian(raw, cv2.CV_32F).var())
    # ---- denoise + normalize
    den = cv2.fastNlMeansDenoising((raw * 255).astype(np.uint8), None,
                                   h=float(np.clip(f["noise_sigma"] * 255 * 1.2, 3, 40)), templateWindowSize=7,
                                   searchWindowSize=21).astype(np.float32) / 255
    sm = filters.gaussian(den, 1.0)
    med = np.median(sm)
    iqr = np.subtract(*np.percentile(sm, [75, 25])) + 1e-6
    z = (sm - med) / iqr
    for p in (0.5, 1, 2, 5, 10, 25, 75, 90, 99):
        f[f"z_p{p}"] = np.percentile(z, p)
    f["den_std"] = den.std()
    # ---- pores: small very dark round blobs
    for thr in (-2.0, -3.0, -4.0):
        m = z < thr
        m = ndi.binary_opening(m, iterations=1)
        lab = measure.label(m)
        props = measure.regionprops(lab)
        areas = np.array([p.area for p in props]) if props else np.zeros(1)
        f[f"pore{thr}_frac"] = m.mean()
        f[f"pore{thr}_n"] = len(props)
        f[f"pore{thr}_area_mean"] = areas.mean()
        f[f"pore{thr}_area_max"] = areas.max()
    # ---- dark phase via multi-otsu on smoothed image
    try:
        th = filters.threshold_multiotsu(filters.gaussian(den, 2.0), classes=3)
        s2 = filters.gaussian(den, 2.0)
        f["phase_dark_frac"] = (s2 < th[0]).mean()
        f["phase_bright_frac"] = (s2 > th[1]).mean()
        f["phase_gap"] = th[1] - th[0]
    except ValueError:
        f["phase_dark_frac"] = f["phase_bright_frac"] = f["phase_gap"] = np.nan
    otsu = filters.threshold_otsu(sm)
    f["otsu_dark_frac"] = (sm < otsu).mean()
    # ---- boundaries / grain size
    for s in (1.0, 2.0, 3.0):
        e = feature.canny(den, sigma=s)
        f[f"canny{s}_density"] = e.mean()
    gm = filters.sobel(sm)
    f["grad_mean"], f["grad_std"] = gm.mean(), gm.std()
    # watershed segmentation of grains
    markers = measure.label(gm < np.percentile(gm, 30))
    ws = segmentation.watershed(gm, markers)
    props = measure.regionprops(ws)
    areas = np.array([p.area for p in props], dtype=float)
    ecc = np.array([p.eccentricity for p in props])
    ori = np.array([p.orientation for p in props])
    big = areas > 10
    f["ws_n"] = int(big.sum())
    f["ws_area_mean"] = areas[big].mean() if big.any() else 0
    f["ws_area_med"] = np.median(areas[big]) if big.any() else 0
    f["ws_area_std"] = areas[big].std() if big.any() else 0
    f["ws_area_cv"] = f["ws_area_std"] / (f["ws_area_mean"] + 1e-6)
    f["ws_d_inv_sqrt"] = 1 / np.sqrt(np.sqrt(f["ws_area_mean"] + 1e-6))
    f["ws_ecc_mean"] = ecc[big].mean() if big.any() else 0
    f["ws_ori_R"] = np.abs(np.mean(np.exp(2j * ori[big]))) if big.any() else 0
    # ---- structure tensor anisotropy
    for s in (2.0, 6.0):
        Arr, Arc, Acc = feature.structure_tensor(sm, sigma=s, order="rc")
        Jrr, Jrc, Jcc = Arr.mean(), Arc.mean(), Acc.mean()
        tr = Jrr + Jcc
        disc = np.sqrt((Jrr - Jcc) ** 2 + 4 * Jrc ** 2)
        f[f"st{s}_coh"] = disc / (tr + 1e-9)
        l1, l2 = (tr + disc) / 2, (tr - disc) / 2
        f[f"st{s}_ratio"] = l2 / (l1 + 1e-9)
        f[f"st{s}_angle_abs"] = abs(0.5 * np.arctan2(2 * Jrc, Jcc - Jrr))
    # ---- autocorrelation lengths along principal axes
    ang = 0.5 * np.arctan2(2 * Jrc, Jcc - Jrr)
    a1 = _autocorr_len(den, (np.sin(ang), np.cos(ang)))
    a2 = _autocorr_len(den, (np.cos(ang), -np.sin(ang)))
    f["ac_len_major"], f["ac_len_minor"] = max(a1, a2), min(a1, a2)
    f["ac_aspect"] = f["ac_len_major"] / (f["ac_len_minor"] + 1e-6)
    f["ac_len_gm"] = np.sqrt(a1 * a2 + 1e-6)
    # ---- FFT radial spectrum bands (on denoised)
    P = np.abs(np.fft.fftshift(np.fft.fft2(den - den.mean()))) ** 2
    prof = _radial_profile(P)
    prof = prof / (prof[1:].sum() + 1e-12)
    edges = [1, 3, 6, 10, 16, 25, 40, 64, 100, 128]
    for a, b in zip(edges[:-1], edges[1:]):
        f[f"fft_{a}_{b}"] = prof[a:b].sum()
    r = np.arange(len(prof))
    f["fft_centroid"] = (prof[1:] * r[1:]).sum()
    # ---- GLCM texture
    q = (np.clip((z + 3) / 6, 0, 1) * 31).astype(np.uint8)
    for d in (1, 3, 8):
        g = feature.graycomatrix(q, [d], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32, symmetric=True, normed=True)
        for prop in ("contrast", "homogeneity", "correlation", "energy"):
            v = feature.graycoprops(g, prop)[0]
            f[f"glcm{d}_{prop}"] = v.mean()
            f[f"glcm{d}_{prop}_aniso"] = v.max() - v.min()
    # ---- LBP histogram
    lbp = feature.local_binary_pattern((den * 255).astype(np.uint8), 8, 1, "uniform")
    h, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    for k, v in enumerate(h):
        f[f"lbp{k}"] = v
    return f


# =====================================================================================================
# v2: physics-oriented measurements (grain size / shape / alignment, dark phase, pores, image quality)
# Images are handled in 8-bit gray units (0..255). Generator palette observed on clean train images:
# matrix grains M ~ 156 (+-7 grain-to-grain), dark-phase grains ~0.80 M, boundaries ~0.83 M (1-2 px),
# pores ~0.3-0.55 M with crisp edges even in blurred images. Blur/noise vary strongly between images.
# =====================================================================================================
DISK1 = morphology.disk(1)
DISK2 = morphology.disk(2)


def _read8(i):
    return np.asarray(cv2.imread(str(img_path(i)), cv2.IMREAD_GRAYSCALE))


def _hist_mode(x, smooth=2.0):
    h, e = np.histogram(x, bins=256, range=(0, 256))
    hs = ndi.gaussian_filter1d(h.astype(float), smooth)
    return float(e[np.argmax(hs)] + 0.5)


def _wmean(v, w):
    w = np.asarray(w, float)
    return float(np.sum(v * w) / w.sum()) if len(v) and w.sum() > 0 else np.nan


def _orient_stats(theta, w):
    """Alignment order parameter R = |<exp(2i theta)>| and mean axis angle (0..pi)."""
    if len(theta) == 0 or np.sum(w) <= 0:
        return 0.0, 0.0
    z = np.sum(np.asarray(w) * np.exp(2j * np.asarray(theta))) / np.sum(w)
    return float(np.abs(z)), float((np.angle(z) / 2) % np.pi)


def _shape_stats(props, prefix, f, min_area=1):
    """Area / axis / aspect / orientation statistics of a list of regionprops."""
    props = [p for p in props if p.area >= min_area]
    n = len(props)
    f[f"{prefix}_n"] = n
    if n == 0:
        for k in ("area_mean", "area_med", "area_cv", "eqd_mean", "eqd_wmean", "maj_mean", "min_mean", "asp_mean",
                  "asp_wmean", "asp_med", "ori_R", "ori_Rel", "elong_align", "solidity"):
            f[f"{prefix}_{k}"] = np.nan
        return None
    A = np.array([p.area for p in props], float)
    maj = np.array([p.axis_major_length for p in props])
    mi = np.array([max(p.axis_minor_length, 1.0) for p in props])
    th = np.array([p.orientation for p in props])
    asp = maj / mi
    eqd = np.sqrt(4 * A / np.pi)
    f[f"{prefix}_area_mean"] = A.mean()
    f[f"{prefix}_area_med"] = float(np.median(A))
    f[f"{prefix}_area_cv"] = A.std() / A.mean()
    f[f"{prefix}_eqd_mean"] = eqd.mean()
    f[f"{prefix}_eqd_wmean"] = _wmean(eqd, A)
    f[f"{prefix}_maj_mean"] = maj.mean()
    f[f"{prefix}_min_mean"] = mi.mean()
    f[f"{prefix}_asp_mean"] = asp.mean()
    f[f"{prefix}_asp_wmean"] = _wmean(asp, A)
    f[f"{prefix}_asp_med"] = float(np.median(asp))
    R, _ = _orient_stats(th, A)
    Rel, _ = _orient_stats(th, A * (asp - 1))
    f[f"{prefix}_ori_R"] = R
    f[f"{prefix}_ori_Rel"] = Rel
    f[f"{prefix}_elong_align"] = R * (f[f"{prefix}_asp_wmean"] - 1)
    f[f"{prefix}_solidity"] = float(np.mean([p.solidity for p in props]))
    return A, maj, mi, th, asp


def _ac_ellipse(img, levels=(0.5, 0.25)):
    """Normalized 2-D autocorrelation (via FFT) -> second-moment ellipse of the central lobe above each level."""
    x = img - img.mean()
    F = np.fft.fft2(x)
    ac = np.real(np.fft.ifft2(F * np.conj(F)))
    ac = np.fft.fftshift(ac)
    c0 = ac[ac.shape[0] // 2, ac.shape[1] // 2]
    ac = ac / (c0 + 1e-12)
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    win = 64
    sub = ac[cy - win:cy + win + 1, cx - win:cx + win + 1]
    yy, xx = np.mgrid[-win:win + 1, -win:win + 1]
    out = {}
    for lv in levels:
        lab = measure.label(sub > lv)
        cl = lab[win, win]
        if cl == 0:
            out[lv] = (np.nan,) * 4
            continue
        m = lab == cl
        y, x_ = yy[m].astype(float), xx[m].astype(float)
        cyy, cxx, cxy = (y * y).mean(), (x_ * x_).mean(), (x_ * y).mean()
        tr, disc = cyy + cxx, np.sqrt((cxx - cyy) ** 2 + 4 * cxy ** 2)
        l1, l2 = (tr + disc) / 2, max((tr - disc) / 2, 1e-6)
        ang = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
        out[lv] = (2 * np.sqrt(l1), 2 * np.sqrt(l2), m.sum(), ang)
    # radial profile of ac (lags 1..48)
    r = np.hypot(yy, xx).astype(int)
    prof = np.bincount(r.ravel(), sub.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    return out, prof


def _intercepts(line_mask, f, prefix):
    """Linear-intercept grain size: boundary crossings per unit length along 4 directions."""
    lm = line_mask.astype(np.int8)
    L = {}
    # horizontal (along x) / vertical (along y)
    cr_h = np.sum(np.diff(lm, axis=1) == 1)
    cr_v = np.sum(np.diff(lm, axis=0) == 1)
    L[0] = lm.shape[0] * lm.shape[1] / max(cr_h, 1)
    L[90] = lm.shape[0] * lm.shape[1] / max(cr_v, 1)
    # diagonals: shift-based transitions
    d1 = (lm[1:, 1:] == 1) & (lm[:-1, :-1] == 0)
    d2 = (lm[1:, :-1] == 1) & (lm[:-1, 1:] == 0)
    ln = (lm.shape[0] - 1) * (lm.shape[1] - 1) * np.sqrt(2)
    L[45] = ln / max(d1.sum(), 1)
    L[135] = ln / max(d2.sum(), 1)
    v = np.array([L[0], L[45], L[90], L[135]])
    f[f"{prefix}_L_mean"] = v.mean()
    f[f"{prefix}_L_max"] = v.max()
    f[f"{prefix}_L_min"] = v.min()
    f[f"{prefix}_L_aniso"] = v.max() / v.min()
    f[f"{prefix}_L_inv_sqrt"] = 1 / np.sqrt(v.mean())


def extract_v2(i):
    raw8 = _read8(i)
    raw = raw8.astype(np.float32)
    f = {"ID": i}
    H, W = raw.shape
    # ------------------------------------------------------------------ image quality
    sig = float(restoration.estimate_sigma(raw))
    f["q_noise"] = sig
    win = np.outer(np.hanning(H), np.hanning(W))
    P = np.abs(np.fft.fftshift(np.fft.fft2((raw - raw.mean()) * win))) ** 2 / (win ** 2).sum()
    prof = _radial_profile(P)
    floor = float(np.median(prof[105:127]))
    f["q_noise_spec"] = np.sqrt(floor)
    sig_prof = np.clip(prof - floor, 1e-3, None)
    for a, b in ((2, 4), (4, 8), (8, 16), (16, 32), (32, 48), (48, 64), (64, 96)):
        f[f"q_spec_{a}_{b}"] = float(np.log10(sig_prof[a:b].mean()))
    f["q_spec_slope_hi"] = f["q_spec_32_48"] - f["q_spec_8_16"]
    f["q_spec_slope_vhi"] = f["q_spec_48_64"] - f["q_spec_16_32"]
    snr_k = np.where(sig_prof < floor * 0.5)[0]
    snr_k = snr_k[snr_k > 4]
    f["q_k_cut"] = float(snr_k[0]) if len(snr_k) else 128.0
    # ------------------------------------------------------------------ denoise
    h_nl = float(np.clip(sig * 1.0, 2.0, 30.0))
    den = cv2.fastNlMeansDenoising(raw8, None, h=h_nl, templateWindowSize=5, searchWindowSize=21).astype(np.float32)
    sm = cv2.GaussianBlur(den, (0, 0), 1.0)
    sm2 = cv2.GaussianBlur(den, (0, 0), 2.0)
    f["q_res_noise"] = float(restoration.estimate_sigma(den))
    # ------------------------------------------------------------------ global levels
    M = _hist_mode(sm)
    f["lv_M"] = M
    for p in (0.1, 1, 5, 25, 50, 75, 95, 99):
        f[f"lv_p{p}"] = float(np.percentile(sm, p)) / M
    f["lv_mean_rel"] = float(sm.mean()) / M
    # illumination non-uniformity: range of a heavy blur
    bg = cv2.GaussianBlur(den, (0, 0), 24.0)
    f["q_illum_range"] = float(np.percentile(bg, 95) - np.percentile(bg, 5)) / M
    # ------------------------------------------------------------------ pores (very dark, compact, crisp)
    pore_masks = {}
    for r in (0.70, 0.60, 0.50):
        m = sm < r * M
        m = ndi.binary_opening(m, structure=DISK1)
        lab = measure.label(m)
        props = [p for p in measure.regionprops(lab, intensity_image=sm) if p.area >= 6]
        keep = np.zeros(lab.max() + 1, bool)
        for p in props:
            keep[p.label] = True
        pore_masks[r] = keep[lab]
        k = f"pore{int(r * 100)}"
        A = np.array([p.area for p in props], float)
        asp = np.array([p.axis_major_length / max(p.axis_minor_length, 1.0) for p in props])
        f[f"{k}_frac"] = A.sum() / (H * W)
        f[f"{k}_n"] = len(A)
        f[f"{k}_n_big"] = int((A >= 20).sum())
        f[f"{k}_n_round"] = int((asp < 2.0).sum())
        f[f"{k}_n_elong"] = int((asp >= 2.0).sum())
        f[f"{k}_frac_round"] = A[asp < 2.0].sum() / (H * W)
        f[f"{k}_frac_elong"] = A[asp >= 2.0].sum() / (H * W)
        f[f"{k}_area_mean"] = A.mean() if len(A) else 0.0
        f[f"{k}_area_max"] = A.max() if len(A) else 0.0
        f[f"{k}_eqr_mean"] = float(np.sqrt(A / np.pi).mean()) if len(A) else 0.0
        f[f"{k}_asp_mean"] = asp.mean() if len(A) else 0.0
        f[f"{k}_depth"] = float(np.mean([p.intensity_min for p in props]) / M) if props else 1.0
    # LoG blob count of dark blobs (scale-normalized), robust count of pores
    pm60 = pore_masks[0.60]
    pore_ex = ndi.binary_dilation(pore_masks[0.70], structure=DISK2, iterations=2)
    # pore-inpainted image (pores -> matrix level) for phase / texture measurements
    inp = sm.copy()
    inp[pore_ex] = M
    inp2 = sm2.copy()
    inp2[pore_ex] = M
    valid = ~pore_ex
    nv = valid.sum()
    # ------------------------------------------------------------------ dark phase (pixel-level)
    for t in (8, 12, 16, 20, 26):
        f[f"ph_px{t}"] = float(((sm < M - t) & valid).sum() / nv)
        f[f"ph_px2_{t}"] = float(((sm2 < M - t) & valid).sum() / nv)
    for r in (0.92, 0.90, 0.87):
        dm = (sm < r * M) & valid
        dm = ndi.binary_opening(dm, structure=DISK1)
        f[f"ph_rel{int(r * 100)}"] = float(dm.sum() / nv)
    f["ph_deficit"] = float(np.clip(M - sm[valid], 0, None).mean()) / M
    f["ph_deficit2"] = float(np.clip(M - sm2[valid], 0, None).mean()) / M
    f["ph_excess"] = float(np.clip(sm[valid] - M, 0, None).mean()) / M
    # dark-phase blobs (opened mask on sigma-2 image) -> size / shape / alignment of dark grains
    dmask = (sm2 < 0.90 * M) & valid
    dmask = ndi.binary_opening(dmask, structure=DISK2)
    dlab = measure.label(dmask)
    dprops = [p for p in measure.regionprops(dlab) if p.area >= 15]
    _shape_stats(dprops, "dk", f)
    f["dk_frac"] = float(sum(p.area for p in dprops) / nv)
    # ------------------------------------------------------------------ grain segmentation (ridge + watershed)
    rid = filters.sato(den, sigmas=[1.0, 1.5], black_ridges=True)
    rs = cv2.GaussianBlur(rid.astype(np.float32), (0, 0), 1.0)
    hmin = 0.15 * float(np.percentile(rs, 99))
    mk = measure.label(morphology.h_minima(rs, hmin))
    ws = segmentation.watershed(rs, mk, watershed_line=True)
    line = ws == 0
    rc = float(rid[line].mean() - np.median(rid[~line]))
    f["q_ridge_contrast"] = rc
    f["q_ridge_snr"] = rc / max(sig, 1e-3)
    f["q_ridge_p99"] = float(np.percentile(rid, 99)) / max(sig, 1e-3)
    # exclude pores from grain stats; classify grains by interior median intensity
    inner = ws.copy()
    inner[ndi.binary_dilation(line, structure=DISK1)] = 0
    gprops_all = measure.regionprops(inner, intensity_image=den)
    gprops = []
    gmed, garea = [], []
    for p in gprops_all:
        if p.area < 8:
            continue
        # pore regions: majority of pixels in pore mask
        sl = p.slice
        if pore_masks[0.60][sl][p.image].mean() > 0.5:
            continue
        gprops.append(p)
        gmed.append(p.intensity_median if hasattr(p, "intensity_median") else np.median(p.image_intensity[p.image]))
        garea.append(p.area)
    gmed, garea = np.array(gmed), np.array(garea, float)
    f["seg_n_regions"] = int(ws.max())
    if len(gmed):
        order = np.argsort(gmed)
        cw = np.cumsum(garea[order]) / garea.sum()
        Mg = float(gmed[order][np.searchsorted(cw, 0.5)])
    else:
        Mg = M
    f["seg_Mg"] = Mg / M
    for t in (12, 15, 20):
        dk = gmed < Mg - t
        f[f"seg_fd{t}"] = float(garea[dk].sum() / max(garea.sum(), 1))
        f[f"seg_nd{t}"] = int(dk.sum())
    dk = gmed < Mg - 15
    f["seg_D_rel"] = float(_wmean(gmed[dk], garea[dk]) / Mg) if dk.sum() >= 2 else np.nan
    mt = gmed >= Mg - 12
    f["seg_matrix_spread"] = float(np.sqrt(_wmean((gmed[mt] - Mg) ** 2, garea[mt]))) if mt.sum() >= 2 else np.nan
    # grain shape stats on full watershed regions (non-pore)
    ws_props = [p for p in measure.regionprops(ws) if p.area >= 20]
    pm_idx = set()
    for p in ws_props:
        if pore_masks[0.60][p.slice][p.image].mean() > 0.5:
            pm_idx.add(p.label)
    ws_props = [p for p in ws_props if p.label not in pm_idx]
    out = _shape_stats(ws_props, "seg", f)
    if out is not None:
        A = out[0]
        f["seg_inv_sqrt_d"] = 1 / np.sqrt(np.sqrt(A.mean()))
        f["seg_count_density"] = len(A) / (H * W) * 1e4
        q = np.percentile(A, [10, 50, 90])
        f["seg_area_p10"], f["seg_area_p90"] = q[0], q[2]
        f["seg_area_p90_p10"] = q[2] / max(q[0], 1)
        # area-weighted (volume) mean vs number mean -> spread / bimodality indicator
        f["seg_area_wmean"] = _wmean(A, A)
        f["seg_area_skew"] = float(((A - A.mean()) ** 3).mean() / (A.std() ** 3 + 1e-9))
    else:
        for k in ("inv_sqrt_d", "count_density", "area_p10", "area_p90", "area_p90_p10", "area_wmean", "area_skew"):
            f[f"seg_{k}"] = np.nan
    # dark-grain shape stats (segmented dark grains)
    dk_lab = [p for p, m_ in zip(gprops, gmed) if m_ < Mg - 15]
    _shape_stats(dk_lab, "segdk", f)
    _intercepts(line, f, "seg")
    # boundary density (watershed line length per area)
    f["seg_line_frac"] = float(line.mean())
    # ------------------------------------------------------------------ autocorrelation on pore-inpainted image
    hp = inp - cv2.GaussianBlur(inp, (0, 0), 32.0)
    ell, acp = _ac_ellipse(hp)
    for lv, (a1, a2, area, ang) in ell.items():
        k = f"ac{int(lv * 100)}"
        f[f"{k}_major"], f[f"{k}_minor"] = a1, a2
        f[f"{k}_aspect"] = a1 / a2 if a2 and a2 == a2 else np.nan
        f[f"{k}_gm"] = np.sqrt(a1 * a2) if a2 == a2 else np.nan
    for lag in (2, 4, 6, 8, 12, 16, 24, 32):
        f[f"acp_{lag}"] = float(acp[lag])
    f["hp_std"] = float(hp[valid].std()) / M
    # same on stronger-smoothed image (degraded images)
    hp2 = inp2 - cv2.GaussianBlur(inp2, (0, 0), 32.0)
    ell2, acp2 = _ac_ellipse(hp2, levels=(0.5,))
    a1, a2, area, ang = ell2[0.5]
    f["ac2_major"], f["ac2_minor"] = a1, a2
    f["ac2_aspect"] = a1 / a2 if a2 == a2 else np.nan
    f["ac2_gm"] = np.sqrt(a1 * a2) if a2 == a2 else np.nan
    # ------------------------------------------------------------------ structure tensor anisotropy (multi-scale)
    for s_in, s_out in ((1.0, 4.0), (2.0, 8.0), (3.0, 16.0)):
        g = cv2.GaussianBlur(inp, (0, 0), s_in)
        gy, gx = np.gradient(g)
        Jxx, Jyy, Jxy = (gx * gx)[valid].mean(), (gy * gy)[valid].mean(), (gx * gy)[valid].mean()
        tr = Jxx + Jyy
        disc = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2)
        f[f"st{int(s_in)}_coh"] = disc / (tr + 1e-9)
        # local coherence averaged (local alignment, independent of global orientation spread)
        Lxx = cv2.GaussianBlur(gx * gx, (0, 0), s_out)
        Lyy = cv2.GaussianBlur(gy * gy, (0, 0), s_out)
        Lxy = cv2.GaussianBlur(gx * gy, (0, 0), s_out)
        ltr = Lxx + Lyy
        ldisc = np.sqrt((Lxx - Lyy) ** 2 + 4 * Lxy ** 2)
        f[f"st{int(s_in)}_lcoh"] = float((ldisc / (ltr + 1e-9))[valid].mean())
        f[f"st{int(s_in)}_energy"] = float(np.sqrt(tr)) / M
    return f


# =====================================================================================================
# v3 block ("ic_" = illumination-corrected). Every intensity is divided by a local matrix level
# (70th-percentile filter over ~80 px), which removes the shading that otherwise dominates global
# thresholds on noisy images. Adds dark-phase chord lengths / autocorrelation along the principal axes.
# =====================================================================================================
def _bg_matrix(den, pct=70, win=20, sub=4):
    small = cv2.resize(den, (den.shape[1] // sub, den.shape[0] // sub), interpolation=cv2.INTER_AREA)
    b = ndi.percentile_filter(small, pct, size=win, mode="reflect")
    b = cv2.GaussianBlur(b, (0, 0), win / 3)
    return cv2.resize(b, den.shape[::-1], interpolation=cv2.INTER_CUBIC)


def _rotate(img, angle_rad, interp=cv2.INTER_NEAREST, border=0):
    """Rotate so that the axis at angle (x-axis convention, counter-clockwise in image coords) becomes horizontal."""
    h, w = img.shape
    R = cv2.getRotationMatrix2D((w / 2, h / 2), -np.degrees(angle_rad), 1.0)
    return cv2.warpAffine(img, R, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def _runs(mask_rows, valid_rows):
    """Mean run length of True runs along rows, only counting runs fully inside the valid area."""
    lengths = []
    for m, v in zip(mask_rows, valid_rows):
        if not v.any():
            continue
        idx = np.where(v)[0]
        seg = m[idx[0]:idx[-1] + 1].astype(np.int8)
        if seg.size < 3:
            continue
        d = np.diff(np.concatenate([[0], seg, [0]]))
        st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        # drop runs touching the ends (censored)
        for s_, e_ in zip(st, en):
            if s_ == 0 or e_ == seg.size:
                continue
            lengths.append(e_ - s_)
    return np.array(lengths, float)


def _chords(mask, valid, angle, prefix, f):
    """Chord (run) lengths of a binary phase mask along the principal axis and perpendicular to it."""
    for tag, ang in (("par", angle), ("perp", angle + np.pi / 2)):
        m = _rotate(mask.astype(np.uint8), ang) > 0
        v = _rotate(valid.astype(np.uint8), ang, border=0) > 0
        v = ndi.binary_erosion(v, iterations=2)
        m &= v
        runs_in = _runs(m, v)
        runs_out = _runs(~m & v, v)
        f[f"{prefix}_chord_{tag}"] = runs_in.mean() if len(runs_in) else np.nan
        f[f"{prefix}_chord_{tag}_med"] = float(np.median(runs_in)) if len(runs_in) else np.nan
        f[f"{prefix}_gap_{tag}"] = runs_out.mean() if len(runs_out) else np.nan
        f[f"{prefix}_nchord_{tag}"] = len(runs_in) / max(v.sum(), 1) * 1e3
    a, b = f[f"{prefix}_chord_par"], f[f"{prefix}_chord_perp"]
    f[f"{prefix}_chord_aspect"] = a / b if (a == a and b == b and b > 0) else np.nan


def _ac2d(x):
    F = np.fft.fft2(x - x.mean())
    ac = np.fft.fftshift(np.real(np.fft.ifft2(F * np.conj(F))))
    return ac


def _ac_features(x, angle, prefix, f, norm_lag1=True, levels=(0.5, 0.3)):
    """Autocorrelation lengths along/perpendicular to the principal axis + lobe ellipse."""
    ac = _ac2d(x)
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    if norm_lag1:  # lag-1 normalisation removes the white-noise spike at lag 0
        c = 0.25 * (ac[cy - 1, cx] + ac[cy + 1, cx] + ac[cy, cx - 1] + ac[cy, cx + 1])
    else:
        c = ac[cy, cx]
    ac = ac / (c + 1e-12)
    t = np.arange(0, 80, 0.5)
    for tag, ang in (("par", angle), ("perp", angle + np.pi / 2)):
        ys, xs = cy - t * np.sin(ang), cx + t * np.cos(ang)
        prof = ndi.map_coordinates(ac, [ys, xs], order=1)
        ys2, xs2 = cy + t * np.sin(ang), cx - t * np.cos(ang)
        prof = 0.5 * (prof + ndi.map_coordinates(ac, [ys2, xs2], order=1))
        for lv in levels:
            below = np.where(prof[2:] < lv)[0]
            f[f"{prefix}_len{int(lv * 100)}_{tag}"] = float(t[2:][below[0]]) if len(below) else 80.0
        f[f"{prefix}_ac8_{tag}"] = float(prof[16])
        f[f"{prefix}_ac16_{tag}"] = float(prof[32])
    for lv in levels:
        a, b = f[f"{prefix}_len{int(lv * 100)}_par"], f[f"{prefix}_len{int(lv * 100)}_perp"]
        f[f"{prefix}_len{int(lv * 100)}_aspect"] = a / max(b, 0.5)
        f[f"{prefix}_len{int(lv * 100)}_gm"] = np.sqrt(a * b)
    # lobe ellipse at 0.5
    win = 64
    sub = ac[cy - win:cy + win + 1, cx - win:cx + win + 1]
    lab = measure.label(sub > 0.5)
    if lab[win, win] > 0:
        m = lab == lab[win, win]
        f[f"{prefix}_lobe_area"] = float(m.sum())
    else:
        f[f"{prefix}_lobe_area"] = 0.0


def _quality_v3(raw):
    """Noise level and blur-sensitive spectral descriptors (per image)."""
    H, W = raw.shape
    sig = float(restoration.estimate_sigma(raw))
    win = np.outer(np.hanning(H), np.hanning(W))
    P = np.abs(np.fft.fftshift(np.fft.fft2((raw - raw.mean()) * win))) ** 2 / (win ** 2).sum()
    prof = _radial_profile(P)
    floor = float(np.median(prof[105:127]))
    sp = np.clip(prof - floor, 1e-3, None)
    q = {"ic_noise": sig}
    for a, b in ((4, 8), (8, 16), (16, 32), (32, 48)):
        q[f"ic_spec_{a}_{b}"] = float(np.log10(sp[a:b].mean()))
    q["ic_spec_slope"] = q["ic_spec_32_48"] - q["ic_spec_8_16"]
    return q


def _quality_row(i):
    return {"ID": i, **_quality_v3(_read8(i).astype(np.float32))}


def extract_v3(i, img=None):
    raw8 = _read8(i) if img is None else img
    raw = raw8.astype(np.float32)
    H, W = raw.shape
    f = {"ID": i}
    f.update(_quality_v3(raw))
    sig = f["ic_noise"]
    den = cv2.fastNlMeansDenoising(raw8, None, h=float(np.clip(sig, 2.0, 30.0)), templateWindowSize=5,
                                   searchWindowSize=21).astype(np.float32)
    bg = _bg_matrix(den)
    f["ic_bg_range"] = float(np.percentile(bg, 98) - np.percentile(bg, 2)) / float(np.median(bg))
    f["ic_bg_level"] = float(np.median(bg))
    rn = den / np.maximum(bg, 1.0)
    sm1 = cv2.GaussianBlur(rn, (0, 0), 1.0)
    sm2 = cv2.GaussianBlur(rn, (0, 0), 2.0)
    # ---------------------------------------------------------------- pores
    pmask = {}
    for t in (0.68, 0.60, 0.52):
        m = ndi.binary_opening(sm1 < t, structure=DISK1)
        lab = measure.label(m)
        props = [p for p in measure.regionprops(lab, intensity_image=sm1) if p.area >= 6]
        keep = np.zeros(lab.max() + 1, bool)
        for p in props:
            keep[p.label] = True
        pmask[t] = keep[lab]
        k = f"ic_pore{int(t * 100)}"
        A = np.array([p.area for p in props], float)
        asp = np.array([p.axis_major_length / max(p.axis_minor_length, 1.0) for p in props])
        f[f"{k}_frac"] = A.sum() / (H * W)
        f[f"{k}_n"] = len(A)
        f[f"{k}_n_round"] = int((asp < 2.0).sum())
        f[f"{k}_n_elong"] = int((asp >= 2.0).sum())
        f[f"{k}_frac_round"] = A[asp < 2.0].sum() / (H * W) if len(A) else 0.0
        f[f"{k}_area_mean"] = A.mean() if len(A) else 0.0
        f[f"{k}_eqr_mean"] = float(np.sqrt(A / np.pi).mean()) if len(A) else 0.0
        f[f"{k}_depth"] = float(np.mean([p.intensity_min for p in props])) if props else 1.0
    # integrated darkness of pore-like blobs (blur-invariant 'pore volume')
    pm = pmask[0.68]
    f["ic_pore_deficit"] = float(np.clip(1.0 - sm1[pm], 0, None).sum()) / (H * W)
    pore_ex = ndi.binary_dilation(pm, structure=DISK2, iterations=2)
    valid = ~pore_ex
    nv = max(valid.sum(), 1)
    # ---------------------------------------------------------------- dark phase fraction
    v2 = sm2[valid]
    v1 = sm1[valid]
    for t in (0.95, 0.93, 0.91, 0.89, 0.87, 0.85):
        f[f"ic_fd2_{int(t * 100)}"] = float((v2 < t).mean())
        m = ndi.binary_opening((sm2 < t) & valid, structure=DISK2)
        f[f"ic_fdo_{int(t * 100)}"] = float(m.sum() / nv)
    for t in (0.92, 0.89):
        f[f"ic_fd1_{int(t * 100)}"] = float((v1 < t).mean())
    f["ic_fd_soft"] = float(np.clip((0.96 - v2) / 0.12, 0, 1).mean())
    f["ic_deficit"] = float(np.clip(1.0 - v2, 0, None).mean())
    f["ic_std"] = float(v2.std())
    for p in (2, 5, 10, 20, 30, 50, 90, 98):
        f[f"ic_p{p}"] = float(np.percentile(v2, p))
    # 2-component GMM on intensities (matrix ~1, dark ~0.8)
    try:
        from sklearn.mixture import GaussianMixture
        xs = v2[:: max(1, len(v2) // 6000)].reshape(-1, 1)
        g = GaussianMixture(2, means_init=[[1.0], [0.82]], random_state=0, max_iter=200).fit(xs)
        mu = g.means_.ravel()
        lo = int(np.argmin(mu))
        f["ic_gmm_w"] = float(g.weights_[lo]) if mu[lo] < 0.95 else 0.0
        f["ic_gmm_mu_lo"], f["ic_gmm_mu_hi"] = float(mu[lo]), float(mu[1 - lo])
        f["ic_gmm_sd_lo"] = float(np.sqrt(g.covariances_.ravel()[lo]))
        f["ic_gmm_sd_hi"] = float(np.sqrt(g.covariances_.ravel()[1 - lo]))
        f["ic_gmm_sep"] = (f["ic_gmm_mu_hi"] - f["ic_gmm_mu_lo"]) / (f["ic_gmm_sd_lo"] + f["ic_gmm_sd_hi"])
    except Exception:
        for k in ("w", "mu_lo", "mu_hi", "sd_lo", "sd_hi", "sep"):
            f[f"ic_gmm_{k}"] = np.nan
    # ---------------------------------------------------------------- principal direction (structure tensor)
    gy, gx = np.gradient(cv2.GaussianBlur(np.where(valid, rn, 1.0).astype(np.float32), (0, 0), 2.0))
    Jxx, Jyy, Jxy = (gx * gx)[valid].mean(), (gy * gy)[valid].mean(), (gx * gy)[valid].mean()
    # gradient orientation is perpendicular to the elongation axis
    grad_ang = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
    elong_ang = grad_ang + np.pi / 2  # angle of grain elongation (x-axis, image y pointing down)
    f["ic_st_coh"] = float(np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / (Jxx + Jyy + 1e-12))
    f["ic_angle_abs"] = float(abs(np.sin(elong_ang)))  # |sin| of elongation axis angle (0 = horizontal)
    # convert to row/col convention for rotation & sampling: image y down -> visual angle = -elong_ang
    ang = -elong_ang
    # ---------------------------------------------------------------- dark-phase mask morphology
    dmask = ndi.binary_opening((sm2 < 0.90) & valid, structure=DISK2)
    _chords(dmask, valid, ang, "ic_dk", f)
    dl = measure.label(dmask)
    dprops = [p for p in measure.regionprops(dl) if p.area >= 15]
    out = _shape_stats(dprops, "ic_dk", f)
    if out is not None:
        A, maj, mi, th, asp = out
        f["ic_dk_maj_wmean"] = _wmean(maj, A)
        f["ic_dk_min_wmean"] = _wmean(mi, A)
        f["ic_dk_count_density"] = len(A) / nv * 1e4
    else:
        f["ic_dk_maj_wmean"] = f["ic_dk_min_wmean"] = np.nan
        f["ic_dk_count_density"] = 0.0
    # ---------------------------------------------------------------- autocorrelation (grey + binary)
    x = np.where(valid, rn - 1.0, 0.0).astype(np.float32)
    x = x - cv2.GaussianBlur(x, (0, 0), 32.0)
    _ac_features(x, ang, "ic_acg", f)
    _ac_features(dmask.astype(np.float32), ang, "ic_acb", f, norm_lag1=False)
    # ---------------------------------------------------------------- grain segmentation on denoised image
    rid = filters.sato(den, sigmas=[1.0, 1.5], black_ridges=True)
    rs = cv2.GaussianBlur(rid.astype(np.float32), (0, 0), 1.0)
    mk = measure.label(morphology.h_minima(rs, 0.15 * float(np.percentile(rs, 99))))
    ws = segmentation.watershed(rs, mk, watershed_line=True)
    line = ws == 0
    f["ic_ridge_snr"] = float(rid[line].mean() - np.median(rid[~line])) / max(sig, 1e-3)
    inner = ws.copy()
    inner[ndi.binary_dilation(line, structure=DISK1)] = 0
    idx = np.arange(1, ws.max() + 1)
    med = np.asarray(ndi.median(rn, inner, idx))
    area = np.asarray(ndi.sum(np.ones_like(rn), inner, idx))
    porefrac = np.asarray(ndi.mean(pmask[0.60].astype(np.float32), inner, idx))
    ok = (area >= 8) & ~(porefrac > 0.5)
    med, area = med[ok], area[ok]
    if area.sum() > 0:
        for t in (0.93, 0.91, 0.89, 0.87):
            f[f"ic_seg_fd{int(t * 100)}"] = float(area[med < t].sum() / area.sum())
        f["ic_seg_nd91"] = int((med < 0.91).sum())
        f["ic_seg_nfrac91"] = float((med < 0.91).mean())
        dk = med < 0.91
        f["ic_seg_dk_area_mean"] = float(area[dk].mean()) if dk.any() else 0.0
        f["ic_seg_mx_area_mean"] = float(area[~dk].mean()) if (~dk).any() else 0.0
        f["ic_seg_dk_level"] = float(_wmean(med[dk], area[dk])) if dk.sum() >= 2 else np.nan
        mt = med >= 0.93
        f["ic_seg_mx_spread"] = float(np.sqrt(_wmean((med[mt] - _wmean(med[mt], area[mt])) ** 2, area[mt]))) if mt.sum() >= 2 else np.nan
    else:
        for k in ("fd93", "fd91", "fd89", "fd87", "nd91", "nfrac91", "dk_area_mean", "mx_area_mean", "dk_level", "mx_spread"):
            f[f"ic_seg_{k}"] = np.nan
    # linear intercepts of boundary network along / across the principal axis
    for tag, a_ in (("par", ang), ("perp", ang + np.pi / 2)):
        lm = _rotate(line.astype(np.uint8), a_) > 0
        vv = ndi.binary_erosion(_rotate(np.ones_like(line, np.uint8), a_) > 0, iterations=2)
        cr = np.sum(np.diff((lm & vv).astype(np.int8), axis=1) == 1)
        f[f"ic_seg_L_{tag}"] = float(vv.sum() / max(cr, 1))
    f["ic_seg_L_aspect"] = f["ic_seg_L_par"] / max(f["ic_seg_L_perp"], 1e-3)
    f["ic_seg_L_gm"] = float(np.sqrt(f["ic_seg_L_par"] * f["ic_seg_L_perp"]))
    return f


def build(ids, fn=extract, n_jobs=-1):
    return pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(fn)(i) for i in ids))


# =====================================================================================================
# Measurement calibration (label-free). Clean TRAIN images (high ridge SNR) are synthetically degraded
# (blur of the grain layer with pores kept crisp, smooth shading, contrast jitter, additive noise) and
# v3 features are recomputed. Regressors learn degraded-features -> clean-image measurements (dark-phase
# fraction from grain segmentation, correlation lengths, pore counts, grain-size stats). Hardness labels are
# never used and no test image is used for fitting; the fitted maps are then applied per image.
#   python -m src.features --cal_build   -> data/cal_pairs.parquet   (synthetic pairs, train images only)
#   python -m src.features --cal_apply   -> data/features_cal.parquet
# =====================================================================================================
CAL_TARGETS_V3 = ["ic_seg_fd93", "ic_seg_fd91", "ic_seg_fd89", "ic_fdo_93", "ic_gmm_w", "ic_acg_len50_par",
                  "ic_acg_len50_perp", "ic_acg_len30_par", "ic_acg_len30_perp", "ic_pore60_n", "ic_pore60_frac",
                  "ic_pore68_n", "ic_dk_maj_wmean", "ic_dk_min_wmean", "ic_dk_area_cv", "ic_seg_L_par",
                  "ic_seg_L_perp", "ic_seg_L_gm", "ic_st_coh", "ic_seg_nfrac91", "ic_seg_dk_area_mean",
                  "ic_seg_mx_area_mean"]
CAL_TARGETS_V2 = ["seg_count_density", "seg_asp_wmean", "seg_area_cv", "seg_ori_R", "segdk_area_cv",
                  "seg_inv_sqrt_d", "seg_elong_align"]


def _degrade(img8, rng):
    img = img8.astype(np.float32)
    den = cv2.fastNlMeansDenoising(img8, None, h=8.0, templateWindowSize=5, searchWindowSize=21).astype(np.float32)
    pore = ndi.binary_dilation(cv2.GaussianBlur(den / _bg_matrix(den), (0, 0), 1.0) < 0.68, iterations=1)
    sb = rng.uniform(0.0, 2.5)
    p = {"sb": sb}
    if sb > 0.3:
        bl = cv2.GaussianBlur(img, (0, 0), sb)
        img = np.where(pore, img, bl)  # pores stay crisp (as observed on real degraded images)
    a = rng.uniform(0.0, 0.07)
    field = cv2.GaussianBlur(rng.standard_normal(img.shape).astype(np.float32), (0, 0), 40.0)
    field /= field.std() + 1e-9
    img = img * (1 + a * field)
    c, b = rng.uniform(0.85, 1.15), rng.uniform(-10, 10)
    img = (img - img.mean()) * c + img.mean() + b
    sn = rng.uniform(0.0, 18.0)
    img = img + rng.normal(0, sn, img.shape)
    p.update(illum=a, contrast=c, noise_add=sn)
    return np.clip(np.round(img), 0, 255).astype(np.uint8), p


def _degrade_v2(img8, rng):
    """Calibration v2 degradation, matched to real train images (descriptor medians per noise band).

    A latent quality level q sets the total noise (4..21). Blur of the grain layer and loss of dark-side
    contrast grow with q: real noisy images are blurrier, and their dark phase/boundaries sit closer to
    the matrix level while the bright-side grain-to-grain spread is preserved. Pores are pasted back crisp.
    """
    img = img8.astype(np.float32)
    sig0 = float(restoration.estimate_sigma(img))
    den = cv2.fastNlMeansDenoising(img8, None, h=float(np.clip(sig0, 2.0, 30.0)), templateWindowSize=5,
                                   searchWindowSize=21).astype(np.float32)
    bg = _bg_matrix(den)
    pore = ndi.binary_dilation(cv2.GaussianBlur(den / bg, (0, 0), 1.0) < 0.68, iterations=1)
    q = rng.uniform(0.0, 1.0)
    sig_t = 4.0 + 17.0 * q                                   # total noise 4..21 grey levels
    sb = rng.uniform(0.0, 0.6 + 2.6 * q ** 2)                # blur grows mainly for the noisiest images
    c = rng.uniform(max(0.3, 1.0 - 0.8 * q ** 1.5), 1.0)     # dark-side contrast factor
    d = den - bg
    grain = bg + np.where(d < 0, c * d, d)                   # compress only the darker-than-matrix side
    grain = grain + (img - den)                               # keep the source's own residual noise
    if sb > 0.3:
        grain = cv2.GaussianBlur(grain, (0, 0), sb)
    img = np.where(pore, img, grain)
    a = rng.uniform(0.0, 0.035)
    field = cv2.GaussianBlur(rng.standard_normal(img.shape).astype(np.float32), (0, 0), 40.0)
    field /= field.std() + 1e-9
    img = img * (1 + a * field)
    gain, off = rng.uniform(0.85, 1.15), rng.uniform(-12, 12)
    img = (img - img.mean()) * gain + img.mean() + off
    sn = float(np.sqrt(max(sig_t ** 2 - sig0 ** 2, 0.0)))
    img = img + rng.normal(0, sn, img.shape)
    # optional mildly correlated noise component (real noisy images keep more spread after denoising)
    cn = rng.uniform(0.0, 0.35) * sig_t if rng.uniform() < 0.5 else 0.0
    if cn > 0:
        g = cv2.GaussianBlur(rng.standard_normal(img.shape).astype(np.float32), (0, 0), 1.2)
        img = img + cn * g / (g.std() + 1e-9)
    return np.clip(np.round(img), 0, 255).astype(np.uint8), {"q": q, "sb": sb, "contrast": c, "noise_t": sig_t,
                                                               "illum": a, "corr_noise": cn}


def _cal_one_v2(i, k, seed):
    rng = np.random.default_rng(seed)
    d, p = _degrade_v2(_read8(i), rng)
    f = extract_v3(i, img=d)
    f.update({f"aug_{kk}": v for kk, v in p.items()})
    f["src"] = i
    f["k"] = k
    return f


def _cal_one(i, k, seed):
    rng = np.random.default_rng(seed)
    img8 = _read8(i)
    d, p = _degrade(img8, rng)
    f = extract_v3(i, img=d)
    f.update({f"aug_{kk}": v for kk, v in p.items()})
    f["src"] = i
    f["k"] = k
    return f


def patch_v3_quality(n_jobs=1):
    """Add the _quality_v3 columns to an existing features_v3.parquet (same code as extract_v3)."""
    v3 = pd.read_parquet(DATA_DIR / "features_v3.parquet")
    q = pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(_quality_row)(i) for i in v3.ID))
    v3 = v3.drop(columns=[c for c in q.columns if c != "ID" and c in v3.columns]).merge(q, on="ID")
    v3.to_parquet(DATA_DIR / "features_v3.parquet", index=False)
    print(v3.shape)


def cal_build(n_aug=8, snr_min=0.9, n_jobs=1, version=1, out=None, seed0=0):
    v3 = pd.read_parquet(DATA_DIR / "features_v3.parquet")
    tr = pd.read_csv(DATA_DIR / "train.csv")
    src = v3[v3.ID.isin(tr.ID) & (v3.ic_ridge_snr > snr_min)].ID.tolist()
    fn = _cal_one if version == 1 else _cal_one_v2
    jobs = [(i, k, seed0 + 1000 * n + k) for n, i in enumerate(src) for k in range(n_aug)]
    rows = Parallel(n_jobs=n_jobs)(delayed(fn)(*j) for j in jobs)
    df = pd.DataFrame(rows)
    out = out or ("cal_pairs.parquet" if version == 1 else "cal2_pairs.parquet")
    df.to_parquet(DATA_DIR / out, index=False)
    print(df.shape, "sources", len(src), "->", out)


def _cal_inputs(df):
    return [c for c in df.columns if c.startswith("ic_")]


def cal_apply(eval_only=False):
    from sklearn.model_selection import GroupKFold
    pairs = pd.read_parquet(DATA_DIR / "cal_pairs.parquet")
    v3 = pd.read_parquet(DATA_DIR / "features_v3.parquet")
    v2 = pd.read_parquet(DATA_DIR / "features_v2.parquet")
    clean = v3.merge(v2[["ID"] + CAL_TARGETS_V2], on="ID").set_index("ID")
    # also train on the clean sources themselves (identity pairs)
    srcs = pairs.src.unique()
    ident = v3[v3.ID.isin(srcs)].copy()
    ident["src"] = ident.ID
    X = pd.concat([pairs[_cal_inputs(v3)], ident[_cal_inputs(v3)]], ignore_index=True)
    groups = np.concatenate([pairs.src.values, ident.src.values])
    targets = CAL_TARGETS_V3 + CAL_TARGETS_V2
    Y = clean.loc[groups, targets].reset_index(drop=True)
    Xall = v3[_cal_inputs(v3)]
    out = pd.DataFrame({"ID": v3.ID})
    report = {}
    for t in targets:
        y = Y[t].values.astype(float)
        ok = np.isfinite(y)
        mk = lambda: lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=1)
        oof = np.full(len(y), np.nan)
        for a, b in GroupKFold(5).split(X[ok], y[ok], groups[ok]):
            ia, ib = np.where(ok)[0][a], np.where(ok)[0][b]
            m = mk().fit(X.iloc[ia], y[ia])
            oof[ib] = m.predict(X.iloc[ib])
        deg = pairs.index  # rows of degraded copies only
        yo, po = y[: len(pairs)], oof[: len(pairs)]
        okd = np.isfinite(yo) & np.isfinite(po)
        r2 = 1 - np.mean((po[okd] - yo[okd]) ** 2) / np.var(yo[okd])
        raw_r = np.corrcoef(pairs[t].values[okd], yo[okd])[0, 1] if t in pairs.columns else np.nan
        report[t] = (round(float(r2), 3), round(float(raw_r), 3) if raw_r == raw_r else None)
        if not eval_only:
            m = mk().fit(X[ok], y[ok])
            out[f"cal_{t}"] = m.predict(Xall)
    for t, (r2, rr) in report.items():
        print(f"{t:22s} calibrated R2 on held-out sources {r2:+.3f}   raw degraded-vs-clean corr {rr}")
    if not eval_only:
        out.to_parquet(DATA_DIR / "features_cal.parquet", index=False)
        print(out.shape, "-> features_cal.parquet")


CAL_FAILED = ["ic_seg_L_par", "ic_seg_L_perp", "ic_seg_L_gm", "ic_seg_mx_area_mean", "seg_area_cv", "segdk_area_cv"]
CAL2_BANDS = [(0.0, 8.5), (8.5, 13.0), (13.0, 99.0)]  # applied by the image's own noise estimate (ic_noise)
DOMAIN_COLS = ["ic_noise", "ic_spec_slope", "ic_spec_32_48", "ic_acg_ac8_perp", "ic_acg_ac8_par", "ic_std",
               "ic_gmm_sep", "ic_gmm_sd_hi", "ic_gmm_mu_lo", "ic_p2", "ic_p98", "ic_bg_range", "ic_ridge_snr"]


def _domain_weights(syn, real, clip=(0.1, 10.0)):
    """Density-ratio weights making synthetic quality descriptors look like real TRAIN images (no labels)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    A = pd.concat([syn[DOMAIN_COLS], real[DOMAIN_COLS]], ignore_index=True)
    A = A.fillna(A.median())
    z = np.r_[np.zeros(len(syn)), np.ones(len(real))]
    sc = StandardScaler().fit(A)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(A), z)
    p = clf.predict_proba(sc.transform(A.iloc[: len(syn)]))[:, 1]
    w = p / (1 - p) * len(syn) / len(real)
    w = np.clip(w, *clip)
    return w / w.mean()


def _lgb_cal(n_jobs=1):
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.04, num_leaves=15, min_child_samples=10,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=n_jobs)


def cal_apply_v2(pairs_file="cal2_pairs.parquet", bands=True, reweight=True, out="features_cal2.parquet",
                 eval_v1=True):
    from sklearn.model_selection import GroupKFold
    pairs = pd.read_parquet(DATA_DIR / pairs_file)
    v3 = pd.read_parquet(DATA_DIR / "features_v3.parquet")
    v2 = pd.read_parquet(DATA_DIR / "features_v2.parquet")
    tr_ids = set(pd.read_csv(DATA_DIR / "train.csv").ID)
    clean = v3.merge(v2[["ID"] + CAL_TARGETS_V2], on="ID").set_index("ID")
    targets = [t for t in CAL_TARGETS_V3 + CAL_TARGETS_V2 if t not in CAL_FAILED]
    inputs = _cal_inputs(v3)
    ident = v3[v3.ID.isin(pairs.src.unique())].copy()
    ident["src"] = ident.ID
    S = pd.concat([pairs[inputs + ["src"]], ident[inputs + ["src"]]], ignore_index=True)
    groups = S.src.values
    Y = clean.loc[groups, targets].reset_index(drop=True)
    w_syn = _domain_weights(S, v3[v3.ID.isin(tr_ids)]) if reweight else np.ones(len(S))
    print(f"domain weights: min {w_syn.min():.2f} max {w_syn.max():.2f} ESS {w_syn.sum() ** 2 / (w_syn ** 2).sum():.0f}/{len(S)}")
    def band_masks(noise, margin=0.0):
        return [(noise >= lo - margin) & (noise < hi + margin) for lo, hi in (CAL2_BANDS if bands else [(0, 99)])]
    tr_masks = band_masks(S.ic_noise.values, margin=1.5)
    ap_masks_S = band_masks(S.ic_noise.values)
    ap_masks_all = band_masks(v3.ic_noise.values)
    n_deg = len(pairs)
    res = {}
    out_df = pd.DataFrame({"ID": v3.ID})
    folds = list(GroupKFold(5).split(S, groups=groups))
    for t in targets:
        y = Y[t].values.astype(float)
        ok = np.isfinite(y)
        oof = np.full(len(y), np.nan)
        for a, b in folds:
            for tm, am in zip(tr_masks, ap_masks_S):
                ia = np.intersect1d(a, np.where(ok & tm)[0])
                ib = np.intersect1d(b, np.where(am)[0])
                if len(ib) == 0 or len(ia) < 30:
                    continue
                m = _lgb_cal().fit(S.iloc[ia][inputs], y[ia], sample_weight=w_syn[ia])
                oof[ib] = m.predict(S.iloc[ib][inputs])
        yo, po, wo = y[:n_deg], oof[:n_deg], w_syn[:n_deg]
        k = np.isfinite(yo) & np.isfinite(po)
        r2 = 1 - np.mean((po[k] - yo[k]) ** 2) / np.var(yo[k])
        r2w = 1 - np.average((po[k] - yo[k]) ** 2, weights=wo[k]) / np.cov(yo[k], aweights=wo[k])
        hi = k & (pairs.ic_noise.values >= 13)
        r2_hi = 1 - np.mean((po[hi] - yo[hi]) ** 2) / np.var(yo[hi]) if hi.sum() > 20 else np.nan
        res[t] = {"R2": round(float(r2), 3), "R2_w": round(float(r2w), 3), "R2_noise13+": round(float(r2_hi), 3)}
        pred = np.full(len(v3), np.nan)
        for tm, am in zip(tr_masks, ap_masks_all):
            ia = np.where(ok & tm)[0]
            m = _lgb_cal().fit(S.iloc[ia][inputs], y[ia], sample_weight=w_syn[ia])
            pred[am] = m.predict(v3.loc[am, inputs])
        out_df[f"cal2_{t}"] = pred
    if eval_v1 and (DATA_DIR / "cal_pairs.parquet").exists():
        # v1 maps (trained on v1 degradations) evaluated on the realistic v2 degradations, same held-out sources
        p1 = pd.read_parquet(DATA_DIR / "cal_pairs.parquet")
        S1 = pd.concat([p1[inputs + ["src"]], ident[inputs + ["src"]]], ignore_index=True)
        Y1 = clean.loc[S1.src.values, targets].reset_index(drop=True)
        for t in targets:
            y1, y2 = Y1[t].values.astype(float), Y[t].values.astype(float)
            po = np.full(n_deg, np.nan)
            for a, b in folds:
                held = set(groups[b])
                tr1 = np.where(np.isfinite(y1) & ~S1.src.isin(held).values)[0]
                m = _lgb_cal().fit(S1.iloc[tr1][inputs], y1[tr1])
                ib = np.intersect1d(b, np.arange(n_deg))
                po[ib] = m.predict(S.iloc[ib][inputs])
            k = np.isfinite(po) & np.isfinite(y2[:n_deg])
            res[t]["R2_v1map_on_v2"] = round(float(1 - np.mean((po[k] - y2[:n_deg][k]) ** 2) / np.var(y2[:n_deg][k])), 3)
    rep = pd.DataFrame(res).T
    print(rep.to_string())
    rep.to_csv(DATA_DIR / (out.replace(".parquet", "_report.csv")))
    out_df.to_parquet(DATA_DIR / out, index=False)
    print(out_df.shape, "->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--v3", action="store_true")
    ap.add_argument("--cal_build", action="store_true")
    ap.add_argument("--patch_v3_quality", action="store_true")
    ap.add_argument("--cal_apply", action="store_true")
    ap.add_argument("--cal_eval", action="store_true")
    ap.add_argument("--cal2_apply", action="store_true")
    ap.add_argument("--no_bands", action="store_true")
    ap.add_argument("--no_reweight", action="store_true")
    ap.add_argument("--cal2_pairs", default="cal2_pairs.parquet")
    ap.add_argument("--n_aug", type=int, default=8)
    ap.add_argument("--snr_min", type=float, default=0.9)
    ap.add_argument("--cal_version", type=int, default=1)
    ap.add_argument("--cal_out", default=None)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--n_jobs", type=int, default=int(os.environ.get("N_JOBS", "-1")))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.patch_v3_quality:
        patch_v3_quality(n_jobs=a.n_jobs)
        raise SystemExit
    if a.cal_build:
        cal_build(n_aug=a.n_aug, snr_min=a.snr_min, n_jobs=a.n_jobs, version=a.cal_version, out=a.cal_out,
                  seed0=a.seed0)
        raise SystemExit
    if a.cal2_apply:
        cal_apply_v2(a.cal2_pairs, bands=not a.no_bands, reweight=not a.no_reweight,
                     out=a.cal_out or "features_cal2.parquet")
        raise SystemExit
    if a.cal_apply or a.cal_eval:
        cal_apply(eval_only=a.cal_eval)
        raise SystemExit
    tr = pd.read_csv(DATA_DIR / "train.csv")
    ids = list(tr.ID) + list(load_test().ID)
    if a.limit:
        ids = ids[: a.limit]
    if a.v3:
        df = build(ids, extract_v3, n_jobs=a.n_jobs)
        out = DATA_DIR / ("features_v3.parquet" if not a.limit else "features_v3_sample.parquet")
    elif a.v2:
        df = build(ids, extract_v2, n_jobs=a.n_jobs)
        out = DATA_DIR / ("features_v2.parquet" if not a.limit else "features_v2_sample.parquet")
    else:
        df = build(ids, extract, n_jobs=a.n_jobs)
        out = DATA_DIR / "features.parquet"
    df.to_parquet(out, index=False)
    print(df.shape, "->", out)
