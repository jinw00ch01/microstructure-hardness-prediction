"""d1010 C mid-noise: label-free per-image formula7 terms from a (denoised) segmentation.

The grain table and the per-image statistics copy d1009/J/cleanhead seg_c.py + feats_c.py exactly (only the pieces
formula7 needs, plus a few descriptive extras), so variant 'raw_s1' reproduces feats_C_{train,test}.parquet.
Variants differ only in the pre-filter applied to the raw 8-bit image before src.het_blocks.segment and in the
segment sigma. Every value is computed from one image; nothing is fitted across images; no labels are read.

Variants (name -> pre-filter, segment sigma):
  raw_s1   raw image,                                                  sigma 1.0   (the clean pipeline)
  raw_s15  raw image,                                                  sigma 1.5
  raw_s2   raw image,                                                  sigma 2.0
  nlm05_s1 cv2.fastNlMeansDenoising h = 0.5 * est_sigma(image), 5/21,   sigma 1.0
  nlm08_s1 same with h = 0.8 * est_sigma,                              sigma 1.0
  nlm05_s15 h = 0.5 * est_sigma,                                       sigma 1.5
  bil_s1   cv2.bilateralFilter d=7, sigmaColor = 2*est_sigma, sigmaSpace = 2, sigma 1.0
est_sigma = skimage.restoration.estimate_sigma of the image itself (the features_v3 ic_noise definition).
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys

import cv2
import numpy as np
import pandas as pd
from skimage import measure, restoration
from sklearn.mixture import GaussianMixture

sys.path.insert(0, '/home/claude/microstructure-hardness-prediction')
from src.het_blocks import segment, shading_norm  # noqa: E402

cv2.setNumThreads(1)
GRID = np.add.outer(np.arange(256) // 64 * 4, np.arange(256) // 64)
VARIANTS = {
    'raw_s1': ('raw', None, 1.0),
    'raw_s15': ('raw', None, 1.5),
    'raw_s2': ('raw', None, 2.0),
    'nlm05_s1': ('nlm', 0.5, 1.0),
    'nlm08_s1': ('nlm', 0.8, 1.0),
    'nlm05_s15': ('nlm', 0.5, 1.5),
    'bil_s1': ('bil', 2.0, 1.0),
}
F7 = ['fd_area', 'pore_frac', 'logasp_aw', 'S_align', 'logN_eff', 'het4', 'bd_frac']
FD_RULES = {'P0': 'fd_area', 'P1': 'fd_area_P1', 'P2': 'fd_area_P2'}   # P0 = the clean rule


def prefilter(im8, kind, k):
    im8 = np.ascontiguousarray(im8.astype(np.uint8))
    if kind == 'raw':
        return im8.astype(np.float32)
    sg = float(restoration.estimate_sigma(im8.astype(np.float32)))
    if kind == 'nlm':
        return cv2.fastNlMeansDenoising(im8, None, h=float(k * sg), templateWindowSize=5,
                                        searchWindowSize=21).astype(np.float32)
    if kind == 'bil':
        return cv2.bilateralFilter(im8.astype(np.float32), 7, float(k * sg), 2.0)
    raise ValueError(kind)


def grain_table(sm, ws):
    rows = []
    for p in measure.regionprops(ws, intensity_image=sm):
        minr, minc, maxr, maxc = p.bbox
        rows.append(dict(label=p.label, area=p.area, cy=p.centroid[0], cx=p.centroid[1], mean=p.intensity_mean,
                         maj=p.axis_major_length, mnr=p.axis_minor_length, ori=p.orientation, perim=p.perimeter,
                         border=minr == 0 or minc == 0 or maxr == 256 or maxc == 256))
    g = pd.DataFrame(rows)
    g['gn'] = shading_norm(g)
    return g


def phase_thr(gp, info=None):
    big = gp[gp.area > 25]
    thr = 0.90
    try:
        gm = GaussianMixture(2, random_state=0).fit(big.gn.values[:, None])
        mu = np.sort(gm.means_.ravel())
        if info is not None:
            info['gmm_gap'] = float(mu[1] - mu[0])
            info['thr_fallback'] = float(mu[1] - mu[0] <= 0.08)
        if mu[1] - mu[0] > 0.08:
            xs = np.linspace(mu[0], mu[1], 200)
            p = gm.predict_proba(xs[:, None])
            lo = np.argmin(gm.means_.ravel())
            thr = xs[np.argmin(np.abs(p[:, lo] - 0.5))]
    except Exception:
        pass
    return thr


def phase_thr_relaxed(gp):
    """phase_thr without the 0.08 minimum gap (posterior 0.5 crossing of the 2-GMM on gn of grains > 25 px)."""
    big = gp[gp.area > 25]
    gm = GaussianMixture(2, random_state=0).fit(big.gn.values[:, None])
    mu = np.sort(gm.means_.ravel())
    xs = np.linspace(mu[0], mu[1], 200)
    p = gm.predict_proba(xs[:, None])
    lo = np.argmin(gm.means_.ravel())
    return xs[np.argmin(np.abs(p[:, lo] - 0.5))]


def feats(g, ws):
    f = {}
    pore = ((g.gn < 0.68) & (g.area < 600)).values
    gp = g[~pore].copy()
    thr = phase_thr(gp, f)
    dark = (gp.gn < thr).values
    A = 65536.0
    area = gp.area.values.astype(float)
    bd = gp.border.values.astype(bool)
    inner = ~bd
    f['thr'] = thr
    f['N_all'] = len(gp)
    f['N_in'] = inner.sum()
    f['N_bd'] = bd.sum()
    f['N_eff'] = inner.sum() + 0.5 * bd.sum()
    f['pore_frac'] = g.area.values[pore].sum() / A
    f['pore_n'] = pore.sum()
    f['fd_area'] = area[dark].sum() / area.sum()
    # mid-band phase rules (only fd_area depends on the rule): P1 = GMM split if gap > 0.04 else 0.95,
    # P2 = fixed 0.955 (the clean 0.90/0.915 split with the dark-matrix gap halved, as in the noisy preset)
    gap = f.get('gmm_gap', 0.0)
    thr1 = thr if gap > 0.08 else (phase_thr_relaxed(gp) if gap > 0.04 else 0.95)
    f['fd_area_P1'] = area[(gp.gn < thr1).values].sum() / area.sum()
    f['fd_area_P2'] = area[(gp.gn < 0.955).values].sum() / area.sum()
    asp = gp.maj.values / np.maximum(gp.mnr.values, 1)
    ai = area[inner]
    f['logasp_aw'] = (np.log(asp[inner]) * ai).sum() / ai.sum()
    th = gp.ori.values
    w = area * (asp - 1)
    z = (w * np.exp(2j * th)).sum() / w.sum()
    f['S_align'] = np.abs(z)
    lab_pore = np.zeros(ws.max() + 1, bool)
    lab_pore[g.label.values[pore]] = True
    a = np.maximum(np.bincount(ws.ravel()).astype(float)[ws], 1)
    m = ~lab_pore[ws]
    vals = []
    for b in range(16):
        s = (GRID == b) & m
        if s.sum() >= 50:
            vals.append(-2 * np.log((a[s] ** -0.5).mean()))
    f['het4'] = float(np.std(vals))
    f['logN_eff'] = np.log(f['N_eff'])
    f['bd_frac'] = f['N_bd'] / f['N_all']
    return f


def image_feats(im8, variant):
    kind, k, sig = VARIANTS[variant]
    pre = prefilter(im8, kind, k)
    sm, ws = segment(pre, sig)
    g = grain_table(sm, ws)
    return feats(g, ws.astype(np.int32))
