"""Handcrafted microstructure features. Run: python -m src.features  -> data/features.parquet"""
import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from skimage import feature, filters, measure, restoration, segmentation

from .common import DATA_DIR, load_test, read_img


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


def build(ids, n_jobs=-1):
    return pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(extract)(i) for i in ids))


if __name__ == "__main__":
    tr = pd.read_csv(DATA_DIR / "train.csv")
    ids = list(tr.ID) + list(load_test().ID)
    df = build(ids)
    df.to_parquet(DATA_DIR / "features.parquet", index=False)
    print(df.shape)
