"""A41: per-grain interior texture ("etch roughness") and per-grain boundary contrast, number-weighted per-image
summaries.  Fixed per-image transform (train or test).
Image: raw uint8 PNG (NOT the NLM image, which removes fine texture), divided by its local matrix level
bg = 70th-percentile filter of GaussianBlur(raw, 1) (same _bg_matrix as a2/v3) -> x = raw / bg.
Grains: cached watershed labels (a2), pore pixels (GaussianBlur(rn,1) < 0.60, opened) excluded, pore share <= 0.5,
interior = label pixels farther than 3 px from any watershed line (>= 15 px).  Phase dark = interior median rn < 0.91.
Per grain (interior pixels):
  lap  = 1.4826 * MAD(x - box3x3(x))                 fine-scale residual (noise + roughness)
  std  = std(x) about the grain mean                  within-grain std
  band = mean((G0.7(x) - G2.0(x))^2) ^ 0.5            band energy
  level = mean(x)
Image noise references: wav = skimage estimate_sigma(raw) / median(bg); pore = lap measured inside pores (eroded 2 px,
>= 40 px, else NaN); rel = median over all grains of the same measure.
Per image and phase (number-weighted, each grain counts once):
  tx_{m}_{ph}_wav / _pore   median over grains of log(m) - log(reference)
  tx_{m}_{ph}_sd, _iqr      spread over grains of log(m);  tx_{m}_{ph}_xs  excess spread over the sampling expectation
  tx_{m}_dkmx               median log(m) dark - matrix
  tx_{m}_{ph}_rho_size / _rho_lvl   Spearman over grains of log(m) with log area / with grain level
Boundary contrast per grain: boundary pixels = watershed-line pixels touching the grain (3x3); darkness of each
boundary pixel = GaussianBlur(x, 1) there;  bc = 1 - mean(boundary) / median(interior of GaussianBlur(x, 1));
bd = 1 - mean(boundary) (relative to the matrix level).  Same from the NLM image rn: bcn.
  bc_{ph}_med, bc_{ph}_sd, bc_{ph}_iqr, bd_{ph}_med, bcn_{ph}_med, bcn_{ph}_sd, bc_dkmx, bc_{ph}_rho_size
Writes CACHE/tb_{split}.parquet.  Usage: python a41_tex_bd.py CACHE_DIR split"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from skimage import restoration

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


def lab_median(v, lab, idx):
    return np.asarray(ndi.median(v, lab, idx))


def rho(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok])[0]) if ok.sum() > 6 else np.nan


rows = []
for k, i in enumerate(ids):
    raw = cv2.imread(str(DATA_DIR / split / f"{i}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    bg = np.maximum(_bg_matrix(cv2.GaussianBlur(raw, (0, 0), 1.0)), 1.0)
    x = raw / bg
    rn = np.asarray(rn_all[k], np.float32)
    w = np.asarray(ws_all[k]).astype(np.int32)
    f = {"ID": i}
    nl = int(w.max())
    if nl < 5:
        rows.append(f)
        continue
    idx = np.arange(1, nl + 1)
    pm = ndi.binary_opening(cv2.GaussianBlur(rn, (0, 0), 1.0) < 0.60, structure=np.ones((3, 3)))
    line = w == 0
    inner = w.copy()
    inner[ndi.binary_dilation(line, iterations=3) | ndi.binary_dilation(pm, iterations=2)] = 0
    n_in = np.asarray(ndi.sum(np.ones_like(x), inner, idx))
    area = np.asarray(ndi.sum(np.ones_like(x), w, idx))
    pf = np.asarray(ndi.mean(pm.astype(np.float32), w, idx))
    med_rn = lab_median(rn, inner, idx)
    lap = x - cv2.blur(x, (3, 3))
    lap_med = lab_median(lap, inner, idx)
    dev = np.abs(lap - np.concatenate([[0.0], lap_med])[inner])
    m_lap = 1.4826 * lab_median(dev, inner, idx)
    m_std = np.sqrt(np.asarray(ndi.variance(x, inner, idx)))
    bnd = cv2.GaussianBlur(x, (0, 0), 0.7) - cv2.GaussianBlur(x, (0, 0), 2.0)
    m_band = np.sqrt(np.asarray(ndi.mean(bnd ** 2, inner, idx)))
    lvl = np.asarray(ndi.mean(x, inner, idx))
    # boundary darkness per grain: line pixels touching the grain (labels of 3x3 neighbours: max and min non-zero)
    g1 = cv2.GaussianBlur(x, (0, 0), 1.0)
    big = nl + 10
    lmax = ndi.maximum_filter(w, 3)
    lmin = ndi.minimum_filter(np.where(w == 0, big, w), 3)
    yy, xx = np.nonzero(line)
    s_b = np.zeros(nl + 1)
    s_n = np.zeros(nl + 1)
    s_bn = np.zeros(nl + 1)
    for L in (lmax[yy, xx], lmin[yy, xx]):
        ok = (L > 0) & (L <= nl)
        np.add.at(s_b, L[ok], g1[yy[ok], xx[ok]])
        np.add.at(s_bn, L[ok], rn[yy[ok], xx[ok]])
        np.add.at(s_n, L[ok], 1.0)
    same = lmax[yy, xx] == lmin[yy, xx]  # pixel touches one grain only: counted twice above, correct the double count
    Ls = lmax[yy, xx][same]
    np.add.at(s_b, Ls, -g1[yy[same], xx[same]])
    np.add.at(s_bn, Ls, -rn[yy[same], xx[same]])
    np.add.at(s_n, Ls, -1.0)
    bd_mean = np.where(s_n[1:] > 0, s_b[1:] / np.maximum(s_n[1:], 1), np.nan)
    bdn_mean = np.where(s_n[1:] > 0, s_bn[1:] / np.maximum(s_n[1:], 1), np.nan)
    int_g1 = lab_median(g1, inner, idx)
    ok = (n_in >= 15) & (pf <= 0.5) & np.isfinite(med_rn) & (area >= 12)
    f["tb_n"] = int(ok.sum())
    if ok.sum() < 10:
        rows.append(f)
        continue
    dk = med_rn[ok] < 0.91
    la = np.log(area[ok])
    nin = n_in[ok]
    wav = float(restoration.estimate_sigma(raw)) / float(np.median(bg))
    pin = pm & ~ndi.binary_dilation(~pm, iterations=2)
    pore_lap = 1.4826 * float(np.median(np.abs(lap[pin] - np.median(lap[pin])))) if pin.sum() >= 40 else np.nan
    f["tb_wav"] = wav
    f["tb_pore_lap_rel"] = float(np.log(pore_lap / wav)) if np.isfinite(pore_lap) and pore_lap > 0 else np.nan
    for mn, mv in (("lap", m_lap[ok]), ("std", m_std[ok]), ("band", m_band[ok])):
        lm = np.log(np.maximum(mv, 1e-6))
        for ph, m in (("dk", dk), ("mx", ~dk)):
            if m.sum() < 3:
                for s in ("wav", "pore", "sd", "iqr", "xs", "rho_size", "rho_lvl"):
                    f[f"tx_{mn}_{ph}_{s}"] = np.nan
                continue
            f[f"tx_{mn}_{ph}_wav"] = float(np.median(lm[m]) - np.log(wav))
            f[f"tx_{mn}_{ph}_pore"] = float(np.median(lm[m]) - np.log(pore_lap)) if np.isfinite(pore_lap) and pore_lap > 0 else np.nan
            f[f"tx_{mn}_{ph}_sd"] = float(lm[m].std())
            f[f"tx_{mn}_{ph}_iqr"] = float(np.subtract(*np.percentile(lm[m], [75, 25])))
            f[f"tx_{mn}_{ph}_xs"] = float(np.sqrt(max(lm[m].var() - np.mean(1.0 / (2.0 * (nin[m] - 1))), 0.0)))
            f[f"tx_{mn}_{ph}_rho_size"] = rho(lm[m], la[m])
            f[f"tx_{mn}_{ph}_rho_lvl"] = rho(lm[m], lvl[ok][m])
        f[f"tx_{mn}_dkmx"] = float(np.median(lm[dk]) - np.median(lm[~dk])) if dk.sum() >= 3 and (~dk).sum() >= 3 else np.nan
    bc = 1.0 - bd_mean[ok] / np.maximum(int_g1[ok], 1e-3)
    bd = 1.0 - bd_mean[ok]
    bcn = 1.0 - bdn_mean[ok] / np.maximum(med_rn[ok], 1e-3)
    for ph, m in (("dk", dk), ("mx", ~dk)):
        mm = m & np.isfinite(bc)
        if mm.sum() < 3:
            for s in ("bc_{}_med", "bc_{}_sd", "bc_{}_iqr", "bd_{}_med", "bcn_{}_med", "bcn_{}_sd", "bc_{}_rho_size"):
                f[s.format(ph)] = np.nan
            continue
        f[f"bc_{ph}_med"] = float(np.median(bc[mm]))
        f[f"bc_{ph}_sd"] = float(bc[mm].std())
        f[f"bc_{ph}_iqr"] = float(np.subtract(*np.percentile(bc[mm], [75, 25])))
        f[f"bd_{ph}_med"] = float(np.median(bd[mm]))
        f[f"bcn_{ph}_med"] = float(np.nanmedian(bcn[mm]))
        f[f"bcn_{ph}_sd"] = float(np.nanstd(bcn[mm]))
        f[f"bc_{ph}_rho_size"] = rho(bc[mm], la[mm])
    f["bc_dkmx"] = f.get("bc_dk_med", np.nan) - f.get("bc_mx_med", np.nan)
    rows.append(f)
    if k % 100 == 0:
        print(k, flush=True)
pd.DataFrame(rows).to_parquet(cache / f"tb_{split}.parquet")
print("done", split, len(rows))
