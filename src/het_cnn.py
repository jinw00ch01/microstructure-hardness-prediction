"""Grid-aligned CNN for the grain-size heterogeneity of noisy-preset images (cnn-trainer, 2026-10-08).

What it estimates: the 16 per-block values b_k of src.het_blocks (fixed 4x4 grid of 64-px blocks aligned to the image
origin, GRID; b_k = -2 log(mean over the block's non-pore pixels of a^-1/2), a = pixel area of the watershed grain
containing the pixel), hence het4 = sd_k(b_k), and log N_eff, from ONE image.
How: trained only on the 267 clean train images (raw ic_noise < 9.5, where the het_blocks segmentation is reliable),
each re-rendered on the fly to look like the noisy preset (render_noisy); the targets are measured on the clean
originals by the het_blocks method. The hardness label is never read. Test images are only passed through trained
models (prediction); nothing is fitted on them.

  python -m src.het_cnn --prep [--threads 3]        # -> data/het_cnn/prep.npz; asserts het4/N_eff == het_blocks_train
  python -m src.het_cnn --selftest                  # D4 / 4x4-grid permutation / TTA unit tests (needs the prep)
  python -m src.het_cnn --calib 100 --calib-dir D   # contrast stats of renders vs real noisy train images + montage
                                                    # (D must be outside the repo: the montage shows competition images)
  python -m src.het_cnn --device cpu --folds 0 1 --epochs 24 --batch 16 --threads 3 --workers 1 --out r18_e24r4_pilot
  python -m src.het_cnn --device cuda --epochs 32 --renders 4 --batch 12 --workers 4 --threads 2 --out ev2s_e32r4
  python -m src.het_cnn --dump ev2s_e32r4 --chunk 0  # 4-decimal ID-ordered text for messages (chunks 0, 1, 2)
  python -m src.het_cnn --ingest ev2s_e32r4 c0.txt c1.txt c2.txt   # cloud: rebuild the parquets from that text
  ... --extra-targets fd pore asp --out ev2s_e32r4_x  # also learn 3 label-free scalars of the clean originals
  ... --render-preset mid --out ev2s_mid_e32r4        # renders for the raw ic_noise 9.5-12 band (RENDER_MID)

Options added 2026-10-09 (defaults reproduce the earlier runs bit-identically; checked on CPU):
--extra-targets {fd,pore,asp}: extra scalar targets measured on the clean originals from the prep arrays (never from
  hardness; extra_target_values): fd = dark-grain area fraction of the label map (prep 'dark' flags, the renderer's
  dark grains; equals prep dark_area), pore = area fraction of the het_blocks pore grains (prep 'pore' flags: grey /
  shading < 0.68 and area < 600, the rule of the generator's pore_frac), asp = area-weighted mean over interior non-pore
  grains of log(major / max(minor, 1)) from the second moments of each grain (the generator's logasp_aw on this label
  map). All three are invariant under the D4 ops. Head: global mean of the last feature map -> one linear output per
  extra target; loss += lambda_x * sum over the extras of the batch mean of ((x-hat - x) / s_x)^2, mu_x / s_x from
  the training fold. Outputs: columns fd_cnn, pore_cnn, asp_cnn appended to het_cnn_{train,test}.parquet (after
  src) and to --dump / --ingest (after bmean_cnn).
--render-preset {noisy,mid}: 'noisy' = RENDER / M_RANGES (unchanged); 'mid' = RENDER_MID / M_RANGES_MID: clean-preset-
  like renders at higher noise (strong boundary lines, little dark-contrast loss), meant for raw ic_noise 9.5-12.
  --calib with --render-preset mid compares against the real train images with 9.5 <= raw ic_noise < 12.

Training, per fold f of data/folds.csv restricted to the clean images: train on the clean images not in fold f; every
epoch renders each of them --renders times with fresh random parameters (DataLoader workers), then one random D4 op
(flips/rot90 map the 64-px grid onto itself: the image is transformed and the 16 targets are permuted, GRID_PERM).
Full 256x256 input, no crop/resize. Raw channel only (an NLM channel per render is too slow), normalised by the pixel
mean/sd of one fixed render of each training image of the fold. Head: last feature map -> adaptive_avg_pool2d(4) ->
1x1 conv -> 16 b-hat (aligned cells, row-major like GRID); global mean -> linear -> log N-hat. Loss = mean over valid
blocks of ((b-hat - b)/s_b)^2 + lambda_het (sd(b-hat) - het4)^2 / s_het^2 + lambda_N ((logN-hat - logN)/s_N)^2,
standardisation constants from the training fold only. AdamW + linear warm-up + cosine, AMP on CUDA, fp32 on CPU.
Fixed schedule; the final weights predict (no checkpoint selection).
Prediction per fold model: (a) held-out clean fold images rendered with K fixed seeds -> stage-1 metrics; (b) all 500
real train and 1000 real test images (raw, prediction only) with 8-op D4 TTA (grid un-permuted, b-hat averaged per
block, then het = sd over the 16 blocks). Aggregation: clean train images get only the fold model that did not train
on them; noisy train images and all test images get the mean over the fold models run (b-hat averaged per block).
Outputs (data/het_cnn/<out>/, git-ignored): fold{f}.pt, fold{f}.npz, fold{f}.json, het_cnn_{train,test}.parquet
(ID, het_cnn, logN_cnn, bmean_cnn, b00..b15), score.json (stage-1 metrics per fold + the exact command).
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi

from .common import DATA_DIR, ROOT, create_timm

OUT_ROOT = DATA_DIR / "het_cnn"
PREP_FP = OUT_ROOT / "prep.npz"
NOISE_MAX = 9.5  # src.het_blocks.NOISE_MAX
RPORE_REL = 0.65  # render-only pore blobs: sigma-1 grey / shading surface below this (prep_one)
GPU_ARCH = "tf_efficientnetv2_s.in21k_ft_in1k"  # as used with src.train_cnn on the laptop GPU (Apache-2.0)
CPU_ARCH = "resnet18.a1_in1k"  # src.train_cnn default backbone

# ---------------------------------------------------------------------------------------------------------- renderer
# Ranges of the cartoon re-render (render_noisy), calibrated 2026-10-08 against the real noisy train images (--calib;
# table and montage in hardness-cache/scripts/het-1008/build/calib_v1_*; with the cartoon's mean-grey fix and a mean
# row: het-1008/review/calib_v2_*).
RENDER = dict(
    p_m=0.25,           # probability of using degrade_m on the original image instead of the cartoon
    c_d=(0.55, 1.0),    # dark grains: deviation from the matrix level x c_d          (task start: 0.35-0.8)
    e=(1.2, 2.0),       # matrix grains: deviation from the matrix level x e          (task start: 0.8-1.5)
    p_line=0.3,         # probability of faint residual boundary lines
    line=(0.0, 0.3),    # their strength (fraction of the original line depth)
    pore=(0.8, 1.0),    # pore depth factor
    blur=(0.3, 1.6),    # gaussian blur sigma (px)                                    (task start: 0.8-3.0)
    illum=(0.0, 0.035),  # amplitude of the smooth illumination field (as degrade_m)
    gain=(0.9, 1.1),
    offset=(-10.0, 10.0),
    noise=(9.5, 21.0),  # white noise sd (grey levels); covers every image above the 9.5 gate (task start: 11-21)
)
# degrade_m ranges: those of the measure agent's matched pairs (gap-1007/noise/code/build_m.py)
M_RANGES = dict(cd=(0.4, 0.75), ex=(1.2, 1.5), sb=(0.8, 2.2), sn=(12.0, 20.5))
# 'mid' preset (--render-preset mid, 2026-10-09): clean-preset-like images at higher noise, for raw ic_noise 9.5-12.
# Boundary lines kept in most renders and strong, dark phase barely compressed, little extra matrix spread, noise 8-13
# (orchestrator's ranges); blur and the degrade_m branch are scaled down to match (see MID_NOTE).
# Calibration 2026-10-09 (hardness-cache scripts/d1009/D/calib/mid_v1_*, 120 renders vs the 68 train images with
# 9.5 <= raw ic_noise < 12): that band is a mix -- 21 line-visible clean-preset-like images (ic_ridge_snr >= 0.5:
# dark offset -17.8, d' 3.35) and 47 noisy-preset-like ones (offset -8.7, d' 1.53). Mid renders: offset -14.3, d' 2.82,
# ic_noise 10.8 (band 10.3), but ic_ridge_snr median 0.32 vs 0.62 for the line-visible band images (0.36 even with
# line 0.8-1.3 and blur 0.2-0.6), so their boundary lines are weaker than in the real line-visible images.
RENDER_MID = dict(RENDER, c_d=(0.8, 1.0), e=(1.0, 1.3), p_line=0.9, line=(0.3, 1.0), noise=(8.0, 13.0),
                  blur=(0.3, 1.0))
M_RANGES_MID = dict(cd=(0.8, 1.0), ex=(1.0, 1.3), sb=(0.3, 1.0), sn=(8.0, 13.0))
MID_NOTE = ("mid: c_d, e, p_line, line, noise as specified for the 9.5-12 band; blur 0.3-1.0 (noisy 0.3-1.6) and the "
            "degrade_m branch (p_m 0.25 as noisy) with the same mild compression / spread / noise and blur 0.3-1.0")
PRESETS = {"noisy": (RENDER, M_RANGES), "mid": (RENDER_MID, M_RANGES_MID)}
CALIB_BAND = {"noisy": (12.0, np.inf), "mid": (NOISE_MAX, 12.0)}  # real train images the --calib table compares to
EXTRA_TARGETS = ("fd", "pore", "asp")  # --extra-targets choices; output columns <name>_cnn


# ------------------------------------------------------------------------------------------------------ D4 geometry
def d4_np(x, k):
    """The 8 dihedral ops on the last two axes, identical to src.train_cnn.d4 (rot90 by k%4, then flip(-1) if k>=4)."""
    x = np.rot90(x, k % 4, axes=(-2, -1))
    return x[..., ::-1] if k >= 4 else x


def d4_t(x, k):  # torch version (src.train_cnn.d4)
    x = torch.rot90(x, k % 4, dims=(-2, -1))
    return x.flip(-1) if k >= 4 else x


# Block j of the transformed image is block GRID_PERM[k][j] of the original: b_transformed = b[GRID_PERM[k]].
GRID_PERM = np.stack([d4_np(np.arange(16).reshape(4, 4), k).ravel() for k in range(8)])
GRID_INV = np.argsort(GRID_PERM, 1)  # b_original = b_transformed[GRID_INV[k]]


# ------------------------------------------------------------------------------------------------------------- prep
def _boundary(ws):
    bd = np.zeros(ws.shape, bool)
    dx, dy = ws[:, 1:] != ws[:, :-1], ws[1:] != ws[:-1]
    bd[:, 1:] |= dx
    bd[:, :-1] |= dx
    bd[1:] |= dy
    bd[:-1] |= dy
    return bd


def block_targets(ws, is_pore):
    """Per-block b (16,), NaN where a block has < 50 non-pore pixels -- the expression of src.het_blocks.one."""
    from .het_blocks import GRID
    a = np.maximum(np.bincount(ws.ravel()).astype(float)[ws], 1)
    m = ~is_pore[ws]
    b = np.full(16, np.nan)
    for k in range(16):
        s = (GRID == k) & m
        if s.sum() >= 50:
            b[k] = -2 * np.log((a[s] ** -0.5).mean())
    return b


def n_eff_of(ws, is_pore):
    """N_eff = interior grains + 0.5 * border grains, pores excluded (src.het_blocks.one), from a label map."""
    from skimage import measure
    n = 0.0
    for p in measure.regionprops(ws):
        if is_pore[p.label]:
            continue
        minr, minc, maxr, maxc = p.bbox
        n += 0.5 if (minr == 0 or minc == 0 or maxr == ws.shape[0] or maxc == ws.shape[1]) else 1.0
    return n


def grain_log_aspect(ws, is_pore):
    """Area-weighted mean over the interior (bbox not touching the image border), non-pore grains of
    log(major / max(minor, 1)); major/minor = 4 sqrt(eigenvalues of the grain's pixel-coordinate covariance), i.e.
    skimage's axis_major/minor_length. The generator's logasp_aw (gap-1007/generator/scripts/feats.py) on this label
    map. ws: int label map (labels >= 1), is_pore: bool per label (len > ws.max())."""
    L = len(is_pore)
    w = ws.ravel()
    yy, xx = np.indices(ws.shape)
    n = np.bincount(w, minlength=L).astype(float)
    nn_ = np.maximum(n, 1)
    my = np.bincount(w, yy.ravel(), L) / nn_
    mx = np.bincount(w, xx.ravel(), L) / nn_
    dy, dx = yy.ravel() - my[w], xx.ravel() - mx[w]
    cyy, cxx, cxy = (np.bincount(w, v, L) / nn_ for v in (dy * dy, dx * dx, dx * dy))
    half_tr = (cyy + cxx) / 2
    disc = np.sqrt(np.maximum(half_tr ** 2 - (cyy * cxx - cxy ** 2), 0))
    l1, l2 = half_tr + disc, np.maximum(half_tr - disc, 0)
    asp = 4 * np.sqrt(l1) / np.maximum(4 * np.sqrt(l2), 1)
    border = np.zeros(L, bool)
    border[np.concatenate([ws[0], ws[-1], ws[:, 0], ws[:, -1]])] = True
    sel = (n > 0) & ~border & ~is_pore
    sel[0] = False
    return float((np.log(asp[sel]) * n[sel]).sum() / n[sel].sum())


def extra_target_values(P, names):
    """(n_prep, len(names)) float64 extra targets of the clean originals, from the prep arrays only (label maps and
    per-grain flags; the hardness label is never read). fd: area fraction of the prep 'dark' grains (= prep
    dark_area); pore: area fraction of the het_blocks pore grains (prep 'pore'); asp: grain_log_aspect."""
    out = np.zeros((len(P["ids"]), len(names)))
    for j in range(len(P["ids"])):
        ws = P["ws"][j].astype(np.int64)
        for c, nm in enumerate(names):
            if nm == "fd":
                out[j, c] = P["dark"][j][ws].mean()
            elif nm == "pore":
                out[j, c] = P["pore"][j][ws].mean()
            elif nm == "asp":
                out[j, c] = grain_log_aspect(ws, P["pore"][j])
            else:
                raise ValueError(f"unknown extra target {nm!r} (choices {EXTRA_TARGETS})")
    return out


def shading_fit(g, v=None):
    """src.het_blocks.shading_norm (same algorithm) on grain greys v (default: g['mean']); also returns the surface
    coefficients so the shading can be evaluated per pixel (shading_surface)."""
    x, y = g.cx.values / 256 - 0.5, g.cy.values / 256 - 0.5
    X = np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], 1)
    w = g.area.values.astype(float)
    v = (g["mean"].values if v is None else np.asarray(v)).astype(float)
    keep = v > 0.9 * np.median(v[g.area.values > 30])
    for _ in range(4):
        W = np.sqrt(w[keep])
        beta, *_ = np.linalg.lstsq(X[keep] * W[:, None], v[keep] * W, rcond=None)
        fit = X @ beta
        keep = v / fit > 0.93
    return v / fit, beta


_YY, _XX = np.indices((256, 256))


def shading_surface(beta):
    x, y = _XX / 256 - 0.5, _YY / 256 - 0.5
    return (beta[0] + beta[1] * x + beta[2] * y + beta[3] * x * x + beta[4] * y * y + beta[5] * x * y).astype(np.float32)


def prep_one(i):
    """One clean train image -> label map, grain table, pores, targets and the arrays the renderer needs.
    Targets (b, het4, N_eff) use the het_blocks pore rule; the renderer's pore pixels (rpore) are a separate,
    render-only mask."""
    import cv2
    from skimage import restoration
    from . import het_blocks as hb
    from .features import _bg_matrix
    cv2.setNumThreads(1)
    im8 = np.array(Image.open(DATA_DIR / "train" / f"{i}.png"))
    im = im8.astype(np.float32)  # exactly as src.het_blocks.one
    sm, ws = hb.segment(im)
    g = hb.grain_table(sm, ws)
    pore_g = ((hb.shading_norm(g) < 0.68) & (g.area < 600)).values  # pore rule of het_blocks.one
    gp = g[~pore_g]
    n_eff = float((~gp.border).sum() + 0.5 * gp.border.sum())
    L = int(ws.max()) + 1
    is_pore = np.zeros(L, bool)
    is_pore[g.label.values[pore_g]] = True
    b = block_targets(ws, is_pore)
    het4 = float(np.std(b[~np.isnan(b)]))
    # per-grain robust grey: median of the raw pixels away from the watershed lines (ridges); fall back to a thinner
    # margin, then to the whole grain, for small grains
    dist = ndi.distance_transform_edt(~_boundary(ws))
    lab = g.label.values
    c3 = np.bincount(ws[dist >= 3], minlength=L)[lab]
    c2 = np.bincount(ws[dist >= 2], minlength=L)[lab]
    m3 = np.asarray(ndi.median(im, np.where(dist >= 3, ws, 0), lab), float)
    m2 = np.asarray(ndi.median(im, np.where(dist >= 2, ws, 0), lab), float)
    m0 = np.asarray(ndi.median(im, ws, lab), float)
    rob = np.where(c3 >= 12, m3, np.where(c2 >= 4, m2, m0))
    sn_rob, beta = shading_fit(g, rob)  # shading surface of het_blocks.shading_norm, fitted to the robust greys
    sn_tab = np.ones(L, np.float32)
    sn_tab[lab] = sn_rob
    dark = np.zeros(L, bool)
    dark[lab] = (sn_rob < 0.93) & ~is_pore[lab]
    S = shading_surface(beta)
    # render-only pore pixels (the targets keep the het_blocks pore rule): het_blocks pore grains, plus very dark
    # compact blobs (thin dark boundary lines removed by an opening) with a 2-px rim, because large round pores
    # (area >= 600) and the dark rims of pores are assigned to dark grains / neighbours by the watershed
    from skimage import morphology
    rel = sm / S
    blob = ndi.binary_opening(rel < RPORE_REL, structure=morphology.disk(2))
    blob = ndi.binary_dilation(blob, structure=morphology.disk(2)) & (rel < 0.9)
    pp = is_pore[ws] | blob
    # residual map: pore pixels -> deviation from the shading surface (pores are kept as they are); pixels near a
    # watershed line -> how much darker than their grain's cartoon grey they are (the boundary line depth)
    resid = np.where(pp, sm - S, np.where(dist < 3, np.minimum(sm - S * sn_tab[ws], 0), 0)).astype(np.float16)
    # deterministic part of degrade_m (NLM-denoised image and matrix background), precomputed once
    sig0 = float(restoration.estimate_sigma(im))
    den = cv2.fastNlMeansDenoising(im8, None, h=float(np.clip(sig0, 2.0, 30.0)), templateWindowSize=5,
                                   searchWindowSize=21)
    bg = _bg_matrix(den.astype(np.float32))
    return dict(ID=i, im8=im8, ws=ws.astype(np.int16), resid=resid, rpore=pp, den=den, bg=bg.astype(np.float16),
                sig0=sig0, beta=beta, sn_tab=sn_tab, dark=dark, pore=is_pore, b=b, het4=het4, n_eff=n_eff,
                dark_area=float(dark[ws].mean()), n_grains=len(g), n_pores=int(pore_g.sum()))


def clean_ids():
    v3 = pd.read_parquet(DATA_DIR / "features_v3.parquet", columns=["ID", "ic_noise"]).set_index("ID")
    ids = pd.read_csv(DATA_DIR / "train.csv", usecols=["ID"]).ID  # IDs only; the hardness column is not read
    return [i for i in ids if v3.loc[i, "ic_noise"] < NOISE_MAX]


def run_prep(threads):
    from joblib import Parallel, delayed
    ids = clean_ids()
    t = time.time()
    rows = Parallel(n_jobs=threads)(delayed(prep_one)(i) for i in ids)
    het4 = np.array([r["het4"] for r in rows])
    n_eff = np.array([r["n_eff"] for r in rows])
    ref_fp = DATA_DIR / "het_blocks_train.parquet"
    if ref_fp.exists():
        ref = pd.read_parquet(ref_fp).set_index("ID").loc[ids]
        dh, dn = np.abs(ref.het4.values - het4).max(), np.abs(ref.N_eff.values - n_eff).max()
        assert dh < 1e-9 and dn < 1e-9, f"prep does not reproduce {ref_fp.name}: max |dhet4| {dh}, |dN_eff| {dn}"
        print(f"prep reproduces {ref_fp.name} on {len(ids)} images (max |dhet4| {dh:.1e}, |dN_eff| {dn:.1e})")
    else:
        print(f"WARNING {ref_fp} missing: het4/N_eff not cross-checked (run python -m src.het_blocks first)")
    L = max(len(r["sn_tab"]) for r in rows)

    def pad(key, dtype, fill):
        out = np.full((len(rows), L), fill, dtype)
        for n, r in enumerate(rows):
            out[n, :len(r[key])] = r[key]
        return out

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = PREP_FP.with_name("prep_tmp.npz")
    np.savez(tmp, ids=np.array(ids), im8=np.stack([r["im8"] for r in rows]), ws=np.stack([r["ws"] for r in rows]),
             resid=np.stack([r["resid"] for r in rows]), rpore=np.stack([r["rpore"] for r in rows]),
             den=np.stack([r["den"] for r in rows]), bg=np.stack([r["bg"] for r in rows]), sig0=np.array([r["sig0"] for r in rows]),
             beta=np.stack([r["beta"] for r in rows]), sn_tab=pad("sn_tab", np.float32, 1.0),
             dark=pad("dark", bool, False), pore=pad("pore", bool, False), b=np.stack([r["b"] for r in rows]),
             het4=het4, n_eff=n_eff, dark_area=np.array([r["dark_area"] for r in rows]))
    os.replace(tmp, PREP_FP)
    nan_blocks = int(np.isnan(np.stack([r["b"] for r in rows])).sum())
    print(f"prep: {len(ids)} clean train images in {time.time() - t:.0f}s -> {PREP_FP} "
          f"({PREP_FP.stat().st_size / 1e6:.0f} MB); NaN blocks {nan_blocks}; het4 sum {het4.sum():.6f}, "
          f"N_eff sum {n_eff.sum():.1f}; dark area frac median {np.median([r['dark_area'] for r in rows]):.3f}; "
          f"grains median {np.median([r['n_grains'] for r in rows]):.0f}, images with pores "
          f"{sum(r['n_pores'] > 0 for r in rows)}")


def load_prep(fp=PREP_FP):
    if not Path(fp).exists():
        raise SystemExit(f"{fp} missing: run python -m src.het_cnn --prep first")
    with np.load(fp) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------------------------------------- renderer
def degrade_m(img8, rng, cd=(0.35, 0.7), ex=(1.0, 1.3), sb=(1.0, 2.8), sn=(12.0, 20.5), den=None, bg=None,
              sig0=None):
    """Copied from the measure agent's /mnt/project-files/work/hardness-cache/scripts/gap-1007/noise/code/degrade.py
    (degrade_m, 2026-10-07), unchanged except that the deterministic NLM/background part can be passed in
    precomputed (den, bg, sig0 from the prep: den and sig0 identical, bg stored as float16, i.e. within ~0.1 grey).
    noisy-preset matched: phase-specific dark contrast loss (dark grains move toward the matrix by factor c_d,
    matrix untouched), bright-side matrix spread x e, blur of everything, white noise."""
    import cv2
    from skimage import restoration
    img = img8.astype(np.float32)
    if den is None:
        from .features import _bg_matrix
        sig0 = float(restoration.estimate_sigma(img))
        den = cv2.fastNlMeansDenoising(img8, None, h=float(np.clip(sig0, 2.0, 30.0)), templateWindowSize=5,
                                       searchWindowSize=21).astype(np.float32)
        bg = _bg_matrix(den)
    den, bg = den.astype(np.float32), bg.astype(np.float32)
    r = cv2.GaussianBlur(den / bg, (0, 0), 1.0)
    pore = ndi.binary_dilation(r < 0.68, iterations=1)
    c_d = rng.uniform(*cd); e = rng.uniform(*ex)
    w = np.clip((0.955 - r) / 0.04, 0, 1)
    dd = den - bg
    grain = bg + dd * (1 - w * (1 - c_d))
    grain = bg + (grain - bg) * np.where(dd > 0, e, 1.0)
    grain = grain + (img - den)
    out = np.where(pore, img, grain)
    s_b = rng.uniform(*sb)
    out = cv2.GaussianBlur(out, (0, 0), s_b)
    a = rng.uniform(0.0, 0.035)
    field = cv2.GaussianBlur(rng.standard_normal(out.shape).astype(np.float32), (0, 0), 40.0)
    field /= field.std() + 1e-9
    out = out * (1 + a * field)
    gain, off = rng.uniform(0.9, 1.1), rng.uniform(-10, 10)
    out = (out - out.mean()) * gain + out.mean() + off
    sig_t = rng.uniform(*sn)
    s_add = float(np.sqrt(max(sig_t ** 2 - (sig0 * 0.5) ** 2, 0.0)))
    out = out + rng.normal(0, s_add, out.shape)
    return np.clip(np.round(out), 0, 255).astype(np.uint8), dict(q=np.nan, sb=s_b, c=c_d, e=e, sig_t=sig_t)


def render_noisy(P, j, rng, R=None, force=None, M=None):
    """Re-render clean prep image j in the noisy-preset style -> (uint8 (256, 256), info). The watershed label map is
    unchanged, so the block targets of the original stay exact. force: None | 'cartoon' | 'm'. R / M: cartoon and
    degrade_m ranges (default RENDER / M_RANGES = the 'noisy' preset; *PRESETS['mid'] for the mid preset)."""
    import cv2
    R = RENDER if R is None else R
    M = M_RANGES if M is None else M
    path = force or ("m" if rng.random() < R["p_m"] else "cartoon")
    if path == "m":
        img, p = degrade_m(P["im8"][j], rng, **M, den=P["den"][j], bg=P["bg"][j], sig0=float(P["sig0"][j]))
        return img, dict(path=1, c_d=p["c"], e=p["e"], blur=p["sb"], noise=p["sig_t"])
    ws = P["ws"][j].astype(np.int64)
    S = shading_surface(P["beta"][j])
    sn, dark = P["sn_tab"][j], P["dark"][j]
    resid = P["resid"][j].astype(np.float32)
    c_d, e = rng.uniform(*R["c_d"]), rng.uniform(*R["e"])
    new = np.where(dark, 1 + c_d * (sn - 1), 1 + e * (sn - 1)).astype(np.float32)  # pore entries unused below
    pp = P["rpore"][j]  # render-only pore pixels (prep_one)
    img = S * new[ws]  # cartoon: every pixel takes its grain's (shading-normalised) grey -> no boundary lines
    img = np.where(pp, S + rng.uniform(*R["pore"]) * resid, img)
    s_line = rng.uniform(*R["line"]) if rng.random() < R["p_line"] else 0.0
    if s_line > 0:
        img = img + np.where(pp, 0, s_line * resid)
    # keep the source's mean grey: dropping the dark boundary lines alone made cartoons ~8.5 grey brighter than their
    # source, while real noisy images have the clean images' mean grey (median 145.2 vs 145.0; review 2026-10-08)
    img = img + (float(P["im8"][j].mean()) - float(img.mean()))
    s_b = rng.uniform(*R["blur"])
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), s_b)
    a = rng.uniform(*R["illum"])
    field = cv2.GaussianBlur(rng.standard_normal(img.shape).astype(np.float32), (0, 0), 40.0)
    field /= field.std() + 1e-9
    img = img * (1 + a * field)
    gain, off = rng.uniform(*R["gain"]), rng.uniform(*R["offset"])
    img = (img - img.mean()) * gain + img.mean() + off
    sd = rng.uniform(*R["noise"])
    img = img + rng.normal(0, sd, img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8), dict(path=0, c_d=c_d, e=e, blur=s_b, noise=sd,
                                                                 line=s_line)


# ------------------------------------------------------------------------------------------- calibration statistics
_N = 256
_KR = np.hypot(np.fft.fftfreq(_N)[:, None], np.fft.fftfreq(_N)[None, :]) * _N


def _gauss_kernel_norm2(s):
    import cv2
    k = cv2.getGaussianKernel(int(2 * np.ceil(4 * s) + 1), s)[:, 0]
    return float((k ** 2).sum() ** 2)


def contrast_stats(im8):
    """Contrast statistics of the gap-1007 'measure' report, per image.
    off (dark-phase offset from the matrix), sdhi, pdepth (pore depth): copy of cstats() in
    hardness-cache/scripts/gap-1007/noise/code/infoloss_build.py; dprime = -off / sdhi (the report's d').
    sb35 (matrix bright-side spread), def35 (mean deficit below the matrix mode), sn (white-noise level): the
    nr_sb3.5 / nr_def35 / nr_sn parts of features() in hardness-cache/scripts/gap-1007/noise/code/nrfeat.py.
    ic_noise: skimage estimate_sigma, as src.features._quality_v3."""
    import cv2
    from skimage import restoration
    from sklearn.mixture import GaussianMixture
    im = im8.astype(np.float64)
    bg = cv2.GaussianBlur(im, (0, 0), 32)
    x = im - bg
    g = cv2.GaussianBlur(x, (0, 0), 3.5)
    v = g[8:-8:2, 8:-8:2].ravel()
    gm = GaussianMixture(2, random_state=0).fit(v[:, None])
    mu = gm.means_[:, 0]
    sd = np.sqrt(gm.covariances_[:, 0, 0])
    lo, hi = np.argmin(mu), np.argmax(mu)
    g1 = cv2.GaussianBlur(x, (0, 0), 2.0)
    f = dict(off=mu[lo] - mu[hi], sdhi=sd[hi], pdepth=np.percentile(g1, 0.05) - np.median(g1))
    f["dprime"] = -f["off"] / f["sdhi"]
    # nrfeat: noise level from the flat high-frequency floor, then the sigma-3.5 grey-level distribution
    Pw = np.abs(np.fft.fft2(im - im.mean())) ** 2 / im.size
    sn = float(np.sqrt(np.median(Pw[_KR > 100])) / np.sqrt(np.log(2)))
    xx = x - x.mean()
    gg = cv2.GaussianBlur(xx, (0, 0), 3.5)
    se = sn * np.sqrt(_gauss_kernel_norm2(3.5))
    vv = gg.ravel()
    h, e = np.histogram(vv, bins=np.arange(-120, 80.5, 1.0))
    mode = float(e[np.argmax(ndi.gaussian_filter1d(h.astype(float), 2.0))] + 0.5)
    u = vv - mode
    up = u[u > 0]
    f["sb35"] = float(np.sqrt(max(np.mean(up ** 2) - se ** 2, 1.0)))
    f["def35"] = float(-np.mean(u))
    f["sn"] = sn
    f["ic_noise"] = float(restoration.estimate_sigma(im8.astype(np.float32)))
    f["mean"] = float(im.mean())  # absolute grey level (every statistic above is offset-invariant)
    return f


def run_calib(a):
    """Contrast stats of a.calib renders vs the real noisy train images (raw ic_noise >= 12) + a montage. With
    --render-preset mid: mid renders vs the real train images with 9.5 <= raw ic_noise < 12 (CALIB_BAND)."""
    import cv2
    from joblib import Parallel, delayed
    cdir = Path(a.calib_dir).resolve()
    if ROOT.resolve() in cdir.parents or cdir == ROOT.resolve():
        raise SystemExit("--calib-dir must be outside the repo (the montage shows competition images)")
    cdir.mkdir(parents=True, exist_ok=True)
    P = load_prep()
    n = len(P["ids"])
    v3 = pd.read_parquet(DATA_DIR / "features_v3.parquet", columns=["ID", "ic_noise"])
    R, M = PRESETS[a.render_preset]
    lo, hi = CALIB_BAND[a.render_preset]
    real_ids = v3[(v3.ic_noise >= lo) & (v3.ic_noise < hi)].ID.tolist()
    real_ids = [i for i in real_ids if i.startswith("TRAIN")]
    order = np.random.default_rng([a.seed, 777]).permutation(n)

    def one_render(t):
        rng = np.random.default_rng([a.seed, 4242, t])
        j = int(order[t % n])
        img, info = render_noisy(P, j, rng, R, M=M)
        return {**contrast_stats(img), **info, "src": P["ids"][j]}

    cv2.setNumThreads(1)
    rows = Parallel(n_jobs=a.threads)(delayed(one_render)(t) for t in range(a.calib))
    Rn = pd.DataFrame(rows)
    real = pd.DataFrame(Parallel(n_jobs=a.threads)(
        delayed(lambda i: {**contrast_stats(np.array(Image.open(DATA_DIR / "train" / f"{i}.png"))), "ID": i})(i)
        for i in real_ids))
    clean = pd.DataFrame(Parallel(n_jobs=a.threads)(
        delayed(lambda j: contrast_stats(P["im8"][j]))(j) for j in order[:60]))
    cols = ["off", "sb35", "pdepth", "def35", "dprime", "sn", "ic_noise", "sdhi", "mean"]
    lines = []
    for c in cols:
        r, s = real[c], Rn[c]
        pct = float((s < r.median()).mean() * 100)
        lines.append(dict(stat=c, real_noisy_med=r.median(), real_q25=r.quantile(.25), real_q75=r.quantile(.75),
                          render_med=s.median(), render_q10=s.quantile(.1), render_q25=s.quantile(.25),
                          render_q75=s.quantile(.75), render_q90=s.quantile(.9),
                          pct_of_real_med_in_renders=pct, cartoon_med=Rn[Rn.path == 0][c].median(),
                          m_med=Rn[Rn.path == 1][c].median(), clean_src_med=clean[c].median()))
    T = pd.DataFrame(lines).set_index("stat")
    pd.set_option("display.width", 250)
    print(f"calibration ({a.render_preset} preset): {len(Rn)} renders ({(Rn.path == 1).sum()} degrade_m) vs "
          f"{len(real)} real train images ({lo} <= ic_noise < {hi}; column real_noisy_*); clean sources: 60 originals")
    print(T.round(2).to_string())
    tag = a.calib_tag or "calib"
    T.round(4).to_csv(cdir / f"{tag}_table.csv")
    Rn.to_csv(cdir / f"{tag}_renders.csv", index=False)
    real.to_csv(cdir / f"{tag}_real.csv", index=False)
    (cdir / f"{tag}_ranges.json").write_text(json.dumps({"preset": a.render_preset, "RENDER": R, "M_RANGES": M},
                                                        indent=1))
    # montage: row 1 real noisy crops; rows 2-3 cartoon renders; row 4 degrade_m renders; row 5 the clean originals
    # of row 2. 128x128 crops at native scale, 2x nearest-neighbour upscale.
    c0, cw = 64, 128
    rng = np.random.default_rng([a.seed, 99])
    pick_real = rng.choice(len(real_ids), 6, replace=False)
    srcs = order[:6]
    tiles = [[np.array(Image.open(DATA_DIR / "train" / f"{real_ids[p]}.png"))[c0:c0 + cw, c0:c0 + cw]
              for p in pick_real]]
    for row, force in ((0, "cartoon"), (1, "cartoon"), (2, "m")):
        tiles.append([render_noisy(P, int(j), np.random.default_rng([a.seed, 55, row, int(j)]), R, force, M)[0]
                      [c0:c0 + cw, c0:c0 + cw] for j in srcs])
    tiles.append([P["im8"][int(j)][c0:c0 + cw, c0:c0 + cw] for j in srcs])
    gap = 4
    H = len(tiles) * (cw + gap)
    W = 6 * (cw + gap)
    canvas = np.full((H, W), 255, np.uint8)
    for r_, row in enumerate(tiles):
        for c_, t in enumerate(row):
            canvas[r_ * (cw + gap):r_ * (cw + gap) + cw, c_ * (cw + gap):c_ * (cw + gap) + cw] = t
    Image.fromarray(canvas).resize((W * 2, H * 2), Image.NEAREST).save(cdir / f"{tag}_montage.png")
    print(f"wrote {cdir}/{tag}_table.csv, _renders.csv, _real.csv, _ranges.json, _montage.png "
          f"(rows: real noisy | cartoon | cartoon (other seed) | degrade_m | clean source)")


# ------------------------------------------------------------------------------------------------------ dataset/model
class RenderSet(torch.utils.data.Dataset):
    """Fresh noisy-style renders of clean prep images. Index idx = epoch * n_items + i (EpochSampler), item i renders
    prep row rows[i % len(rows)] with rng seeded by (seed, fold, epoch, i), so results do not depend on the number of
    workers. The prep arrays are loaded lazily per process (nothing big is pickled to Windows spawn workers).
    preset: render preset (PRESETS); xt: optional (n_prep, n_extra) extra targets (extra_target_values), appended to
    each item as a 6th tensor (D4-invariant, so not permuted)."""

    def __init__(self, prep_fp, rows, renders, seed, fold, preset="noisy", xt=None):
        self.fp, self.rows, self.R, self.seed, self.fold = str(prep_fp), np.asarray(rows), renders, seed, fold
        self.n_items = len(self.rows) * renders
        self.preset = preset
        self.xt = None if xt is None else np.asarray(xt, np.float32)
        self._P = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_P"] = None
        return d

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        if self._P is None:
            self._P = load_prep(self.fp)
        P = self._P
        ep, i = divmod(int(idx), self.n_items)
        j = int(self.rows[i % len(self.rows)])
        rng = np.random.default_rng([self.seed, self.fold, ep, i])
        R, M = PRESETS[self.preset]
        img, _ = render_noisy(P, j, rng, R, M=M)
        k = int(rng.integers(8))
        img = np.ascontiguousarray(d4_np(img, k))
        b = P["b"][j][GRID_PERM[k]]
        valid = ~np.isnan(b)
        item = (torch.from_numpy(img)[None], torch.from_numpy(np.nan_to_num(b).astype(np.float32)),
                torch.from_numpy(valid.astype(np.float32)), torch.tensor(float(P["het4"][j])),
                torch.tensor(float(np.log(P["n_eff"][j]))))
        if self.xt is not None:
            item = item + (torch.from_numpy(self.xt[j].copy()),)
        return item


class EpochSampler(torch.utils.data.Sampler):
    def __init__(self, n, seed):
        self.n, self.seed, self.epoch = n, seed, 0

    def __iter__(self):
        perm = np.random.default_rng([self.seed, 31337, self.epoch]).permutation(self.n)
        return iter((self.epoch * self.n + perm).tolist())

    def __len__(self):
        return self.n


def _worker_init(_):
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


class HetNet(nn.Module):
    """forward -> (b (B, 16), logN (B,)), plus x (B, n_extra) from a linear head on the same globally pooled features
    when n_extra > 0 (the extra layer is created after the others, so n_extra=0 initialises exactly as before)."""

    def __init__(self, arch, pretrained=True, n_extra=0):
        super().__init__()
        self.body = create_timm(arch, pretrained=pretrained, num_classes=0, global_pool="", in_chans=1)
        nf = self.body.num_features
        self.cell = nn.Conv2d(nf, 1, 1)
        self.glob = nn.Linear(nf, 1)
        for m in (self.cell, self.glob):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)
        self.n_extra = n_extra
        if n_extra:
            self.extra = nn.Linear(nf, n_extra)
            nn.init.normal_(self.extra.weight, std=0.01)
            nn.init.zeros_(self.extra.bias)

    def forward(self, x):
        f = self.body.forward_features(x)
        with torch.autocast(x.device.type, enabled=False):  # pooling + heads in fp32
            f = f.float()
            assert f.shape[-1] % 4 == 0 and f.shape[-2] % 4 == 0, f"feature map {tuple(f.shape[-2:])} not divisible by 4"
            b = self.cell(F.adaptive_avg_pool2d(f, 4)).flatten(1)  # (B, 16), row-major = GRID block ids
            g = f.mean((-2, -1))
            n = self.glob(g).squeeze(-1)
            if self.n_extra:
                return b, n, self.extra(g)
        return b, n


def masked_sd(v, m):
    n = m.sum(1).clamp(min=1)
    mu = (v * m).sum(1) / n
    return ((((v - mu[:, None]) ** 2) * m).sum(1) / n + 1e-8).sqrt()


# ------------------------------------------------------------------------------------------------------- prediction
@torch.no_grad()
def predict(model, X8, mu, sd, dev, amp_dtype, tta=8, bs=32):
    """uint8 (N, 256, 256) -> (b_z (N, 16), n_z (N,)): mean over the first `tta` D4 views, each view's 4x4 output
    un-permuted to the original grid (GRID_INV) before averaging. Values in the training fold's z-units.
    A model with extra outputs (HetNet n_extra > 0) -> (b_z, n_z, x_z (N, n_extra)), x averaged over the views."""
    model.eval()
    outb, outn, outx = [], [], []
    inv = [torch.as_tensor(GRID_INV[k], device=dev) for k in range(8)]
    for i in range(0, len(X8), bs):
        x = torch.from_numpy(np.ascontiguousarray(X8[i:i + bs])).to(dev).float().div_(255.0)[:, None]
        x = (x - mu) / sd
        sb, sn, sx = 0, 0, 0
        for k in range(tta):
            with torch.autocast(dev.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                o = model(d4_t(x, k).contiguous(memory_format=torch.channels_last))
            sb = sb + o[0].float()[:, inv[k]]
            sn = sn + o[1].float()
            if len(o) > 2:
                sx = sx + o[2].float()
        outb.append((sb / tta).cpu())
        outn.append((sn / tta).cpu())
        if len(o) > 2:
            outx.append((sx / tta).cpu())
    res = (torch.cat(outb).numpy().astype(np.float64), torch.cat(outn).numpy().astype(np.float64))
    if outx:
        res = res + (torch.cat(outx).numpy().astype(np.float64),)
    return res


def _pcorr(x, y, z):
    """Partial Pearson corr of x and y given z (both residualised on [1, z])."""
    Z = np.column_stack([np.ones(len(z)), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def stage1_metrics(bh, nh, b, het4, logn):
    """bh (M, 16) raw b-hat, nh (M,) logN-hat; b (M, 16) targets with NaN; het4, logn (M,)."""
    v = ~np.isnan(b)
    sdh = np.array([np.std(bh[r][v[r]]) for r in range(len(bh))])
    within_h = bh - np.array([bh[r][v[r]].mean() for r in range(len(bh))])[:, None]
    within_t = b - np.array([b[r][v[r]].mean() for r in range(len(b))])[:, None]
    return dict(n=int(len(bh)), corr_het=float(np.corrcoef(sdh, het4)[0, 1]),
                pcorr_het_given_logN=_pcorr(sdh, het4, logn),
                corr_block=float(np.corrcoef(bh[v], b[v])[0, 1]),
                corr_block_within=float(np.corrcoef(within_h[v], within_t[v])[0, 1]),
                corr_logN=float(np.corrcoef(nh, logn)[0, 1]),
                rmse_logN=float(np.sqrt(np.mean((nh - logn) ** 2))),
                het_hat_mean=float(sdh.mean()), het4_mean=float(het4.mean()))


def stage1_extra(xh, x, names):
    """Per extra target: corr, RMSE, R^2 vs the truth (sd of the truth over these rows as the scale)."""
    out = {}
    for c, nm in enumerate(names):
        h, t = xh[:, c], x[:, c]
        out[nm] = dict(corr=float(np.corrcoef(h, t)[0, 1]), rmse=float(np.sqrt(np.mean((h - t) ** 2))),
                       r2=float(1 - np.mean((h - t) ** 2) / max(np.var(t), 1e-30)), sd_true=float(t.std()),
                       mean_hat=float(h.mean()), mean_true=float(t.mean()))
    return out


def _fmt_extra(m):
    return ", ".join(f"{nm} corr {v['corr']:.3f} R2 {v['r2']:.3f}" for nm, v in m.items())


def read_u8(i):
    return np.array(Image.open(DATA_DIR / ("train" if i.startswith("TRAIN") else "test") / f"{i}.png").convert("L"))


# ---------------------------------------------------------------------------------------------------------- training
def train_fold(a, f, P, folds_of, dev, amp_dtype, pred_ids, out, xt=None):
    """xt: None or (n_prep, n_extra) extra targets (extra_target_values) for a.extra_targets."""
    seed = a.seed * 1000 + f
    xnames = list(a.extra_targets or [])
    nx = len(xnames)
    rR, rM = PRESETS[a.render_preset]  # (M is the module alias below)
    torch.manual_seed(seed)
    ids = P["ids"]
    tr_rows = np.where(folds_of != f)[0]
    va_rows = np.where(folds_of == f)[0]
    if a.smoke:
        tr_rows, va_rows = tr_rows[:24], va_rows[:8]
    b = P["b"]
    logn = np.log(P["n_eff"])
    # standardisation constants: training fold only
    bt = b[tr_rows]
    mu_b, s_b = float(np.nanmean(bt)), float(np.nanstd(bt))
    s_het = float(P["het4"][tr_rows].std())
    mu_n, s_n = float(logn[tr_rows].mean()), float(logn[tr_rows].std())
    # input normalisation: pixel mean/sd of one fixed render of each training image of this fold
    px = np.stack([render_noisy(P, int(j), np.random.default_rng([a.seed, f, 8888, int(j)]), rR, M=rM)[0]
                   for j in tr_rows])
    mu_px, sd_px = float(px.mean() / 255.0), float(px.std() / 255.0)
    del px
    mu_t = torch.tensor(mu_px, device=dev)
    sd_t = torch.tensor(sd_px, device=dev)
    consts = dict(mu_b=mu_b, s_b=s_b, s_het=s_het, mu_n=mu_n, s_n=s_n, mu_px=mu_px, sd_px=sd_px)
    if nx:  # extra targets: standardisation from the training fold only
        mu_x, s_x = xt[tr_rows].mean(0), xt[tr_rows].std(0)
        for c, nm in enumerate(xnames):
            consts[f"mu_{nm}"], consts[f"s_{nm}"] = float(mu_x[c]), float(s_x[c])
        mu_xt = torch.tensor(mu_x, dtype=torch.float32, device=dev)
        s_xt = torch.tensor(s_x, dtype=torch.float32, device=dev)
    print(f"fold {f}: train {len(tr_rows)} / held-out {len(va_rows)} clean images; consts "
          + " ".join(f"{k} {v:.4f}" for k, v in consts.items()), flush=True)

    model = HetNet(a.arch, pretrained=not a.scratch, n_extra=nx).to(dev).to(memory_format=torch.channels_last)
    decay, no_decay = [], []
    for nme, p in model.named_parameters():
        (decay if p.ndim > 1 else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr)
    from . import het_cnn as M  # pickle as src.het_cnn.* (not __main__.*) for spawn workers (Windows)
    ds = M.RenderSet(a.prep, tr_rows, a.renders, a.seed, f, preset=a.render_preset, xt=xt)
    sampler = M.EpochSampler(len(ds), seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, sampler=sampler, num_workers=a.workers,
                                     drop_last=True, persistent_workers=a.workers > 0, worker_init_fn=M._worker_init,
                                     pin_memory=dev.type == "cuda")
    spe = len(dl)
    total = a.epochs * spe
    warm = max(1, int(round(a.warmup * total)))
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        a.final_lr + (1 - a.final_lr) * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        sampler.epoch = ep
        tl = np.zeros(4 + nx)
        te = time.time()
        for batch in dl:
            x, bb, vv, hh, nn_ = batch[:5]
            x = x.to(dev, non_blocking=True).float().div_(255.0)
            x = ((x - mu_t) / sd_t).contiguous(memory_format=torch.channels_last)
            bb, vv, hh, nn_ = (t.to(dev, non_blocking=True) for t in (bb, vv, hh, nn_))
            with torch.autocast(dev.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                o = model(x)
            bz, nz = o[0].float(), o[1].float()
            l_b = (((bz - (bb - mu_b) / s_b) ** 2) * vv).sum() / vv.sum()
            l_h = ((s_b * masked_sd(bz, vv) - hh) ** 2).mean() / s_het ** 2
            l_n = ((nz - (nn_ - mu_n) / s_n) ** 2).mean()
            loss = l_b + a.lambda_het * l_h + a.lambda_n * l_n
            if nx:  # lambda_x * standardised squared error per extra target
                xx_ = batch[5].to(dev, non_blocking=True)
                l_x = ((o[2].float() - (xx_ - mu_xt) / s_xt) ** 2).mean(0)
                loss = loss + a.lambda_x * l_x.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if a.clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            scaler.step(opt)
            scaler.update()
            sch.step()
            tl += [loss.item(), l_b.item(), l_h.item(), l_n.item()] + ([float(v) for v in l_x.tolist()] if nx else [])
        tl /= spe
        xl = "".join(f" {nm} {tl[4 + c]:.4f}" for c, nm in enumerate(xnames))
        print(f"fold {f} ep {ep + 1:3d}/{a.epochs} loss {tl[0]:.4f} (block {tl[1]:.4f} het {tl[2]:.4f} "
              f"logN {tl[3]:.4f}{xl}) lr {sch.get_last_lr()[0]:.2e} | {len(ds)} renders {time.time() - te:.0f}s "
              f"| {time.time() - t0:.0f}s", flush=True)
    t_train = time.time() - t0
    ckpt = {"state_dict": model.state_dict(), "consts": consts, "arch": a.arch}
    if nx or a.render_preset != "noisy":
        ckpt.update(extra_targets=xnames, render_preset=a.render_preset)
    torch.save(ckpt, out / f"fold{f}.pt")

    # (a) stage 1: held-out clean images rendered with K fixed seeds (independent of the fold and of --seed)
    t1 = time.time()
    K = 2 if a.smoke else a.k_val
    Xv, meta = [], []
    for j in va_rows:
        for k in range(K):
            img, info = render_noisy(P, int(j), np.random.default_rng([20261008, k, int(j)]), rR, M=rM)
            Xv.append(img)
            meta.append((int(j), k, info["path"]))
    Xv = np.stack(Xv)
    meta = np.array(meta)
    pv = predict(model, Xv, mu_t, sd_t, dev, amp_dtype, tta=a.tta, bs=a.pred_batch)
    bz, nz = pv[0], pv[1]
    bh, nh = mu_b + s_b * bz, mu_n + s_n * nz
    rows = meta[:, 0]
    m_all = stage1_metrics(bh, nh, b[rows], P["het4"][rows], logn[rows])
    cart = meta[:, 2] == 0
    m_cart = stage1_metrics(bh[cart], nh[cart], b[rows[cart]], P["het4"][rows[cart]], logn[rows[cart]])
    # K-averaged b-hat per image (the renders' mean prediction)
    bk = np.stack([bh[rows == j].mean(0) for j in va_rows])
    nk = np.array([nh[rows == j].mean() for j in va_rows])
    m_kavg = stage1_metrics(bk, nk, b[va_rows], P["het4"][va_rows], logn[va_rows])
    if nx:
        xh = mu_x + s_x * pv[2]
        mx_all = stage1_extra(xh, xt[rows], xnames)
        mx_cart = stage1_extra(xh[cart], xt[rows[cart]], xnames)
        mx_kavg = stage1_extra(np.stack([xh[rows == j].mean(0) for j in va_rows]), xt[va_rows], xnames)
    t_val = time.time() - t1

    # (b) real images, raw, prediction only
    t2 = time.time()
    Xr = np.stack([read_u8(i) for i in pred_ids])
    pr_ = predict(model, Xr, mu_t, sd_t, dev, amp_dtype, tta=a.tta, bs=a.pred_batch)
    bz, nz = pr_[0], pr_[1]
    bR, nR = mu_b + s_b * bz, mu_n + s_n * nz
    xR = mu_x + s_x * pr_[2] if nx else None
    t_pred = time.time() - t2
    # diagnostics on the REAL held-out clean images (their own clean rendering, no re-render)
    pos = {i: n for n, i in enumerate(pred_ids)}
    have = [j for j in va_rows if ids[j] in pos]
    m_real = mx_real = None
    if len(have) >= 5:
        pr = np.array([pos[ids[j]] for j in have])
        m_real = stage1_metrics(bR[pr], nR[pr], b[have], P["het4"][have], logn[have])
        if nx:
            mx_real = stage1_extra(xR[pr], xt[have], xnames)
    res = dict(fold=f, n_train=int(len(tr_rows)), n_heldout=int(len(va_rows)), K=K, consts=consts,
               stage1_renders=m_all, stage1_renders_cartoon=m_cart, stage1_renders_Kavg=m_kavg,
               heldout_real_clean=m_real, t_train_s=round(t_train, 1), t_val_s=round(t_val, 1),
               t_pred_s=round(t_pred, 1), epochs=a.epochs, renders=a.renders, steps=total)
    extra_npz = {}
    if nx or a.render_preset != "noisy":
        res.update(render_preset=a.render_preset, extra_targets=xnames)
    if nx:
        res.update(lambda_x=a.lambda_x, stage1_extra_renders=mx_all, stage1_extra_renders_cartoon=mx_cart,
                   stage1_extra_renders_Kavg=mx_kavg, heldout_real_clean_extra=mx_real)
        extra_npz = dict(x=xR, val_x=xh, x_names=np.array(xnames))
    np.savez(out / f"fold{f}.npz", pred_ids=np.array(pred_ids), b=bR, logn=nR, train_ids=ids[tr_rows],
             val_rows=meta, val_b=bh, val_logn=nh, **extra_npz)
    (out / f"fold{f}.json").write_text(json.dumps(res, indent=1))
    s = m_all
    print(f"fold {f} stage-1 (held-out renders, n={s['n']}): corr(sd b-hat, het4) {s['corr_het']:.3f}, partial | logN "
          f"{s['pcorr_het_given_logN']:.3f}, block corr {s['corr_block']:.3f} (within-image {s['corr_block_within']:.3f}),"
          f" corr logN {s['corr_logN']:.3f} | cartoon only: het {m_cart['corr_het']:.3f} partial "
          f"{m_cart['pcorr_het_given_logN']:.3f} | K-avg: het {m_kavg['corr_het']:.3f} partial "
          f"{m_kavg['pcorr_het_given_logN']:.3f}", flush=True)
    if m_real is not None:
        print(f"fold {f} real held-out clean images (no re-render, n={m_real['n']}): corr het {m_real['corr_het']:.3f},"
              f" partial {m_real['pcorr_het_given_logN']:.3f}, block {m_real['corr_block']:.3f}, logN "
              f"{m_real['corr_logN']:.3f}", flush=True)
    if nx:
        print(f"fold {f} extra targets, held-out renders: {_fmt_extra(mx_all)} | K-avg: {_fmt_extra(mx_kavg)}"
              + (f" | real held-out clean: {_fmt_extra(mx_real)}" if mx_real else ""), flush=True)
    print(f"fold {f}: train {t_train:.0f}s, held-out renders {t_val:.0f}s, {len(pred_ids)} real images "
          f"(tta{a.tta}) {t_pred:.0f}s", flush=True)
    return res


def aggregate(a, out, P, folds_of, pred_ids):
    """fold{f}.npz -> het_cnn_{train,test}.parquet + score.json."""
    done = [f for f in range(5) if (out / f"fold{f}.npz").exists()]
    if not done:
        print("nothing to aggregate")
        return
    Z = {}
    for f in done:
        with np.load(out / f"fold{f}.npz") as z:
            Z[f] = {k: z[k] for k in z.files}
    ids = list(Z[done[0]]["pred_ids"])
    for f in done:
        assert list(Z[f]["pred_ids"]) == ids, f"fold {f} predicted another image list"
    xn = {f: list(Z[f]["x_names"]) if "x_names" in Z[f] else [] for f in done}
    xnames = xn[done[0]]
    assert all(v == xnames for v in xn.values()), f"folds were trained with different --extra-targets: {xn}"
    fold_of_clean = dict(zip(P["ids"].tolist(), folds_of.tolist()))
    B = np.stack([Z[f]["b"] for f in done])  # (nf, N, 16)
    Nn = np.stack([Z[f]["logn"] for f in done])
    Xx = np.stack([Z[f]["x"] for f in done]) if xnames else None  # (nf, N, n_extra)
    rows_b, rows_n, rows_x, src = [], [], [], []
    for n, i in enumerate(ids):
        if i in fold_of_clean:  # clean train image: only the fold model that did not train on it
            fo = fold_of_clean[i]
            if fo in done and i not in set(Z[fo]["train_ids"].tolist()):
                rows_b.append(B[done.index(fo), n]); rows_n.append(Nn[done.index(fo), n]); src.append(f"oof{fo}")
                if xnames:
                    rows_x.append(Xx[done.index(fo), n])
            else:
                rows_b.append(np.full(16, np.nan)); rows_n.append(np.nan); src.append("none")
                if xnames:
                    rows_x.append(np.full(len(xnames), np.nan))
        else:  # noisy train image or test image: mean over the fold models run (b-hat averaged per block)
            rows_b.append(B[:, n].mean(0)); rows_n.append(Nn[:, n].mean()); src.append(f"mean{len(done)}")
            if xnames:
                rows_x.append(Xx[:, n].mean(0))
    bb = np.array(rows_b)
    D = pd.DataFrame({"ID": ids, "het_cnn": bb.std(1), "logN_cnn": rows_n, "bmean_cnn": bb.mean(1)})
    for k in range(16):
        D[f"b{k:02d}"] = bb[:, k]
    D["src"] = src
    if xnames:  # extra columns appended after the old ones
        xx = np.array(rows_x)
        for c, nm in enumerate(xnames):
            D[f"{nm}_cnn"] = xx[:, c]
    for split, pre in (("train", "TRAIN"), ("test", "TEST")):
        d = D[D.ID.str.startswith(pre)].reset_index(drop=True)
        if len(d):
            d.to_parquet(out / f"het_cnn_{split}.parquet", index=False)
            print(f"wrote {out / f'het_cnn_{split}.parquet'}: {len(d)} images, sources "
                  f"{d.src.value_counts().to_dict()}; sums het_cnn {d.het_cnn.sum():.4f} logN_cnn "
                  f"{d.logN_cnn.sum():.4f} bmean_cnn {d.bmean_cnn.sum():.4f}"
                  + "".join(f" {nm}_cnn {d[f'{nm}_cnn'].sum():.4f}" for nm in xnames))
    per = {f: json.loads((out / f"fold{f}.json").read_text()) for f in done}
    # pooled stage-1 over the done folds' held-out renders
    vb = np.concatenate([Z[f]["val_b"] for f in done])
    vn = np.concatenate([Z[f]["val_logn"] for f in done])
    vr = np.concatenate([Z[f]["val_rows"][:, 0] for f in done]).astype(int)
    pooled = stage1_metrics(vb, vn, P["b"][vr], P["het4"][vr], np.log(P["n_eff"][vr])) if len(done) > 1 else None
    pooled_x = None
    if xnames and len(done) > 1:
        vx = np.concatenate([Z[f]["val_x"] for f in done])
        pooled_x = stage1_extra(vx, extra_target_values(P, xnames)[vr], xnames)
    prev = json.loads((out / "score.json").read_text()) if (out / "score.json").exists() else {}
    cmds = prev.get("commands", [])
    cmd = "python -m src.het_cnn " + " ".join(sys.argv[1:])
    if cmd not in cmds:
        cmds.append(cmd)
    R, M = PRESETS[a.render_preset]
    score = dict(folds_done=done, stage1_per_fold={str(f): per[f] for f in done}, stage1_pooled=pooled,
                 command=cmd, commands=cmds, args={k: v for k, v in vars(a).items()},
                 render=R, m_ranges=M,
                 notes="targets from src.het_blocks on the clean originals; hardness never read; test images "
                       "prediction only; clean train rows = OOF fold model, noisy train/test = mean over folds_done")
    if xnames:
        score.update(extra_targets=xnames, stage1_extra_pooled=pooled_x,
                     notes_extra="extra targets fd/pore/asp from the prep arrays of the clean originals "
                                 "(extra_target_values); columns <name>_cnn appended to the parquets")
    if a.render_preset != "noisy":
        score.update(render_preset=a.render_preset, render_preset_note=MID_NOTE)
    (out / "score.json").write_text(json.dumps(score, indent=1, default=str))
    print(f"wrote {out / 'score.json'} (folds {done})")


DUMP_COLS = ["het_cnn", "logN_cnn", "bmean_cnn"]
X_COLS = [f"{nm}_cnn" for nm in EXTRA_TARGETS]  # appended to a dump when the run has them


def chunk_frame(out, chunk):
    """ID-ordered chunk 0 = the 500 train images, 1 = test images 0-499, 2 = test images 500-999 (sample_submission
    order) with the three aggregate columns, then any extra-target columns of the run (X_COLS order)."""
    if chunk == 0:
        ids, split = pd.read_csv(DATA_DIR / "train.csv", usecols=["ID"]).ID, "train"
    else:
        ids, split = pd.read_csv(DATA_DIR / "sample_submission.csv", usecols=["ID"]).ID[500 * (chunk - 1):500 * chunk], "test"
    d = pd.read_parquet(out / f"het_cnn_{split}.parquet").set_index("ID")
    miss = [i for i in ids if i not in d.index]
    if miss:
        raise SystemExit(f"{out.name}: {len(miss)} of the {len(ids)} chunk-{chunk} images are not in het_cnn_{split}"
                         f".parquet (a --smoke run?)")
    cols = DUMP_COLS + [c for c in X_COLS if c in d.columns]
    return d.loc[ids, cols].reset_index()


def run_dump(a):
    """Print chunk(s) as CSV text with 4 decimals plus per-column sums of the printed (rounded) values."""
    out = OUT_ROOT / a.dump
    for c in (a.chunk if a.chunk is not None else [0, 1, 2]):
        d = chunk_frame(out, c)
        cols = list(d.columns[1:])
        print(f"# het_cnn {a.dump} chunk {c} ({len(d)} rows)")
        print("ID," + ",".join(cols))
        for r in d.itertuples(index=False):
            print(r.ID + "," + ",".join("nan" if not np.isfinite(v) else f"{v:.4f}" for v in r[1:]))
        sums = {k: float(np.nansum(np.round(d[k].values, 4))) for k in cols}
        print(f"# sums chunk {c}: " + " ".join(f"{k} {v:.4f}" for k, v in sums.items()) + f" nan {int(d[cols].isna().sum().sum())}")


def run_ingest(a):
    """Cloud side: rebuild data/het_cnn/<name>/het_cnn_{train,test}.parquet (the 3 aggregate columns, plus any extra
    columns the dump has) from the pasted chunk texts of --dump (files in chunk order 0, 1, 2), checking each chunk's
    IDs and '# sums' line (every column)."""
    import io
    name, files = a.ingest[0], a.ingest[1:]
    parts = []
    for c, fp in enumerate(files):
        txt = Path(fp).read_text(encoding="utf-8-sig")  # tolerates a BOM (PowerShell Out-File -Encoding utf8)
        rows = [ln for ln in txt.splitlines() if ln and not ln.startswith("#")]
        d = pd.read_csv(io.StringIO("\n".join(rows)))
        sl = [ln for ln in txt.splitlines() if ln.startswith("# sums")]
        assert sl, f"{fp}: no '# sums' line"
        tok = sl[0].split(":", 1)[1].split()
        ref = {tok[i]: float(tok[i + 1]) for i in range(0, len(tok) - 1, 2)}
        cols = [k for k in d.columns if k != "ID"]
        assert cols[:len(DUMP_COLS)] == DUMP_COLS and all(k in X_COLS for k in cols[len(DUMP_COLS):]), \
            f"{fp}: unexpected columns {cols}"
        assert c == 0 or cols == list(parts[0].columns[1:]), f"{fp}: columns differ from chunk 0"
        for k in cols:
            assert k in ref, f"{fp}: no sum for {k} in the '# sums' line"
            assert abs(np.nansum(d[k].values) - ref[k]) < 5e-3, f"{fp}: {k} sum {np.nansum(d[k].values)} != {ref[k]}"
        exp = (pd.read_csv(DATA_DIR / "train.csv", usecols=["ID"]).ID if c == 0 else
               pd.read_csv(DATA_DIR / "sample_submission.csv", usecols=["ID"]).ID[500 * (c - 1):500 * c])
        assert list(d.ID) == list(exp), f"{fp}: IDs are not chunk {c} in ID order"
        parts.append(d)
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    parts[0].to_parquet(out / "het_cnn_train.parquet", index=False)
    if len(parts) == 3:
        pd.concat(parts[1:], ignore_index=True).to_parquet(out / "het_cnn_test.parquet", index=False)
    print(f"ingested {len(parts)} chunks -> {out}")


def main(a):
    if a.prep_only:
        run_prep(a.threads)
        return
    if a.selftest:
        selftest()
        return
    if a.calib:
        run_calib(a)
        return
    if a.dump:
        run_dump(a)
        return
    if a.ingest:
        run_ingest(a)
        return
    if a.out is None:
        raise SystemExit("--out NAME is required for training")
    use_cuda = a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available())
    dev = torch.device("cuda" if use_cuda else "cpu")
    a.arch = a.arch or (GPU_ARCH if use_cuda else CPU_ARCH)
    amp_dtype = None
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.set_num_threads(a.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    import cv2
    cv2.setNumThreads(1)
    P = load_prep(a.prep)
    fo = pd.read_csv(DATA_DIR / "folds.csv").set_index("ID")
    folds_of = fo.loc[P["ids"], "fold"].values.astype(int)
    xt = extra_target_values(P, a.extra_targets) if a.extra_targets else None
    if xt is not None:
        print("extra targets (clean originals, from the prep arrays): " + ", ".join(
            f"{nm} mean {xt[:, c].mean():.4f} sd {xt[:, c].std():.4f}" for c, nm in enumerate(a.extra_targets)),
            flush=True)
    pred_ids = (pd.read_csv(DATA_DIR / "train.csv", usecols=["ID"]).ID.tolist()
                + pd.read_csv(DATA_DIR / "sample_submission.csv", usecols=["ID"]).ID.tolist())
    if a.smoke:  # a few clean + noisy train images (incl. held-out fold-0 clean ones) and a few test images
        tr_ids = pred_ids[:500]
        clean = set(P["ids"].tolist())
        va0 = [i for i in P["ids"][folds_of == (a.folds or [0])[0]][:8]]
        pred_ids = va0 + [i for i in tr_ids if i not in clean][:8] + pred_ids[500:508]
    out = OUT_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    print(f"het_cnn: device {dev} amp {amp_dtype} arch {a.arch} threads {a.threads} workers {a.workers} "
          f"epochs {a.epochs} renders {a.renders} batch {a.batch} lr {a.lr} -> {out}"
          + (f" | extra targets {a.extra_targets} lambda_x {a.lambda_x}" if a.extra_targets else "")
          + (f" | render preset {a.render_preset}" if a.render_preset != "noisy" else ""), flush=True)
    t = time.time()
    for f in (a.folds if a.folds is not None else range(5)):
        if (out / f"fold{f}.npz").exists() and not a.overwrite:
            prev = json.loads((out / f"fold{f}.json").read_text()) if (out / f"fold{f}.json").exists() else {}
            got = (prev.get("render_preset", "noisy"), prev.get("extra_targets", []))
            want = (a.render_preset, list(a.extra_targets or []))
            if got != want:
                raise SystemExit(f"{out}/fold{f}: cached with (render preset, extra targets) {got}, but this call "
                                 f"asks for {want}: use another --out or --overwrite")
            print(f"fold {f}: cached", flush=True)
            continue
        train_fold(a, f, P, folds_of, dev, amp_dtype, pred_ids, out, xt=xt)
    aggregate(a, out, P, folds_of, pred_ids)
    print(f"total wall {time.time() - t:.0f}s", flush=True)


# --------------------------------------------------------------------------------------------------------- selftest
def selftest(n_img=6):
    """Unit tests: (1) numpy and torch D4 ops agree; (2) targets recomputed from a D4-transformed label map (and pore
    map) equal the GRID_PERM-permuted targets, N_eff unchanged; (3) GRID_INV undoes GRID_PERM; (4) the TTA path of
    predict (transform, model, un-permute, average) returns the original-frame block values for a model that outputs
    exact 64-px block means; (5) renders are seeded-deterministic uint8 256x256."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 256, 256)).astype(np.float32)
    for k in range(8):
        assert np.array_equal(d4_np(x, k), d4_t(torch.from_numpy(x), k).numpy()), f"d4 numpy/torch differ at k={k}"
    print("ok 1: numpy and torch D4 ops agree (8 ops)")
    P = load_prep()
    rows = list(range(min(n_img, len(P["ids"]))))
    nchk = 0
    for j in rows:
        ws = P["ws"][j].astype(np.int64)
        pore = P["pore"][j]
        b0 = block_targets(ws, pore)
        assert np.allclose(b0, P["b"][j], equal_nan=True), "prep targets not reproduced"
        n0 = n_eff_of(ws, pore)
        assert abs(n0 - P["n_eff"][j]) < 1e-9, "prep N_eff not reproduced"
        for k in range(8):
            wk = np.ascontiguousarray(d4_np(ws, k))
            bk = block_targets(wk, pore)
            perm = P["b"][j][GRID_PERM[k]]
            assert np.array_equal(np.isnan(bk), np.isnan(perm)), f"NaN pattern differs (img {j}, op {k})"
            assert np.allclose(bk, perm, rtol=0, atol=1e-12, equal_nan=True), f"targets differ (img {j}, op {k})"
            assert np.allclose(bk[GRID_INV[k]], P["b"][j], rtol=0, atol=1e-12, equal_nan=True)
            assert abs(np.std(bk[~np.isnan(bk)]) - P["het4"][j]) < 1e-12
            assert abs(n_eff_of(wk, pore) - P["n_eff"][j]) < 1e-9
            nchk += 1
    # synthetic NaN blocks (the clean prep has none): every grain touching block 0 or block 6 counts as a pore
    from .het_blocks import GRID
    ws = P["ws"][0].astype(np.int64)
    pore = P["pore"][0].copy()
    pore[np.unique(ws[(GRID == 0) | (GRID == 6)])] = True
    b0 = block_targets(ws, pore)
    assert np.isnan(b0).sum() >= 2
    for k in range(8):
        bk = block_targets(np.ascontiguousarray(d4_np(ws, k)), pore)
        assert np.array_equal(np.isnan(bk), np.isnan(b0[GRID_PERM[k]]))
        assert np.allclose(bk, b0[GRID_PERM[k]], rtol=0, atol=1e-12, equal_nan=True)
        nchk += 1
    print(f"ok 2: targets recomputed from D4-transformed label maps == permuted targets ({nchk} image-op pairs, "
          f"incl. a synthetic case with {int(np.isnan(b0).sum())} NaN blocks); het4, N_eff invariant")
    for k in range(8):
        assert np.array_equal(GRID_PERM[k][GRID_INV[k]], np.arange(16))
    print("ok 3: GRID_INV inverts GRID_PERM")

    class BlockMean(nn.Module):  # 'model' whose cell output is the exact 64-px block mean of its input
        def forward(self, x):
            return F.avg_pool2d(x, 64).flatten(1), x.mean((1, 2, 3))

    X8 = np.stack([P["im8"][j] for j in rows[:3]])
    ref = X8.astype(np.float64).reshape(3, 4, 64, 4, 64).mean((2, 4)).reshape(3, 16) / 255.0
    for tta in (1, 8):
        bz, _ = predict(BlockMean(), X8, torch.tensor(0.0), torch.tensor(1.0), torch.device("cpu"), None, tta=tta)
        assert np.allclose(bz, ref, atol=1e-5), f"TTA un-permutation wrong (tta={tta})"
    for k in range(8):  # one view alone, un-permuted, is the original-frame block mean too
        xk = torch.from_numpy(X8.astype(np.float32) / 255.0)[:, None]
        bk = F.avg_pool2d(d4_t(xk, k), 64).flatten(1).numpy()
        assert np.allclose(bk[:, GRID_INV[k]], ref, atol=1e-5) and np.allclose(bk, ref[:, GRID_PERM[k]], atol=1e-5)
    print("ok 4: TTA path (D4 view -> model -> GRID_INV -> mean) returns original-frame block values")
    for force in ("cartoon", "m"):
        r1, _ = render_noisy(P, 0, np.random.default_rng(5), force=force)
        r2, _ = render_noisy(P, 0, np.random.default_rng(5), force=force)
        assert r1.dtype == np.uint8 and r1.shape == (256, 256) and np.array_equal(r1, r2)
    ds = RenderSet(PREP_FP, [0, 1], 2, 0, 0)
    i1, i2 = ds[5], ds[5]
    assert all(torch.equal(u, v) for u, v in zip(i1, i2))
    print("ok 5: renders and dataset items are seeded-deterministic uint8 256x256")

    # (6) extra targets: definitions, D4 invariance, dataset item, TTA path
    from skimage import measure
    names = list(EXTRA_TARGETS)
    xt = extra_target_values(P, names)
    assert np.array_equal(xt[:, 0], P["dark_area"]), "fd != prep dark_area"
    for j in rows[:3]:
        ws = P["ws"][j].astype(np.int64)
        pore = P["pore"][j]
        rp = measure.regionprops(ws)
        lab = np.array([q.label for q in rp])
        ar = np.array([q.area for q in rp], float)
        asp = np.array([q.axis_major_length for q in rp]) / np.maximum([q.axis_minor_length for q in rp], 1)
        inner = np.array([q.bbox[0] > 0 and q.bbox[1] > 0 and q.bbox[2] < 256 and q.bbox[3] < 256 for q in rp])
        s = inner & ~pore[lab]
        ref_asp = (np.log(asp[s]) * ar[s]).sum() / ar[s].sum()
        assert abs(xt[j, 2] - ref_asp) < 1e-9, f"asp differs from skimage regionprops ({xt[j, 2]} vs {ref_asp})"
        assert abs(xt[j, 1] - ar[pore[lab]].sum() / ws.size) < 1e-12, "pore fraction differs from the grain areas"
        for k in range(8):
            Pk = {**P, "ids": P["ids"][j:j + 1], "ws": d4_np(P["ws"][j:j + 1], k), "dark": P["dark"][j:j + 1],
                  "pore": P["pore"][j:j + 1]}
            xk = extra_target_values(Pk, names)[0]
            assert np.allclose(xk, xt[j], rtol=0, atol=1e-12), f"extra targets not D4-invariant (img {j}, op {k})"
    dsx = RenderSet(PREP_FP, [0, 1], 2, 0, 0, xt=xt)
    for idx in (0, 3, 5):
        a_, b_ = ds[idx], dsx[idx]
        assert len(a_) == 5 and len(b_) == 6 and all(torch.equal(u, v) for u, v in zip(a_, b_))
        j = [0, 1][idx % 2]
        assert torch.equal(b_[5], torch.from_numpy(xt[j].astype(np.float32)))
    net = HetNet("resnet18", pretrained=False, n_extra=3).eval()
    out3 = net(torch.zeros(2, 1, 256, 256))
    assert len(out3) == 3 and out3[2].shape == (2, 3)
    assert len(HetNet("resnet18", pretrained=False).eval()(torch.zeros(1, 1, 256, 256))) == 2

    class BlockMeanX(nn.Module):  # third output: D4-invariant scalars (image mean, mean of squares)
        def forward(self, x):
            return (F.avg_pool2d(x, 64).flatten(1), x.mean((1, 2, 3)),
                    torch.stack([x.mean((1, 2, 3)), (x * x).mean((1, 2, 3))], 1))

    ref_x = np.stack([X8.astype(np.float64).mean((1, 2)) / 255.0, ((X8.astype(np.float64) / 255.0) ** 2).mean((1, 2))], 1)
    bz, nz, xz = predict(BlockMeanX(), X8, torch.tensor(0.0), torch.tensor(1.0), torch.device("cpu"), None, tta=8)
    assert np.allclose(bz, ref, atol=1e-5) and np.allclose(xz, ref_x, atol=1e-5)
    print(f"ok 6: extra targets fd == prep dark_area, asp == skimage regionprops logasp_aw, pore == pore-grain area; "
          f"all D4-invariant ({len(rows[:3]) * 8} image-op pairs); dataset items unchanged + extras; HetNet / predict "
          f"return the extra outputs (TTA-averaged)")

    # (7) render presets: 'noisy' = the defaults exactly; 'mid' deterministic, in range, different
    for t in range(6):
        for force in (None, "cartoon", "m"):
            r0, i0 = render_noisy(P, t % len(P["ids"]), np.random.default_rng([7, t]), force=force)
            r1, i1_ = render_noisy(P, t % len(P["ids"]), np.random.default_rng([7, t]), RENDER, force, M_RANGES)
            assert np.array_equal(r0, r1) and i0 == i1_
            r2, i2_ = render_noisy(P, t % len(P["ids"]), np.random.default_rng([7, t]), PRESETS["mid"][0], force,
                                   PRESETS["mid"][1])
            r3, _ = render_noisy(P, t % len(P["ids"]), np.random.default_rng([7, t]), PRESETS["mid"][0], force,
                                 PRESETS["mid"][1])
            assert np.array_equal(r2, r3) and r2.dtype == np.uint8 and r2.shape == (256, 256)
            assert not np.array_equal(r0, r2)
            if i2_["path"] == 0:
                for key, rk in (("c_d", "c_d"), ("e", "e"), ("noise", "noise"), ("blur", "blur")):
                    lo_, hi_ = RENDER_MID[rk]
                    assert lo_ <= i2_[key] <= hi_, (key, i2_[key])
                assert i2_["line"] == 0 or 0.3 <= i2_["line"] <= 1.0
            else:
                assert 0.8 <= i2_["c_d"] <= 1.0 and 1.0 <= i2_["e"] <= 1.3 and 8 <= i2_["noise"] <= 13
    dm = RenderSet(PREP_FP, [0, 1], 2, 0, 0, preset="mid")
    assert all(torch.equal(u, v) for u, v in zip(dm[5], dm[5])) and not torch.equal(dm[5][0], ds[5][0])
    print("ok 7: render preset 'noisy' == the default renderer bit for bit; 'mid' renders deterministic, in its ranges")
    print("selftest passed")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prep", nargs="?", const="__BUILD__", default=str(PREP_FP),
                    help="bare --prep: build data/het_cnn/prep.npz and exit; --prep PATH: use that prep file")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--calib", type=int, default=0, help="number of renders for the calibration table (0 = off)")
    ap.add_argument("--calib-dir", default=None, help="output folder for the calibration table/montage (not the repo)")
    ap.add_argument("--calib-tag", default=None)
    ap.add_argument("--dump", default=None, help="print the aggregate of run NAME as 4-decimal CSV chunks + sums")
    ap.add_argument("--chunk", type=int, nargs="*", default=None, help="--dump: chunks 0 (train), 1, 2 (test)")
    ap.add_argument("--ingest", nargs="+", default=None, help="NAME chunk0.txt [chunk1.txt chunk2.txt]")
    ap.add_argument("--arch", default=None, help=f"timm name (default: {GPU_ARCH} on cuda, {CPU_ARCH} on cpu)")
    ap.add_argument("--scratch", action="store_true", help="no pretrained weights")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=32)
    ap.add_argument("--renders", type=int, default=4, help="renders of every training image per epoch")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--pred-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--warmup", type=float, default=0.05, help="fraction of steps with linear warm-up")
    ap.add_argument("--final-lr", type=float, default=0.01, help="cosine floor as a fraction of --lr")
    ap.add_argument("--clip", type=float, default=2.0)
    ap.add_argument("--lambda-het", type=float, default=1.0)
    ap.add_argument("--lambda-n", type=float, default=0.5)
    ap.add_argument("--extra-targets", nargs="*", default=None, choices=EXTRA_TARGETS,
                    help="extra label-free targets of the clean originals (fd pore asp); default none = as before")
    ap.add_argument("--lambda-x", type=float, default=0.5, help="loss weight of each extra target (standardised SE)")
    ap.add_argument("--render-preset", default="noisy", choices=sorted(PRESETS),
                    help="noisy (default, as before) | mid (RENDER_MID, for raw ic_noise 9.5-12)")
    ap.add_argument("--k-val", type=int, default=4, help="fixed-seed renders per held-out clean image")
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=None, help="run name -> data/het_cnn/<out>/")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run: 24 train / 8 held-out images, K=2, "
                                                         "24 real images predicted")
    a = ap.parse_args(argv)
    if a.extra_targets is not None:  # dedupe, keep the given order; an empty list means none
        a.extra_targets = list(dict.fromkeys(a.extra_targets)) or None
    a.prep_only = a.prep == "__BUILD__"
    if a.prep_only:
        a.prep = str(PREP_FP)
    return a


if __name__ == "__main__":
    main(parse())
