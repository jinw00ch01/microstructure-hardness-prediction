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
  ... --render-preset real --eval-presets noisy real   # D1 prior (RENDER_REAL) + stage-1 on fixed noisy / real renders
  python -m src.het_cnn --eval-only ev2s_loc_e32r4 --folds 0 1 --eval-presets noisy real   # score_eval.json only

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

Options added 2026-10-09 (evening), for within-image localisation (defaults reproduce the earlier runs bit-identically:
render RNG streams, model init, losses, outputs; checked against the pre-options file (git: parent of the commit that
added them) by --selftest check 12 and a 1-epoch CPU run). New randomness comes from a separate generator
default_rng([seed, fold, epoch, i, 9001]) (AUG_TAG) and only when an option is on; the render parameters, the path draw
(cartoon / degrade_m) and the D4 op stay on the item's own generator. Zoom and mosaic act on cartoon-path renders only
(the degrade_m path, p_m = 25%, is never augmented), so their probabilities are per cartoon render.
--zoom LO HI [--p-zoom P (0.5; 0-1, only with --zoom)]: label-map rescaling (zoom_source). s ~ logUniform(LO, HI);
  output pixel (y, x) samples the source at (oy + (y + .5)/s - .5, ox + (x + .5)/s - .5) with symmetric reflection
  beyond the image edge (s < 1 sees reflected copies around the whole original; s > 1 a random window inside it).
  Nearest neighbour for the label map and the render-pore mask, bilinear for resid and the source grey (used only for
  the mean-grey fix); the shading surface is the source's, evaluated on the output canvas. Then split_merge(): relabel()
  makes connected components (4-connectivity) of equal labels separate grains, each carrying sn_tab / dark / pore of its
  source label; NON-PORE components below MIN_GRAIN = 16 px (seam slivers, necks broken by nearest-neighbour
  down-scaling; het_blocks.segment never yields grains below 17 px, so the prep maps have none) are absorbed by
  neighbouring non-pore grains (grown in from the rim); pore pieces are never merged, so every pixel keeps its pore flag
  and the target pores stay inside the render-pore mask (a sliver with no non-pore neighbour stays its own grain).
  Targets are recomputed on the final map with the original definitions (targets_of: block_targets_fast ==
  block_targets, het4 = sd of the valid b, n_eff_fast == n_eff_of, extras -- asp falls back to all non-pore grains when
  no interior one is left -- and cells). Caveats: boundary-line depth (resid) and pore blobs scale with s; small grains
  of the source vanish at s < 1 while their old boundary lines stay in resid (s_line > 0). Zoom-out (s < 1) reflects the
  source at its edges, and a grain cut by the original border merges with its mirror copy (2x, 4x at corners; no
  boundary line there, so image and targets agree): the canvas corner / edge blocks then hold bigger grains than in real
  images, where border truncation makes them smaller. Mean within-image block deviation corner / edge / centre (60 draws
  each): prep -0.034 / -0.013 / +0.060; s=0.85 +0.134 / -0.037 / -0.060 (het4 0.224 vs 0.178 for the same sources);
  s=0.70 +0.047 / -0.017 / -0.014; zoom-in s=1.2 -0.109 / -0.011 / +0.132 (the border truncation of larger grains, as in
  real images). Splitting the mirror copies into separate grains overshoots the other way (corners -0.11 at s=0.85: thin
  truncated slivers), so it is not done; use LO >= 1 (zoom-in only) when that corner pattern matters.
--mosaic P: with probability P a cartoon render is a composite of image j and a second training image j2 of the fold
  (mosaic_mask: 75% a Gaussian-smoothed white-noise field, sigma 32-72 px (periodic FFT filter), thresholded at a
  quantile 0.3-0.7; 25% a straight line at a random angle through a random point of the central half). Composite label
  map = j inside the mask, j2 (labels offset) outside, relabelled as for zoom; j2's resid and grey are rescaled by S_j /
  S_j2 per pixel and the composite uses j's shading surface, so both regions sit on the same illumination (no brightness
  step at the seam); ONE draw of the render parameters for the whole composite; mean-grey fix to the composite source
  grey. Zoom composes: each source is zoomed independently (own s, own p-zoom draw) before compositing. Caveat:
  any-partner composites of a large- and a small-grain image lie far outside the real het4 range (200 forced mosaics:
  het4 mean 0.33 sd 0.17 vs prep 0.18 / 0.07, 23% above the prep max; sd of het4 given logN 0.167 vs 0.039).
  --mosaic-dlogn D: draw j2 only among the training images with |log N_eff(j2) - log N_eff(j)| <= D (the nearest one if
  none; same draws, narrower candidate list). 200 forced mosaics: D 0.5 -> het4 0.212 sd 0.065, 1% above the prep max,
  sd given logN 0.049; D 0.3 -> 0.197 / 0.040; D 0.15 -> 0.186 / 0.037 (log N_eff sd 0.62). fold{f}.json 'train_het4'
  reports the het4 of the training renders actually drawn.
--head {grid4,fpn}: grid4 = HetNet (unchanged). fpn = FPNNet: timm features_only backbone, stride-8/16/32 maps -> 1x1
  laterals (--fpn-ch, 128) merged at stride 16 (nearest up-sampling of s32, 2x2 average pooling of s8) -> 3x3 conv + BN
  + ReLU -> block head (adaptive_avg_pool2d(4) + 1x1 conv, GRID order) and cell head (1x1 conv -> 16x16 map of 16-px
  cells); logN / extras from the global mean of the stride-32 map. --lambda-cell (1.0 with fpn): + mean over valid cells
  of ((b-hat_cell - b_cell)/s_b)^2, b_cell = cell_targets (valid when >= 32 of its 256 pixels are non-pore),
  D4-transformed with the image. predict() un-transforms the cell map with the inverse D4 op (d4_inv_t); het stays sd of
  the 16 block b-hat; stage-1 also reports pcorr_het_cell (cells -> blocks by -2 log mean exp(-b_cell/2)),
  corr_cell(_within).
--lambda-within W: + W * mean over images of the mean over valid blocks of (((b-hat_k - mean b-hat) - (b_k - mean b)) /
  s_w)^2, s_w = sd of the within-image block deviations of the training fold's prep targets.
--screen: design screening -- the real-image pass predicts only each fold's held-out real clean images
  (heldout_real_clean); fold{f}.json / score.json as usual (plus stage-1 on held-out renders split at the fold's median
  het4: stage1_renders_het4_hi / _lo, and per-epoch wall / data-wait seconds); no het_cnn_{train,test}.parquet (those of
  an earlier full run in the same --out are renamed *.parquet.stale).
All runs: stage-1 dicts gain slope_het4_on_hat_given_logN (OLS slope of het4 on het-hat, both residualised on logN =
partial corr x sd ratio: > 1 only when het-hat is shrunk AND ranks het4 well, a poor ranking pulls it toward 0) and
sdratio_het4_over_hat_given_logN (sd(het4 | logN) / sd(het-hat | logN), scale only: > 1 = shrinkage); read them with
pcorr_het_given_logN. Fold json: train_het4 (het4 of the training renders actually drawn vs the prep's) and per-epoch
het4 mean / max in ep_log. Fold caches record the non-default options ('opts') and are not reused across different
options; aggregate() refuses to pool folds whose (render preset, extra targets, options) differ from the call's.

Options added 2026-10-09 (night), route d1010 K (defaults reproduce the earlier runs bit-identically: selftest check 13
compares the default flags with the file before these options, git: parent of the commit that added RENDER_REAL):
--render-preset real: RENDER_REAL / M_RANGES_REAL (= RENDER with p_line 0.9, line 0.2-1.1, blur 0.3-1.3, e 1.0-2.0;
  M_RANGES with sb 0.4-1.4), the D1 training domain; --calib with it compares against raw ic_noise >= 12 (as noisy).
--eval-presets PRESET [PRESET ...]: after a fold is trained, score its model on held-out renders of every listed preset
  (eval_render_set: the fold's held-out clean images, K = --k-val renders each, seeds default_rng([EVAL_SEED, k, j]),
  i.e. the stage-1 seeds; they depend on neither --seed, the fold model nor the training preset, so the renders of a
  preset are identical across runs; for the training preset they are exactly the stage-1 renders). It also predicts
  the real line-free noisy TRAIN images (G1: raw ic_noise >= 9.5 and ic_ridge_snr < 0.5 in features_v3; prediction
  only, no label) for the logN / agreement sanity checks. Writes fold{f}_eval.json / fold{f}_eval.npz; score.json gains
  one new key 'stage1_eval' (per fold, folds-mean P / W, pooled metrics per preset, pooled real held-out clean R, real
  G1 logN summary); every existing key is unchanged. P = pcorr_het_given_logN, W = corr_block_within of all K renders.
--eval-only RUN: no training. Loads data/het_cnn/RUN/fold{f}.pt (--folds, default every fold present), rebuilds the
  model from the checkpoint (arch, head / fpn_ch, extra targets; no pretrained download) and computes only the
  --eval-presets metrics (as above) with the held-out rows of fold{f}.npz when present (else data/folds.csv); writes
  data/het_cnn/RUN/score_eval.json and score_eval_fold{f}.npz; never touches score.json, fold{f}.json/.npz or parquets.
  Example: python -m src.het_cnn --eval-only ev2s_loc_e32r4 --folds 0 1 --eval-presets noisy real --device cuda

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
# 'real' preset (--render-preset real, 2026-10-09 night, route d1010 K, design D1): the noisy preset widened toward what
# the label-free render-gap study inferred for the real line-free noisy images (G1: raw ic_noise >= 9.5 and ic_ridge_snr
# < 0.5): real G1 keeps faint boundary lines (inferred line about 0.75) and is less blurred (blur about 0.6) than the
# noisy renders (fast-statistic domain AUC 0.90). P7r (p_line 1.0, line 0.4-1.0, blur 0.3-0.9, e 1.0-1.6, sb 0.4-1.2)
# cut the AUC to 0.77 but left about 45% of real G1 outside its box; these ranges cover that part too.
RENDER_REAL = dict(RENDER, p_line=0.9, line=(0.2, 1.1), blur=(0.3, 1.3), e=(1.0, 2.0))
M_RANGES_REAL = dict(M_RANGES, sb=(0.4, 1.4))
REAL_NOTE = ("real: RENDER with p_line 0.9, line 0.2-1.1, blur 0.3-1.3, e 1.0-2.0; degrade_m branch (p_m 0.25) with sb "
             "0.4-1.4 (M_RANGES otherwise); label-free gates in the d1010 K pre-registration")
PRESETS = {"noisy": (RENDER, M_RANGES), "mid": (RENDER_MID, M_RANGES_MID), "real": (RENDER_REAL, M_RANGES_REAL)}
CALIB_BAND = {"noisy": (12.0, np.inf), "mid": (NOISE_MAX, 12.0), "real": (12.0, np.inf)}  # --calib's real train images
PRESET_NOTES = {"mid": MID_NOTE, "real": REAL_NOTE}  # score.json render_preset_note
EVAL_SEED = 20261008  # held-out stage-1 renders: default_rng([EVAL_SEED, k, j]) (train_fold, eval_render_set)
G1_RIDGE_MAX = 0.5  # line-free noisy images (G1): raw ic_noise >= NOISE_MAX and ic_ridge_snr < G1_RIDGE_MAX
EXTRA_TARGETS = ("fd", "pore", "asp")  # --extra-targets choices; output columns <name>_cnn
AUG_TAG = 9001  # last seed word of the augmentation generator (zoom / mosaic), separate from the render stream
HEADS = ("grid4", "fpn")


# ------------------------------------------------------------------------------------------------------ D4 geometry
def d4_np(x, k):
    """The 8 dihedral ops on the last two axes, identical to src.train_cnn.d4 (rot90 by k%4, then flip(-1) if k>=4)."""
    x = np.rot90(x, k % 4, axes=(-2, -1))
    return x[..., ::-1] if k >= 4 else x


def d4_t(x, k):  # torch version (src.train_cnn.d4)
    x = torch.rot90(x, k % 4, dims=(-2, -1))
    return x.flip(-1) if k >= 4 else x


def d4_inv_t(x, k):
    """Inverse of d4_t: d4_inv_t(d4_t(x, k), k) == x."""
    if k >= 4:
        x = x.flip(-1)
    return torch.rot90(x, -(k % 4), dims=(-2, -1))


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


def block_targets_fast(ws, is_pore):
    """block_targets bit for bit (the same pixel values in the same row-major order reach the same np.mean), ~6x
    faster: each block is a 64x64 slice instead of a full-image mask."""
    a = np.maximum(np.bincount(ws.ravel()).astype(float)[ws], 1)
    m = ~is_pore[ws]
    b = np.full(16, np.nan)
    for k in range(16):
        r, c = divmod(k, 4)
        mb = m[64 * r:64 * r + 64, 64 * c:64 * c + 64]
        if mb.sum() >= 50:
            b[k] = -2 * np.log((a[64 * r:64 * r + 64, 64 * c:64 * c + 64][mb] ** -0.5).mean())
    return b


def n_eff_fast(ws, is_pore):
    """n_eff_of without regionprops: present non-pore labels, 0.5 for those with a pixel on the image border (= bbox
    touching the border), 1 otherwise. Sums of halves and ones are exact, so the value equals n_eff_of."""
    L = max(len(is_pore), int(ws.max()) + 1)
    sel = np.bincount(ws.ravel(), minlength=L) > 0
    sel[:len(is_pore)] &= ~is_pore
    sel[0] = False  # regionprops ignores label 0
    border = np.zeros(L, bool)
    border[np.concatenate([ws[0], ws[-1], ws[:, 0], ws[:, -1]])] = True
    return float(sel.sum() - 0.5 * (sel & border).sum())


_CELL = (np.arange(256)[:, None] // 16 * 16 + np.arange(256)[None, :] // 16).ravel()  # 16-px cell id, row-major


def cell_targets(ws, is_pore, min_px=32):
    """(16, 16) b of the 16-px cells (-2 log mean over the cell's non-pore pixels of a^-1/2, a = grain area), NaN where
    a cell has < min_px non-pore pixels. Row-major cells; block k = (cy // 4) * 4 + cx // 4."""
    a = np.maximum(np.bincount(ws.ravel()).astype(float)[ws.ravel()], 1)
    m = ~is_pore[ws.ravel()]
    cnt = np.bincount(_CELL[m], minlength=256)
    s = np.bincount(_CELL[m], a[m] ** -0.5, minlength=256)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where(cnt >= min_px, -2 * np.log(s / np.maximum(cnt, 1)), np.nan)
    return b.reshape(16, 16)


def cells_to_blocks(c):
    """(..., 16, 16) cell b -> (..., 16) block b by -2 log(mean over the block's 16 cells of exp(-b_cell / 2)) (the
    block value if every cell had the same number of non-pore pixels); NaN cells are skipped."""
    c = np.asarray(c, float)
    sh = c.shape[:-2]
    t = c.reshape(sh + (4, 4, 4, 4))  # (..., by, cy, bx, cx)
    t = np.moveaxis(t, -3, -2).reshape(sh + (16, 16))  # (..., block, cell-in-block)
    with np.errstate(invalid="ignore"):
        return -2 * np.log(np.nanmean(np.exp(-t / 2), -1))


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
    # render-only pore pixels: P["rpore"][j] (prep_one)
    return _cartoon(ws, S, sn, dark, resid, P["rpore"][j], float(P["im8"][j].mean()), rng, R)


def _cartoon(ws, S, sn, dark, resid, pp, mean_grey, rng, R):
    """Cartoon branch of render_noisy on explicit arrays (label map, shading surface, per-grain sn / dark tables,
    resid, render-pore mask, target mean grey): the same operations and draws in the same order as before."""
    import cv2
    c_d, e = rng.uniform(*R["c_d"]), rng.uniform(*R["e"])
    new = np.where(dark, 1 + c_d * (sn - 1), 1 + e * (sn - 1)).astype(np.float32)  # pore entries unused below
    img = S * new[ws]  # cartoon: every pixel takes its grain's (shading-normalised) grey -> no boundary lines
    img = np.where(pp, S + rng.uniform(*R["pore"]) * resid, img)
    s_line = rng.uniform(*R["line"]) if rng.random() < R["p_line"] else 0.0
    if s_line > 0:
        img = img + np.where(pp, 0, s_line * resid)
    # keep the source's mean grey: dropping the dark boundary lines alone made cartoons ~8.5 grey brighter than their
    # source, while real noisy images have the clean images' mean grey (median 145.2 vs 145.0; review 2026-10-08)
    img = img + (mean_grey - float(img.mean()))
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


# ------------------------------------------------------------------------------- zoom / mosaic label-map augmentation
GRAIN_TABS = ("sn", "dark", "pore")  # per-grain tables carried through relabel()


def source_of(P, j):
    """The cartoon renderer's inputs for prep image j (+ the source grey 'im', used only for the mean-grey fix)."""
    return dict(ws=P["ws"][j].astype(np.int64), S=shading_surface(P["beta"][j]), sn=P["sn_tab"][j],
                dark=P["dark"][j], pore=P["pore"][j], resid=P["resid"][j].astype(np.float32), rpore=P["rpore"][j],
                im=P["im8"][j].astype(np.float32))


def render_source(src, rng, R):
    """Cartoon render of a (possibly augmented) source; render_source(source_of(P, j), rng, R) == render_noisy(P, j,
    rng, R, 'cartoon')."""
    return _cartoon(src["ws"], src["S"], src["sn"], src["dark"], src["resid"], src["rpore"],
                    float(src["im"].mean(dtype=np.float64)), rng, R)


def relabel(ws, tabs):
    """Connected components (4-connectivity) of equal labels -> new labels 1..n; tabs (per-grain tables indexed by the
    old labels) re-indexed to the new labels (each new grain inherits its source label's entries)."""
    from skimage import measure
    new = measure.label(ws, background=-1, connectivity=1)
    src = np.zeros(int(new.max()) + 1, np.int64)
    src[new.ravel()] = ws.ravel()
    return new.astype(np.int64), {k: v[src] for k, v in tabs.items()}


MIN_GRAIN = 16  # het_blocks.segment never yields grains below 17 px (markers >= 6 px + their share of the ridges)


def split_merge(ws, tabs, min_px=MIN_GRAIN):
    """relabel(), then merge every NON-PORE component below min_px pixels (seam slivers of a mosaic, necks broken by
    nearest-neighbour down-scaling: pieces the het_blocks segmentation could never produce) into a neighbouring
    non-pore grain: their pixels repeatedly take the label of a 4-neighbour that is non-pore and outside such
    components (priority up, down, left, right), growing inward from the component's rim. Pore components are never
    merged (small pore pieces stay pore grains, so the pore flag of every pixel is unchanged and the target pore set
    stays the one the render-pore mask shows), and a sliver with no non-pore neighbour stays its own grain. Every
    filled pixel touches a pixel of the label it takes, so all grains stay connected (no second relabel; the removed
    labels just stop occurring). The prep label maps have no grain below 17 px, so an unaugmented map passes through
    unchanged."""
    ws, tabs = relabel(ws, tabs)
    pore = tabs["pore"]
    bad = ((np.bincount(ws.ravel()) < min_px) & ~pore)[ws]
    if not bad.any():
        return ws, tabs
    ws = ws.copy()
    for _ in range(64):
        cand = np.zeros_like(ws)
        for dst, src_ in (((slice(1, None), slice(None)), (slice(None, -1), slice(None))),  # from the pixel above
                          ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),  # below
                          ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),  # left
                          ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):  # right
            c, w = cand[dst], ws[src_]
            take = (c == 0) & ~bad[src_] & ~pore[w]
            c[take] = w[take]
        fill = bad & (cand > 0)
        if not fill.any():  # what is left has no non-pore neighbour: keep those slivers as grains
            break
        ws[fill] = cand[fill]
        bad &= ~fill
        if not bad.any():
            break
    return ws, tabs


def _reflect(i, n=256):
    i = np.mod(i, 2 * n)
    return np.where(i >= n, 2 * n - 1 - i, i)


def _bilinear(a, cy, cx):
    """Separable bilinear sampling of a (256, 256) at rows cy, columns cx, symmetric reflection outside."""
    y0, x0 = np.floor(cy), np.floor(cx)
    wy, wx = (cy - y0).astype(np.float32)[:, None], (cx - x0).astype(np.float32)[None, :]
    y0, x0 = y0.astype(np.int64), x0.astype(np.int64)
    r = a[_reflect(y0)] * (1 - wy) + a[_reflect(y0 + 1)] * wy
    return (r[:, _reflect(x0)] * (1 - wx) + r[:, _reflect(x0 + 1)] * wx).astype(np.float32)


def zoom_source(src, s, oy, ox):
    """Rescale a source by s (> 1 magnifies): output pixel (y, x) samples source coordinate (oy + (y + .5)/s - .5,
    ox + (x + .5)/s - .5), symmetric reflection beyond the edges. Nearest neighbour for the label map and the
    render-pore mask, bilinear for resid and im; the shading surface S stays the canvas one. Relabelled, slivers
    merged (split_merge)."""
    u = (np.arange(256) + 0.5) / s - 0.5
    cy, cx = oy + u, ox + u
    iy, ix = _reflect(np.floor(cy + 0.5).astype(np.int64)), _reflect(np.floor(cx + 0.5).astype(np.int64))
    ws, tabs = split_merge(src["ws"][iy[:, None], ix[None, :]], {k: src[k] for k in GRAIN_TABS})
    return dict(ws=ws, S=src["S"], resid=_bilinear(src["resid"], cy, cx), rpore=src["rpore"][iy[:, None], ix[None, :]],
                im=_bilinear(src["im"], cy, cx), **tabs)


def mosaic_source(s1, s2, mask):
    """Composite: s1 where mask, s2 elsewhere. s2's labels are offset, the map relabelled; s2's resid and grey are
    rescaled by S1/S2 per pixel and the composite keeps S1, so both regions sit on s1's illumination. Relabelled,
    slivers merged (split_merge)."""
    off = int(s1["ws"].max()) + 1
    r = s1["S"] / s2["S"]
    tabs = {k: np.concatenate([s1[k][:off], s2[k]]) for k in GRAIN_TABS}
    ws, tabs = split_merge(np.where(mask, s1["ws"], s2["ws"] + off), tabs)
    return dict(ws=ws, S=s1["S"], resid=np.where(mask, s1["resid"], s2["resid"] * r).astype(np.float32),
                rpore=np.where(mask, s1["rpore"], s2["rpore"]), im=np.where(mask, s1["im"], s2["im"] * r), **tabs)


def mosaic_mask(ra):
    """Random region mask (True = first image): 25% a straight line at a random angle through a random point of the
    central half; 75% a white-noise field smoothed by a periodic Gaussian (sigma 32-72 px, FFT) thresholded at a random
    quantile 0.3-0.7."""
    if ra.random() < 0.25:
        th = ra.uniform(0, 2 * np.pi)
        py, px = ra.uniform(64, 192, 2)
        return (_XX - px) * np.cos(th) + (_YY - py) * np.sin(th) > 0
    sig, q = ra.uniform(32, 72), ra.uniform(0.3, 0.7)
    fy, fx = np.fft.fftfreq(256), np.fft.rfftfreq(256)
    H = np.exp(-2 * (np.pi * sig) ** 2 * (fy[:, None] ** 2 + fx[None, :] ** 2))
    g = np.fft.irfft2(np.fft.rfft2(ra.standard_normal((256, 256))) * H, s=(256, 256))
    kq = int(q * (g.size - 1))
    return g > np.partition(g.ravel(), kq)[kq]


def _zoom_draw(ra, lo, hi):
    s = float(np.exp(ra.uniform(np.log(lo), np.log(hi))))
    w = 256 / s
    a, b = min(0.0, 256 - w), max(0.0, 256 - w)
    return s, float(ra.uniform(a, b)), float(ra.uniform(a, b))


def augment_source(P, j, rows, ra, aug):
    """Zoom / mosaic source for training image j (rows: the fold's training prep rows, j2 drawn among them), or None
    when this render draws no augmentation. aug: dict(zoom=(lo, hi) | None, p_zoom, mosaic[, mosaic_dlogn]). All
    draws from ra (the separate augmentation generator), in a fixed order; mosaic_dlogn (None = any partner) only
    narrows the candidate list, the draws are the same."""
    z = aug.get("zoom")
    do_mos = aug.get("mosaic", 0) > 0 and ra.random() < aug["mosaic"]
    z1 = z is not None and ra.random() < aug["p_zoom"]
    if not (do_mos or z1):
        return None
    src = source_of(P, j)
    if z1:
        src = zoom_source(src, *_zoom_draw(ra, *z))
    if do_mos:
        cand = rows[rows != j]
        dl = aug.get("mosaic_dlogn")
        if dl is not None:  # partner of similar grain count: |log N_eff(j2) - log N_eff(j)| <= dl (else the nearest)
            d = np.abs(np.log(P["n_eff"][cand]) - np.log(P["n_eff"][j]))
            cand = cand[d <= dl] if (d <= dl).any() else cand[[int(np.argmin(d))]]
        j2 = int(cand[ra.integers(len(cand))])
        s2 = source_of(P, j2)
        if z is not None and ra.random() < aug["p_zoom"]:
            s2 = zoom_source(s2, *_zoom_draw(ra, *z))
        src = mosaic_source(src, s2, mosaic_mask(ra))
    return src


def targets_of(src, xnames=(), cells=False):
    """Targets of a (possibly augmented) source with the definitions of the prep: b (block_targets), het4 = sd of the
    valid b, n_eff (n_eff_of), extras (extra_target_values), cells (cell_targets)."""
    ws, pore = src["ws"], src["pore"]
    b = block_targets_fast(ws, pore)
    t = dict(b=b, het4=float(np.std(b[~np.isnan(b)])), n_eff=n_eff_fast(ws, pore))
    if xnames:
        x = []
        for nm in xnames:
            if nm == "fd":
                x.append(src["dark"][ws].mean())
            elif nm == "pore":
                x.append(pore[ws].mean())
            elif nm == "asp":
                with np.errstate(invalid="ignore", divide="ignore"):
                    v = grain_log_aspect(ws, pore)
                if not np.isfinite(v):  # no interior non-pore grain (strong zoom-in): use every non-pore grain
                    v = grain_log_aspect(np.pad(ws, 1), pore)
                x.append(v)
            else:
                raise ValueError(nm)
        t["x"] = np.array(x, np.float32)
    if cells:
        t["cells"] = cell_targets(ws, pore)
    return t


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
    each item as a 6th tensor (D4-invariant, so not permuted).
    aug: None (as before) or dict(zoom=(lo, hi) | None, p_zoom, mosaic) -- cartoon-path renders may be zoomed /
    composited (augment_source, draws from default_rng([seed, fold, epoch, i, AUG_TAG])) and then carry targets
    recomputed on the transformed label map (targets_of); xnames: the extra-target names (needed to recompute xt).
    cells: append the (16, 16) cell targets (cell_targets, D4-transformed with the image, NaN -> 0) and their validity
    mask as two more tensors."""

    def __init__(self, prep_fp, rows, renders, seed, fold, preset="noisy", xt=None, aug=None, cells=False,
                 xnames=None):
        self.fp, self.rows, self.R, self.seed, self.fold = str(prep_fp), np.asarray(rows), renders, seed, fold
        self.n_items = len(self.rows) * renders
        self.preset = preset
        self.xt = None if xt is None else np.asarray(xt, np.float32)
        self.aug, self.cells, self.xnames = aug, cells, list(xnames or [])
        if aug is not None and self.xt is not None:
            assert len(self.xnames) == self.xt.shape[1], "aug with extra targets needs xnames"
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
        tg = None
        if self.aug is None:
            img, _ = render_noisy(P, j, rng, R, M=M)
        else:  # same path draw as render_noisy; augmentation draws from a separate generator
            path = "m" if rng.random() < R["p_m"] else "cartoon"
            src = None
            if path == "cartoon":
                src = augment_source(P, j, self.rows, np.random.default_rng([self.seed, self.fold, ep, i, AUG_TAG]),
                                     self.aug)
            if src is None:  # identical to render_noisy(P, j, rng, R, M=M) with the same generator
                img, _ = render_noisy(P, j, rng, R, force=path, M=M)
            else:
                img, _ = render_source(src, rng, R)
                tg = targets_of(src, self.xnames if self.xt is not None else (), self.cells)
        k = int(rng.integers(8))
        img = np.ascontiguousarray(d4_np(img, k))
        if tg is None:
            b = P["b"][j][GRID_PERM[k]]
            valid = ~np.isnan(b)
            item = (torch.from_numpy(img)[None], torch.from_numpy(np.nan_to_num(b).astype(np.float32)),
                    torch.from_numpy(valid.astype(np.float32)), torch.tensor(float(P["het4"][j])),
                    torch.tensor(float(np.log(P["n_eff"][j]))))
            if self.xt is not None:
                item = item + (torch.from_numpy(self.xt[j].copy()),)
        else:
            b = tg["b"][GRID_PERM[k]]
            valid = ~np.isnan(b)
            item = (torch.from_numpy(img)[None], torch.from_numpy(np.nan_to_num(b).astype(np.float32)),
                    torch.from_numpy(valid.astype(np.float32)), torch.tensor(tg["het4"]),
                    torch.tensor(float(np.log(tg["n_eff"]))))
            if self.xt is not None:
                item = item + (torch.from_numpy(tg["x"]),)
        if self.cells:
            c = tg["cells"] if tg is not None else cell_targets(P["ws"][j].astype(np.int64), P["pore"][j])
            c = np.ascontiguousarray(d4_np(c, k))
            item = item + (torch.from_numpy(np.nan_to_num(c).astype(np.float32)),
                           torch.from_numpy((~np.isnan(c)).astype(np.float32)))
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


class FPNNet(nn.Module):
    """--head fpn. timm features_only backbone; the stride-8/16/32 maps -> 1x1 laterals (ch) merged at stride 16
    (nearest up-sampling of s32, 2x2 average pooling of s8; all D4-equivariant) -> 3x3 conv + BN + ReLU = the merged
    map (16x16 at 256 input). forward -> (b (B, 16) block head on adaptive_avg_pool2d(merged, 4), logN (B,) from the
    global mean of the stride-32 map, [x (B, n_extra)], cells (B, 16, 16) cell head on the merged map); has_cells tells
    predict() and the training loop that the last output is the cell map."""
    has_cells = True

    def __init__(self, arch, pretrained=True, n_extra=0, ch=128):
        super().__init__()
        self.body = create_timm(arch, pretrained=pretrained, features_only=True, in_chans=1)
        red, chs = self.body.feature_info.reduction(), self.body.feature_info.channels()
        self.idx = [red.index(r) for r in (8, 16, 32)]
        self.lat = nn.ModuleList(nn.Conv2d(chs[i], ch, 1) for i in self.idx)
        self.smooth = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        nf = chs[self.idx[2]]
        self.block = nn.Conv2d(ch, 1, 1)
        self.cellh = nn.Conv2d(ch, 1, 1)
        self.glob = nn.Linear(nf, 1)
        heads = [self.block, self.cellh, self.glob]
        self.n_extra = n_extra
        if n_extra:
            self.extra = nn.Linear(nf, n_extra)
            heads.append(self.extra)
        for m in heads:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        fs = self.body(x)
        f8, f16, f32 = (fs[i] for i in self.idx)
        m = (self.lat[1](f16) + F.interpolate(self.lat[2](f32), size=f16.shape[-2:], mode="nearest")
             + F.avg_pool2d(self.lat[0](f8), 2))
        m = self.smooth(m)
        with torch.autocast(x.device.type, enabled=False):  # pooling + heads in fp32
            m = m.float()
            assert m.shape[-1] % 4 == 0 and m.shape[-2] % 4 == 0, f"merged map {tuple(m.shape[-2:])} not divisible by 4"
            b = self.block(F.adaptive_avg_pool2d(m, 4)).flatten(1)
            c = self.cellh(m)[:, 0]
            g = f32.float().mean((-2, -1))
            n = self.glob(g).squeeze(-1)
            if self.n_extra:
                return b, n, self.extra(g), c
        return b, n, c


def make_model(a, nx):
    if a.head == "fpn":
        return FPNNet(a.arch, pretrained=not a.scratch, n_extra=nx, ch=a.fpn_ch)
    return HetNet(a.arch, pretrained=not a.scratch, n_extra=nx)


def masked_sd(v, m):
    n = m.sum(1).clamp(min=1)
    mu = (v * m).sum(1) / n
    return ((((v - mu[:, None]) ** 2) * m).sum(1) / n + 1e-8).sqrt()


# ------------------------------------------------------------------------------------------------------- prediction
@torch.no_grad()
def predict(model, X8, mu, sd, dev, amp_dtype, tta=8, bs=32):
    """uint8 (N, 256, 256) -> (b_z (N, 16), n_z (N,)): mean over the first `tta` D4 views, each view's 4x4 output
    un-permuted to the original grid (GRID_INV) before averaging. Values in the training fold's z-units.
    A model with extra outputs (HetNet n_extra > 0) -> (b_z, n_z, x_z (N, n_extra)), x averaged over the views.
    A model with has_cells (FPNNet) -> one more element at the end: cells_z (N, 16, 16), each view's cell map
    un-transformed by the inverse D4 op (d4_inv_t) before averaging."""
    model.eval()
    has_cells = getattr(model, "has_cells", False)
    outb, outn, outx, outc = [], [], [], []
    inv = [torch.as_tensor(GRID_INV[k], device=dev) for k in range(8)]
    for i in range(0, len(X8), bs):
        x = torch.from_numpy(np.ascontiguousarray(X8[i:i + bs])).to(dev).float().div_(255.0)[:, None]
        x = (x - mu) / sd
        sb, sn, sx, sc = 0, 0, 0, 0
        for k in range(tta):
            with torch.autocast(dev.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                o = model(d4_t(x, k).contiguous(memory_format=torch.channels_last))
            if has_cells:
                sc = sc + d4_inv_t(o[-1].float(), k)
                o = o[:-1]
            sb = sb + o[0].float()[:, inv[k]]
            sn = sn + o[1].float()
            if len(o) > 2:
                sx = sx + o[2].float()
        outb.append((sb / tta).cpu())
        outn.append((sn / tta).cpu())
        if len(o) > 2:
            outx.append((sx / tta).cpu())
        if has_cells:
            outc.append((sc / tta).cpu())
    res = (torch.cat(outb).numpy().astype(np.float64), torch.cat(outn).numpy().astype(np.float64))
    if outx:
        res = res + (torch.cat(outx).numpy().astype(np.float64),)
    if has_cells:
        res = res + (torch.cat(outc).numpy().astype(np.float64),)
    return res


def _pcorr(x, y, z):
    """Partial Pearson corr of x and y given z (both residualised on [1, z])."""
    Z = np.column_stack([np.ones(len(z)), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def _resid(v, z):
    Z = np.column_stack([np.ones(len(z)), z])
    return v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]


def _psdratio(x, y, z):
    """sd(y | z) / sd(x | z) (residuals on [1, z]): scale only; > 1 = x has less spread than y (shrinkage)."""
    return float(np.std(_resid(y, z)) / max(np.std(_resid(x, z)), 1e-30))


def _pslope(x, y, z):
    """OLS slope of y on x after residualising both on [1, z] = partial corr x sd(y | z) / sd(x | z). It exceeds 1 only
    when x is shrunk AND ranks y well; a poor ranking pulls it toward 0 (attenuation), so read it with _psdratio."""
    Z = np.column_stack([np.ones(len(z)), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float((rx @ ry) / max(rx @ rx, 1e-30))


def _within(v, ok):
    return np.where(ok, v - np.array([v[r][ok[r]].mean() for r in range(len(v))])[:, None], np.nan)


def stage1_metrics(bh, nh, b, het4, logn, ch=None, ct=None):
    """bh (M, 16) raw b-hat, nh (M,) logN-hat; b (M, 16) targets with NaN; het4, logn (M,).
    ch (M, 16, 16): optional raw cell b-hat (FPNNet) -> pcorr_het_cell etc. (het-hat = sd of the cells aggregated to
    blocks, cells_to_blocks); ct (M, 16, 16): cell targets (NaN = invalid) -> corr_cell(_within)."""
    v = ~np.isnan(b)
    sdh = np.array([np.std(bh[r][v[r]]) for r in range(len(bh))])
    within_h = bh - np.array([bh[r][v[r]].mean() for r in range(len(bh))])[:, None]
    within_t = b - np.array([b[r][v[r]].mean() for r in range(len(b))])[:, None]
    out = dict(n=int(len(bh)), corr_het=float(np.corrcoef(sdh, het4)[0, 1]),
               pcorr_het_given_logN=_pcorr(sdh, het4, logn),
               corr_block=float(np.corrcoef(bh[v], b[v])[0, 1]),
               corr_block_within=float(np.corrcoef(within_h[v], within_t[v])[0, 1]),
               corr_logN=float(np.corrcoef(nh, logn)[0, 1]),
               rmse_logN=float(np.sqrt(np.mean((nh - logn) ** 2))),
               het_hat_mean=float(sdh.mean()), het4_mean=float(het4.mean()))
    out["slope_het4_on_hat_given_logN"] = _pslope(sdh, het4, logn)  # additive keys (2026-10-09 evening)
    out["sdratio_het4_over_hat_given_logN"] = _psdratio(sdh, het4, logn)
    if ch is not None:
        bc = cells_to_blocks(ch)
        sdc = np.array([np.std(bc[r][v[r]]) for r in range(len(bc))])
        wc = bc - np.array([bc[r][v[r]].mean() for r in range(len(bc))])[:, None]
        out.update(pcorr_het_cell=_pcorr(sdc, het4, logn), corr_het_cell=float(np.corrcoef(sdc, het4)[0, 1]),
                   slope_het4_on_cellhat_given_logN=_pslope(sdc, het4, logn),
                   sdratio_het4_over_cellhat_given_logN=_psdratio(sdc, het4, logn),
                   corr_block_within_cell=float(np.corrcoef(wc[v], within_t[v])[0, 1]),
                   het_hat_cell_mean=float(sdc.mean()))
        if ct is not None:
            ok = ~np.isnan(ct.reshape(len(ct), -1))
            hc, tc = ch.reshape(len(ch), -1), ct.reshape(len(ct), -1)
            out.update(corr_cell=float(np.corrcoef(hc[ok], tc[ok])[0, 1]),
                       corr_cell_within=float(np.corrcoef(_within(hc, ok)[ok], _within(tc, ok)[ok])[0, 1]),
                       cell_sd_hat=float(np.nanstd(_within(hc, ok))), cell_sd_true=float(np.nanstd(_within(tc, ok))))
    return out


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


# ----------------------------------------------------------- fixed-seed preset evaluation (--eval-presets, --eval-only)
def eval_render_set(P, rows, preset, K):
    """K held-out renders of every prep row in rows, in preset's style, with the stage-1 seeds
    default_rng([EVAL_SEED, k, j]) -> (uint8 (len(rows) * K, 256, 256), meta (n, 3) int = (row, k, path)). The same
    loop and seeds as train_fold's stage 1, so for the training preset these are exactly its stage-1 renders; the seeds
    depend on neither the run (--seed), the fold model nor the training preset, so every run sees identical renders."""
    R, M = PRESETS[preset]
    X, meta = [], []
    for j in rows:
        for k in range(K):
            img, info = render_noisy(P, int(j), np.random.default_rng([EVAL_SEED, k, int(j)]), R, M=M)
            X.append(img)
            meta.append((int(j), k, info["path"]))
    return np.stack(X), np.array(meta)


def g1_train_ids():
    """The real line-free noisy TRAIN images (G1: raw ic_noise >= NOISE_MAX and ic_ridge_snr < G1_RIDGE_MAX in
    features_v3.parquet, label-free image statistics) in train.csv order; used for prediction only."""
    v3 = pd.read_parquet(DATA_DIR / "features_v3.parquet", columns=["ID", "ic_noise", "ic_ridge_snr"]).set_index("ID")
    ids = pd.read_csv(DATA_DIR / "train.csv", usecols=["ID"]).ID  # IDs only; the hardness column is not read
    return [i for i in ids if v3.loc[i, "ic_noise"] >= NOISE_MAX and v3.loc[i, "ic_ridge_snr"] < G1_RIDGE_MAX]


def eval_presets_fold(model, consts, P, va_rows, presets, K, dev, amp_dtype, tta=8, bs=32, g1_ids=()):
    """One fold model: stage-1 metrics on eval_render_set(P, va_rows, preset, K) for every preset (all K renders,
    cartoon-path renders only, K-averaged; cell metrics for an FPNNet), plus predictions on the real G1 train images
    g1_ids (prediction only; logN-hat mean / sd vs the prep's). consts: the fold's standardisation constants (ckpt
    'consts'). -> (json-able dict, dict of arrays for an npz). P = pcorr_het_given_logN, W = corr_block_within (all K
    renders), md5 = hash of the rendered uint8 stack (equal md5 = identical renders)."""
    import hashlib
    mu_t, sd_t = torch.tensor(consts["mu_px"], device=dev), torch.tensor(consts["sd_px"], device=dev)
    mu_b, s_b, mu_n, s_n = (consts[k] for k in ("mu_b", "s_b", "mu_n", "s_n"))
    b, het4, logn = P["b"], P["het4"], np.log(P["n_eff"])
    va_rows = np.asarray(va_rows, int)
    use_cells = getattr(model, "has_cells", False)
    ct_of = {int(j): cell_targets(P["ws"][j].astype(np.int64), P["pore"][j]) for j in va_rows} if use_cells else {}
    sel = lambda v, m: None if v is None else v[m]  # noqa: E731
    met, arr = {}, {}
    for pr in presets:
        X, meta = eval_render_set(P, va_rows, pr, K)
        pv = predict(model, X, mu_t, sd_t, dev, amp_dtype, tta=tta, bs=bs)
        bh, nh = mu_b + s_b * pv[0], mu_n + s_n * pv[1]
        rows = meta[:, 0]
        ch = mu_b + s_b * pv[-1] if use_cells else None
        ct = np.stack([ct_of[int(j)] for j in rows]) if use_cells else None
        m_all = stage1_metrics(bh, nh, b[rows], het4[rows], logn[rows], ch, ct)
        cart = meta[:, 2] == 0
        m_cart = (stage1_metrics(bh[cart], nh[cart], b[rows[cart]], het4[rows[cart]], logn[rows[cart]], sel(ch, cart),
                                 sel(ct, cart)) if cart.sum() >= 4 else None)
        bk = np.stack([bh[rows == j].mean(0) for j in va_rows])
        nk = np.array([nh[rows == j].mean() for j in va_rows])
        chk = None if ch is None else np.stack([ch[rows == j].mean(0) for j in va_rows])
        m_k = stage1_metrics(bk, nk, b[va_rows], het4[va_rows], logn[va_rows], chk,
                             None if ch is None else np.stack([ct_of[int(j)] for j in va_rows]))
        met[pr] = dict(P=m_all["pcorr_het_given_logN"], W=m_all["corr_block_within"], n=int(len(rows)), K=int(K),
                       n_cartoon=int(cart.sum()), md5=hashlib.md5(np.ascontiguousarray(X).tobytes()).hexdigest(),
                       renders=m_all, renders_cartoon=m_cart, renders_Kavg=m_k)
        arr.update({f"{pr}_meta": meta, f"{pr}_b": bh, f"{pr}_logn": nh})
        if use_cells:
            arr[f"{pr}_cells"] = ch.astype(np.float32)
    if len(g1_ids):
        Xg = np.stack([read_u8(i) for i in g1_ids])
        pg = predict(model, Xg, mu_t, sd_t, dev, amp_dtype, tta=tta, bs=bs)
        bg, ng = mu_b + s_b * pg[0], mu_n + s_n * pg[1]
        met["real_g1_train"] = dict(n=int(len(g1_ids)), logN_mean=float(ng.mean()), logN_sd=float(ng.std()),
                                    het_mean=float(bg.std(1).mean()), het_sd=float(bg.std(1).std()),
                                    prep_logN_mean=float(logn.mean()), prep_logN_sd=float(logn.std()))
        arr.update(g1_ids=np.array(g1_ids), g1_b=bg, g1_logn=ng)
    return met, arr


def summarize_eval(met, arr, P, presets):
    """{fold: eval_presets_fold output} -> dict(folds, per_fold, mean_folds {preset: P, W, P_Kavg, W_Kavg (means over
    the folds)}, pooled {preset: stage1_metrics over all folds' renders (> 1 fold)}, real_g1_train (blocks averaged over
    the folds, then het = sd over the 16 blocks, as aggregate() does; logN-hat mean / sd vs the prep's))."""
    folds = sorted(met)
    out = dict(folds=folds, per_fold={str(f): met[f] for f in folds}, mean_folds={}, pooled={})
    logn = np.log(P["n_eff"])
    for pr in presets:
        if not all(pr in met[f] for f in folds):
            continue
        mf = [met[f][pr] for f in folds]
        out["mean_folds"][pr] = dict(
            P=float(np.mean([m["P"] for m in mf])), W=float(np.mean([m["W"] for m in mf])),
            P_Kavg=float(np.mean([m["renders_Kavg"]["pcorr_het_given_logN"] for m in mf])),
            W_Kavg=float(np.mean([m["renders_Kavg"]["corr_block_within"] for m in mf])))
        if len(folds) > 1 and all(f"{pr}_b" in arr[f] for f in folds):
            rows = np.concatenate([arr[f][f"{pr}_meta"][:, 0] for f in folds]).astype(int)
            bh = np.concatenate([arr[f][f"{pr}_b"] for f in folds])
            nh = np.concatenate([arr[f][f"{pr}_logn"] for f in folds])
            ch = ct = None
            if all(f"{pr}_cells" in arr[f] for f in folds):
                ch = np.concatenate([arr[f][f"{pr}_cells"] for f in folds]).astype(np.float64)
                cto = {int(j): cell_targets(P["ws"][j].astype(np.int64), P["pore"][j]) for j in np.unique(rows)}
                ct = np.stack([cto[int(j)] for j in rows])
            out["pooled"][pr] = stage1_metrics(bh, nh, P["b"][rows], P["het4"][rows], logn[rows], ch, ct)
    if folds and all("g1_b" in arr[f] for f in folds):
        ids0 = list(arr[folds[0]]["g1_ids"])
        assert all(list(arr[f]["g1_ids"]) == ids0 for f in folds), "folds predicted different G1 image lists"
        bg = np.mean([arr[f]["g1_b"] for f in folds], 0)
        ng = np.mean([arr[f]["g1_logn"] for f in folds], 0)
        out["real_g1_train"] = dict(n=len(ids0), folds_averaged=folds, logN_mean=float(ng.mean()),
                                    logN_sd=float(ng.std()), het_mean=float(bg.std(1).mean()),
                                    het_sd=float(bg.std(1).std()), prep_logN_mean=float(logn.mean()),
                                    prep_logN_sd=float(logn.std()),
                                    d_logN_mean=float(ng.mean() - logn.mean()), d_logN_sd=float(ng.std() - logn.std()))
    return out


def model_from_ckpt(ck):
    """Rebuild a fold model from a fold{f}.pt dict (arch, opts head / fpn_ch, extra targets) without downloading
    pretrained weights, and load its state_dict (strict)."""
    opts = ck.get("opts", {}) or {}
    nx = len(ck.get("extra_targets", []) or [])
    if opts.get("head", "grid4") == "fpn":
        m = FPNNet(ck["arch"], pretrained=False, n_extra=nx, ch=opts.get("fpn_ch", 128))
    else:
        m = HetNet(ck["arch"], pretrained=False, n_extra=nx)
    m.load_state_dict(ck["state_dict"])
    return m


def _eval_line(tag, m, presets):
    s = " | ".join(f"{pr}: P {m[pr]['P']:.4f} W {m[pr]['W']:.4f} (K-avg P "
                   f"{m[pr]['renders_Kavg']['pcorr_het_given_logN']:.4f}, n {m[pr]['n']}, md5 {m[pr]['md5'][:8]})"
                   for pr in presets if pr in m)
    g = m.get("real_g1_train")
    if g:
        s += (f" | real G1 train n {g['n']}: logN-hat mean {g['logN_mean']:.3f} sd {g['logN_sd']:.3f} (prep "
              f"{g['prep_logN_mean']:.3f} / {g['prep_logN_sd']:.3f})")
    return f"{tag} eval presets: {s}"


def run_eval_only(a):
    """--eval-only RUN: score RUN's fold checkpoints on the --eval-presets renders; writes score_eval.json and
    score_eval_fold{f}.npz in data/het_cnn/RUN (score.json, fold json / npz and parquets are never written)."""
    out = OUT_ROOT / a.eval_only
    folds = a.folds if a.folds is not None else [f for f in range(5) if (out / f"fold{f}.pt").exists()]
    miss = [f for f in folds if not (out / f"fold{f}.pt").exists()]
    if not folds or miss:
        raise SystemExit(f"--eval-only {a.eval_only}: no fold checkpoints {miss or ''} in {out}")
    use_cuda = a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available())
    dev = torch.device("cuda" if use_cuda else "cpu")
    amp_dtype = None
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.set_num_threads(a.threads)
    import cv2
    cv2.setNumThreads(1)
    P = load_prep(a.prep)
    fo = pd.read_csv(DATA_DIR / "folds.csv").set_index("ID")
    folds_of = fo.loc[P["ids"], "fold"].values.astype(int)
    K = 2 if a.smoke else a.k_val
    g1 = g1_train_ids()
    g1 = g1[:8] if a.smoke else g1
    print(f"het_cnn --eval-only {a.eval_only}: folds {folds}, presets {a.eval_presets}, K {K}, device {dev} amp "
          f"{amp_dtype}, tta {a.tta}; {len(g1)} real G1 train images (prediction only)", flush=True)
    met, arr, info = {}, {}, {}
    t0 = time.time()
    for f in folds:
        ck = torch.load(out / f"fold{f}.pt", map_location="cpu", weights_only=False)
        va_rows = np.where(folds_of == f)[0]
        va_rows = va_rows[:8] if a.smoke else va_rows
        if (out / f"fold{f}.npz").exists():  # the run's own held-out rows (a --smoke run held out the first 8)
            with np.load(out / f"fold{f}.npz") as z:
                vr = np.unique(z["val_rows"][:, 0]).astype(int)
            if not np.array_equal(vr, np.where(folds_of == f)[0][:len(vr)]):
                raise SystemExit(f"{out}/fold{f}.npz: held-out rows are not fold {f} of data/folds.csv")
            va_rows = vr
        model = model_from_ckpt(ck).to(dev).to(memory_format=torch.channels_last)
        t = time.time()
        m, ar = eval_presets_fold(model, ck["consts"], P, va_rows, a.eval_presets, K, dev, amp_dtype, a.tta,
                                  a.pred_batch, g1)
        m.update(fold=f, train_preset=ck.get("render_preset", "noisy"), n_heldout=int(len(va_rows)),
                 t_s=round(time.time() - t, 1))
        met[f], arr[f] = m, ar
        info[f] = dict(arch=ck["arch"], train_preset=ck.get("render_preset", "noisy"), opts=ck.get("opts", {}),
                       extra_targets=ck.get("extra_targets", []))
        np.savez(out / f"score_eval_fold{f}.npz", **ar)
        print(_eval_line(f"fold {f}", m, a.eval_presets), flush=True)
        del model
    S = summarize_eval(met, arr, P, a.eval_presets)
    cmd = "python -m src.het_cnn " + " ".join(sys.argv[1:])
    S = dict(run=a.eval_only, mode="eval-only", presets=a.eval_presets, K=K,
             seed_form=f"default_rng([{EVAL_SEED}, k, j])", tta=a.tta, device=str(dev), amp=str(amp_dtype), ckpt=info,
             ranges={pr: dict(render=PRESETS[pr][0], m_ranges=PRESETS[pr][1]) for pr in a.eval_presets},
             command=cmd, t_s=round(time.time() - t0, 1), **S)
    (out / "score_eval.json").write_text(json.dumps(S, indent=1, default=str))
    for pr, v in S["mean_folds"].items():
        pl = S["pooled"].get(pr)
        print(f"{a.eval_only} {pr}: mean over folds {folds} P {v['P']:.4f} W {v['W']:.4f}"
              + (f" | pooled P {pl['pcorr_het_given_logN']:.4f} W {pl['corr_block_within']:.4f}" if pl else ""),
              flush=True)
    print(f"wrote {out / 'score_eval.json'} ({time.time() - t0:.0f}s)", flush=True)


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
    use_cells = a.head == "fpn"
    if a.lambda_within > 0:  # sd of the within-image block deviations of the training fold's prep targets
        consts["s_w"] = float(np.nanstd(bt - np.nanmean(bt, 1, keepdims=True)))
        s_w = consts["s_w"]
    print(f"fold {f}: train {len(tr_rows)} / held-out {len(va_rows)} clean images; consts "
          + " ".join(f"{k} {v:.4f}" for k, v in consts.items()), flush=True)

    model = make_model(a, nx).to(dev).to(memory_format=torch.channels_last)
    decay, no_decay = [], []
    for nme, p in model.named_parameters():
        (decay if p.ndim > 1 else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr)
    from . import het_cnn as M  # pickle as src.het_cnn.* (not __main__.*) for spawn workers (Windows)
    aug = None
    if a.zoom is not None or a.mosaic > 0:
        aug = dict(zoom=tuple(a.zoom) if a.zoom is not None else None, p_zoom=a.p_zoom, mosaic=a.mosaic,
                   mosaic_dlogn=getattr(a, "mosaic_dlogn", None))
    ds = M.RenderSet(a.prep, tr_rows, a.renders, a.seed, f, preset=a.render_preset, xt=xt, aug=aug, cells=use_cells,
                     xnames=xnames)
    ci = 5 + (nx > 0)  # batch index of the cell targets (FPNNet)
    new_l = (["within"] if a.lambda_within > 0 else []) + (["cell"] if use_cells and a.lambda_cell > 0 else [])
    ep_log, h_all = [], []
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
        tl = np.zeros(4 + nx + len(new_l))
        te = time.time()
        t_wait, tw = 0.0, time.time()
        h_ep = []  # het4 of this epoch's training renders (targets of augmented renders differ from the prep's)
        for batch in dl:
            t_wait += time.time() - tw
            x, bb, vv, hh, nn_ = batch[:5]
            h_ep.append(hh.numpy().astype(np.float64))
            x = x.to(dev, non_blocking=True).float().div_(255.0)
            x = ((x - mu_t) / sd_t).contiguous(memory_format=torch.channels_last)
            bb, vv, hh, nn_ = (t.to(dev, non_blocking=True) for t in (bb, vv, hh, nn_))
            with torch.autocast(dev.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                o = model(x)
            if use_cells:
                oc = o[-1].float()
                o = o[:-1]
            bz, nz = o[0].float(), o[1].float()
            l_b = (((bz - (bb - mu_b) / s_b) ** 2) * vv).sum() / vv.sum()
            l_h = ((s_b * masked_sd(bz, vv) - hh) ** 2).mean() / s_het ** 2
            l_n = ((nz - (nn_ - mu_n) / s_n) ** 2).mean()
            loss = l_b + a.lambda_het * l_h + a.lambda_n * l_n
            if nx:  # lambda_x * standardised squared error per extra target
                xx_ = batch[5].to(dev, non_blocking=True)
                l_x = ((o[2].float() - (xx_ - mu_xt) / s_xt) ** 2).mean(0)
                loss = loss + a.lambda_x * l_x.sum()
            ln = []
            if a.lambda_within > 0:  # within-image deviations, valid blocks only
                nv = vv.sum(1, keepdim=True).clamp(min=1)
                d_h = s_b * (bz - (bz * vv).sum(1, keepdim=True) / nv)
                d_t = bb - (bb * vv).sum(1, keepdim=True) / nv
                l_w = ((((d_h - d_t) / s_w) ** 2 * vv).sum(1) / nv[:, 0]).mean()
                loss = loss + a.lambda_within * l_w
                ln.append(l_w)
            if use_cells and a.lambda_cell > 0:
                cb, cv = batch[ci].to(dev, non_blocking=True), batch[ci + 1].to(dev, non_blocking=True)
                assert oc.shape == cb.shape, f"cell map {tuple(oc.shape)} vs targets {tuple(cb.shape)}"
                l_c = (((oc - (cb - mu_b) / s_b) ** 2) * cv).sum() / cv.sum().clamp(min=1)
                loss = loss + a.lambda_cell * l_c
                ln.append(l_c)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if a.clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            scaler.step(opt)
            scaler.update()
            sch.step()
            tl += ([loss.item(), l_b.item(), l_h.item(), l_n.item()] + ([float(v) for v in l_x.tolist()] if nx else [])
                   + [v.item() for v in ln])
            tw = time.time()
        tl /= spe
        xl = "".join(f" {nm} {tl[4 + c]:.4f}" for c, nm in enumerate(xnames))
        xl += "".join(f" {nm} {tl[4 + nx + c]:.4f}" for c, nm in enumerate(new_l))
        t_ep = time.time() - te
        h_ep = np.concatenate(h_ep) if h_ep else np.zeros(0)
        h_all.append(h_ep)
        ep_log.append(dict(ep=ep + 1, wall_s=round(t_ep, 2), data_wait_s=round(t_wait, 2),
                           het4_mean=round(float(h_ep.mean()), 4) if len(h_ep) else None,
                           het4_max=round(float(h_ep.max()), 4) if len(h_ep) else None))
        print(f"fold {f} ep {ep + 1:3d}/{a.epochs} loss {tl[0]:.4f} (block {tl[1]:.4f} het {tl[2]:.4f} "
              f"logN {tl[3]:.4f}{xl}) lr {sch.get_last_lr()[0]:.2e} | {len(ds)} renders {time.time() - te:.0f}s "
              f"| {time.time() - t0:.0f}s"
              + (f" | epoch wall {t_ep:.1f}s, data wait {t_wait:.1f}s, {len(ds) / t_ep:.1f} renders/s" if a.screen
                 else ""), flush=True)
    t_train = time.time() - t0
    h_all = np.concatenate(h_all) if h_all else np.zeros(0)
    h_tr = P["het4"][tr_rows]
    train_het4 = dict(n=int(len(h_all)), mean=float(h_all.mean()), sd=float(h_all.std()),
                      p95=float(np.percentile(h_all, 95)), max=float(h_all.max()),
                      frac_above_prep_max=float((h_all > h_tr.max() + 1e-6).mean()),  # f32 items
                      prep_mean=float(h_tr.mean()), prep_sd=float(h_tr.std()), prep_max=float(h_tr.max())) \
        if len(h_all) else None
    if train_het4 is not None:
        print(f"fold {f} training-render het4: mean {train_het4['mean']:.3f} sd {train_het4['sd']:.3f} max "
              f"{train_het4['max']:.3f} ({100 * train_het4['frac_above_prep_max']:.1f}% above the prep max) vs prep "
              f"mean {train_het4['prep_mean']:.3f} sd {train_het4['prep_sd']:.3f} max {train_het4['prep_max']:.3f}",
              flush=True)
    ckpt = {"state_dict": model.state_dict(), "consts": consts, "arch": a.arch}
    if nx or a.render_preset != "noisy":
        ckpt.update(extra_targets=xnames, render_preset=a.render_preset)
    opts = new_opts(a)
    if opts:
        ckpt.update(opts=opts)
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
    ch = ct = None
    if use_cells:  # raw cell b-hat and the cell targets of the held-out originals
        ch = mu_b + s_b * pv[-1]
        ct_of = {int(j): cell_targets(P["ws"][j].astype(np.int64), P["pore"][j]) for j in va_rows}
        ct = np.stack([ct_of[int(j)] for j in rows])
    sel = lambda v, m: None if v is None else v[m]  # noqa: E731
    m_all = stage1_metrics(bh, nh, b[rows], P["het4"][rows], logn[rows], ch, ct)
    cart = meta[:, 2] == 0
    m_cart = stage1_metrics(bh[cart], nh[cart], b[rows[cart]], P["het4"][rows[cart]], logn[rows[cart]],
                            sel(ch, cart), sel(ct, cart))
    # K-averaged b-hat per image (the renders' mean prediction)
    bk = np.stack([bh[rows == j].mean(0) for j in va_rows])
    nk = np.array([nh[rows == j].mean() for j in va_rows])
    chk = None if ch is None else np.stack([ch[rows == j].mean(0) for j in va_rows])
    m_kavg = stage1_metrics(bk, nk, b[va_rows], P["het4"][va_rows], logn[va_rows], chk,
                            None if ch is None else np.stack([ct_of[int(j)] for j in va_rows]))
    m_split = None
    if a.screen:  # where does localisation fail: held-out renders of images above / below the fold's median het4
        med = float(np.median(P["het4"][va_rows]))
        m_split = {"het4_median": med}
        for nm_, msk in (("hi", P["het4"][rows] > med), ("lo", P["het4"][rows] <= med)):
            m_split[nm_] = (stage1_metrics(bh[msk], nh[msk], b[rows[msk]], P["het4"][rows[msk]], logn[rows[msk]],
                                           sel(ch, msk), sel(ct, msk)) if msk.sum() >= 4 else None)
    if nx:
        xh = mu_x + s_x * pv[2]
        mx_all = stage1_extra(xh, xt[rows], xnames)
        mx_cart = stage1_extra(xh[cart], xt[rows[cart]], xnames)
        mx_kavg = stage1_extra(np.stack([xh[rows == j].mean(0) for j in va_rows]), xt[va_rows], xnames)
    t_val = time.time() - t1

    # (b) real images, raw, prediction only (--screen: only this fold's held-out real clean images)
    t2 = time.time()
    if a.screen:
        pred_ids = [ids[j] for j in va_rows]
    Xr = np.stack([read_u8(i) for i in pred_ids])
    pr_ = predict(model, Xr, mu_t, sd_t, dev, amp_dtype, tta=a.tta, bs=a.pred_batch)
    bz, nz = pr_[0], pr_[1]
    bR, nR = mu_b + s_b * bz, mu_n + s_n * nz
    xR = mu_x + s_x * pr_[2] if nx else None
    cR = mu_b + s_b * pr_[-1] if use_cells else None
    t_pred = time.time() - t2
    # diagnostics on the REAL held-out clean images (their own clean rendering, no re-render)
    pos = {i: n for n, i in enumerate(pred_ids)}
    have = [j for j in va_rows if ids[j] in pos]
    m_real = mx_real = None
    if len(have) >= 5:
        pr = np.array([pos[ids[j]] for j in have])
        m_real = stage1_metrics(bR[pr], nR[pr], b[have], P["het4"][have], logn[have], sel(cR, pr),
                                None if cR is None else np.stack([cell_targets(P["ws"][j].astype(np.int64),
                                                                               P["pore"][j]) for j in have]))
        if nx:
            mx_real = stage1_extra(xR[pr], xt[have], xnames)
    res = dict(fold=f, n_train=int(len(tr_rows)), n_heldout=int(len(va_rows)), K=K, consts=consts,
               stage1_renders=m_all, stage1_renders_cartoon=m_cart, stage1_renders_Kavg=m_kavg,
               heldout_real_clean=m_real, t_train_s=round(t_train, 1), t_val_s=round(t_val, 1),
               t_pred_s=round(t_pred, 1), epochs=a.epochs, renders=a.renders, steps=total,
               train_het4=train_het4)
    extra_npz = {}
    if nx or a.render_preset != "noisy":
        res.update(render_preset=a.render_preset, extra_targets=xnames)
    if nx:
        res.update(lambda_x=a.lambda_x, stage1_extra_renders=mx_all, stage1_extra_renders_cartoon=mx_cart,
                   stage1_extra_renders_Kavg=mx_kavg, heldout_real_clean_extra=mx_real)
        extra_npz = dict(x=xR, val_x=xh, x_names=np.array(xnames))
    if opts:
        res.update(opts=opts, ep_log=ep_log)
    if use_cells:
        extra_npz.update(cells=cR.astype(np.float32), val_cells=ch.astype(np.float32))
    if m_split is not None:
        res.update(stage1_renders_het4_hi=m_split["hi"], stage1_renders_het4_lo=m_split["lo"],
                   het4_median=m_split["het4_median"])
    np.savez(out / f"fold{f}.npz", pred_ids=np.array(pred_ids), b=bR, logn=nR, train_ids=ids[tr_rows],
             val_rows=meta, val_b=bh, val_logn=nh, **extra_npz)
    (out / f"fold{f}.json").write_text(json.dumps(res, indent=1))
    s = m_all
    print(f"fold {f} stage-1 (held-out renders, n={s['n']}): corr(sd b-hat, het4) {s['corr_het']:.3f}, partial | logN "
          f"{s['pcorr_het_given_logN']:.3f}, block corr {s['corr_block']:.3f} (within-image {s['corr_block_within']:.3f}),"
          f" corr logN {s['corr_logN']:.3f} | cartoon only: het {m_cart['corr_het']:.3f} partial "
          f"{m_cart['pcorr_het_given_logN']:.3f} | K-avg: het {m_kavg['corr_het']:.3f} partial "
          f"{m_kavg['pcorr_het_given_logN']:.3f}", flush=True)
    print(f"fold {f} stage-1 shrinkage (held-out renders): slope het4 on het-hat | logN "
          f"{s['slope_het4_on_hat_given_logN']:.3f} (= partial corr x sd ratio; sd ratio het4/het-hat | logN "
          f"{s['sdratio_het4_over_hat_given_logN']:.3f}, > 1 = shrunk), het-hat mean {s['het_hat_mean']:.3f} vs het4 "
          f"{s['het4_mean']:.3f}"
          + (f" | cells: pcorr_het_cell {s['pcorr_het_cell']:.3f}, slope {s['slope_het4_on_cellhat_given_logN']:.3f},"
             f" corr_cell {s['corr_cell']:.3f} (within {s['corr_cell_within']:.3f}), block-within from cells "
             f"{s['corr_block_within_cell']:.3f}" if use_cells else "")
          + (f" | K-avg pcorr_het_cell {m_kavg['pcorr_het_cell']:.3f}" if use_cells else ""), flush=True)
    if m_split is not None:
        print(f"fold {f} stage-1 by het4 (median {m_split['het4_median']:.3f}): " + " | ".join(
            f"{nm_}: n {v['n']} pcorr {v['pcorr_het_given_logN']:.3f} within {v['corr_block_within']:.3f} "
            f"het-hat {v['het_hat_mean']:.3f} vs {v['het4_mean']:.3f}" for nm_, v in
            (("hi", m_split["hi"]), ("lo", m_split["lo"])) if v is not None), flush=True)
    if m_real is not None:
        print(f"fold {f} real held-out clean images (no re-render, n={m_real['n']}): corr het {m_real['corr_het']:.3f},"
              f" partial {m_real['pcorr_het_given_logN']:.3f}, block {m_real['corr_block']:.3f}, logN "
              f"{m_real['corr_logN']:.3f}", flush=True)
    if nx:
        print(f"fold {f} extra targets, held-out renders: {_fmt_extra(mx_all)} | K-avg: {_fmt_extra(mx_kavg)}"
              + (f" | real held-out clean: {_fmt_extra(mx_real)}" if mx_real else ""), flush=True)
    print(f"fold {f}: train {t_train:.0f}s, held-out renders {t_val:.0f}s, {len(pred_ids)} real images "
          f"(tta{a.tta}) {t_pred:.0f}s", flush=True)
    if getattr(a, "eval_presets", None):  # --eval-presets: fixed-seed renders of each listed preset + real G1 train
        t3 = time.time()
        g1 = g1_train_ids()
        em, ea = eval_presets_fold(model, consts, P, va_rows, a.eval_presets, K, dev, amp_dtype, a.tta, a.pred_batch,
                                   g1[:8] if a.smoke else g1)
        em.update(fold=f, train_preset=a.render_preset, n_heldout=int(len(va_rows)), t_s=round(time.time() - t3, 1))
        np.savez(out / f"fold{f}_eval.npz", **ea)
        (out / f"fold{f}_eval.json").write_text(json.dumps(em, indent=1))
        print(_eval_line(f"fold {f}", em, a.eval_presets) + f" ({time.time() - t3:.0f}s)", flush=True)
    return res


def _cfg_got(out, f):
    """(render preset, extra targets, options) a cached fold was trained with (from fold{f}.json)."""
    prev = json.loads((out / f"fold{f}.json").read_text()) if (out / f"fold{f}.json").exists() else {}
    return prev.get("render_preset", "noisy"), prev.get("extra_targets", []), prev.get("opts", {})


def _cfg_want(a):
    return a.render_preset, list(a.extra_targets or []), json.loads(json.dumps(new_opts(a)))


def aggregate(a, out, P, folds_of, pred_ids):
    """fold{f}.npz -> het_cnn_{train,test}.parquet + score.json. Every done fold must have been trained with this
    call's (render preset, extra targets, options); --screen removes stale parquets of an earlier full run."""
    done = [f for f in range(5) if (out / f"fold{f}.npz").exists()]
    if not done:
        print("nothing to aggregate")
        return
    want = _cfg_want(a)
    bad = {f: _cfg_got(out, f) for f in done if _cfg_got(out, f) != want}
    if bad:
        raise SystemExit(f"{out}: folds {sorted(bad)} were trained with (render preset, extra targets, options) "
                         f"{bad}, but this call has {want}; not pooling them: use another --out or --overwrite those "
                         f"folds")
    Z = {}
    for f in done:
        with np.load(out / f"fold{f}.npz") as z:
            Z[f] = {k: z[k] for k in z.files}
    ids = list(Z[done[0]]["pred_ids"])
    screen = bool(getattr(a, "screen", False))
    for f in done:
        assert screen or list(Z[f]["pred_ids"]) == ids, f"fold {f} predicted another image list"
    xn = {f: list(Z[f]["x_names"]) if "x_names" in Z[f] else [] for f in done}
    xnames = xn[done[0]]
    assert all(v == xnames for v in xn.values()), f"folds were trained with different --extra-targets: {xn}"
    if screen:  # --screen: no parquets, stage-1 only
        for split in ("train", "test"):
            fp = out / f"het_cnn_{split}.parquet"
            if fp.exists():  # from an earlier full run in this --out: they no longer match these folds
                fp.replace(fp.with_name(fp.name + ".stale"))
                print(f"--screen: renamed the stale {fp.name} of an earlier run to {fp.name}.stale")
        _write_score(a, out, P, Z, done, xnames)
        return
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
    _write_score(a, out, P, Z, done, xnames)


def _write_score(a, out, P, Z, done, xnames):
    per = {f: json.loads((out / f"fold{f}.json").read_text()) for f in done}
    # pooled stage-1 over the done folds' held-out renders
    vb = np.concatenate([Z[f]["val_b"] for f in done])
    vn = np.concatenate([Z[f]["val_logn"] for f in done])
    vr = np.concatenate([Z[f]["val_rows"][:, 0] for f in done]).astype(int)
    vc = vct = None
    if all("val_cells" in Z[f] for f in done):  # FPNNet runs: cell maps of the held-out renders
        vc = np.concatenate([Z[f]["val_cells"] for f in done]).astype(np.float64)
        vct = np.stack([cell_targets(P["ws"][j].astype(np.int64), P["pore"][j]) for j in vr])
    pooled = (stage1_metrics(vb, vn, P["b"][vr], P["het4"][vr], np.log(P["n_eff"][vr]), vc, vct) if len(done) > 1
              else None)
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
                 command=cmd, commands=cmds,
                 args={k: v for k, v in vars(a).items() if not (k in K_ARGS and v is None)},  # unset K options omitted
                 render=R, m_ranges=M,
                 notes="targets from src.het_blocks on the clean originals; hardness never read; test images "
                       "prediction only; clean train rows = OOF fold model, noisy train/test = mean over folds_done")
    if xnames:
        score.update(extra_targets=xnames, stage1_extra_pooled=pooled_x,
                     notes_extra="extra targets fd/pore/asp from the prep arrays of the clean originals "
                                 "(extra_target_values); columns <name>_cnn appended to the parquets")
    if a.render_preset != "noisy":
        score.update(render_preset=a.render_preset, render_preset_note=PRESET_NOTES[a.render_preset])
    if new_opts(a):
        score.update(opts=new_opts(a))
    if getattr(a, "screen", False):
        score.update(notes_screen="--screen: only each fold's held-out real clean images were predicted; no parquets")
    if getattr(a, "eval_presets", None):  # new key only; every key above is unchanged
        score.update(stage1_eval=_stage1_eval_score(a, out, P, Z, done))
    (out / "score.json").write_text(json.dumps(score, indent=1, default=str))
    print(f"wrote {out / 'score.json'} (folds {done})")


K_ARGS = ("eval_presets", "eval_only")  # 2026-10-09 (night) options: left out of score.json 'args' when unset


def _stage1_eval_score(a, out, P, Z, done):
    """score.json 'stage1_eval' (--eval-presets): summarize_eval over the done folds that have fold{f}_eval.json/.npz
    (a fold cached from a run without --eval-presets has none: folds_missing), plus the real held-out clean R per fold
    (heldout_real_clean pcorr_het_given_logN), its mean over the folds, and R pooled over the folds' held-out real clean
    predictions (fold{f}.npz; no re-render)."""
    have = [f for f in done if (out / f"fold{f}_eval.json").exists() and (out / f"fold{f}_eval.npz").exists()]
    met, arr = {}, {}
    for f in have:
        met[f] = json.loads((out / f"fold{f}_eval.json").read_text())
        with np.load(out / f"fold{f}_eval.npz") as z:
            arr[f] = {k: z[k] for k in z.files}
    S = summarize_eval(met, arr, P, a.eval_presets) if have else dict(folds=[], per_fold={}, mean_folds={}, pooled={})
    ids, logn = P["ids"], np.log(P["n_eff"])
    per_r, bR, nR, jj = {}, [], [], []
    for f in done:
        pos = {i: n for n, i in enumerate(Z[f]["pred_ids"])}
        hv = [j for j in np.unique(Z[f]["val_rows"][:, 0]).astype(int) if ids[j] in pos]
        if hv:
            pr = np.array([pos[ids[j]] for j in hv])
            bR.append(Z[f]["b"][pr])
            nR.append(Z[f]["logn"][pr])
            jj.append(np.array(hv))
        hr = json.loads((out / f"fold{f}.json").read_text()).get("heldout_real_clean")
        per_r[str(f)] = None if hr is None else hr["pcorr_het_given_logN"]
    j = np.concatenate(jj) if jj else np.zeros(0, int)
    rv = [v for v in per_r.values() if v is not None]
    S.update(presets=list(a.eval_presets), seed_form=f"default_rng([{EVAL_SEED}, k, j])",
             K=(met[have[0]][a.eval_presets[0]]["K"] if have else None),
             folds_missing=[f for f in done if f not in have], R_per_fold=per_r,
             R_mean_folds=float(np.mean(rv)) if rv else None,
             heldout_real_clean_pooled=(stage1_metrics(np.concatenate(bR), np.concatenate(nR), P["b"][j], P["het4"][j],
                                                       logn[j]) if len(j) >= 5 else None),
             ranges={pr: dict(render=PRESETS[pr][0], m_ranges=PRESETS[pr][1]) for pr in a.eval_presets})
    return S


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


def new_opts(a):
    """The 2026-10-09 (evening) options that differ from their defaults ({} = a run of the earlier code)."""
    o = {}
    if getattr(a, "zoom", None) is not None:
        o.update(zoom=list(a.zoom), p_zoom=a.p_zoom)
    if getattr(a, "mosaic", 0) > 0:
        o.update(mosaic=a.mosaic)
        if getattr(a, "mosaic_dlogn", None) is not None:
            o.update(mosaic_dlogn=a.mosaic_dlogn)
    if getattr(a, "head", "grid4") != "grid4":
        o.update(head=a.head, lambda_cell=a.lambda_cell, fpn_ch=a.fpn_ch)
    if getattr(a, "lambda_within", 0) > 0:
        o.update(lambda_within=a.lambda_within)
    if getattr(a, "screen", False):
        o.update(screen=True)
    return o


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
    if a.eval_only:
        run_eval_only(a)
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
          + (f" | render preset {a.render_preset}" if a.render_preset != "noisy" else "")
          + (f" | options {new_opts(a)}" if new_opts(a) else "")
          + (f" | eval presets {a.eval_presets}" if a.eval_presets else ""), flush=True)
    t = time.time()
    for f in (a.folds if a.folds is not None else range(5)):
        if (out / f"fold{f}.npz").exists() and not a.overwrite:
            got, want = _cfg_got(out, f), _cfg_want(a)
            if got != want:
                raise SystemExit(f"{out}/fold{f}: cached with (render preset, extra targets, options) {got}, but this "
                                 f"call asks for {want}: use another --out or --overwrite")
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
    selftest_aug(P, rows)
    selftest_k(P, rows)
    print("selftest passed")


def _ref_relabel(raw):
    """Independent reference for relabel(): scipy 4-connected components per label value (find_objects windows)."""
    out = np.zeros(raw.shape, np.int64)
    n = 0
    lab_ids = np.unique(raw)
    remap = np.zeros(int(raw.max()) + 1, np.int64)
    remap[lab_ids] = np.arange(1, len(lab_ids) + 1)
    dense = remap[raw]
    for v, sl in enumerate(ndi.find_objects(dense), start=1):
        if sl is None:
            continue
        cc, m = ndi.label(dense[sl] == v)  # default structure = 4-connectivity
        w = out[sl]
        w[cc > 0] = cc[cc > 0] + n
        n += m
    return out


def _small_grains_ok(ws, pore):
    """split_merge invariant: every grain below MIN_GRAIN is a pore grain or touches (4-neighbour) no non-pore grain
    of >= MIN_GRAIN px."""
    big_np = (np.bincount(ws.ravel(), minlength=len(pore)) >= MIN_GRAIN) & ~pore
    small_np = ~big_np & ~pore
    s_ = small_np[ws]
    if not s_.any():
        return True
    b_ = big_np[ws]
    touch = (s_[1:] & b_[:-1]).any() | (s_[:-1] & b_[1:]).any() | (s_[:, 1:] & b_[:, :-1]).any() | \
        (s_[:, :-1] & b_[:, 1:]).any()
    return not touch


def _same_partition(x, y):
    pairs = np.unique(np.stack([x.ravel(), y.ravel()]), axis=1)
    return pairs.shape[1] == len(np.unique(x)) == len(np.unique(y))


def selftest_aug(P, rows):
    """Checks of the 2026-10-09 (evening) options: (8) zoom s=1 and mosaic with an all-true / all-false mask reproduce
    the prep targets and the plain cartoon render exactly; fast target functions == the originals; (9) mosaic of an
    image with itself along random masks: relabel() == an independent scipy relabelling, targets == recomputed on it;
    (10) augmented sources / dataset items: targets == the slow definitions, het4 == sd(valid b), finite logN;
    (11) cells: D4 transform of the cell targets == targets of the transformed map, cells -> blocks, TTA path of
    predict for cell maps, FPNNet outputs, RenderSet cell tensors; (12) defaults == the pre-options file (git show)."""
    import importlib.util
    import subprocess
    import tempfile
    n = len(P["ids"])
    # (8)
    for j in rows:
        src = source_of(P, j)
        other = source_of(P, (j + 1) % n)
        ws0, pore0 = src["ws"], src["pore"]
        assert np.array_equal(block_targets_fast(ws0, pore0), P["b"][j], equal_nan=True)
        assert n_eff_fast(ws0, pore0) == n_eff_of(ws0, pore0) == P["n_eff"][j]
        cands = [("zoom s=1", zoom_source(src, 1.0, 0.0, 0.0)),
                 ("mosaic all-true", mosaic_source(src, other, np.ones((256, 256), bool))),
                 ("mosaic all-false", mosaic_source(other, src, np.zeros((256, 256), bool))),
                 ("zoom s=1 + mosaic all-true", mosaic_source(zoom_source(src, 1.0, 0.0, 0.0), other,
                                                              np.ones((256, 256), bool)))]
        for nm, s_ in cands:
            t = targets_of(s_, list(EXTRA_TARGETS), cells=True)
            assert np.array_equal(t["b"], P["b"][j], equal_nan=True), nm
            assert np.array_equal(block_targets(s_["ws"], s_["pore"]), P["b"][j], equal_nan=True), nm
            assert t["het4"] == P["het4"][j] and t["n_eff"] == P["n_eff"][j], nm
            assert np.allclose(t["x"], extra_target_values({**P, "ids": P["ids"][j:j + 1], "ws": P["ws"][j:j + 1],
                                                            "dark": P["dark"][j:j + 1], "pore": P["pore"][j:j + 1]},
                                                           list(EXTRA_TARGETS))[0], rtol=0, atol=1e-6), nm
            assert np.array_equal(t["cells"], cell_targets(ws0, pore0), equal_nan=True), nm
            assert _same_partition(s_["ws"], ws0), nm
            for force_seed in range(2 if "all-false" not in nm else 0):  # all-false sits on the other's shading
                r0, i0 = render_noisy(P, j, np.random.default_rng([3, force_seed, j]), force="cartoon")
                r1, i1 = render_source(s_, np.random.default_rng([3, force_seed, j]), RENDER)
                assert np.array_equal(r0, r1) and i0 == i1, f"{nm}: render differs from the plain cartoon"
    print(f"ok 8: zoom s=1 / mosaic all-true / all-false / both reproduce b, het4, N_eff, extras, cells and the plain "
          f"cartoon render exactly ({len(rows)} images); block_targets_fast == block_targets, n_eff_fast == n_eff_of")
    # (9)
    nmask = nsmall = 0
    for j in rows[:3]:
        src = source_of(P, j)
        for t_ in range(4):
            ra = np.random.default_rng([11, j, t_])
            mask = mosaic_mask(ra)
            off = int(src["ws"].max()) + 1
            raw = np.where(mask, src["ws"], src["ws"] + off)
            tabs = {k: np.concatenate([src[k][:off], src[k]]) for k in GRAIN_TABS}
            # pure relabel vs an independent scipy relabelling; targets on it == recomputed with the slow definitions
            rl, rt = relabel(raw, tabs)
            ref = _ref_relabel(raw)
            assert _same_partition(rl, ref), "relabel != scipy reference"
            pore_ref = np.zeros(int(ref.max()) + 1, bool)
            pore_ref[ref.ravel()] = src["pore"][raw.ravel() % off]
            dark_ref = np.zeros_like(pore_ref)
            dark_ref[ref.ravel()] = src["dark"][raw.ravel() % off]
            assert np.array_equal(rt["pore"][rl], pore_ref[ref]) and np.array_equal(rt["dark"][rl], dark_ref[ref])
            b_ref = block_targets(ref, pore_ref)
            t = targets_of(dict(ws=rl, **rt))
            assert np.array_equal(t["b"], b_ref, equal_nan=True)
            assert t["n_eff"] == n_eff_of(ref, pore_ref) and t["het4"] == float(np.std(b_ref[~np.isnan(b_ref)]))
            assert len(np.unique(rl)) >= len(np.unique(src["ws"]))  # splitting only adds grains
            # full mosaic_source (split_merge): only pixels of non-pore components < MIN_GRAIN change grain; no
            # non-pore grain below MIN_GRAIN next to a bigger non-pore grain is left; the pore flag of EVERY pixel is
            # unchanged; tables per pixel unchanged elsewhere; targets == slow definitions on the final map
            mos = mosaic_source(src, src, mask)
            small = ((np.bincount(rl.ravel()) < MIN_GRAIN) & ~rt["pore"])[rl]
            assert _same_partition(np.where(small, -1, mos["ws"]), np.where(small, -1, rl))
            assert _small_grains_ok(mos["ws"], mos["pore"])
            assert _same_partition(mos["ws"], relabel(mos["ws"], {})[0])  # every grain connected
            assert np.array_equal(mos["pore"][mos["ws"]], rt["pore"][rl])
            for k_ in GRAIN_TABS:
                assert np.array_equal(mos[k_][mos["ws"]][~small], rt[k_][rl][~small])
            tm = targets_of(mos)
            assert np.array_equal(tm["b"], block_targets(mos["ws"], mos["pore"]), equal_nan=True)
            assert tm["n_eff"] == n_eff_of(mos["ws"], mos["pore"])
            nmask += 1
            nsmall += int(small.sum())
    print(f"ok 9: mosaic of an image with itself ({nmask} random masks): relabel == scipy per-label 4-connected "
          f"components; per-grain tables carried; b, het4, N_eff == recomputed on that map; split_merge only moved "
          f"the {nsmall} pixels of non-pore components < {MIN_GRAIN} px (pore flag of every pixel unchanged)")
    # (10)
    aug = dict(zoom=(0.7, 1.4), p_zoom=0.5, mosaic=0.5)
    trows = np.arange(min(n, 40))
    n_aug = 0
    for i in range(16):
        j = int(trows[i % len(trows)])
        s_ = augment_source(P, j, trows, np.random.default_rng([0, 0, 0, i, AUG_TAG]), aug)
        if s_ is None:
            continue
        n_aug += 1
        t = targets_of(s_, list(EXTRA_TARGETS), cells=True)
        assert np.array_equal(t["b"], block_targets(s_["ws"], s_["pore"]), equal_nan=True)
        v = ~np.isnan(t["b"])
        assert t["het4"] == float(np.std(t["b"][v])) and np.isfinite(np.log(t["n_eff"]))
        assert t["n_eff"] == n_eff_of(s_["ws"], s_["pore"])
        assert abs(t["x"][2] - grain_log_aspect(s_["ws"], s_["pore"])) < 1e-6
        assert len(s_["sn"]) >= s_["ws"].max() + 1 and s_["ws"].min() >= 1
        assert _small_grains_ok(s_["ws"], s_["pore"])
        assert _same_partition(s_["ws"], relabel(s_["ws"], {})[0])  # every grain connected
        assert not (s_["pore"][s_["ws"]] & ~s_["rpore"]).any()  # target pores only where the render draws pores
        assert s_["resid"].dtype == np.float32 and s_["im"].shape == (256, 256)
    xt = extra_target_values(P, list(EXTRA_TARGETS))
    dsa = RenderSet(PREP_FP, trows, 2, 0, 0, xt=xt, aug=aug, cells=True, xnames=list(EXTRA_TARGETS))
    dsp = RenderSet(PREP_FP, trows, 2, 0, 0, xt=xt, cells=True)
    n_same = 0
    for idx in range(0, 2 * dsa.n_items, 7):
        it = dsa[idx]
        assert len(it) == 8 and it[0].dtype == torch.uint8 and it[6].shape == (16, 16)
        bv = it[1][it[2] > 0].double().numpy()
        assert abs(float(np.std(bv)) - float(it[3])) < 1e-5 and torch.isfinite(it[4])
        assert all(torch.equal(u, w) for u, w in zip(it, dsa[idx]))
        n_same += all(torch.equal(u, w) for u, w in zip(it, dsp[idx]))
    print(f"ok 10: {n_aug} of 16 augmented sources: targets == slow definitions, het4 == sd(valid b), finite logN; "
          f"augmented dataset items deterministic, het4 == sd(valid b) ({n_same} of {len(range(0, 2 * dsa.n_items, 7))}"
          f" items drew no augmentation and equal the plain items)")
    # (11)
    for j in rows[:3]:
        ws, pore = P["ws"][j].astype(np.int64), P["pore"][j]
        c0 = cell_targets(ws, pore)
        for k in range(8):
            ck = cell_targets(np.ascontiguousarray(d4_np(ws, k)), pore)
            assert np.array_equal(np.isnan(ck), np.isnan(d4_np(c0, k)))
            assert np.allclose(ck, d4_np(c0, k), rtol=0, atol=1e-12, equal_nan=True)
        cnt = np.bincount(_CELL[~pore[ws.ravel()]], minlength=256).reshape(4, 4, 4, 4).transpose(0, 2, 1, 3)
        full = (cnt == 256).all((2, 3)).ravel()  # blocks whose 16 cells have no pore pixel: cells -> blocks exact
        assert np.allclose(cells_to_blocks(c0)[full], P["b"][j][full], rtol=0, atol=1e-9)
    x = torch.randn(2, 3, 16, 16)
    assert all(torch.equal(d4_inv_t(d4_t(x, k), k), x) for k in range(8))

    class CellMean(nn.Module):  # exact 64-px block means, image mean, exact 16-px cell means
        has_cells = True

        def forward(self, x):
            return F.avg_pool2d(x, 64).flatten(1), x.mean((1, 2, 3)), F.avg_pool2d(x, 16)[:, 0]

    X8 = np.stack([P["im8"][j] for j in rows[:3]])
    refc = X8.astype(np.float64).reshape(3, 16, 16, 16, 16).mean((2, 4)) / 255.0
    for tta in (1, 8):
        r_ = predict(CellMean(), X8, torch.tensor(0.0), torch.tensor(1.0), torch.device("cpu"), None, tta=tta)
        assert len(r_) == 3 and np.allclose(r_[2], refc, atol=1e-5), f"cell TTA un-transform wrong (tta={tta})"
    for nx_ in (0, 3):
        net = FPNNet("resnet18", pretrained=False, n_extra=nx_).eval()
        o = net(torch.zeros(2, 1, 256, 256))
        assert len(o) == 3 + (nx_ > 0) and o[0].shape == (2, 16) and o[1].shape == (2,) and o[-1].shape == (2, 16, 16)
        r_ = predict(net, X8[:2], torch.tensor(0.5), torch.tensor(0.2), torch.device("cpu"), None, tta=2)
        assert len(r_) == 3 + (nx_ > 0) and r_[-1].shape == (2, 16, 16)
    dsc = RenderSet(PREP_FP, [0, 1], 2, 0, 0, cells=True)
    for idx in (0, 3, 5):
        it, it0 = dsc[idx], ds_plain_item(idx)
        assert all(torch.equal(u, w) for u, w in zip(it[:5], it0))
        ep, i = divmod(idx, dsc.n_items)
        rng = np.random.default_rng([0, 0, ep, i])
        render_noisy(P, [0, 1][i % 2], rng)
        k = int(rng.integers(8))
        ck = d4_np(cell_targets(P["ws"][[0, 1][i % 2]].astype(np.int64), P["pore"][[0, 1][i % 2]]), k)
        assert np.array_equal(it[6].numpy() > 0, ~np.isnan(ck)) and np.allclose(it[5].numpy(), np.nan_to_num(ck),
                                                                                 atol=1e-5)
    print("ok 11: cell targets D4-equivariant (24 image-op pairs), cells -> blocks exact on pore-free blocks, "
          "d4_inv_t inverts d4_t, predict un-transforms cell maps (tta 1/8), FPNNet output shapes (+extras), "
          "RenderSet cell tensors == D4-transformed cell targets")
    # (12) defaults vs the version of this file from before the 2026-10-09 (evening) options: the parent of the first
    # commit that contains AUG_TAG (HEAD while the options are uncommitted), so the check stays meaningful after the
    # commit. Loaded from a temp dir under a per-process module name (nothing is written into src/, no .pyc).
    try:
        log = subprocess.run(["git", "log", "--format=%H", "--reverse", "-S", "AUG_TAG = 9001", "--",
                              "src/het_cnn.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        rev = f"{log[0]}^" if log else "HEAD"
        sha = subprocess.run(["git", "rev-parse", "--short", rev], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.strip()
        txt = subprocess.run(["git", "show", f"{rev}:src/het_cnn.py"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout
    except Exception as e:  # noqa: BLE001
        print(f"skip 12: git could not provide the pre-options src/het_cnn.py ({e})")
        return
    assert "AUG_TAG" not in txt, f"reference {rev} already has the options"
    mod_name = f"{__package__}._het_cnn_ref_{os.getpid()}"
    tmp = tempfile.TemporaryDirectory(prefix="het_cnn_ref_")
    ref_fp = Path(tmp.name) / f"_het_cnn_ref_{os.getpid()}.py"
    ref_fp.write_text(txt)
    dwb = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(mod_name, ref_fp)
        ref = importlib.util.module_from_spec(spec)  # __package__ = 'src': its relative imports resolve to src.*
        sys.modules[mod_name] = ref
        spec.loader.exec_module(ref)
        nchk = 0
        for preset in ("noisy", "mid"):
            for use_x in (False, True):
                kw = dict(preset=preset, xt=xt if use_x else None)
                d_new = RenderSet(PREP_FP, np.arange(10), 1, 0, 1, **kw)
                d_ref = ref.RenderSet(PREP_FP, np.arange(10), 1, 0, 1, **kw)
                for idx in (0, 1, 4, 7, 8, 9, 10, 12, 15, 17, 18, 19):  # 12 items across 2 epochs
                    a_, b_ = d_new[idx], d_ref[idx]
                    assert len(a_) == len(b_) and all(torch.equal(u, w) for u, w in zip(a_, b_)), (preset, idx)
                    nchk += 1
        for t in range(6):
            for force in (None, "cartoon", "m"):
                for preset in ("noisy", "mid"):
                    R_, M_ = PRESETS[preset]
                    r0, i0 = render_noisy(P, t, np.random.default_rng([9, t]), R_, force, M_)
                    r1, i1 = ref.render_noisy(P, t, np.random.default_rng([9, t]), R_, force, M_)
                    assert np.array_equal(r0, r1) and i0 == i1
        for nx_ in (0, 3):
            torch.manual_seed(0)
            m_new = HetNet("resnet18.a1_in1k", pretrained=False, n_extra=nx_)
            torch.manual_seed(0)
            m_ref = ref.HetNet("resnet18.a1_in1k", pretrained=False, n_extra=nx_)
            sd_n, sd_r = m_new.state_dict(), m_ref.state_dict()
            assert list(sd_n) == list(sd_r) and all(torch.equal(sd_n[k], sd_r[k]) for k in sd_n)
            xin = torch.randn(2, 1, 256, 256, generator=torch.Generator().manual_seed(1))
            for mode in ("train", "eval"):
                getattr(m_new, mode)(), getattr(m_ref, mode)()
                o_n, o_r = m_new(xin), m_ref(xin)
                assert len(o_n) == len(o_r) and all(torch.equal(u, w) for u, w in zip(o_n, o_r))
            pn = predict(m_new, X8, torch.tensor(0.5), torch.tensor(0.2), torch.device("cpu"), None, tta=8)
            pr = ref.predict(m_ref, X8, torch.tensor(0.5), torch.tensor(0.2), torch.device("cpu"), None, tta=8)
            assert len(pn) == len(pr) and all(np.array_equal(u, w) for u, w in zip(pn, pr))
        bh = np.random.default_rng(2).normal(size=(30, 16))
        args = (bh, bh[:, 0], P["b"][:30], P["het4"][:30], np.log(P["n_eff"][:30]))
        m_n, m_r = stage1_metrics(*args), ref.stage1_metrics(*args)
        assert all(m_n[k] == m_r[k] for k in m_r) and set(m_n) - set(m_r) == {"slope_het4_on_hat_given_logN",
                                                                              "sdratio_het4_over_hat_given_logN"}
        print(f"ok 12: default flags == src/het_cnn.py at {rev} ({sha}, before the evening options) bit for bit: "
              f"{nchk} RenderSet items (2 epochs, presets noisy/mid, +-extras), 36 renders, HetNet init state_dict + "
              f"train/eval outputs + TTA predict (n_extra 0/3), stage1_metrics old keys (two additive keys)")
    finally:
        sys.dont_write_bytecode = dwb
        sys.modules.pop(mod_name, None)
        tmp.cleanup()


def ds_plain_item(idx):
    """Item idx of the default RenderSet(PREP_FP, [0, 1], 2, 0, 0) (selftest helper)."""
    return RenderSet(PREP_FP, [0, 1], 2, 0, 0)[idx]


def selftest_k(P, rows):
    """Check 13, the 2026-10-09 (night) options of route K: the 'real' preset is RENDER / M_RANGES with exactly the
    specified changes, its renders are deterministic and inside its ranges on both paths; eval_render_set == the
    stage-1 render loop of train_fold (seed 20261008) for every preset and identical on repeat; eval_presets_fold /
    summarize_eval on an exact block / cell-mean model (pooled == metrics of the concatenated folds); model_from_ckpt
    rebuilds FPNNet / HetNet (+extras) from a saved checkpoint with identical outputs; and the default flags == the file
    before these options (git: parent of the first commit with RENDER_REAL; HEAD while uncommitted) bit for bit."""
    import hashlib
    import importlib.util
    import subprocess
    import tempfile
    Rr, Mr = PRESETS["real"]
    assert list(Rr) == list(RENDER) and list(Mr) == list(M_RANGES)
    assert {k: v for k, v in Rr.items() if RENDER[k] != v} == dict(p_line=0.9, line=(0.2, 1.1), blur=(0.3, 1.3),
                                                                     e=(1.0, 2.0))
    assert {k: v for k, v in Mr.items() if M_RANGES[k] != v} == dict(sb=(0.4, 1.4))
    assert CALIB_BAND["real"] == CALIB_BAND["noisy"]
    n_line = n_cart = n_m = 0
    for t in range(12):
        j = t % len(P["ids"])
        for force in (None, "cartoon", "m"):
            r0, i0 = render_noisy(P, j, np.random.default_rng([13, t]), Rr, force, Mr)
            r1, i1 = render_noisy(P, j, np.random.default_rng([13, t]), Rr, force, Mr)
            assert np.array_equal(r0, r1) and i0 == i1 and r0.dtype == np.uint8 and r0.shape == (256, 256)
            if i0["path"] == 0:
                n_cart += 1
                for key in ("c_d", "e", "blur", "noise"):
                    assert Rr[key][0] <= i0[key] <= Rr[key][1], (key, i0[key])
                assert i0["line"] == 0 or Rr["line"][0] <= i0["line"] <= Rr["line"][1], i0["line"]
                n_line += i0["line"] > 0
            else:
                n_m += 1
                for key, mk in (("c_d", "cd"), ("e", "ex"), ("blur", "sb"), ("noise", "sn")):
                    assert Mr[mk][0] <= i0[key] <= Mr[mk][1], (key, i0[key])
    dn, dr = RenderSet(PREP_FP, [0, 1], 2, 0, 0), RenderSet(PREP_FP, [0, 1], 2, 0, 0, preset="real")
    assert all(torch.equal(u, v) for u, v in zip(dr[5], dr[5])) and not torch.equal(dr[5][0], dn[5][0])
    # eval_render_set == train_fold's stage-1 loop (the literal seed of that loop), for every preset; repeatable
    va = np.asarray(rows[:3])
    md5 = {}
    for pr in sorted(PRESETS):
        rR, rM = PRESETS[pr]
        Xv, meta = [], []
        for j in va:
            for k in range(2):
                img, info = render_noisy(P, int(j), np.random.default_rng([20261008, k, int(j)]), rR, M=rM)
                Xv.append(img)
                meta.append((int(j), k, info["path"]))
        X1, m1 = eval_render_set(P, va, pr, 2)
        X2, m2 = eval_render_set(P, va, pr, 2)
        assert np.array_equal(np.stack(Xv), X1) and np.array_equal(np.array(meta), m1)
        assert np.array_equal(X1, X2) and np.array_equal(m1, m2)
        md5[pr] = hashlib.md5(X1.tobytes()).hexdigest()[:8]
    assert len(set(md5.values())) == len(md5)

    class CellMean(nn.Module):  # exact 64-px block means, image mean, exact 16-px cell means
        has_cells = True

        def forward(self, x):
            return F.avg_pool2d(x, 64).flatten(1), x.mean((1, 2, 3)), F.avg_pool2d(x, 16)[:, 0]

    consts = dict(mu_b=0.0, s_b=1.0, mu_n=0.0, s_n=1.0, mu_px=0.0, sd_px=1.0)
    g1 = g1_train_ids()
    folds_rows = {0: np.asarray(rows[:6]), 1: np.asarray(rows[6:12]) if len(rows) >= 12 else np.arange(6, 12)}
    met, arr = {}, {}
    for f, vr in folds_rows.items():
        met[f], arr[f] = eval_presets_fold(CellMean(), consts, P, vr, ["noisy", "real"], 2, torch.device("cpu"), None,
                                           tta=8, bs=8, g1_ids=g1[:3])
        for pr in ("noisy", "real"):
            Xp, _ = eval_render_set(P, vr, pr, 2)
            ref = Xp.astype(np.float64).reshape(len(Xp), 4, 64, 4, 64).mean((2, 4)).reshape(len(Xp), 16) / 255.0
            assert np.allclose(arr[f][f"{pr}_b"], ref, atol=1e-5) and np.isfinite(met[f][pr]["P"])
            assert arr[f][f"{pr}_cells"].shape == (len(Xp), 16, 16)
        assert met[f]["real_g1_train"]["n"] == 3 and arr[f]["g1_b"].shape == (3, 16)
    S = summarize_eval(met, arr, P, ["noisy", "real"])
    for pr in ("noisy", "real"):
        rws = np.concatenate([arr[f][f"{pr}_meta"][:, 0] for f in (0, 1)]).astype(int)
        cto = np.stack([cell_targets(P["ws"][j].astype(np.int64), P["pore"][j]) for j in rws])
        ref = stage1_metrics(np.concatenate([arr[f][f"{pr}_b"] for f in (0, 1)]),
                             np.concatenate([arr[f][f"{pr}_logn"] for f in (0, 1)]), P["b"][rws], P["het4"][rws],
                             np.log(P["n_eff"][rws]),
                             np.concatenate([arr[f][f"{pr}_cells"] for f in (0, 1)]).astype(np.float64), cto)
        assert all(np.isclose(S["pooled"][pr][k], ref[k], rtol=0, atol=1e-12) for k in ref), pr
        assert abs(S["mean_folds"][pr]["P"] - (met[0][pr]["P"] + met[1][pr]["P"]) / 2) < 1e-12
    assert S["real_g1_train"]["n"] == 3
    json.dumps(S)  # json-able
    # model_from_ckpt round trip
    nck = 0
    with tempfile.TemporaryDirectory(prefix="het_cnn_ck_") as td:
        for head, nx_ in (("fpn", 0), ("fpn", 3), ("grid4", 0), ("grid4", 3)):
            torch.manual_seed(0)
            m = (FPNNet("resnet18", pretrained=False, n_extra=nx_, ch=64) if head == "fpn"
                 else HetNet("resnet18", pretrained=False, n_extra=nx_)).eval()
            ck = {"state_dict": m.state_dict(), "consts": consts, "arch": "resnet18"}
            if nx_:
                ck.update(extra_targets=list(EXTRA_TARGETS), render_preset="noisy")
            if head == "fpn":
                ck.update(opts=dict(head="fpn", lambda_cell=1.0, fpn_ch=64))
            torch.save(ck, Path(td) / "fold0.pt")
            m2 = model_from_ckpt(torch.load(Path(td) / "fold0.pt", map_location="cpu", weights_only=False)).eval()
            assert type(m2) is type(m)
            x = torch.randn(2, 1, 256, 256, generator=torch.Generator().manual_seed(3))
            with torch.no_grad():
                assert all(torch.equal(u, w) for u, w in zip(m(x), m2(x)))
            nck += 1
    print(f"ok 13: 'real' preset = RENDER / M_RANGES with exactly p_line 0.9, line 0.2-1.1, blur 0.3-1.3, e 1.0-2.0 / "
          f"sb 0.4-1.4; {n_cart} cartoon ({n_line} with lines) + {n_m} degrade_m renders deterministic and in range; "
          f"eval_render_set == the stage-1 loop for {sorted(PRESETS)} (md5 {md5}); eval_presets_fold / summarize_eval "
          f"exact on block / cell means (pooled == concatenated folds); model_from_ckpt round trip ({nck} heads); "
          f"{len(g1)} real G1 train images")
    # defaults vs the file before these options (loaded from git into a temp dir, as check 12 does)
    try:
        log = subprocess.run(["git", "log", "--format=%H", "--reverse", "-S", "RENDER_REAL = dict(", "--",
                              "src/het_cnn.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        rev = f"{log[0]}^" if log else "HEAD"
        sha = subprocess.run(["git", "rev-parse", "--short", rev], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.strip()
        txt = subprocess.run(["git", "show", f"{rev}:src/het_cnn.py"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout
    except Exception as e:  # noqa: BLE001
        print(f"skip 13b: git could not provide the pre-K src/het_cnn.py ({e})")
        return
    assert "RENDER_REAL" not in txt, f"reference {rev} already has the K options"
    mod_name = f"{__package__}._het_cnn_refk_{os.getpid()}"
    tmp = tempfile.TemporaryDirectory(prefix="het_cnn_refk_")
    ref_fp = Path(tmp.name) / f"_het_cnn_refk_{os.getpid()}.py"
    ref_fp.write_text(txt)
    dwb = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(mod_name, ref_fp)
        ref = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = ref
        spec.loader.exec_module(ref)
        for pr in ("noisy", "mid"):
            assert PRESETS[pr] == ref.PRESETS[pr] and CALIB_BAND[pr] == ref.CALIB_BAND[pr]
        assert set(ref.PRESETS) == {"noisy", "mid"}
        xt = extra_target_values(P, list(EXTRA_TARGETS))
        aug = dict(zoom=(1.0, 1.4), p_zoom=0.5, mosaic=0.5, mosaic_dlogn=0.5)  # the ev2s_loc_e32r4 options
        nitem = 0
        for kw in (dict(preset="noisy"), dict(preset="mid", xt=xt), dict(preset="noisy", aug=aug, cells=True),
                   dict(preset="noisy", xt=xt, aug=aug, cells=True, xnames=list(EXTRA_TARGETS))):
            d_new = RenderSet(PREP_FP, np.arange(12), 1, 0, 1, **kw)
            d_ref = ref.RenderSet(PREP_FP, np.arange(12), 1, 0, 1, **kw)
            for idx in (0, 1, 5, 7, 11, 12, 13, 18, 23):  # 9 items across 2 epochs
                a_, b_ = d_new[idx], d_ref[idx]
                assert len(a_) == len(b_) and all(torch.equal(u, w) for u, w in zip(a_, b_)), (kw.keys(), idx)
                nitem += 1
        nr = 0
        for t in range(4):
            for force in (None, "cartoon", "m"):
                for pr in ("noisy", "mid"):
                    R_, M_ = PRESETS[pr]
                    r0, i0 = render_noisy(P, t, np.random.default_rng([9, t]), R_, force, M_)
                    r1, i1 = ref.render_noisy(P, t, np.random.default_rng([9, t]), R_, force, M_)
                    assert np.array_equal(r0, r1) and i0 == i1
                    nr += 1
        for pr in ("noisy", "mid"):  # held-out stage-1 renders (the ref has no eval_render_set: its loop by hand)
            R_, M_ = PRESETS[pr]
            X1, _ = eval_render_set(P, va, pr, 4)
            X0 = np.stack([ref.render_noisy(P, int(j), np.random.default_rng([20261008, k, int(j)]), R_, M=M_)[0]
                           for j in va for k in range(4)])
            assert np.array_equal(X0, X1)
        X8 = np.stack([P["im8"][j] for j in rows[:3]])
        for nx_ in (0, 3):
            torch.manual_seed(0)
            m_new = FPNNet("resnet18.a1_in1k", pretrained=False, n_extra=nx_)
            torch.manual_seed(0)
            m_ref = ref.FPNNet("resnet18.a1_in1k", pretrained=False, n_extra=nx_)
            sd_n, sd_r = m_new.state_dict(), m_ref.state_dict()
            assert list(sd_n) == list(sd_r) and all(torch.equal(sd_n[k], sd_r[k]) for k in sd_n)
            xin = torch.randn(2, 1, 256, 256, generator=torch.Generator().manual_seed(1))
            for mode in ("train", "eval"):
                getattr(m_new, mode)(), getattr(m_ref, mode)()
                o_n, o_r = m_new(xin), m_ref(xin)
                assert len(o_n) == len(o_r) and all(torch.equal(u, w) for u, w in zip(o_n, o_r))
            pn = predict(m_new, X8, torch.tensor(0.5), torch.tensor(0.2), torch.device("cpu"), None, tta=8)
            pr_ = ref.predict(m_ref, X8, torch.tensor(0.5), torch.tensor(0.2), torch.device("cpu"), None, tta=8)
            assert len(pn) == len(pr_) and all(np.array_equal(u, w) for u, w in zip(pn, pr_))
        bh = np.random.default_rng(2).normal(size=(30, 16))
        ch = np.random.default_rng(3).normal(size=(30, 16, 16))
        ct = np.stack([cell_targets(P["ws"][j].astype(np.int64), P["pore"][j]) for j in range(30)])
        args = (bh, bh[:, 0], P["b"][:30], P["het4"][:30], np.log(P["n_eff"][:30]), ch, ct)
        m_n, m_r = stage1_metrics(*args), ref.stage1_metrics(*args)
        assert list(m_n) == list(m_r) and all(m_n[k] == m_r[k] for k in m_r)
        loc = ["--head", "fpn", "--mosaic", "0.5", "--mosaic-dlogn", "0.5", "--zoom", "1.0", "1.4", "--lambda-within",
               "1", "--folds", "0", "1", "--screen", "--out", "x"]
        for argv in ([], ["--out", "x"], loc, ["--render-preset", "mid", "--extra-targets", "fd", "asp"]):
            vn, vr_ = vars(parse(argv)), vars(ref.parse(argv))
            assert {k: v for k, v in vn.items() if k not in K_ARGS} == vr_ and all(vn[k] is None for k in K_ARGS)
            assert new_opts(parse(argv)) == ref.new_opts(ref.parse(argv))
            assert _cfg_want(parse(argv)) == ref._cfg_want(ref.parse(argv))
        print(f"ok 13b: default flags == src/het_cnn.py at {rev} ({sha}, before the K options) bit for bit: {nitem} "
              f"RenderSet items (presets noisy/mid, +-extras, +-zoom/mosaic/cells), {nr} renders, the stage-1 held-out "
              f"renders (noisy/mid), FPNNet init state_dict + train/eval outputs + TTA predict (n_extra 0/3), "
              f"stage1_metrics with cells (all keys), parse / new_opts / cache keys (only the two new args, None)")
    finally:
        sys.dont_write_bytecode = dwb
        sys.modules.pop(mod_name, None)
        tmp.cleanup()


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
                    help="noisy (default, as before) | mid (RENDER_MID, for raw ic_noise 9.5-12) | real (RENDER_REAL, "
                         "the widened realistic prior of route K / D1)")
    ap.add_argument("--eval-presets", nargs="+", default=None, choices=sorted(PRESETS), metavar="PRESET",
                    help="stage-1 evaluation of each fold model on fixed-seed held-out renders of these presets "
                         f"({', '.join(sorted(PRESETS))}; identical across runs) and on the real G1 train images -> "
                         "fold{f}_eval.json/.npz and score.json 'stage1_eval'; default none = as before")
    ap.add_argument("--eval-only", default=None, metavar="RUN",
                    help="no training: score data/het_cnn/RUN/fold{f}.pt (--folds, default all present) on "
                         "--eval-presets -> data/het_cnn/RUN/score_eval.json (score.json / parquets untouched)")
    ap.add_argument("--zoom", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="label-map zoom augmentation of cartoon renders, s ~ logUniform(LO, HI), e.g. 0.7 1.4 "
                         "(targets recomputed on the zoomed label map); default off")
    ap.add_argument("--p-zoom", type=float, default=None,
                    help="--zoom: probability per source image, 0-1 (default 0.5; only with --zoom)")
    ap.add_argument("--mosaic", type=float, default=0.0,
                    help="probability that a cartoon render composites two training images along a smooth random "
                         "mask or a straight line (targets from the composite label map), 0-1; default 0 = off")
    ap.add_argument("--mosaic-dlogn", type=float, default=None,
                    help="--mosaic: draw the second image only among training images with |log N_eff - log N_eff(j)| "
                         "<= D (nearest one if none), e.g. 0.5, so composites stay in the real het4 range; default "
                         "none = any training image")
    ap.add_argument("--head", default="grid4", choices=HEADS,
                    help="grid4 (default, HetNet as before) | fpn (FPNNet: stride-8/16/32 FPN at stride 16 with "
                         "block head + 16x16 cell head)")
    ap.add_argument("--fpn-ch", type=int, default=128, help="--head fpn: channels of the merged map")
    ap.add_argument("--lambda-cell", type=float, default=None,
                    help="--head fpn: weight of the 16-px cell loss (default 1.0 with fpn; grid4 has no cells)")
    ap.add_argument("--lambda-within", type=float, default=0.0,
                    help="weight of the within-image block-deviation loss (targets the shrinkage); default 0 = off")
    ap.add_argument("--screen", action="store_true",
                    help="design screening: predict only the held-out real clean images, no parquets; per-epoch "
                         "timing and het4-split stage-1 in the logs / fold json")
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
    if a.head == "fpn":
        a.lambda_cell = 1.0 if a.lambda_cell is None else a.lambda_cell
    else:
        if a.lambda_cell:
            ap.error("--lambda-cell needs --head fpn")
        a.lambda_cell = 0.0
    if a.zoom is not None and not (1 / 3 <= a.zoom[0] <= a.zoom[1]):
        ap.error("--zoom LO HI needs 1/3 <= LO <= HI")
    if a.p_zoom is not None and a.zoom is None:
        ap.error("--p-zoom needs --zoom")
    if a.zoom is not None:
        a.p_zoom = 0.5 if a.p_zoom is None else a.p_zoom
        if not 0 <= a.p_zoom <= 1:
            ap.error("--p-zoom must be in [0, 1]")
    if not 0 <= a.mosaic <= 1:
        ap.error("--mosaic must be in [0, 1]")
    if a.mosaic_dlogn is not None and (a.mosaic <= 0 or a.mosaic_dlogn < 0):
        ap.error("--mosaic-dlogn D needs --mosaic P > 0 and D >= 0")
    if a.eval_presets is not None:  # dedupe, keep the given order
        a.eval_presets = list(dict.fromkeys(a.eval_presets))
    if a.eval_only is not None and not a.eval_presets:
        ap.error("--eval-only RUN needs --eval-presets PRESET [PRESET ...]")
    a.prep_only = a.prep == "__BUILD__"
    if a.prep_only:
        a.prep = str(PREP_FP)
    return a


if __name__ == "__main__":
    main(parse())
