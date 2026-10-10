"""degrade_sim: calibrated clean -> noisy-preset degradation simulator (d1011 G / forensics; label-free, train-only).

Turns a CLEAN-preset TRAIN image (estimated white-noise sd ~4-10.5, boundary lines visible) into a look-alike of the
NOISY preset: 'G1hi' (noise sd 12-20.5, the line-free high-noise group) or 'midLF' (noise sd 9.6-12.3, the
noisy-preset part of the mid / line-free group, 'midLF_NP'). The priors below are FIXED (embedded here, not read from a
file). They were fitted only on statistics of real TRAIN images (no label, no test image): ABC on ~70 image statistics,
quadratic-surrogate moment matching, then Bayesian optimisation of a cross-validated real-vs-simulated domain
classifier (f5 / f8 / f9 in this folder; REPORT.txt sections 4-5).

    import sys; sys.path.insert(0, '/mnt/project-files/work/hardness-cache/scripts/d1011/G/forensics')
    import degrade_sim as D
    img, p = D.degrade(clean_im8, prior='G1hi', seed=0)            # any clean-preset uint8 (256, 256) image
    img, p = D.degrade_train(j, prior='midLF', seed=0)            # j-th of the 267 cached clean train images (fast)
    img, p = D.degrade(clean_im8, prior='G1hi', params={'noise': 15.0})   # override any drawn parameter
    python degrade_sim.py in.png out.png [G1hi|midLF|noisy] [seed]

What one call does (fx_sim.render; the scene comes from fx_sim.decompose = src.het_blocks watershed segmentation,
per-grain robust grey over a quadratic shading surface S, NLM-denoised pore and boundary-line deficit maps):
  dark (second-phase) grains: grey offset dev x c_d_i with a per-grain factor c_d_i = clip(c_d + c_d_sd N, 0, 1.3)
      -> the phase contrast falls to ~0.5 of the clean render (d' ~5 -> ~2), unevenly per grain;
  matrix grains: e x dev + N(0, jit) per grain -> the matrix grain greys are re-spread (sd 0.034 -> ~0.048 of S),
      largely independently of the clean grain grey;
  boundary lines: each grain keeps its side of the line with probability line_p (~0.5), at line_k (~1.1) of the clean
      line profile, spread by an extra Gaussian of sd line_s (~1.1 px) -> weak, sparse, soft lines (NOT removed);
  pores: same masks, depth x pore_k (~0.86);
  PSF: Gaussian, total sigma blur (~1.23 px; clean preset B0 = 1.0 -> ~0.7 px extra in quadrature);
  illumination: extra random quadratic shading (relative amplitude U(0, 0.043)) and a faint smooth field;
  noise: additive white Gaussian, signal-independent, sd drawn from the empirical TRAIN distribution of the target.
Every structural parameter has per-image spread (truncated normal); none depends on the noise level (in real noisy
TRAIN images no structural statistic correlates with the noise sd, |Spearman| < 0.1).

Validation (real noisy TRAIN images vs one simulated copy of each of the 267 clean TRAIN images, 5-fold
StratifiedGroupKFold, sims grouped by clean source; f6_domain.py; target was AUC < 0.65):
                                 G1hi (161 real)   midLF_NP (40 real; AUC sd ~0.03)
  ~70 forensic stats, logistic        0.786            0.767
  ~70 forensic stats, LightGBM        0.802            0.782
  + d1010/K imstats, logistic/LGBM    0.811 / 0.767    0.785 / 0.786
  frozen resnet18 GAP+std, logistic   0.813            0.885
  small CNN from scratch (D4 crops)   0.487            0.561
  (CNN sanity: the crude d1010/C-style copy recipe on G1hi -> CNN 0.866, resnet18 0.966)
  fidelity floor of the scene model (clean image vs its own clean-parameter re-render, both at G1hi noise): 0.85-0.90
So the copies fool a from-scratch CNN but not hand-crafted statistics or ImageNet features; the residual is diffuse
(grey-level kurtosis, ridge-filter quantiles at sigma 2.5, high-frequency PSD share, edge width) and is bounded below by
the decomposition fidelity of the clean scene, not by a mis-set degradation parameter.

Rules: uses only TRAIN images and the repository's own code; never call it on test images for training purposes.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import fx_sim as _X  # noqa: E402  (scene decomposition + renderer)

# ------------------------------------------------------------------------------------------------ fixed priors
# empirical white-noise sd (PSD floor, fx_stats 'sn') of the real TRAIN target images
_SN_G1HI = [
    12.0, 12.22, 12.26, 12.27, 12.28, 12.3, 12.34, 12.37, 12.39, 12.45, 12.52, 12.58, 12.7, 12.8, 12.82, 12.84, 13.09,
    13.09, 13.11, 13.13, 13.14, 13.19, 13.27, 13.42, 13.44, 13.48, 13.55, 13.65, 13.66, 13.68, 13.69, 13.69, 13.72,
    13.8, 13.84, 13.95, 13.96, 13.97, 13.99, 14.14, 14.26, 14.37, 14.45, 14.56, 14.65, 14.82, 14.87, 15.03, 15.05,
    15.22, 15.28, 15.29, 15.3, 15.32, 15.47, 15.48, 15.52, 15.57, 15.58, 15.59, 15.67, 15.77, 15.79, 15.87, 15.91,
    15.95, 15.99, 16.02, 16.04, 16.06, 16.07, 16.09, 16.11, 16.11, 16.15, 16.24, 16.25, 16.28, 16.34, 16.35, 16.36,
    16.36, 16.36, 16.48, 16.51, 16.51, 16.56, 16.59, 16.6, 16.6, 16.64, 16.64, 16.72, 16.84, 16.85, 16.87, 16.9, 16.91,
    16.96, 17.0, 17.0, 17.15, 17.26, 17.26, 17.28, 17.35, 17.37, 17.38, 17.48, 17.49, 17.55, 17.62, 17.74, 17.75,
    17.77, 17.8, 17.83, 17.86, 17.86, 18.0, 18.03, 18.05, 18.05, 18.08, 18.11, 18.2, 18.21, 18.22, 18.23, 18.27, 18.3,
    18.32, 18.35, 18.36, 18.39, 18.52, 18.59, 18.6, 18.66, 18.73, 18.74, 18.75, 18.76, 18.92, 18.94, 19.14, 19.23,
    19.26, 19.29, 19.34, 19.35, 19.37, 19.51, 19.55, 19.58, 19.59, 19.67, 19.71, 19.9, 20.09, 20.28]
_SN_MIDLF = [
    9.63, 9.71, 9.89, 10.09, 10.11, 10.16, 10.17, 10.21, 10.28, 10.3, 10.36, 10.37, 10.39, 10.47, 10.6, 10.74, 10.78,
    10.83, 10.85, 10.87, 10.88, 10.88, 11.12, 11.14, 11.24, 11.32, 11.38, 11.48, 11.52, 11.54, 11.55, 11.64, 11.65,
    11.67, 11.74, 11.77, 11.95, 11.98, 12.02, 12.29]

# structure of the noisy preset (G1hi_v2 fit, priors.json; REPORT.txt section 3)
# ('n', mean, sd, lo, hi) truncated normal per image; ('c', v) constant; ('u', lo, hi) uniform; ('emp', values)
STRUCTURE = {
    'c_d': ('n', 0.4919, 0.12, 0.05, 1.2),      # dark-phase offset factor (mean over grains)
    'c_d_sd': ('c', 0.2567),                    # per-grain sd of the dark-phase factor
    'e': ('n', 0.5632, 0.15, 0.0, 2.0),         # matrix grain deviation factor
    'jit': ('n', 0.0434, 0.012, 0.0, 0.1),      # per-grain grey jitter (fraction of S)
    'line_k': ('n', 1.1248, 0.225, 0.0, 2.0),   # line depth factor where a line is kept
    'line_s': ('c', 1.1103),                    # extra line spread (px)
    'line_p': ('c', 0.517),                     # probability that a grain keeps its side of the boundary line
    'pore_k': ('n', 0.8642, 0.075, 0.6, 1.2),   # pore depth factor
    'blur': ('n', 1.2321, 0.15, 0.5, 2.5),      # total Gaussian PSF sigma (px); clean preset = 1.0
    'q_amp': ('u', 0.0, 0.0427),                # extra quadratic shading amplitude
    'r_amp': ('u', 0.0, 0.006),                 # smooth random illumination field amplitude
}
# midLF_NP refinement (f7 batches m1 / m2, priors.json 'midLF_v3'): a little more remaining phase contrast, less blur,
# more shading than G1hi (40 real rows only: weakly determined; the G1hi shape on midLF_NP is close, AUC +0.03)
STRUCTURE_MIDLF = dict(STRUCTURE, c_d=('n', 0.6, 0.12, 0.05, 1.2), blur=('n', 1.15, 0.15, 0.5, 2.5),
                       q_amp=('u', 0.0, 0.06), r_amp=('u', 0.0, 0.02))
PRIORS = {
    'G1hi': dict(STRUCTURE, noise=('emp', _SN_G1HI)),
    'midLF': dict(STRUCTURE_MIDLF, noise=('emp', _SN_MIDLF)),
    'noisy': dict(STRUCTURE, noise=('u', 10.0, 20.5)),   # whole noisy-preset noise range, G1hi structure
}
# install into fx_sim so fx_sim.draw_params / fx_sim.degrade see the same fixed priors under 'fixed_<name>'
for _k, _v in PRIORS.items():
    _X.PRIORS['fixed_' + _k] = _v


def draw_params(rng, prior='G1hi'):
    return {k: _X._draw(v, rng) for k, v in PRIORS[prior].items()}


def degrade(clean_img8, prior='G1hi', seed=0, params=None, scene=None):
    """clean_img8: uint8 (256, 256) clean-preset image. Returns (uint8 image, dict of parameters used).
    scene: optional fx_sim.decompose(clean_img8) result to reuse across several draws."""
    rng = np.random.default_rng(seed)
    p = draw_params(rng, prior)
    if params:
        p.update(params)
    sc = scene if scene is not None else _X.decompose(np.asarray(clean_img8, np.uint8))
    return _X.render(sc, p, rng), p


def degrade_train(j, prior='G1hi', seed=0, params=None):
    """Same, for the j-th of the 267 clean train images cached in data/het_cnn/prep.npz (ids: train_ids()).
    The cached per-grain tables are padded; they are cut to the image's labels so that the result is identical to
    degrade(<that PNG>, prior, seed, params) (up to float rounding: rarely 1 grey level at a pixel)."""
    sc = dict(_X.from_prep(j))
    n = int(sc['ws'].max()) + 1
    for k in ('dev', 'dark', 'is_pore'):
        sc[k] = np.asarray(sc[k])[:n]
    return degrade(None, prior, seed, params, scene=sc)


def train_ids():
    return _X.prep_ids()


if __name__ == '__main__':
    from PIL import Image
    src, dst = sys.argv[1], sys.argv[2]
    pr = sys.argv[3] if len(sys.argv) > 3 else 'G1hi'
    sd = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    im, used = degrade(np.array(Image.open(src).convert('L')), pr, sd)
    Image.fromarray(im).save(dst)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in used.items()})
