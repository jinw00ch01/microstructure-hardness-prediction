"""Degradation simulator for the DACON hardness images (d1011 G / forensics; label-free, train-only).

degrade(clean_img8, params=None, seed=0, prior='G1hi') turns a CLEAN-preset image (raw ic_noise < 9.5, boundary lines
visible) into a look-alike of the noisy preset (G1hi: ic_noise >= 12, line-free; midLF: 9.5-12 line-free, noisy-preset
part). No label is used anywhere; the priors below were fitted on statistics of real TRAIN images only (see REPORT.txt).

Scene model (decompose): the clean image is split into
  - a watershed label map ws (src.het_blocks.segment, sigma 1.0) and a per-grain robust grey relative to a quadratic
    shading surface S (sn = grain grey / S; 'dark' = second phase, sn < 0.93),
  - a boundary-line band (the watershed boundary pixels, 2 px wide) with a per-image line depth,
  - pores as hard masks (half-depth contour of the NLM-denoised image) with a flat core level,
  - the image's mean grey.
Render (render): S * (1 + dev'[ws]) with dev' = c_d * dev (dark grains) or e * dev (matrix grains) + per-grain jitter;
lines x line_k; pores at their core level x pore_k; mean grey kept; illumination (quadratic + smooth random field);
Gaussian PSF of sigma blur; additive white Gaussian noise of sd noise; round and clip to uint8.
"""
import numpy as np
import cv2
from scipy import ndimage as ndi

REPO = '/home/claude/microstructure-hardness-prediction'
_YY, _XX = np.indices((256, 256))
_XN, _YN = _XX / 256 - 0.5, _YY / 256 - 0.5

# ---------------------------------------------------------------------------------------------- fixed parameter priors
# Fitted label-free on real TRAIN images (f3 inference + f4 classifier-guided refinement; REPORT.txt section 4).
# Each entry: ('u', lo, hi) uniform, ('n', mean, sd, lo, hi) truncated normal, ('emp', values) empirical draw.
PRIORS = {
    # placeholders, overwritten at import from priors.json when present (written by f4 / f5)
}


B0 = 1.0  # intrinsic PSF sigma of the clean preset (pore edge-spread; REPORT.txt section 2); den-derived maps carry it


def _boundary(ws):
    bd = np.zeros(ws.shape, bool)
    dx, dy = ws[:, 1:] != ws[:, :-1], ws[1:] != ws[:-1]
    bd[:, 1:] |= dx
    bd[:, :-1] |= dx
    bd[1:] |= dy
    bd[:-1] |= dy
    return bd


def _shading_fit(cx, cy, area, v):
    """src.het_cnn.shading_fit / src.het_blocks.shading_norm algorithm on per-grain values."""
    x, y = cx / 256 - 0.5, cy / 256 - 0.5
    X = np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], 1)
    w = area.astype(float)
    keep = v > 0.9 * np.median(v[area > 30])
    beta = None
    for _ in range(4):
        W = np.sqrt(w[keep])
        beta, *_ = np.linalg.lstsq(X[keep] * W[:, None], v[keep] * W, rcond=None)
        fit = X @ beta
        keep = v / fit > 0.93
    return v / (X @ beta), beta


def _surface(beta):
    return (beta[0] + beta[1] * _XN + beta[2] * _YN + beta[3] * _XN ** 2 + beta[4] * _YN ** 2
            + beta[5] * _XN * _YN).astype(np.float64)


def decompose(im8):
    """Clean image -> scene dict (see module docstring). ~0.3 s per image."""
    import sys
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from skimage import restoration, morphology
    from src import het_blocks as hb
    im = im8.astype(np.float32)
    sm, ws = hb.segment(im)
    g = hb.grain_table(sm, ws)
    pore_g = ((hb.shading_norm(g) < 0.68) & (g.area < 600)).values
    L = int(ws.max()) + 1
    is_pore = np.zeros(L, bool)
    is_pore[g.label.values[pore_g]] = True
    bd = _boundary(ws)
    dist = ndi.distance_transform_edt(~bd)
    lab = g.label.values
    c3 = np.bincount(ws[dist >= 3], minlength=L)[lab]
    c2 = np.bincount(ws[dist >= 2], minlength=L)[lab]
    m3 = np.asarray(ndi.median(im, np.where(dist >= 3, ws, 0), lab), float)
    m2 = np.asarray(ndi.median(im, np.where(dist >= 2, ws, 0), lab), float)
    m0 = np.asarray(ndi.median(im, ws, lab), float)
    rob = np.where(c3 >= 12, m3, np.where(c2 >= 4, m2, m0))
    sn_rob, beta = _shading_fit(g.cx.values, g.cy.values, g.area.values, rob)
    sn_tab = np.ones(L)
    sn_tab[lab] = sn_rob
    dark = np.zeros(L, bool)
    dark[lab] = (sn_rob < 0.93) & ~is_pore[lab]
    S = _surface(beta)
    sig0 = float(restoration.estimate_sigma(im))
    den = cv2.fastNlMeansDenoising(im8, None, h=float(np.clip(sig0, 2.0, 30.0)), templateWindowSize=5,
                                   searchWindowSize=21).astype(np.float64)
    return _finish(ws, S, sn_tab, dark, is_pore, den, float(im.mean()), sig0, beta, im8=im8)


def _finish(ws, S, sn_tab, dark, is_pore, den, mean_grey, sig0, beta, im8=None):
    """Pore and line deficit maps from the denoised image; shared by decompose and from_prep.
    base grain level g = 1 + dev (pore grains set to the matrix level, dev 0); deficit d = den / S - G_B0(g).
    pore map: d inside the pore regions (very dark compact blobs of den / S < 0.65 and het_blocks pore grains, dilated
    by 3 px); line map: min(d, 0) within 2 px of a watershed boundary, outside the pore regions."""
    from skimage import morphology
    ws = ws.astype(np.int64)
    sn_tab = np.asarray(sn_tab, np.float64).copy()
    dev = sn_tab - 1.0
    dev[np.asarray(is_pore, bool)] = 0.0
    rel = den / S
    g1 = cv2.GaussianBlur(rel, (0, 0), 1.0)
    blob = ndi.binary_opening(g1 < 0.65, structure=morphology.disk(2)) | np.asarray(is_pore, bool)[ws]
    preg = ndi.binary_dilation(blob, iterations=3)
    base0 = cv2.GaussianBlur(1.0 + dev[ws], (0, 0), B0, borderType=cv2.BORDER_REFLECT)
    d = rel - base0  # deficit against the B0-blurred mosaic: only true lines / pores remain
    bd = _boundary(ws)
    dist = ndi.distance_transform_edt(~bd)
    pore_map = np.where(preg, d, 0.0)
    # line: mean raw deficit against the B0-blurred mosaic by distance class to the watershed boundary band (0 = band
    # pixel, 1, 2, 3); the generator's lines look uniform, so a per-image mean profile is used (no NLM softening)
    dcls = np.minimum(np.round(dist).astype(int), 4)
    ok = ~ndi.binary_dilation(preg, iterations=2)
    dr = im8.astype(np.float64) / S - base0 if im8 is not None else d
    prof = np.zeros(5)
    for k in range(4):
        m = ok & (dcls == k)
        if m.sum() > 50:
            prof[k] = float(np.mean(dr[m]))
    prof = prof - prof[3]  # deficit relative to the grain interior at 3 px
    prof[3:] = 0.0
    # continuous distance -> interpolated profile (avoids the jagged rounded-distance classes)
    line_map = np.where(preg, 0.0, np.interp(dist, np.arange(5), prof))
    # 'raw' alternative: the actual raw deficit in the band (dist <= 2.5), smoothed inside the band (normalised
    # convolution, sigma 0.6) -- keeps each line's own depth and sharpness; carries a little of the clean image's noise
    if im8 is not None:
        band = (dist <= 2.5) & ok
        w = cv2.GaussianBlur(band.astype(np.float64), (0, 0), 0.6)
        v = cv2.GaussianBlur(np.where(band, dr, 0.0), (0, 0), 0.6)
        interior = np.where((dist >= 3) & ok, dr, np.nan)
        lvl = np.nanmedian(interior) if np.isfinite(interior).any() else 0.0
        line_raw = np.where(band, v / np.maximum(w, 1e-6) - lvl, 0.0)
    else:
        line_raw = line_map
    # 'den' alternative: per-pixel deficit of the NLM-denoised image (own depth per boundary, smooth, no staircase)
    line_den = np.where((dist <= 2.5) & ok, np.minimum(d, 0.0), 0.0)
    return dict(ws=ws, S=S, dev=dev, dark=np.asarray(dark, bool) & ~np.asarray(is_pore, bool), is_pore=is_pore,
                pore_map=pore_map, line_map=line_map, line_raw=line_raw, line_den=line_den, line_prof=prof[:4], preg=preg, bd=bd,
                mean=mean_grey,
                sig0=sig0, beta=beta)


_PREP = {}


def from_prep(j):
    """Scene of clean train image j of src.het_cnn's cached prep (data/het_cnn/prep.npz: the 267 clean train images,
    same segmentation / robust greys / shading as decompose). Cached per process."""
    if ('scene', j) in _PREP:
        return _PREP[('scene', j)]
    sc = _from_prep(j)
    _PREP[('scene', j)] = sc
    return sc


def _from_prep(j):
    if 'P' not in _PREP:
        _PREP['P'] = np.load(REPO + '/data/het_cnn/prep.npz', allow_pickle=True)
    P = _PREP['P']
    ws = P['ws'][j].astype(np.int64)
    beta = P['beta'][j]
    S = _surface(beta)
    sn = P['sn_tab'][j].astype(np.float64)
    return _finish(ws, S, sn, P['dark'][j], P['pore'][j], P['den'][j].astype(np.float64),
                   float(P['im8'][j].mean()), float(P['sig0'][j]), beta, im8=P['im8'][j])


def prep_ids():
    if 'P' not in _PREP:
        _PREP['P'] = np.load(REPO + '/data/het_cnn/prep.npz', allow_pickle=True)
    return [str(i) for i in _PREP['P']['ids']]


DEFAULT = dict(c_d=1.0, c_d_sd=0.0, e=1.0, jit=0.0, jit_d=0.0, jit_m=0.0, line_k=1.0, line_s=0.0, line_p=1.0,
               line_dark=0.0, line_mode='mean', pore_k=1.0, blur=1.0, q_amp=0.0, r_amp=0.0, noise=6.7, psf='gauss',
               keep_mean=True)


def _blur(img, s, psf):
    if s <= 0.05:
        return img
    if psf == 'gauss':
        return cv2.GaussianBlur(img, (0, 0), s, borderType=cv2.BORDER_REFLECT)
    k = max(int(round(s * np.sqrt(12))), 1)
    return cv2.blur(img, (k, k), borderType=cv2.BORDER_REFLECT)


def render(scene, p, rng):
    """Render a scene with parameters p (keys of DEFAULT). Returns uint8 (256, 256).
    The hard-edged grain mosaic is blurred with sigma blur; the den-derived line / pore maps already carry the clean
    preset's PSF (B0) and get only the extra sqrt(blur^2 - B0^2)."""
    q = dict(DEFAULT, **p)
    ws, S, dev, dark = scene['ws'], scene['S'], scene['dev'], scene['dark']
    if q['c_d_sd'] > 0:  # per-grain dark-phase contrast factor: c_d_i = clip(c_d + c_d_sd * N(0, 1), 0, 1.3)
        cdi = np.clip(q['c_d'] + q['c_d_sd'] * rng.normal(0, 1, len(dev)), 0.0, 1.3)
    else:
        cdi = q['c_d']
    nd = np.where(dark, cdi * dev, q['e'] * dev)
    if q['jit'] > 0 or q['jit_d'] > 0 or q['jit_m'] > 0:  # per-grain grey jitter (all / dark-only / matrix-only)
        sdj = np.sqrt(q['jit'] ** 2 + np.where(dark, q['jit_d'], q['jit_m']) ** 2)
        nd = nd + rng.normal(0, 1, len(nd)) * sdj * ~np.asarray(scene['is_pore'], bool)
    base = S * (1.0 + nd[ws])
    bx = float(np.sqrt(max(q['blur'] ** 2 - B0 ** 2, 0.0)))
    img = _blur(base, q['blur'], q['psf']) + _blur(S * q['pore_k'] * scene['pore_map'], bx, q['psf'])
    if q['line_k'] != 0:  # line_s: extra Gaussian spread of the boundary lines only (same integrated darkness)
        bl = float(np.sqrt(bx ** 2 + q['line_s'] ** 2))
        lm = scene[{'raw': 'line_raw', 'den': 'line_den'}.get(q['line_mode'], 'line_map')]
        if q['line_p'] < 1.0 or q['line_dark'] > 0:  # sparse lines: a grain keeps its side of the boundary line with
            # probability line_p, or always when it is a dark grain and line_dark = 1
            keep = (rng.random(len(dev)) < q['line_p']) | (dark & (q['line_dark'] > 0))
            lm = lm * keep[ws]
        img = img + _blur(S * q['line_k'] * lm, bl, q['psf'])
    if q['keep_mean']:
        img = img + (scene['mean'] - img.mean())
    if q['q_amp'] > 0:  # extra quadratic illumination: random quadratic surface with unit sd, x amplitude
        c = rng.normal(0, 1, 5)
        Q = c[0] * _XN + c[1] * _YN + c[2] * (_XN ** 2 - 1 / 12) + c[3] * (_YN ** 2 - 1 / 12) + c[4] * _XN * _YN
        Q = Q / (Q.std() + 1e-12)
        img = img * (1.0 + q['q_amp'] * Q)
    if q['r_amp'] > 0:
        fld = cv2.GaussianBlur(rng.standard_normal((256, 256)), (0, 0), 40.0)
        fld /= fld.std() + 1e-12
        img = img * (1.0 + q['r_amp'] * fld)
    img = img + rng.normal(0, q['noise'], img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------------------------------------- priors / API
def _draw(spec, rng):
    kind = spec[0]
    if kind == 'u':
        return float(rng.uniform(spec[1], spec[2]))
    if kind == 'n':
        for _ in range(100):
            v = rng.normal(spec[1], spec[2])
            if spec[3] <= v <= spec[4]:
                return float(v)
        return float(np.clip(spec[1], spec[3], spec[4]))
    if kind == 'emp':
        return float(rng.choice(spec[1]))
    if kind == 'c':
        return spec[1]
    raise ValueError(spec)


def draw_params(rng, prior='G1hi'):
    pr = PRIORS[prior]
    return {k: _draw(v, rng) for k, v in pr.items()}


def degrade(clean_img8, params=None, seed=0, prior='G1hi', scene=None):
    """Public entry point. clean_img8: uint8 (256, 256) clean-preset image; params: dict overriding the prior draw;
    prior: 'G1hi' or 'midLF'. Returns (uint8 image, params used)."""
    rng = np.random.default_rng(seed)
    p = draw_params(rng, prior) if prior is not None else {}
    if params:
        p.update(params)
    sc = scene if scene is not None else decompose(clean_img8)
    return render(sc, p, rng), p


def _load_priors():
    import json
    import os
    fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'priors.json')
    if os.path.exists(fp):
        d = json.load(open(fp))
        for k, v in d.items():
            PRIORS[k] = {kk: tuple(vv) if not (isinstance(vv, list) and vv and vv[0] == 'emp') else ('emp', vv[1])
                         for kk, vv in v.items()}


_load_priors()
