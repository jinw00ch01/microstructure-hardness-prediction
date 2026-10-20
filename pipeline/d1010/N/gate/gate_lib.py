"""d1010 N/gate: per-image, label-free measurement-stability quantities for the UNCHANGED cleanhead formula7 pipeline.

For one 8-bit image (array) this runs the cleanhead pipeline (d1009/J/cleanhead seg_c.py grain table + feats_c.feats,
imported unchanged) under five segmentation settings and one phase-threshold shift:
  V0 = (sigma 1.0, tf 0.6)  -> identical to src.het_blocks.segment (the unchanged pipeline; asserted by the callers)
  V1 = (0.8, 0.6), V2 = (1.2, 0.6), V3 = (1.0, 0.45), V4 = (1.0, 0.8)    (tf = factor on the Otsu ridge threshold)
Each variant re-derives its own per-image GMM phase threshold, pores, N_eff, het4 exactly as feats_c does.
On V0 only: fd_area with the phase threshold moved by -0.02 / +0.02 (gn units), and the GMM phase gap / fallback flag
(same GaussianMixture(2, random_state=0) call as feats_c.phase_thr; the threshold it implies is asserted equal).
Nothing is fitted across images; no label is read here.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation
from sklearn.mixture import GaussianMixture

REPO = '/home/claude/microstructure-hardness-prediction'
CH = '/mnt/project-files/work/hardness-cache/scripts/d1009/J/cleanhead/'
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if CH not in sys.path:
    sys.path.insert(0, CH)
from src.het_blocks import segment, shading_norm  # noqa: E402
import feats_c  # noqa: E402

TERMS = ['fd_area', 'pore_frac', 'logasp_aw', 'S_align', 'logN_eff', 'het4', 'bd_frac']
VARIANTS = {'V0': (1.0, 0.6), 'V1': (0.8, 0.6), 'V2': (1.2, 0.6), 'V3': (1.0, 0.45), 'V4': (1.0, 0.8)}
DTHR = 0.02


def seg(im, sig, tf):
    """src.het_blocks.segment with the Otsu factor exposed (copy of cleanhead seg_alt.seg)."""
    sm = ndi.gaussian_filter(im, sig)
    r = filters.sato(sm, sigmas=(sig, 1.5 * sig), black_ridges=True, mode="reflect")
    interior = ndi.binary_opening(~(r > filters.threshold_otsu(r) * tf), structure=morphology.disk(1))
    lab, _ = ndi.label(interior)
    small = np.bincount(lab.ravel()) < 6
    small[0] = False
    lab[small[lab]] = 0
    lab, _, _ = segmentation.relabel_sequential(lab)
    return sm, segmentation.watershed(r, markers=lab, connectivity=1)


def grain_table(sm, ws):
    """Copy of cleanhead seg_c.one's grain table."""
    rows = []
    for p in measure.regionprops(ws, intensity_image=sm):
        minr, minc, maxr, maxc = p.bbox
        rows.append(dict(label=p.label, area=p.area, cy=p.centroid[0], cx=p.centroid[1], mean=p.intensity_mean,
                         maj=p.axis_major_length, mnr=p.axis_minor_length, ori=p.orientation, perim=p.perimeter,
                         border=minr == 0 or minc == 0 or maxr == 256 or maxc == 256,
                         nb=int((minr == 0) + (minc == 0) + (maxr == 256) + (maxc == 256))))
    g = pd.DataFrame(rows)
    g['gn'] = shading_norm(g)
    return g


def gmm_gap(gp):
    big = gp[gp.area > 25]
    try:
        gm = GaussianMixture(2, random_state=0).fit(big.gn.values[:, None])
        mu = np.sort(gm.means_.ravel())
        return float(mu[1] - mu[0])
    except Exception:
        return np.nan


def terms_of(f):
    return dict(fd_area=float(f['fd_area']), pore_frac=float(f['pore_frac']), logasp_aw=float(f['logasp_aw']),
                S_align=float(f['S_align']), logN_eff=float(np.log(f['N_eff'])), het4=float(f['het4']),
                bd_frac=float(f['N_bd'] / f['N_all']))


def image_quantities(im, check_v0=False):
    """im: float32 2-D array (as seg_c reads it). Returns a flat dict."""
    out = {}
    for name, (sig, tf) in VARIANTS.items():
        sm, ws = seg(im, sig, tf)
        if name == 'V0' and check_v0:
            sm0, ws0 = segment(im)
            assert np.array_equal(ws, ws0) and np.array_equal(sm, sm0), 'V0 != src.het_blocks.segment'
        ws = ws.astype(np.int16).astype(np.int32)      # the int16 round trip of seg_c (ws_C npz)
        g = grain_table(sm, ws)
        f = feats_c.feats(g.reset_index(drop=True), ws)
        for k, v in terms_of(f).items():
            out[f'{name}__{k}'] = v
        out[f'{name}__thr'] = float(f['thr'])
        out[f'{name}__N_all'] = float(f['N_all'])
        pore = ((g.gn < 0.68) & (g.area < 600)).values
        gp = g[~pore]
        gap = gmm_gap(gp)
        out[f'{name}__gap'] = gap
        out[f'{name}__fallback'] = float(not (gap > 0.08))
        if name == 'V0':
            area = gp.area.values.astype(float)
            thr = float(f['thr'])
            for s, d in (('m', -DTHR), ('p', DTHR)):
                out[f'V0__fd_thr{s}'] = float(area[(gp.gn < thr + d).values].sum() / area.sum())
            # share of non-pore grain area whose gn lies within +-0.02 of the threshold (ambiguous grains)
            out['V0__amb_area'] = float(area[np.abs(gp.gn.values - thr) < DTHR].sum() / area.sum())
            out['V0__n_grains_raw'] = float(len(g))
    return out


def design(F):
    """formula7 design from a cleanhead feats frame (as l_test.design)."""
    D = pd.DataFrame(index=F.index)
    for c in ['fd_area', 'pore_frac', 'logasp_aw', 'S_align', 'het4']:
        D[c] = F[c].astype(float)
    D['logN_eff'] = np.log(F['N_eff'].astype(float))
    D['bd_frac'] = F['N_bd'].astype(float) / F['N_all'].astype(float)
    return D[TERMS]


def f7_apply(b, X):
    return np.column_stack([np.ones(len(X)), np.asarray(X, float)]) @ b


def reliability(Q, b):
    """Q: frame of image_quantities rows. b: formula7 coefficients (intercept first). Returns a frame of scores."""
    R = pd.DataFrame(index=Q.index)
    P = np.column_stack([f7_apply(b, Q[[f'{v}__{t}' for t in TERMS]].values) for v in VARIANTS])
    R['f7_V0'] = P[:, 0]
    for j, v in enumerate(VARIANTS):
        R[f'f7_{v}'] = P[:, j]
    R['sd_seg'] = P.std(axis=1)                              # SD of f7 over the 5 segmentations (ddof 0)
    R['rng_seg'] = P.max(axis=1) - P.min(axis=1)
    Xm = Q[[f'V0__{t}' for t in TERMS]].copy(); Xm.columns = TERMS
    Xp = Xm.copy()
    Xm['fd_area'] = Q['V0__fd_thrm'].values; Xp['fd_area'] = Q['V0__fd_thrp'].values
    R['d_thr'] = np.abs(f7_apply(b, Xp.values) - f7_apply(b, Xm.values)) / 2
    R['W'] = np.sqrt(R.sd_seg ** 2 + R.d_thr ** 2)
    R['gap'] = Q['V0__gap'].fillna(0.0).values
    R['fallback'] = Q['V0__fallback'].values
    R['nfb_var'] = Q[[f'{v}__fallback' for v in VARIANTS]].sum(axis=1).values   # variants that fell back (0..5)
    R['amb_area'] = Q['V0__amb_area'].values
    return R
