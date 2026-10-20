"""Label-free, train-only: find a degradation (contrast c, blur b, noise to a target ic_noise) that turns clean TRAIN
images into images whose features_v3 quality statistics look like the line-free mid-noise TRAIN rows
(9.5 <= ic_noise < 12, ic_ridge_snr < 0.5). Only image statistics are compared (no labels, no test images).
Output: calib_degrade.json (per-combo medians + distance; chosen combo = smallest distance).
kd < 1 additionally lifts each dark grain (clean seg of that train image) to kd x its gap below the matrix shading
(the noisy preset compresses dark-phase contrast); added after the first 9 combos left a gmm_sep gap.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import json
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy import ndimage as ndi
from skimage import restoration

REPO = '/home/claude/microstructure-hardness-prediction'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
import src.features as sf  # noqa: E402

STATS = ['ic_noise', 'ic_ridge_snr', 'ic_spec_slope', 'ic_spec_32_48', 'ic_gmm_sep', 'ic_std', 'ic_acg_ac8_par']


CH = '/mnt/project-files/work/hardness-cache/scripts/d1009/J/cleanhead/'
_G = {}


def dark_shift_map(i):
    """Per-pixel shift that brings each dark grain (clean seg, gn < per-image GMM thr) up to its matrix shading level."""
    if not _G:
        _G['g'] = pd.read_parquet(CH + 'grains_C_train.parquet')
        _G['w'] = np.load(CH + 'ws_C_train.npz')
        _G['f'] = pd.read_parquet(CH + 'feats_C_train.parquet')
    g = _G['g'][_G['g'].ID == i]
    ws = _G['w'][i].astype(np.int32)
    thr = float(_G['f'].loc[i, 'thr'])
    pore = (g.gn < 0.68) & (g.area < 600)
    dk = g[(~pore) & (g.gn < thr)]
    lut = np.zeros(ws.max() + 1)
    lut[dk.label.values] = dk['mean'].values * (1 / dk.gn.values - 1)
    return lut[ws]


def degrade(im8, c, b, target, seed, i=None, kd=1.0):
    x = im8.astype(np.float64)
    if kd < 1.0:
        x = x + (1 - kd) * dark_shift_map(i)
    x = x.mean() + c * (x - x.mean())
    if b > 0:
        x = ndi.gaussian_filter(x, b)
    n1 = float(restoration.estimate_sigma(x))
    s = np.sqrt(max(target ** 2 - n1 ** 2, 0.0))
    rng = np.random.default_rng(seed)
    x = x + rng.normal(0, s, x.shape)
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def one(i, c, b, target, seed, kd=1.0):
    im8 = np.array(Image.open(f'{REPO}/data/train/{i}.png'))
    d = degrade(im8, c, b, target, seed, i=i, kd=kd)
    f = sf.extract_v3(i, img=d)
    return {k: f[k] for k in STATS}


if __name__ == '__main__':
    v3 = pd.read_parquet(REPO + '/data/features_v3.parquet').set_index('ID')
    tr = pd.read_csv(REPO + '/data/train.csv')
    x = v3.loc[tr.ID]
    clean = x[x.ic_noise < 9.5].index.values
    midlf = x[(x.ic_noise >= 9.5) & (x.ic_noise < 12) & (x.ic_ridge_snr < 0.5)]
    ref_med = midlf[STATS].median()
    ref_iqr = (x[STATS].quantile(.75) - x[STATS].quantile(.25))
    rng = np.random.default_rng(20261010)
    pick = rng.choice(clean, 30, replace=False)
    combos = [(c, b, 1.0) for c in (1.0, 0.8, 0.65) for b in (0.0, 0.6, 1.0)]
    combos += [(c, 1.0, kd) for c in (1.0, 0.8) for kd in (0.7, 0.5)]   # added after the first 9: gmm_sep gap
    out = {'ref_midLF_median': ref_med.round(4).to_dict(), 'clean_median': x.loc[clean, STATS].median().round(4).to_dict(),
           'combos': {}}
    for c, b, kd in combos:
        R = pd.DataFrame(Parallel(n_jobs=3)(delayed(one)(i, c, b, 10.7, 1000 + j, kd) for j, i in enumerate(pick)))
        med = R.median()
        dist = float(np.sqrt(np.mean(((med - ref_med) / ref_iqr)[STATS[1:]] ** 2)))
        out['combos'][f'c{c}_b{b}_kd{kd}'] = dict(median=med.round(4).to_dict(), dist=dist)
        print(f'c={c} b={b} kd={kd} dist={dist:.3f}', med.round(3).to_dict(), flush=True)
    best = min(out['combos'], key=lambda k: out['combos'][k]['dist'])
    out['chosen'] = best
    print('ref midLF', ref_med.round(3).to_dict())
    print('chosen', best)
    json.dump(out, open(HERE + '/calib_degrade.json', 'w'), indent=1)
