"""Label-free mid-noise segmentation features for TRAIN and TEST rows with 9.5 <= raw ic_noise < 12 (features_v3).

All seven mid_seg variants are computed per image; columns '<variant>__<feature>'. 'linefree' = ic_ridge_snr < 0.5
(the evaluated band); line-visible band rows are kept for description only. Nothing is fitted across images and
no labels are read. Output: mid_feats_{train,test}.parquet (index ID).
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mid_seg as M  # noqa: E402

REPO = '/home/claude/microstructure-hardness-prediction'
KEEP = M.F7 + ['fd_area_P1', 'fd_area_P2', 'N_all', 'N_in', 'N_bd', 'pore_n', 'thr', 'gmm_gap', 'thr_fallback']


def one(split, i):
    im8 = np.array(Image.open(f'{REPO}/data/{split}/{i}.png'))
    out = {}
    for v in M.VARIANTS:
        f = M.image_feats(im8, v)
        out.update({f'{v}__{k}': float(f.get(k, np.nan)) for k in KEEP})
    return out


if __name__ == '__main__':
    v3 = pd.read_parquet(REPO + '/data/features_v3.parquet').set_index('ID')
    for split, csv in (('train', 'train.csv'), ('test', 'sample_submission.csv')):
        ids = pd.read_csv(f'{REPO}/data/{csv}').ID
        x = v3.loc[ids]
        band = x[(x.ic_noise >= 9.5) & (x.ic_noise < 12)]
        R = pd.DataFrame(Parallel(n_jobs=3)(delayed(one)(split, i) for i in band.index),
                         index=pd.Index(band.index, name='ID'))
        R['ic_noise'] = band.ic_noise.values
        R['ic_ridge_snr'] = band.ic_ridge_snr.values
        R['linefree'] = (band.ic_ridge_snr.values < 0.5).astype(float)
        R.to_parquet(f'{HERE}/mid_feats_{split}.parquet')
        print(split, R.shape, 'linefree', int(R.linefree.sum()))
