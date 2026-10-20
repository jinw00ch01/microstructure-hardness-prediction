"""d1010 N/gate stage 1 (label-free): gate_lib.image_quantities on every train / test image.

  python gate_real.py <split> <part> <nparts>   -> q_<split>_<part>of<nparts>.parquet  (2 workers)
No label is read (train.csv usecols=['ID']). Test images are only segmented; nothing is fitted across images.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

import gate_lib as L

REPO = L.REPO


def run(split, i, check):
    im = np.array(Image.open(f'{REPO}/data/{split}/{i}.png')).astype(np.float32)
    q = L.image_quantities(im, check_v0=check)
    q['ID'] = i
    return q


if __name__ == '__main__':
    split, part, nparts = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    out = f'{HERE}/q_{split}_{part}of{nparts}.parquet'
    assert not os.path.exists(out), f'refusing to overwrite {out}'
    csv = 'train.csv' if split == 'train' else 'sample_submission.csv'
    ids = pd.read_csv(f'{REPO}/data/{csv}', usecols=['ID']).ID.tolist()
    ids = ids[part::nparts]
    t0 = time.time()
    res = Parallel(n_jobs=2)(delayed(run)(split, i, j < 4) for j, i in enumerate(ids))
    Q = pd.DataFrame(res).set_index('ID')
    Q.to_parquet(out)
    print(split, part, nparts, Q.shape, 'sec', round(time.time() - t0, 1))
