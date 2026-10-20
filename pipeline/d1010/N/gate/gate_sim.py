"""d1010 N/gate stage 2 (label-free): gate_lib.image_quantities on DEGRADED copies of the 267 clean TRAIN images.

Purpose: check, without any hardness label, whether the per-image reliability quantities predict how far the unchanged
pipeline's formula7 value on a degraded image moves from its value on the original clean image.
Degradation = d1010/C/mid-noise/calib_degrade.degrade (imported unchanged): dark grains lifted to kd x their gap,
contrast c, Gaussian blur b, Gaussian noise up to a target ic_noise.
Scenarios (fixed before any result was seen):
  mid : c 1.0, b 1.0, kd 0.5; targets / seeds exactly as C/mid-noise/fidelity.py (rng 20261010 over the 47 line-free
        mid TRAIN ic_noise values, seed 5000 + j)  -> V0 must reproduce fid_feats raw_s1 columns (asserted)
  lv  : c 1.0, b 0.0, kd 1.0 (noise only = clean preset at LV noise); targets rng 20261011 over the 21 LV TRAIN
        ic_noise values, seed 6000 + j
  g1  : c 1.0, b 1.0, kd 0.5; targets rng 20261012 over the 161 G1hi TRAIN ic_noise values (>= 12, ridge < 0.5),
        seed 7000 + j
Only train images are used. Labels are not read here.
  python gate_sim.py <scenario>   -> sim_<scenario>.parquet
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MID = '/mnt/project-files/work/hardness-cache/scripts/d1010/C/mid-noise'
sys.path.insert(0, MID)
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

import gate_lib as L
from calib_degrade import degrade  # noqa: E402

SC = {'mid': (1.0, 1.0, 0.5, 20261010, 5000), 'lv': (1.0, 0.0, 1.0, 20261011, 6000),
      'g1': (1.0, 1.0, 0.5, 20261012, 7000)}


def run(i, c, b, kd, t, seed):
    im8 = np.array(Image.open(f'{L.REPO}/data/train/{i}.png'))
    d = degrade(im8, c, b, t, seed, i=i, kd=kd)
    q = L.image_quantities(d.astype(np.float32))
    q['ID'] = i
    q['target_noise'] = float(t)
    return q


if __name__ == '__main__':
    sc = sys.argv[1]
    out = f'{HERE}/sim_{sc}.parquet'
    assert not os.path.exists(out), f'refusing to overwrite {out}'
    c, b, kd, rs, s0 = SC[sc]
    v3 = pd.read_parquet(L.REPO + '/data/features_v3.parquet').set_index('ID')
    tr = pd.read_csv(L.REPO + '/data/train.csv', usecols=['ID'])
    x = v3.loc[tr.ID]
    nz, rd = x.ic_noise, x.ic_ridge_snr
    pool = {'mid': x[(nz >= 9.5) & (nz < 12) & (rd < 0.5)].ic_noise.values,
            'lv': x[(nz >= 9.5) & (nz < 12) & (rd >= 0.5)].ic_noise.values,
            'g1': x[(nz >= 12) & (rd < 0.5)].ic_noise.values}[sc]
    ids = pd.read_parquet(L.CH + 'feats_C_train.parquet').index.tolist()
    assert len(ids) == 267
    tgt = np.random.default_rng(rs).choice(pool, len(ids), replace=True)
    t0 = time.time()
    res = Parallel(n_jobs=2)(delayed(run)(i, c, b, kd, float(t), s0 + j) for j, (i, t) in enumerate(zip(ids, tgt)))
    Q = pd.DataFrame(res).set_index('ID')
    if sc == 'mid':   # reproduction of the C mid-noise degraded raw_s1 features
        ref = pd.read_parquet(MID + '/fid_feats_c1.0_b1.0_kd0.5.parquet')
        for t in L.TERMS:
            dmax = float(np.nanmax(np.abs(Q[f'V0__{t}'].values - ref.loc[Q.index, f'raw_s1__{t}'].values)))
            print('repro mid raw_s1', t, 'max abs diff', dmax)
            assert dmax < 1e-9, t
    Q.to_parquet(out)
    print(sc, Q.shape, 'sec', round(time.time() - t0, 1))
