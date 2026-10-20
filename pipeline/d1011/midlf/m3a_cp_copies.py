"""d1011 midlf M3a (label-free; no label is read; train images only).
'Clean-preset + noise' copies at the midLF noise levels: the 267 clean TRAIN images degraded by NOISE ONLY
(d1010/C/mid-noise calib_degrade.degrade with c 1.0, blur 0.0, kd 1.0), with exactly the noise targets and seeds of the
midLF-look copies (fidelity.py / gate_sim 'mid' / f6: default_rng(20261010).choice(47 midLF train ic_noise values),
seed 5000 + j). So each source image has two copies that differ only in blur 1.0 + dark-gap compression kd 0.5.
Per copy: src.features.extract_v3 (134 ic_ columns, as f6) and N/gate gate_lib.image_quantities (clean pipeline V0 plus
4 segmentation variants, as gate_sim).
python m3a_cp_copies.py -> m3a_cp_copies.parquet  (2 workers, BLAS threads 1)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import sys
sys.dont_write_bytecode = True
import time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from PIL import Image

REPO = '/home/claude/microstructure-hardness-prediction'
CACHE = '/mnt/project-files/work/hardness-cache'
MID = CACHE + '/scripts/d1010/C/mid-noise'
GATE = CACHE + '/scripts/d1010/N/gate'
CH = CACHE + '/scripts/d1009/J/cleanhead/'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE + '/m3a_cp_copies.parquet'


def run(i, t, seed):
    for p in (REPO, MID, GATE):
        if p not in sys.path:
            sys.path.insert(0, p)
    from calib_degrade import degrade
    from src.features import extract_v3
    import gate_lib as L
    im8 = np.array(Image.open(f'{REPO}/data/train/{i}.png'))
    d = degrade(im8, 1.0, 0.0, t, seed, i=i, kd=1.0)
    v = extract_v3('x', img=d)
    v.pop('ID', None)
    v = {k: v[k] for k in v if k.startswith('ic_')}
    q = L.image_quantities(d.astype(np.float32))
    v.update(q)
    v['ID'] = i
    v['target_noise'] = float(t)
    return v


if __name__ == '__main__':
    assert not os.path.exists(OUT), 'refusing to overwrite'
    v3 = pd.read_parquet(REPO + '/data/features_v3.parquet').set_index('ID')
    tr = pd.read_csv(REPO + '/data/train.csv', usecols=['ID'])
    x = v3.loc[tr.ID]
    midlf_noise = x[(x.ic_noise >= 9.5) & (x.ic_noise < 12) & (x.ic_ridge_snr < 0.5)].ic_noise.values
    assert len(midlf_noise) == 47
    ids = pd.read_parquet(CH + 'feats_C_train.parquet').index.tolist()
    assert len(ids) == 267
    tgt = np.random.default_rng(20261010).choice(midlf_noise, len(ids), replace=True)
    # same targets as the midLF-look copies (gate_sim 'mid' stored them)
    sm = pd.read_parquet(GATE + '/sim_mid.parquet')
    assert np.allclose(sm.loc[ids, 'target_noise'].values, tgt)
    t0 = time.time()
    res = Parallel(n_jobs=2)(delayed(run)(i, float(t), 5000 + j) for j, (i, t) in enumerate(zip(ids, tgt)))
    Q = pd.DataFrame(res).set_index('ID')
    Q.to_parquet(OUT)
    print(Q.shape, 'sec', round(time.time() - t0, 1))
