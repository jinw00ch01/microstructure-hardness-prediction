"""d1010 C residual-probe (post-hoc distillation #2, PREREG addendum 2): 5-segmentation averages, label-free.
Inputs: segalt_{train,test}.parquet (this dir, 4 alternative segmentations), cleanhead feats_C / rowcnt / colcnt (base).
  python build_avg5.py -> feats_avg5_{train,test}.parquet (ID index: bdf_avg5, ringall_avg5, lN_avg5)
"""
import os
import numpy as np, pandas as pd
CH = '/mnt/project-files/work/hardness-cache/scripts/d1009/J/cleanhead/'
HERE = os.path.dirname(os.path.abspath(__file__)) + '/'
for split in ('train', 'test'):
    F = pd.read_parquet(CH + f'feats_C_{split}.parquet'); ids = list(F.index)
    R = pd.read_parquet(CH + f'rowcnt_{split}.parquet').loc[ids]; C = pd.read_parquet(CH + f'colcnt_{split}.parquet').loc[ids]
    R.columns = R.columns.astype(int); C.columns = C.columns.astype(int)
    Rv, Cv = R[list(range(256))].values.astype(float), C[list(range(256))].values.astype(float)
    ring = np.mean([(Rv[:, k] + Rv[:, 255 - k] + Cv[:, k] + Cv[:, 255 - k]) / 4 for k in range(1, 5)], axis=0)
    allm = (Rv.mean(1) + Cv.mean(1)) / 2
    base = pd.DataFrame({'bdf': (F.N_bd / F.N_all).values, 'lra': np.log(ring / allm), 'lN': np.log(F.N_all.values.astype(float))},
                        index=ids)
    L = pd.read_parquet(HERE + f'segalt_{split}.parquet')
    L['bdf'] = L.N_bd / L.N_all; L['lra'] = np.log(L.ring / L.allm); L['lN'] = np.log(L.N_all.astype(float))
    alt = [L[(L.sig == s) & (L.tf == t)].set_index('ID').loc[ids, ['bdf', 'lra', 'lN']] for s, t in
           [(0.8, 0.6), (1.3, 0.6), (1.0, 0.45), (1.0, 0.8)]]
    M = (base + sum(alt)) / 5
    D = pd.DataFrame({'bdf_avg5': M.bdf, 'ringall_avg5': M.lra, 'lN_avg5': M.lN}, index=pd.Index(ids, name='ID'))
    assert np.isfinite(D.values).all() and len(D) == len(ids)
    D.to_parquet(HERE + f'feats_avg5_{split}.parquet')
    print(split, D.shape, D.describe().loc[['mean', 'std']].round(4).to_dict(),
          'corr with base:', {c: round(float(np.corrcoef(D[c], base[k])[0, 1]), 4) for c, k in zip(D.columns, ['bdf', 'lra', 'lN'])})
