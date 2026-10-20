"""Label-free (for the mid rows) choice of the mid-noise segmentation variant.

The 267 clean TRAIN images are degraded to look like line-free mid-noise images (calib_degrade.json: contrast c,
Gaussian blur b, dark-phase gap x kd, Gaussian noise up to a target ic_noise drawn from the 47 line-free mid TRAIN
rows' ic_noise values). Each segmentation variant (mid_seg.VARIANTS) is run on the degraded images and compared
with the clean pipeline on the ORIGINAL image (cleanhead feats_C_train = raw_s1 on the undegraded image).
Criteria: per-term Pearson corr and the corr / RMSE of the formula7 prediction (coefficients = OLS on the 267
clean train rows' ORIGINAL features and labels; no mid-row label is read). No test image is used.
Output: fidelity_<combo>.json and fid_feats_<combo>.parquet (scratch-size, kept for audit).
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mid_seg as M  # noqa: E402
from calib_degrade import degrade  # noqa: E402

REPO = '/home/claude/microstructure-hardness-prediction'
CH = '/mnt/project-files/work/hardness-cache/scripts/d1009/J/cleanhead/'


def parse(combo):
    c, b, kd = [float(t[len(p):]) for t, p in zip(combo.split('_'), ('c', 'b', 'kd'))]
    return c, b, kd


def one(i, c, b, kd, target, seed):
    im8 = np.array(Image.open(f'{REPO}/data/train/{i}.png'))
    d = degrade(im8, c, b, target, seed, i=i, kd=kd)
    out = {}
    for v in M.VARIANTS:
        f = M.image_feats(d, v)
        out.update({f'{v}__{k}': f[k] for k in M.F7 + ['N_all', 'fd_area_P1', 'fd_area_P2']})
    return out


if __name__ == '__main__':
    combo = sys.argv[1] if len(sys.argv) > 1 else json.load(open(HERE + '/calib_degrade.json'))['chosen']
    c, b, kd = parse(combo)
    v3 = pd.read_parquet(REPO + '/data/features_v3.parquet').set_index('ID')
    tr = pd.read_csv(REPO + '/data/train.csv')
    x = v3.loc[tr.ID]
    midlf_noise = x[(x.ic_noise >= 9.5) & (x.ic_noise < 12) & (x.ic_ridge_snr < 0.5)].ic_noise.values
    F = pd.read_parquet(CH + 'feats_C_train.parquet')
    F['logN_eff'] = np.log(F.N_eff)
    F['bd_frac'] = F.N_bd / F.N_all
    ids = F.index.tolist()
    rng = np.random.default_rng(20261010)
    tgt = rng.choice(midlf_noise, len(ids), replace=True)
    R = pd.DataFrame(Parallel(n_jobs=3)(delayed(one)(i, c, b, kd, float(t), 5000 + j)
                                        for j, (i, t) in enumerate(zip(ids, tgt))), index=pd.Index(ids, name='ID'))
    R.to_parquet(HERE + f'/fid_feats_{combo}.parquet')
    # formula7 coefficients on ORIGINAL clean features (clean labels only)
    y = tr.set_index('ID').hardness.loc[ids].values.astype(float)
    X0 = F.loc[ids, M.F7].values
    D0 = np.column_stack([np.ones(len(ids)), X0])
    beta = np.linalg.lstsq(D0, y, rcond=None)[0]
    f0 = D0 @ beta
    res = {'combo': combo, 'n': len(ids), 'variants': {}}

    def score(cols):
        Xd = R[cols].values
        fd = np.column_stack([np.ones(len(ids)), Xd]) @ beta
        ok = np.isfinite(fd)
        return dict(f7_corr=float(np.corrcoef(fd[ok], f0[ok])[0, 1]),
                    f7_rmse=float(np.sqrt(np.mean((fd[ok] - f0[ok]) ** 2))),
                    f7_bias=float(np.mean(fd[ok] - f0[ok])), n_nonfinite=int((~ok).sum()),
                    term_corr={k: float(np.corrcoef(R[c][ok], F.loc[ids, k][ok])[0, 1]) for k, c in zip(M.F7, cols)})

    for v in M.VARIANTS:
        for rule, fdc in M.FD_RULES.items():
            cols = [f'{v}__{fdc}' if k == 'fd_area' else f'{v}__{k}' for k in M.F7]
            r = score(cols)
            r['N_ratio_med'] = float(np.median(R[f'{v}__N_all'] / F.loc[ids, 'N_all']))
            res['variants'][f'{v}|{rule}'] = r
            print(f'{v:10s} {rule} f7 corr {r["f7_corr"]:.3f} rmse {r["f7_rmse"]:6.2f} bias {r["f7_bias"]:+6.2f} '
                  f'N ratio {r["N_ratio_med"]:.2f} | ' + ' '.join(f'{k}:{val:.2f}' for k, val in r['term_corr'].items()),
                  flush=True)
    # composite: per term, the (variant, rule) column with the highest per-term corr
    comp = []
    for k in M.F7:
        cands = [(res['variants'][key]['term_corr'][k], key) for key in res['variants']]
        _, key = max(cands)
        v, rule = key.split('|')
        comp.append(f'{v}__{M.FD_RULES[rule]}' if k == 'fd_area' else f'{v}__{k}')
    r = score(comp)
    r['columns'] = comp
    res['variants']['composite'] = r
    print(f'composite  f7 corr {r["f7_corr"]:.3f} rmse {r["f7_rmse"]:6.2f} bias {r["f7_bias"]:+6.2f} | {comp}')
    best = max(res['variants'], key=lambda kk: res['variants'][kk]['f7_corr'])
    res['chosen'] = best
    print('chosen (max f7 corr):', best)
    res['f7_sd_orig'] = float(np.std(f0))
    print('sd of f7 on original clean features', round(float(np.std(f0)), 2))
    json.dump(res, open(HERE + f'/fidelity_{combo}.json', 'w'), indent=1)
