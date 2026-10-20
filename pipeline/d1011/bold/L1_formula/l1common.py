"""d1011 bold L1 (label formula on clean rows): shared loader + nested evaluator.

Rows: the 267 clean train rows (raw features_v3 ic_noise < 9.5). Folds: data/folds.csv (shared 5 folds).
Base B8 = v28n clean rows = 0.5*blend_v24 OOF + 0.5*formula8 (formula7 + ringall_avg5), formula cross-fitted.
Every fitted quantity in a candidate (OLS coefs, NLS params, exponents, w, set choice) is fitted inside training folds.
Test rows are never touched here.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
import pandas as pd

REPO = '/home/claude/microstructure-hardness-prediction'
CACHE = '/mnt/project-files/work/hardness-cache'
CH = CACHE + '/scripts/d1009/J/cleanhead/'
RP = CACHE + '/scripts/d1010/C/residual-probe/'
HERE = CACHE + '/scripts/d1011/bold/L1_formula/'
F7 = ['fd_area', 'pore_frac', 'logasp_aw', 'S_align', 'logN_eff', 'het4', 'bd_frac']
F8 = F7 + ['ringall_avg5']


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def load():
    tr = pd.read_csv(REPO + '/data/train.csv').merge(pd.read_csv(REPO + '/data/folds.csv'), on='ID')
    v3 = pd.read_parquet(REPO + '/data/features_v3.parquet').set_index('ID')
    clean = v3.loc[tr.ID, 'ic_noise'].values < 9.5
    ids = tr.ID.values[clean]
    F = pd.read_parquet(CH + 'feats_C_train.parquet').loc[ids]
    A = pd.read_parquet(RP + 'feats_avg5_train.parquet').loc[ids]
    D = pd.DataFrame(index=ids)
    for c in ['fd_area', 'pore_frac', 'logasp_aw', 'S_align', 'het4']:
        D[c] = F[c].astype(float)
    D['logN_eff'] = np.log(F.N_eff.astype(float))
    D['bd_frac'] = F.N_bd.astype(float) / F.N_all.astype(float)
    D['ringall_avg5'] = A.ringall_avg5.astype(float)
    y = tr.set_index('ID').loc[ids, 'hardness'].values.astype(float)
    fold = tr.set_index('ID').loc[ids, 'fold'].values.astype(int)
    sub = CACHE + '/submissions/'
    o24 = pd.read_csv(sub + 'blend_v24_oof.csv').set_index('ID').hardness.loc[ids].values
    o28 = pd.read_csv(sub + 'blend_v28n_oof.csv').set_index('ID').hardness.loc[ids].values
    return dict(ids=ids, y=y, fold=fold, D=D, F=F, o24=o24, o28=o28, tr=tr, clean=clean)


def ols_fit(X, y):
    Xc = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(Xc, y, rcond=None)[0]


def ols_pred(b, X):
    return np.column_stack([np.ones(len(X)), X]) @ b


def xfit_ols(X, y, fold, mask=None):
    """Cross-fitted OLS on the 5 shared folds (optionally restricted to rows in mask for training)."""
    p = np.empty(len(y))
    for k in range(5):
        m = fold != k
        b = ols_fit(X[m], y[m])
        p[~m] = ols_pred(b, X[~m])
    return p


def signflip_p(d, nflip=20000, seed=20261011):
    rng = np.random.default_rng(seed)
    obs, n, cnt = d.mean(), len(d), 0
    for _ in range(nflip // 5000):
        s = rng.integers(0, 2, size=(5000, n)).astype(float) * 2 - 1
        cnt += int(np.sum((s @ d) / n <= obs + 1e-12))
    return (1 + cnt) / (nflip + 1)


def compare(pred, base, y, fold, N=None):
    ec, eb = pred - y, base - y
    out = dict(cand=rmse(ec), base=rmse(eb), gain=rmse(eb) - rmse(ec),
               fold_delta=[round(rmse(ec[fold == k]) - rmse(eb[fold == k]), 3) for k in range(5)])
    out['folds_better'] = int(sum(x < 0 for x in out['fold_delta']))
    d = ec ** 2 - eb ** 2
    out['signflip_p'] = signflip_p(d)
    if N is not None:
        t = pd.qcut(N, 3, labels=False)
        out['terc_cand'] = [round(rmse(ec[t == j]), 3) for j in range(3)]
        out['terc_base'] = [round(rmse(eb[t == j]), 3) for j in range(3)]
    return out
