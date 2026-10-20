"""t9 (label-free build on the 267 clean TRAIN images, then a NESTED swap test; family fixed before running).
Does the "average over L1's 6 maps" idea also help the biggest formula term, the dark area fraction fd_area?
Per map (sigma 0.7/0.8/0.9 x tf 0.5/0.6, pores refilled for the area count but excluded as cleanhead does):
grain table (regionprops on the smoothed image), gn = src.het_blocks.shading_norm, pores = gn < 0.68 & area < 600,
phase threshold = cleanhead phase_thr (2-GMM on gn of grains > 25 px, default 0.90), fd = dark area / non-pore area.
Variants (W1 rule as base, i.e. V4 in both halves; only fd_area swapped in f8):
  FD6  mean fd over the 6 maps      FD6t  same but with cleanhead's per-image threshold (feats_C thr) on every map
Stats vs W1: fold deltas, sign-flip, 100 re-splits."""
import os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from skimage import measure
sys.path.insert(0, '/mnt/project-files/work/hardness-cache/scripts/d1011/bold/L1_formula')
from l1common import *
from s14_segfam import seg  # noqa
from src.het_blocks import shading_norm  # noqa
from sklearn.mixture import GaussianMixture
OUT = '/mnt/project-files/work/hardness-cache/scripts/d1011/bold/L8_wild/'
MAPS = [(s, tf) for s in (0.7, 0.8, 0.9) for tf in (0.5, 0.6)]


def phase_thr(gp):
    big = gp[gp.area > 25]; thr = 0.90
    try:
        gm = GaussianMixture(2, random_state=0).fit(big.gn.values[:, None])
        mu = np.sort(gm.means_.ravel())
        if mu[1] - mu[0] > 0.08:
            xs = np.linspace(mu[0], mu[1], 200); p = gm.predict_proba(xs[:, None]); lo = np.argmin(gm.means_.ravel())
            thr = xs[np.argmin(np.abs(p[:, lo] - 0.5))]
    except Exception:
        pass
    return thr


def one(i, thr_c):
    im = np.array(Image.open(f'{REPO}/data/train/{i}.png')).astype(np.float32)
    out = dict(ID=i)
    a, b = [], []
    for s, tf in MAPS:
        sm, ws = seg(im, s, tf)
        g = pd.DataFrame([dict(label=p.label, area=p.area, cy=p.centroid[0], cx=p.centroid[1], mean=p.intensity_mean)
                          for p in measure.regionprops(ws, intensity_image=sm)])
        g['gn'] = shading_norm(g)
        gp = g[~((g.gn < 0.68) & (g.area < 600))]
        ar = gp.area.values.astype(float)
        a.append(ar[(gp.gn < phase_thr(gp)).values].sum() / ar.sum())
        b.append(ar[(gp.gn < thr_c).values].sum() / ar.sum())
    out['FD6'] = float(np.mean(a)); out['FD6t'] = float(np.mean(b)); out['FD6_sd'] = float(np.std(a))
    return out


if __name__ == '__main__':
    C = load(); y, fold, D, o24, o28, ids = C['y'], C['fold'], C['D'], C['o24'], C['o28'], C['ids']
    fp = OUT + 't9_fd_train.parquet'
    if not os.path.exists(fp):
        thr = C['F'].thr.astype(float)
        X = pd.DataFrame(Parallel(n_jobs=3)(delayed(one)(i, thr.loc[i]) for i in ids)).set_index('ID')
        X.to_parquet(fp)
    X = pd.read_parquet(fp).loc[ids]
    print('label-free: corr with cleanhead fd_area: FD6 %.4f FD6t %.4f | mean across-map SD %.4f | means %.4f %.4f vs %.4f' % (
        np.corrcoef(X.FD6, D.fd_area)[0, 1], np.corrcoef(X.FD6t, D.fd_area)[0, 1], X.FD6_sd.mean(), X.FD6.mean(), X.FD6t.mean(), D.fd_area.mean()))
    n = len(y)
    o21b = pd.read_csv(CACHE + '/submissions/blend_v21base_oof.csv').set_index('ID').hardness.loc[ids].values
    hb = pd.read_parquet(REPO + '/data/het_blocks_train.parquet').set_index('ID').loc[ids]
    V4 = pd.read_parquet(HERE + 's26_train.parquet').loc[ids, 'V4'].values
    Xh = np.column_stack([V4, hb.N_eff.values ** 0.25])

    def fitp(A, yt, At):
        bb = np.linalg.lstsq(np.column_stack([np.ones(len(A)), A]), yt, rcond=None)[0]
        return np.column_stack([np.ones(len(At)), At]) @ bb

    def run(folds, fdcol):
        Dx = D.copy(); Dx['het4'] = V4
        if fdcol is not None:
            Dx['fd_area'] = X[fdcol].values
        X8 = Dx[F8].values; p = np.empty(n)
        for k in range(5):
            tr = folds != k; te = ~tr
            p[te] = 0.5 * (o21b[te] + fitp(Xh[tr], y[tr] - o21b[tr], Xh[te])) + 0.5 * fitp(X8[tr], y[tr], X8[te])
        return p

    W1 = run(fold, None)
    print('W1 clean %.4f (t7: 8.4176)' % rmse(W1 - y))
    rng = np.random.default_rng(20261011); yb = pd.qcut(y, 10, labels=False)
    splits = []
    for it in range(100):
        f = np.empty(n, int)
        for bb in range(10):
            idx = np.where(yb == bb)[0]; rng.shuffle(idx); f[idx] = np.arange(len(idx)) % 5
        splits.append(f)
    for v in ['FD6', 'FD6t']:
        p = run(fold, v); r = compare(p, W1, y, fold, C['F'].N_eff.values)
        rs = np.array([rmse(run(f, None) - y) - rmse(run(f, v) - y) for f in splits])
        print('%-5s clean %.4f | vs W1 %+.4f (%d/5 %s p %.3f) | re-split mean %+.4f P(<=0) %.2f' % (
            v, r['cand'], r['gain'], r['folds_better'], r['fold_delta'], r['signflip_p'], rs.mean(), (rs <= 0).mean()))
