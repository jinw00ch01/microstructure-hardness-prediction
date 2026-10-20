"""s26 (label-free build on the 267 clean TRAIN images + cross-fitted DESCRIPTIVE look; evidence for idea 2 only):
can segmentation consensus raise the fidelity of the origin-grid het further than qavg (mean of het4q over sigma
0.7/0.8/0.9 at tf 0.6)? Pre-declared variants (p = 0.25 grid het, pores refilled):
  V1 sigma 0.8 tf 0.5      V2 sigma 0.8 tf 0.7
  V3 pixel-wise MEDIAN of log full-area maps over 6 maps (sigma 0.7/0.8/0.9 x tf 0.5/0.6), then the grid het
  V4 mean of the 6 maps' het4q (het-level average)
Reference: qavg (s14). R form (f8 with het4 replaced) and the 0.5 mix vs v28n."""
import os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
sys.path.insert(0, '/mnt/project-files/work/hardness-cache/scripts/d1011/bold/L1_formula')
from l1common import *
sys.path.insert(0, HERE)
from s14_segfam import seg, pores_of, fill_pores, REPO  # noqa
from joblib import Parallel, delayed
from PIL import Image
GRID = np.add.outer(np.arange(256) // 64 * 4, np.arange(256) // 64)


def la_map(im, s, tf):
    sm, ws = seg(im, s, tf); ws = fill_pores(ws.astype(np.int64), pores_of(sm, ws))
    _, inv = np.unique(ws, return_inverse=True); ws = inv.reshape(ws.shape)
    return np.log(np.bincount(ws.ravel()).astype(float))[ws]


def hq(la):
    return np.std([-4 * np.log(np.exp(-0.25 * la[GRID == b]).mean()) for b in range(16)])


def one(i):
    im = np.array(Image.open(f'{REPO}/data/train/{i}.png')).astype(np.float32)
    maps = {(s, tf): la_map(im, s, tf) for s in (0.7, 0.8, 0.9) for tf in (0.5, 0.6)}
    f = dict(ID=i, V1=hq(maps[(0.8, 0.5)]), V2=hq(la_map(im, 0.8, 0.7)),
             V3=hq(np.median(np.stack(list(maps.values())), 0)), V4=np.mean([hq(m) for m in maps.values()]))
    return f


if __name__ == '__main__':
    C = load(); y, fold, D, o24, o28, ids = C['y'], C['fold'], C['D'], C['o24'], C['o28'], C['ids']
    X = pd.DataFrame(Parallel(n_jobs=2)(delayed(one)(i) for i in ids)).set_index('ID').loc[ids]
    X.to_parquet(HERE + 's26_train.parquet')
    q = pd.read_parquet(HERE + 's14_train.parquet').pivot(index='ID', columns='sig', values='het4q').loc[ids]
    X['qavg'] = ((q[0.7] + q[0.8] + q[0.9]) / 3).values
    N = C['F'].N_eff.values
    for c in ['qavg', 'V1', 'V2', 'V3', 'V4']:
        Dx = D.copy(); Dx['het4'] = X[c].values
        f = xfit_ols(Dx[F8].values, y, fold); r = compare(0.5 * o24 + 0.5 * f, o28, y, fold, N)
        print('%-5s formula %.3f mix %.4f gain %+.4f folds %d/5 %s p %.4f terc %s' % (c, rmse(f - y), r['cand'], r['gain'], r['folds_better'],
              r['fold_delta'], r['signflip_p'], r['terc_cand']))
