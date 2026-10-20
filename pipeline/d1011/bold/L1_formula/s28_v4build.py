"""s28 (LABEL-FREE build): V4 = mean over 6 segmentations (sigma 0.7/0.8/0.9 x tf 0.5/0.6; pores refilled) of the
p = 0.25 origin-grid het, for the clean TEST rows (519), the LV rows (21 train / 42 test), with a 5-image check that it
reproduces s26 on clean train. Same functions as s26 (imported). One image at a time; nothing fitted.
Writes s28_V4_{clean_test,LV_train,LV_test}.parquet (ID, V4 and the 6 per-map values)."""
import os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
sys.path.insert(0, '/mnt/project-files/work/hardness-cache/scripts/d1011/bold/L1_formula')
from l1common import *
sys.path.insert(0, HERE)
from s26_consensus import la_map, hq  # noqa
from s14_segfam import REPO  # noqa
from joblib import Parallel, delayed
from PIL import Image
MAPS = [(s, tf) for s in (0.7, 0.8, 0.9) for tf in (0.5, 0.6)]


def one(split, i):
    im = np.array(Image.open(f'{REPO}/data/{split}/{i}.png')).astype(np.float32)
    v = {f'hq_{s}_{tf}': hq(la_map(im, s, tf)) for s, tf in MAPS}
    v['V4'] = float(np.mean(list(v.values()))); v['ID'] = i
    return v


if __name__ == '__main__':
    nj = int(os.environ.get('NJ', 2))
    s26 = pd.read_parquet(HERE + 's26_train.parquet')
    chk = [one('train', i) for i in s26.index[:5]]
    dmax = max(abs(c['V4'] - s26.loc[c['ID'], 'V4']) for c in chk)
    print('reproduce s26 V4 on 5 clean train images: max diff %.2e' % dmax); assert dmax < 1e-12
    LVD = CACHE + '/scripts/d1010/L/'
    jobs = {'clean_test': ('test', sorted(pd.read_parquet(CH + 'feats_C_test.parquet').index))}
    for split in ['train', 'test']:
        F = pd.read_parquet(LVD + f'feats_LV_{split}.parquet'); jobs[f'LV_{split}'] = (split, sorted(F.index[F.group == 'LV']))
    for nm, (split, ids) in jobs.items():
        X = pd.DataFrame(Parallel(n_jobs=nj)(delayed(one)(split, i) for i in ids)).set_index('ID')
        X.to_parquet(HERE + f's28_V4_{nm}.parquet'); print(nm, X.shape, 'V4 mean %.4f' % X.V4.mean(), flush=True)
