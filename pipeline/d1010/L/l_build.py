"""d1010 L builder (run only after l_test.py PASS): blend_v26L = blend_v25 with the line-visible mid-noise (LV) rows
replaced by 0.5 * blend_v25 + 0.5 * formula7 (clean-fitted, unchanged; PREREG.txt).

LV rows: 9.5 <= raw features_v3 ic_noise < 12 AND ic_ridge_snr >= 0.5 (21 train / 42 test).
formula7 = OLS with intercept of hardness on [fd_area, pore_frac, logasp_aw, S_align, log N_eff, het4,
bd_frac = N_bd / N_all], fitted on the 267 clean train rows only (cleanhead feats_C_train). LV features: stage 1
(feats_LV_{train,test}.parquet, the unchanged cleanhead pipeline). Test: fit on all 267 clean train rows, predict the
42 LV test rows (prediction only). OOF: per outer fold k, fit on the clean rows with fold != k, predict the LV train
rows with fold == k (identical to l_test.py, asserted). Every other row is blend_v25 bit-identical.
Test images are used only through label-free per-image features; nothing is fitted on test rows; the json holds
train-side numbers only. Writes ONLY to CACHE/submissions (never the repo); refuses to overwrite.

  python l_build.py
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import hashlib
import json
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

REPO = '/home/claude/microstructure-hardness-prediction'
CACHE = '/mnt/project-files/work/hardness-cache'
CH = CACHE + '/scripts/d1009/J/cleanhead/'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = CACHE + '/submissions'
NAME = 'blend_v26L'
W = 0.5
TERMS = ['fd_area', 'pore_frac', 'logasp_aw', 'S_align', 'logN_eff', 'het4', 'bd_frac']
MD5 = {'blend_v25.csv': 'f1e6dbc47ecd6136a0cacd1a14d034cb', 'blend_v25_oof.csv': 'f20fb0a346d85308ce96f237b9082086',
       'blend_v24.csv': 'f49530982200297df1e57d33e5027837', 'blend_v24_oof.csv': 'f1bb2163b732b3f950c3bd8b52e93447'}
sys.path.insert(0, REPO)
from src.common import load_train  # noqa: E402


def design(F):
    D = pd.DataFrame(index=F.index)
    for c in ['fd_area', 'pore_frac', 'logasp_aw', 'S_align', 'het4']:
        D[c] = F[c].astype(float)
    D['logN_eff'] = np.log(F['N_eff'].astype(float))
    D['bd_frac'] = F['N_bd'].astype(float) / F['N_all'].astype(float)
    D = D[TERMS]
    assert np.isfinite(D.values).all()
    return D


def ols(X, y):
    return np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]


def apply(b, X):
    return np.column_stack([np.ones(len(X)), X]) @ b


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


# ---------------------------------------------------------------- gate: l_test PASS with the recorded script
res = json.load(open(HERE + '/l_test_result.json'))
sha_rec = open(HERE + '/l_test.sha256').read().split()[0]
assert res['verdict'] == 'PASS' and all(res['bar_items'].values()), 'l_test did not PASS: build nothing'
assert res['script_sha256'] == sha_rec == hashlib.sha256(open(HERE + '/l_test.py', 'rb').read()).hexdigest()

# ---------------------------------------------------------------- inputs
paths = {k: f'{OUT_DIR}/{NAME}{s}' for k, s in (('csv', '.csv'), ('oof', '_oof.csv'), ('json', '.json'))}
for p in paths.values():
    assert not os.path.exists(p), f'refusing to overwrite {p}'
for c, m in MD5.items():
    assert hashlib.md5(open(REPO + '/submissions/' + c, 'rb').read()).hexdigest() == m == \
        hashlib.md5(open(CACHE + '/submissions/' + c, 'rb').read()).hexdigest(), c
tr = load_train()
sub = pd.read_csv(REPO + '/data/sample_submission.csv')
v3 = pd.read_parquet(REPO + '/data/features_v3.parquet').set_index('ID')
assert len(tr) == 500 and len(sub) == 1000


def masks(ids):
    nz, rd = v3.loc[ids, 'ic_noise'].values, v3.loc[ids, 'ic_ridge_snr'].values
    return nz < 9.5, (nz >= 9.5) & (nz < 12) & (rd >= 0.5)


tr_clean, tr_lv = masks(tr.ID)
te_clean, te_lv = masks(sub.ID)
assert (tr_clean.sum(), tr_lv.sum(), te_clean.sum(), te_lv.sum()) == (267, 21, 519, 42)
assert not (tr_clean & tr_lv).any() and not (te_clean & te_lv).any()

v25 = pd.read_csv(CACHE + '/submissions/blend_v25.csv', float_precision='round_trip')
v24 = pd.read_csv(CACHE + '/submissions/blend_v24.csv', float_precision='round_trip')
assert (v25.ID.values == sub.ID.values).all() and (v24.ID.values == sub.ID.values).all()
o25 = pd.read_csv(CACHE + '/submissions/blend_v25_oof.csv', float_precision='round_trip').set_index('ID').loc[tr.ID].hardness.values
o24 = pd.read_csv(CACHE + '/submissions/blend_v24_oof.csv', float_precision='round_trip').set_index('ID').loc[tr.ID].hardness.values
p25 = v25.hardness.values.astype(float); p24 = v24.hardness.values.astype(float)
assert (p25[te_lv] == p24[te_lv]).all() and (o25[tr_lv] == o24[tr_lv]).all(), 'v25 != v24 on LV rows'

Fc = pd.read_parquet(CH + 'feats_C_train.parquet')
ids_c = tr.ID.values[tr_clean]
assert set(Fc.index) == set(ids_c)
Xc = design(Fc).loc[ids_c].values
y = tr.hardness.values.astype(float); yc = y[tr_clean]; fold = tr.fold.values; fc = fold[tr_clean]
FLtr = pd.read_parquet(HERE + '/feats_LV_train.parquet'); FLte = pd.read_parquet(HERE + '/feats_LV_test.parquet')
ids_lv_tr, ids_lv_te = tr.ID.values[tr_lv], sub.ID.values[te_lv]
assert set(FLtr.index[FLtr.group == 'LV']) == set(ids_lv_tr) and set(FLte.index[FLte.group == 'LV']) == set(ids_lv_te)
Xlv_tr = design(FLtr).loc[ids_lv_tr].values; Xlv_te = design(FLte).loc[ids_lv_te].values

# ---------------------------------------------------------------- formula7 (clean-fitted) on LV rows
b_full = ols(Xc, yc)
f7_te = apply(b_full, Xlv_te)
f7_oof = np.full(len(ids_lv_tr), np.nan); b_folds = []
flv = fold[tr_lv]
for k in range(5):
    b = ols(Xc[fc != k], yc[fc != k]); b_folds.append(b)
    m = flv == k
    if m.any():
        f7_oof[m] = apply(b, Xlv_tr[m])
assert np.isfinite(f7_te).all() and np.isfinite(f7_oof).all()
assert np.allclose(b_full, list(res['coef_full'].values()), rtol=0, atol=1e-9)

# ---------------------------------------------------------------- build
p26 = p25.copy(); p26[te_lv] = (1 - W) * p25[te_lv] + W * f7_te
o26 = o25.copy(); o26[tr_lv] = (1 - W) * o25[tr_lv] + W * f7_oof
lt = pd.read_csv(HERE + '/l_test_oof.csv', float_precision='round_trip')
lt = lt[lt.group == 'LV'].set_index('ID').loc[ids_lv_tr]
assert (lt.new_oof.values == o26[tr_lv]).all() and (lt.f7_oof.values == f7_oof).all(), 'OOF != l_test.py OOF'

out = pd.DataFrame({'ID': sub.ID.values, 'hardness': p26})
oof = pd.DataFrame({'ID': tr.ID.values, 'hardness': o26})
assert len(out) == 1000 and (out.ID.values == sub.ID.values).all() and np.isfinite(out.hardness).all()
assert len(oof) == 500 and np.isfinite(oof.hardness).all() and (oof.ID.values == tr.ID.values).all()
same = out.hardness.values == p25
assert same[~te_lv].all(), 'rows outside the LV test rows changed'
assert (~same).sum() == 42, (~same).sum()
osame = oof.hardness.values == o25
assert osame[~tr_lv].all() and (~osame).sum() == 21

# ---------------------------------------------------------------- train-side stats only
e25, e26 = o25 - y, o26 - y
lvf = lambda e: [rmse(e[tr_lv & (fold == k)]) if (tr_lv & (fold == k)).any() else None for k in range(5)]
stats = {
    'base': 'blend_v25',
    'rule': 'LV rows (9.5 <= raw features_v3 ic_noise < 12 and ic_ridge_snr >= 0.5): 0.5*blend_v25 + 0.5*formula7 '
            '(clean-fitted, unchanged); other rows = blend_v25 bit-identical',
    'prereg': 'scripts/d1010/L/PREREG.txt', 'l_test_sha256': sha_rec, 'l_test_verdict': res['verdict'],
    'l_test_bar_items': res['bar_items'],
    'formula7_terms': ['intercept'] + TERMS,
    'coef_full': dict(zip(['intercept'] + TERMS, map(float, b_full))),
    'coef_folds': [dict(zip(['intercept'] + TERMS, map(float, b))) for b in b_folds],
    'mix_weight': W, 'n_train_LV': int(tr_lv.sum()), 'n_test_LV': int(te_lv.sum()),
    'LV_rows_per_fold': [int((tr_lv & (fold == k)).sum()) for k in range(5)],
    'LV_rmse_v25': rmse(e25[tr_lv]), 'LV_rmse_new': rmse(e26[tr_lv]), 'LV_gain': rmse(e25[tr_lv]) - rmse(e26[tr_lv]),
    'LV_fold_rmse_v25': lvf(e25), 'LV_fold_rmse_new': lvf(e26),
    'LV_formula7_alone_xfit': rmse(f7_oof - y[tr_lv]),
    'LV_signflip_p': res['LV']['signflip_p'], 'LV_droptop_gain': res['LV']['droptop_gain'],
    'nested_cv_all': rmse(e26), 'nested_cv_all_v25': rmse(e25),
    'fold_rmse': [rmse(e26[fold == k]) for k in range(5)], 'fold_rmse_v25': [rmse(e25[fold == k]) for k in range(5)],
    'notes': 'PREREG d1010/L: the unchanged clean formula7 (d1009/J) applied to the line-visible mid-noise rows, fixed '
             'w 0.5, no refit on LV rows, no selection; stage-1 features from the cleanhead pipeline (verified exact on '
             '5 clean train images); one stage-2 test (l_test.py) passed all four bar items. Test images are used only '
             'for label-free per-image features; nothing fitted on test.',
}

# ---------------------------------------------------------------- write (cache only), read back
os.makedirs(OUT_DIR, exist_ok=True)
out.to_csv(paths['csv'], index=False); oof.to_csv(paths['oof'], index=False)
json.dump(stats, open(paths['json'], 'w'), indent=1)
chk = pd.read_csv(paths['csv'], float_precision='round_trip')
assert len(chk) == 1000 and (chk.ID.values == sub.ID.values).all() and (chk.hardness.values == p26).all()
assert (chk.hardness.values == p25)[~te_lv].all() and (chk.hardness.values != p25).sum() == 42
chko = pd.read_csv(paths['oof'], float_precision='round_trip')
assert (chko.hardness.values == o26).all() and (chko.hardness.values != o25).sum() == 21
md5 = {k: hashlib.md5(open(p, 'rb').read()).hexdigest() for k, p in paths.items()}
print('written', paths)
print('md5', md5)
print('LV train: v25 %.4f -> new %.4f (gain %+.4f) | per fold v25 %s new %s | n per fold %s'
      % (stats['LV_rmse_v25'], stats['LV_rmse_new'], stats['LV_gain'],
         [None if v is None else round(v, 3) for v in stats['LV_fold_rmse_v25']],
         [None if v is None else round(v, 3) for v in stats['LV_fold_rmse_new']], stats['LV_rows_per_fold']))
print('all-500 nested %.4f (v25 %.4f)' % (stats['nested_cv_all'], stats['nested_cv_all_v25']))
# label-free test-side summary of the change (printed only; never stored in the json)
dlt = p26[te_lv] - p25[te_lv]
print('test change on %d LV rows: mean %+.3f sd %.3f max|d| %.3f' % (len(dlt), dlt.mean(), dlt.std(), np.abs(dlt).max()))
dtr = o26[tr_lv] - o25[tr_lv]
print('train OOF change on %d LV rows: mean %+.3f sd %.3f max|d| %.3f' % (len(dtr), dtr.mean(), dtr.std(), np.abs(dtr).max()))
