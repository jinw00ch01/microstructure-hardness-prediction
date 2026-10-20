"""J builder: blend_v25 = blend_v24 with clean rows (raw ic_noise < 9.5, features_v3) replaced by
0.5 * blend_v24 + 0.5 * formula7.

formula7 = OLS (with intercept) of hardness on
    [fd_area, pore_frac, logasp_aw, S_align, log N_eff, het4, bd_frac = N_bd / N_all]
from the clean-row segmentation features (cleanhead/feats_C_{train,test}.parquet, src.het_blocks.segment).
Test side: fitted on all 267 clean train rows, applied to the 519 clean test rows (prediction only).
OOF side: cross-fitted on the shared 5 folds (data/folds.csv), clean train rows only.
Test images are used only through label-free per-image features; nothing is fitted on test rows.
The json holds train-side numbers only (no test statistics).

Usage:
  python j_build.py [--sub-dir REPO/submissions] [--cache-dir CACHE/submissions] [--name blend_v25]
"""
import argparse, hashlib, json, os, shutil, subprocess, sys
import numpy as np, pandas as pd

REPO = '/home/claude/microstructure-hardness-prediction'
CACHE = '/mnt/project-files/work/hardness-cache'
FEAT = CACHE + '/scripts/d1009/J/cleanhead/'
sys.path.insert(0, REPO)
from src.common import load_train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--sub-dir', default=REPO + '/submissions')
ap.add_argument('--cache-dir', default=CACHE + '/submissions')
ap.add_argument('--name', default='blend_v25')
ap.add_argument('--w', type=float, default=0.5)
a = ap.parse_args()

TERMS = ['fd_area', 'pore_frac', 'logasp_aw', 'S_align', 'logN_eff', 'het4', 'bd_frac']
NZ_CUT = 9.5


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
    Xc = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(Xc, y, rcond=None)[0]


def apply(b, X):
    return np.column_stack([np.ones(len(X)), X]) @ b


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


# ---------------------------------------------------------------- inputs
tr = load_train()                                   # ID, hardness, fold (shared folds)
sub = pd.read_csv(REPO + '/data/sample_submission.csv')
v3 = pd.read_parquet(REPO + '/data/features_v3.parquet').set_index('ID')
assert len(tr) == 500 and len(sub) == 1000
tr_clean = v3.loc[tr.ID, 'ic_noise'].values < NZ_CUT
te_clean = v3.loc[sub.ID, 'ic_noise'].values < NZ_CUT
assert tr_clean.sum() == 267 and te_clean.sum() == 519, (tr_clean.sum(), te_clean.sum())

v24 = pd.read_csv(REPO + '/submissions/blend_v24.csv', float_precision='round_trip')
v24o = pd.read_csv(REPO + '/submissions/blend_v24_oof.csv', float_precision='round_trip')
assert (v24.ID.values == sub.ID.values).all(), 'blend_v24 order != sample_submission'
v24o = v24o.set_index('ID').loc[tr.ID].reset_index()
for c in ['blend_v24.csv', 'blend_v24_oof.csv']:   # repo copy == cache copy (where the cache has one)
    if os.path.exists(CACHE + '/submissions/' + c):
        assert hashlib.md5(open(REPO + '/submissions/' + c, 'rb').read()).hexdigest() == \
            hashlib.md5(open(CACHE + '/submissions/' + c, 'rb').read()).hexdigest(), c
    else:
        print('note: no cache copy of', c)

Ftr = pd.read_parquet(FEAT + 'feats_C_train.parquet')
Fte = pd.read_parquet(FEAT + 'feats_C_test.parquet')
assert set(Ftr.index) == set(tr.ID[tr_clean]) and len(Ftr) == 267
assert set(Fte.index) == set(sub.ID[te_clean]) and len(Fte) == 519
ids_tr = tr.ID[tr_clean].values; ids_te = sub.ID[te_clean].values
Xtr = design(Ftr).loc[ids_tr].values; Xte = design(Fte).loc[ids_te].values
y = tr.hardness.values.astype(float); yc = y[tr_clean]; fold = tr.fold.values; fc = fold[tr_clean]

# ---------------------------------------------------------------- formula7
b_full = ols(Xtr, yc)
f7_te = apply(b_full, Xte)
f7_oof = np.zeros(len(yc)); b_folds = []
for k in range(5):
    m = fc != k
    b = ols(Xtr[m], yc[m]); b_folds.append(b); f7_oof[~m] = apply(b, Xtr[~m])
assert np.isfinite(f7_te).all() and np.isfinite(f7_oof).all()

# ---------------------------------------------------------------- build
w = a.w
p24 = v24.hardness.values.astype(float); o24 = v24o.hardness.values.astype(float)
p25 = p24.copy(); p25[te_clean] = (1 - w) * p24[te_clean] + w * f7_te
o25 = o24.copy(); o25[tr_clean] = (1 - w) * o24[tr_clean] + w * f7_oof

out = pd.DataFrame({'ID': sub.ID.values, 'hardness': p25})
oof = pd.DataFrame({'ID': tr.ID.values, 'hardness': o25})
assert len(out) == 1000 and (out.ID.values == sub.ID.values).all() and np.isfinite(out.hardness).all()
assert len(oof) == 500 and np.isfinite(oof.hardness).all()
same = (out.hardness.values == p24)
assert same[~te_clean].all(), 'rows outside clean test rows changed'
assert (~same).sum() == 519, (~same).sum()
assert (oof.hardness.values == o24)[~tr_clean].all() and (oof.hardness.values != o24).sum() == 267

# ---------------------------------------------------------------- train-side stats
e24 = o24 - y; e25 = o25 - y
fr = lambda e, mask=None: [rmse(e[(fold == k) & (mask if mask is not None else True)]) for k in range(5)]
stats = {
    'base': 'blend_v24',
    'rule': f'clean rows (raw ic_noise < {NZ_CUT}, features_v3): {1 - w:g}*blend_v24 + {w:g}*formula7; other rows = blend_v24 bit-identical',
    'formula7_terms': ['intercept'] + TERMS,
    'formula7_terms_def': 'cleanhead feats_C (src.het_blocks.segment, sigma 1.0; pores = gn<0.68 & area<600 excluded): '
                          'logN_eff = log(N_in + 0.5*N_bd); bd_frac = N_bd/N_all (share of non-pore grains touching the image edge)',
    'coef_full': dict(zip(['intercept'] + TERMS, map(float, b_full))),
    'coef_folds': [dict(zip(['intercept'] + TERMS, map(float, b))) for b in b_folds],
    'mix_weight': w,
    'n_train_clean': int(tr_clean.sum()), 'n_test_clean': int(te_clean.sum()),
    'nested_cv_all': rmse(e25), 'nested_cv_all_v24': rmse(e24),
    'clean_nested': rmse(e25[tr_clean]), 'clean_v24': rmse(e24[tr_clean]),
    'clean_formula7_alone_xfit': rmse(f7_oof - yc),
    'noisy_nested': rmse(e25[~tr_clean]),
    'fold_rmse': fr(e25), 'fold_rmse_v24': fr(e24),
    'fold_rmse_clean': fr(e25, tr_clean), 'fold_rmse_clean_v24': fr(e24, tr_clean),
    'notes': 'PREREG d1009/J (cleanhead formula7 + bd_frac). formula7 cross-fitted on the shared folds for OOF, '
             'fitted on all 267 clean train rows for test; test images used only for label-free per-image features '
             '(segmentation, per-image GMM phase threshold), nothing fitted on test. Mix weight 0.5 and the bd_frac term '
             'were chosen on all train labels (nested choice picks 0.5 in 5/5 folds and bd_frac in 5/5 folds, d20/d8).',
}

# ---------------------------------------------------------------- write
os.makedirs(a.sub_dir, exist_ok=True)
paths = {'csv': f'{a.sub_dir}/{a.name}.csv', 'oof': f'{a.sub_dir}/{a.name}_oof.csv', 'json': f'{a.sub_dir}/{a.name}.json'}
for p in paths.values():
    assert not os.path.exists(p), f'refusing to overwrite {p}'
out.to_csv(paths['csv'], index=False); oof.to_csv(paths['oof'], index=False)
json.dump(stats, open(paths['json'], 'w'), indent=1)
# read-back check: written file round-trips to the same floats
chk = pd.read_csv(paths['csv'], float_precision='round_trip')
assert (chk.hardness.values == out.hardness.values).all() and (chk.ID.values == sub.ID.values).all()
assert (chk.hardness.values == p24)[~te_clean].all() and (chk.hardness.values != p24).sum() == 519
if os.path.realpath(a.sub_dir).startswith(REPO):
    for k in ['csv', 'oof']:
        r = subprocess.run(['git', '-C', REPO, 'check-ignore', '-q', paths[k]])
        assert r.returncode == 0, f'{paths[k]} is NOT git-ignored'
if a.cache_dir:
    os.makedirs(a.cache_dir, exist_ok=True)
    for p in paths.values():
        dst = os.path.join(a.cache_dir, os.path.basename(p))
        assert not os.path.exists(dst), f'refusing to overwrite {dst}'
        shutil.copy2(p, dst)

md5 = {k: hashlib.md5(open(p, 'rb').read()).hexdigest() for k, p in paths.items()}
print('written', paths)
print('md5', md5)
print('nested CV all %.4f (v24 %.4f) | clean %.4f (v24 %.4f) | formula7 alone (clean, xfit) %.4f | noisy %.4f'
      % (stats['nested_cv_all'], stats['nested_cv_all_v24'], stats['clean_nested'], stats['clean_v24'],
         stats['clean_formula7_alone_xfit'], stats['noisy_nested']))
print('fold RMSE all  ', np.round(stats['fold_rmse'], 4), 'v24', np.round(stats['fold_rmse_v24'], 4))
print('fold RMSE clean', np.round(stats['fold_rmse_clean'], 4), 'v24', np.round(stats['fold_rmse_clean_v24'], 4),
      'delta', np.round(np.subtract(stats['fold_rmse_clean'], stats['fold_rmse_clean_v24']), 3))
print('coef full', {k: round(v, 3) for k, v in stats['coef_full'].items()})
# label-free test-side summary of the change (printed only; never stored in the json)
dlt = p25[te_clean] - p24[te_clean]
print('test change on %d clean rows: mean %+.3f sd %.3f max|d| %.3f | f7-v24 gap mean %+.3f sd %.3f'
      % (len(dlt), dlt.mean(), dlt.std(), np.abs(dlt).max(), (f7_te - p24[te_clean]).mean(), (f7_te - p24[te_clean]).std()))
dtr = o25[tr_clean] - o24[tr_clean]
print('train OOF change on 267 clean rows: mean %+.3f sd %.3f max|d| %.3f' % (dtr.mean(), dtr.std(), np.abs(dtr).max()))
