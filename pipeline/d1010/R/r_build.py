"""d1010 R builder: blend_v26r = blend_v25 with the clean rows (raw ic_noise < 9.5, features_v3) re-mixed as
0.5 * blend_v24 + 0.5 * formula8.

BOLD PUBLIC PROBE. No new label test: ringall_avg5 was already scored on the 267 clean train labels (d1010 C residual
probe, post-hoc family) and refuted by the statistics skeptic's re-splits (honest clean gain about 0.06). This file only
lets the public split test it. Reading lines: R/READING.txt (written before any public score of this file).

formula8 = OLS (with intercept) of hardness on the formula7 terms
    [fd_area, pore_frac, logasp_aw, S_align, log N_eff, het4, bd_frac = N_bd / N_all]   (cleanhead feats_C)
  + ringall_avg5 = mean over 5 segmentations of log(ring / allm)                        (residual-probe feats_avg5)
Test side: fitted on all 267 clean train rows, applied to the 519 clean test rows (prediction only).
OOF side: cross-fitted on the shared 5 folds (data/folds.csv), clean train rows only.
Mix weight fixed at 0.5 (the v25 form; no w selection). Every non-clean row = blend_v25 bit-identical.
Test images are used only through label-free per-image features; nothing is fitted on test rows.
The json holds train-side numbers only (no test statistics).

Writes only to CACHE/submissions (never the repo):
  python r_build.py [--out-dir CACHE/submissions] [--name blend_v26r]
"""
import os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '3'
sys.dont_write_bytecode = True
import argparse, hashlib, json, time
import numpy as np, pandas as pd

REPO = '/home/claude/microstructure-hardness-prediction'
CACHE = '/mnt/project-files/work/hardness-cache'
FEAT = CACHE + '/scripts/d1009/J/cleanhead/'
RP = CACHE + '/scripts/d1010/C/residual-probe/'

ap = argparse.ArgumentParser()
ap.add_argument('--out-dir', default=CACHE + '/submissions')
ap.add_argument('--name', default='blend_v26r')
a = ap.parse_args()
assert not os.path.realpath(a.out_dir).startswith(os.path.realpath(REPO)), 'never write into the repo'

TERMS7 = ['fd_area', 'pore_frac', 'logasp_aw', 'S_align', 'logN_eff', 'het4', 'bd_frac']
TERMS8 = TERMS7 + ['ringall_avg5']
NZ_CUT = 9.5
W = 0.5
MD5 = {'blend_v24.csv': 'f49530982200297df1e57d33e5027837', 'blend_v24_oof.csv': 'f1bb2163b732b3f950c3bd8b52e93447',
       'blend_v25.csv': 'f1e6dbc47ecd6136a0cacd1a14d034cb', 'blend_v25_oof.csv': 'f20fb0a346d85308ce96f237b9082086'}


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def design(F, A=None):
    D = pd.DataFrame(index=F.index)
    for c in ['fd_area', 'pore_frac', 'logasp_aw', 'S_align', 'het4']:
        D[c] = F[c].astype(float)
    D['logN_eff'] = np.log(F['N_eff'].astype(float))
    D['bd_frac'] = F['N_bd'].astype(float) / F['N_all'].astype(float)
    if A is None:
        D = D[TERMS7]
    else:
        D['ringall_avg5'] = A.loc[F.index, 'ringall_avg5'].astype(float)
        D = D[TERMS8]
    assert np.isfinite(D.values).all()
    return D


def ols(X, y):
    Xc = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(Xc, y, rcond=None)[0]


def apply(b, X):
    return np.column_stack([np.ones(len(X)), X]) @ b


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


# ---------------------------------------------------------------- inputs (all read-only)
tr = pd.read_csv(REPO + '/data/train.csv').merge(pd.read_csv(REPO + '/data/folds.csv'), on='ID')
sub = pd.read_csv(REPO + '/data/sample_submission.csv')
v3 = pd.read_parquet(REPO + '/data/features_v3.parquet').set_index('ID')
assert len(tr) == 500 and len(sub) == 1000 and tr.ID.is_unique and sub.ID.is_unique
tr_clean = v3.loc[tr.ID, 'ic_noise'].values < NZ_CUT
te_clean = v3.loc[sub.ID, 'ic_noise'].values < NZ_CUT
assert tr_clean.sum() == 267 and te_clean.sum() == 519, (tr_clean.sum(), te_clean.sum())

for c, h in MD5.items():   # pinned base files: repo copy == cache copy == recorded md5
    assert md5(REPO + '/submissions/' + c) == h and md5(CACHE + '/submissions/' + c) == h, c
rd = lambda p: pd.read_csv(p, float_precision='round_trip')
v24 = rd(REPO + '/submissions/blend_v24.csv'); v25 = rd(REPO + '/submissions/blend_v25.csv')
v24o = rd(REPO + '/submissions/blend_v24_oof.csv'); v25o = rd(REPO + '/submissions/blend_v25_oof.csv')
assert (v24.ID.values == sub.ID.values).all() and (v25.ID.values == sub.ID.values).all()
v24o = v24o.set_index('ID').loc[tr.ID].reset_index(); v25o = v25o.set_index('ID').loc[tr.ID].reset_index()

Ftr = pd.read_parquet(FEAT + 'feats_C_train.parquet'); Fte = pd.read_parquet(FEAT + 'feats_C_test.parquet')
Atr = pd.read_parquet(RP + 'feats_avg5_train.parquet'); Ate = pd.read_parquet(RP + 'feats_avg5_test.parquet')
ids_tr = tr.ID[tr_clean].values; ids_te = sub.ID[te_clean].values
for F, ids in ((Ftr, ids_tr), (Fte, ids_te), (Atr, ids_tr), (Ate, ids_te)):
    assert F.index.is_unique and set(F.index) == set(ids) and len(F) == len(ids)
X7tr = design(Ftr).loc[ids_tr].values; X7te = design(Fte).loc[ids_te].values
X8tr = design(Ftr, Atr).loc[ids_tr].values; X8te = design(Fte, Ate).loc[ids_te].values
assert np.array_equal(X8tr[:, :7], X7tr) and np.array_equal(X8te[:, :7], X7te)
y = tr.hardness.values.astype(float); yc = y[tr_clean]; fold = tr.fold.values; fc = fold[tr_clean]

p24 = v24.hardness.values.astype(float); o24 = v24o.hardness.values.astype(float)
p25 = v25.hardness.values.astype(float); o25 = v25o.hardness.values.astype(float)


def fit_xfit(Xtr, Xte):
    b_full = ols(Xtr, yc); f_te = apply(b_full, Xte)
    f_oof = np.zeros(len(yc)); b_folds = []
    for k in range(5):
        m = fc != k
        b = ols(Xtr[m], yc[m]); b_folds.append(b); f_oof[~m] = apply(b, Xtr[~m])
    assert np.isfinite(f_te).all() and np.isfinite(f_oof).all()
    return b_full, b_folds, f_te, f_oof


# ---------------------------------------------------------------- v25 reproduction check (formula7)
_, _, f7_te, f7_oof = fit_xfit(X7tr, X7te)
r25 = p24.copy(); r25[te_clean] = (1 - W) * p24[te_clean] + W * f7_te
r25o = o24.copy(); r25o[tr_clean] = (1 - W) * o24[tr_clean] + W * f7_oof
v25_repro = {'test_maxabs': float(np.abs(r25 - p25).max()), 'oof_maxabs': float(np.abs(r25o - o25).max()),
             'test_exact': bool((r25 == p25).all()), 'oof_exact': bool((r25o == o25).all())}
assert v25_repro['test_maxabs'] < 1e-9 and v25_repro['oof_maxabs'] < 1e-9, v25_repro

# ---------------------------------------------------------------- formula8 and build
b_full, b_folds, f8_te, f8_oof = fit_xfit(X8tr, X8te)
p26 = p25.copy(); p26[te_clean] = (1 - W) * p24[te_clean] + W * f8_te
o26 = o25.copy(); o26[tr_clean] = (1 - W) * o24[tr_clean] + W * f8_oof

out = pd.DataFrame({'ID': sub.ID.values, 'hardness': p26})
oof = pd.DataFrame({'ID': tr.ID.values, 'hardness': o26})
assert len(out) == 1000 and (out.ID.values == sub.ID.values).all() and np.isfinite(out.hardness).all()
assert len(oof) == 500 and (oof.ID.values == tr.ID.values).all() and np.isfinite(oof.hardness).all()
same = out.hardness.values == p25
assert same[~te_clean].all(), 'test rows outside the clean rows changed vs blend_v25'
assert (~same).sum() == 519 and (~same == te_clean).all(), (~same).sum()
same_o = oof.hardness.values == o25
assert same_o[~tr_clean].all(), 'OOF rows outside the clean rows changed vs blend_v25'
assert (~same_o).sum() == 267 and (~same_o == tr_clean).all(), (~same_o).sum()

# ---------------------------------------------------------------- train-side stats (no test statistics)
e24, e25, e26 = o24 - y, o25 - y, o26 - y
fr = lambda e, mask=None: [rmse(e[(fold == k) & (mask if mask is not None else True)]) for k in range(5)]
dtr = o26 - o25
ci = lambda b: dict(zip(['intercept'] + TERMS8, map(float, b)))
stats = {
    'name': a.name,
    'base': 'blend_v25',
    'rule': f'clean rows (raw ic_noise < {NZ_CUT}, features_v3): {1 - W:g}*blend_v24 + {W:g}*formula8; '
            'other rows = blend_v25 bit-identical (= blend_v24 there)',
    'status': 'BOLD PUBLIC PROBE: ringall_avg5 passed only a post-hoc family on the 267 clean labels and was refuted by '
              're-splits (honest clean gain about 0.06; d1010/C SYNTHESIS.md, skeptic-stats-ringall). No new label test. '
              'Reading lines in scripts/d1010/R/READING.txt.',
    'formula8_terms': ['intercept'] + TERMS8,
    'formula8_terms_def': 'formula7 terms as blend_v25 (cleanhead feats_C) + ringall_avg5 = mean over 5 segmentations '
                          '(sigma/tf 1.0/0.6, 0.8/0.6, 1.3/0.6, 1.0/0.45, 1.0/0.8) of log(ring/allm); ring = mean pieces '
                          'per row/col on lines 1-4 from the border, allm = mean pieces per row/col (residual-probe '
                          'build_avg5.py, verified from the PNGs by R/r_verify.py)',
    'coef_full': ci(b_full),
    'coef_folds': [ci(b) for b in b_folds],
    'mix_weight': W,
    'n_train_clean': int(tr_clean.sum()), 'n_test_clean': int(te_clean.sum()),
    'v25_reproduction': v25_repro,
    'nested_cv_all': rmse(e26), 'nested_cv_all_v25': rmse(e25), 'nested_cv_all_v24': rmse(e24),
    'clean_nested': rmse(e26[tr_clean]), 'clean_v25': rmse(e25[tr_clean]), 'clean_v24': rmse(e24[tr_clean]),
    'clean_formula8_alone_xfit': rmse(f8_oof - yc), 'clean_formula7_alone_xfit': rmse(f7_oof - yc),
    'noisy_nested': rmse(e26[~tr_clean]),
    'clean_mse_drop_vs_v25': float(np.mean(e25[tr_clean] ** 2) - np.mean(e26[tr_clean] ** 2)),
    'all500_mse_drop_vs_v25': float(np.mean(e25 ** 2) - np.mean(e26 ** 2)),
    'train_d_vs_v25': {'mean_d2_all500': float(np.mean(dtr ** 2)), 'mean_de_all500': float(np.mean(dtr * e25)),
                       'mean_d2_clean': float(np.mean(dtr[tr_clean] ** 2)), 'mean_de_clean': float(np.mean((dtr * e25)[tr_clean])),
                       'def': 'd = v26r OOF - v25 OOF, e = v25 OOF - y; delta MSE = mean d^2 + 2 mean d e'},
    'fold_rmse': fr(e26), 'fold_rmse_v25': fr(e25),
    'fold_rmse_clean': fr(e26, tr_clean), 'fold_rmse_clean_v25': fr(e25, tr_clean),
    'fold_delta_clean_vs_v25': list(np.subtract(fr(e26, tr_clean), fr(e25, tr_clean))),
    'clean_folds_better': int(sum(x < 0 for x in np.subtract(fr(e26, tr_clean), fr(e25, tr_clean)))),
    'notes': 'd1010 R. formula8 cross-fitted on the shared folds for OOF, fitted on all 267 clean train rows for test; '
             'fixed w 0.5 (v25 form). Test images used only for label-free per-image features (segmentations); nothing '
             'fitted on test. The clean OOF gain reported here is the shared-split figure of a column chosen post hoc; '
             're-split mean of the fixed-w form 0.141 (sd 0.015, 200 splits) still carries the menu-selection bias, the '
             'honest nested-procedure re-split gain is about 0.06.',
}

# ---------------------------------------------------------------- write (refuse to overwrite) + read-back
os.makedirs(a.out_dir, exist_ok=True)
paths = {'csv': f'{a.out_dir}/{a.name}.csv', 'oof': f'{a.out_dir}/{a.name}_oof.csv', 'json': f'{a.out_dir}/{a.name}.json'}
for p in paths.values():
    assert not os.path.exists(p), f'refusing to overwrite {p}'
out.to_csv(paths['csv'], index=False); oof.to_csv(paths['oof'], index=False)
json.dump(stats, open(paths['json'], 'w'), indent=1)
md5s = {k: md5(p) for k, p in paths.items()}


def readback():
    chk = rd(paths['csv']); chko = rd(paths['oof'])
    assert (chk.ID.values == sub.ID.values).all() and (chk.hardness.values == p26).all()
    assert (chk.hardness.values == p25)[~te_clean].all() and (chk.hardness.values != p25).sum() == 519
    assert (chko.ID.values == tr.ID.values).all() and (chko.hardness.values == o26).all()
    assert (chko.hardness.values == o25)[~tr_clean].all() and (chko.hardness.values != o25).sum() == 267
    assert json.load(open(paths['json']))['clean_nested'] == stats['clean_nested']
    assert {k: md5(p) for k, p in paths.items()} == md5s


readback()
time.sleep(10)
readback()   # shared filesystem: read back again about 10 s after writing

print('written', paths)
print('md5', md5s)
print('v25 reproduction', v25_repro)
print('nested CV all %.4f (v25 %.4f, v24 %.4f) | clean %.4f (v25 %.4f, v24 %.4f) | f8 alone %.4f (f7 %.4f) | noisy %.4f'
      % (stats['nested_cv_all'], stats['nested_cv_all_v25'], stats['nested_cv_all_v24'], stats['clean_nested'],
         stats['clean_v25'], stats['clean_v24'], stats['clean_formula8_alone_xfit'], stats['clean_formula7_alone_xfit'],
         stats['noisy_nested']))
print('fold RMSE clean', np.round(stats['fold_rmse_clean'], 4), 'v25', np.round(stats['fold_rmse_clean_v25'], 4),
      'delta', np.round(stats['fold_delta_clean_vs_v25'], 3))
print('fold RMSE all  ', np.round(stats['fold_rmse'], 4), 'v25', np.round(stats['fold_rmse_v25'], 4))
print('clean MSE drop %.3f | all-500 MSE drop %.3f' % (stats['clean_mse_drop_vs_v25'], stats['all500_mse_drop_vs_v25']))
print('coef full', {k: round(v, 3) for k, v in stats['coef_full'].items()})
print('ringall coef by fold', [round(b['ringall_avg5'], 2) for b in stats['coef_folds']])
dte = p26[te_clean] - p25[te_clean]   # label-free test-side summary (printed only; never stored in the json)
print('test change vs v25 on %d clean rows: mean %+.3f sd %.3f max|d| %.3f' % (len(dte), dte.mean(), dte.std(), np.abs(dte).max()))
print('train OOF change vs v25 on 267 clean rows: mean %+.3f sd %.3f max|d| %.3f'
      % (dtr[tr_clean].mean(), dtr[tr_clean].std(), np.abs(dtr[tr_clean]).max()))
