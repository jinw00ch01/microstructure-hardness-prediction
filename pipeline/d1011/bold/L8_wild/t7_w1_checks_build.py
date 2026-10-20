"""t7: robustness of t6 W1 (descriptive, label used, cross-fitted) and a CANDIDATE builder (NOT a submission; writes
only into this folder; never copied to submissions/).
W1 clean rule: 0.5*(v21base + off_V4) + 0.5*f8V4, where off_V4 = OLS of (y - v21base OOF) on [1, V4, N_eff^0.25]
(src.het_offset form; N_eff from data/het_blocks) and f8V4 = f8 with het4 <- L1's V4. Both fitted on the 267 clean
train rows. Test: V4 from L1 s28_V4_clean_test.parquet, f8V4 test rows taken from L1's
CANDIDATE_v28n_gridhet_V4.csv (re-derived here and asserted equal), v21base from submissions/blend_v21base.csv.
Every non-clean test row stays bit-identical to blend_v28n. Sanity: with the OLD het4 in the offset, the rule must
reproduce v28n's clean test rows (old offset refit on the same rows reproduces v24 = v21)."""
import sys, json, hashlib
sys.path.insert(0, '/mnt/project-files/work/hardness-cache/scripts/d1011/bold/L1_formula')
from l1common import *
OUT = '/mnt/project-files/work/hardness-cache/scripts/d1011/bold/L8_wild/'

C = load(); y, fold, D, o24, o28, ids = C['y'], C['fold'], C['D'], C['o24'], C['o28'], C['ids']
n = len(y); sub = CACHE + '/submissions/'
o21b = pd.read_csv(sub + 'blend_v21base_oof.csv').set_index('ID').hardness.loc[ids].values
hbt = pd.read_parquet(REPO + '/data/het_blocks_train.parquet').set_index('ID').loc[ids]
V4 = pd.read_parquet(HERE + 's26_train.parquet').loc[ids, 'V4'].values
X8 = D[F8].values.copy(); X8[:, F8.index('het4')] = V4
Xh = np.column_stack([V4, hbt.N_eff.values ** 0.25]); Xo = np.column_stack([hbt.het4.values, hbt.N_eff.values ** 0.25])


def fitb(X, yt):
    return np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), yt, rcond=None)[0]


def pr(b, X):
    return np.column_stack([np.ones(len(X)), X]) @ b


R1 = np.empty(n); W1 = np.empty(n); OFFH = np.empty(n)
for k in range(5):
    tr = fold != k; te = ~tr
    f8 = pr(fitb(X8[tr], y[tr]), X8[te]); half = o21b[te] + pr(fitb(Xh[tr], y[tr] - o21b[tr]), Xh[te])
    R1[te] = 0.5 * o24[te] + 0.5 * f8; W1[te] = 0.5 * half + 0.5 * f8; OFFH[te] = half
e0, e1, eR = o28 - y, W1 - y, R1 - y
print('clean: v28n %.4f | V4 (R1) %.4f | W1 %.4f | v24 half %.4f -> V4 half %.4f' % (rmse(e0), rmse(eR), rmse(e1), rmse(o24 - y), rmse(OFFH - y)))
res = {}
for nm, base in [('vs_v28n', o28), ('vs_V4', R1)]:
    r = compare(W1, base, y, fold, C['F'].N_eff.values)
    d = (W1 - y) ** 2 - (base - y) ** 2; o = np.argsort(d)
    r['droptop1'] = rmse((base - y)[o[1:]]) - rmse((W1 - y)[o[1:]]); r['droptop3'] = rmse((base - y)[o[3:]]) - rmse((W1 - y)[o[3:]])
    rng = np.random.default_rng(7); cnt = 0
    for _ in range(2000):   # paired row-wise swap null
        s = rng.random(n) < 0.5
        a = np.where(s, W1, base); b = np.where(s, base, W1)
        cnt += (rmse(b - y) - rmse(a - y)) >= r['gain']
    r['swap_p'] = (1 + cnt) / 2001
    res[nm] = r
    print('%s: gain %+.4f folds %d/5 %s signflip p %.4f swap p %.4f drop-top1 %+.4f drop-top3 %+.4f terc %s (base %s)' % (
        nm, r['gain'], r['folds_better'], r['fold_delta'], r['signflip_p'], r['swap_p'], r['droptop1'], r['droptop3'], r['terc_cand'], r['terc_base']))
N = np.exp(D.logN_eff.values)
for nm, e in [('v28n', e0), ('V4', eR), ('W1', e1)]:
    ab = np.linalg.lstsq(np.column_stack([np.ones(n), 1 / N]), e ** 2, rcond=None)[0]
    print('variance law %-5s e2 = %.1f + %.0f/N' % (nm, ab[0], ab[1]))
coef_old = fitb(Xo, y - o21b); coef_new = fitb(Xh, y - o21b)
print('offset coef [1, het, N^0.25]: old het4 %s | V4 %s' % (np.round(coef_old, 2), np.round(coef_new, 2)))
# all-500 nested (clean rows replaced, others = v28n nested OOF)
tr_all = C['tr']; full = pd.read_csv(sub + 'blend_v28n_oof.csv').set_index('ID').hardness.loc[tr_all.ID].values.copy()
yall = tr_all.hardness.values; pos = pd.Index(tr_all.ID).get_indexer(ids)
base_all = rmse(full - yall); full[pos] = W1; res['all500'] = dict(v28n=base_all, W1=rmse(full - yall))
print('all-500 nested: v28n %.4f -> W1 %.4f (dMSE %+.2f)' % (base_all, rmse(full - yall), rmse(full - yall) ** 2 - base_all ** 2))

# ---------------- candidate build
te = pd.read_csv(REPO + '/data/sample_submission.csv')
v3 = pd.read_parquet(REPO + '/data/features_v3.parquet').set_index('ID')
cl = v3.loc[te.ID, 'ic_noise'].values < 9.5; tid = te.ID.values[cl]; assert cl.sum() == 519
rd = lambda nm: pd.read_csv(nm, float_precision='round_trip').set_index('ID').hardness.loc[te.ID].values
v28 = rd(sub + 'blend_v28n.csv'); v24 = rd(sub + 'blend_v24.csv'); v21b = rd(sub + 'blend_v21base.csv')
L1V4 = rd(HERE + 'CANDIDATE_v28n_gridhet_V4.csv')
hbe = pd.read_parquet(REPO + '/data/het_blocks_test.parquet').set_index('ID').loc[tid]
Xo_t = np.column_stack([hbe.het4.values, hbe.N_eff.values ** 0.25])
rep = v21b[cl] + pr(coef_old, Xo_t)
d_old = float(np.abs(rep - v24[cl]).max()); print('old offset reproduces v24 clean test rows: max diff %.2e' % d_old); assert d_old < 1e-6
Ft = pd.read_parquet(CH + 'feats_C_test.parquet').loc[tid]; At = pd.read_parquet(RP + 'feats_avg5_test.parquet').loc[tid]
Dt = pd.DataFrame(index=tid)
for c in ['fd_area', 'pore_frac', 'logasp_aw', 'S_align']:
    Dt[c] = Ft[c].astype(float)
Dt['logN_eff'] = np.log(Ft.N_eff.astype(float)); Dt['bd_frac'] = Ft.N_bd / Ft.N_all; Dt['ringall_avg5'] = At.ringall_avg5
V4t = pd.read_parquet(HERE + 's28_V4_clean_test.parquet').loc[tid, 'V4'].values
Dt['het4'] = V4t
f8t = pr(fitb(X8, y), Dt[F8].values)
d_l1 = float(np.abs(0.5 * v24[cl] + 0.5 * f8t - L1V4[cl]).max()); print('f8V4 test rows match L1 V4 candidate: max diff %.2e' % d_l1); assert d_l1 < 1e-6
half_t = v21b[cl] + pr(coef_new, np.column_stack([V4t, hbe.N_eff.values ** 0.25]))
new = v28.copy(); new[cl] = 0.5 * half_t + 0.5 * f8t
assert np.isfinite(new).all() and np.array_equal(new[~cl], v28[~cl])
outp = OUT + 'CANDIDATE_v28n_gridhet_V4_both.csv'
pd.DataFrame({'ID': te.ID, 'hardness': new}).to_csv(outp, index=False)
pd.Series(W1, index=pd.Index(ids, name='ID'), name='hardness').to_csv(OUT + 'CANDIDATE_v28n_gridhet_V4_both_oof_clean.csv')
chg_te = new[cl] - L1V4[cl]; chg_tr = W1 - R1
info = dict(rule='clean test rows: 0.5*(v21base + off_V4) + 0.5*f8V4; other rows = blend_v28n', coef_off_V4=coef_new.tolist(),
            coef_off_old=coef_old.tolist(), clean_nested=dict(v28n=rmse(e0), V4=rmse(eR), W1=rmse(e1)), all500=res['all500'],
            vs_v28n=res['vs_v28n'], vs_V4=res['vs_V4'],
            test_change_vs_L1V4=dict(mean=float(chg_te.mean()), sd=float(chg_te.std())),
            train_change_vs_V4=dict(mean=float(chg_tr.mean()), sd=float(chg_tr.std())),
            md5=hashlib.md5(open(outp, 'rb').read()).hexdigest())
print('test change vs L1 V4 candidate: mean %+.3f sd %.3f | train OOF change: mean %+.3f sd %.3f' % (
    chg_te.mean(), chg_te.std(), chg_tr.mean(), chg_tr.std()))
print('V4 shift train vs test: mean %.4f / %.4f' % (V4.mean(), V4t.mean()))
print('md5', info['md5'])
json.dump(info, open(OUT + 'CANDIDATE_v28n_gridhet_V4_both.json', 'w'), indent=1, default=float)
