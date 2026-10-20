"""CANDIDATE builder for L1 idea 1 (NOT a submission; writes only into this folder).
blend_v28n with the 519 clean test rows (raw features_v3 ic_noise < 9.5) re-mixed as
    0.5 * blend_v24 + 0.5 * f8'   where f8' = f8 with het4 replaced by the re-measured origin-grid het (--het).
--het h08  : het4 on the sigma-0.8 map (pores refilled), p = 0.5         (s14 column het4 @ 0.8)
--het qavg : mean over sigma 0.7/0.8/0.9 of the p = 0.25 grid het         (s14 column het4q)
--het V4   : mean over 6 maps (sigma 0.7/0.8/0.9 x tf 0.5/0.6) of the p = 0.25 grid het (s26 train, s28 test)
f8' is fitted on all 267 clean train rows (OLS with intercept) and only applied to test rows. Every other row of
blend_v28n is copied bit-identically. Sanity first: f8 (unchanged) must reproduce v28n's clean test rows to 1e-6.
Also writes the clean-row nested OOF (shared folds, w 0.5) for the same design and the json of train-side numbers.
Test images enter only through label-free per-image features (s14_test.parquet, cleanhead feats_C_test,
residual-probe feats_avg5_test); nothing is fitted on test rows; no test-prediction statistic is used."""
import sys, json, argparse, hashlib
sys.path.insert(0, '/mnt/project-files/work/hardness-cache/scripts/d1011/bold/L1_formula')
from l1common import *

ap = argparse.ArgumentParser(); ap.add_argument('--het', default='h08', choices=['h08', 'qavg', 'V4'])
ap.add_argument('--lv', action='store_true', help='also re-mix the 42 LV test rows (v26L rule) with the re-measured het')
a = ap.parse_args()
C = load(); y, fold, D, o24, ids = C['y'], C['fold'], C['D'], C['o24'], C['ids']
sub = CACHE + '/submissions/'
te = pd.read_csv(REPO + '/data/sample_submission.csv')
v3 = pd.read_parquet(REPO + '/data/features_v3.parquet').set_index('ID')
clean_te = v3.loc[te.ID, 'ic_noise'].values < 9.5
tid = te.ID.values[clean_te]
assert clean_te.sum() == 519
v28 = pd.read_csv(sub + 'blend_v28n.csv', float_precision='round_trip').set_index('ID').hardness.loc[te.ID].values
v24 = pd.read_csv(sub + 'blend_v24.csv', float_precision='round_trip').set_index('ID').hardness.loc[te.ID].values
Ft = pd.read_parquet(CH + 'feats_C_test.parquet').loc[tid]
At = pd.read_parquet(RP + 'feats_avg5_test.parquet').loc[tid]
Dt = pd.DataFrame(index=tid)
for c in ['fd_area', 'pore_frac', 'logasp_aw', 'S_align', 'het4']:
    Dt[c] = Ft[c].astype(float)
Dt['logN_eff'] = np.log(Ft.N_eff.astype(float)); Dt['bd_frac'] = Ft.N_bd / Ft.N_all; Dt['ringall_avg5'] = At.ringall_avg5
# sanity: reproduce v28n clean test rows with the unchanged f8
b8 = ols_fit(D[F8].values, y)
rep = 0.5 * v24[clean_te] + 0.5 * ols_pred(b8, Dt[F8].values)
dmax = float(np.abs(rep - v28[clean_te]).max())
print('reproduce v28n clean test rows with f8: max diff %.2e' % dmax)
assert dmax < 1e-6, 'v28n clean rows not reproduced; stop'


def het_col(split, idx):
    if a.het == 'V4':
        f = HERE + ('s26_train.parquet' if split == 'train' else 's28_V4_clean_test.parquet')
        return pd.read_parquet(f).loc[idx, 'V4'].values
    L = pd.read_parquet(HERE + f's14_{split}.parquet')
    if a.het == 'h08':
        return L.pivot(index='ID', columns='sig', values='het4').loc[idx, 0.8].values
    q = L.pivot(index='ID', columns='sig', values='het4q').loc[idx]
    return ((q[0.7] + q[0.8] + q[0.9]) / 3).values


Dn, Dtn = D.copy(), Dt.copy()
Dn['het4'] = het_col('train', ids); Dtn['het4'] = het_col('test', tid)
b = ols_fit(Dn[F8].values, y)
new = v28.copy()
new[clean_te] = 0.5 * v24[clean_te] + 0.5 * ols_pred(b, Dtn[F8].values)
assert np.isfinite(new).all()
oof = 0.5 * o24 + 0.5 * xfit_ols(Dn[F8].values, y, fold)
lvinfo = None
if a.lv:
    # v28n LV rows = 0.5*v25 + 0.5*f7 (clean-fitted on all 267 rows, cleanhead sigma-1.0 het4; d1010/L). Reproduce first.
    v25 = pd.read_csv(sub + 'blend_v25.csv', float_precision='round_trip').set_index('ID').hardness.loc[te.ID].values
    FL = pd.read_parquet(CACHE + '/scripts/d1010/L/feats_LV_test.parquet'); FL = FL[FL.group == 'LV']
    lv_te = te.ID.isin(FL.index).values; assert lv_te.sum() == 42
    lid = te.ID.values[lv_te]; FL = FL.loc[lid]
    def d7(Fx, het):
        X = pd.DataFrame(index=Fx.index)
        for c in ['fd_area', 'pore_frac', 'logasp_aw', 'S_align']:
            X[c] = Fx[c].astype(float)
        X['logN_eff'] = np.log(Fx.N_eff.astype(float)); X['het4'] = het; X['bd_frac'] = Fx.N_bd / Fx.N_all
        return X[F7].values
    b7 = ols_fit(D[F7].values, y)
    rep7 = 0.5 * v25[lv_te] + 0.5 * ols_pred(b7, d7(FL, FL.het4.values))
    dlv = float(np.abs(rep7 - v28[lv_te]).max()); print('reproduce v28n LV test rows: max diff %.2e' % dlv); assert dlv < 1e-6
    LT = pd.read_parquet(HERE + 's23_LV_test.parquet')
    if a.het == 'V4':
        hl = pd.read_parquet(HERE + 's28_V4_LV_test.parquet').loc[lid, 'V4'].values
    elif a.het == 'h08':
        hl = LT.pivot(index='ID', columns='sig', values='het4').loc[lid, 0.8].values
    else:
        ql = LT.pivot(index='ID', columns='sig', values='het4q').loc[lid]; hl = ((ql[0.7] + ql[0.8] + ql[0.9]) / 3).values
    b7n = ols_fit(Dn[F7].values, y)
    new[lv_te] = 0.5 * v25[lv_te] + 0.5 * ols_pred(b7n, d7(FL, hl))
    lvinfo = dict(rows=int(lv_te.sum()), reproduce_maxdiff=dlv, coef_f7_new=dict(zip(['intercept'] + F7, b7n.tolist())))
    clean_te = clean_te | lv_te
out = HERE + f'CANDIDATE_v28n_gridhet_{a.het}{"_lv" if a.lv else ""}.csv'
pd.DataFrame({'ID': te.ID, 'hardness': new}).to_csv(out, index=False)
pd.Series(oof, index=pd.Index(ids, name='ID'), name='hardness').to_csv(out.replace('.csv', '_oof_clean.csv'))
info = dict(het=a.het, lv=lvinfo, coef_full=dict(zip(['intercept'] + F8, b.tolist())), coef_f8_full=dict(zip(['intercept'] + F8, b8.tolist())),
            clean_nested_oof_rmse=rmse(oof - y), v28n_clean=rmse(C['o28'] - y), rows_changed=int(clean_te.sum()),
            other_rows_identical=bool(np.array_equal(new[~clean_te], v28[~clean_te])),
            md5=hashlib.md5(open(out, 'rb').read()).hexdigest(), reproduce_maxdiff=dmax)
json.dump(info, open(out.replace('.csv', '.json'), 'w'), indent=1)
print(json.dumps(info, indent=1))
