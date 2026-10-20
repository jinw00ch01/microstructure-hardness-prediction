"""d1010 N/fresh F6 (label-free; no hardness label read; fixed clean coef_full only defines the clean f7 target).
midLF regression-calibrated f7 with BOTH input families on the SAME degraded copies:
  - re-create exactly the 267 degraded clean TRAIN copies of d1010/C/mid-noise/fidelity.py (combo c1.0_b1.0_kd0.5,
    target noise rng default_rng(20261010).choice(midLF train ic_noise), seed 5000 + j), compute src.features.extract_v3
    on each copy (134 ic_ columns) -> joined with fid_feats_c1.0_b1.0_kd0.5.parquet (7 pipelines x 10 columns).
  - Targets: clean f7 (cleanhead feats_C_train, coef_full) and its fd / size parts.
  - Models (fixed, no tuning): ridge alpha 10 on standardized inputs; LightGBM (g1/t2 settings) on ic_ only and on both.
    Cross-fitted on the shared folds by source.
Real midLF rows: inputs = features_v3 ic_ + mid-noise mid_feats_{train,test}; full-data model -> f7-hat; corr with v26L
(predictions only), sd, domain AUC copies vs real midLF train (both input families).
python f6_rc_mid2.py -> f6_copies_v3.parquet, f6_rc_mid2.json"""
import os, sys, json, time
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, lightgbm as lgb
from PIL import Image
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
REPO = '/home/claude/microstructure-hardness-prediction'
CACHE = '/mnt/project-files/work/hardness-cache'
MID = CACHE + '/scripts/d1010/C/mid-noise/'
CH = CACHE + '/scripts/d1009/J/cleanhead/'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
COEF = json.load(open(f'{CACHE}/scripts/d1010/L/l_test_result.json'))['coef_full']
TERMS = ['fd_area', 'pore_frac', 'logasp_aw', 'S_align', 'logN_eff', 'het4', 'bd_frac']
F = pd.read_parquet(CH + 'feats_C_train.parquet'); F['logN_eff'] = np.log(F.N_eff); F['bd_frac'] = F.N_bd / F.N_all
ids = F.index.tolist()
v3 = pd.read_parquet(REPO + '/data/features_v3.parquet').set_index('ID')
IC = [c for c in v3.columns if c.startswith('ic_')]

def one(i, t, seed):
    sys.path.insert(0, MID)
    from calib_degrade import degrade
    from src.features import extract_v3
    im8 = np.array(Image.open(f'{REPO}/data/train/{i}.png'))
    d = degrade(im8, 1.0, 1.0, t, seed, i=i, kd=0.5)
    v = extract_v3('x', img=d); v.pop('ID')
    return v

out_fp = f'{HERE}/f6_copies_v3.parquet'
if not os.path.exists(out_fp):
    tr = pd.read_csv(REPO + '/data/train.csv'); x = v3.loc[tr.ID]
    midlf_noise = x[(x.ic_noise >= 9.5) & (x.ic_noise < 12) & (x.ic_ridge_snr < 0.5)].ic_noise.values
    rng = np.random.default_rng(20261010); tgt = rng.choice(midlf_noise, len(ids), replace=True)
    t0 = time.time()
    V = pd.DataFrame(Parallel(n_jobs=2)(delayed(one)(i, float(t), 5000 + j) for j, (i, t) in enumerate(zip(ids, tgt))),
                     index=pd.Index(ids, name='ID'))
    V.to_parquet(out_fp); print('extract', time.time() - t0, flush=True)
V = pd.read_parquet(out_fp)[IC].astype(float)
V = V.replace([np.inf, -np.inf], np.nan); vmed = V.median().fillna(0); V = V.fillna(vmed)
D = pd.read_parquet(MID + 'fid_feats_c1.0_b1.0_kd0.5.parquet').loc[ids]
Rtr = pd.read_parquet(MID + 'mid_feats_train.parquet'); Rte = pd.read_parquet(MID + 'mid_feats_test.parquet')
PC = [c for c in D.columns if c in Rtr.columns]
def prep_p(X):
    X = X[PC].astype(float).copy()
    for c in X.columns:
        if c.endswith('__N_all'):
            X[c] = np.log(X[c].clip(lower=1))
    return X
P = prep_p(D); med = P.median(); P = P.fillna(med)
f7 = COEF['intercept'] + sum(COEF[t] * F[t] for t in TERMS)
tg = {'f7': f7, 'p_fd': COEF['fd_area'] * F.fd_area,
      'p_size': COEF['logN_eff'] * F.logN_eff + COEF['bd_frac'] * F.bd_frac + COEF['het4'] * F.het4}
folds = pd.read_csv(REPO + '/data/folds.csv').set_index('ID').fold.loc[ids].values
mk_r = lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0))
mk_g = lambda: lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=10, subsample=0.8,
                                 subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=2, random_state=0)
INP = {'ridge_pipes': (P, mk_r), 'ridge_ic': (V, mk_r), 'ridge_both': (pd.concat([P, V], axis=1), mk_r),
       'lgb_ic': (V, mk_g), 'lgb_both': (pd.concat([P, V], axis=1), mk_g)}
rep = {'n': len(ids)}
oofs = {}
for nm, (X, mk) in INP.items():
    rep[nm] = {}
    for tn, y in tg.items():
        oof = np.zeros(len(y))
        for k in range(5):
            m = mk().fit(X.values[folds != k], y.values[folds != k]); oof[folds == k] = m.predict(X.values[folds == k])
        rep[nm][tn] = dict(corr=float(np.corrcoef(oof, y)[0, 1]), r2=float(1 - ((y - oof) ** 2).mean() / y.var(ddof=0)),
                           rmse=float(np.sqrt(((y - oof) ** 2).mean())))
        if tn == 'f7':
            oofs[nm] = oof
# equal average of ridge_both and lgb_both (fixed, not chosen)
avg = 0.5 * (oofs['ridge_both'] + oofs['lgb_both'])
rep['avg_ridge_lgb_both'] = {'f7': dict(corr=float(np.corrcoef(avg, f7)[0, 1]), r2=float(1 - ((f7 - avg) ** 2).mean() / f7.var(ddof=0)),
                                        rmse=float(np.sqrt(((f7 - avg) ** 2).mean())))}
# by clean N tercile for the avg
terc = pd.qcut(F.N_eff, 3, labels=['coarse', 'mid', 'fine']).values
rep['avg_by_tercile'] = {t: dict(corr=float(np.corrcoef(avg[terc == t], f7[terc == t])[0, 1]),
                                 rmse=float(np.sqrt(((f7[terc == t] - avg[terc == t]) ** 2).mean()))) for t in ['coarse', 'mid', 'fine']}
# real midLF rows
def midlf(ix):
    x = v3.loc[ix]; return x.index[(x.ic_noise >= 9.5) & (x.ic_noise < 12) & (x.ic_ridge_snr < 0.5)]
itr, ite = midlf(Rtr.index), midlf(Rte.index)
Xr = {s: pd.concat([prep_p(R.loc[ii]).fillna(med), v3.loc[ii, IC].astype(float).replace([np.inf, -np.inf], np.nan).fillna(vmed)], axis=1) for s, R, ii in [('train', Rtr, itr), ('test', Rte, ite)]}
Xall = pd.concat([P, V], axis=1)
fr = mk_r().fit(Xall.values, f7.values); fg = mk_g().fit(Xall.values, f7.values)
S = CACHE + '/submissions/'
pv = {'train': pd.read_csv(S + 'blend_v26L_oof.csv').set_index('ID').hardness, 'test': pd.read_csv(S + 'blend_v26L.csv').set_index('ID').hardness}
for s in ['train', 'test']:
    h = 0.5 * (fr.predict(Xr[s].values) + fg.predict(Xr[s].values)); pp = pv[s].loc[Xr[s].index].values
    rep[f'real_{s}'] = dict(n=int(len(h)), f7hat_mean=float(h.mean()), f7hat_sd=float(h.std()), v26L_sd=float(pp.std()),
                            corr_v26L=float(np.corrcoef(h, pp)[0, 1]), change_rms_w025=float(np.sqrt(((0.25 * (h - pp)) ** 2).mean())),
                            change_rms_w05=float(np.sqrt(((0.5 * (h - pp)) ** 2).mean())))
    pd.DataFrame({'ID': Xr[s].index, 'f7hat': h, 'v26L': pp}).to_parquet(f'{HERE}/f6_real_{s}.parquet')
for nm, X in [('ic', V), ('both', Xall)]:
    Z = pd.concat([X, (Xr['train'][X.columns])]); lab = np.r_[np.zeros(len(X)), np.ones(len(Xr['train']))]
    pr = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)), Z.values, lab,
                           cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
    rep[f'domain_auc_{nm}'] = float(roc_auc_score(lab, pr))
json.dump(rep, open(f'{HERE}/f6_rc_mid2.json', 'w'), indent=1)
print(json.dumps(rep, indent=1))
