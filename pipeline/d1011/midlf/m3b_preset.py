"""d1011 midlf M3b (label-free; no hardness label of any LV / midLF / G1hi row is read; coef_full only defines the f7
target of the clean copies).

Hypothesis (label-free facts in m3_facts below): the midLF group (9.5 <= ic_noise < 12, ic_ridge_snr < 0.5) is a MIXTURE of
 (CP) clean-preset images whose drawn boundary lines are drowned by noise (the same image type as LV rows, minus visible
      lines; clean rows show a line-free share that rises with noise: 0.05 at 6-7 -> 0.38 at 9-9.5), and
 (NP) noisy-preset images (blur, compressed dark-phase contrast).
If so, the v26L lesson (unchanged clean formula7 works where the clean pipeline measures its terms) applies to the CP part.

Gate defined ONLY from degraded clean TRAIN copies (no midLF label; no test image used for any choice):
  CP copies: m3a_cp_copies.parquet (noise only), NP copies: midLF-look copies (N/fresh f6_copies_v3 + N/gate sim_mid);
  same 267 sources, same noise targets and seeds; the two copies of a source differ only by blur 1.0 + kd 0.5.
  Descriptors (fixed): the five degradation-calibration statistics except noise and ridge:
     D5 = ic_spec_slope, ic_spec_32_48, ic_gmm_sep, ic_std, ic_acg_ac8_par.
  (ridge is excluded on purpose: midLF is selected on ridge < 0.5, so ridge would push CP rows toward NP; noise is
   excluded so that the noise profile of the gated rows is an independent check.)
  Classifier: standardized logistic regression, C = 1.0 (fixed), cross-fitted by source fold (both copies of a source in
  the same fold). Gate: P(CP) >= 0.5 (fixed).
Reports: OOF AUC on copies; fidelity of the unchanged clean pipeline (V0 terms, coef_full) on CP copies, NP copies, and
the gated mixture; P(CP) on real groups (LV = known clean preset, clean rows by noise band, G1hi = noisy preset);
midLF CP / NP counts by split; noise profile of gated midLF rows; mixture share pi in real midLF TRAIN rows (EM on the
classifier logit with the copies' class-conditional densities); expected precision of the gate; overlap with the N/gate
G_both gate; label-free change size of the v26L recipe (0.5 * (f7_V0 - v28n)) on gated test rows.
python m3b_preset.py -> m3b_preset.json, m3b_rows.parquet (per-row gate, train+test, label-free)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import sys
sys.dont_write_bytecode = True
import json, hashlib
import numpy as np, pandas as pd
from scipy import stats as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

REPO = '/home/claude/microstructure-hardness-prediction'
CACHE = '/mnt/project-files/work/hardness-cache'
D10 = CACHE + '/scripts/d1010/'
GATE = D10 + 'N/gate/'
CH = CACHE + '/scripts/d1009/J/cleanhead/'
S = CACHE + '/submissions/'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE + '/m3b_preset.json'
OUTR = HERE + '/m3b_rows.parquet'
assert not os.path.exists(OUT) and not os.path.exists(OUTR), 'refusing to overwrite'
sys.path.insert(0, GATE)
import gate_lib as L  # noqa: E402


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


TERMS = L.TERMS
cf = json.load(open(D10 + 'L/l_test_result.json'))['coef_full']
b = np.array([cf['intercept']] + [cf[t] for t in TERMS])
D5 = ['ic_spec_slope', 'ic_spec_32_48', 'ic_gmm_sep', 'ic_std', 'ic_acg_ac8_par']

F = pd.read_parquet(CH + 'feats_C_train.parquet')
ids = F.index.tolist()
f7o = pd.Series(L.f7_apply(b, L.design(F).values), index=ids)
CPc = pd.read_parquet(HERE + '/m3a_cp_copies.parquet').loc[ids]
NPv = pd.read_parquet(D10 + 'N/fresh/f6_copies_v3.parquet').loc[ids]
NPq = pd.read_parquet(GATE + 'sim_mid.parquet').loc[ids]
assert np.allclose(CPc.target_noise.values, NPq.target_noise.values)
RCP, RNP = L.reliability(CPc, b), L.reliability(NPq, b)
folds = pd.read_csv(REPO + '/data/folds.csv').set_index('ID').fold
fk = folds.loc[ids].values
rep = {'inputs_sha256': {'m3a_cp_copies': sha(HERE + '/m3a_cp_copies.parquet'),
                         'f6_copies_v3': sha(D10 + 'N/fresh/f6_copies_v3.parquet'),
                         'sim_mid': sha(GATE + 'sim_mid.parquet'), 'feats_C_train': sha(CH + 'feats_C_train.parquet'),
                         'features_v3': sha(REPO + '/data/features_v3.parquet'), 'folds': sha(REPO + '/data/folds.csv'),
                         'blend_v28n': sha(S + 'blend_v28n.csv'), 'blend_v28n_oof': sha(S + 'blend_v28n_oof.csv')}}
rep['copies_noise_check'] = dict(cp_ic_noise_med=float(CPc.ic_noise.median()), np_ic_noise_med=float(NPv.ic_noise.median()),
                                 corr_cp_np_noise=float(np.corrcoef(CPc.ic_noise, NPv.ic_noise)[0, 1]))


def fid(g, t):
    g, t = np.asarray(g, float), np.asarray(t, float)
    if len(g) < 5:
        return dict(n=int(len(g)))
    e = g - t
    return dict(n=int(len(g)), rms=float(np.sqrt(np.mean(e ** 2))), bias=float(e.mean()), sd_e=float(e.std()),
                corr=float(np.corrcoef(g, t)[0, 1]), kappa_sim=float(np.cov(t, g)[0, 1] / np.var(g, ddof=1)))


# ------------------------------------------------------------------ classifier on copies
X = pd.concat([CPc[D5], NPv[D5]]).astype(float).values
y = np.r_[np.ones(len(ids)), np.zeros(len(ids))]
fk2 = np.r_[fk, fk]
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
oof = np.zeros(len(y))
for k in range(5):
    m = mk().fit(X[fk2 != k], y[fk2 != k])
    oof[fk2 == k] = m.predict_proba(X[fk2 == k])[:, 1]
full = mk().fit(X, y)
rep['classifier'] = dict(desc=D5, C=1.0, oof_auc=float(roc_auc_score(y, oof)),
                         coef_std=dict(zip(D5, [float(c) for c in full[-1].coef_.ravel()])),
                         cp_copies_pass=float((oof[:len(ids)] >= 0.5).mean()),
                         np_copies_pass=float((oof[len(ids):] >= 0.5).mean()))
# single-feature robustness (spec slope only)
oof1 = np.zeros(len(y))
for k in range(5):
    m = mk().fit(X[fk2 != k][:, :1], y[fk2 != k])
    oof1[fk2 == k] = m.predict_proba(X[fk2 == k][:, :1])[:, 1]
rep['classifier_spec_only_oof_auc'] = float(roc_auc_score(y, oof1))

# ------------------------------------------------------------------ fidelity of the unchanged pipeline
pc, pn = oof[:len(ids)], oof[len(ids):]
nz = CPc.ic_noise.values
fidr = {'CP_all': fid(RCP.f7_V0, f7o), 'NP_all': fid(RNP.f7_V0, f7o),
        'CP_noise_le_10.5': fid(RCP.f7_V0[nz <= 10.5], f7o[nz <= 10.5]),
        'CP_noise_gt_10.5': fid(RCP.f7_V0[nz > 10.5], f7o[nz > 10.5]),
        'CP_gated': fid(RCP.f7_V0[pc >= 0.5], f7o[pc >= 0.5]),
        'NP_gated_false_pass': fid(RNP.f7_V0[pn >= 0.5], f7o[pn >= 0.5]),
        'NP_rejected': fid(RNP.f7_V0[pn < 0.5], f7o[pn < 0.5])}
Sl = pd.read_parquet(GATE + 'sim_lv.parquet').loc[ids]
fidr['ref_LV_look_copies'] = fid(L.reliability(Sl, b).f7_V0, f7o)
rep['fidelity_unchanged_pipeline'] = fidr
rep['stability_copies_median'] = {nm: dict(W=float(R.W.median()), gap=float(R.gap.median()), nfb_var=float(R.nfb_var.median()),
                                           fallback=float(R.fallback.mean()), N_all_V0=float(Q['V0__N_all'].median()))
                                  for nm, R, Q in (('CP', RCP, CPc), ('NP', RNP, NPq))}

# ------------------------------------------------------------------ real rows
v3 = pd.read_parquet(REPO + '/data/features_v3.parquet').set_index('ID')
Rtr, Rte = pd.read_parquet(GATE + 'rel_train.parquet'), pd.read_parquet(GATE + 'rel_test.parquet')
v28 = {'train': pd.read_csv(S + 'blend_v28n_oof.csv').set_index('ID').hardness,
       'test': pd.read_csv(S + 'blend_v28n.csv').set_index('ID').hardness}
rows = []
for s, R in (('train', Rtr), ('test', Rte)):
    x = v3.loc[R.index]
    p = full.predict_proba(x[D5].astype(float).values)[:, 1]
    df = pd.DataFrame({'split': s, 'group': R.group.values, 'ic_noise': x.ic_noise.values,
                       'ic_ridge_snr': x.ic_ridge_snr.values, 'p_cp': p, 'f7_V0': R.f7_V0.values, 'W': R.W.values,
                       'gap': R.gap.values, 'nfb_var': R.nfb_var.values, 'fallback': R.fallback.values,
                       'G_both': R.G_both.values, 'mahal': R.mahal.values, 'v28n': v28[s].loc[R.index].values},
                      index=R.index)
    rows.append(df)
Rr = pd.concat(rows)
Rr['gate_cp'] = Rr.p_cp >= 0.5
Rr.to_parquet(OUTR)
real = {}
for s in ('train', 'test'):
    for grp in ('clean', 'LV', 'midLF', 'HLV', 'G1hi'):
        d = Rr[(Rr.split == s) & (Rr.group == grp)]
        real[f'{grp}_{s}'] = dict(n=int(len(d)), cp_pass=int(d.gate_cp.sum()), p_cp_med=float(d.p_cp.median()))
# clean rows by noise band (train+test): does the classifier call the clean preset 'CP' at high clean noise?
for lo, hi in ((0, 7), (7, 8.5), (8.5, 9.5)):
    d = Rr[(Rr.group == 'clean') & (Rr.ic_noise >= lo) & (Rr.ic_noise < hi)]
    for lf in (True, False):
        dd = d[(d.ic_ridge_snr < 0.5) == lf]
        real[f'clean_{lo}-{hi}_{"LF" if lf else "lines"}'] = dict(n=int(len(dd)), cp_rate=float(dd.gate_cp.mean()) if len(dd) else None)
rep['real_pass'] = real

# midLF: noise profile of gated vs rejected rows, and by noise band
M = Rr[Rr.group == 'midLF']
prof = {}
for lo, hi in ((9.5, 10.0), (10.0, 10.5), (10.5, 11.0), (11.0, 11.5), (11.5, 12.0)):
    d = M[(M.ic_noise >= lo) & (M.ic_noise < hi)]
    prof[f'{lo}-{hi}'] = dict(n=int(len(d)), cp_pass=int(d.gate_cp.sum()), mean_p_cp=float(d.p_cp.mean()) if len(d) else None)
rep['midLF_by_noise'] = prof
rep['midLF_noise_median'] = dict(cp=float(M[M.gate_cp].ic_noise.median()), np=float(M[~M.gate_cp].ic_noise.median()),
                                 LV=float(Rr[Rr.group == 'LV'].ic_noise.median()),
                                 mannwhitney_p_cp_lt_np=float(st.mannwhitneyu(M[M.gate_cp].ic_noise, M[~M.gate_cp].ic_noise,
                                                                              alternative='less').pvalue))
# label-free stability of gated midLF rows vs LV rows
stab = {}
for nm, d in (('midLF_CP', M[M.gate_cp]), ('midLF_NP', M[~M.gate_cp]), ('LV', Rr[Rr.group == 'LV']),
              ('clean_8.5-9.5', Rr[(Rr.group == 'clean') & (Rr.ic_noise >= 8.5)])):
    stab[nm] = dict(n=int(len(d)), W_med=float(d.W.median()), gap_med=float(d.gap.median()),
                    nfb_var_med=float(d.nfb_var.median()), fallback_rate=float(d.fallback.mean()),
                    ridge_med=float(d.ic_ridge_snr.median()), mahal_med=float(d.mahal.median()))
rep['real_stability'] = stab
# overlap with the N/gate G_both gate and train-vs-test pass-rate check
rep['overlap_with_G_both'] = {s: dict(cp=int(M[(M.split == s)].gate_cp.sum()), G_both=int(M[(M.split == s)].G_both.sum()),
                                      both=int((M[(M.split == s)].gate_cp & M[(M.split == s)].G_both).sum()))
                              for s in ('train', 'test')}
a, bb = M[M.split == 'train'].gate_cp, M[M.split == 'test'].gate_cp
rep['midLF_pass_fisher_train_vs_test'] = float(st.fisher_exact([[a.sum(), (~a).sum()], [bb.sum(), (~bb).sum()]])[1])

# ------------------------------------------------------------------ mixture share pi (EM on the logit, train midLF only)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
kc, kn = st.gaussian_kde(lg(pc)), st.gaussian_kde(lg(pn))
zt = lg(M[M.split == 'train'].p_cp.values)
fc, fn = kc(zt), kn(zt)
pi = 0.5
for _ in range(500):
    r = pi * fc / (pi * fc + (1 - pi) * fn)
    pi = float(r.mean())
tpr, fpr = float((pc >= 0.5).mean()), float((pn >= 0.5).mean())
prec = pi * tpr / (pi * tpr + (1 - pi) * fpr) if pi > 0 else 0.0
# bootstrap of pi over the 47 train rows
rng = np.random.default_rng(20261011)
pis = []
for _ in range(500):
    zz = rng.choice(zt, len(zt), replace=True)
    a_, b_ = kc(zz), kn(zz)
    q = 0.5
    for _ in range(200):
        q = float((q * a_ / (q * a_ + (1 - q) * b_)).mean())
    pis.append(q)
rep['mixture'] = dict(pi_train=pi, pi_boot_q05_q50_q95=[float(v) for v in np.quantile(pis, [0.05, 0.5, 0.95])],
                      tpr_copies=tpr, fpr_copies=fpr, expected_precision_train=float(prec),
                      expected_fidelity_gated_rows_rms=float(np.sqrt(prec * fidr['CP_gated'].get('rms', np.nan) ** 2 +
                                                                      (1 - prec) * fidr['NP_gated_false_pass'].get('rms', np.nan) ** 2))
                      if fidr['NP_gated_false_pass'].get('n', 0) >= 5 else None)
# domain check: CP copies vs real LV rows (both 'clean preset + noise') on D5 (5-fold CV logistic)
from sklearn.model_selection import cross_val_predict, StratifiedKFold
for nm, real_rows in (('CPcopies_vs_LV_train_test', Rr[Rr.group == 'LV']),
                      ('NPcopies_vs_G1hi_train', Rr[(Rr.group == 'G1hi') & (Rr.split == 'train')]),
                      ('NPcopies_vs_midLF_NP_train', M[(M.split == 'train') & ~M.gate_cp]),
                      ('CPcopies_vs_midLF_CP_train', M[(M.split == 'train') & M.gate_cp])):
    base = CPc if nm.startswith('CP') else NPv
    Z = np.r_[base[D5].astype(float).values, v3.loc[real_rows.index, D5].astype(float).values]
    lab = np.r_[np.zeros(len(base)), np.ones(len(real_rows))]
    if len(real_rows) >= 5:
        pr = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)), Z, lab,
                               cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
        rep.setdefault('domain_auc', {})[nm] = dict(n_real=int(len(real_rows)), auc=float(roc_auc_score(lab, pr)))

# ------------------------------------------------------------------ v26L-recipe change size on gated midLF rows
for s in ('train', 'test'):
    d = M[(M.split == s) & M.gate_cp]
    ch = 0.5 * (d.f7_V0 - d.v28n)
    rep[f'recipe_change_{s}'] = dict(n=int(len(d)), rms=float(np.sqrt((ch ** 2).mean())) if len(d) else None,
                                     mean=float(ch.mean()) if len(d) else None, max_abs=float(ch.abs().max()) if len(d) else None,
                                     corr_f7_v28n=float(np.corrcoef(d.f7_V0, d.v28n)[0, 1]) if len(d) > 3 else None,
                                     n_mahal_gt_clean_p99=int((d.mahal > 5.982726487820969).sum()))
# per-fold counts of gated train rows
ft = folds.loc[M[(M.split == 'train') & M.gate_cp].index]
rep['gated_train_rows_per_fold'] = {int(k): int(v) for k, v in ft.value_counts().sort_index().items()}
json.dump(rep, open(OUT, 'w'), indent=1, default=float)
print(json.dumps(rep, indent=1, default=float))
