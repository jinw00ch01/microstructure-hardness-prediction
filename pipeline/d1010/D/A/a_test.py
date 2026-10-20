"""d1010 D / FILE A: the ONE registered nested bar for blend_v28n (D/A/PREREG.txt, registered 2026-10-10 13:46 KST).
Run ONCE. Its sha256 must be recorded in D/A/a_test.sha256 before the run (the script refuses to run otherwise).

Rule (blend_v28n = blend_v27u with a half-step slope stretch on the LOW-NOISE half of the G1hi rows):
  G1hi = raw features_v3 ic_noise >= 12 AND ic_ridge_snr < 0.5            (assert 161 train / 320 test)
  m    = median raw ic_noise over the train G1hi rows                    (assert 16.280704 +- 1e-6)
  file rows = G1hi AND ic_noise < m                                      (assert 80 train / 167 test; others 420 / 833)
  over the file's train rows: c = mean of the blend_v27u nested OOF      (assert 192.9155)
                              b = OLS slope, with intercept, of y on that OOF (assert 1.4922)
  rho = 0.9795141681843882 (fixed constant, D/mech m4), k = 1 + 0.5 * (b / rho - 1) (assert 1.26171)
  new = c + k * (p - c), p = blend_v27u, on the file rows; every other row bit-identical to blend_v27u.
Nested bar: for each outer fold f of data/folds.csv recompute m_f, the low-half rows, c_f, b_f, k_f from the rows with
fold != f only (rho fixed); set new = c_f + k_f (p - c_f) on the fold-f rows with G1hi and ic_noise < m_f.
Required (all five):
  (a) all-500 nested MSE change vs the blend_v27u OOF <= -1.00;
  (b) SSE lower on the changed held-out rows in >= 3 of the 5 folds (a fold with no changed row counts as not better);
  (c) paired sign-flip p < 0.20 (one-sided, 20000 flips, seed 20261010, d_i = (new_i - y_i)^2 - (p_i - y_i)^2 over the
      changed rows; same routine as d1010/C/harness.py and L/l_test.py);
  (d) after dropping the 3 rows with the largest SSE gain (no refit), the all-500 MSE change is still <= 0
      (mean of d over the 497 remaining rows; its sign equals that of the sum);
  (e) 300 stratified 5-fold re-splits (seed 20261010; strata clean / LV / midLF / HLV / G1hi; the design audits' split
      routine: per stratum shuffle, fold = (rank + one random offset) mod 5; same nested procedure):
      mean all-500 MSE change <= -0.50 and P(change >= 0) <= 0.10.
Report only (never selectable): per-fold m_f, c_f, b_f, k_f and changed-row counts; the in-sample all-500 change; the
LOO-EB and B_lo_eb factors (skeptic-rules r7 / r4 definitions) and the splits at the train G1hi 40% / 60% noise quantiles,
each under the same harness (fixed folds + the same 300 re-splits); the full step b / rho.
Labels: read once (src.common.load_train), after every label-free check. The bar uses only G1hi labels (d = 0 on
every other row); the report-only EB factors also use the clean / LV / midLF slopes, as r7 defines them.
No test prediction is read here; test rows enter only as label-free per-image gate counts from features_v3.
Writes ONLY a_test_result.json and a_test_oof.csv next to this file; refuses to overwrite either.
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
HERE = os.path.dirname(os.path.abspath(__file__))
D_DIR = os.path.dirname(HERE)
PREREG_SHA256 = '123b281da8a2359999e89d42278d44e366cdea68341446afa590c3043a5c9b88'
MD5 = {'blend_v27u.csv': '97a7cf2bbec5e90a65f8f7e388f4819f', 'blend_v27u_oof.csv': 'c0a2cda36bfc0e0394bb332f2ee54338'}
RHO = 0.9795141681843882
SEED, N_FLIP, N_SPLIT, N_DROP = 20261010, 20000, 300, 3
NOISE_HI, RIDGE_LO = 12.0, 0.5
# design values asserted by the PREREG (tolerance = half a unit of the last stated digit; m as stated)
EXPECT = dict(n_G1hi_train=161, n_G1hi_test=320, n_file_train=80, n_file_test=167, n_other_train=420,
              n_other_test=833, m=16.280704, c=192.9155, b=1.4922, k=1.26171)
TOL = dict(m=1e-6, c=5e-5, b=5e-5, k=5e-6)
BAR = dict(a_max_all500_dmse=-1.00, b_min_folds_better=3, c_max_signflip_p=0.20, d_n_drop=N_DROP,
           d_max_all500_dmse_after_drop=0.0, e_max_resplit_mean=-0.50, e_max_resplit_P_ge0=0.10)
REPORT_STATED = dict(k_loo_eb=1.2567, k_b_lo_eb=1.3549, k_full_step=1.5234)
STRATA = ('clean', 'LV', 'midLF', 'HLV', 'G1hi')


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def groups(nz, rs):
    return dict(clean=nz < 9.5, LV=(nz >= 9.5) & (nz < 12) & (rs >= 0.5), midLF=(nz >= 9.5) & (nz < 12) & (rs < 0.5),
                HLV=(nz >= NOISE_HI) & (rs >= RIDGE_LO), G1hi=(nz >= NOISE_HI) & (rs < RIDGE_LO))


def ols_slope(p, y):
    """OLS slope, with intercept, of y on p."""
    X = np.column_stack([np.ones(len(p)), p])
    return float(np.linalg.lstsq(X, y, rcond=None)[0][1])


def hc1(p, y):
    """OLS slope with intercept and its HC1 SE (copy of skeptic-rules r4 / r7 hc1)."""
    n = len(p); X = np.c_[np.ones(n), p]; be = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ be
    XtXi = np.linalg.inv(X.T @ X); V = XtXi @ (X.T * e ** 2) @ X @ XtXi * n / (n - 2)
    return float(be[1]), float(np.sqrt(V[1, 1]))


def half_step(b):
    return 1.0 + 0.5 * (b / RHO - 1.0)


def signflip_p(d, seed=SEED, nflip=N_FLIP, chunk=5000):
    """One-sided: small p when mean(d) is negative (copy of d1010/C/harness.py _signflip_p)."""
    rng = np.random.default_rng(seed)
    obs, n, cnt, done = d.mean(), len(d), 0, 0
    while done < nflip:
        c = min(chunk, nflip - done)
        s = rng.integers(0, 2, size=(c, n), dtype=np.int8).astype(float) * 2 - 1
        cnt += int(np.sum((s @ d) / n <= obs + 1e-12))
        done += c
    return (1 + cnt) / (nflip + 1)


def threshold(x, q):
    return float(np.median(x)) if q == 0.5 else float(np.quantile(x, q))


class Ctx:
    """Train arrays in data/folds.csv order (= blend_v27u_oof.csv order, asserted)."""

    def __init__(self, po, nz, G, grp, y):
        self.po, self.nz, self.G, self.grp, self.y = po, nz, G, grp, y


def fit_rule(C, fm, form):
    """Factor from the rows in fm only. form: ('half', q) registered for q = 0.5; ('eb', loo) report only."""
    kind, arg = form
    g_f = C.G & fm
    thr = threshold(C.nz[g_f], arg if kind == 'half' else 0.5)
    lo_f = g_f & (C.nz < thr)
    c = float(C.po[lo_f].mean())
    if kind == 'half':
        b = ols_slope(C.po[lo_f], C.y[lo_f])
        return dict(m=thr, n_fit=int(lo_f.sum()), c=c, b=b, k=half_step(b))
    # skeptic-rules r7 eb_k: tau2 from the HC1 slopes of clean / LV / midLF / lo / hi (lo left out when loo)
    subs = dict(clean=C.grp['clean'] & fm, LV=C.grp['LV'] & fm, midLF=C.grp['midLF'] & fm, lo=lo_f,
                hi=g_f & (C.nz >= thr))
    st = {s: hc1(C.po[msk], C.y[msk]) for s, msk in subs.items()}
    contrib = {s: max((b_ - 1) ** 2 - se_ ** 2, 0) for s, (b_, se_) in st.items()}
    tau2 = float(np.mean([contrib[s] for s in st if not (arg and s == 'lo')]))
    b, se = st['lo']; w = tau2 / (tau2 + se ** 2)
    return dict(m=thr, n_fit=int(lo_f.sum()), c=c, b=b, se=se, tau2=tau2, w=w, k=float(1 + w * (b - 1)))


def nested(C, folds_, form):
    q = C.po.copy(); changed = np.zeros(len(q), bool); per = []
    for f in range(5):
        par = fit_rule(C, folds_ != f, form)
        app = C.G & (folds_ == f) & (C.nz < par['m'])
        q[app] = par['c'] + par['k'] * (C.po[app] - par['c'])
        changed |= app
        per.append(dict(fold=f, **par, n_changed=int(app.sum())))
    return q, changed, per


def evaluate(C, q, changed, folds_, ids, do_signflip=True):
    y, po = C.y, C.po
    d = (q - y) ** 2 - (po - y) ** 2
    assert (d[~changed] == 0).all()
    per = []
    for f in range(5):
        mf = changed & (folds_ == f)
        s_new, s_base = float(np.sum((q[mf] - y[mf]) ** 2)), float(np.sum((po[mf] - y[mf]) ** 2))
        per.append(dict(fold=f, n_changed=int(mf.sum()), sse_v27u=s_base, sse_new=s_new, dsse=s_new - s_base,
                        better=bool(mf.any() and s_new < s_base)))
    order = np.argsort(d, kind='stable')                 # most negative d (largest SSE gain) first
    drop = order[:N_DROP]; keep = np.ones(len(d), bool); keep[drop] = False
    res = dict(all500_dmse=float(d.mean()), n_changed=int(changed.sum()), changed_dmse_mean=float(d[changed].mean()),
               folds=per, folds_better=int(sum(p['better'] for p in per)),
               rows_better=int((d[changed] < 0).sum()),
               droptop_ids=[str(ids[i]) for i in drop], droptop_sse_gain=[float(-d[i]) for i in drop],
               droptop_all500_dmse=float(d[keep].mean()), droptop_dsse_sum=float(d[keep].sum()),
               mse_v27u=float(np.mean((po - y) ** 2)), mse_new=float(np.mean((q - y) ** 2)),
               rmse_v27u=float(np.sqrt(np.mean((po - y) ** 2))), rmse_new=float(np.sqrt(np.mean((q - y) ** 2))),
               fold_rmse_v27u=[float(np.sqrt(np.mean((po - y)[folds_ == f] ** 2))) for f in range(5)],
               fold_rmse_new=[float(np.sqrt(np.mean((q - y)[folds_ == f] ** 2))) for f in range(5)])
    res['signflip_p'] = float(signflip_p(d[changed])) if (do_signflip and changed.any()) else None
    return res


def strat(rng, glab):
    """Design audits' re-split (skeptic-rules r4 / r7 strat): per stratum shuffle, fold = (rank + offset) mod 5."""
    f = np.empty(len(glab), int)
    for gl in range(len(STRATA)):
        idx = np.where(glab == gl)[0]; rng.shuffle(idx); f[idx] = (np.arange(len(idx)) + rng.integers(5)) % 5
    return f


def resplits(C, splits, form):
    R = np.array([float(((nested(C, ff, form)[0] - C.y) ** 2 - (C.po - C.y) ** 2).mean()) for ff in splits])
    return dict(n=int(len(R)), mean=float(R.mean()), sd=float(R.std()), P_ge0=float((R >= 0).mean()),
                q05=float(np.quantile(R, 0.05)), q50=float(np.quantile(R, 0.5)), q95=float(np.quantile(R, 0.95)))


def main():
    out_js, out_oof = HERE + '/a_test_result.json', HERE + '/a_test_oof.csv'
    for p in (out_js, out_oof):
        assert not os.path.exists(p), f'refusing to overwrite {p} (the test runs once)'
    sha = hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()
    sha_file = HERE + '/a_test.sha256'
    assert os.path.exists(sha_file), 'a_test.sha256 must be recorded before the single run'
    sha_rec = open(sha_file).read().split()[0]
    assert sha_rec == sha, f'recorded sha256 {sha_rec} != this script {sha}'
    prereg_sha = hashlib.sha256(open(HERE + '/PREREG.txt', 'rb').read()).hexdigest()
    reg = dict(l.split()[::-1] for l in open(D_DIR + '/PREREG.sha256').read().splitlines() if l.strip())
    assert prereg_sha == PREREG_SHA256 == reg[HERE + '/PREREG.txt'], 'PREREG.txt changed since registration'

    # ------------------------------------------------------------ inputs (md5 pins), label-free checks first
    input_md5 = {}
    for fn, m in MD5.items():
        input_md5[fn] = md5(f'{CACHE}/submissions/{fn}')
        assert input_md5[fn] == m, fn
    input_md5['data/folds.csv'] = md5(REPO + '/data/folds.csv')
    input_md5['data/features_v3.parquet'] = md5(REPO + '/data/features_v3.parquet')
    fo = pd.read_csv(REPO + '/data/folds.csv')
    ids = fo.ID.astype(str).to_numpy(); fold = fo.fold.to_numpy().astype(int)
    assert len(ids) == 500 and len(set(ids)) == 500 and set(np.unique(fold)) == set(range(5))
    te_ids = pd.read_csv(REPO + '/data/sample_submission.csv').ID.astype(str).to_numpy()
    assert len(te_ids) == 1000 and len(set(te_ids)) == 1000
    oof = pd.read_csv(f'{CACHE}/submissions/blend_v27u_oof.csv', float_precision='round_trip')
    assert len(oof) == 500 and (oof.ID.astype(str).to_numpy() == ids).all(), 'v27u OOF not in folds.csv order'
    po = oof.hardness.to_numpy().astype(float)
    assert np.isfinite(po).all()
    v3 = pd.read_parquet(REPO + '/data/features_v3.parquet').set_index('ID')
    nz, rs = v3.loc[ids, 'ic_noise'].to_numpy().astype(float), v3.loc[ids, 'ic_ridge_snr'].to_numpy().astype(float)
    nzt, rst = v3.loc[te_ids, 'ic_noise'].to_numpy().astype(float), v3.loc[te_ids, 'ic_ridge_snr'].to_numpy().astype(float)
    assert np.isfinite(nz).all() and np.isfinite(rs).all() and np.isfinite(nzt).all() and np.isfinite(rst).all()
    grp, grpt = groups(nz, rs), groups(nzt, rst)
    cover = np.sum([grp[s] for s in STRATA], 0)
    assert (cover == 1).all() and (np.sum([grpt[s] for s in STRATA], 0) == 1).all(), 'strata do not partition rows'
    glab = np.select([grp[s] for s in STRATA], list(range(len(STRATA))), default=-1)
    G, Gt = grp['G1hi'], grpt['G1hi']
    m = float(np.median(nz[G]))
    file_tr, file_te = G & (nz < m), Gt & (nzt < m)
    counts = dict(G1hi_train=int(G.sum()), G1hi_test=int(Gt.sum()), file_train=int(file_tr.sum()),
                  file_test=int(file_te.sum()), other_train=int((~file_tr).sum()), other_test=int((~file_te).sum()),
                  strata_train={s: int(grp[s].sum()) for s in STRATA})
    assert counts['G1hi_train'] == EXPECT['n_G1hi_train'] and counts['G1hi_test'] == EXPECT['n_G1hi_test'], counts
    assert abs(m - EXPECT['m']) <= TOL['m'], m
    assert counts['file_train'] == EXPECT['n_file_train'] and counts['file_test'] == EXPECT['n_file_test'], counts
    assert counts['other_train'] == EXPECT['n_other_train'] and counts['other_test'] == EXPECT['n_other_test'], counts

    # ------------------------------------------------------------ the single label read
    sys.path.insert(0, REPO)
    from src.common import load_train
    tr = load_train()
    assert len(tr) == 500 and set(tr.ID.astype(str)) == set(ids)
    trx = tr.assign(ID=tr.ID.astype(str)).set_index('ID').loc[ids]
    assert (trx.fold.to_numpy().astype(int) == fold).all(), 'load_train fold != data/folds.csv'
    y = trx.hardness.to_numpy().astype(float)
    assert np.isfinite(y).all()
    C = Ctx(po, nz, G, grp, y)
    ALL = np.ones(500, bool)
    REG = ('half', 0.5)

    # ------------------------------------------------------------ full-data factor (train only), PREREG asserts
    full = fit_rule(C, ALL, REG)
    assert full['m'] == m and full['n_fit'] == EXPECT['n_file_train']
    for key in ('c', 'b', 'k'):
        assert abs(full[key] - EXPECT[key]) <= TOL[key], (key, full[key], EXPECT[key])
    eb_std, eb_loo = fit_rule(C, ALL, ('eb', False)), fit_rule(C, ALL, ('eb', True))
    report_factors = dict(k_full_step=full['b'] / RHO, k_b_lo_eb=eb_std['k'], k_loo_eb=eb_loo['k'],
                          b_lo_eb_detail=eb_std, loo_eb_detail=eb_loo, stated=REPORT_STATED)
    report_factors['match_stated_4dp'] = {k: bool(abs(report_factors[k] - v) <= 5e-5) for k, v in REPORT_STATED.items()}
    # in-sample all-500 change (report only): full-data c, k on the 80 train file rows
    q_in = po.copy(); q_in[file_tr] = full['c'] + full['k'] * (po[file_tr] - full['c'])
    insample_all500 = float(((q_in - y) ** 2 - (po - y) ** 2).mean())

    # ------------------------------------------------------------ nested bar on data/folds.csv
    q, changed, per_fit = nested(C, fold, REG)
    ev = evaluate(C, q, changed, fold, ids)
    assert ((q != po) == changed).all(), 'a selected row is unchanged or an unselected row changed'
    for pf, pe in zip(per_fit, ev['folds']):
        pe.update(m_f=pf['m'], c_f=pf['c'], b_f=pf['b'], k_f=pf['k'], n_fit_lo=pf['n_fit'])
        assert pe['n_changed'] == pf['n_changed']

    # ------------------------------------------------------------ 300 stratified re-splits (same nested procedure)
    rng = np.random.default_rng(SEED)
    splits = [strat(rng, glab) for _ in range(N_SPLIT)]
    for ff in splits:
        for gl in range(len(STRATA)):
            cnt = np.bincount(ff[glab == gl], minlength=5)
            assert cnt.max() - cnt.min() <= 1
    rs_main = resplits(C, splits, REG)

    bars = dict(a_all500_dmse=bool(ev['all500_dmse'] <= BAR['a_max_all500_dmse']),
                b_folds_better=bool(ev['folds_better'] >= BAR['b_min_folds_better']),
                c_signflip=bool(ev['signflip_p'] is not None and ev['signflip_p'] < BAR['c_max_signflip_p']),
                d_droptop3=bool(ev['droptop_all500_dmse'] <= BAR['d_max_all500_dmse_after_drop']),
                e_resplits=bool(rs_main['mean'] <= BAR['e_max_resplit_mean'] and rs_main['P_ge0'] <= BAR['e_max_resplit_P_ge0']))
    e_detail = dict(mean_ok=bool(rs_main['mean'] <= BAR['e_max_resplit_mean']),
                    P_ge0_ok=bool(rs_main['P_ge0'] <= BAR['e_max_resplit_P_ge0']))
    verdict = 'PASS' if all(bars.values()) else 'FAIL'

    # ------------------------------------------------------------ report only, same harness (never selectable)
    report_harness = {}
    for name, form in (('loo_eb', ('eb', True)), ('b_lo_eb', ('eb', False)), ('split_q40', ('half', 0.4)),
                       ('split_q60', ('half', 0.6))):
        f_full = fit_rule(C, ALL, form)
        qq, ch, pf = nested(C, fold, form)
        e = evaluate(C, qq, ch, fold, ids)
        report_harness[name] = dict(full=f_full, all500_dmse=e['all500_dmse'], folds_better=e['folds_better'],
                                    fold_dsse=[p['dsse'] for p in e['folds']], n_changed=e['n_changed'],
                                    signflip_p=e['signflip_p'], droptop_all500_dmse=e['droptop_all500_dmse'],
                                    per_fold_fit=pf, resplits=resplits(C, splits, form))

    # ------------------------------------------------------------ write (this folder only), read back
    oof_df = pd.DataFrame(dict(ID=ids, fold=fold, file_row=changed.astype(int), v27u_oof=po, hardness=q))
    oof_df.to_csv(out_oof, index=False)
    chk = pd.read_csv(out_oof, float_precision='round_trip')
    assert (chk.ID.astype(str).to_numpy() == ids).all() and (chk.hardness.to_numpy() == q).all()
    assert (chk.v27u_oof.to_numpy() == po).all() and (chk.file_row.to_numpy().astype(bool) == changed).all()
    res = dict(prereg='scripts/d1010/D/A/PREREG.txt', prereg_sha256=prereg_sha, script='a_test.py', script_sha256=sha,
               name='blend_v28n', base='blend_v27u', input_md5=input_md5, counts=counts,
               rule=dict(G1hi='raw features_v3 ic_noise >= 12 and ic_ridge_snr < 0.5',
                         file_rows='G1hi and ic_noise < m (m = median train G1hi ic_noise)',
                         factor='c = mean blend_v27u nested OOF, b = OLS slope (with intercept) of y on it, over the '
                                'train file rows; k = 1 + 0.5 (b / rho - 1); new = c + k (p - c)', rho=RHO),
               factor=dict(m=m, c=full['c'], b=full['b'], rho=RHO, k=full['k'], n_fit=full['n_fit'],
                           expected=EXPECT, tol=TOL),
               nested=ev, resplits=rs_main, split_seed=SEED, n_splits=N_SPLIT, signflip_seed=SEED, n_flips=N_FLIP,
               bar=BAR, bar_items=bars, e_detail=e_detail, verdict=verdict,
               report_only=dict(insample_all500_dmse=insample_all500, factors=report_factors, harness=report_harness),
               oof_file='a_test_oof.csv', oof_md5=md5(out_oof))
    json.dump(res, open(out_js, 'w'), indent=1)
    assert json.load(open(out_js))['verdict'] == verdict

    print('script sha256', sha)
    print('counts', counts)
    print('full-data factor: m %.6f c %.4f b %.4f rho %.10f k %.5f (n %d)' % (m, full['c'], full['b'], RHO, full['k'], full['n_fit']))
    print('nested all-500 MSE change %+.4f (RMSE %.4f -> %.4f) | changed rows %d | folds better %d/5 | sign-flip p %.4f '
          '| drop-top-3 all-500 change %+.4f (%s)' % (ev['all500_dmse'], ev['rmse_v27u'], ev['rmse_new'], ev['n_changed'],
                                                     ev['folds_better'], ev['signflip_p'], ev['droptop_all500_dmse'],
                                                     ','.join(ev['droptop_ids'])))
    print('per fold (m_f, k_f, n changed, dSSE):', [(round(p['m_f'], 4), round(p['k_f'], 4), p['n_changed'], round(p['dsse'], 1))
                                                    for p in ev['folds']])
    print('re-splits: mean %+.4f sd %.4f P(>=0) %.4f' % (rs_main['mean'], rs_main['sd'], rs_main['P_ge0']))
    print('report only: in-sample all-500 %+.4f | full step %.4f | k LOO-EB %.4f, B_lo_eb %.4f'
          % (insample_all500, report_factors['k_full_step'], eb_loo['k'], eb_std['k']))
    for name, r in report_harness.items():
        print('   [%s] k %.4f all-500 %+.4f folds %d/5 p %.4f drop3 %+.4f re-split mean %+.4f P(>=0) %.4f'
              % (name, r['full']['k'], r['all500_dmse'], r['folds_better'], r['signflip_p'], r['droptop_all500_dmse'],
                 r['resplits']['mean'], r['resplits']['P_ge0']))
    print('bar items', bars, '->', verdict)


if __name__ == '__main__':
    main()
