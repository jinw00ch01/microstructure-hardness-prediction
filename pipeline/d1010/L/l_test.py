"""d1010 L stage 2: the ONE pre-registered label test (PREREG.txt). Run once; its sha256 is recorded before the run.

Rows: LV train rows (9.5 <= raw features_v3 ic_noise < 12 AND ic_ridge_snr >= 0.5), features from stage 1
(feats_LV_train.parquet = the unchanged cleanhead pipeline, verified exact on 5 clean train images).
Model (no refit on LV rows, no selection): formula7 = OLS with intercept of hardness on
  [fd_area, pore_frac, logasp_aw, S_align, log N_eff, het4, bd_frac = N_bd / N_all]
fitted on the clean train rows only (cleanhead feats_C_train, 267 rows). OOF: for each outer fold k of data/folds.csv
fit on the clean rows with fold != k, predict the LV train rows with fold == k. Mix fixed w = 0.5:
  new = 0.5 * blend_v25_oof + 0.5 * f7 on the LV rows (blend_v25 = blend_v24 there, asserted).
BAR (all required): (a) gain = RMSE(v25) - RMSE(new) on the LV rows >= 0.40; (b) RMSE lower in >= 3 of 5 folds
(a fold without LV rows counts as not better); (c) one-sided paired sign-flip p < 0.20 (20000 flips, seed 20261010,
d_i = (new_i - y_i)^2 - (v25_i - y_i)^2; same routine as d1010/C/harness.py); (d) after dropping the single row with
the largest SSE gain (no refit) the gain is still >= 0.20.
Report only (never part of the bar): all-500 nested RMSE, per-fold numbers and row counts, w = 0.25 / 0.75, formula7
alone, the 4 HLV train rows (ic_noise >= 12 AND ridge >= 0.5) with the same procedure.
No test image or test row is used here. Output: l_test_result.json, l_test_oof.csv (LV + HLV train rows).
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
TERMS = ['fd_area', 'pore_frac', 'logasp_aw', 'S_align', 'logN_eff', 'het4', 'bd_frac']
W = 0.5
SEED, N_FLIP = 20261010, 20000
BAR = dict(min_gain=0.40, min_folds_better=3, max_signflip_p=0.20, min_droptop_gain=0.20)
MD5 = {'blend_v25_oof.csv': 'f20fb0a346d85308ce96f237b9082086', 'blend_v24_oof.csv': 'f1bb2163b732b3f950c3bd8b52e93447'}


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
    e = np.asarray(e, float)
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else float('nan')


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


def xfit_f7(Xc, yc, fc, Xt, ft):
    """Per outer fold k: fit on clean rows with fold != k, predict target rows with fold == k."""
    out = np.full(len(Xt), np.nan); coefs = []
    for k in range(5):
        b = ols(Xc[fc != k], yc[fc != k]); coefs.append(b)
        m = ft == k
        if m.any():
            out[m] = apply(b, Xt[m])
    assert np.isfinite(out).all()
    return out, coefs


def evaluate(y, base, f7, fold, w=W, do_signflip=True):
    new = (1 - w) * base + w * f7
    eb, en = base - y, new - y
    rb, rn = rmse(eb), rmse(en)
    per = []
    for k in range(5):
        m = fold == k
        per.append(dict(fold=k, n=int(m.sum()), rmse_v25=rmse(eb[m]), rmse_new=rmse(en[m]),
                        delta=(rmse(en[m]) - rmse(eb[m])) if m.any() else None,
                        better=bool(m.any() and rmse(en[m]) < rmse(eb[m]))))
    d = en ** 2 - eb ** 2
    top = int(np.argmax(-d)); keep = np.arange(len(d)) != top
    res = dict(n=int(len(y)), w=w, rmse_v25=rb, rmse_new=rn, gain=rb - rn, folds=per,
               folds_better=int(sum(p['better'] for p in per)), mean_d=float(d.mean()),
               droptop_idx=top, droptop_row_sse_gain=float(-d[top]),
               droptop_gain=(rmse(eb[keep]) - rmse(en[keep])) if keep.any() else float('nan'),
               rows_better=int((d < 0).sum()), bias_v25=float(eb.mean()), bias_new=float(en.mean()))
    res['signflip_p'] = float(signflip_p(d)) if do_signflip else None
    return res, new


def main():
    out_js, out_oof = HERE + '/l_test_result.json', HERE + '/l_test_oof.csv'
    for p in (out_js, out_oof):
        assert not os.path.exists(p), f'refusing to overwrite {p} (the test runs once)'
    sha = hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()
    sys.path.insert(0, REPO)
    from src.common import load_train
    tr = load_train()                                     # ID, hardness, fold (the single label read)
    assert len(tr) == 500 and tr.fold.isin(range(5)).all()
    v3 = pd.read_parquet(REPO + '/data/features_v3.parquet').set_index('ID')
    nz, rd = v3.loc[tr.ID, 'ic_noise'].values, v3.loc[tr.ID, 'ic_ridge_snr'].values
    clean = nz < 9.5
    lv = (nz >= 9.5) & (nz < 12) & (rd >= 0.5)
    hlv = (nz >= 12) & (rd >= 0.5)
    counts = dict(clean=int(clean.sum()), LV=int(lv.sum()), HLV=int(hlv.sum()))
    assert counts['clean'] == 267, counts
    ids, y, fold = tr.ID.values, tr.hardness.values.astype(float), tr.fold.values

    md5 = {}
    for c in MD5:
        md5[c] = hashlib.md5(open(REPO + '/submissions/' + c, 'rb').read()).hexdigest()
        assert md5[c] == MD5[c] == hashlib.md5(open(CACHE + '/submissions/' + c, 'rb').read()).hexdigest(), c
    o25 = pd.read_csv(REPO + '/submissions/blend_v25_oof.csv', float_precision='round_trip').set_index('ID').loc[ids].hardness.values
    o24 = pd.read_csv(REPO + '/submissions/blend_v24_oof.csv', float_precision='round_trip').set_index('ID').loc[ids].hardness.values
    assert (o25[lv] == o24[lv]).all() and (o25[hlv] == o24[hlv]).all(), 'blend_v25 != blend_v24 on LV/HLV rows'

    # formula7 on the clean rows (cleanhead feats_C_train), same row order as d1009/J/j_build.py
    Fc = pd.read_parquet(CH + 'feats_C_train.parquet')
    assert set(Fc.index) == set(ids[clean]) and len(Fc) == 267
    Xc = design(Fc).loc[ids[clean]].values; yc = y[clean]; fc = fold[clean]
    # fidelity of the model: clean OOF of blend_v25 must be reproduced by these per-fold fits
    f7c, coefs = xfit_f7(Xc, yc, fc, Xc, fc)
    rep = 0.5 * o24[clean] + 0.5 * f7c
    rep_maxdiff = float(np.max(np.abs(rep - o25[clean])))
    assert rep_maxdiff < 1e-9, rep_maxdiff
    b_full = ols(Xc, yc)

    FL = pd.read_parquet(HERE + '/feats_LV_train.parquet')
    assert set(FL.index[FL.group == 'LV']) == set(ids[lv]) and set(FL.index[FL.group == 'HLV']) == set(ids[hlv])
    res = dict(prereg='scripts/d1010/L/PREREG.txt', script='l_test.py', script_sha256=sha, counts=counts,
               input_md5=md5, clean_oof_reproduction_maxdiff=rep_maxdiff,
               coef_full=dict(zip(['intercept'] + TERMS, map(float, b_full))),
               coef_folds=[dict(zip(['intercept'] + TERMS, map(float, b))) for b in coefs])
    oof_rows = []
    for name, mask in (('LV', lv), ('HLV', hlv)):
        Xt = design(FL).loc[ids[mask]].values
        f7, _ = xfit_f7(Xc, yc, fc, Xt, fold[mask])
        r, new = evaluate(y[mask], o25[mask], f7, fold[mask])
        r['droptop_id'] = str(ids[mask][r.pop('droptop_idx')])
        r['f7_alone_rmse'] = rmse(f7 - y[mask]); r['f7_bias'] = float(np.mean(f7 - y[mask]))
        r['corr_f7_y'] = float(np.corrcoef(f7, y[mask])[0, 1]); r['corr_v25_y'] = float(np.corrcoef(o25[mask], y[mask])[0, 1])
        r['report_w'] = {}
        for wr in (0.25, 0.75):
            rr, _ = evaluate(y[mask], o25[mask], f7, fold[mask], w=wr)
            rr.pop('droptop_idx')
            r['report_w'][str(wr)] = dict(gain=rr['gain'], rmse_new=rr['rmse_new'], folds_better=rr['folds_better'],
                                          signflip_p=rr['signflip_p'], droptop_gain=rr['droptop_gain'],
                                          fold_delta=[p['delta'] for p in rr['folds']])
        o_all = o25.copy(); o_all[mask] = new
        r['all500_v25'] = rmse(o25 - y); r['all500_new'] = rmse(o_all - y)
        r['all500_fold_v25'] = [rmse((o25 - y)[fold == k]) for k in range(5)]
        r['all500_fold_new'] = [rmse((o_all - y)[fold == k]) for k in range(5)]
        res[name] = r
        oof_rows.append(pd.DataFrame(dict(ID=ids[mask], group=name, fold=fold[mask], f7_oof=f7, v25_oof=o25[mask],
                                          new_oof=new)))
    L = res['LV']
    bars = dict(a_gain=bool(L['gain'] >= BAR['min_gain']), b_folds=bool(L['folds_better'] >= BAR['min_folds_better']),
                c_signflip=bool(L['signflip_p'] < BAR['max_signflip_p']),
                d_droptop=bool(L['droptop_gain'] >= BAR['min_droptop_gain']))
    res['bar'] = BAR; res['bar_items'] = bars
    res['verdict'] = 'PASS' if all(bars.values()) else 'FAIL'
    json.dump(res, open(out_js, 'w'), indent=1)
    pd.concat(oof_rows).to_csv(out_oof, index=False)

    print('script sha256', sha)
    print('counts', counts, '| clean OOF reproduction max diff %.2e' % rep_maxdiff)
    for name in ('LV', 'HLV'):
        r = res[name]
        print(f"[{name}{' report only' if name == 'HLV' else ''}] n {r['n']} | RMSE v25 {r['rmse_v25']:.4f} -> new {r['rmse_new']:.4f} "
              f"gain {r['gain']:+.4f} | folds better {r['folds_better']}/5 | signflip p {r['signflip_p']:.4f} | "
              f"drop-top gain {r['droptop_gain']:+.4f} ({r['droptop_id']}) | f7 alone {r['f7_alone_rmse']:.3f} bias {r['f7_bias']:+.2f}")
        print('   per fold (n, v25, new, delta):',
              [(p['n'], round(p['rmse_v25'], 3), round(p['rmse_new'], 3), None if p['delta'] is None else round(p['delta'], 3))
               for p in r['folds']])
        print('   all-500 nested %.4f -> %.4f' % (r['all500_v25'], r['all500_new']),
              '| w 0.25 gain %+.4f, w 0.75 gain %+.4f' % (r['report_w']['0.25']['gain'], r['report_w']['0.75']['gain']))
    print('bar items', bars, '->', res['verdict'])


if __name__ == '__main__':
    main()
