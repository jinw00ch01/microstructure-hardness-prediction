"""d1010 D / FILE A builder (run only after a_test.py PASS): blend_v28n = blend_v27u with the low-noise half of the G1hi
test rows stretched by the full-data half-step factor (D/A/PREREG.txt):
  G1hi = raw features_v3 ic_noise >= 12 AND ic_ridge_snr < 0.5 (161 train / 320 test)
  m = median raw ic_noise over the 161 train G1hi rows (16.280704); file rows = G1hi AND ic_noise < m (80 / 167)
  new = c + k (p - c), p = blend_v27u, on the 167 file test rows; the 833 other rows bit-identical to blend_v27u.
  c = mean blend_v27u nested OOF over the 80 train file rows (192.9155), k = 1 + 0.5 (b / rho - 1) (1.26171) with
  b = OLS slope (with intercept) of y on that OOF (1.4922), rho = 0.9795141681843882.
blend_v28n_oof.csv = the bar's nested OOF (a_test_oof.csv), unchanged.
Label-free: no label is read here. b (the only label-derived factor) comes from a_test_result.json, written by the
registered a_test.py whose sha256 must equal both a_test.sha256 and the script on disk; m, c, the row sets and the
nested OOF's per-fold m_f / c_f are recomputed label-free and must equal the recorded values. Test images enter only
through label-free per-image gate features; nothing is fitted on test rows; no test statistic enters any factor.
Writes ONLY CACHE/submissions/blend_v28n{.csv,_oof.csv,.json} and D/A/a_build_md5.json; refuses to overwrite.
The json (for the public repo) holds counts and train-side numbers only.

  python a_build.py
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import hashlib
import json
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

REPO = '/home/claude/microstructure-hardness-prediction'
CACHE = '/mnt/project-files/work/hardness-cache'
HERE = os.path.dirname(os.path.abspath(__file__))
D_DIR = os.path.dirname(HERE)
OUT_DIR = CACHE + '/submissions'
NAME = 'blend_v28n'
PREREG_SHA256 = '123b281da8a2359999e89d42278d44e366cdea68341446afa590c3043a5c9b88'
MD5 = {'blend_v27u.csv': '97a7cf2bbec5e90a65f8f7e388f4819f', 'blend_v27u_oof.csv': 'c0a2cda36bfc0e0394bb332f2ee54338'}
RHO = 0.9795141681843882
EXPECT = dict(n_G1hi_train=161, n_G1hi_test=320, n_file_train=80, n_file_test=167, n_other_train=420,
              n_other_test=833, m=16.280704, c=192.9155, b=1.4922, k=1.26171)
TOL = dict(m=1e-6, c=5e-5, b=5e-5, k=5e-6)
BAR_KEYS = {'a_all500_dmse', 'b_folds_better', 'c_signflip', 'd_droptop3', 'e_resplits'}
READBACK_WAIT_S = 10


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def main():
    # ------------------------------------------------------------ gate: a_test PASS with the recorded script
    res = json.load(open(HERE + '/a_test_result.json'))
    sha_rec = open(HERE + '/a_test.sha256').read().split()[0]
    sha_now = hashlib.sha256(open(HERE + '/a_test.py', 'rb').read()).hexdigest()
    items = res.get('bar_items', {})
    assert res.get('verdict') == 'PASS', 'a_test did not PASS: build nothing'
    assert set(items) == BAR_KEYS and all(v is True for v in items.values()), f'bar items not all true: {items}'
    assert res['script_sha256'] == sha_rec == sha_now, 'a_test.py / a_test.sha256 / result json sha256 disagree'
    prereg_sha = hashlib.sha256(open(HERE + '/PREREG.txt', 'rb').read()).hexdigest()
    reg = dict(l.split()[::-1] for l in open(D_DIR + '/PREREG.sha256').read().splitlines() if l.strip())
    assert prereg_sha == PREREG_SHA256 == reg[HERE + '/PREREG.txt'] == res['prereg_sha256'], 'PREREG changed'
    assert md5(HERE + '/a_test_oof.csv') == res['oof_md5'], 'a_test_oof.csv changed since the test'

    # ------------------------------------------------------------ outputs: refuse to overwrite
    paths = {k: f'{OUT_DIR}/{NAME}{s}' for k, s in (('csv', '.csv'), ('oof', '_oof.csv'), ('json', '.json'))}
    md5_record = HERE + '/a_build_md5.json'
    for p in list(paths.values()) + [md5_record]:
        assert not os.path.exists(p), f'refusing to overwrite {p}'

    # ------------------------------------------------------------ inputs (md5 pins; same data files as the test)
    for fn, m_ in MD5.items():
        assert md5(f'{OUT_DIR}/{fn}') == m_ == res['input_md5'][fn], fn
    assert md5(REPO + '/data/folds.csv') == res['input_md5']['data/folds.csv'], 'folds.csv changed since the test'
    assert md5(REPO + '/data/features_v3.parquet') == res['input_md5']['data/features_v3.parquet'], 'features_v3 changed'
    fo = pd.read_csv(REPO + '/data/folds.csv')
    ids = fo.ID.astype(str).to_numpy(); fold = fo.fold.to_numpy().astype(int)
    te_ids = pd.read_csv(REPO + '/data/sample_submission.csv').ID.astype(str).to_numpy()
    assert len(ids) == 500 and len(set(ids)) == 500 and len(te_ids) == 1000 and len(set(te_ids)) == 1000
    v27 = pd.read_csv(f'{OUT_DIR}/blend_v27u.csv', float_precision='round_trip')
    assert list(v27.columns) == ['ID', 'hardness'] and len(v27) == 1000
    assert (v27.ID.astype(str).to_numpy() == te_ids).all(), 'blend_v27u not in sample order'
    p = v27.hardness.to_numpy().astype(float)
    o27 = pd.read_csv(f'{OUT_DIR}/blend_v27u_oof.csv', float_precision='round_trip')
    assert list(o27.columns) == ['ID', 'hardness'] and (o27.ID.astype(str).to_numpy() == ids).all()
    po = o27.hardness.to_numpy().astype(float)
    assert np.isfinite(p).all() and np.isfinite(po).all()
    v3 = pd.read_parquet(REPO + '/data/features_v3.parquet').set_index('ID')
    nz, rs = v3.loc[ids, 'ic_noise'].to_numpy().astype(float), v3.loc[ids, 'ic_ridge_snr'].to_numpy().astype(float)
    nzt, rst = v3.loc[te_ids, 'ic_noise'].to_numpy().astype(float), v3.loc[te_ids, 'ic_ridge_snr'].to_numpy().astype(float)
    assert np.isfinite(nz).all() and np.isfinite(rs).all() and np.isfinite(nzt).all() and np.isfinite(rst).all()

    # ------------------------------------------------------------ rows (label-free gates; train-derived threshold)
    G, Gt = (nz >= 12) & (rs < 0.5), (nzt >= 12) & (rst < 0.5)
    m = float(np.median(nz[G]))
    file_tr, file_te = G & (nz < m), Gt & (nzt < m)
    counts = dict(G1hi_train=int(G.sum()), G1hi_test=int(Gt.sum()), file_train=int(file_tr.sum()),
                  file_test=int(file_te.sum()), other_train=int((~file_tr).sum()), other_test=int((~file_te).sum()))
    assert (counts['G1hi_train'], counts['G1hi_test']) == (EXPECT['n_G1hi_train'], EXPECT['n_G1hi_test']), counts
    assert (counts['file_train'], counts['file_test']) == (EXPECT['n_file_train'], EXPECT['n_file_test']), counts
    assert (counts['other_train'], counts['other_test']) == (EXPECT['n_other_train'], EXPECT['n_other_test']), counts
    assert counts == {k: v for k, v in res['counts'].items() if k in counts}, 'counts differ from the test'
    assert abs(m - EXPECT['m']) <= TOL['m'] and m == res['factor']['m'], m

    # ------------------------------------------------------------ factor: c recomputed, b from the PASSed test
    fac = res['factor']
    c = float(po[file_tr].mean())
    assert c == fac['c'] and abs(c - EXPECT['c']) <= TOL['c'], (c, fac['c'])
    b = float(fac['b'])
    assert abs(b - EXPECT['b']) <= TOL['b'], b
    assert fac['rho'] == RHO
    k = 1.0 + 0.5 * (b / RHO - 1.0)
    assert k == fac['k'] and abs(k - EXPECT['k']) <= TOL['k'], (k, fac['k'])

    # ------------------------------------------------------------ build the test file (row-wise transform only)
    new = p.copy()
    new[file_te] = c + k * (p[file_te] - c)
    out = pd.DataFrame({'ID': te_ids, 'hardness': new})
    assert len(out) == 1000 and (out.ID.to_numpy() == te_ids).all() and np.isfinite(out.hardness.to_numpy()).all()
    same = out.hardness.to_numpy() == p
    assert same[~file_te].all() and int((~file_te).sum()) == 833, 'rows outside the 167 file rows changed'
    assert np.max(np.abs(out.hardness.to_numpy()[file_te] - (c + k * (p[file_te] - c)))) <= 1e-9
    assert int((~same).sum()) == 167, int((~same).sum())

    # ------------------------------------------------------------ the bar's nested OOF, checked label-free
    at = pd.read_csv(HERE + '/a_test_oof.csv', float_precision='round_trip')
    assert list(at.columns) == ['ID', 'fold', 'file_row', 'v27u_oof', 'hardness'] and len(at) == 500
    assert (at.ID.astype(str).to_numpy() == ids).all() and (at.fold.to_numpy().astype(int) == fold).all()
    assert (at.v27u_oof.to_numpy() == po).all(), 'a_test_oof v27u column != blend_v27u_oof'
    onew = at.hardness.to_numpy().astype(float); frow = at.file_row.to_numpy().astype(int) == 1
    assert np.isfinite(onew).all()
    nested_sel = np.zeros(500, bool)
    for pf in res['nested']['folds']:
        f = pf['fold']
        g_f = G & (fold != f)
        m_f = float(np.median(nz[g_f]))
        lo_f = g_f & (nz < m_f)
        c_f = float(po[lo_f].mean())
        assert m_f == pf['m_f'] and c_f == pf['c_f'] and int(lo_f.sum()) == pf['n_fit_lo'], f
        k_f = 1.0 + 0.5 * (pf['b_f'] / RHO - 1.0)
        assert k_f == pf['k_f'], f
        app = G & (fold == f) & (nz < m_f)
        assert int(app.sum()) == pf['n_changed'], f
        assert np.max(np.abs(onew[app] - (c_f + k_f * (po[app] - c_f))), initial=0.0) <= 1e-9, f
        nested_sel |= app
    assert (nested_sel == frow).all() and ((onew != po) == nested_sel).all(), 'nested OOF rows inconsistent'
    assert int(nested_sel.sum()) == res['nested']['n_changed']
    oof = pd.DataFrame({'ID': ids, 'hardness': onew})
    assert len(oof) == 500 and np.isfinite(oof.hardness.to_numpy()).all()

    # ------------------------------------------------------------ train-side json (counts + train numbers only)
    nv = res['nested']; rsp = res['resplits']; ro = res['report_only']
    stats = {
        'name': NAME, 'base': 'blend_v27u',
        'rule': 'G1hi = raw features_v3 ic_noise >= 12 and ic_ridge_snr < 0.5; file rows = G1hi and ic_noise < m '
                '(m = median train G1hi ic_noise); new = c + k (p - c) on the file rows, c = mean blend_v27u nested OOF '
                'and b = OLS slope of y on it over the train file rows, k = 1 + 0.5 (b / rho - 1); other rows = '
                'blend_v27u bit-identical',
        'prereg': 'scripts/d1010/D/A/PREREG.txt', 'prereg_sha256': prereg_sha,
        'a_test_sha256': sha_rec, 'a_test_verdict': res['verdict'], 'a_test_bar_items': items, 'bar': res['bar'],
        'counts': counts,
        'factor': dict(m=m, c=c, b=b, rho=RHO, k=k, k_full_step=b / RHO, n_fit=int(file_tr.sum())),
        'report_only_factors': dict(k_loo_eb=ro['factors']['k_loo_eb'], k_b_lo_eb=ro['factors']['k_b_lo_eb']),
        'nested_all500_dmse': nv['all500_dmse'], 'nested_changed_rows': nv['n_changed'],
        'nested_folds_better': nv['folds_better'], 'nested_signflip_p': nv['signflip_p'],
        'nested_droptop3_all500_dmse': nv['droptop_all500_dmse'],
        'nested_per_fold': [dict(fold=pf['fold'], m_f=pf['m_f'], c_f=pf['c_f'], b_f=pf['b_f'], k_f=pf['k_f'],
                                 n_changed=pf['n_changed'], dsse=pf['dsse'], better=pf['better']) for pf in nv['folds']],
        'resplits': dict(n=rsp['n'], mean_all500_dmse=rsp['mean'], sd=rsp['sd'], P_ge0=rsp['P_ge0']),
        'insample_all500_dmse': ro['insample_all500_dmse'],
        'nested_cv_all': nv['rmse_new'], 'nested_cv_all_v27u': nv['rmse_v27u'],
        'fold_rmse': nv['fold_rmse_new'], 'fold_rmse_v27u': nv['fold_rmse_v27u'],
        'notes': 'PREREG d1010/D/A: half-step slope stretch (k = 1 + 0.5 (b / rho - 1), rho fixed from train-only inner CV) '
                 'on the low-noise half of G1hi; form found post hoc in the D design stage; one registered nested bar '
                 '(a_test.py) passed all five items. Test images are used only through label-free per-image gate '
                 'features; nothing fitted on test; no test statistic in any factor.',
    }

    # ------------------------------------------------------------ write (cache only), read back after a wait
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(paths['csv'], index=False)
    oof.to_csv(paths['oof'], index=False)
    with open(paths['json'], 'w') as fh:
        json.dump(stats, fh, indent=1)
    md5_w = {k_: md5(p_) for k_, p_ in paths.items()}
    time.sleep(READBACK_WAIT_S)
    md5_r = {k_: md5(p_) for k_, p_ in paths.items()}
    assert md5_r == md5_w, (md5_w, md5_r)
    chk = pd.read_csv(paths['csv'], float_precision='round_trip')
    assert list(chk.columns) == ['ID', 'hardness'] and len(chk) == 1000
    assert (chk.ID.astype(str).to_numpy() == te_ids).all() and (chk.hardness.to_numpy() == new).all()
    hv = chk.hardness.to_numpy()
    assert np.isfinite(hv).all() and (hv == p)[~file_te].all() and int((hv != p).sum()) == 167
    assert np.max(np.abs(hv[file_te] - (c + k * (p[file_te] - c)))) <= 1e-9
    chko = pd.read_csv(paths['oof'], float_precision='round_trip')
    assert list(chko.columns) == ['ID', 'hardness'] and (chko.ID.astype(str).to_numpy() == ids).all()
    assert (chko.hardness.to_numpy() == onew).all() and int((chko.hardness.to_numpy() != po).sum()) == int(nested_sel.sum())
    assert json.load(open(paths['json'])) == json.loads(json.dumps(stats))
    rec = dict(written={k_: paths[k_] for k_ in paths}, md5=md5_r, a_test_sha256=sha_rec,
               build_script_sha256=hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest(),
               readback_wait_s=READBACK_WAIT_S)
    with open(md5_record, 'w') as fh:
        json.dump(rec, fh, indent=1)

    print('written', paths)
    print('md5', md5_r)
    print('factor: m %.6f c %.4f b %.4f rho %.10f k %.5f | counts %s' % (m, c, b, RHO, k, counts))
    print('nested OOF: all-500 MSE change %+.4f, RMSE %.4f -> %.4f, changed %d, folds better %d/5'
          % (nv['all500_dmse'], nv['rmse_v27u'], nv['rmse_new'], nv['n_changed'], nv['folds_better']))
    # label-free test-side size of the change (stdout only; never stored in the json)
    dlt = new[file_te] - p[file_te]
    print('test change on %d file rows: rms %.3f mean %+.3f max|d| %.3f' % (len(dlt), np.sqrt(np.mean(dlt ** 2)), dlt.mean(),
                                                                          np.abs(dlt).max()))


if __name__ == '__main__':
    main()
