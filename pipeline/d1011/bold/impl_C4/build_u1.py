"""impl_C4: U1 = one union file for the 10/11 submission = blend_v28n + C1 (W1 clean + V4 LV) + C2 riders + C3 T-CP.
Bold-round triage TRIAGE.txt section 2 (C1, C2, C3, C4). Single script; builds every piece from cached inputs.
NOT a submission and never written to submissions/: outputs go to bold/impl_C4/ only.

Row sets (test, raw features_v3 ic_noise / ic_ridge_snr; d1011/midlf m3b gate for T-CP):
  clean  ic_noise < 9.5                                    519 test / 267 train
  LV     9.5 <= ic_noise < 12 and ic_ridge_snr >= 0.5       42 test /  21 train
  T-CP   midLF (9.5 <= ic_noise < 12, ridge < 0.5) and m3b gate_cp   19 test / 7 train
Rules per row set (every fitted quantity is fitted on clean TRAIN rows; inside the training folds for the OOF):
  C1 clean (W1, L8 t7):  0.5*(v21base + off_V4) + 0.5*f8V4
                         off_V4 = OLS of (y - v21base) on [1, V4, N_eff^0.25];  f8V4 = OLS of y on F8 with het4 <- V4
  C2 clean (L8 t10 RALL): 0.5*(v21base + off_V4) + 0.5*f8_C2,  f8_C2 = OLS of log(y) on F8 with het4 <- V4,
                         fd_area <- FD6t (6-map mean dark fraction at cleanhead's per-image thr), logN_eff <- N_eff^0.25;
                         back-transform exp(p + s2/2), s2 = training residual variance (np.var, ddof 0)
                         (N_eff = cleanhead feats_C N_eff, which equals data/het_blocks N_eff; asserted)
  C1 LV (L1 V4_lv):      0.5*v25 + 0.5*f7V4,  f7V4 = OLS of y on F7 with het4 <- V4 (clean rows), applied to LV rows
  C3 T-CP (L4 tcp):      0.5*v28n + 0.5*f7,   f7 = blend_v26L coef_full on the V0 terms (old het, as validated)
  U1: clean from C2 (C1 if C2 fails its pre-declared asserts), LV from C1, T-CP from C3, all other rows = v28n bit-identical.
Pre-declared C2 asserts (TRIAGE C2 step 3): clean nested OOF reproduces 8.3609 +- 0.0005; non-clean rows identical to C1;
  test change vs C1 has SD within [0.5, 2.0] x the train OOF change SD (both reported). If any fails: C2 is dropped.
Label-free test feature FD6t (519 clean test images) is computed here once (cached to fd6t_test.parquet); test images
enter only through this per-image function; nothing is fitted on test rows; no statistic of test predictions is used.
OOF (all 500 train rows, data/folds.csv): v28n nested OOF, with clean rows <- C2 (or C1) nested, LV rows <- the V4 LV
rule nested (f7V4 fitted on clean rows of the other folds; 0.5 mix with v25 nested OOF), the 7 T-CP train rows <-
0.5*v28n OOF + 0.5*f7_k (f7_k fitted on clean rows of the other folds). A second file U1_oof_recipe.csv keeps LV rows at
v28n (the triage recipe's convention, because L1's LV number was descriptive).
usage (repo root): flock SLOT timeout 1500 python .../impl_C4/build_u1.py [--nj 3]"""
import os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import json, hashlib, argparse, subprocess
sys.path.insert(0, '/mnt/project-files/work/hardness-cache/scripts/d1011/bold/L1_formula')
from l1common import *  # noqa: E402  (REPO, CACHE, CH, RP, HERE, F7, F8, rmse, load, ols_fit, ols_pred, signflip_p)
from s14_segfam import seg  # noqa: E402  (also puts REPO on sys.path)
from src.het_blocks import shading_norm  # noqa: E402

OUT = CACHE + '/scripts/d1011/bold/impl_C4/'
SUB = CACHE + '/submissions/'
L8 = CACHE + '/scripts/d1011/bold/L8_wild/'
L4 = CACHE + '/scripts/d1011/bold/L4_noisy/'
L7 = CACHE + '/scripts/d1011/bold/L7_public/'
MAPS = [(s, tf) for s in (0.7, 0.8, 0.9) for tf in (0.5, 0.6)]
V28N_MD5 = '396a52f2263c4ec857b8dbee1316a3ae'
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
LOG = []


def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)


# ----------------------------------------------------------------------------------------------- FD6t (label-free)
def phase_thr(gp):  # copied from L8_wild/t9_fd_ens.py (used for FD6, reported only)
    from sklearn.mixture import GaussianMixture
    big = gp[gp.area > 25]; thr = 0.90
    try:
        gm = GaussianMixture(2, random_state=0).fit(big.gn.values[:, None])
        mu = np.sort(gm.means_.ravel())
        if mu[1] - mu[0] > 0.08:
            xs = np.linspace(mu[0], mu[1], 200); p = gm.predict_proba(xs[:, None]); lo = np.argmin(gm.means_.ravel())
            thr = xs[np.argmin(np.abs(p[:, lo] - 0.5))]
    except Exception:
        pass
    return thr


def fd_one(split, i, thr_c):  # L8_wild/t9_fd_ens.py one(i, thr) with the image split as a parameter
    from PIL import Image
    from skimage import measure
    im = np.array(Image.open(f'{REPO}/data/{split}/{i}.png')).astype(np.float32)
    a, b = [], []
    for s, tf in MAPS:
        sm, ws = seg(im, s, tf)
        g = pd.DataFrame([dict(label=p.label, area=p.area, cy=p.centroid[0], cx=p.centroid[1], mean=p.intensity_mean)
                          for p in measure.regionprops(ws, intensity_image=sm)])
        g['gn'] = shading_norm(g)
        gp = g[~((g.gn < 0.68) & (g.area < 600))]
        ar = gp.area.values.astype(float)
        a.append(ar[(gp.gn < phase_thr(gp)).values].sum() / ar.sum())
        b.append(ar[(gp.gn < thr_c).values].sum() / ar.sum())
    return dict(ID=i, FD6=float(np.mean(a)), FD6t=float(np.mean(b)), FD6_sd=float(np.std(a)))


def fitp(A, yt, At):
    bb = np.linalg.lstsq(np.column_stack([np.ones(len(A)), A]), yt, rcond=None)[0]
    return np.column_stack([np.ones(len(At)), At]) @ bb, np.column_stack([np.ones(len(A)), A]) @ bb, bb


def fit_log(A, yt, At):
    pt, pi, bb = fitp(A, np.log(yt), At)
    s2 = np.var(np.log(yt) - pi)
    return np.exp(pt + s2 / 2), bb, s2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--nj', type=int, default=3); ap.add_argument('--resplits', type=int, default=100)
    a = ap.parse_args()
    assert md5(SUB + 'blend_v28n.csv') == V28N_MD5

    # ------------------------------------------------------------------------------------------ train side
    C = load(); y, fold, D, o24, o28c, ids = C['y'], C['fold'], C['D'], C['o24'], C['o28'], C['ids']
    tr = C['tr']; aid = tr.ID.values; yall = tr.hardness.values.astype(float); fall = tr.fold.values.astype(int)
    nc = len(y); assert nc == 267
    v3 = pd.read_parquet(REPO + '/data/features_v3.parquet').set_index('ID')
    nz, rs = v3.loc[aid, 'ic_noise'].values, v3.loc[aid, 'ic_ridge_snr'].values
    G = np.full(500, 'G1hi', dtype=object)
    G[nz < 9.5] = 'clean'; G[(nz >= 9.5) & (nz < 12) & (rs >= 0.5)] = 'LV'; G[(nz >= 9.5) & (nz < 12) & (rs < 0.5)] = 'midLF'
    G[(nz >= 12) & (rs >= 0.5)] = 'HLV'
    cnt = {g: int((G == g).sum()) for g in ['clean', 'LV', 'midLF', 'HLV', 'G1hi']}
    assert cnt == dict(clean=267, LV=21, midLF=47, HLV=4, G1hi=161), cnt
    rdo = lambda n: pd.read_csv(SUB + f'{n}_oof.csv', float_precision='round_trip').set_index('ID').hardness.loc[aid].values
    O28, O25 = rdo('blend_v28n'), rdo('blend_v25')
    pos_c = pd.Index(aid).get_indexer(ids); assert (G[pos_c] == 'clean').all() and float(np.abs(O28[pos_c] - o28c).max()) < 1e-9
    o21b = pd.read_csv(SUB + 'blend_v21base_oof.csv').set_index('ID').hardness.loc[ids].values
    hbt = pd.read_parquet(REPO + '/data/het_blocks_train.parquet').set_index('ID').loc[ids]
    N = C['F'].N_eff.values.astype(float); assert np.array_equal(N, hbt.N_eff.values.astype(float))
    V4 = pd.read_parquet(HERE + 's26_train.parquet').loc[ids, 'V4'].values
    FD = pd.read_parquet(L8 + 't9_fd_train.parquet').loc[ids, 'FD6t'].values
    Xh = np.column_stack([V4, N ** 0.25])
    X8o = D[F8].values
    Dv = D.copy(); Dv['het4'] = V4; X8v = Dv[F8].values
    Dr = Dv.copy(); Dr['fd_area'] = FD; Dr['logN_eff'] = N ** 0.25; X8r = Dr[F8].values
    X7o = D[F7].values; X7v = Dv[F7].values
    # LV train rows (21)
    FLt = pd.read_parquet(CACHE + '/scripts/d1010/L/feats_LV_train.parquet'); FLt = FLt[FLt.group == 'LV']
    lv_ids = FLt.index.values; pos_lv = pd.Index(aid).get_indexer(lv_ids); assert (G[pos_lv] == 'LV').all() and len(lv_ids) == 21

    def d7(Fx, het):
        X = pd.DataFrame(index=Fx.index)
        for c in ['fd_area', 'pore_frac', 'logasp_aw', 'S_align']:
            X[c] = Fx[c].astype(float)
        X['logN_eff'] = np.log(Fx.N_eff.astype(float)); X['het4'] = het; X['bd_frac'] = Fx.N_bd / Fx.N_all
        return X[F7].values
    hl_tr = pd.read_parquet(HERE + 's28_V4_LV_train.parquet').loc[lv_ids, 'V4'].values
    Xlo, Xlv = d7(FLt, FLt.het4.values.astype(float)), d7(FLt, hl_tr)
    # T-CP train rows (7): m3b split train, midLF and gate_cp (only these columns are read)
    TERMS = F7
    q = pd.read_parquet(CACHE + '/scripts/d1010/N/gate/q_train_0of1.parquet').loc[aid]
    XV0 = q[[f'V0__{t}' for t in TERMS]].to_numpy().astype(float)
    assert np.array_equal(XV0[pos_c], X7o)
    mb = pd.read_parquet(CACHE + '/scripts/d1011/midlf/m3b_rows.parquet', columns=['split', 'group', 'gate_cp'])
    mb = mb[mb.split == 'train'].loc[aid]
    CPm = (mb.group.to_numpy() == 'midLF') & mb.gate_cp.to_numpy().astype(bool)
    assert CPm.sum() == 7 and (G[CPm] == 'midLF').all()
    pos_cp = np.where(CPm)[0]

    def components(fa):
        """All fitted pieces under fold vector fa (500 rows, train.csv order); fits use clean rows with fold != k."""
        fc, fl, fp = fa[pos_c], fa[pos_lv], fa[pos_cp]
        half = np.empty(nc); f8o = np.empty(nc); f8v = np.empty(nc); f8r = np.empty(nc)
        f7o_lv = np.empty(21); f7v_lv = np.empty(21); f7_cp = np.empty(len(pos_cp))
        for k in range(5):
            t = fc != k; e = ~t
            half[e] = o21b[e] + fitp(Xh[t], y[t] - o21b[t], Xh[e])[0]
            f8o[e] = fitp(X8o[t], y[t], X8o[e])[0]
            f8v[e] = fitp(X8v[t], y[t], X8v[e])[0]
            f8r[e] = fit_log(X8r[t], y[t], X8r[e])[0]
            m = fl == k
            f7o_lv[m] = fitp(X7o[t], y[t], Xlo[m])[0]; f7v_lv[m] = fitp(X7v[t], y[t], Xlv[m])[0]
            m = fp == k
            f7_cp[m] = fitp(X7o[t], y[t], XV0[pos_cp][m])[0]
        return dict(base_c=0.5 * o24 + 0.5 * f8o, W1=0.5 * half + 0.5 * f8v, C2=0.5 * half + 0.5 * f8r,
                    base_lv=0.5 * O25[pos_lv] + 0.5 * f7o_lv, lv=0.5 * O25[pos_lv] + 0.5 * f7v_lv,
                    cp=0.5 * O28[pos_cp] + 0.5 * f7_cp)

    P = components(fall)
    r_base_c = float(np.abs(P['base_c'] - o28c).max()); r_base_lv = float(np.abs(P['base_lv'] - O28[pos_lv]).max())
    say('reproduce v28n nested OOF: clean rows max diff %.2e | LV rows max diff %.2e' % (r_base_c, r_base_lv))
    assert r_base_c < 1e-8 and r_base_lv < 1e-8
    e = lambda p, yy: rmse(p - yy)
    say('C1 W1 clean nested %.4f (expect 8.4176; v28n %.4f)' % (e(P['W1'], y), e(o28c, y)))
    assert abs(e(P['W1'], y) - 8.4176) <= 0.00005
    w1_ref = pd.read_csv(L8 + 'CANDIDATE_v28n_gridhet_V4_both_oof_clean.csv', float_precision='round_trip').set_index('ID').hardness.loc[ids].values
    assert float(np.abs(w1_ref - P['W1']).max()) < 1e-9
    say('C2 RALL clean nested %.4f (expect 8.3609 +- 0.0005)' % e(P['C2'], y))
    ylv = yall[pos_lv]
    say('C1 LV (V4) nested n=21: %.3f vs v28n %.3f (L1 s29 descriptive: 9.496 / 10.522)' % (e(P['lv'], ylv), e(O28[pos_lv], ylv)))
    ycp = yall[pos_cp]
    say('C3 T-CP nested n=7: %.3f vs v28n %.3f (L4 q6: 7.754 / 9.063)' % (e(P['cp'], ycp), e(O28[pos_cp], ycp)))
    assert abs(e(P['cp'], ycp) - 7.7535) < 0.0005

    # ------------------------------------------------------------------------------------------ test side
    te = pd.read_csv(REPO + '/data/sample_submission.csv'); tid_all = te.ID.values
    rd = lambda p: pd.read_csv(p, float_precision='round_trip').set_index('ID').hardness.loc[tid_all].values
    v28 = rd(SUB + 'blend_v28n.csv'); v21b = rd(SUB + 'blend_v21base.csv'); v25 = rd(SUB + 'blend_v25.csv'); v24 = rd(SUB + 'blend_v24.csv')
    nzt, rst = v3.loc[tid_all, 'ic_noise'].values, v3.loc[tid_all, 'ic_ridge_snr'].values
    cl = nzt < 9.5; lvm = (nzt >= 9.5) & (nzt < 12) & (rst >= 0.5); mid = (nzt >= 9.5) & (nzt < 12) & (rst < 0.5)
    assert cl.sum() == 519 and lvm.sum() == 42 and mid.sum() == 113
    tid = tid_all[cl]
    Ft = pd.read_parquet(CH + 'feats_C_test.parquet').loc[tid]; At = pd.read_parquet(RP + 'feats_avg5_test.parquet').loc[tid]
    hbe = pd.read_parquet(REPO + '/data/het_blocks_test.parquet').set_index('ID').loc[tid]
    Nt = Ft.N_eff.values.astype(float); assert np.array_equal(Nt, hbe.N_eff.values.astype(float))
    Dt = pd.DataFrame(index=tid)
    for c in ['fd_area', 'pore_frac', 'logasp_aw', 'S_align', 'het4']:
        Dt[c] = Ft[c].astype(float)
    Dt['logN_eff'] = np.log(Nt); Dt['bd_frac'] = Ft.N_bd / Ft.N_all; Dt['ringall_avg5'] = At.ringall_avg5
    # sanity: unchanged f8 reproduces v28n clean test rows
    rep = 0.5 * v24[cl] + 0.5 * fitp(X8o, y, Dt[F8].values)[0]
    d_rep = float(np.abs(rep - v28[cl]).max()); say('reproduce v28n clean test rows (old f8): max diff %.2e' % d_rep); assert d_rep < 1e-6
    V4t = pd.read_parquet(HERE + 's28_V4_clean_test.parquet').loc[tid, 'V4'].values
    Dtv = Dt.copy(); Dtv['het4'] = V4t
    half_t, _, coef_off = fitp(Xh, y - o21b, np.column_stack([V4t, Nt ** 0.25]))
    half_t = v21b[cl] + half_t
    f8v_t = fitp(X8v, y, Dtv[F8].values)[0]
    C1 = v28.copy(); C1[cl] = 0.5 * half_t + 0.5 * f8v_t
    # LV test rows: 0.5*v25 + 0.5*f7V4
    FL = pd.read_parquet(CACHE + '/scripts/d1010/L/feats_LV_test.parquet'); FL = FL[FL.group == 'LV']
    lv_te = te.ID.isin(FL.index).values; assert lv_te.sum() == 42 and np.array_equal(lv_te, lvm)
    lid = tid_all[lv_te]; FL = FL.loc[lid]
    rep7 = 0.5 * v25[lv_te] + 0.5 * fitp(X7o, y, d7(FL, FL.het4.values.astype(float)))[0]
    d_lv = float(np.abs(rep7 - v28[lv_te]).max()); say('reproduce v28n LV test rows (old f7): max diff %.2e' % d_lv); assert d_lv < 1e-6
    hl = pd.read_parquet(HERE + 's28_V4_LV_test.parquet').loc[lid, 'V4'].values
    C1[lv_te] = 0.5 * v25[lv_te] + 0.5 * fitp(X7v, y, d7(FL, hl))[0]
    # C1 vs the lens files (values; their md5 recorded)
    W1f, LVf = L8 + 'CANDIDATE_v28n_gridhet_V4_both.csv', HERE + 'CANDIDATE_v28n_gridhet_V4_lv.csv'
    W1v, LVv = rd(W1f), rd(LVf)
    dW1 = float(np.abs(C1[cl] - W1v[cl]).max()); dLV = float(np.abs(C1[lv_te] - LVv[lv_te]).max())
    say('C1 vs L8 W1 file clean rows max diff %.2e (md5 %s) | vs L1 V4_lv LV rows max diff %.2e (md5 %s)' % (dW1, md5(W1f)[:8], dLV, md5(LVf)[:8]))
    assert dW1 < 1e-9 and dLV < 1e-9 and np.array_equal(W1v[~cl], v28[~cl])
    assert np.array_equal(W1v[lv_te], v28[lv_te]) and float(np.abs(LVv[cl] - (0.5 * v24[cl] + 0.5 * f8v_t)).max()) < 1e-9
    ch1 = C1 != v28; assert ch1.sum() == 561 and np.array_equal(ch1, cl | lv_te) and np.isfinite(C1).all()

    # FD6t on the 519 clean test images (label-free; cached)
    fdp = OUT + 'fd6t_test.parquet'
    from joblib import Parallel, delayed
    chk = [fd_one('train', i, float(C['F'].thr.loc[i])) for i in ids[:3]]
    t9 = pd.read_parquet(L8 + 't9_fd_train.parquet')
    d_fd = max(abs(c['FD6t'] - t9.loc[c['ID'], 'FD6t']) for c in chk)
    say('copied FD function reproduces t9 FD6t on 3 clean train images: max diff %.2e' % d_fd); assert d_fd < 1e-12
    if not os.path.exists(fdp):
        thr = Ft.thr.astype(float)
        X = pd.DataFrame(Parallel(n_jobs=a.nj)(delayed(fd_one)('test', i, float(thr.loc[i])) for i in tid)).set_index('ID')
        X.to_parquet(fdp)
    FDt = pd.read_parquet(fdp).loc[tid]
    say('FD6t test: corr with cleanhead fd_area %.4f (train %.4f) | mean %.4f (train %.4f)' % (
        np.corrcoef(FDt.FD6t, Dt.fd_area)[0, 1], np.corrcoef(FD, D.fd_area)[0, 1], FDt.FD6t.mean(), FD.mean()))
    Dtr = Dtv.copy(); Dtr['fd_area'] = FDt.FD6t.values; Dtr['logN_eff'] = Nt ** 0.25
    f8r_t, coef_log, s2_full = fit_log(X8r, y, Dtr[F8].values)
    C2c = 0.5 * half_t + 0.5 * f8r_t
    C2 = C1.copy(); C2[cl] = C2c
    # pre-declared C2 asserts
    chg_te = C2[cl] - C1[cl]; chg_tr = P['C2'] - P['W1']
    sd_ratio = float(chg_te.std() / chg_tr.std())
    c2_checks = dict(clean_nested=e(P['C2'], y), reproduce_ok=bool(abs(e(P['C2'], y) - 8.3609) <= 0.0005),
                     nonclean_identical_to_C1=bool(np.array_equal(C2[~cl], C1[~cl])),
                     test_change_mean=float(chg_te.mean()), test_change_sd=float(chg_te.std()),
                     train_change_mean=float(chg_tr.mean()), train_change_sd=float(chg_tr.std()), sd_ratio=sd_ratio,
                     sd_ratio_ok=bool(0.5 <= sd_ratio <= 2.0), max_abs_test_change=float(np.abs(chg_te).max()),
                     max_abs_train_change=float(np.abs(chg_tr).max()), finite=bool(np.isfinite(C2c).all()))
    c2_ok = c2_checks['reproduce_ok'] and c2_checks['nonclean_identical_to_C1'] and c2_checks['sd_ratio_ok'] and c2_checks['finite']
    c2_checks['PASS'] = bool(c2_ok)
    say('C2 checks: ' + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c2_checks.items()}))

    # C3 T-CP (inline; then cross-checked against L4's tcp_build.py run into this folder)
    cf = json.load(open(SUB + 'blend_v26L.json'))['coef_full']
    assert float(np.abs(ols_fit(X7o, y) - np.array([cf['intercept']] + [cf[t] for t in TERMS])).max()) < 1e-9
    qt = pd.concat([pd.read_parquet(CACHE + f'/scripts/d1010/N/gate/q_test_{k}of2.parquet') for k in (0, 1)]).loc[tid_all]
    f7t = cf['intercept'] + sum(cf[t] * qt[f'V0__{t}'].to_numpy().astype(float) for t in TERMS)
    mt = pd.read_parquet(CACHE + '/scripts/d1011/midlf/m3b_rows.parquet', columns=['split', 'group', 'gate_cp'])
    mt = mt[mt.split == 'test'].loc[tid_all]
    tcp = (mt.group.to_numpy() == 'midLF') & mt.gate_cp.to_numpy().astype(bool)
    assert tcp.sum() == 19 and mid[tcp].all() and np.isfinite(f7t[tcp]).all()
    C3 = v28.copy(); C3[tcp] = 0.5 * v28[tcp] + 0.5 * f7t[tcp]
    r = subprocess.run([sys.executable, L4 + 'tcp_build.py', '--name', 'C3_tcp', '--out-dir', OUT.rstrip('/')], cwd=REPO,
                       capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    assert r.returncode == 0, r.stderr
    C3f = rd(OUT + 'C3_tcp.csv'); d_c3 = float(np.abs(C3f - C3).max())
    say('C3 inline vs L4 tcp_build.py output: max diff %.2e (rows changed %d)' % (d_c3, int((C3f != v28).sum()))); assert d_c3 < 1e-9

    # ------------------------------------------------------------------------------------------ U1 union
    clean_src = C2 if c2_ok else C1
    U = v28.copy(); U[cl] = clean_src[cl]; U[lv_te] = C1[lv_te]; U[tcp] = C3[tcp]
    assert not (cl & lv_te).any() and not (cl & tcp).any() and not (lv_te & tcp).any()
    chg = U != v28
    assert chg.sum() == 580 and np.array_equal(chg, cl | lv_te | tcp) and np.array_equal(U[~chg], v28[~chg])
    assert np.isfinite(U).all() and len(U) == 1000
    outp = OUT + 'test.csv'
    sub_df = pd.DataFrame({'ID': tid_all, 'hardness': U}); sub_df.to_csv(outp, index=False)
    back = pd.read_csv(outp, float_precision='round_trip')
    assert (back.ID.values == tid_all).all() and np.array_equal(back.hardness.values, U)
    pd.DataFrame({'ID': tid_all, 'hardness': C1}).to_csv(OUT + 'C1_core.csv', index=False)
    pd.DataFrame({'ID': tid, 'hardness': C2c}).to_csv(OUT + 'C2_clean_test.csv', index=False)

    # ------------------------------------------------------------------------------------------ OOF all 500
    oof = O28.copy(); oof[pos_c] = P['C2'] if c2_ok else P['W1']; oof[pos_cp] = P['cp']
    oof_recipe = oof.copy()          # LV rows at v28n (triage recipe convention)
    oof[pos_lv] = P['lv']            # final candidate: LV rows nested with the V4 rule
    c1_oof = O28.copy(); c1_oof[pos_c] = P['W1']
    pd.DataFrame({'ID': aid, 'hardness': oof}).to_csv(OUT + 'oof.csv', index=False)
    pd.DataFrame({'ID': aid, 'hardness': oof_recipe}).to_csv(OUT + 'U1_oof_recipe.csv', index=False)
    pd.DataFrame({'ID': aid, 'hardness': c1_oof}).to_csv(OUT + 'C1_core_oof.csv', index=False)
    pd.Series(P['C2'], index=pd.Index(ids, name='ID'), name='hardness').to_csv(OUT + 'C2_oof_clean.csv')
    pd.DataFrame({'ID': aid[pos_cp], 'hardness': P['cp'], 'v28n_oof': O28[pos_cp]}).to_csv(OUT + 'C3_oof_rows.csv', index=False)
    pd.DataFrame({'ID': aid[pos_lv], 'hardness': P['lv'], 'v28n_oof': O28[pos_lv]}).to_csv(OUT + 'LV_oof_rows.csv', index=False)

    dens = pd.read_parquet(REPO + '/data/features_cal.parquet').set_index('ID').loc[aid, 'cal_seg_count_density'].values
    terc = np.digitize(dens, np.quantile(dens, [1 / 3, 2 / 3]))   # 0 = coarse (fewest grains)

    def stats(cand, label):
        d = (cand - yall) ** 2 - (O28 - yall) ** 2
        fr = [round(rmse((cand - yall)[fall == k]) - rmse((O28 - yall)[fall == k]), 4) for k in range(5)]
        bg = {g: [round(rmse((O28 - yall)[G == g]), 3), round(rmse((cand - yall)[G == g]), 3), int((G == g).sum())]
              for g in ['clean', 'LV', 'midLF', 'HLV', 'G1hi']}
        bt = {nm: [round(rmse((O28 - yall)[terc == j]), 3), round(rmse((cand - yall)[terc == j]), 3)] for j, nm in enumerate(['coarse', 'mid', 'fine'])}
        o = np.argsort(d); keep = np.ones(500, bool); keep[o[:3]] = False
        s = dict(label=label, nested_rmse_all=rmse(cand - yall), v28n_rmse_all=rmse(O28 - yall), dMSE_all500=float(d.mean()),
                 fold_delta_rmse=fr, folds_better=int(sum(x < 0 for x in fr)), signflip_p=float(signflip_p(d)),
                 droptop3_dMSE=float(d[keep].mean()), rows_changed_train=int((np.abs(cand - O28) > 1e-12).sum()),
                 by_group_v28n_cand_n=bg, by_N_tercile_v28n_cand=bt)
        say('%s: all-500 %.4f vs v28n %.4f (dMSE %+.3f) folds better %d/5 %s signflip p %.4f drop-top3 dMSE %+.3f' % (
            label, s['nested_rmse_all'], s['v28n_rmse_all'], s['dMSE_all500'], s['folds_better'], fr, s['signflip_p'], s['droptop3_dMSE']))
        say('   by group [v28n, cand, n]: %s' % bg); say('   by grain-count tercile [v28n, cand]: %s' % bt)
        return s
    S = stats(oof, 'U1 (LV nested)'); S_rec = stats(oof_recipe, 'U1 recipe (LV = v28n)'); S_c1 = stats(c1_oof, 'C1 clean only (W1)')

    # re-splits (descriptive; formula parts refit on each split, base OOFs v24/v21base/v25/v28n fixed as in L8 t10)
    rng = np.random.default_rng(20261011); yb = pd.qcut(yall, 10, labels=False); rsd = []
    for it in range(a.resplits):
        fa = np.empty(500, int)
        for bb in range(10):
            idx = np.where(yb == bb)[0]; rng.shuffle(idx); fa[idx] = np.arange(len(idx)) % 5
        Q = components(fa)
        base = O28.copy(); base[pos_c] = Q['base_c']; base[pos_lv] = Q['base_lv']
        cand = O28.copy(); cand[pos_c] = Q['C2'] if c2_ok else Q['W1']; cand[pos_lv] = Q['lv']; cand[pos_cp] = Q['cp']
        rsd.append(float(np.mean((cand - yall) ** 2 - (base - yall) ** 2)))
    rsd = np.array(rsd)
    RS = dict(n=int(a.resplits), mean_dMSE=float(rsd.mean()), sd=float(rsd.std()), q05=float(np.quantile(rsd, .05)),
              q95=float(np.quantile(rsd, .95)), P_ge0=float((rsd >= 0).mean()))
    say('re-splits (%d): all-500 dMSE mean %+.3f sd %.3f q05 %+.3f q95 %+.3f P(>=0) %.2f' % (
        RS['n'], RS['mean_dMSE'], RS['sd'], RS['q05'], RS['q95'], RS['P_ge0']))

    # qualify.py (L7) lane check, M = 36
    QL = {}
    for nm, fp in [('U1_LV_nested', OUT + 'oof.csv'), ('U1_recipe', OUT + 'U1_oof_recipe.csv')]:
        r = subprocess.run([sys.executable, L7 + 'qualify.py', '--base', 'blend_v28n', '--cand', fp, '--rows', 'changed', '--M', '36'],
                           cwd=REPO, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        QL[nm] = json.loads(r.stdout); say('qualify %s: %s' % (nm, json.dumps(QL[nm])))

    info = dict(id='C4', name='U1', rule=__doc__.split('\n\n')[0], test_md5=md5(outp), v28n_md5=V28N_MD5,
                rows_changed_test=int(chg.sum()), rows=dict(clean=int(cl.sum()), LV=int(lv_te.sum()), TCP=int(tcp.sum())),
                clean_source='C2' if c2_ok else 'C1', c2_checks=c2_checks,
                coef_off_V4=coef_off.tolist(), coef_f8_C2_log=dict(zip(['intercept'] + F8, coef_log.tolist())), s2_full=float(s2_full),
                inputs_md5={k: md5(v)[:12] for k, v in dict(W1=W1f, V4_lv=LVf, C3_tcp=OUT + 'C3_tcp.csv', fd6t_test=fdp, C1_core=OUT + 'C1_core.csv').items()},
                test_change=dict(rms_all=float(np.sqrt(np.mean((U - v28) ** 2))), clean_mean=float((U - v28)[cl].mean()),
                                 clean_rms=float(np.sqrt(np.mean((U - v28)[cl] ** 2))), lv_mean=float((U - v28)[lv_te].mean()),
                                 tcp_mean=float((U - v28)[tcp].mean())),
                component_nested=dict(W1_clean=e(P['W1'], y), C2_clean=e(P['C2'], y), v28n_clean=e(o28c, y),
                                      LV_V4=e(P['lv'], ylv), LV_v28n=e(O28[pos_lv], ylv), TCP=e(P['cp'], ycp), TCP_v28n=e(O28[pos_cp], ycp)),
                final=S, recipe_LV_v28n=S_rec, C1_only=S_c1, resplits=RS, qualify_M36=QL, log=LOG)
    json.dump(info, open(OUT + 'build_u1.json', 'w'), indent=1, default=float)
    say('test.csv md5', info['test_md5'])


if __name__ == '__main__':
    main()
