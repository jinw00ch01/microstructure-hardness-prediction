"""N4 / clean: the clean-row mix w (and off_V4 scale s) and the LV-row mix u of blend_v29g, re-chosen by a
selection-honest nested procedure. Option set and procedure: N4/clean/options.txt (written before any label number here).

Components reuse impl_C4/build_u1.py (imported, not edited; its main() is not run): fitp, fit_log and the same feature
tables / base files read the same way, so the O0 option (w 0.5, s 1.0, u 0.5) reproduces blend_v29g OOF and test exactly
on the clean and LV rows. Labels: train only (lib_l4 -> src.common.load_train; l1common reads data/train.csv the same way
build_u1 did). Test rows: only label-free per-image features + fixed base test files; nothing is fitted on or chosen by
test rows. usage (repo root): nice -n 10 timeout 1800 python <N4>/clean/build_clean.py [--resplits 40]
"""
import os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
sys.dont_write_bytecode = True
import json, argparse, hashlib, time
N4D = '/mnt/project-files/work/hardness-cache/scripts/d1011/N4'
sys.path.insert(0, N4D)
import n4lib as NL  # noqa: E402
from n4lib import np, pd  # noqa: E402
sys.path.insert(0, NL.CACHE + '/scripts/d1011/bold/impl_C4')
import build_u1 as BU  # noqa: E402  (fitp, fit_log; module import only)
import l1common as L1  # noqa: E402  (already on sys.path via build_u1)

OUT = N4D + '/clean/'
SUBD = NL.CACHE + '/submissions/'
IMPL = NL.CACHE + '/scripts/d1011/bold/impl_C4/'
L8 = NL.CACHE + '/scripts/d1011/bold/L8_wild/'
F7, F8, HERE = L1.F7, L1.F8, L1.HERE
LOG = []


def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)


# ------------------------------------------------------------------------------------------- options (options.txt)
CLEANV = [(0.25, 1.0), (0.25, 1.25), (0.5, 1.0), (0.5, 1.25), (0.75, 1.0), (0.75, 1.25), (1.0, 1.0)]
LVV = [0.5, 0.75, 1.0]
O0 = (0.5, 1.0, 0.5)
OPTS = [O0] + [(w, s, u) for (w, s) in CLEANV for u in LVV if (w, s, u) != O0]
assert len(OPTS) == 21 and len(set(OPTS)) == 21
oname = lambda o: 'w%.2f_s%.2f_u%.2f' % o


def d7(Fx, het):  # copied from build_u1.main
    X = pd.DataFrame(index=Fx.index)
    for c in ['fd_area', 'pore_frac', 'logasp_aw', 'S_align']:
        X[c] = Fx[c].astype(float)
    X['logN_eff'] = np.log(Fx.N_eff.astype(float)); X['het4'] = het; X['bd_frac'] = Fx.N_bd / Fx.N_all
    return X[F7].values


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--resplits', type=int, default=40); a = ap.parse_args()
    t0 = time.time()
    D = NL.load29()
    ids, y, fold, p29, t29, te_ids = D['ids'], D['y'], D['fold'], D['p29'], D['t29'], D['te_ids']
    cl, lv, clt, lvt = D['grp']['clean'], D['grp']['LV'], D['grpt']['clean'], D['grpt']['LV']
    # ---------------------------------------------------------------- train side, as build_u1 (clean rows in train.csv order)
    C = L1.load(); ids_c, yc, Dc = C['ids'], C['y'], C['D']
    assert np.array_equal(ids_c, ids[cl]) and np.array_equal(yc, y[cl]) and np.array_equal(C['fold'], fold[cl])
    idx_c = np.where(cl)[0]
    o21c = pd.read_csv(SUBD + 'blend_v21base_oof.csv').set_index('ID').hardness.loc[ids_c].values  # as build_u1 (no round_trip)
    N = C['F'].N_eff.values.astype(float)
    V4 = pd.read_parquet(HERE + 's26_train.parquet').loc[ids_c, 'V4'].values
    FD = pd.read_parquet(L8 + 't9_fd_train.parquet').loc[ids_c, 'FD6t'].values
    Xh = np.column_stack([V4, N ** 0.25])
    Dv = Dc.copy(); Dv['het4'] = V4
    Dr = Dv.copy(); Dr['fd_area'] = FD; Dr['logN_eff'] = N ** 0.25; X8r = Dr[F8].values
    X7v = Dv[F7].values
    FLt = pd.read_parquet(NL.CACHE + '/scripts/d1010/L/feats_LV_train.parquet'); FLt = FLt[FLt.group == 'LV']
    lv_ids = FLt.index.values; pos_lv = pd.Index(ids).get_indexer(lv_ids)
    assert len(lv_ids) == 21 and lv[pos_lv].all() and lv.sum() == 21
    Xlv = d7(FLt, pd.read_parquet(HERE + 's28_V4_LV_train.parquet').loc[lv_ids, 'V4'].values)
    v25o = D['p25']  # round_trip read of blend_v25_oof (as build_u1 rdo)
    o21 = np.full(500, np.nan); o21[idx_c] = o21c

    def fitpred(t, ec, el):
        """fit on clean rows t (bool over the 267), predict clean rows ec (bool over 267) and LV rows el (bool over 21)."""
        off = BU.fitp(Xh[t], yc[t] - o21c[t], Xh[ec])[0]
        f8 = BU.fit_log(X8r[t], yc[t], X8r[ec])[0]
        f7 = BU.fitp(X7v[t], yc[t], Xlv[el])[0]
        return off, f8, f7

    def comps(fa, excl):
        """component arrays (500, NaN elsewhere) for eval fold(s) e with fits on clean rows of folds not in excl(e)."""
        fc, fl = fa[cl], fa[pos_lv]
        out = dict(off=np.full(500, np.nan), f8=np.full(500, np.nan), f7=np.full(500, np.nan))
        for e in range(5):
            if e in excl:
                continue
            t = np.ones(len(fc), bool)
            for x in excl + [e]:
                t &= fc != x
            ec, el = fc == e, fl == e
            off, f8, f7 = fitpred(t, ec, el)
            out['off'][idx_c[ec]] = off; out['f8'][idx_c[ec]] = f8; out['f7'][pos_lv[el]] = f7
        return out

    def pred(o, P):
        w, s, u = o
        q = p29.copy()
        q[cl] = (1 - w) * (o21[cl] + s * P['off'][cl]) + w * P['f8'][cl]
        q[lv] = (1 - u) * v25o[lv] + u * P['f7'][lv]
        return q

    def nested(fa, opts, verbose=False):
        """selection-honest nested OOF over opts (ties -> earlier); also returns the fixed-option nested OOFs."""
        Po = comps(fa, [])                       # outer: fits on folds != k for fold-k rows
        fixed = {o: pred(o, Po) for o in OPTS}
        qh = p29.copy(); picks = []
        for k in range(5):
            Pi = comps(fa, [k])                  # inner: fits on folds not in {j, k} for fold-j rows, j != k
            R = (fa != k) & (cl | lv)
            sc = []
            for o in opts:
                q = pred(o, Pi); sc.append(float(np.sum((q[R] - y[R]) ** 2)))
            b = int(np.argmin(sc)); picks.append(opts[b])
            E = (fa == k) & (cl | lv)
            qh[E] = fixed[opts[b]][E]
            if verbose:
                say('  outer fold %d: pick %s inner SSE %.1f (O0 %.1f) | top3 %s' % (
                    k, oname(opts[b]), sc[b], sc[opts.index(O0)] if O0 in opts else float('nan'),
                    [(oname(opts[i]), round(sc[i] - sc[b], 1)) for i in np.argsort(sc, kind='stable')[:3]]))
        return qh, picks, fixed

    # ---------------------------------------------------------------- reproduction of v29g (OOF)
    qh, picks, fixed = nested(fold, OPTS, verbose=True)
    rep = float(np.abs(fixed[O0] - p29).max()); say('O0 fixed nested OOF vs blend_v29g OOF: max abs diff %.3e' % rep)
    assert rep < 1e-9
    say('O0 bit-identical on all 500 rows: %s' % bool(np.array_equal(fixed[O0], p29)))

    # ---------------------------------------------------------------- test side, as build_u1
    te = pd.read_csv(NL.REPO + '/data/sample_submission.csv'); tid_all = te.ID.values
    assert np.array_equal(tid_all.astype(str), te_ids)
    rd = lambda p: pd.read_csv(p, float_precision='round_trip').set_index('ID').hardness.loc[tid_all].values
    v21b, v25t = rd(SUBD + 'blend_v21base.csv'), rd(SUBD + 'blend_v25.csv')
    tid = tid_all[clt]
    Ft = pd.read_parquet(L1.CH + 'feats_C_test.parquet').loc[tid]; At = pd.read_parquet(L1.RP + 'feats_avg5_test.parquet').loc[tid]
    hbe = pd.read_parquet(NL.REPO + '/data/het_blocks_test.parquet').set_index('ID').loc[tid]
    Nt = Ft.N_eff.values.astype(float); assert np.array_equal(Nt, hbe.N_eff.values.astype(float))
    Dt = pd.DataFrame(index=tid)
    for c in ['fd_area', 'pore_frac', 'logasp_aw', 'S_align', 'het4']:
        Dt[c] = Ft[c].astype(float)
    Dt['logN_eff'] = np.log(Nt); Dt['bd_frac'] = Ft.N_bd / Ft.N_all; Dt['ringall_avg5'] = At.ringall_avg5
    V4t = pd.read_parquet(HERE + 's28_V4_clean_test.parquet').loc[tid, 'V4'].values
    Dtr = Dt.copy(); Dtr['het4'] = V4t
    FDt = pd.read_parquet(IMPL + 'fd6t_test.parquet').loc[tid]
    Dtr['fd_area'] = FDt.FD6t.values; Dtr['logN_eff'] = Nt ** 0.25
    off_t = BU.fitp(Xh, yc - o21c, np.column_stack([V4t, Nt ** 0.25]))[0]
    f8_t = BU.fit_log(X8r, yc, Dtr[F8].values)[0]
    FL = pd.read_parquet(NL.CACHE + '/scripts/d1010/L/feats_LV_test.parquet'); FL = FL[FL.group == 'LV']
    lv_te = te.ID.isin(FL.index).values; assert np.array_equal(lv_te, lvt) and lv_te.sum() == 42
    lid = tid_all[lv_te]; FL = FL.loc[lid]
    f7_t = BU.fitp(X7v, yc, d7(FL, pd.read_parquet(HERE + 's28_V4_LV_test.parquet').loc[lid, 'V4'].values))[0]

    def test_of(o):
        w, s, u = o
        T = t29.copy()
        T[clt] = (1 - w) * (v21b[clt] + s * off_t) + w * f8_t
        T[lvt] = (1 - u) * v25t[lvt] + u * f7_t
        return T
    T0 = test_of(O0); rep_t = float(np.abs(T0 - t29).max())
    say('O0 test vs blend_v29g.csv: max abs diff %.3e, bit-identical %s' % (rep_t, bool(np.array_equal(T0, t29))))
    assert rep_t < 1e-9

    # ---------------------------------------------------------------- in-sample (fixed option) numbers
    say('IN-SAMPLE fixed-option nested (data/folds.csv; the option is NOT chosen honestly here):')
    ins = {}
    for o in OPTS:
        r = NL.vs29(fixed[o], D, nflip=5000)
        ins[oname(o)] = dict(all500_dmse=r['all500_dmse'], folds_better=r['folds_better'], fold_dsse=r['fold_dsse'],
                             signflip_p=r['signflip_p'], droptop3=r['droptop3_all500_dmse'],
                             dmse_clean=r['by_group']['clean']['dmse_group'], dmse_LV=r['by_group']['LV']['dmse_group'],
                             test_change_rms=float(np.sqrt(np.mean((test_of(o) - t29) ** 2))))
        say('  %s  dMSE %+.3f  folds %d/5 %s  clean %+.2f LV %+.2f  test rms %.2f' % (
            oname(o), r['all500_dmse'], r['folds_better'], r['fold_dsse'], ins[oname(o)]['dmse_clean'],
            ins[oname(o)]['dmse_LV'], ins[oname(o)]['test_change_rms']))
    sse = {o: float(np.sum((fixed[o] - y) ** 2)) for o in OPTS}
    pick = OPTS[int(np.argmin([sse[o] for o in OPTS]))]
    nonO0 = OPTS[1:]
    pick_b = nonO0[int(np.argmin([sse[o] for o in nonO0]))]
    say('full-data pick: %s | best non-O0: %s' % (oname(pick), oname(pick_b)))

    # ---------------------------------------------------------------- honest selection-nested
    say('HONEST selection-nested (21 options):')
    H = NL.vs29(qh, D); say('  picks %s' % [oname(o) for o in picks]); say('  ' + json.dumps({k: H[k] for k in ('all500_dmse', 'folds_better', 'fold_dsse', 'signflip_p', 'droptop3_all500_dmse', 'n_changed')}))
    say('HONEST forced-change selection-nested (20 non-O0 options):')
    qf, picks_f, _ = nested(fold, nonO0, verbose=True)
    # POST HOC (added after the pre-declared run above was seen): the joint argmin decomposes exactly into a clean-row
    # argmin and an LV-row argmin (disjoint rows); report each sub-procedure separately (LV / clean fixed at O0).
    CLEAN_ONLY = [o for o in OPTS if o[2] == 0.5]; LV_ONLY = [o for o in OPTS if o[:2] == (0.5, 1.0)]
    assert CLEAN_ONLY[0] == O0 and LV_ONLY[0] == O0 and len(CLEAN_ONLY) == 7 and len(LV_ONLY) == 3
    qc, picks_c, _ = nested(fold, CLEAN_ONLY); qlv, picks_l, _ = nested(fold, LV_ONLY)
    Hc, Hl = NL.vs29(qc, D), NL.vs29(qlv, D)
    for nm, hh, pk in (('clean-only (7 opts, u = 0.5)', Hc, picks_c), ('LV-only (3 opts, w 0.5 s 1)', Hl, picks_l)):
        say('POST HOC honest sub-procedure %s: picks %s %s' % (nm, [oname(o) for o in pk], json.dumps(
            {k: hh[k] for k in ('all500_dmse', 'folds_better', 'fold_dsse', 'signflip_p', 'droptop3_all500_dmse', 'n_changed')})))
    pick_c = CLEAN_ONLY[int(np.argmin([sse[o] for o in CLEAN_ONLY]))]
    Hf = NL.vs29(qf, D); say('  picks %s' % [oname(o) for o in picks_f]); say('  ' + json.dumps({k: Hf[k] for k in ('all500_dmse', 'folds_better', 'fold_dsse', 'signflip_p', 'droptop3_all500_dmse', 'n_changed')}))

    # ---------------------------------------------------------------- re-splits
    rng = np.random.default_rng(20261015)
    rs_h, rs_f, rs_c, rs_l, rs_fix, rs_picks = [], [], [], [], {oname(o): [] for o in OPTS}, {}
    for it in range(a.resplits):
        fa = NL.strat_split(rng, D['glab'])
        qh_r, pk, fx = nested(fa, OPTS)
        qf_r, _, _ = nested(fa, nonO0)
        qc_r, _, _ = nested(fa, CLEAN_ONLY); ql_r, _, _ = nested(fa, LV_ONLY)
        b0 = (fx[O0] - y) ** 2
        rs_c.append(float(np.mean((qc_r - y) ** 2 - b0))); rs_l.append(float(np.mean((ql_r - y) ** 2 - b0)))
        rs_h.append(float(np.mean((qh_r - y) ** 2 - b0))); rs_f.append(float(np.mean((qf_r - y) ** 2 - b0)))
        for o in OPTS:
            rs_fix[oname(o)].append(float(np.mean((fx[o] - y) ** 2 - b0)))
        for o in pk:
            rs_picks[oname(o)] = rs_picks.get(oname(o), 0) + 1
    summ = lambda v: dict(n=len(v), mean=float(np.mean(v)), sd=float(np.std(v)), P_ge0=float(np.mean(np.array(v) >= 0)))
    RS = dict(honest=summ(rs_h), forced_change=summ(rs_f), picks=dict(sorted(rs_picks.items(), key=lambda kv: -kv[1])),
              fixed_mean={k: round(float(np.mean(v)), 3) for k, v in rs_fix.items()},
              posthoc_clean_only=summ(rs_c), posthoc_LV_only=summ(rs_l))
    say('re-splits (%d, strat_split seed 20261015): honest %s | forced-change %s' % (a.resplits, RS['honest'], RS['forced_change']))
    say('  picks over %d outer folds: %s' % (5 * a.resplits, RS['picks']))
    say('  fixed-option mean dMSE vs O0 (descriptive): %s' % RS['fixed_mean'])
    say('  POST HOC sub-procedures on the same re-splits: clean-only %s | LV-only %s' % (RS['posthoc_clean_only'], RS['posthoc_LV_only']))

    # ---------------------------------------------------------------- write
    tier = lambda h: 'SAFE' if (h['all500_dmse'] < 0 and h['folds_better'] >= 3) else ('BOLD?' if h['all500_dmse'] <= 0.5 else 'FAIL')
    common = dict(options_file=OUT + 'options.txt', options_sha256=open(OUT + 'options.sha256').read().split()[0],
                  builder=OUT + 'build_clean.py', options=[oname(o) for o in OPTS], insample=ins, resplits=RS,
                  honest_picks=[oname(o) for o in picks], honest=H, forced_change_picks=[oname(o) for o in picks_f],
                  forced_change_honest=Hf, reproduce=dict(oof_max_abs=rep, test_max_abs=rep_t),
                  posthoc=dict(note='added after the pre-declared result was seen; not pre-declared', clean_only_picks=[oname(o) for o in picks_c],
                               clean_only_honest=Hc, LV_only_picks=[oname(o) for o in picks_l], LV_only_honest=Hl,
                               clean_only_full_pick=oname(pick_c)))
    T = test_of(pick)
    meta = dict(common, name='clean', full_data_pick=oname(pick), tier_rule=tier(H),
                note='oof.csv = honest selection-nested OOF; test.csv = full-data pick applied to v29g (clean+LV rows)')
    res = NL.write('clean', qh, T, D, meta)
    say('clean: pick %s, test md5 %s, rows changed %d, public %s' % (
        oname(pick), res['test_md5'], int((T != t29).sum()), json.dumps({k: res['public'][k] for k in ('cv_hyp_public_dmse', 'cv_hyp_sd', 'null_public_dmse', 'null_sd')})))
    if pick == O0:
        Tb = test_of(pick_b)
        metab = dict(common, name='clean_bold', option=oname(pick_b), tier_rule_forced_change=tier(Hf),
                     note='oof.csv = fixed-option nested OOF of the forced option (in-sample about the choice); '
                          'forced_change_honest = selection-honest over the 20 non-O0 options')
        resb = NL.write('clean_bold', fixed[pick_b], Tb, D, metab)
        say('clean_bold: %s, test md5 %s, rows changed %d, test change rms %.3f, public %s' % (
            oname(pick_b), resb['test_md5'], int((Tb != t29).sum()), float(np.sqrt(np.mean((Tb - t29) ** 2))),
            json.dumps({k: resb['public'][k] for k in ('cv_hyp_public_dmse', 'cv_hyp_sd', 'null_public_dmse', 'null_sd')})))
    if pick_c != O0:
        Tc = test_of(pick_c)
        metac = dict(common, name='clean/posthoc_clean_only', option=oname(pick_c), tier_rule_posthoc=tier(Hc),
                     note='POST HOC (sub-procedure chosen after the pre-declared joint result was seen): clean rows only, '
                          'LV rows = v29g. oof.csv = honest selection-nested OOF of the 7-option clean-only sub-procedure.')
        resc = NL.write('clean/posthoc_clean_only', qc, Tc, D, metac)
        say('posthoc_clean_only: %s, test md5 %s, rows changed %d, test change rms %.3f, public %s' % (
            oname(pick_c), resc['test_md5'], int((Tc != t29).sum()), float(np.sqrt(np.mean((Tc - t29) ** 2))),
            json.dumps({k: resc['public'][k] for k in ('cv_hyp_public_dmse', 'cv_hyp_sd', 'null_public_dmse', 'null_sd')})))
    json.dump(dict(common, log=LOG, seconds=time.time() - t0), open(OUT + 'options_eval.json', 'w'), indent=1, default=float)
    say('done in %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
