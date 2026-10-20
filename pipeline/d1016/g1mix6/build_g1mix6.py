"""d1016/g1mix6 builder (options.txt, sha256 asserted) = d1016/g1mix5/build_g1mix5.py with the g1mix6 reader lists, the XNfixed
reference (= the exact blend_v35xn rule, md5 asserted), kriging18 and the g1mix6 SAFE/BOLD flags (main thread, 2026-10-16).
usage (repo root, after ingesting the new readers): nice -n 5 python /mnt/project-files/work/hardness-cache/scripts/d1016/g1mix6/build_g1mix6.py
env OUT=<dir>: write the option folders there instead of d1016/g1mix6. env ALLOW_PARTIAL=1: also build options whose families
miss seeds (self-tests and scratch looks only; options.txt builds an option only when every seed of its families is in).
Writes <OUT>/<opt>/{oof.csv,test.csv,result.json} and <OUT>/summary.json (test files stay in the cache; never in the repo).
Labels: train only. Test rows enter only as label-free group membership, the T-CP gate flag and the reader test outputs."""
import os, sys, io, json, math, hashlib, contextlib, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
SPEC = '/mnt/project-files/work/hardness-cache/scripts/d1016/g1mix6'
OPT_SHA = 'a585cfbdca25435a4eb8bbf81208cd43f1c55543fc5a7451d0095f8b6fbbbecf'
assert hashlib.sha256(open(SPEC + '/options.txt', 'rb').read()).hexdigest() == OPT_SHA, 'options.txt changed'
sys.path.insert(0, '/mnt/project-files/work/hardness-cache/scripts/d1011/N4/g1')
sys.path.insert(0, '/mnt/project-files/work/hardness-cache/scripts/d1011/N5')
import g1core
from g1core import *  # noqa  (rebinds HERE to N4/g1: never use HERE below)
import n5lib
OUT = os.environ.get('OUT', SPEC)

t0 = time.time()
A3 = 'ev2s_locK_real5'
FAM_RUNS = dict(A3f=[A3, A3 + '_s1', A3 + '_s2', A3 + '_s3', A3 + '_s4'],
                B1f=['ev2s_locK_fsim', os.environ.get('B1S1_RUN', 'ev2s_locK_fsim_s1'), os.environ.get('B1S2_RUN', 'ev2s_locK_fsim_s2')],
                B4f=['ev2m_locK_real5', os.environ.get('B4S1_RUN', 'ev2m_locK_real5_s1')],
                NSf=['b5ns_locK_real5', 'b5ns_locK_real5_s1', 'b5ns_locK_real5_s2'],
                B7f=['ev2l_locK_fsim', 'ev2l_locK_fsim_s1', 'ev2l_locK_fsim_s2'],
                B8f=['b5ns_locK_fsim', 'b5ns_locK_fsim_s1', 'b5ns_locK_fsim_s2'])
NEW_RUNS = set(FAM_RUNS['B1f'][1:] + FAM_RUNS['B4f'][1:] + FAM_RUNS['NSf'] + FAM_RUNS['B7f'] + FAM_RUNS['B8f'])
OPTIONS = dict(XN8=['B1f', 'B4f', 'NSf', 'B8f'], XN7=['B1f', 'B4f', 'NSf', 'B7f'], XN78=['B1f', 'B4f', 'NSf', 'B7f', 'B8f'],
               X8=['B1f', 'B4f', 'B8f'], N2=['NSf', 'B8f'])
DIAG = dict(XNf=['B1f', 'B4f', 'NSf'], NSf=['NSf'], B8f=['B8f'])
ALLOW_PARTIAL = os.environ.get('ALLOW_PARTIAL', '') == '1'
present = lambda r: all(os.path.exists(f'{HETD}{r}/het_cnn_{s}.parquet') for s in ('train', 'test'))

# label-free health gate on the 208 G1 train rows: het corr with the A3 family mean >= 0.80
D5 = n5lib.load30()
G1 = D5['G1']
ids_tr = D5['ids']
hA3, _ = avg_dirs([r for r in FAM_RUNS['A3f'] if present(r)], 'train', ids_tr)
gate = {}
for f, runs in FAM_RUNS.items():
    for r in runs:
        if not present(r):
            gate[r] = 'missing'; continue
        if r not in NEW_RUNS:
            gate[r] = 'ok (old reader)'; continue
        h, _ = avg_dirs([r], 'train', ids_tr)
        c = float(np.corrcoef(h[G1], hA3[G1])[0, 1])
        gate[r] = f'ok (corr {c:.3f})' if c >= 0.80 else f'DROPPED: broken run (corr {c:.3f} < 0.80)'
FAM = {f: [r for r in runs if gate[r].startswith('ok')] for f, runs in FAM_RUNS.items()}
FAM = {f: v for f, v in FAM.items() if v}
print('gate:', json.dumps(gate), flush=True)
print('families:', json.dumps(FAM), flush=True)


def runs_of(fams):
    fams = [f for f in fams if f in FAM]
    if not fams:
        return [], []
    L = 1
    for f in fams:
        L = L * len(FAM[f]) // math.gcd(L, len(FAM[f]))
    return fams, [r for f in fams for r in FAM[f] for _ in range(L // len(FAM[f]))]


ALLOPT = {}
for k, fl in list(OPTIONS.items()) + list(DIAG.items()):
    used, runs = runs_of(fl)
    if not runs:
        print(f'{k}: no family present, skipped', flush=True); continue
    short = [f for f in fl if len(FAM.get(f, [])) < len(FAM_RUNS[f])]
    if short and not ALLOW_PARTIAL:
        print(f'{k}: families missing seeds {short}, not built (options.txt seed rule)', flush=True); continue
    key = 'G6_' + k
    assert key not in g1core.EST_DIRS
    g1core.EST_DIRS[key] = runs; g1core.HSPEC['replace_' + key] = ([key], False)
    ALLOPT[k] = dict(families=used, missing=[f for f in fl if f not in used], runs=runs, kind='option' if k in OPTIONS else 'diagnostic')
# fair references besides the fixed replace_A3/0.75 rule: the fixed d1015 XB4 option (B1 seed 0 + B4 seed 0) and the exact
# v35xn rule (= g1mix4 XN: B1f 3 seeds x2, B4f 2 seeds x3, NS seed 0 x6, the runs_of order of build_g1mix4.py)
g1core.EST_DIRS['G3_REF_XB4'] = ['ev2s_locK_fsim', 'ev2m_locK_real5']; g1core.HSPEC['replace_G3_REF_XB4'] = (['G3_REF_XB4'], False)
REF_XN_RUNS = [r for r in FAM_RUNS['B1f'] for _ in range(2)] + [r for r in FAM_RUNS['B4f'] for _ in range(3)] + ['b5ns_locK_real5'] * 6
assert all(present(r) for r in REF_XN_RUNS), 'XNfixed readers missing'
g1core.EST_DIRS['G6_REF_XN'] = REF_XN_RUNS; g1core.HSPEC['replace_G6_REF_XN'] = (['G6_REF_XN'], False)
C = Ctx(ests=('LOC', 'A2', 'A3', 'A23', 'G3_REF_XB4', 'G6_REF_XN') + tuple('G6_' + k for k in ALLOPT))
D = C.D; y, fold, ids, te = C.y, C.fold, C.ids, D['te_ids']
G1t = D5['G1t']
p30, t30, t21, t27, t28, t29 = D5['p30'], D5['t30'], D5['t21'], D5['t27'], D5['t28'], D5['t29']
nzt, rst = D['nzt'], D['rst']
mb = pd.read_parquet(CACHE + '/scripts/d1011/midlf/m3b_rows.parquet', columns=['split', 'group', 'gate_cp'])
mt = mb[mb.split == 'test'].loc[te]
tcp_t = (mt.group.to_numpy() == 'midLF') & mt.gate_cp.to_numpy().astype(bool)
assert tcp_t.sum() == 19
wt = np.where(tcp_t, 0.5, 1.0) * G1t


def mapt(n):
    d = n - t28; d[np.abs(d) < 1e-9] = 0.0
    return t29 + wt * d


def test_g1_clip(opt):
    """= d1015/g1mix2 test_g1_clip: full-data fits on train G1 rows only."""
    h, s = opt
    names, avg = HSPEC[h]; assert not avg and len(names) == 1
    n = names[0]
    G, nz, r = C.G, C.nz, C.r
    Gt = (nzt >= 12) & (rst < 0.5)
    hh, ll = C.EST[n]; hht, llt = avg_dirs(g1core.EST_DIRS[n], 'test', te)
    g = np.polyfit(ll[G1], hh[G1], 1)
    Hm = (hh - np.polyval(g, ll))[:, None]; Hmt = (hht - np.polyval(g, llt))[:, None]
    assert np.isfinite(Hmt[G1t]).all()
    beta = np.linalg.lstsq(Hm[G1], r[G1] - r[G1].mean(), rcond=None)[0]
    off_tr, off_te = Hm @ beta, Hmt @ beta
    cap = float(np.abs(off_tr[G1]).max())
    base_t = t27.copy(); base_t[G1t] = t21[G1t] + np.clip(off_te, -cap, cap)[G1t]
    bo = C.base(h, C.fold, C.ALL)
    m = float(np.median(nz[G])); lo = G & (nz < m)
    c = float(bo[lo].mean()); b = float(np.polyfit(bo[lo], y[lo], 1)[0]); k = 1 + s * (b / RHO - 1)
    fte = Gt & (nzt < m)
    new_t = base_t.copy(); new_t[fte] = c + k * (base_t[fte] - c)
    return new_t, dict(g=[float(v) for v in g], beta=float(beta[0]), cap=cap,
                       n_test_beyond_cap=int((np.abs(off_te[G1t]) > cap).sum()), m=m, c=c, b=b, k=k,
                       offset_sd_train_G1=float(off_tr[G1].std()), offset_sd_test_G1=float(off_te[G1t].std()))


# D1: machinery reproduces v30b (procedure OOF and test) with the original readers
q0, pk0, _ = C.nested(fold, OPTS)
assert np.max(np.abs(C.mapped(q0) - p30)) < 1e-9
n0, _ = test_g1_clip(('replace_A3', 0.75))
assert np.max(np.abs(mapt(n0) - t30)) < 1e-9
print('D1 ok: v30b OOF and test reproduced', flush=True)

KNS = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.environ.get('KRIG', CACHE + '/scripts/d1016/pub/kriging16.py')).read(), KNS)  # kriging16 = forensics15 + the 4 10-15 readings (15:05 KST 10-15)
KNS17, KNS18 = {}, {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.environ.get('KRIG17', CACHE + '/scripts/d1016/pub/kriging17.py')).read(), KNS17)  # kriging17 = kriging16 + v34l2
    exec(open(os.environ.get('KRIG18', CACHE + '/scripts/d1016/pub/kriging18.py')).read(), KNS18)  # kriging18 = kriging17 + v35xn (the g1mix6 bars)
SUBMITTED = dict(v33xb='7b3e5c08483c6ff58f7ce3cfc42c804b', v33b4='de0684eb', v33t3='07ab14d2', v33s5='9e991b2f',
                 v34l2='698f5eb67d2f6af5e646cbd878ee767f', v35xn='d1eab1318577e0b8f962e253632b9e8d')
TSUB = {}
for n_, m_ in SUBMITTED.items():
    fp = CACHE + f'/submissions/blend_{n_}.csv'
    assert n5lib.md5(fp).startswith(m_), f'{n_} md5'
    TSUB[n_] = pd.read_csv(fp, float_precision='round_trip').set_index('ID').hardness.loc[te].to_numpy(float)
BEST_PUB, BOLD_KR = 9.9722, 9.9854

rng = np.random.default_rng(20261015)
SPL = [strat_split(rng, D['glab']) for _ in range(40)]
# fair references: the fixed replace_A3/0.75 rule (v30b's test rule) and the fixed d1015 XB4 option
REFS = {'A3fixed': ('replace_A3', 0.75), 'XB4fixed': ('replace_G3_REF_XB4', 0.75), 'XNfixed': ('replace_G6_REF_XN', 0.75)}
REFQ = {k: C.mapped(C.build(o, fold)) for k, o in REFS.items()}
REFR = {k: [C.mapped(C.build(o, ff)) for ff in SPL] for k, o in REFS.items()}
t33xb = mapt(test_g1_clip(REFS['XB4fixed'])[0])
x4md5 = hashlib.md5(pd.DataFrame({'ID': te, 'hardness': t33xb}).to_csv(index=False).encode()).hexdigest()
assert x4md5 == '7b3e5c08483c6ff58f7ce3cfc42c804b', 'v33xb test not reproduced'
print('XB4 reference test md5', x4md5, '(d1015 g1mix2 XB4: 7b3e5c08483c6ff58f7ce3cfc42c804b)', flush=True)
t35xn = mapt(test_g1_clip(REFS['XNfixed'])[0])
xnmd5 = hashlib.md5(pd.DataFrame({'ID': te, 'hardness': t35xn}).to_csv(index=False).encode()).hexdigest()
assert xnmd5 == 'd1eab1318577e0b8f962e253632b9e8d', 'v35xn test not reproduced'
assert np.array_equal(t35xn, TSUB['v35xn'])
print('XN reference test md5', xnmd5, '(blend_v35xn: d1eab1318577e0b8f962e253632b9e8d)', flush=True)


def vs_ref(q, opt, k):
    d = (q - y) ** 2 - (REFQ[k] - y) ** 2
    per = [float(d[fold == j].sum()) for j in range(5)]
    o = np.argsort(d); keep = np.ones(500, bool); keep[o[:3]] = False
    R = np.array([float(np.mean((C.mapped(C.build(opt, ff)) - y) ** 2) - np.mean((REFR[k][i] - y) ** 2)) for i, ff in enumerate(SPL)])
    return dict(oof_dmse=float(d.mean()), folds_better=int(sum(x < 0 for x in per)), fold_dsse=[round(x, 1) for x in per],
                droptop3=float(d[keep].mean()), resplit40=dict(mean=float(R.mean()), sd=float(R.std()), P_ge0=float((R >= 0).mean())))


out = {}
for nm, spec in ALLOPT.items():
    opt = ('replace_G6_' + nm, 0.75)
    q = C.mapped(C.build(opt, fold))
    S = n5lib.vs30(q, D5)
    VA, VX, VN = vs_ref(q, opt, 'A3fixed'), vs_ref(q, opt, 'XB4fixed'), vs_ref(q, opt, 'XNfixed')
    n_, info = test_g1_clip(opt)
    tnew = mapt(n_)
    assert ((tnew != t30) <= G1t).all() and np.array_equal(tnew[~G1t], t30[~G1t])
    n5lib.check_test(tnew, D5)
    u = tnew - t30
    kr = KNS['kpred'](u)
    kr17 = KNS17['kpred'](u)
    kr18 = KNS18['kpred'](u)
    rms_sub = {n_: float(np.sqrt(np.mean((tnew - t_)[G1t] ** 2))) for n_, t_ in TSUB.items()}
    safe = VN['oof_dmse'] < 0 and VN['folds_better'] >= 3 and VN['resplit40']['P_ge0'] <= 0.5 and kr18['rmse'] < BEST_PUB
    bold = VN['oof_dmse'] <= 0.42 and kr18['rmse'] <= BOLD_KR and rms_sub['v35xn'] >= 0.5
    skip = min(rms_sub.values()) < 0.25
    flag = 'diagnostic' if spec['kind'] != 'option' else ('SKIP(rms<0.25)' if skip else ('SAFE' if safe else ('BOLD' if bold else 'no')))
    d = f'{OUT}/{nm}'; os.makedirs(d, exist_ok=True)
    pd.DataFrame({'ID': ids, 'hardness': q}).to_csv(d + '/oof.csv', index=False)
    pd.DataFrame({'ID': te, 'hardness': tnew}).to_csv(d + '/test.csv', index=False)
    ch = tnew != t30
    res = dict(option=nm, kind=spec['kind'], families=spec['families'], families_missing=spec['missing'],
               family_runs={f: FAM[f] for f in spec['families']}, n_reader_entries=len(spec['runs']), gate=gate, step=0.75,
               options_sha256=OPT_SHA, honest_fixed_option_vs_v30b_p30=S, vs_A3fixed=VA, vs_XB4fixed=VX, vs_XNfixed=VN,
               test_rule=info, n_test_changed=int(ch.sum()), test_change=dict(rms=float(np.sqrt(np.mean(u[ch] ** 2))), max_abs=float(np.abs(u).max())),
               test_change_vs_v33xb=dict(rms_G1t=float(np.sqrt(np.mean((tnew - t33xb)[G1t] ** 2))), max_abs=float(np.abs(tnew - t33xb).max())),
               kriging_public=kr, kriging17_public=kr17, kriging18_public=kr18, rms_G1t_vs_submitted=rms_sub, slate_flag=flag,
               test_md5=n5lib.md5(d + '/test.csv'))
    json.dump(res, open(d + '/result.json', 'w'), indent=1, default=float)
    out[nm] = res
    print(f"{nm:6s} [{spec['kind'][:4]}] fams {'+'.join(spec['families'])}{' (missing ' + ','.join(spec['missing']) + ')' if spec['missing'] else ''} | "
          f"vs p30 {S['all500_dmse']:+.3f} | vs A3fixed {VA['oof_dmse']:+.3f} {VA['folds_better']}/5 dt3 {VA['droptop3']:+.3f} rs {VA['resplit40']['mean']:+.3f} (P {VA['resplit40']['P_ge0']:.2f}) | "
          f"vs XB4fixed {VX['oof_dmse']:+.3f} {VX['folds_better']}/5 | vs XNfixed {VN['oof_dmse']:+.3f} {VN['folds_better']}/5 {VN['fold_dsse']} dt3 {VN['droptop3']:+.3f} rs {VN['resplit40']['mean']:+.3f} (P {VN['resplit40']['P_ge0']:.2f}) | "
          f"test rms {res['test_change']['rms']:.3f} vs33xb {res['test_change_vs_v33xb']['rms_G1t']:.3f} | kriging {kr['pred']:+.3f} +- {kr['sd']:.3f} -> {kr['rmse']:.4f} | k17 {kr17['rmse']:.4f} | k18 {kr18['pred']:+.3f} +- {kr18['sd']:.3f} -> {kr18['rmse']:.4f} | rms vs35xn {rms_sub['v35xn']:.3f} min-sub {min(rms_sub.values()):.3f} | {flag} | md5 {res['test_md5']}", flush=True)
json.dump({k: dict(kind=v['kind'], families=v['families'], vs_p30=v['honest_fixed_option_vs_v30b_p30']['all500_dmse'],
                   vs_A3fixed=v['vs_A3fixed'], vs_XB4fixed=v['vs_XB4fixed'], kr=v['kriging_public']['pred'], kr_sd=v['kriging_public']['sd'], kr_rmse=v['kriging_public']['rmse'], rms_vs33xb=v['test_change_vs_v33xb']['rms_G1t'],
                   vs_XNfixed=v['vs_XNfixed'], kr17_rmse=v['kriging17_public']['rmse'], kr18=v['kriging18_public']['pred'], kr18_sd=v['kriging18_public']['sd'],
                   kr18_rmse=v['kriging18_public']['rmse'], rms_sub=v['rms_G1t_vs_submitted'], flag=v['slate_flag'],
                   md5=v['test_md5']) for k, v in out.items()}, open(OUT + '/summary.json', 'w'), indent=1)
print('elapsed %.0fs' % (time.time() - t0))
