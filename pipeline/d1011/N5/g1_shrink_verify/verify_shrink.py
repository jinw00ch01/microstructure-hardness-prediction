"""Independent rebuild of N5/g1_shrink from its options.txt (not from g1_shrink.py).
Reuses only the shared machinery: g1core.Ctx (data loading, EST readers, r = y - v21 OOF, mapped()), g1core.OPTS/HSPEC,
n5lib.load30/vs30/strat_split. The lambda-shrunk offsets, base, stretch, two-stage nested, full pick and the test vector are
written here from scratch. Writes only into N5/g1_shrink_verify/.
usage (repo root): nice -n 10 timeout 1800 python <this file>"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, json, time, hashlib
sys.path.insert(0, '/mnt/project-files/work/hardness-cache/scripts/d1011/N4/g1')
sys.path.insert(0, '/mnt/project-files/work/hardness-cache/scripts/d1011/N5')
import g1core
from g1core import Ctx, OPTS, HSPEC, EST_DIRS, avg_dirs
import n5lib
from n5lib import np, pd, RHO, SUB, CACHE, strat_split

OUT = '/mnt/project-files/work/hardness-cache/scripts/d1011/N5/g1_shrink_verify'
CAND = '/mnt/project-files/work/hardness-cache/scripts/d1011/N5/g1_shrink'
LAMS = {'s100': 1.0, 's90': 0.9, 's80': 0.8, 's70': 0.7, 's60': 0.6}
t0 = time.time()
D = n5lib.load30()
C = Ctx()
y, fold, ids, te = C.y, C.fold, C.ids, D['te_ids']
assert np.array_equal(y, D['y']) and np.array_equal(fold, D['fold'])
G1, G, nz = C.G1, C.G, C.nz
ALL = np.ones(500, bool)


# ---------------------------------------------------------------- own lambda recipe
def my_offsets(names, avg, ff, fm, lam):
    off = np.zeros(500)
    E, r = C.EST, C.r
    for j in range(5):
        fit = G1 & fm & (ff != j); app = G1 & (ff == j)
        if avg:
            hh = np.mean([E[n][0] for n in names], 0); ll = E['LOC'][1]
            cf = np.polyfit(ll[fit], hh[fit], 1)
            H = (hh - np.polyval(cf, ll)).reshape(-1, 1)
        else:
            cols = []
            for n in names:
                cf = np.polyfit(E[n][1][fit], E[n][0][fit], 1)
                cols.append(E[n][0] - np.polyval(cf, E[n][1]))
            H = np.stack(cols, 1)
        rc = r[fit] - r[fit].mean()
        b0 = np.linalg.lstsq(H[fit], rc, rcond=None)[0]
        off[app] = H[app] @ (lam * b0)
    return off


_bc = {}
def my_base(h, ff, fm, lam):
    key = (h, ff.tobytes(), fm.tobytes(), lam)
    if key not in _bc:
        names, avg = HSPEC[h]
        b = C.p27.copy()
        b[G1] = C.o21[G1] + my_offsets(names, avg, ff, fm, lam)[G1]
        _bc[key] = b
        if len(_bc) > 6000:
            _bc.clear(); _bc[key] = b
    return _bc[key]


def my_stretch(base, ff, fm, s):
    q = base.copy()
    for j in range(5):
        g = G & fm & (ff != j)
        m = float(np.median(nz[g])); lo = g & (nz < m)
        c = float(base[lo].mean())
        b = float(np.polyfit(base[lo], y[lo], 1)[0])
        kk = 1.0 + s * (b / RHO - 1.0)
        a = G & (ff == j) & (nz < m)
        q[a] = c + kk * (base[a] - c)
    return q


def my_build(opt, ff, fm, lam):
    h, s = opt
    return my_stretch(my_base(h, ff, fm, lam), ff, fm, s)


MAP = C.mapped
mse = lambda a, b: float(np.mean((a - b) ** 2))


def v30_procedure(ff):
    q = np.zeros(500); picks = []
    for k in range(5):
        T = ff != k
        sc = {nm: mse(MAP(my_build(o, ff, T, 1.0))[T], y[T]) for nm, o in OPTS.items()}
        best = min(sc, key=sc.get); picks.append(best)
        q[ff == k] = my_build(OPTS[best], ff, ALL, 1.0)[ff == k]
    return MAP(q), picks


def two_stage(ff, lams=LAMS):
    q = np.zeros(500); hp = []; lp = []
    for k in range(5):
        T = ff != k
        sc = {nm: mse(MAP(my_build(o, ff, T, 1.0))[T], y[T]) for nm, o in OPTS.items()}
        h = min(sc, key=sc.get)
        sl = {ln: mse(MAP(my_build(OPTS[h], ff, T, lv))[T], y[T]) for ln, lv in lams.items()}
        l = min(sl, key=sl.get)
        hp.append(h); lp.append(l)
        q[ff == k] = my_build(OPTS[h], ff, ALL, lams[l])[ff == k]
    return MAP(q), hp, lp


rep = {}
# check: lambda forced to 1 -> p30 ; v30 procedure -> p30
q1, _, _ = two_stage(fold, {'s100': 1.0})
pv, pv_picks = v30_procedure(fold)
rep['forced1_vs_p30'] = float(np.max(np.abs(q1 - D['p30'])))
rep['v30proc_vs_p30'] = float(np.max(np.abs(pv - D['p30'])))
print('checks', rep, flush=True)

qh, hp, lp = two_stage(fold)
S = n5lib.vs30(qh, D)
print(f"HONEST: dMSE {S['all500_dmse']:+.4f} folds {S['folds_better']}/5 {S['fold_dsse']} droptop3 {S['droptop3_all500_dmse']:+.4f} "
      f"rmse {S['rmse_v30b']:.4f}->{S['rmse_new']:.4f}; h {hp} lam {lp}", flush=True)

cand_oof = pd.read_csv(CAND + '/oof.csv', float_precision='round_trip')
assert (cand_oof.ID.astype(str).to_numpy() == ids).all()
rep['max_abs_oof'] = float(np.max(np.abs(cand_oof.hardness.to_numpy() - qh)))
print('max|d| OOF vs candidate', rep['max_abs_oof'], flush=True)

# full pick
sch = {nm: mse(MAP(my_build(o, fold, ALL, 1.0)), y) for nm, o in OPTS.items()}
h_star = min(sch, key=sch.get)
scl = {ln: mse(MAP(my_build(OPTS[h_star], fold, ALL, lv)), y) for ln, lv in LAMS.items()}
l_star = min(scl, key=scl.get)
print('full pick', h_star, {k: round(v, 4) for k, v in scl.items()}, '->', l_star, flush=True)
insample = {ln: float(np.mean((MAP(my_build(OPTS[h_star], fold, ALL, lv)) - y) ** 2) - np.mean((D['p30'] - y) ** 2)) for ln, lv in LAMS.items()}
print('insample fixed vs p30', {k: round(v, 4) for k, v in insample.items()}, flush=True)

# ---------------------------------------------------------------- own test vector
nzt, rst = D['nzt'], D['rst']
G1t = (nzt >= 9.5) & (rst < 0.5); Gt = (nzt >= 12) & (rst < 0.5)
assert np.array_equal(Gt, D['grpt']['G1hi'])
t21, t27, t28, t29, t30 = D['t21'], D['t27'], D['t28'], D['t29'], D['t30']
mb = pd.read_parquet(CACHE + '/scripts/d1011/midlf/m3b_rows.parquet', columns=['split', 'group', 'gate_cp'])
mt = mb[mb.split == 'test'].loc[te]
tcp_t = (mt.group.to_numpy() == 'midLF') & mt.gate_cp.to_numpy().astype(bool)
assert tcp_t.sum() == 19
# independent cross-check of the T-CP rows: v29g differs from v28n exactly there among G1 test rows
assert np.array_equal(tcp_t, G1t & (np.abs(t29 - t28) > 1e-9))
wt = np.where(tcp_t, 0.5, 1.0) * G1t


def my_test(h, s, lam):
    names, avg = HSPEC[h]
    assert not avg and names == ['A3']
    trh = pd.read_parquet(f'{g1core.HETD}ev2s_locK_real5/het_cnn_train.parquet').set_index('ID').loc[ids]
    teh = pd.read_parquet(f'{g1core.HETD}ev2s_locK_real5/het_cnn_test.parquet').set_index('ID').loc[te]
    het, lgn = trh.het_cnn.to_numpy(), trh.logN_cnn.to_numpy()
    hett, lgnt = teh.het_cnn.to_numpy(), teh.logN_cnn.to_numpy()
    assert np.allclose(het, C.EST['A3'][0]) and np.allclose(lgn, C.EST['A3'][1])
    cf = np.polyfit(lgn[G1], het[G1], 1)
    H = het - np.polyval(cf, lgn); Ht = hett - np.polyval(cf, lgnt)
    r = C.r
    b0 = float(np.linalg.lstsq(H[G1].reshape(-1, 1), r[G1] - r[G1].mean(), rcond=None)[0][0])
    beta = lam * b0
    off_tr, off_te = beta * H, beta * Ht
    cap = float(np.abs(off_tr[G1]).max())
    nb = int((np.abs(off_te[G1t]) > cap).sum())
    base_t = t27.copy(); base_t[G1t] = t21[G1t] + np.clip(off_te, -cap, cap)[G1t]
    bo = my_base(h, fold, ALL, lam)
    m = float(np.median(nz[G])); lo = G & (nz < m)
    c = float(bo[lo].mean()); b = float(np.polyfit(bo[lo], y[lo], 1)[0]); k = 1 + s * (b / RHO - 1)
    fte = Gt & (nzt < m)
    nt = base_t.copy(); nt[fte] = c + k * (base_t[fte] - c)
    d = nt - t28; d[np.abs(d) < 1e-9] = 0.0
    tm = t29 + wt * d
    info = dict(b0=b0, beta=beta, cap=cap, n_beyond=nb, m=m, c=c, b=b, k=k, n_stretch=int(fte.sum()),
                sd_ratio=float(off_te[G1t].std() / off_tr[G1].std()))
    return tm, info, fte


h, s = OPTS[h_star]
t1, i1, _ = my_test(h, s, 1.0)
rep['test_lam1_vs_t30'] = float(np.max(np.abs(t1 - t30)))
rep['test_lam1_nonG1_bitequal'] = bool(np.array_equal(t1[~G1t], t30[~G1t]))
print('lam1 test vs v30b', rep['test_lam1_vs_t30'], 'info', i1, flush=True)
tn, inf, fte = my_test(h, s, LAMS[l_star])
print('test info', inf, flush=True)
cand_t = pd.read_csv(CAND + '/test.csv', float_precision='round_trip')
ss = pd.read_csv('data/sample_submission.csv')
rep['ids_match_sample'] = bool((cand_t.ID.astype(str).to_numpy() == ss.ID.astype(str).to_numpy()).all() and len(cand_t) == 1000)
rep['ids_match_te'] = bool((cand_t.ID.astype(str).to_numpy() == te).all())
ct = cand_t.hardness.to_numpy()
rep['finite'] = bool(np.isfinite(ct).all())
rep['max_abs_test'] = float(np.max(np.abs(ct - tn)))
v30 = pd.read_csv(SUB + '/blend_v30b.csv', float_precision='round_trip')
assert (v30.ID.astype(str).to_numpy() == te).all()
rep['nonG1_bitequal_v30b'] = bool(np.array_equal(ct[~G1t], v30.hardness.to_numpy()[~G1t]))
# also the raw text of the non-G1 rows
lines_c = open(CAND + '/test.csv').read().splitlines(); lines_v = open(SUB + '/blend_v30b.csv').read().splitlines()
rep['header_c'] = lines_c[0]; rep['header_v'] = lines_v[0]
rep['nonG1_text_equal'] = bool(all(lines_c[i + 1] == lines_v[i + 1] for i in np.where(~G1t)[0]))
rep['n_changed_vs_v30b'] = int((ct != t30).sum())
rep['test_md5'] = hashlib.md5(open(CAND + '/test.csv', 'rb').read()).hexdigest()
dd = ct - t30
grp = {'midLF': D['grpt']['midLF'], 'G1hi_lowhalf_stretch': fte, 'G1hi_tophalf': Gt & ~fte, 'TCP': tcp_t}
rep['per_group_test_change'] = {g: dict(n=int(mk.sum()), rms=float(np.sqrt(np.mean(dd[mk] ** 2))), mean=float(dd[mk].mean()),
                                        maxabs=float(np.abs(dd[mk]).max())) for g, mk in grp.items()}
rep['all_G1_change_rms'] = float(np.sqrt(np.mean(dd[G1t] ** 2)))
print(json.dumps(rep, indent=1), flush=True)

# magnitude decomposition: expected change from shrinking the clipped offsets by 0.3 (old cap) plus the stretch change
_, _, _ = None, None, None

# ---------------------------------------------------------------- re-splits
rng = np.random.default_rng(20261015); R = []; LF = {k: 0 for k in LAMS}
for i in range(40):
    ff = strat_split(rng, D['glab'])
    qn, _, lp_ = two_stage(ff); qb, _ = v30_procedure(ff)
    R.append(mse(qn, y) - mse(qb, y))
    for l in lp_:
        LF[l] += 1
R = np.array(R)
print(f'resplit40 mean {R.mean():+.4f} sd {R.std():.4f} P>=0 {(R >= 0).mean():.3f}; lambda freq {LF}', flush=True)
res = dict(checks=rep, honest=dict(S=S, h=hp, lam=lp), full_pick=dict(h=h_star, lam=l_star, scores=scl), insample=insample,
           test_info=inf, test_lam1_info=i1, resplit=dict(mean=float(R.mean()), sd=float(R.std()), P=float((R >= 0).mean()), freq=LF),
           elapsed=time.time() - t0)
json.dump(res, open(OUT + '/verify_result.json', 'w'), indent=1, default=float)
pd.DataFrame({'ID': ids, 'hardness': qh}).to_csv(OUT + '/verify_oof.csv', index=False)
pd.DataFrame({'ID': te, 'hardness': tn}).to_csv(OUT + '/verify_test.csv', index=False)
print('elapsed %.1fs' % (time.time() - t0))
