"""N4/g1 core (train side + test-row recipe). Recipe = L4_noisy/gpu/posthoc/nested_select_g1.py (het offset on the G1
rows, then the v28n G1hi low-half stretch) with the stretch step s as a second option axis: k = 1 + s (b / RHO - 1).
Options and procedure are pre-declared in options.txt (sha256 in options.sha256). Labels: train only (lib_l4.load ->
src.common.load_train). Test rows enter only as label-free group membership, the T-CP gate flag and the het CNN test
outputs; no test prediction is used to choose anything."""
import sys, os, hashlib
N4DIR = '/mnt/project-files/work/hardness-cache/scripts/d1011/N4'
if N4DIR not in sys.path:
    sys.path.insert(0, N4DIR)
from n4lib import *  # noqa  (np, pd, REPO, CACHE, SUB, STRATA, RHO, load29, vs29, public_expect, write, strat_split, ...)

HERE = N4DIR + '/g1'
HETD = REPO + '/data/het_cnn/'
EST_DIRS = dict(LOC=['ev2s_loc_e32r4', 'ev2s_loc_e32r4_s1'], A2=['ev2s_loc_fsim'], A3=['ev2s_locK_real5'],
                A23=['ev2s_loc_fsim', 'ev2s_locK_real5'])
HSPEC = {'LOC': (['LOC'], False), 'replace_A2': (['A2'], False), 'replace_A3': (['A3'], False),
         'replace_A23': (['A23'], False), 'avg_A23': (['LOC', 'A23'], True), 'twobeta_A23': (['LOC', 'A23'], False)}
STEPS = (0.5, 0.75, 1.0)
OPTS = {f'{h}/{s}': (h, s) for h in HSPEC for s in STEPS}          # declared order: H outer, S inner (18)
OPTS_STRETCH = {k: v for k, v in OPTS.items() if v[0] == 'LOC'}    # secondary procedure (3)
NOCHANGE = 'LOC/0.5'
OPTIONS_SHA256 = 'c4bd0675df9e0569523109e6c7cc05b571b0a468590d9a5172bb9d67ae31bb84'
assert len(OPTS) == 18 and list(OPTS)[0] == NOCHANGE


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def avg_dirs(dirs, split, ids):
    Hs = [pd.read_parquet(f'{HETD}{d}/het_cnn_{split}.parquet').set_index('ID').loc[ids] for d in dirs]
    return np.mean([h.het_cnn.values for h in Hs], 0), np.mean([h.logN_cnn.values for h in Hs], 0)


class Ctx:
    def __init__(self, ests=('LOC', 'A2', 'A3', 'A23')):
        assert hashlib.sha256(open(HERE + '/options.txt', 'rb').read()).hexdigest() == OPTIONS_SHA256, 'options.txt changed'
        D = load29(); self.D = D
        self.ids, self.y, self.fold, self.nz, self.rs = D['ids'], D['y'], D['fold'], D['nz'], D['rs']
        self.G1 = (self.nz >= 9.5) & (self.rs < 0.5); self.G = D['grp']['G1hi']
        assert self.G1.sum() == 208 and np.array_equal(self.G1, D['grp']['midLF'] | D['grp']['G1hi'])
        rd = lambda n: pd.read_csv(f'{SUB}/{n}', float_precision='round_trip').set_index('ID').hardness.loc[self.ids].to_numpy().astype(float)
        self.o21 = rd('blend_v21_oof.csv'); self.p27 = D['p27u']; self.p28n = D['p28n']; self.p29 = D['p29']
        self.r = self.y - self.o21
        self.EST = {n: avg_dirs(EST_DIRS[n], 'train', self.ids) for n in ests}
        for n in ests:
            assert np.isfinite(self.EST[n][0][self.G1]).all() and np.isfinite(self.EST[n][1][self.G1]).all(), n
        # T-CP train rows (v29g mixed them 0.5 v28n + 0.5 f7): impl_C4/C3_oof_rows.csv, cross-checked with the m3b gate
        cp = pd.read_csv(CACHE + '/scripts/d1011/bold/impl_C4/C3_oof_rows.csv', float_precision='round_trip')
        self.tcp = np.isin(self.ids, cp.ID.astype(str).to_numpy())
        mb = pd.read_parquet(CACHE + '/scripts/d1011/midlf/m3b_rows.parquet', columns=['split', 'group', 'gate_cp'])
        mtr = mb[mb.split == 'train'].loc[self.ids]
        assert np.array_equal(self.tcp, (mtr.group.to_numpy() == 'midLF') & mtr.gate_cp.to_numpy().astype(bool))
        assert self.tcp.sum() == 7 and (self.tcp <= D['grp']['midLF']).all()
        self.w = np.where(self.tcp, 0.5, 1.0) * self.G1
        # v29g == v28n on the non-T-CP G1 rows (train OOF)
        assert np.array_equal(self.p29[self.G1 & ~self.tcp], self.p28n[self.G1 & ~self.tcp])
        self.ALL = np.ones(500, bool)
        self._cache = {}

    # ------------------------------------------------------------------ recipe (train OOF under fold vector ff, fits on fm)
    def offsets(self, names, ff, fm, avg=False):
        EST, G1, r = self.EST, self.G1, self.r
        off = np.zeros(500)
        for k in range(5):
            fit, app = G1 & fm & (ff != k), G1 & (ff == k)
            if avg:
                hh = np.mean([EST[n][0] for n in names], 0); ll = EST['LOC'][1]
                Hm = (hh - np.polyval(np.polyfit(ll[fit], hh[fit], 1), ll))[:, None]
            else:
                Hm = np.column_stack([EST[n][0] - np.polyval(np.polyfit(EST[n][1][fit], EST[n][0][fit], 1), EST[n][1]) for n in names])
            b = np.linalg.lstsq(Hm[fit], r[fit] - r[fit].mean(), rcond=None)[0]; off[app] = Hm[app] @ b
        return off

    def base(self, h, ff, fm):
        key = (h, ff.tobytes(), fm.tobytes())
        if key not in self._cache:
            names, avg = HSPEC[h]
            b = self.p27.copy(); b[self.G1] = (self.o21 + self.offsets(names, ff, fm, avg))[self.G1]
            if len(self._cache) > 4000:
                self._cache.clear()
            self._cache[key] = b
        return self._cache[key]

    def stretch(self, base, ff, fm, s):
        G, nz, y = self.G, self.nz, self.y
        q = base.copy()
        for k in range(5):
            g = G & fm & (ff != k); m = float(np.median(nz[g])); lo = g & (nz < m)
            c = float(base[lo].mean()); b = float(np.polyfit(base[lo], y[lo], 1)[0]); kk = 1 + s * (b / RHO - 1)
            a = G & (ff == k) & (nz < m); q[a] = c + kk * (base[a] - c)
        return q

    def build(self, opt, ff, fm=None):
        h, s = opt
        fm = self.ALL if fm is None else fm
        return self.stretch(self.base(h, ff, fm), ff, fm, s)

    def mapped(self, q):
        """v29g form: v29g + w (newG1 - v28n), w = 0.5 on T-CP rows, 1 on other G1 rows, 0 elsewhere."""
        d = q - self.p28n; d[np.abs(d) < 1e-9] = 0.0           # float noise of the rebuild -> rows stay bit-identical
        return self.p29 + self.w * d

    # ------------------------------------------------------------------ selection-honest nested
    def nested(self, ff, opts):
        y = self.y; q = np.zeros(500); picks = []; inner = []
        for k in range(5):
            T = ff != k
            sc = {nm: float(np.mean((self.mapped(self.build(o, ff, T))[T] - y[T]) ** 2)) for nm, o in opts.items()}
            best = min(sc, key=sc.get)          # ties -> earlier in the declared order
            picks.append(best); inner.append(sc)
            q[ff == k] = self.build(opts[best], ff)[ff == k]
        return q, picks, inner

    def full_pick(self, opts):
        y = self.y
        sc = {nm: float(np.mean((self.mapped(self.build(o, self.fold)) - y) ** 2)) for nm, o in opts.items()}
        return min(sc, key=sc.get), sc


# ---------------------------------------------------------------------- test-row recipe (full-data fits)
def test_new_g1(ctx, opt, te_ids, nzt, rst, t21, t27):
    """newG1 test vector in v28n form for option opt: G1 test rows = v21 + full-fit offset, then the stretch with c, b
    from the option's nested base OOF on the train low-half rows (m from all train G1hi rows); other rows = v27u.
    Raises SystemExit with the ingest commands if a needed het test parquet is missing."""
    h, s = opt
    names, avg = HSPEC[h]
    dirs = sorted({d for n in names + (['LOC'] if avg else []) for d in EST_DIRS[n]})
    miss = [d for d in dirs if not os.path.exists(f'{HETD}{d}/het_cnn_test.parquet')]
    if miss:
        cmds = '\n'.join(f'  python -m src.het_cnn --ingest {d} {CACHE}/laptop-gpu-2026-10-11/{d}_c0.txt <{d}_c1.txt> <{d}_c2.txt>' for d in miss)
        raise SystemExit(f'\n*** PENDING_TEST_INPUT: option {h}/{s} needs het_cnn_test.parquet for {miss} (not ingested).\n'
                         f'Run from the repo root once the laptop chunks (section 15) arrive:\n{cmds}\nthen re-run this builder.\n')
    G1, G, y, nz, r = ctx.G1, ctx.G, ctx.y, ctx.nz, ctx.r
    G1t = (nzt >= 9.5) & (rst < 0.5); Gt = (nzt >= 12) & (rst < 0.5)
    ESTt = {n: avg_dirs(EST_DIRS[n], 'test', te_ids) for n in set(names) | ({'LOC'} if avg else set())}
    EST = ctx.EST
    fit = G1
    if avg:
        hh = np.mean([EST[n][0] for n in names], 0); ll = EST['LOC'][1]
        hht = np.mean([ESTt[n][0] for n in names], 0); llt = ESTt['LOC'][1]
        g = np.polyfit(ll[fit], hh[fit], 1)
        Hm = (hh - np.polyval(g, ll))[:, None]; Hmt = (hht - np.polyval(g, llt))[:, None]; gs = [g.tolist()]
    else:
        cols, colt, gs = [], [], []
        for n in names:
            g = np.polyfit(EST[n][1][fit], EST[n][0][fit], 1); gs.append(g.tolist())
            cols.append(EST[n][0] - np.polyval(g, EST[n][1])); colt.append(ESTt[n][0] - np.polyval(g, ESTt[n][1]))
        Hm, Hmt = np.column_stack(cols), np.column_stack(colt)
    assert np.isfinite(Hmt[G1t]).all(), 'non-finite het on G1 test rows'
    beta = np.linalg.lstsq(Hm[fit], r[fit] - r[fit].mean(), rcond=None)[0]
    base_t = t27.copy(); base_t[G1t] = t21[G1t] + (Hmt @ beta)[G1t]
    bo = ctx.base(h, ctx.fold, ctx.ALL)                      # nested base OOF (v28n's c, b come from the v27u OOF)
    m = float(np.median(nz[G])); lo = G & (nz < m)
    c = float(bo[lo].mean()); b = float(np.polyfit(bo[lo], y[lo], 1)[0]); k = 1 + s * (b / RHO - 1)
    fte = Gt & (nzt < m)
    new_t = base_t.copy(); new_t[fte] = c + k * (base_t[fte] - c)
    off_tr, off_te = (Hm @ beta)[G1], (Hmt @ beta)[G1t]       # report only (label-free sanity of the ingested test het)
    info = dict(option=f'{h}/{s}', het_dirs=dirs, g=gs, beta=[float(x) for x in beta], m=m, c=c, b=b, k=k,
                report_only_offset_sd=dict(train_G1=float(off_tr.std()), test_G1=float(off_te.std()),
                                           train_max_abs=float(np.abs(off_tr).max()), test_max_abs=float(np.abs(off_te).max())),
                n_fit_G1=int(fit.sum()), n_fit_lo=int(lo.sum()), n_test_G1=int(G1t.sum()), n_test_stretch=int(fte.sum()))
    return new_t, info
