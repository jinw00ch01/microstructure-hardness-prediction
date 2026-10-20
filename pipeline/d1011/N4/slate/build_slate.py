"""N4 slate for 2026-10-11 slots 2-5 (v29g took slot 1). Builds the union files of the screened N4 pieces, scores every
file with n4lib (vs29 / public_expect), simulates the slate's branch rule, and copies READY files to
submissions/blend_v30<letter>.csv (+ .json). Never overwrites an existing submission file (a PENDING placeholder .json
written by this script is the only file it replaces, and only with its own csv).

Pieces (each a delta on blend_v29g; nested OOF = the piece's selection-honest nested OOF in v29g form):
  G  N4/g1             honest -2.396 (4/5)  SAFE   G1 rows (midLF + G1hi), het A3 + stretch 0.75. Test PENDING (A3 test het).
  S  N4/g1/g1_stretch  honest -0.360 (3/5)  SAFE   G1hi low-half rows, LOC het + stretch 0.75 (S is G's LOC-only sub-procedure).
  C  N4/clean/posthoc_clean_only  honest -0.507 (4/5) but POST HOC -> BOLD   clean rows, w 0.75 formula weight.
  (cnn FAIL and clean-joint (+LV u) are not used: FAIL / dominated, LV knob closed.)
G and S change the same rows (alternatives, never summed). G or S and C change disjoint rows, so a union's nested OOF
and test are exactly v29g + the two deltas, and the union's public MSE change is exactly the sum of the pieces' changes.

usage (repo root /home/claude/microstructure-hardness-prediction):
  nice -n 10 timeout 1800 python <N4>/slate/build_slate.py              # ready files (c, d, e) + placeholders for a, b
  nice -n 10 timeout 1800 python <N4>/slate/build_slate.py --with-g1    # after N4/g1/test.csv exists: writes a (G+C), b (G)
  python <N4>/slate/build_slate.py --next v30a=10.0412 v30b=10.0450     # branch rule: which file goes next (no build)
"""
import sys, os, json, hashlib, argparse, shutil, math
N4DIR = '/mnt/project-files/work/hardness-cache/scripts/d1011/N4'
if N4DIR not in sys.path:
    sys.path.insert(0, N4DIR)
from n4lib import *  # noqa  (np, pd, SUB, STRATA, load29, vs29, public_expect, write, ...)

HERE = N4DIR + '/slate'
BASE_MSE = 10.0564339191 ** 2           # v29g public MSE, as in n4lib.public_expect
R29_SHOWN = 10.0564
TOL = 0.002                             # MSE: 4-decimal rounding of two readings (+-0.001 each)
L10, L9 = 100.0 - BASE_MSE, 81.0 - BASE_MSE   # public dMSE needed for <= 10.0 and <= 9.0
Phi = lambda z: 0.5 * math.erfc(-z / math.sqrt(2))
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()

PIECES = {
    'G': dict(oof=N4DIR + '/g1/g1_honest_oof.csv', test=N4DIR + '/g1/test.csv', res=N4DIR + '/g1/result.json',
              nested=N4DIR + '/g1/nested_result.json', posthoc=False, resplit=-0.5191668917256684,
              insample=dict(option='replace_A3/0.75', all500_dmse=-2.6676387939789046, fold_dsse=[-150.6, -519.1, -191.9, -851.2, 379.0]),
              desc='g1: G1 rows (raw ic_noise >= 9.5, ridge < 0.5): v21 + A3 (ev2s_locK_real5) het offset, then G1hi low-half '
                   'stretch step 0.75; mapped to v29g (w 0.5 on T-CP rows)', builder=N4DIR + '/g1/build_g1_test.py'),
    'S': dict(oof=N4DIR + '/g1/g1_stretch/oof.csv', test=N4DIR + '/g1/g1_stretch/test.csv', md5='63dbb249c3b08577772603aa2e841f36',
              res=N4DIR + '/g1/g1_stretch/result.json', posthoc=False, resplit=0.04037114164450486,
              insample=dict(option='LOC/0.75', all500_dmse=-0.3595351952683885, fold_dsse=[-256.4, -239.9, 226.8, 92.1, -2.4]),
              desc='g1_stretch: G1hi low-half stretch step 0.75 (k 1.3926) with the LOC het', builder=N4DIR + '/g1/build_g1_test.py'),
    'C': dict(oof=N4DIR + '/clean/posthoc_clean_only/oof.csv', test=N4DIR + '/clean/posthoc_clean_only/test.csv',
              md5='4877dedcf68466d83a774be78fe7cec9', res=N4DIR + '/clean/posthoc_clean_only/result.json', posthoc=True,
              resplit=0.21127355558729008,
              insample=dict(option='w0.75_s1.00_u0.50', all500_dmse=-0.5070286268341758, fold_dsse=[-24.8, -129.8, -98.0, -141.9, 141.0]),
              desc='clean_only (POST HOC): clean rows 0.25*(v21base+off_V4) + 0.75*f8_C2 (v29g 0.5/0.5); LV rows = v29g',
              builder=N4DIR + '/clean/build_clean.py'),
}
# letter, name, pieces  (letters in the PRIMARY slot order; e is the fallback-only file)
FILES = [('a', 'GC', ['G', 'C']), ('b', 'G', ['G']), ('c', 'C', ['C']), ('d', 'S', ['S']), ('e', 'SC', ['S', 'C'])]
MECH = {'G': 'a better het reader (A3, trained on a realistic render prior) on the G1 rows plus a larger low-half stretch',
        'S': 'a larger G1hi low-half stretch (v28n direction, k 1.26 -> 1.39)',
        'C': 'more weight on the f8_C2 formula on clean rows (same direction as the v25 public-optimal 0.66)'}


def rd(p, ids):
    o = pd.read_csv(p, float_precision='round_trip')
    assert (o.ID.astype(str).to_numpy() == ids).all(), p
    return o.hardness.to_numpy().astype(float)


def g1_ready():
    if not (os.path.exists(PIECES['G']['test']) and os.path.exists(PIECES['G']['res'])):
        return False
    r = json.load(open(PIECES['G']['res']))
    return r.get('procedure') == 'g1' and not os.path.exists(N4DIR + '/g1/g1_PENDING.json')


def load_pieces(D, with_g1):
    P = {}
    for k, s in PIECES.items():
        q = rd(s['oof'], D['ids'])
        t = None
        if k == 'G':
            NR = json.load(open(s['nested']))
            assert NR['g1']['full_pick'] == 'replace_A3/0.75'
            if with_g1:
                assert g1_ready(), 'g1 test not built yet: run the ingest and N4/g1/build_g1_test.py first'
                r = json.load(open(s['res']))
                assert r['full_pick'] == 'replace_A3/0.75' and md5(s['test']) == r['test_md5']
                q2 = rd(N4DIR + '/g1/oof.csv', D['ids'])
                assert float(np.max(np.abs(q2 - q))) < 1e-9, 'g1/oof.csv differs from the honest OOF used by the slate'
                t = rd(s['test'], D['te_ids'])
                s['md5'] = r['test_md5']
        else:
            assert md5(s['test']) == s['md5'] == json.load(open(s['res']))['test_md5'], k
            t = rd(s['test'], D['te_ids'])
        # the piece's OOF must reproduce its own reported honest number
        r = json.load(open(s['res'])) if os.path.exists(s['res']) and k != 'G' else None
        v = vs29(q, D, nflip=2000)
        if k == 'G':
            assert abs(v['all500_dmse'] - (-2.396266157028484)) < 1e-9 and v['folds_better'] == 4
        else:
            assert abs(v['all500_dmse'] - r['vs_v29g']['all500_dmse']) < 1e-9, k
        P[k] = dict(q=q, t=t, dq=q - D['p29'], dt=None if t is None else t - D['t29'])
    # supports: G1 rows for G/S, clean rows for C; G and S overlap (alternatives), G/S vs C disjoint
    G1 = D['grp']['midLF'] | D['grp']['G1hi']; G1t = D['grpt']['midLF'] | D['grpt']['G1hi']
    assert ((P['G']['dq'] != 0) <= G1).all() and ((P['S']['dq'] != 0) <= D['grp']['G1hi']).all()
    assert ((P['C']['dq'] != 0) <= D['grp']['clean']).all() and ((P['C']['dt'] != 0) <= D['grpt']['clean']).all()
    assert ((P['S']['dt'] != 0) <= D['grpt']['G1hi']).all()
    if P['G']['dt'] is not None:
        assert ((P['G']['dt'] != 0) <= G1t).all()
    return P


def proxy_test_delta(dq, D):
    """PENDING test: changed test rows = every test row of a group whose train rows change (G changes all 208 G1 train
    rows, so all 433 G1 test rows); magnitude = the train change rms of that group. Only the CV-hypothesis count and the
    null's size use it; it is a stand-in, never a prediction."""
    dt = np.zeros(1000)
    for g in STRATA:
        m = D['grp'][g] & (dq != 0)
        if m.any():
            dt[D['grpt'][g]] = float(np.sqrt(np.mean(dq[m] ** 2)))
    return dt


def hyp(E, sd):
    out = {}
    for nm, f in (('full', 1.0), ('half', 0.5)):
        e = f * E
        r = math.sqrt(BASE_MSE + e)
        out[nm] = dict(public_dmse=e, sd_mse=sd, public_rmse=r, sd_rmse=sd / (2 * r),
                       P_le_10=Phi((L10 - e) / sd) if sd > 0 else float(e <= L10),
                       P_le_9=Phi((L9 - e) / sd) if sd > 0 else float(e <= L9),
                       P_better_v29g=Phi((0 - e) / sd) if sd > 0 else float(e < 0))
    return out


def build_file(name, pieces, P, D):
    q = D['p29'] + sum(P[k]['dq'] for k in pieces)
    ready = all(P[k]['dt'] is not None for k in pieces)
    t = D['t29'] + sum(P[k]['dt'] for k in pieces) if ready else None
    v = vs29(q, D)
    if ready:
        pub = public_expect(q, t, D); pend = False
    else:
        dproxy = sum((P[k]['dt'] if P[k]['dt'] is not None else proxy_test_delta(P[k]['dq'], D)) for k in pieces)
        pub = public_expect(q, D['t29'] + dproxy, D); pend = True
    E, sd = pub['cv_hyp_public_dmse'], pub['cv_hyp_sd']
    posthoc = any(PIECES[k]['posthoc'] for k in pieces)
    if v['all500_dmse'] < 0 and v['folds_better'] >= 3 and not posthoc:
        tier = 'SAFE'
    elif v['all500_dmse'] <= 0.5:
        tier = 'BOLD' + (' (post hoc piece C)' if posthoc else '')
    else:
        tier = 'FAIL'
    ins = dict(all500_dmse=sum(PIECES[k]['insample']['all500_dmse'] for k in pieces),
               fold_dsse=[round(sum(PIECES[k]['insample']['fold_dsse'][f] for k in pieces), 1) for f in range(5)],
               options={k: PIECES[k]['insample']['option'] for k in pieces})
    ins['folds_better'] = int(sum(x < 0 for x in ins['fold_dsse']))
    out = dict(name=name, pieces=pieces, ready=ready, tier=tier,
               honest=dict(all500_dmse=v['all500_dmse'], folds_better=v['folds_better'], fold_dsse=v['fold_dsse'],
                           signflip_p=v['signflip_p'], droptop3=v['droptop3_all500_dmse'], rmse_new=v['rmse_new'],
                           n_train_changed=v['n_changed']),
               insample_fixed_pick=ins,
               resplit_mean_sum=sum(PIECES[k]['resplit'] for k in pieces),
               public=dict(cv_hyp_dmse=E, cv_hyp_sd=sd, null_dmse=pub['null_public_dmse'], null_sd=pub['null_sd'],
                           pending_proxy=pend, groups=pub['groups']),
               hyp=hyp(E, sd),
               keep_line=f'public <= {R29_SHOWN - 0.0001:.4f} (better than v29g 10.0564)',
               close_line_rmse=math.sqrt(BASE_MSE + pub['null_public_dmse']))
    if ready:
        dt = t - D['t29']; ch = dt != 0
        out['test_change'] = dict(n_rows=int(ch.sum()), rms_changed=float(np.sqrt(np.mean(dt[ch] ** 2))),
                                  rms_all=float(np.sqrt(np.mean(dt ** 2))), max_abs=float(np.abs(dt).max()))
    else:
        out['test_change'] = dict(n_rows=None, note='PENDING (A3 test het not ingested)')
    return out, q, t


# --------------------------------------------------------------------------------------------- branch rule
def next_slot(readings, g1_is_ready):
    """readings: dict letter -> public RMSE (4 decimals) of files already submitted on 10-11 (after v29g).
    Returns (letter or None, reason). Pieces: G/S on G1 rows, C on clean rows; d = public MSE change vs v29g."""
    M = {k: v * v for k, v in readings.items()}
    d = lambda k: M[k] - BASE_MSE
    done = list(readings)
    conf = lambda x: x < -TOL                       # direction confirmed (beyond 4-decimal rounding)
    if not done:
        return ('a', 'PRIMARY slot 2: G+C (highest expected)') if g1_is_ready else ('e', 'FALLBACK slot 2: S+C (g1 test not built)')
    if done[0] == 'a':                                                   # ---------------- PRIMARY
        if len(done) == 1:
            return 'b', 'slot 3: G alone (SAFE) -> splits G+C into dG and dC'
        dG, dC = d('b'), d('a') - d('b')
        if len(done) == 2:
            if conf(dC) and not conf(dG):
                return 'c', f'slot 4: dC {dC:+.3f} confirmed, dG {dG:+.3f} not -> C alone (BOLD 2; public exactly v29g + dC)'
            if not conf(dG):
                return 'd', f'slot 4: dG {dG:+.3f} not confirmed, dC {dC:+.3f} -> S (SAFE, the other G1 reader)'
            return None, f'slot 4 empty: dG {dG:+.3f} confirmed (best = {"a" if conf(dC) else "b"}); nothing left that can beat it without re-tuning'
        if len(done) == 3 and done[2] == 'c':
            return 'd', 'slot 5: S (SAFE, the other G1 reader); if it confirms, S+C (v30e) goes first on 10-12 (BOLD budget of 10-11 used)'
        return None, 'slot 5 empty'
    if done[0] == 'e':                                                   # ---------------- FALLBACK
        if len(done) == 1:
            return 'd', 'slot 3: S alone (SAFE) -> splits S+C into dS and dC'
        dS, dC = d('d'), d('e') - d('d')
        if len(done) == 2:
            if g1_is_ready:
                return 'b', f'slot 4: g1 test now built -> G alone (SAFE); dS {dS:+.3f}, dC {dC:+.3f}'
            if conf(dC) and not conf(dS):
                return 'c', f'slot 4: dC {dC:+.3f} confirmed, dS {dS:+.3f} not -> C alone (BOLD 2)'
            return None, f'slot 4 empty (g1 not built; dS {dS:+.3f}, dC {dC:+.3f})'
        if len(done) == 3 and done[2] == 'b':
            dG = d('b')
            if conf(dC) and conf(dG) and dG < dS - TOL:
                return 'a', f'slot 5: dC {dC:+.3f} and dG {dG:+.3f} confirmed, G beats S -> G+C (BOLD 2; public exactly v29g + dG + dC)'
            if conf(dC) and not conf(dS) and not conf(dG):
                return 'c', f'slot 5: only C confirmed (dC {dC:+.3f}) -> C alone (BOLD 2)'
            return None, f'slot 5 empty (dS {dS:+.3f}, dC {dC:+.3f}, dG {dG:+.3f})'
        return None, 'slot 5 empty'
    return None, 'unknown order'


def simulate(pieces_E, pieces_sd, rhoGS, nsim=40000, seed=20261011):
    """Monte Carlo of the slate under one hypothesis: piece public dMSE (G, S, C) jointly normal; C independent of G/S
    (disjoint rows). Applies next_slot mechanically; best = min(v29g, submitted files)."""
    rng = np.random.default_rng(seed)
    mu = np.array([pieces_E['G'], pieces_E['S'], pieces_E['C']])
    sG, sS, sC = pieces_sd['G'], pieces_sd['S'], pieces_sd['C']
    cov = np.array([[sG * sG, rhoGS * sG * sS, 0], [rhoGS * sG * sS, sS * sS, 0], [0, 0, sC * sC]])
    X = rng.multivariate_normal(mu, cov, size=nsim)
    out = {}
    for branch, ready in (('primary', True), ('fallback_no_g1', False)):
        best = np.zeros(nsim); nfiles = np.zeros(nsim); nbold = np.zeros(nsim); final = {}
        for i in range(nsim):
            dG, dS, dC = X[i]
            val = {'a': dG + dC, 'b': dG, 'c': dC, 'd': dS, 'e': dS + dC}
            rd_ = {}
            for _ in range(4):
                nx, _r = next_slot(rd_, ready)
                if nx is None:
                    break
                rd_[nx] = math.sqrt(max(BASE_MSE + val[nx], 1.0))
            vals = [val[k] for k in rd_]
            b = min([0.0] + vals); best[i] = b
            fk = 'v29g' if b == 0.0 else min(rd_, key=lambda k: val[k])
            final[fk] = final.get(fk, 0) + 1
            nfiles[i] = len(rd_); nbold[i] = sum(k in 'ace' for k in rd_)
        n = len(best)
        out[branch] = dict(n=n, P_best_le_10=float(np.mean(best <= L10)), P_best_le_9=float(np.mean(best <= L9)),
                           P_best_better_v29g=float(np.mean(best < 0)), E_best_rmse=float(np.mean(np.sqrt(BASE_MSE + best))),
                           mean_files=float(nfiles.mean()), max_bold=int(nbold.max()),
                           final_freq={k: round(v / n, 3) for k, v in sorted(final.items())})
    return out


def rho_GS(P, D):
    """Correlation of the public dMSE noise of G and S: per-group covariance of the train per-row dSE (all rows of the
    group, zeros included), scaled by the test row counts of the group."""
    y, p = D['y'], D['p29']
    dse = {k: (P[k]['q'] - y) ** 2 - (p - y) ** 2 for k in ('G', 'S')}
    c = vg = vs = 0.0
    for g in ('midLF', 'G1hi'):
        m = D['grp'][g]; nt = int(D['grpt'][g].sum())
        C_ = np.cov(dse['G'][m], dse['S'][m])
        c += nt * C_[0, 1]; vg += nt * C_[0, 0]; vs += nt * C_[1, 1]
    return float(c / math.sqrt(vg * vs)), float(np.corrcoef(P['G']['dq'][P['G']['dq'] != 0], P['S']['dq'][P['G']['dq'] != 0])[0, 1])


def copy_out(letter, name, info, t, meta_extra):
    """submissions/blend_v30<letter>.csv + .json; never overwrite, except our own PENDING placeholder json."""
    csv_p, js_p = f'{SUB}/blend_v30{letter}.csv', f'{SUB}/blend_v30{letter}.json'
    if os.path.exists(csv_p):
        if md5(csv_p) == info.get('test_md5'):
            print(f'  blend_v30{letter}.csv already present with the same md5 - kept'); return
        raise SystemExit(f'refuse: {csv_p} exists with a different md5')
    if os.path.exists(js_p):
        old = json.load(open(js_p))
        if not (old.get('status') == 'PENDING_TEST_INPUT' and old.get('written_by') == HERE + '/build_slate.py'):
            raise SystemExit(f'refuse: {js_p} exists and is not this script\'s PENDING placeholder')
    src = f'{HERE}/{name}/test.csv'
    shutil.copyfile(src, csv_p)
    assert md5(csv_p) == info['test_md5']
    meta = dict(name=f'blend_v30{letter}', status='READY', written_by=HERE + '/build_slate.py', md5=info['test_md5'],
                source=src, base='blend_v29g', base_md5=V29G_MD5, **meta_extra)
    json.dump(meta, open(js_p, 'w'), indent=1)
    print(f'  wrote {csv_p} (md5 {info["test_md5"]}) + .json')


def placeholder(letter, name, info, meta_extra):
    js_p, csv_p = f'{SUB}/blend_v30{letter}.json', f'{SUB}/blend_v30{letter}.csv'
    if os.path.exists(csv_p):
        print(f'  blend_v30{letter}.csv exists - placeholder not written'); return
    if os.path.exists(js_p):
        old = json.load(open(js_p))
        if not (old.get('status') == 'PENDING_TEST_INPUT' and old.get('written_by') == HERE + '/build_slate.py'):
            raise SystemExit(f'refuse: {js_p} exists and is not this script\'s placeholder')
    meta = dict(name=f'blend_v30{letter}', status='PENDING_TEST_INPUT', written_by=HERE + '/build_slate.py',
                finish_with=f'nice -n 10 timeout 1800 python {HERE}/build_slate.py --with-g1 (after the A3 ingest and '
                            f'{N4DIR}/g1/build_g1_test.py)', base='blend_v29g', base_md5=V29G_MD5, **meta_extra)
    json.dump(meta, open(js_p, 'w'), indent=1)
    print(f'  wrote placeholder {js_p} (no csv until --with-g1)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--with-g1', action='store_true')
    ap.add_argument('--next', nargs='*', default=None, help='readings like v30a=10.0412 (submission order)')
    ap.add_argument('--no-copy', action='store_true')
    a = ap.parse_args()
    if a.next is not None:
        rdg = {}
        for s in a.next:
            k, v = s.split('=')
            rdg[k.replace('blend_', '').replace('v30', '')] = float(v)
        print(next_slot(rdg, g1_ready()))
        return
    D = load29()
    P = load_pieces(D, a.with_g1)
    res = {}; built = {}
    for letter, name, pcs in FILES:
        info, q, t = build_file(name, pcs, P, D)
        info['letter'] = letter
        if t is not None:
            if len(pcs) > 1:
                r = write(f'slate/{name}', q, t, D, dict(name=name, pieces=pcs, rule='v29g + sum of piece deltas (disjoint rows)'))
                info['test_md5'] = r['test_md5']
                assert abs(r['vs_v29g']['all500_dmse'] - info['honest']['all500_dmse']) < 1e-12
            else:
                os.makedirs(f'{HERE}/{name}', exist_ok=True)
                shutil.copyfile(PIECES[pcs[0]]['test'], f'{HERE}/{name}/test.csv')
                shutil.copyfile(PIECES[pcs[0]]['oof'], f'{HERE}/{name}/oof.csv')
                info['test_md5'] = md5(f'{HERE}/{name}/test.csv'); assert info['test_md5'] == PIECES[pcs[0]]['md5']
            built[letter] = (name, info, t)
        res[letter] = info
        h = info['hyp']
        print(f"[{letter}] {name:3s} {info['tier']:28s} honest {info['honest']['all500_dmse']:+.3f} {info['honest']['folds_better']}/5 "
              f"{info['honest']['fold_dsse']} sf {info['honest']['signflip_p']:.3f} dt3 {info['honest']['droptop3']:+.3f} | "
              f"pub full {h['full']['public_rmse']:.4f} half {h['half']['public_rmse']:.4f} sd {h['half']['sd_rmse']:.4f} "
              f"P10 {h['half']['P_le_10']:.3f} P9 {h['half']['P_le_9']:.1e} | null {info['public']['null_dmse']:+.3f} "
              f"close>= {info['close_line_rmse']:.4f} | resplit-sum {info['resplit_mean_sum']:+.3f} | md5 {info.get('test_md5', 'PENDING')}"
              f"{'' if info['ready'] else '  (public: proxy test rows)'}")
    # test-change overlap / correlation (ready files) and the train OOF delta correlation G vs S
    rGS, rGSq = rho_GS(P, D)
    corr = {}
    for x, y_ in (('d', 'c'), ('d', 'e'), ('c', 'e')):
        if x in built and y_ in built:
            dx, dy = built[x][2] - D['t29'], built[y_][2] - D['t29']
            m = (dx != 0) | (dy != 0)
            corr[f'{x}-{y_}'] = dict(overlap_rows=int(((dx != 0) & (dy != 0)).sum()), corr_on_union=float(np.corrcoef(dx[m], dy[m])[0, 1]))
    if 'b' in built and 'd' in built:
        dx, dy = built['b'][2] - D['t29'], built['d'][2] - D['t29']; m = (dx != 0) | (dy != 0)
        corr['b-d'] = dict(overlap_rows=int(((dx != 0) & (dy != 0)).sum()), corr_on_union=float(np.corrcoef(dx[m], dy[m])[0, 1]))
    print('test-change overlap/corr:', json.dumps(corr), '| public-noise corr(G,S) (train dSE model)', round(rGS, 3),
          '| train OOF delta corr(G,S) on G1 rows', round(rGSq, 3))
    # Monte Carlo of the branch rule
    E = {k: res[l]['public']['cv_hyp_dmse'] for k, l in (('G', 'b'), ('S', 'd'), ('C', 'c'))}
    S = {k: res[l]['public']['cv_hyp_sd'] for k, l in (('G', 'b'), ('S', 'd'), ('C', 'c'))}
    sim = {nm: simulate({k: f * v for k, v in E.items()}, S, rGS) for nm, f in (('full', 1.0), ('half', 0.5))}
    rs = {'G': PIECES['G']['resplit'], 'S': PIECES['S']['resplit'], 'C': PIECES['C']['resplit']}
    # re-split hypothesis: public dMSE ~ re-split mean x (test rows / train rows ratio of the pieces' CV-hyp numbers)
    scale = {k: E[k] / v for k, v in (('G', -2.396266157028484), ('S', -0.3595351952683885), ('C', -0.5070286268341758))}
    sim['resplit'] = simulate({k: rs[k] * scale[k] for k in E}, S, rGS)
    # null (no signal): mean = sum(delta^2) share, sd = null sd (n4lib; G uses the proxy test rows while pending)
    sim['null'] = simulate({k: res[l]['public']['null_dmse'] for k, l in (('G', 'b'), ('S', 'd'), ('C', 'c'))},
                           {k: res[l]['public']['null_sd'] for k, l in (('G', 'b'), ('S', 'd'), ('C', 'c'))}, rGS)
    for nm, s in sim.items():
        print(f'slate MC [{nm}]', json.dumps(s))
    out = dict(files=res, corr=corr, rho_GS_public=rGS, rho_GS_train_delta=rGSq, slate_mc=sim,
               piece_public=dict(E_full=E, sd=S, resplit_scaled={k: rs[k] * scale[k] for k in E}), with_g1=a.with_g1,
               base_mse=BASE_MSE, L10=L10, L9=L9)
    json.dump(out, open(f'{HERE}/slate_result{"_with_g1" if a.with_g1 else ""}.json', 'w'), indent=1)
    if a.no_copy:
        return
    print('copy to submissions:')
    for letter, name, pcs in FILES:
        info = res[letter]
        meta = dict(pieces=pcs, piece_desc={k: PIECES[k]['desc'] for k in pcs}, mechanism={k: MECH[k] for k in pcs},
                    piece_builders={k: PIECES[k]['builder'] for k in pcs}, slate_builder=HERE + '/build_slate.py',
                    plan=HERE + '/PLAN_N4.txt', tier_N4=info['tier'],
                    rule='blend_v29g + ' + ' + '.join(f'delta({k})' for k in pcs) + ' (pieces on disjoint rows)',
                    honest_nested=info['honest'], insample_fixed_pick=info['insample_fixed_pick'],
                    resplit_mean_sum=info['resplit_mean_sum'], public_expectation=info['public'], hypotheses=info['hyp'],
                    reading=dict(keep=info['keep_line'], close_if_rmse_ge=round(info['close_line_rmse'], 4)),
                    test_change=info['test_change'], public_score=None)
        if info['ready']:
            copy_out(letter, name, info, built[letter][2], meta)
        else:
            placeholder(letter, name, info, meta)
    if a.with_g1:
        with open(HERE + '/PLAN_N4.txt', 'a') as fh:
            fh.write('\nADDENDUM (auto, build_slate.py --with-g1): g1 test built; real public numbers replace the proxy ones\n')
            for letter in ('a', 'b'):
                i = res[letter]; h = i['hyp']
                fh.write(f"  v30{letter} {i['name']} md5 {i['test_md5']} test rows {i['test_change']['n_rows']} | full {h['full']['public_rmse']:.4f} "
                         f"half {h['half']['public_rmse']:.4f} (sd {h['half']['sd_rmse']:.4f}) P10 {h['half']['P_le_10']:.2f} | "
                         f"close if public >= {i['close_line_rmse']:.4f}\n")
            fh.write(f"  slate MC: {HERE}/slate_result_with_g1.json\n")


if __name__ == '__main__':
    main()
