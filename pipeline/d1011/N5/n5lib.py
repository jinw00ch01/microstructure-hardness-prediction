"""N5 (2026-10-11 evening, new-direction round on blend_v30b): one shared harness so every candidate is scored the same way.

Every candidate file = blend_v30b + the candidate's own delta. Scoring (criterion N4, now against v30b):
  honest   = selection-honest nested OOF on data/folds.csv: every fit for fold-k rows uses other folds only, every choice
             (option, weight, strength, feature set) is made by inner CV on the outer-training rows, ties -> the earlier
             option in the PRE-DECLARED order, and the option set starts with 'none' (no change = v30b).
  SAFE     = honest all-500 dMSE vs v30b's honest OOF < 0, folds_better >= 3/5, 40 re-split mean < 0, drop-top-3 < 0.
  BOLD     = honest dMSE <= +0.5, a stated mechanism, and a large test change (max 2/day; one bold slot is left on 10-11).
  FAIL     = anything else -> its test file is written as test_NOT_A_CANDIDATE.csv.
Public expectation vs v30b (public RMSE 10.0206885658): E = null + 0.47 (cv_hyp - null) (N4 calibration).
Labels: train only (src.common.load_train via lib_l4). Test rows enter only as label-free inputs and as the candidate's own
test predictions: nothing is fitted on test rows, no test prediction or test statistic chooses anything.

Generic nested selection (use it unless the candidate has a reason not to):
  build(opt, ff, fm) -> length-500 vector: for every fold j of the fold vector ff, rows ff == j get the prediction of a fit
  on rows fm & (ff != j). Then nested(build, opts, ff) picks per outer fold k the option with the lowest inner MSE on
  T = ff != k using build(opt, ff, T), and takes rows ff == k from build(best, ff, ALL).
  Layered candidates (fits on top of v30b's honest OOF p30) use p30 as a fixed input under every ff (stated caveat:
  p30 itself was built on data/folds.csv).
usage:  sys.path.insert(0, '/mnt/project-files/work/hardness-cache/scripts/d1011/N5'); from n5lib import *
        D = load30()
"""
import sys, os, json, hashlib
sys.path.insert(0, '/mnt/project-files/work/hardness-cache/scripts/d1011/N4')
from n4lib import *  # noqa  (np, pd, REPO, CACHE, SUB, N4, STRATA, RHO, groups, load, load29, signflip_p, strat_split)

N5 = CACHE + '/scripts/d1011/N5'
V30B_MD5 = '26291df03fcddd4a6418be9dd72dc132'
PUB30 = 10.0206885658
ALL = np.ones(500, bool)


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def load30():
    """lib_l4.load + v29g + v30b (honest OOF p30 = N4/g1clip/oof.csv, test t30 = blend_v30b.csv) + G1 masks + member tests."""
    D = load29()
    ids, te = D['ids'], D['te_ids']
    rdo = lambda p: pd.read_csv(p, float_precision='round_trip').set_index('ID').hardness.loc[ids].to_numpy().astype(float)
    rdt = lambda n: pd.read_csv(f'{SUB}/{n}', float_precision='round_trip').set_index('ID').hardness.loc[te].to_numpy().astype(float)
    assert md5(f'{SUB}/blend_v30b.csv') == V30B_MD5
    D['p30'] = rdo(N4 + '/g1clip/oof.csv')
    D['t30'] = rdt('blend_v30b.csv')
    D['p21'] = rdo(f'{SUB}/blend_v21_oof.csv')
    for n, f in (('t21', 'blend_v21.csv'), ('t27', 'blend_v27u.csv'), ('t28', 'blend_v28n.csv')):
        D[n] = rdt(f)
    D['G1'] = (D['nz'] >= 9.5) & (D['rs'] < 0.5)
    D['G1t'] = (D['nzt'] >= 9.5) & (D['rst'] < 0.5)
    assert D['G1'].sum() == 208 and D['G1t'].sum() == 433
    # v30b differs from v29g only on G1 test rows; equals it elsewhere
    assert np.array_equal(D['t30'][~D['G1t']], D['t29'][~D['G1t']])
    D['rmse30'] = float(np.sqrt(np.mean((D['p30'] - D['y']) ** 2)))
    return D


def vs30(q, D, nflip=20000):
    """All-500 comparison of a candidate honest OOF q against v30b's honest OOF (same folds)."""
    y, fold, p = D['y'], D['fold'], D['p30']
    d = (q - y) ** 2 - (p - y) ** 2
    ch = d != 0
    per = [float(d[fold == k].sum()) for k in range(5)]
    o = np.argsort(d); keep = np.ones(500, bool); keep[o[:3]] = False
    return dict(all500_dmse=float(d.mean()), rmse_v30b=float(np.sqrt(np.mean((p - y) ** 2))),
                rmse_new=float(np.sqrt(np.mean((q - y) ** 2))), n_changed=int(ch.sum()),
                folds_better=int(sum(x < 0 for x in per)), fold_dsse=[round(x, 1) for x in per],
                signflip_p=float(signflip_p(d[ch], nflip=nflip)) if ch.any() else 1.0,
                droptop3_all500_dmse=float(d[keep].mean()),
                by_group={g: dict(n_changed=int((ch & D['grp'][g]).sum()), dmse_group=float(d[D['grp'][g]].mean())) for g in STRATA})


def public_vs30(q, tnew, D):
    """Expected public dMSE vs v30b by row group (public = random 300 of 1000 test rows); N4: E = null + 0.47 (cvhyp - null)."""
    y, p, t30 = D['y'], D['p30'], D['t30']
    dtr = q - p; dte = tnew - t30
    dse = (q - y) ** 2 - (p - y) ** 2
    E = V = En = Vn = 0.0; rows = {}
    for g in STRATA:
        mtr = D['grp'][g] & (dtr != 0); mte = D['grpt'][g] & (dte != 0); nte = int(mte.sum())
        if nte == 0:
            continue
        s = float(np.std((p - y)[D['grp'][g]]))
        m, v = (float(dse[mtr].mean()), float(dse[mtr].var())) if mtr.sum() >= 2 else (0.0, 0.0)
        E += 0.3 * nte * m; V += 0.3 * nte * v
        dd = dte[mte]; En += 0.3 * float((dd ** 2).sum()); Vn += 0.3 * float((4 * dd ** 2 * s ** 2).sum())
        rows[g] = dict(n_test_changed=nte, test_change_rms=float(np.sqrt(np.mean(dd ** 2))),
                       train_change_rms=float(np.sqrt(np.mean(dtr[mtr] ** 2))) if mtr.any() else 0.0, per_row_dse_train=m)
    E /= 300; V = V ** 0.5 / 300; En /= 300; Vn = Vn ** 0.5 / 300
    base = PUB30 ** 2; n4 = En + 0.47 * (E - En)
    return dict(cv_hyp_public_dmse=E, cv_hyp_sd=V, null_public_dmse=En, null_sd=Vn, n4_expect_dmse=n4,
                cv_hyp_public_rmse=float(np.sqrt(base + E)), null_public_rmse=float(np.sqrt(base + En)),
                n4_expect_public_rmse=float(np.sqrt(base + n4)), groups=rows)


def nested(build, opts, ff, D):
    """Selection-honest nested OOF. opts: dict name -> option (declared order; first must be the no-change option)."""
    y = D['y']; q = np.zeros(500); picks = []; inner = []
    for k in range(5):
        T = ff != k
        sc = {nm: float(np.mean((build(o, ff, T)[T] - y[T]) ** 2)) for nm, o in opts.items()}
        best = min(sc, key=sc.get)
        picks.append(best); inner.append(sc)
        q[ff == k] = build(opts[best], ff, ALL)[ff == k]
    return q, picks, inner


def full_pick(build, opts, D):
    """Full-data pick for the test file: option with the lowest MSE of its OOF on data/folds.csv (ties -> earlier)."""
    y, fold = D['y'], D['fold']
    sc = {nm: float(np.mean((build(o, fold, ALL) - y) ** 2)) for nm, o in opts.items()}
    return min(sc, key=sc.get), sc


def resplit(new_fn, base_fn, D, n=40, seed=20261015):
    """40 stratified re-splits: new_fn(ff) and base_fn(ff) each return a length-500 OOF under fold vector ff.
    Returns the array of MSE(new) - MSE(base). For a candidate layered on p30, base_fn = lambda ff: D['p30']."""
    y = D['y']; rng = np.random.default_rng(seed); R = []
    for _ in range(n):
        ff = strat_split(rng, D['glab'])
        R.append(float(np.mean((new_fn(ff) - y) ** 2) - np.mean((base_fn(ff) - y) ** 2)))
    return np.array(R)


def verdict(S, R, mechanism=None):
    safe = S['all500_dmse'] < 0 and S['folds_better'] >= 3 and float(np.mean(R)) < 0 and S['droptop3_all500_dmse'] < 0
    if safe:
        return 'SAFE'
    if mechanism and S['all500_dmse'] <= 0.5:
        return 'BOLD'
    return 'FAIL'


def freeze_options(dirname, text):
    """Write N5/<dirname>/options.txt once (pre-declared before any label number); returns its sha256. Refuses to overwrite
    a different text."""
    d = f'{N5}/{dirname}'; os.makedirs(d, exist_ok=True); p = d + '/options.txt'
    if os.path.exists(p):
        old = open(p).read()
        assert old == text, f'{p} exists with a different text; options are frozen'
    else:
        open(p, 'w').write(text)
    sha = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    open(d + '/options.sha256', 'w').write(sha + '\n')
    return sha


def check_test(tnew, D):
    assert len(tnew) == 1000 and np.isfinite(tnew).all()
    assert (tnew > 0).all() and (tnew < 1000).all()


def write(dirname, q, tnew, D, meta, R, mechanism=None):
    """Writes N5/<dirname>/oof.csv, result.json, and test.csv (SAFE/BOLD) or test_NOT_A_CANDIDATE.csv (FAIL).
    test files stay in the shared cache only (never in the repo)."""
    d = f'{N5}/{dirname}'; os.makedirs(d, exist_ok=True)
    assert len(q) == 500 and np.isfinite(q).all(); check_test(tnew, D)
    S = vs30(q, D); v = verdict(S, R, mechanism)
    pd.DataFrame({'ID': D['ids'], 'hardness': q}).to_csv(d + '/oof.csv', index=False)
    tp = d + ('/test.csv' if v != 'FAIL' else '/test_NOT_A_CANDIDATE.csv')
    for old in (d + '/test.csv', d + '/test_NOT_A_CANDIDATE.csv'):
        if os.path.exists(old) and old != tp:
            os.remove(old)
    pd.DataFrame({'ID': D['te_ids'], 'hardness': tnew}).to_csv(tp, index=False)
    ch = tnew != D['t30']
    res = dict(meta); res.update(verdict=v, mechanism=mechanism, honest_vs_v30b=S,
                                 resplit40=dict(mean=float(np.mean(R)), sd=float(np.std(R)), P_ge0=float(np.mean(np.asarray(R) >= 0))),
                                 public_vs_v30b=public_vs30(q, tnew, D), n_test_changed=int(ch.sum()),
                                 test_change=dict(rms=float(np.sqrt(np.mean((tnew - D['t30'])[ch] ** 2))) if ch.any() else 0.0,
                                                  max_abs=float(np.abs(tnew - D['t30']).max())),
                                 test_file=tp, test_md5=md5(tp))
    json.dump(res, open(d + '/result.json', 'w'), indent=1, default=float)
    return res
