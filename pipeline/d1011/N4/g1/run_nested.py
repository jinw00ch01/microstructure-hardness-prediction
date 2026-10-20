"""N4/g1 steps 2-4: reproduction asserts, fixed-option in-sample numbers, selection-honest nested OOF of the two
pre-declared procedures (g1: 18 options; g1_stretch: 3 LOC options), 40 stratified re-splits, full-data picks.
Writes nested_result.json, g1_honest_oof.csv, g1_stretch_honest_oof.csv (train OOF only; no test prediction read).
usage (repo root): nice -n 10 timeout 1800 python /mnt/project-files/work/hardness-cache/scripts/d1011/N4/g1/run_nested.py"""
import sys, json, time
sys.path.insert(0, '/mnt/project-files/work/hardness-cache/scripts/d1011/N4/g1')
from g1core import *  # noqa

t0 = time.time()
C = Ctx(); y, fold, D = C.y, C.fold, C.D
q0 = C.build(OPTS[NOCHANGE], fold)
d28 = float(np.max(np.abs(q0 - C.p28n))); d29 = float(np.max(np.abs(C.mapped(q0) - C.p29)))
print(f'reproduce: LOC/0.5 vs v28n OOF max|d| {d28:.2e}; mapped vs v29g OOF max|d| {d29:.2e}', flush=True)
assert d28 < 1e-9 and d29 < 1e-9
res = dict(options_sha256=OPTIONS_SHA256, options=list(OPTS), options_stretch=list(OPTS_STRETCH),
           reproduce=dict(v28n_max_abs=d28, v29g_max_abs=d29))

slim = lambda s: {k: v for k, v in s.items() if k != 'by_group'} | {'by_group_dmse': {g: round(v['dmse_group'], 3) for g, v in s['by_group'].items()}}
fixed = {}
print('IN-SAMPLE fixed options (no selection; 5-fold nested recipe) vs v29g:')
for nm, o in OPTS.items():
    q = C.mapped(C.build(o, fold)); s = vs29(q, D, nflip=20000)
    fixed[nm] = slim(s)
    print(f"  {nm:18s} all500 dMSE {s['all500_dmse']:+.3f} folds {s['folds_better']}/5 {s['fold_dsse']} sf {s['signflip_p']:.3f} "
          f"droptop3 {s['droptop3_all500_dmse']:+.3f}", flush=True)
res['insample_fixed'] = fixed

for tag, opts in [('g1', OPTS), ('g1_stretch', OPTS_STRETCH)]:
    q, picks, inner = C.nested(fold, opts)
    qm = C.mapped(q)
    s = vs29(qm, D, nflip=20000)
    pick_full, sc_full = C.full_pick(opts)
    rng = np.random.default_rng(20261015); R, PK = [], []
    for _ in range(40):
        ff = strat_split(rng, D['glab'])
        qq, pk, _ = C.nested(ff, opts)
        bb = C.mapped(C.build(OPTS[NOCHANGE], ff))
        R.append(float(np.mean((C.mapped(qq) - y) ** 2) - np.mean((bb - y) ** 2))); PK += pk
    R = np.array(R)
    pick_freq = {k: PK.count(k) for k in sorted(set(PK), key=PK.count, reverse=True)}
    pd.DataFrame({'ID': C.ids, 'hardness': qm}).to_csv(f'{HERE}/{tag}_honest_oof.csv', index=False)
    res[tag] = dict(honest=slim(s), picks=picks, inner_best=[round(min(x.values()), 4) for x in inner],
                    inner_nochange=[round(x[NOCHANGE], 4) for x in inner],
                    resplit40=dict(mean=float(R.mean()), sd=float(R.std()), P_ge0=float((R >= 0).mean()), pick_freq=pick_freq),
                    full_pick=pick_full, full_scores={k: round(v, 4) for k, v in sc_full.items()},
                    full_pick_insample_dmse=fixed[pick_full]['all500_dmse'])
    print(f"\n[{tag}] HONEST nested vs v29g: all500 dMSE {s['all500_dmse']:+.3f} folds {s['folds_better']}/5 {s['fold_dsse']} "
          f"sf {s['signflip_p']:.3f} droptop3 {s['droptop3_all500_dmse']:+.3f} rmse {s['rmse_v29g']:.4f}->{s['rmse_new']:.4f}")
    print(f'  picks per outer fold: {picks}')
    print(f'  resplit40: mean {R.mean():+.3f} sd {R.std():.3f} P>=0 {(R >= 0).mean():.2f}; pick freq {pick_freq}')
    print(f'  full-data pick: {pick_full} (in-sample dMSE {fixed[pick_full]["all500_dmse"]:+.3f})', flush=True)

res['elapsed_s'] = time.time() - t0
json.dump(res, open(HERE + '/nested_result.json', 'w'), indent=1)
print('elapsed %.0fs' % res['elapsed_s'])
