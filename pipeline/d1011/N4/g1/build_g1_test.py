"""N4/g1 step 5: test builder for the two pre-declared procedures (options.txt).
  1. Reproduces blend_v28n's G1 test rows exactly from LOC/0.5 (v21 + full-fit LOC offset, then v28n's half-step stretch)
     and, after the T-CP mapping, blend_v29g.csv on all 1000 rows (max abs < 1e-9).
  2. Re-runs the selection-honest nested OOF and the full-data pick of each procedure (train only) and checks them
     against run_nested.py's saved outputs (guards against a changed train het parquet, e.g. after an ingest).
  3. g1_stretch (LOC x {0.5, 0.75, 1.0}; buildable now) -> n4lib.write('g1/g1_stretch', ...).
  4. g1 (18 options) -> n4lib.write('g1', ...). If its full-data pick needs an A2/A3 het test parquet that is not
     ingested yet, it stops LOUDLY (exit code 3), prints the ingest commands and writes g1_PENDING.json.
Test rows: v29g + w (newG1 - v28n) on G1 test rows (w = 0.5 on the 19 T-CP rows, 1 on the other 414), other rows = v29g.
Nothing is fitted on test rows; no test prediction is used to choose anything (the picks come from train labels only).
usage (repo root): nice -n 10 timeout 1800 python /mnt/project-files/work/hardness-cache/scripts/d1011/N4/g1/build_g1_test.py"""
import sys, json
sys.path.insert(0, '/mnt/project-files/work/hardness-cache/scripts/d1011/N4/g1')
from g1core import *  # noqa

PIN = {'blend_v21.csv': 'a57dc83224112bfd940003654914ba18', 'blend_v27u.csv': '97a7cf2bbec5e90a65f8f7e388f4819f',
       'blend_v28n.csv': '396a52f2263c4ec857b8dbee1316a3ae', 'blend_v28n_oof.csv': 'a7b357b872a80f780804c30df432fd07'}
for fn, m_ in PIN.items():
    assert md5(f'{SUB}/{fn}') == m_, fn
C = Ctx(); D = C.D; y, fold = C.y, C.fold
te = D['te_ids']; nzt, rst = D['nzt'], D['rst']
rd = lambda n: pd.read_csv(f'{SUB}/{n}', float_precision='round_trip').set_index('ID').hardness.loc[te].to_numpy().astype(float)
t21, t27, t28, t29 = rd('blend_v21.csv'), rd('blend_v27u.csv'), rd('blend_v28n.csv'), D['t29']
G1t = (nzt >= 9.5) & (rst < 0.5)
assert G1t.sum() == 433 and np.array_equal(G1t, D['grpt']['midLF'] | D['grpt']['G1hi'])
mb = pd.read_parquet(CACHE + '/scripts/d1011/midlf/m3b_rows.parquet', columns=['split', 'group', 'gate_cp'])
mt = mb[mb.split == 'test'].loc[te]
tcp_t = (mt.group.to_numpy() == 'midLF') & mt.gate_cp.to_numpy().astype(bool)
assert tcp_t.sum() == 19 and (tcp_t <= D['grpt']['midLF']).all()
assert np.array_equal(t29[G1t & ~tcp_t], t28[G1t & ~tcp_t]), 'v29g G1 rows outside T-CP are not v28n'
wt = np.where(tcp_t, 0.5, 1.0) * G1t
def mapt(n):
    d = n - t28; d[np.abs(d) < 1e-9] = 0.0                    # float noise of the rebuild -> rows stay bit-identical
    return t29 + wt * d

# ---------------------------------------------------------------- 1. reproduction of v28n / v29g from LOC/0.5
n0, i0 = test_new_g1(C, OPTS[NOCHANGE], te, nzt, rst, t21, t27)
r28 = float(np.max(np.abs(n0 - t28))); r29 = float(np.max(np.abs(mapt(n0) - t29)))
v28j = json.load(open(f'{SUB}/blend_v28n.json'))['factor']
print(f'reproduce LOC/0.5: vs blend_v28n test max|d| {r28:.2e} (G1 rows {np.max(np.abs(n0 - t28)[G1t]):.2e}); '
      f'mapped vs blend_v29g.csv max|d| {r29:.2e}; c {i0["c"]:.6f} b {i0["b"]:.6f} k {i0["k"]:.6f} (v28n {v28j["c"]:.6f} {v28j["b"]:.6f} {v28j["k"]:.6f})')
assert r28 < 1e-9 and r29 < 1e-9
assert abs(i0['c'] - v28j['c']) < 1e-9 and abs(i0['b'] - v28j['b']) < 1e-9 and abs(i0['k'] - v28j['k']) < 1e-9

# ---------------------------------------------------------------- 2. honest OOFs + full-data picks (train only)
NR = json.load(open(HERE + '/nested_result.json'))
assert NR['options_sha256'] == OPTIONS_SHA256
HON = {}
for tag, opts in [('g1', OPTS), ('g1_stretch', OPTS_STRETCH)]:
    q, picks, _ = C.nested(fold, opts); qm = C.mapped(q)
    saved = pd.read_csv(f'{HERE}/{tag}_honest_oof.csv', float_precision='round_trip')
    assert (saved.ID.astype(str).to_numpy() == C.ids).all() and float(np.max(np.abs(saved.hardness.to_numpy() - qm))) < 1e-9, tag
    pk, sc = C.full_pick(opts)
    assert pk == NR[tag]['full_pick'] and picks == NR[tag]['picks'], tag
    HON[tag] = (qm, picks, pk)
    print(f'[{tag}] honest OOF matches run_nested.py; per-fold picks {picks}; full-data pick {pk}')


def meta_for(tag, info):
    nr = NR[tag]
    return dict(key='g1', procedure=tag, options_file=HERE + '/options.txt', options_sha256=OPTIONS_SHA256,
                options=nr and (NR['options'] if tag == 'g1' else NR['options_stretch']),
                rule='G1 rows (raw ic_noise >= 9.5 and ridge < 0.5): v21 + het offset (option het spec) then G1hi low-half '
                     'stretch c + k (p - c), k = 1 + s (b / RHO - 1); mapped to v29g as v29g + w (newG1 - v28n), w = 0.5 on '
                     'the 19 T-CP test rows, 1 on the other G1 rows; other rows = v29g',
                honest_nested=dict(picks=nr['picks'], all500_dmse=nr['honest']['all500_dmse'], folds_better=nr['honest']['folds_better'],
                                   fold_dsse=nr['honest']['fold_dsse'], signflip_p=nr['honest']['signflip_p'],
                                   droptop3=nr['honest']['droptop3_all500_dmse']),
                resplit40=nr['resplit40'], full_pick=nr['full_pick'],
                full_pick_insample_dmse=nr['full_pick_insample_dmse'], test_rule=info,
                builder=HERE + '/build_g1_test.py', honest_oof_note='oof.csv = the selection-honest nested OOF (v29g form)')


def report(tag, res):
    v, p = res['vs_v29g'], res['public']
    print(f"[{tag}] written: honest dMSE {v['all500_dmse']:+.3f} folds {v['folds_better']}/5 sf {v['signflip_p']:.3f} "
          f"droptop3 {v['droptop3_all500_dmse']:+.3f} | public CV-hyp dMSE {p['cv_hyp_public_dmse']:+.3f} sd {p['cv_hyp_sd']:.3f} "
          f"null {p['null_public_dmse']:+.3f} sd {p['null_sd']:.3f} | groups {json.dumps(p['groups'])} | md5 {res['test_md5']}")


# ---------------------------------------------------------------- 3. g1_stretch (ready now)
qm, picks, pk = HON['g1_stretch']
nt, info = test_new_g1(C, OPTS[pk], te, nzt, rst, t21, t27)
tnew = mapt(nt); ch = tnew != t29
assert (ch <= G1t).all() and np.isfinite(tnew).all()
info.update(n_test_changed=int(ch.sum()), test_change_rms_changed=float(np.sqrt(np.mean((tnew - t29)[ch] ** 2))) if ch.any() else 0.0)
res = write('g1/g1_stretch', qm, tnew, D, meta_for('g1_stretch', info))
report('g1_stretch', res)

# ---------------------------------------------------------------- 4. g1 (main; may need the A2/A3 test halves)
qm, picks, pk = HON['g1']
try:
    nt, info = test_new_g1(C, OPTS[pk], te, nzt, rst, t21, t27)
except SystemExit as e:
    msg = str(e)
    json.dump(dict(key='g1', status='PENDING_TEST_INPUT', full_pick=pk, message=msg,
                   honest=NR['g1']['honest'], resplit40=NR['g1']['resplit40']), open(HERE + '/g1_PENDING.json', 'w'), indent=1)
    print(msg, file=sys.stderr)
    sys.exit(3)
tnew = mapt(nt); ch = tnew != t29
assert (ch <= G1t).all() and np.isfinite(tnew).all()
info.update(n_test_changed=int(ch.sum()), test_change_rms_changed=float(np.sqrt(np.mean((tnew - t29)[ch] ** 2))))
res = write('g1', qm, tnew, D, meta_for('g1', info))
report('g1', res)
if os.path.exists(HERE + '/g1_PENDING.json'):
    os.remove(HERE + '/g1_PENDING.json')
