"""Write blend_v30b (= v29g + G_clip) and blend_v30a (= v29g + G_clip + C) to hardness-cache submissions/, replacing only the
PENDING placeholder .json files that build_slate.py wrote. G_clip = N4/g1clip (G with test offsets clipped to the train
range, see clip_g1.py); C = N4/clean/posthoc_clean_only (clean rows, disjoint from G1 rows). Unions are exact deltas.
Public expectation per PLAN_N4 addendum B: E = null + f (cv - null), f = 1 and f = 0.47; sd from n4lib.public_expect."""
import sys, json, hashlib, math, os
sys.path.insert(0, '/mnt/project-files/work/hardness-cache/scripts/d1011/N4')
from n4lib import *  # noqa
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
Phi = lambda z: 0.5 * math.erfc(-z / math.sqrt(2))
D = load29(); ids, tid = D['ids'], D['te_ids']
def rd(p, ix):
    o = pd.read_csv(p, float_precision='round_trip'); assert (o.ID.astype(str).to_numpy() == ix).all(), p
    return o.hardness.to_numpy().astype(float)
Gq, Gt = rd(N4 + '/g1clip/oof.csv', ids), rd(N4 + '/g1clip/test.csv', tid)
assert md5(N4 + '/g1clip/test.csv') == '26291df03fcddd4a6418be9dd72dc132'
Cq, Ct = rd(N4 + '/clean/posthoc_clean_only/oof.csv', ids), rd(N4 + '/clean/posthoc_clean_only/test.csv', tid)
assert md5(N4 + '/clean/posthoc_clean_only/test.csv') == '4877dedcf68466d83a774be78fe7cec9'
dG, dC = Gt - D['t29'], Ct - D['t29']; assert not ((dG != 0) & (dC != 0)).any()
qG, qC = Gq - D['p29'], Cq - D['p29']; assert not ((qG != 0) & (qC != 0)).any()
BASE = 10.0564339191 ** 2
out = {}
for letter, pieces, tier in [('b', ['G_clip'], 'SAFE'), ('a', ['G_clip', 'C'], 'BOLD (C is post hoc)')]:
    q = D['p29'] + qG + (qC if 'C' in pieces else 0); t = D['t29'] + dG + (dC if 'C' in pieces else 0)
    name = f'blend_v30{letter}'; path = f'{SUB}/{name}.csv'
    assert not os.path.exists(path), f'{path} exists: never overwrite'
    j_old = json.load(open(f'{SUB}/{name}.json')); assert j_old.get('status') == 'PENDING_TEST_INPUT', name
    pd.DataFrame({'ID': tid, 'hardness': t}).to_csv(path, index=False)
    v = vs29(q, D); p = public_expect(q, t, D)
    E1, sd, En = p['cv_hyp_public_dmse'], p['cv_hyp_sd'], p['null_public_dmse']
    E47 = En + 0.47 * (E1 - En)
    J = dict(name=name, status='READY', base='blend_v29g', base_md5=V29G_MD5, md5=md5(path), tier=tier, pieces=pieces,
             rule=('v29g + G_clip' + (' + C' if 'C' in pieces else '') + '. G_clip: G1 rows (raw features_v3 ic_noise >= 9.5 and '
                   'ic_ridge_snr < 0.5; 433 test rows) = v21 + A3 (ev2s_locK_real5) het offset (beta and g fitted on the 208 train G1 '
                   'rows), test offsets clipped to +-12.8368 (train max |offset|; 9 test rows), then the G1hi low-half stretch at step '
                   '0.75 (k 1.3723), mapped to v29g with w 0.5 on the 19 T-CP rows. C: clean rows 0.25*(v21base+off_V4) + 0.75*f8_C2.'),
             builders=[N4 + '/g1/build_g1_test.py', N4 + '/g1clip/clip_g1.py', N4 + '/g1clip/build_v30ab.py'] + ([N4 + '/clean/build_clean.py'] if 'C' in pieces else []),
             honest_vs_v29g=v, strict_honest_G=-2.202, resplit_mean=(-0.519 + (0.211 if 'C' in pieces else 0)),
             n_test_changed=int((t != D['t29']).sum()), test_change_rms_changed=float(np.sqrt(np.mean((t - D['t29'])[t != D['t29']] ** 2))),
             public=dict(f1=dict(dmse=E1, rmse=math.sqrt(BASE + E1)), f047=dict(dmse=E47, rmse=math.sqrt(BASE + E47)), sd_dmse=sd,
                         null=dict(dmse=En, rmse=math.sqrt(BASE + En)), P_le_10_f047=Phi((100 - BASE - E47) / sd), P_le_9='< 0.001'),
             reading=dict(read_better='public < 10.0564 (v29g)', close_G_family='v30b >= 10.1505 (dG >= +1.90 MSE)',
                          C_split='dC = M(v30a) - M(v30b) in public MSE; C reaches the final only if v30a reads below v30b'),
             gate=dict(D1_same_run='PASS (c0 sums equal)', D2_ingest='PASS', D3_sd_ratio=0.974, D4_beyond_train_max='9/433 = 2.08% (FAIL by one row) -> offsets clipped to the train range (label-free); LOC reader 5/433 on the same images'))
    json.dump(J, open(f'{SUB}/{name}.json', 'w'), indent=1)
    out[name] = J
    print(name, J['md5'], 'honest %.3f folds %d' % (v['all500_dmse'], v['folds_better']), 'changed', J['n_test_changed'],
          'public f1 %.4f f.47 %.4f sd(dMSE) %.2f null %.4f P10 %.2f' % (J['public']['f1']['rmse'], J['public']['f047']['rmse'], sd, J['public']['null']['rmse'], J['public']['P_le_10_f047']))
