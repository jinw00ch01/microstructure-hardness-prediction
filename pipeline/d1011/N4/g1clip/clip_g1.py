"""N4 g1clip (2026-10-11 ~16:20 KST): G (N4/g1, replace_A3/0.75) with its TEST het offsets clipped to the training range.
Why: the pre-registered label-free build gate (PLAN_N4 addendum D4: < 2% of test G1 offsets beyond the train max |offset|)
read 9/433 = 2.08% (fail by one row). Same-run check (D1), ingest checks (D2) and the SD ratio (D3, 0.974) passed; the LOC
reader already in v29g has 5/433 beyond its own train max on the same images (they are extreme for both readers, corr 0.945).
Fix: clip each test G1 offset to +-max|train G1 offset| (full-data fit, 12.8368). Label-free; uses only train values; no
test statistic chooses anything; the honest OOF is unchanged (clipping touches test rows only).
The G pipeline is linear in the offset on test rows: v29g + w * (stretch(v21 + off) - v28n), stretch slope k on the G1hi
low half (nz < m) and 1 elsewhere, w = 0.5 on the 19 T-CP rows and 1 elsewhere, so the clipped file is
G_test - w * kk * (off - clip(off)). Checked against a direct recompute of the unclipped offsets (they must reproduce G - v29g
up to the LOC part, see assert below)."""
import sys, json, hashlib
sys.path.insert(0, '/mnt/project-files/work/hardness-cache/scripts/d1011/N4')
from n4lib import *  # noqa
N = N4 + '/g1'
D = load29()
R = json.load(open(N + '/result.json'))['test_rule']
assert R['option'] == 'replace_A3/0.75' and R['het_dirs'] == ['ev2s_locK_real5']
g, beta, m, k = np.array(R['g'][0]), R['beta'][0], R['m'], R['k']
G1 = (D['nz'] >= 9.5) & (D['rs'] < 0.5); G1t = (D['nzt'] >= 9.5) & (D['rst'] < 0.5)
G1hit = D['grpt']['G1hi']
tr = pd.read_parquet(REPO + '/data/het_cnn/ev2s_locK_real5/het_cnn_train.parquet').set_index('ID').loc[D['ids']]
te = pd.read_parquet(REPO + '/data/het_cnn/ev2s_locK_real5/het_cnn_test.parquet').set_index('ID').loc[D['te_ids']]
off_tr = beta * (tr.het_cnn.values - np.polyval(g, tr.logN_cnn.values))
off_te = beta * (te.het_cnn.values - np.polyval(g, te.logN_cnn.values))
cap = float(np.abs(off_tr[G1]).max())
assert abs(cap - R['report_only_offset_sd']['train_max_abs']) < 1e-6, (cap, R['report_only_offset_sd'])
assert abs(off_te[G1t].std() - R['report_only_offset_sd']['test_G1']) < 1e-6
excess = np.where(G1t, off_te - np.clip(off_te, -cap, cap), 0.0)
t28 = pd.read_csv(SUB + '/blend_v28n.csv', float_precision='round_trip').hardness.to_numpy()
tcp = G1t & (np.abs(D['t29'] - t28) > 1e-9)
assert tcp.sum() == 19, tcp.sum()
w = np.where(tcp, 0.5, 1.0)
kk = np.where(G1hit & (D['nzt'] < m), k, 1.0)
G = pd.read_csv(N + '/test.csv', float_precision='round_trip')
assert (G.ID.astype(str).to_numpy() == D['te_ids']).all()
gt = G.hardness.to_numpy()
assert hashlib.md5(open(N + '/test.csv', 'rb').read()).hexdigest() == '7b75ed4e74a5f9364f8d1415aa42d0ce'
new = gt - w * kk * excess
n = int((excess != 0).sum())
print(f'cap {cap:.4f}; clipped rows {n} (of {int(G1t.sum())}); their offsets {np.round(off_te[excess != 0], 2).tolist()}; '
      f'file change on them {np.round((new - gt)[excess != 0], 2).tolist()}')
q = pd.read_csv(N + '/oof.csv', float_precision='round_trip').hardness.to_numpy()
res = write('g1clip', q, new, D, dict(name='g1clip', rule='N4/g1 (replace_A3/0.75) with test G1 offsets clipped to +-max|train G1 offset| '
            f'({cap:.4f}); {n} test rows clipped; OOF identical to N4/g1', base_test_md5='7b75ed4e74a5f9364f8d1415aa42d0ce',
            clipped_ids=D['te_ids'][excess != 0].tolist(), builder=N4 + '/g1clip/clip_g1.py'))
print('written g1clip: test md5', res['test_md5'], 'honest', round(res['vs_v29g']['all500_dmse'], 3), 'folds', res['vs_v29g']['folds_better'],
      'public CV-hyp %.3f sd %.3f null %.3f' % (res['public']['cv_hyp_public_dmse'], res['public']['cv_hyp_sd'], res['public']['null_public_dmse']))
