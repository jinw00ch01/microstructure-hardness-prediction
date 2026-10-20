"""IDEA 2 builder (NOT RUN by the L4 lens; run only on the orchestrator's decision): blend_v29t = blend_v28n with the
v26L recipe on the 19 T-CP test rows (midLF AND d1011/midlf m3b gate_cp; label-free gate trained on degraded clean
TRAIN copies only), optionally (--lv75) LV test rows at w 0.75 instead of 0.5.
  T-CP rows: new = 0.5 * v28n + 0.5 * f7, f7 = coef_full (blend_v26L.json, clean OLS on 267 clean train rows) . V0
             terms (d1010/N/gate q_test_*.parquet = the unchanged cleanhead pipeline).
  --lv75:    LV rows: new = 0.25 * v25 + 0.75 * f7 (v28n there = 0.5 v25 + 0.5 f7, asserted to 1e-6).
Every other row is bit-identical to blend_v28n (asserted). Writes OUT.csv (and a json with md5 and row counts) to
--out-dir, never to submissions/. No label is read.
usage: python tcp_build.py --name blend_v29t --out-dir DIR [--lv75]"""
import argparse, hashlib, json, os, sys
import numpy as np, pandas as pd

REPO = '/home/claude/microstructure-hardness-prediction'
C = '/mnt/project-files/work/hardness-cache'
ap = argparse.ArgumentParser(); ap.add_argument('--name', required=True); ap.add_argument('--out-dir', required=True)
ap.add_argument('--lv75', action='store_true'); A = ap.parse_args()
TERMS = ['fd_area', 'pore_frac', 'logasp_aw', 'S_align', 'logN_eff', 'het4', 'bd_frac']
te = pd.read_csv(REPO + '/data/sample_submission.csv').ID.astype(str).to_numpy()
v28 = pd.read_csv(C + '/submissions/blend_v28n.csv').set_index('ID').loc[te].hardness.to_numpy()
assert hashlib.md5(open(C + '/submissions/blend_v28n.csv', 'rb').read()).hexdigest() == '396a52f2263c4ec857b8dbee1316a3ae'
cf = json.load(open(C + '/submissions/blend_v26L.json'))['coef_full']
q = pd.concat([pd.read_parquet(C + f'/scripts/d1010/N/gate/q_test_{k}of2.parquet') for k in (0, 1)]).loc[te]
f7 = cf['intercept'] + sum(cf[t] * q[f'V0__{t}'].to_numpy().astype(float) for t in TERMS)
m = pd.read_parquet(C + '/scripts/d1011/midlf/m3b_rows.parquet', columns=['split', 'group', 'gate_cp', 'ic_noise', 'ic_ridge_snr'])
m = m[m.split == 'test'].loc[te]
tcp = (m.group.to_numpy() == 'midLF') & m.gate_cp.to_numpy().astype(bool)
assert tcp.sum() == 19 and np.isfinite(f7[tcp]).all()
new = v28.copy(); new[tcp] = 0.5 * v28[tcp] + 0.5 * f7[tcp]
info = dict(name=A.name, base='blend_v28n', tcp_rows=int(tcp.sum()), lv75=A.lv75)
if A.lv75:
    v3 = pd.read_parquet(REPO + '/data/features_v3.parquet').set_index('ID').loc[te]
    nz, rs = v3.ic_noise.to_numpy(), v3.ic_ridge_snr.to_numpy()
    lv = (nz >= 9.5) & (nz < 12) & (rs >= 0.5); assert lv.sum() == 42
    v25 = pd.read_csv(C + '/submissions/blend_v25.csv').set_index('ID').loc[te].hardness.to_numpy()
    assert np.max(np.abs(v28[lv] - (0.5 * v25[lv] + 0.5 * f7[lv]))) < 1e-6, 'v28n LV rows != 0.5 v25 + 0.5 f7'
    new[lv] = 0.25 * v25[lv] + 0.75 * f7[lv]; info['lv_rows'] = int(lv.sum())
ch = new != v28
assert ch.sum() == tcp.sum() + (info.get('lv_rows', 0)) and np.isfinite(new).all()
os.makedirs(A.out_dir, exist_ok=True)
fp = f'{A.out_dir}/{A.name}.csv'
pd.DataFrame({'ID': te, 'hardness': new}).to_csv(fp, index=False)
info['md5'] = hashlib.md5(open(fp, 'rb').read()).hexdigest(); info['rows_changed'] = int(ch.sum())
json.dump(info, open(f'{A.out_dir}/{A.name}.json', 'w'), indent=1)
print(json.dumps(info))
