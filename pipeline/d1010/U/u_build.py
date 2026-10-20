"""blend_v27u = blend_v25 with the 519 clean rows from blend_v26r and the 42 line-visible mid rows from blend_v26L.
Built ahead; submitted only if BOTH v26L and v26r read 'carried' (PLAN_1010.md). Label-free on test; OOF combined the same way."""
import hashlib, os, numpy as np, pandas as pd
S = '/mnt/project-files/work/hardness-cache/submissions/'
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
PIN = {'blend_v26L.csv': '5e47b2fd6b71fda6b668629d92df3943', 'blend_v26r.csv': '1b6037e79823fd18c298feab9e60da3d',
       'blend_v25.csv': 'f1e6dbc47ecd6136a0cacd1a14d034cb'}
for f, m in PIN.items():
    assert md5(S + f) == m, f
rd = lambda f: pd.read_csv(S + f, float_precision='round_trip')
for kind in ('', '_oof'):
    v25, L, R = rd(f'blend_v25{kind}.csv'), rd(f'blend_v26L{kind}.csv'), rd(f'blend_v26r{kind}.csv')
    assert (v25.ID == L.ID).all() and (v25.ID == R.ID).all()
    dl, dr = L.hardness.values != v25.hardness.values, R.hardness.values != v25.hardness.values
    assert not (dl & dr).any(), 'overlapping rows'
    assert dl.sum() == (42 if kind == '' else 21) and dr.sum() == (519 if kind == '' else 267), (dl.sum(), dr.sum())
    u = v25.hardness.values.copy(); u[dl] = L.hardness.values[dl]; u[dr] = R.hardness.values[dr]
    out = pd.DataFrame({'ID': v25.ID, 'hardness': u}); assert np.isfinite(out.hardness).all() and len(out) in (1000, 500)
    p = S + f'blend_v27u{kind}.csv'; assert not os.path.exists(p), p
    out.to_csv(p, index=False)
    chk = pd.read_csv(p, float_precision='round_trip'); assert (chk.hardness.values == u).all()
    print(p, md5(p), 'changed vs v25:', int((u != v25.hardness.values).sum()))
