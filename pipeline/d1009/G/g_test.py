"""d1009 G stage 2 (PREREG.txt): localized het CNN offset on G1 vs blend_v23. Run ONCE per estimator.
usage: python g_test.py --het DIR [DIR ...] --name NAME [--p-bar 0.025]
DIR: a src.het_cnn output dir with het_cnn_train.parquet / het_cnn_test.parquet (several dirs = seed average of het_cnn
and logN_cnn, as v22b). Train rows only are used here (the test parquet is not read). Writes G/NAME_result.json and
G/NAME_oof.csv (the cross-fitted OOF of this variant)."""
import argparse, json, sys, numpy as np, pandas as pd
REPO = '/home/claude/microstructure-hardness-prediction'
sys.path.insert(0, REPO)
from src.common import load_train, DATA_DIR
G = '/mnt/project-files/work/hardness-cache/scripts/d1009/G'
ap = argparse.ArgumentParser(); ap.add_argument('--het', nargs='+', required=True); ap.add_argument('--name', required=True)
ap.add_argument('--p-bar', type=float, default=0.025); ap.add_argument('--nperm', type=int, default=2000)
a = ap.parse_args()
tr = load_train(); y = tr.hardness.values; folds = tr.fold.values
v3 = pd.read_parquet(DATA_DIR / 'features_v3.parquet').set_index('ID')
Hs = [pd.read_parquet(f'{d}/het_cnn_train.parquet').set_index('ID').loc[tr.ID] for d in a.het]
h = np.mean([H.het_cnn.values for H in Hs], 0); ln = np.mean([H.logN_cnn.values for H in Hs], 0)
nz, rs = v3.loc[tr.ID, 'ic_noise'].values, v3.loc[tr.ID, 'ic_ridge_snr'].values
G1 = (nz >= 9.5) & (rs < 0.5); keep = (nz >= 12) & ~G1
assert G1.sum() == 208 and keep.sum() == 4
assert np.isfinite(h[G1]).all() and np.isfinite(ln[G1]).all()
# every G1 row must be a mean over fold models (never trained on: G1 rows are not clean prep rows)
for H in Hs:
    if 'src' in H.columns:
        assert not H.src.values[G1].astype(str).tolist().count('none'), 'G1 rows without a prediction'
rd = lambda f: tr[['ID']].merge(pd.read_csv(f'{REPO}/submissions/{f}'), on='ID').hardness.values
o21, o22, o23 = rd('blend_v21_oof.csv'), rd('blend_v22b_oof.csv'), rd('blend_v23_oof.csv')
r = y - o21
rm = lambda p, m: float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))


def crossfit(hv, lv):
    off = np.zeros(500); betas = []
    for f in range(5):
        fit, app = G1 & (folds != f), G1 & (folds == f)
        g = np.polyfit(lv[fit], hv[fit], 1); hr = hv - np.polyval(g, lv)
        b = (hr[fit] @ (r[fit] - r[fit].mean())) / (hr[fit] @ hr[fit]); off[app] = b * hr[app]; betas.append(float(b))
    oof = o21 + off; oof[keep] = o22[keep]
    return oof, betas


oof, betas = crossfit(h, ln)
g1_new, g1_v23 = rm(oof, G1), rm(o23, G1)
fold_new = [rm(oof, G1 & (folds == f)) for f in range(5)]
fold_v23 = [rm(o23, G1 & (folds == f)) for f in range(5)]
nbetter = int(sum(fn < fo for fn, fo in zip(fold_new, fold_v23)))
# (c) permutation: het shuffled within G1, logN kept, full cross-fit per shuffle
rng = np.random.default_rng(20261009)
idx = np.where(G1)[0]; null = []
for _ in range(a.nperm):
    hp = h.copy(); hp[idx] = h[rng.permutation(idx)]
    null.append(rm(crossfit(hp, ln)[0], G1))
null = np.array(null); p = float((1 + (null <= g1_new).sum()) / (1 + a.nperm))
# (d) drop the single G1 row with the largest SSE gain vs v23 (no refit)
gain = (o23 - y) ** 2 - (oof - y) ** 2; gain[~G1] = -np.inf
top = int(np.argmax(gain)); m2 = G1.copy(); m2[top] = False
d_drop = rm(o23, m2) - rm(oof, m2)
bars = dict(a=g1_new <= g1_v23 - 0.10, b=nbetter >= 4, c=p < a.p_bar, d=d_drop >= 0.05)
res = dict(name=a.name, het=a.het, n_G1=int(G1.sum()), G1_new=g1_new, G1_v23=g1_v23, G1_gain=g1_v23 - g1_new,
           fold_G1_new=fold_new, fold_G1_v23=fold_v23, folds_better=nbetter, perm_p=p, perm_null_median=float(np.median(null)),
           drop_top_row=tr.ID.values[top], drop_top_gain_left=float(d_drop), fold_betas=betas,
           all_new=rm(oof, np.ones(500, bool)), all_v23=rm(o23, np.ones(500, bool)),
           G1_lt12_new=rm(oof, G1 & (nz < 12)), G1_lt12_v23=rm(o23, G1 & (nz < 12)),
           G1_ge12_new=rm(oof, G1 & (nz >= 12)), G1_ge12_v23=rm(o23, G1 & (nz >= 12)),
           bars={k: bool(v) for k, v in bars.items()}, PASS=bool(all(bars.values())), p_bar=a.p_bar)
# label-free context: relation to the old seed-averaged het CNN on G1 (partial given logN)
D = '/mnt/project-files/work/hardness-cache/laptop-gpu-2026-10-08/'
old = [pd.read_csv(f'{D}{s}/train.csv').set_index('ID').loc[tr.ID] for s in ('het_ev2s_e32r4', 'het_ev2s_e32r4_s1')]
ho = np.mean([o.het_cnn.values for o in old], 0); lo = np.mean([o.logN_cnn.values for o in old], 0)
Z = np.column_stack([np.ones(G1.sum()), ln[G1]])
res_new = h[G1] - Z @ np.linalg.lstsq(Z, h[G1], rcond=None)[0]
Zo = np.column_stack([np.ones(G1.sum()), lo[G1]])
res_old = ho[G1] - Zo @ np.linalg.lstsq(Zo, ho[G1], rcond=None)[0]
res['corr_resid_new_old_G1'] = float(np.corrcoef(res_new, res_old)[0, 1])
res['corr_logN_new_old_G1'] = float(np.corrcoef(ln[G1], lo[G1])[0, 1])
# secondary, report only: two-estimator offset (no intercept) on G1
off2 = np.zeros(500)
for f in range(5):
    fit, app = G1 & (folds != f), G1 & (folds == f)
    X = []
    for hv, lv in ((h, ln), (ho, lo)):
        g = np.polyfit(lv[fit], hv[fit], 1); X.append(hv - np.polyval(g, lv))
    X = np.column_stack(X)
    bb = np.linalg.lstsq(X[fit], r[fit] - r[fit].mean(), rcond=None)[0]; off2[app] = X[app] @ bb
oof2 = o21 + off2; oof2[keep] = o22[keep]
res['secondary_two_estimator_G1'] = rm(oof2, G1)
json.dump(res, open(f'{G}/{a.name}_result.json', 'w'), indent=1)
pd.DataFrame({'ID': tr.ID, 'hardness': oof}).to_csv(f'{G}/{a.name}_oof.csv', index=False)
print(json.dumps(res, indent=1))
print('PASS' if res['PASS'] else 'FAIL', {k: bool(v) for k, v in bars.items()})
