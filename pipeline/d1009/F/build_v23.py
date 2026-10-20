"""blend_v23 = blend_v21 + beta*(het_cnn - g(logN_cnn)) on G1 = raw ic_noise >= 9.5 and ic_ridge_snr < 0.5 (noisy-preset-like
images, gate chosen from label-free calibration, PREREG.txt P1); rows with ic_noise >= 12 and ic_ridge_snr >= 0.5 keep the
blend_v22b offset. g linear and beta (no intercept) fitted on the G1 train rows against r = y - blend_v21 nested OOF; het_cnn and
logN_cnn averaged over seeds 0 and 1 (src.het_cnn ev2s_e32r4, ev2s_e32r4_s1). Test images are only passed through.
Writes submissions/blend_v23.csv, blend_v23_oof.csv, blend_v23.json."""
import json, sys, numpy as np, pandas as pd
REPO = '/home/claude/microstructure-hardness-prediction'
sys.path.insert(0, REPO)
from src.common import load_train, DATA_DIR
OUT = sys.argv[1] if len(sys.argv) > 1 else 'blend_v23'
tr = load_train(); y = tr.hardness.values; folds = tr.fold.values
te_ids = pd.read_csv(DATA_DIR / 'sample_submission.csv').ID
v3 = pd.read_parquet(DATA_DIR / 'features_v3.parquet').set_index('ID')
D = '/mnt/project-files/work/hardness-cache/laptop-gpu-2026-10-08/'
S = ('het_ev2s_e32r4', 'het_ev2s_e32r4_s1')
def avg(split, ids):
    Hs = [pd.read_csv(f'{D}{s}/{split}.csv').set_index('ID').loc[ids] for s in S]
    return np.mean([H.het_cnn.values for H in Hs], 0), np.mean([H.logN_cnn.values for H in Hs], 0)
h, ln = avg('train', tr.ID); ht, lnt = avg('test', te_ids)
nz, rs = v3.loc[tr.ID, 'ic_noise'].values, v3.loc[tr.ID, 'ic_ridge_snr'].values
nzt, rst = v3.loc[te_ids, 'ic_noise'].values, v3.loc[te_ids, 'ic_ridge_snr'].values
G1, G1t = (nz >= 9.5) & (rs < 0.5), (nzt >= 9.5) & (rst < 0.5)
keep, keept = (nz >= 12) & ~G1, (nzt >= 12) & ~G1t
o21 = tr[['ID']].merge(pd.read_csv(f'{REPO}/submissions/blend_v21_oof.csv'), on='ID').hardness.values
o22 = tr[['ID']].merge(pd.read_csv(f'{REPO}/submissions/blend_v22b_oof.csv'), on='ID').hardness.values
t21 = pd.read_csv(f'{REPO}/submissions/blend_v21.csv'); t22 = pd.read_csv(f'{REPO}/submissions/blend_v22b.csv')
assert (t21.ID.values == te_ids.values).all() and (t22.ID.values == te_ids.values).all()
r = y - o21
rm = lambda p, mm: float(np.sqrt(np.mean((p[mm] - y[mm]) ** 2)))
off = np.zeros(500); betas = []
for f in range(5):
    fit, app = G1 & (folds != f), G1 & (folds == f)
    g = np.polyfit(ln[fit], h[fit], 1); hr = h - np.polyval(g, ln)
    b = (hr[fit] @ (r[fit] - r[fit].mean())) / (hr[fit] @ hr[fit]); off[app] = b * hr[app]; betas.append(float(b))
oof = o21 + off; oof[keep] = o22[keep]
g = np.polyfit(ln[G1], h[G1], 1); hr = h - np.polyval(g, ln); hrt = ht - np.polyval(g, lnt)
b = float((hr[G1] @ (r[G1] - r[G1].mean())) / (hr[G1] @ hr[G1]))
pred = t21.hardness.values + np.where(G1t, b * hrt, 0.0); pred[keept] = t22.hardness.values[keept]
assert len(pred) == 1000 and np.isfinite(pred).all()
pd.DataFrame({'ID': te_ids, 'hardness': pred}).to_csv(f'{REPO}/submissions/{OUT}.csv', index=False)
pd.DataFrame({'ID': tr.ID, 'hardness': oof}).to_csv(f'{REPO}/submissions/{OUT}_oof.csv', index=False)
ALL = np.ones(500, bool)
info = dict(base='blend_v21', gate='G1: raw ic_noise >= 9.5 and ic_ridge_snr < 0.5 (features_v3); ic_noise >= 12 and ridge >= 0.5 keep blend_v22b',
            n_train_G1=int(G1.sum()), n_train_keep=int(keep.sum()), n_test_G1=int(G1t.sum()), n_test_keep=int(keept.sum()),
            seeds=list(S), g=[float(x) for x in g], beta=b, fold_betas=betas,
            nested_cv_all=rm(oof, ALL), nested_cv_v22b=rm(o22, ALL), G1_nested=rm(oof, G1), G1_v22b=rm(o22, G1),
            fold_rmse=[rm(oof, folds == f) for f in range(5)], fold_rmse_v22b=[rm(o22, folds == f) for f in range(5)],
            notes='PREREG d1009/F P1; het_cnn from src.het_cnn trained on clean-train renders only; test images prediction only')
json.dump(info, open(f'{REPO}/submissions/{OUT}.json', 'w'), indent=1)
print(json.dumps(info, indent=1))
d = pred - t22.hardness.values
print(f'LOCAL ONLY: test rows changed vs v22b {int((np.abs(d) > 1e-9).sum())}, rms change {np.sqrt(np.mean(d[np.abs(d) > 1e-9] ** 2)):.3f}')
