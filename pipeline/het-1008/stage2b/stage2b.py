"""Secondary (post-hoc design, fixed before the seed-1 run arrived) stage-2 test of CNN het on noisy images.
Design chosen after seeing the seed-0 primary test fail: gate raw ic_noise >= 12 (the band the renderer was
calibrated on); offset = beta * (het_cnn - g(logN_cnn)), g linear in-fold on gated train rows, beta cross-fitted,
no intercept. Base = blend_v21 nested OOF.
Bar (fixed 2026-10-08 ~17:15 KST, before seed 1): with the het_cnn averaged over the available seeds,
gated RMSE improves >= 0.10, >= 4/5 folds better, permutation p < 0.05 (2000 shuffles of het_cnn within gated rows).
usage: python stage2b.py train_seed0.csv [train_seed1.csv ...]"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/claude/microstructure-hardness-prediction')
from src.common import load_train
tr = load_train(); y = tr.hardness.values; folds = tr.fold.values
Hs = [pd.read_csv(p).set_index('ID').loc[tr.ID] for p in sys.argv[1:]]
h = np.mean([H.het_cnn.values for H in Hs], 0); ln = np.mean([H.logN_cnn.values for H in Hs], 0)
nz = pd.read_parquet('/home/claude/microstructure-hardness-prediction/data/features_v3.parquet').set_index('ID').loc[tr.ID, 'ic_noise'].values
o = tr[['ID']].merge(pd.read_csv('/home/claude/microstructure-hardness-prediction/submissions/blend_v21_oof.csv'), on='ID').hardness.values
r = y - o; m = nz >= 12
rm = lambda a, mm: np.sqrt(np.mean((a[mm] - y[mm]) ** 2))
def cv(hh):
    off = np.zeros(500)
    for f in range(5):
        fit, app = m & (folds != f), m & (folds == f)
        g = np.polyfit(ln[fit], hh[fit], 1); hr = hh - np.polyval(g, ln)
        b = (hr[fit] @ (r[fit] - r[fit].mean())) / (hr[fit] @ hr[fit]); off[app] = b * hr[app]
    return off
off = cv(h); p = o + off
fb = sum(rm(p, m & (folds == f)) < rm(o, m & (folds == f)) for f in range(5))
rng = np.random.default_rng(0); idx = np.where(m)[0]; obs = rm(p, m); c = 0
for _ in range(2000):
    hp = h.copy(); hp[idx] = h[rng.permutation(idx)]
    c += rm(o + cv(hp), m) <= obs
print(f'seeds {len(Hs)} gated n={m.sum()} base {rm(o,m):.3f} -> {obs:.3f} (d {obs-rm(o,m):+.3f}) folds better {fb}/5 perm p {(c+1)/2001:.4f} | all {rm(o,np.ones(500,bool)):.3f} -> {rm(p,np.ones(500,bool)):.3f}')
