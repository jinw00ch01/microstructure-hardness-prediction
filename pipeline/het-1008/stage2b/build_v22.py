"""Build blend_v22 = blend_v21 + beta * (het_cnn - g(logN_cnn)) on images with raw ic_noise >= 12, ONLY if stage2b.py
passed its fixed bar. Same model as stage2b.py: het_cnn / logN_cnn averaged over the seed runs given; g linear and
beta (no intercept) fitted on all 165 gated train rows against r = y - blend_v21 nested OOF. Test images are only
passed through (their het_cnn comes from CNNs trained on clean train renders; nothing is fitted on them).
Writes submissions/blend_v22.csv, blend_v22_oof.csv (v21 OOF + the cross-fitted offset), blend_v22.json.
usage: python build_v22.py --train s0_train.csv [s1_train.csv] --test s0_test.csv [s1_test.csv]"""
import argparse, json, sys, numpy as np, pandas as pd
REPO = '/home/claude/microstructure-hardness-prediction'
sys.path.insert(0, REPO)
from src.common import load_train
ap = argparse.ArgumentParser(); ap.add_argument('--train', nargs='+'); ap.add_argument('--test', nargs='+'); ap.add_argument('--out', default='blend_v22')
a = ap.parse_args()
tr = load_train(); y = tr.hardness.values; folds = tr.fold.values
te_ids = pd.read_csv(f'{REPO}/data/sample_submission.csv').ID
v3 = pd.read_parquet(f'{REPO}/data/features_v3.parquet').set_index('ID')
def avg(paths, ids):
    Hs = [pd.read_csv(p).set_index('ID').loc[ids] for p in paths]
    return np.mean([H.het_cnn.values for H in Hs], 0), np.mean([H.logN_cnn.values for H in Hs], 0)
h, ln = avg(a.train, tr.ID); ht, lnt = avg(a.test, te_ids)
m = v3.loc[tr.ID, 'ic_noise'].values >= 12; mt = v3.loc[te_ids, 'ic_noise'].values >= 12
o = tr[['ID']].merge(pd.read_csv(f'{REPO}/submissions/blend_v21_oof.csv'), on='ID').hardness.values
v21 = pd.read_csv(f'{REPO}/submissions/blend_v21.csv'); assert (v21.ID.values == te_ids.values).all()
r = y - o
rm = lambda p, mm: float(np.sqrt(np.mean((p[mm] - y[mm]) ** 2)))
# nested OOF offset (identical to stage2b.cv)
off = np.zeros(500); betas = []
for f in range(5):
    fit, app = m & (folds != f), m & (folds == f)
    g = np.polyfit(ln[fit], h[fit], 1); hr = h - np.polyval(g, ln)
    b = (hr[fit] @ (r[fit] - r[fit].mean())) / (hr[fit] @ hr[fit]); off[app] = b * hr[app]; betas.append(float(b))
# final fit on all gated train rows
g = np.polyfit(ln[m], h[m], 1); hr = h - np.polyval(g, ln); hrt = ht - np.polyval(g, lnt)
b = float((hr[m] @ (r[m] - r[m].mean())) / (hr[m] @ hr[m]))
offt = np.where(mt, b * hrt, 0.0)
p22 = v21.hardness.values + offt
assert len(p22) == 1000 and np.isfinite(p22).all()
pd.DataFrame({'ID': te_ids, 'hardness': p22}).to_csv(f'{REPO}/submissions/{a.out}.csv', index=False)
pd.DataFrame({'ID': tr.ID, 'hardness': o + off}).to_csv(f'{REPO}/submissions/{a.out}_oof.csv', index=False)
info = dict(base='blend_v21', gate='raw ic_noise >= 12', n_train_gated=int(m.sum()), n_test_gated=int(mt.sum()),
            seeds_train=a.train, seeds_test=a.test, g=[float(x) for x in g], beta=b, fold_betas=betas,
            nested_cv_all=rm(o + off, np.ones(500, bool)), nested_cv_v21=rm(o, np.ones(500, bool)),
            gated_nested=rm(o + off, m), gated_v21=rm(o, m),
            hr_mean_train_gated=float(hr[m].mean()), hr_sd_train_gated=float(hr[m].std()),
            offset_oof_rms=float(np.sqrt(np.mean(off[m] ** 2))),
            notes='secondary post-hoc design fixed before seed 1 (stage2b.py); het_cnn from src.het_cnn on clean '
                  'train renders; test images prediction only')
json.dump(info, open(f'{REPO}/submissions/{a.out}.json', 'w'), indent=1)
print(json.dumps(info, indent=1))
# local check only, never committed (statistics of test-image values)
print(f'LOCAL ONLY: test gated offset rms {np.sqrt(np.mean(offt[mt]**2)):.3f} mean {offt[mt].mean():+.3f}; '
      f'hr test gated mean {hrt[mt].mean():+.4f} sd {hrt[mt].std():.4f}')
