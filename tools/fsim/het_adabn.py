"""het_adabn: AdaBN re-prediction of a trained localized het CNN run (src.het_cnn fold{f}.pt), label-free.
For each fold model: BN running statistics are re-estimated (cumulative average) on REAL noisy line-free TRAIN images
(G1 = raw features_v3 ic_noise >= 9.5 and ic_ridge_snr < 0.5; 208 images; --adapt-views D4 views each), then the model
predicts in eval mode. Test images are never part of the adaptation set: they are only predicted (--with-test).
Output: OUT/het_cnn_train.parquet (+ _test) in src.het_cnn's columns (het_cnn = sd over the 16 fold-averaged block
values, logN_cnn, bmean_cnn); clean train rows get the oof fold model, noisy rows the mean of the 5 fold models.
--control: same code path without adaptation (sanity / paired comparison).
usage: python het_adabn.py --run DIR --out DIR [--control] [--adapt-views 2] [--tta 4] [--with-test] [--device cpu]"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--repo', default=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
ap.add_argument('--control', action='store_true'); ap.add_argument('--adapt-views', type=int, default=2)
ap.add_argument('--tta', type=int, default=4); ap.add_argument('--with-test', action='store_true')
ap.add_argument('--device', default='cpu'); ap.add_argument('--threads', type=int, default=1)
ap.add_argument('--g1-only', action='store_true', help='predict only the 208 G1 train rows (quick check)')
A = ap.parse_args()
sys.path.insert(0, A.repo)
import torch
torch.set_num_threads(A.threads)
from PIL import Image
from src import het_cnn as H
dev = torch.device(A.device)
assert dev.type == 'cpu' or torch.cuda.is_available(), '--device cuda but CUDA is not available'
print(f'het_adabn: device {dev} run {A.run} control {A.control} adapt_views {A.adapt_views} tta {A.tta}', flush=True)
REPO = A.repo
v3 = pd.read_parquet(f'{REPO}/data/features_v3.parquet').set_index('ID')
tr_ids = pd.read_csv(f'{REPO}/data/train.csv', usecols=['ID']).ID.astype(str).tolist()     # IDs only, no label
te_ids = pd.read_csv(f'{REPO}/data/sample_submission.csv', usecols=['ID']).ID.astype(str).tolist()
nz, rs = v3.loc[tr_ids, 'ic_noise'].values, v3.loc[tr_ids, 'ic_ridge_snr'].values
g1_ids = [i for i, a, b in zip(tr_ids, nz, rs) if a >= 9.5 and b < 0.5]
assert len(g1_ids) == 208, len(g1_ids)
P = H.load_prep(H.PREP_FP); clean = set(str(i) for i in P['ids'])
fo = pd.read_csv(f'{REPO}/data/folds.csv').set_index('ID')
rd = lambda split, i: np.array(Image.open(f'{REPO}/data/{split}/{i}.png'))
pred_ids = g1_ids if A.g1_only else tr_ids + (te_ids if A.with_test else [])
X = np.stack([rd('train' if i.startswith('TRAIN') else 'test', i) for i in pred_ids])
XA = np.stack([rd('train', i) for i in g1_ids])
os.makedirs(A.out, exist_ok=True)
B, N, t0 = {}, {}, time.time()
for f in range(5):
    ck = torch.load(f'{A.run}/fold{f}.pt', map_location='cpu', weights_only=False)
    m = H.model_from_ckpt(ck).to(dev).to(memory_format=torch.channels_last); c = ck['consts']
    mu, sd = torch.tensor(c['mu_px'], device=dev), torch.tensor(c['sd_px'], device=dev)
    if not A.control:
        bns = [x for x in m.modules() if isinstance(x, torch.nn.modules.batchnorm._BatchNorm)]
        for bn in bns:
            bn.reset_running_stats(); bn.momentum = None
        m.eval()
        for bn in bns:
            bn.train()
        with torch.no_grad():
            for k in range(A.adapt_views):
                for s in range(0, len(XA), 32):
                    x = torch.from_numpy(XA[s:s + 32]).to(dev).float().div_(255.0)[:, None]
                    m(H.d4_t((x - mu) / sd, k).contiguous(memory_format=torch.channels_last))
        m.eval()
    with torch.no_grad():
        pv = H.predict(m, X, mu, sd, dev, None, tta=A.tta, bs=32)
    B[f] = c['mu_b'] + c['s_b'] * pv[0]; N[f] = c['mu_n'] + c['s_n'] * pv[1]
    print(f'fold {f} done ({time.time() - t0:.0f}s)', flush=True)
rows = []
for n, i in enumerate(pred_ids):
    if i in clean:
        f = int(fo.loc[i, 'fold']); b, ln, src = B[f][n], N[f][n], f'oof{f}'
    else:
        b, ln, src = np.mean([B[f][n] for f in range(5)], 0), float(np.mean([N[f][n] for f in range(5)])), 'mean5'
    rows.append(dict(ID=i, het_cnn=float(np.std(b)), logN_cnn=float(ln), bmean_cnn=float(np.mean(b)), src=src))
Df = pd.DataFrame(rows)
full_tr = pd.DataFrame({'ID': tr_ids}).merge(Df, on='ID', how='left')
full_tr.to_parquet(f'{A.out}/het_cnn_train.parquet', index=False)
if A.with_test:
    Df[Df.ID.str.startswith('TEST')].to_parquet(f'{A.out}/het_cnn_test.parquet', index=False)
json.dump(dict(run=A.run, control=A.control, adapt_views=A.adapt_views, tta=A.tta, n_adapt=len(g1_ids), adapt_set='G1 train images',
               g1_only=A.g1_only), open(f'{A.out}/adabn.json', 'w'), indent=1)
print(f'wrote {A.out} ({time.time() - t0:.0f}s)')
