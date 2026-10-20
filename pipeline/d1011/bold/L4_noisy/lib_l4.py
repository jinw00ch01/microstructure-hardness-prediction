"""L4 (bold, noisy rows) shared loader. Train rows in data/folds.csv order. Labels are read via src.common.load_train.
Nothing here reads a test prediction; test rows enter only as label-free group counts."""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, json
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
REPO = '/home/claude/microstructure-hardness-prediction'
CACHE = '/mnt/project-files/work/hardness-cache'
SUB = CACHE + '/submissions'
sys.path.insert(0, REPO)
from src.common import load_train  # noqa

STRATA = ('clean', 'LV', 'midLF', 'HLV', 'G1hi')
RHO = 0.9795141681843882
M_HALF = 16.28070373209019


def groups(nz, rs):
    return dict(clean=nz < 9.5, LV=(nz >= 9.5) & (nz < 12) & (rs >= 0.5), midLF=(nz >= 9.5) & (nz < 12) & (rs < 0.5),
                HLV=(nz >= 12) & (rs >= 0.5), G1hi=(nz >= 12) & (rs < 0.5))


def load():
    fo = pd.read_csv(REPO + '/data/folds.csv')
    ids = fo.ID.astype(str).to_numpy(); fold = fo.fold.to_numpy().astype(int)
    tr = load_train(); trx = tr.assign(ID=tr.ID.astype(str)).set_index('ID').loc[ids]
    assert (trx.fold.to_numpy().astype(int) == fold).all()
    y = trx.hardness.to_numpy().astype(float)
    te_ids = pd.read_csv(REPO + '/data/sample_submission.csv').ID.astype(str).to_numpy()
    v3 = pd.read_parquet(REPO + '/data/features_v3.parquet').set_index('ID')
    def rd(n):
        o = pd.read_csv(f'{SUB}/{n}_oof.csv', float_precision='round_trip')
        assert (o.ID.astype(str).to_numpy() == ids).all(), n
        return o.hardness.to_numpy().astype(float)
    D = dict(ids=ids, fold=fold, y=y, te_ids=te_ids, v3=v3,
             p28n=rd('blend_v28n'), p27u=rd('blend_v27u'), p28p=rd('blend_v28p'), p24=rd('blend_v24'),
             p25=rd('blend_v25'))
    nz = v3.loc[ids, 'ic_noise'].to_numpy().astype(float); rs = v3.loc[ids, 'ic_ridge_snr'].to_numpy().astype(float)
    nzt = v3.loc[te_ids, 'ic_noise'].to_numpy().astype(float); rst = v3.loc[te_ids, 'ic_ridge_snr'].to_numpy().astype(float)
    D.update(nz=nz, rs=rs, nzt=nzt, rst=rst, grp=groups(nz, rs), grpt=groups(nzt, rst))
    D['glab'] = np.select([D['grp'][s] for s in STRATA], list(range(len(STRATA))), default=-1)
    c = {s: int(D['grp'][s].sum()) for s in STRATA}; ct = {s: int(D['grpt'][s].sum()) for s in STRATA}
    assert c == dict(clean=267, LV=21, midLF=47, HLV=4, G1hi=161), c
    assert ct == dict(clean=519, LV=42, midLF=113, HLV=6, G1hi=320), ct
    return D


def signflip_p(d, seed=20261010, nflip=20000, chunk=5000):
    rng = np.random.default_rng(seed)
    obs, n, cnt, done = d.mean(), len(d), 0, 0
    while done < nflip:
        c = min(chunk, nflip - done)
        s = rng.integers(0, 2, size=(c, n), dtype=np.int8).astype(float) * 2 - 1
        cnt += int(np.sum((s @ d) / n <= obs + 1e-12)); done += c
    return (1 + cnt) / (nflip + 1)


def strat_split(rng, glab):
    f = np.empty(len(glab), int)
    for gl in range(len(STRATA)):
        idx = np.where(glab == gl)[0]; rng.shuffle(idx); f[idx] = (np.arange(len(idx)) + rng.integers(5)) % 5
    return f


def score(q, base, y, fold, do_sf=True):
    d = (q - y) ** 2 - (base - y) ** 2
    ch = d != 0
    per = []
    for f in range(5):
        m = ch & (fold == f)
        per.append(float(d[m].sum()))
    out = dict(all500_dmse=float(d.mean()), n_changed=int(ch.sum()),
               folds_better=int(sum(1 for f in range(5) if per[f] < 0)), fold_dsse=[round(x, 1) for x in per],
               rmse_base=float(np.sqrt(np.mean((base - y) ** 2))), rmse_new=float(np.sqrt(np.mean((q - y) ** 2))))
    if do_sf and ch.any():
        out['signflip_p'] = float(signflip_p(d[ch], nflip=5000))
        o = np.argsort(d); keep = np.ones(len(d), bool); keep[o[:3]] = False
        out['droptop3_all500_dmse'] = float(d[keep].mean())
    return out
