"""het_fsim: train / evaluate the localized het CNN (src.het_cnn, unchanged) with a share of degrade_sim ('fsim') items.

Training (all src.het_cnn options pass through unchanged; only training items change, see fsim_set.py):
  python -W ignore tools/fsim/het_fsim.py --p-fsim 0.5 [--p-g1hi 0.77] <src.het_cnn training options> --out RUN
  Writes data/het_cnn/RUN/fsim.json (the fsim settings); a resume with other fsim settings is refused.
Label-free evaluation in the S0d protocol (d1011/G/s0/s0_het.py: held-out clean sources of each fold, K = 4 fixed-seed
copies per source in two worlds, 'render' = src.het_cnn noisy-preset render, 'fsim' = degrade_sim 'G1hi' with seeds
[20261011, fold, j, k], TTA 1, src.het_cnn.stage1_metrics on the prep targets):
  python -W ignore tools/fsim/het_fsim.py --eval-fsim RUN [RUN ...] [--device cuda]
  Writes data/het_cnn/RUN/score_fsim.json and prints one 'EVAL' line per run and world (send these back).
No hardness label is read anywhere. Test images are only passed through trained models (src.het_cnn's own prediction
step); nothing is fitted on them, and no statistic of test predictions is used by any gate."""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fsim_set  # noqa: E402

SEED0, K_EVAL = 20261011, 4


def own_args(argv):
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--p-fsim', type=float, default=None)
    ap.add_argument('--p-g1hi', type=float, default=0.77)
    ap.add_argument('--repo', default=os.path.abspath(os.path.join(HERE, '..', '..')))
    ap.add_argument('--out-root', default=None, help='override src.het_cnn.OUT_ROOT (smoke tests outside data/)')
    ap.add_argument('--eval-fsim', nargs='+', default=None, metavar='RUN')
    ap.add_argument('--eval-smoke', action='store_true', help='eval on 3 held-out sources per fold only')
    ap.add_argument('--selfcheck', action='store_true', help='p 0 items == RenderSet items; p 1 items carry prep targets')
    ap.add_argument('--spawn', action='store_true', help='force spawn workers (Windows behaviour) for a smoke test')
    return ap.parse_known_args(argv)


def eval_runs(o, rest):
    import pandas as pd
    import torch
    from pathlib import Path
    H, _ = fsim_set.install(o.repo, 0.0, o.p_g1hi)
    D = fsim_set._sim()
    dev_s = 'cuda' if ('--device' in rest and rest[rest.index('--device') + 1] == 'cuda' and torch.cuda.is_available()) else 'cpu'
    dev = torch.device(dev_s)
    torch.set_num_threads(int(rest[rest.index('--threads') + 1]) if '--threads' in rest else 1)
    import cv2
    cv2.setNumThreads(1)
    root = Path(o.out_root) if o.out_root else H.OUT_ROOT
    P = H.load_prep(H.PREP_FP)
    ids = np.array([str(i) for i in P['ids']])
    assert list(ids) == D.train_ids(), 'degrade_sim prep ids differ from src.het_cnn prep ids'
    logn = np.log(P['n_eff'])
    fo = pd.read_csv(H.DATA_DIR / 'folds.csv', usecols=['ID', 'fold']).set_index('ID')
    folds_of = fo.loc[ids, 'fold'].values.astype(int)
    R, M = H.PRESETS['noisy']
    for run in o.eval_fsim:
        d = root / run
        t0 = time.time()
        res = dict(run=run, protocol='S0d: held-out sources x K=4, seeds [20261011, f, j, k], TTA 1', per_fold={})
        acc = {w: dict(bh=[], nh=[], ch=[], rows=[]) for w in ('render', 'fsim')}
        ct_all = {}
        for f in range(5):
            if not (d / f'fold{f}.pt').exists():
                continue
            ck = torch.load(d / f'fold{f}.pt', map_location='cpu', weights_only=False)
            fj = json.load(open(d / f'fold{f}.json'))
            va = np.where(folds_of == f)[0]
            if len(va) != fj['n_heldout']:          # a --smoke run held out the first 8
                va = va[:fj['n_heldout']]
            if o.eval_smoke:
                va = va[:3]
            model = H.model_from_ckpt(ck).to(dev).to(memory_format=torch.channels_last).eval()
            c = ck['consts']
            mu_t, sd_t = torch.tensor(c['mu_px']), torch.tensor(c['sd_px'])
            ct_of = {int(j): H.cell_targets(P['ws'][j].astype(np.int64), P['pore'][j]) for j in va}
            ct_all.update(ct_of)
            res['per_fold'][f] = {}
            for w in ('render', 'fsim'):
                X, rows = [], []
                for j in va:
                    for k in range(K_EVAL):
                        seed = [SEED0, int(f), int(j), int(k)]
                        if w == 'render':
                            img, _ = H.render_noisy(P, int(j), np.random.default_rng(seed), R, M=M)
                        else:
                            img, _ = D.degrade_train(int(j), 'G1hi', seed=seed)
                        X.append(img); rows.append(int(j))
                pv = H.predict(model, np.stack(X), mu_t, sd_t, dev, None, tta=1, bs=32)
                bh = c['mu_b'] + c['s_b'] * pv[0]; nh = c['mu_n'] + c['s_n'] * pv[1]; ch = c['mu_b'] + c['s_b'] * pv[-1]
                rows = np.array(rows)
                m = H.stage1_metrics(bh, nh, P['b'][rows], P['het4'][rows], logn[rows], ch,
                                     np.stack([ct_of[int(j)] for j in rows]))
                res['per_fold'][f][w] = dict(n=int(len(rows)), pc=m['pcorr_het_given_logN'], corr_logN=m['corr_logN'],
                                             W=m['corr_block_within'])
                a_ = acc[w]; a_['bh'].append(bh); a_['nh'].append(nh); a_['ch'].append(ch); a_['rows'].append(rows)
            del model
        res['pooled'] = {}
        for w, a_ in acc.items():
            if not a_['rows']:
                continue
            rows = np.concatenate(a_['rows'])
            m = H.stage1_metrics(np.concatenate(a_['bh']), np.concatenate(a_['nh']), P['b'][rows], P['het4'][rows],
                                 logn[rows], np.concatenate(a_['ch']), np.stack([ct_all[int(j)] for j in rows]))
            res['pooled'][w] = dict(n=int(len(rows)), pc=m['pcorr_het_given_logN'], corr_logN=m['corr_logN'],
                                    W=m['corr_block_within'], stage1=m)
            print(f"EVAL {run} {w}: n {len(rows)} pc(het|logN) {m['pcorr_het_given_logN']:.4f} corr_logN "
                  f"{m['corr_logN']:.4f} W {m['corr_block_within']:.4f} (folds {sorted(res['per_fold'])}; "
                  f"{time.time() - t0:.0f}s)", flush=True)
        json.dump(res, open(d / ('score_fsim_smoke.json' if o.eval_smoke else 'score_fsim.json'), 'w'), indent=1,
                  default=float)


def selfcheck(o):
    import torch
    if o.repo not in sys.path:
        sys.path.insert(0, o.repo)
    from src import het_cnn as H0
    orig = H0.RenderSet
    H, cls = fsim_set.install(o.repo, 0.0, o.p_g1hi)
    assert issubclass(cls, orig) and H.RenderSet is cls
    rows = np.arange(6)
    kw = dict(cells=True)
    for aug in (None, dict(zoom=(1.0, 1.4), p_zoom=0.5, mosaic=0.5, mosaic_dlogn=0.5)):
        a0, a1 = orig(H.PREP_FP, rows, 2, 0, 1, aug=aug, **kw), cls(H.PREP_FP, rows, 2, 0, 1, aug=aug, **kw)
        for i in range(len(a0)):
            for t0, t1 in zip(a0[i], a1[i]):
                assert torch.equal(t0, t1), ('p_fsim 0 differs', i)
    fsim_set.FsimRenderSetBase.p_fsim = 1.0
    a1 = cls(H.PREP_FP, rows, 2, 0, 1, **kw)
    P = H.load_prep(H.PREP_FP)
    hp = []
    for i in range(len(a1)):
        it = a1[i]; j = int(rows[i % len(rows)])
        assert abs(float(it[3]) - float(P['het4'][j])) < 1e-6 and abs(float(it[4]) - float(np.log(P['n_eff'][j]))) < 1e-5
        im = it[0][0].numpy().astype(np.float32)
        hp.append(float(np.std(im[1:, :] - im[:-1, :]) / np.sqrt(2)))
    print(f'selfcheck passed: p_fsim 0 items bit-identical to RenderSet (plain + zoom/mosaic); p_fsim 1 items carry the '
          f'prep targets; fsim copies' + f" pixel-diff noise sd {np.mean(hp):.1f} (G1hi-like 12-20)", flush=True)


def train(o, rest):
    H, _ = fsim_set.install(o.repo, o.p_fsim, o.p_g1hi)
    if o.out_root:
        from pathlib import Path
        H.OUT_ROOT = Path(o.out_root)
    a = H.parse(rest)
    assert a.out, '--out RUN is required'
    d = H.OUT_ROOT / a.out
    d.mkdir(parents=True, exist_ok=True)
    want = dict(p_fsim=o.p_fsim, p_g1hi=o.p_g1hi, fsim_tag=fsim_set.FSIM_TAG, priors=['G1hi', 'midLF'])
    fp = d / 'fsim.json'
    if fp.exists():
        got = json.load(open(fp))
        if got != want:
            raise SystemExit(f'{fp}: cached fsim settings {got} != {want}; use another --out')
    else:
        json.dump(want, open(fp, 'w'), indent=1)
    print(f'het_fsim: p_fsim {o.p_fsim} p_g1hi {o.p_g1hi} -> {d}', flush=True)
    H.main(a)


if __name__ == '__main__':
    o, rest = own_args(sys.argv[1:])
    if o.spawn:
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
    if o.selfcheck:
        selfcheck(o)
    elif o.eval_fsim:
        eval_runs(o, rest)
    else:
        assert o.p_fsim is not None and 0.0 <= o.p_fsim <= 1.0, '--p-fsim P is required for training'
        train(o, rest)
