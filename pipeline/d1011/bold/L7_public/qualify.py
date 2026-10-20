"""L7 qualify: which public lane a candidate file belongs to (train nested OOF only; no test file is read).

Usage (from the repo root):
  python qualify.py --base blend_v28n --cand /path/to/cand_oof.csv --rows coarse|G1hi|clean|LV|midLF|all|changed --M 1
  python qualify.py --backtest          # the lane each historical step would have taken, against its public reading

Quantities (train rows; test composition via fixed group counts where the row set is a group):
  f        share of rows the candidate changes (target set)
  c        rms prediction change on those rows (HV)
  MSE_R    base nested MSE on those rows
  ratio    null harm / paired public sd for a pure-noise change: f c^2 / (2 c sqrt(f MSE_R / 300)) = 8.66 c sqrt(f / MSE_R)
           (a null that is a weaker but informative model, q 0.3, has 0.4 x this ratio)
  delta    nested all-500 dRMSE (candidate minus base) at the private level (x 1/22 of dMSE)
  sd_tr    sd of that estimate (500 rows);  sd_pub = sd_tr / 0.775
  bias     forking discount 0.7 * E[max of M std normals] * sd_tr  (M = variants looked at on these labels)
Strict gate (s5; required for every submitted file): delta + bias <= -1.0 sd_tr  -> strict_gate_pass
Lanes (size decides the process, not the bar):
  NO        delta + bias > +0.66 sd_tr (the auto-final break-even, s2 A)
  FAST      ratio >= 2.5 and not NO: big change; if strict_gate_pass, submit at full strength the same day, no judge round
  PIPELINE  ratio < 1.0: small change; needs the usual pre-registered PASS (honest delta + bias < 0)
  DANGER    1.0 <= ratio < 2.5: a null slips in with P 0.6-16%; only with a pre-registered PASS, never as a hedge
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, json, argparse, numpy as np, pandas as pd
sys.dont_write_bytecode = True
REPO = '/home/claude/microstructure-hardness-prediction'
sys.path.insert(0, REPO)
from src.common import load_train, DATA_DIR
SUB = '/mnt/project-files/work/hardness-cache/submissions'
OUT = '/mnt/project-files/work/hardness-cache/scripts/d1011/bold/L7_public'
EMAX = {1: 0.0, 2: 0.564, 3: 0.846, 4: 1.029, 5: 1.163, 10: 1.539, 20: 1.867, 30: 2.043, 50: 2.249, 100: 2.508, 128: 2.58}

tr = load_train(); y = tr.hardness.values; ids = tr.ID.values
v3 = pd.read_parquet(DATA_DIR / 'features_v3.parquet').set_index('ID')
nz = v3.loc[ids, 'ic_noise'].values; rs = v3.loc[ids, 'ic_ridge_snr'].values
G = np.full(500, 'G1hi', dtype=object)
G[nz < 9.5] = 'clean'; G[(nz >= 9.5) & (nz < 12) & (rs >= 0.5)] = 'LV'; G[(nz >= 9.5) & (nz < 12) & (rs < 0.5)] = 'midLF'
G[(nz >= 12) & (rs >= 0.5)] = 'HLV'
dens = pd.read_parquet(DATA_DIR / 'features_cal.parquet').set_index('ID').loc[ids, 'cal_seg_count_density'].values
coarse = dens <= np.quantile(dens, 1 / 3)

def rd(x):
    path = x if x.endswith('.csv') else f'{SUB}/{x}_oof.csv'
    p = pd.DataFrame({'ID': ids}).merge(pd.read_csv(path), on='ID', how='left').hardness.values
    assert np.isfinite(p).all(); return p

def emax(M):
    k = max(m for m in EMAX if m <= M); return EMAX[k]

def lane(pb, pc, rows='changed', M=1):
    dp = pc - pb
    if rows == 'changed': m = np.abs(dp) > 1e-9
    elif rows == 'coarse': m = coarse
    elif rows == 'all': m = np.ones(500, bool)
    else: m = G == rows
    f = m.mean(); c = float(np.sqrt(np.mean(dp[m] ** 2))); mse_r = float(np.mean((y - pb)[m] ** 2))
    ratio = 8.66 * c * np.sqrt(f / mse_r)
    d = (y - pc) ** 2 - (y - pb) ** 2
    delta = d.mean() / 22.0; sd_tr = d.std() / np.sqrt(500) / 22.0; bias = 0.7 * emax(M) * sd_tr
    if delta + bias > 0.66 * sd_tr: L = 'NO'
    elif ratio >= 2.5: L = 'FAST'
    elif ratio < 1.0: L = 'PIPELINE'
    else: L = 'DANGER'
    strict = bool(delta + bias <= -1.0 * sd_tr)      # the one gate every submitted file must pass (s5)
    return dict(rows=int(m.sum()), f=float(f), c=c, mse_r=mse_r, ratio=float(ratio), ratio_q03=float(0.4 * ratio), delta=float(delta),
                sd_tr=float(sd_tr), sd_pub=float(sd_tr / 0.775), bias=float(bias), M=M, lane=L, strict_gate_pass=strict)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--base', default='blend_v28n'); ap.add_argument('--cand')
    ap.add_argument('--rows', default='changed'); ap.add_argument('--M', type=int, default=1); ap.add_argument('--backtest', action='store_true')
    a = ap.parse_args()
    if not a.backtest:
        print(json.dumps(lane(rd(a.base), rd(a.cand), a.rows, a.M), indent=1)); sys.exit()
    PUB = dict(blend_v18=11.7304, blend_v19=11.0751, blend_v20=11.6924, blend_v21=11.0620, blend_v22=10.9742, blend_v24=10.5130,
               blend_v25=10.2394, blend_v26L=10.1302, blend_v26r=10.2136, blend_v27u=10.1041, blend_v28n=10.0712, blend_v28p=10.0796)
    # M = variants looked at on the same labels before the file (rough, from the record; 1 = pre-registered single test)
    steps = [('blend_v18', 'blend_v19', 20, 'het offset on clean rows (label-scanned form)'),
             ('blend_v18', 'blend_v20', 1, 'cell CNN share 0.8 probe'),
             ('blend_v21', 'blend_v22', 2, 'GPU het-CNN offset, ic_noise >= 12'),
             ('blend_v22', 'blend_v24', 1, 'localized het CNN on G1 (pre-registered; v23 gate inside)'),
             ('blend_v24', 'blend_v25', 10, 'formula7 on clean rows (label-explored)'),
             ('blend_v25', 'blend_v26L', 1, 'formula7 on LV rows (pre-registered)'),
             ('blend_v25', 'blend_v26r', 128, 'ringall on clean rows (post hoc, 128 candidates)'),
             ('blend_v27u', 'blend_v28n', 4, 'G1hi low-half stretch (post hoc)'),
             ('blend_v27u', 'blend_v28p', 4, 'G1hi pooled stretch')]
    res = []
    print('step                 lane      strict rows  f     c(HV)  ratio  delta    sd_tr  bias   | public dRMSE  (z vs delta, sd_pub)')
    for b, c, M, note in steps:
        r = lane(rd(b), rd(c), 'changed', M); r.update(step=f'{b[6:]}->{c[6:]}', note=note)
        dpub = PUB[c] - PUB[b]; r['public_drmse'] = dpub
        # public dRMSE at public level vs delta at private level: compare in MSE
        dpub_mse = PUB[c] ** 2 - PUB[b] ** 2; r['z'] = float((dpub_mse - r['delta'] * 22) / (r['sd_pub'] * 22))
        res.append(r)
        print(f"{r['step']:20s} {r['lane']:8s} {str(r['strict_gate_pass']):6s} {r['rows']:4d}  {r['f']:.2f}  {r['c']:5.2f}  {r['ratio']:5.2f}  {r['delta']:+.3f}  "
              f"{r['sd_tr']:.3f}  {r['bias']:.3f}  | {dpub:+.4f}  ({r['z']:+.2f})   {note}")
    # what the lanes would do for hypothetical coarse-third and G1hi candidates of a given size (null = pure noise)
    print('\nminimum rms change for ratio 2.5 / 3.0 (FAST lane, null slips in with P 0.6% / 0.1%):')
    e = y - rd('blend_v28n'); hyp = {}
    for nm, m in [('coarse third', coarse), ('G1hi', G == 'G1hi'), ('clean', G == 'clean'), ('all rows', np.ones(500, bool))]:
        f = m.mean(); mse_r = float(np.mean(e[m] ** 2))
        c25 = 2.5 / (8.66 * np.sqrt(f / mse_r)); c30 = 3.0 / (8.66 * np.sqrt(f / mse_r))
        gain_q08 = -0.6 * f * c25 ** 2 / 22
        hyp[nm] = dict(f=float(f), mse_r=mse_r, c_ratio25=float(c25), c_ratio30=float(c30), gain_if_real_q08_at_c25=float(gain_q08))
        print(f'  {nm:12s} f {f:.2f} MSE_R {mse_r:6.1f}: c >= {c25:.1f} / {c30:.1f} HV; a real lever (q 0.8) of that size is worth {gain_q08:+.3f} RMSE')
    json.dump(dict(backtest=res, fast_lane_sizes=hyp), open(f'{OUT}/qualify_backtest.json', 'w'), indent=1)
