"""forensics18 = forensics17 plus the blend_v35xn reading (user, 2026-10-16 14:51 KST; main thread). forensics17 = forensics16 plus the blend_v34l2 reading (user, 2026-10-16 14:43 KST; main thread). Was: d1016 copy of d1015/g1mix/forensics15.py with the four 10-15 readings added (main thread, 2026-10-15). N6 / pub lens: public-reading forensics (label-free; never reads data/train.csv hardness or any residual built from it).
Inputs: submitted test files (cache submissions/blend_*.csv), public readings (CONTEXT.txt), features_v3 noise columns,
reader dumps data/het_cnn/*/het_cnn_{train,test}.parquet (label-free CNN outputs), and builder scalars printed in
N5/g1_shrink_verify/verify_result.json (beta, cap, stretch c/k: already-public summary numbers of the v30b / v31s rule).
Nothing here fits anything on test rows; test statistics are report-only.

Part 1  reading table: for each submitted file B and the file A it was built on, d = B - A on the 1000 test rows,
        m2 = mean_1000(d^2) (null expectation of the public dMSE), sd of the public dMSE under a random 300-row public set
        (residual sd per group from the v30b error budget in CONTEXT.txt), z = (obs - m2) / sd and the public step
        lambda* = -(obs - m2) / (2 m2) (1 = the step was right, 0 = no value, < 0 = wrong sign) with its SE.
Part 2  Gaussian-process (kriging) reading of all 16 readings jointly: o_j = (pub_j^2 - pub_30b^2) - m2_j is linear in the
        unknown public residual e30 of v30b; prior e_i ~ (0, sigma_g^2) independent, public set random 30%. Posterior of
        h_U = mean_P(U e30) for named pieces U -> public-implied step for each piece, information share.
Part 3  label-free tests of the v31s hypotheses (a) reliability / errors-in-variables, (b) train/test shift on G1,
        (c) clipping, (d) stretch base.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import json, sys
import numpy as np, pandas as pd
from scipy import stats

REPO = '/home/claude/microstructure-hardness-prediction'
CACHE = '/mnt/project-files/work/hardness-cache'
SUB = CACHE + '/submissions'
OUT = '/mnt/project-files/work/hardness-cache/scripts/d1016/pub/kriging_out18'
HD = REPO + '/data/het_cnn/'

PUB = dict(v19=11.0751, v21=11.0620, v22=10.9742, v23=10.8445, v23m15=10.8393, v24=10.5130, v25=10.2394,
           v26L=10.1302, v26r=10.2136, v27u=10.1041, v28n=10.0712, v28p=10.0796, v29g=10.0564,
           v30b=10.0206885658, v30a=10.0312450482, v31s=10.1953136309, v31f=10.1704996257, v32m=10.0298274485,
           v33xb=9.9868332159, v33t3=9.9921641853, v33s5=10.0031466453, v33b4=9.9960660597,  # 10-15 readings (user, 12:56 KST)
           v34l2=10.0054603343,  # 10-15 slot 5 (g1mix4 B6x3 L2), user 10-16 14:43 KST
           v35xn=9.9721726754)  # 10-16 slot 1 (g1mix4 XN), user 10-16 14:51 KST
BUILT_ON = dict(v21='v19', v22='v21', v23='v22', v23m15='v23', v24='v23', v25='v24', v26L='v25', v26r='v25',
                v27u='v26L', v28n='v27u', v28p='v27u', v29g='v28n', v30b='v29g', v30a='v30b', v31s='v30b', v31f='v31s', v32m='v30b',
                v33xb='v30b', v33t3='v30b', v33s5='v30b', v33b4='v30b', v34l2='v30b', v35xn='v30b')
SIG = dict(clean=8.36, LV=9.50, midLF=10.45, HLV=7.63, G1hi=13.48)   # v30b nested RMSE by group (CONTEXT.txt)
STRATA = ('clean', 'LV', 'midLF', 'HLV', 'G1hi')
NPUB, NTE = 300, 1000
f = 0.3


def groups(nz, rs):
    return dict(clean=nz < 9.5, LV=(nz >= 9.5) & (nz < 12) & (rs >= 0.5), midLF=(nz >= 9.5) & (nz < 12) & (rs < 0.5),
                HLV=(nz >= 12) & (rs >= 0.5), G1hi=(nz >= 12) & (rs < 0.5))


ids = pd.read_csv(REPO + '/data/folds.csv', usecols=['ID']).ID.astype(str).to_numpy()
te = pd.read_csv(REPO + '/data/sample_submission.csv', usecols=['ID']).ID.astype(str).to_numpy()
v3 = pd.read_parquet(REPO + '/data/features_v3.parquet', columns=['ID', 'ic_noise', 'ic_ridge_snr']).set_index('ID')
nz, rs = v3.loc[ids, 'ic_noise'].to_numpy(float), v3.loc[ids, 'ic_ridge_snr'].to_numpy(float)
nzt, rst = v3.loc[te, 'ic_noise'].to_numpy(float), v3.loc[te, 'ic_ridge_snr'].to_numpy(float)
grp, grpt = groups(nz, rs), groups(nzt, rst)
G1, G1t = grp['midLF'] | grp['G1hi'], grpt['midLF'] | grpt['G1hi']
assert G1.sum() == 208 and G1t.sum() == 433
sig_t = np.zeros(NTE)
for g in STRATA:
    sig_t[grpt[g]] = SIG[g]

T = {n: pd.read_csv(f'{SUB}/blend_{n}.csv', float_precision='round_trip').set_index('ID').hardness.loc[te].to_numpy(float)
     for n in PUB}
V = json.load(open(CACHE + '/scripts/d1011/N5/g1_shrink_verify/verify_result.json'))
i1, iS = V['test_lam1_info'], V['test_info']
M_HALF = i1['m']
st_t = grpt['G1hi'] & (nzt < M_HALF)                     # G1hi low-noise half: stretch rows (167)
ns_t = G1t & ~st_t                                        # other G1 rows (266)
tcp_t = G1t & (np.abs(T['v29g'] - T['v28n']) > 1e-9)      # T-CP rows (w 0.5), as in N5 magnitude.py
w_t = np.where(tcp_t, 0.5, 1.0)
assert st_t.sum() == 167 and tcp_t.sum() == 19
BLK = dict(clean=grpt['clean'], LV=grpt['LV'], HLV=grpt['HLV'], midLF=grpt['midLF'],
           G1hi_ns=grpt['G1hi'] & ~st_t, G1hi_st=st_t)

R = {}
P = lambda *a: print(*a, flush=True)

# ------------------------------------------------------------------ Part 1: reading table
def null_sd(d):
    """sd of mean_P(d^2 + 2 d e) - m2 over a random 300-of-1000 public set and e_i ~ (0, sig_i^2)."""
    x = d ** 2
    v_s = np.var(x, ddof=1) / NPUB * (1 - f)   # finite-population variance of a 300-of-1000 mean
    v_e = f * np.sum(4 * d ** 2 * sig_t ** 2) / NPUB ** 2
    return float(np.sqrt(v_s + v_e)), float(np.sqrt(v_e))


rows = []
for b, a in BUILT_ON.items():
    d = T[b] - T[a]; m2 = float(np.mean(d ** 2)); obs = PUB[b] ** 2 - PUB[a] ** 2
    sd, sde = null_sd(d)
    lam = -(obs - m2) / (2 * m2) if m2 > 0 else np.nan
    blk = {k: dict(n=int(((d != 0) & m).sum()), share_m2=float(np.sum(d[m] ** 2) / max(np.sum(d ** 2), 1e-12)),
                   share_var=float(np.sum((d[m] * sig_t[m]) ** 2) / max(np.sum((d * sig_t) ** 2), 1e-12)))
           for k, m in BLK.items()}
    rows.append(dict(file=b, base=a, n_changed=int((np.abs(d) > 1e-9).sum()), rms_changed=float(np.sqrt(np.mean(d[np.abs(d) > 1e-9] ** 2))) if (np.abs(d) > 1e-9).any() else 0.0,
                     obs_dmse=obs, null_m2=m2, sd=sd, z_vs_null=(obs - m2) / sd, lam_star=lam, lam_se=sd / (2 * m2),
                     info_snr=m2 / sd, blocks=blk))
R['readings'] = rows
sel = [r for r in rows if r['file'] not in ('v31s', 'v27u')]
wv = np.array([1 / r['lam_se'] ** 2 for r in sel]); lv = np.array([r['lam_star'] for r in sel])
R['lam_star_ivw_excl_v31s'] = dict(mean=float(np.sum(wv * lv) / wv.sum()), se=float(1 / np.sqrt(wv.sum())),
                                   chi2_dof=float(np.sum(wv * (lv - np.sum(wv * lv) / wv.sum()) ** 2) / (len(lv) - 1)))
R['lam_star_ivw_since_v29g'] = (lambda ss: dict(mean=float(sum(r['lam_star'] / r['lam_se'] ** 2 for r in ss) / sum(1 / r['lam_se'] ** 2 for r in ss)),
                                se=float(1 / np.sqrt(sum(1 / r['lam_se'] ** 2 for r in ss)))))([r for r in rows if r['file'] in ('v29g', 'v30b', 'v30a', 'v31f')])
P('Part 1: public readings (dMSE = pub_B^2 - pub_A^2; m2 = null expectation; lambda* = public-optimal multiple of the step)')
P(f"{'file':7s} {'base':6s} {'nch':>4s} {'rms':>5s} {'obs':>7s} {'m2':>6s} {'sd':>5s} {'z':>5s} {'lam*':>6s} {'se':>5s} {'snr':>5s}  main blocks (share of sigma^2-weighted change)")
for r in rows:
    top = sorted(r['blocks'].items(), key=lambda kv: -kv[1]['share_var'])[:3]
    P(f"{r['file']:7s} {r['base']:6s} {r['n_changed']:4d} {r['rms_changed']:5.2f} {r['obs_dmse']:+7.3f} {r['null_m2']:6.3f} {r['sd']:5.2f} {r['z_vs_null']:+5.2f} {r['lam_star']:+6.2f} {r['lam_se']:5.2f} {r['info_snr']:5.2f}  "
      + ' '.join(f"{k}:{v['share_var']:.2f}" for k, v in top))
P('inverse-variance mean lambda* (all readings but v31s, v27u):', R['lam_star_ivw_excl_v31s'], ' since v29g (v29g, v30b, v30a, v31f):', R['lam_star_ivw_since_v29g'])

# ------------------------------------------------------------------ pieces (directions on test rows)
tr5 = lambda d, s: pd.read_parquet(f'{HD}{d}/het_cnn_{s}.parquet').set_index('ID')
A3tr, A3te = tr5('ev2s_locK_real5', 'train').loc[ids], tr5('ev2s_locK_real5', 'test').loc[te]
gA3 = np.polyfit(A3tr.logN_cnn.values[G1], A3tr.het_cnn.values[G1], 1)
devA3_t = A3te.het_cnn.values - np.polyval(gA3, A3te.logN_cnn.values)
devA3_tr = A3tr.het_cnn.values - np.polyval(gA3, A3tr.logN_cnn.values)
off1 = i1['beta'] * devA3_t; off1c = np.clip(off1, -i1['cap'], i1['cap'])
t21 = pd.read_csv(f'{SUB}/blend_v21.csv', float_precision='round_trip').set_index('ID').hardness.loc[te].to_numpy(float)
k1, c1, kS, cS = i1['k'], i1['c'], iS['k'], iS['c']
# exact rebuild of the v31s - v30b change from the pieces
old = np.where(st_t, c1 + k1 * (t21 + off1c - c1), t21 + off1c)
d31s = T['v31s'] - T['v30b']
recon = np.zeros(NTE)
recon[ns_t] = (-0.3 * w_t * off1c)[ns_t]
new_st = cS + kS * (t21 + 0.7 * off1c - cS)
recon[st_t] = (w_t * (new_st - old))[st_t]
R['rebuild_v31s_maxabs'] = float(np.abs(recon - d31s).max())
P('\nrebuild of v31s - v30b from pieces: max|diff|', R['rebuild_v31s_maxabs'])

z = np.zeros(NTE)
def piece(mask, v):
    u = z.copy(); u[mask] = v[mask]; return u

U = {}
U['A3off_all'] = piece(G1t, w_t * np.where(st_t, k1, 1.0) * off1c)            # +1 = double the effective A3 offset
U['A3off_ns'] = piece(ns_t, w_t * off1c)
U['A3off_ns_midLF'] = piece(ns_t & grpt['midLF'], w_t * off1c)
U['A3off_ns_G1hi'] = piece(ns_t & grpt['G1hi'], w_t * off1c)
U['A3off_st'] = piece(st_t, w_t * k1 * off1c)
U['stretch_dev'] = piece(st_t, w_t * (t21 + off1c - c1))                    # +1 = k up by 1 on the stretch rows
U['v31s_shrink_only'] = piece(G1t, -0.3 * w_t * np.where(st_t, kS, 1.0) * off1c)
U['v31s_stretch_only'] = piece(st_t, w_t * ((kS - k1) * (t21 + off1c - c1) + (1 - kS) * (cS - c1)))
U['v31s'] = d31s
U['v30b_minus_v29g'] = T['v30b'] - T['v29g']
U['v30b_minus_v29g_G1hi'] = piece(grpt['G1hi'], T['v30b'] - T['v29g'])
U['v30b_minus_v29g_midLF'] = piece(grpt['midLF'], T['v30b'] - T['v29g'])
U['v31f_minus_v31s'] = T['v31f'] - T['v31s']
U['v29g_minus_v28n'] = T['v29g'] - T['v28n']
U['v30a_minus_v30b(C)'] = T['v30a'] - T['v30b']
U['v28n_minus_v27u'] = T['v28n'] - T['v27u']
U['v28p_minus_v27u'] = T['v28p'] - T['v27u']
U['CNNshare_proxy(v21-v19 G1)'] = piece(G1t, T['v21'] - T['v19'])
dsh, dst = U['v31s_shrink_only'], U['v31s_stretch_only']
R['v31s_decomp'] = dict(m2_total=float(np.mean(d31s ** 2)), m2_shrink=float(np.mean(dsh ** 2)), m2_stretch=float(np.mean(dst ** 2)),
                        m2_cross=float(2 * np.mean(dsh * dst)), max_resid=float(np.abs(dsh + dst - d31s).max()),
                        sig2w_share_shrink=float(np.sum((dsh * sig_t) ** 2) / np.sum((d31s * sig_t) ** 2)),
                        m2_ns=float(np.mean(d31s[ns_t] ** 2) * ns_t.mean()), m2_st=float(np.mean(d31s[st_t] ** 2) * st_t.mean()))
P('v31s decomposition', json.dumps({k: round(v, 4) for k, v in R['v31s_decomp'].items()}))

# ------------------------------------------------------------------ Part 2: GP / kriging over all readings (relative to v30b)
names = [n for n in PUB if n != 'v30b']
Dm = np.array([T[n] - T['v30b'] for n in names])                 # (J, 1000)
o = np.array([PUB[n] ** 2 - PUB['v30b'] ** 2 - np.mean((T[n] - T['v30b']) ** 2) for n in names])
X2 = Dm ** 2
Cs = -f * (1 - f) / (NTE - 1) * np.ones((NTE, NTE)); np.fill_diagonal(Cs, f * (1 - f))
K = (X2 @ Cs @ X2.T + f * 4 * (Dm * sig_t ** 2) @ Dm.T) / NPUB ** 2
# readings are rounded to 1e-4 RMSE (dMSE rounding sd ~ 2*10*5e-5/sqrt(3) ~ 6e-4); v27u-v26L == v26r-v25 on disjoint rows makes K
# near-singular without it
K = K + np.eye(len(names)) * (6e-4 ** 2 * 2)
ev = np.linalg.eigvalsh(K); print('K eigen min/max', ev.min(), ev.max())


def krig(u, use=None):
    idx = np.arange(len(names)) if use is None else np.array(use)
    Ki = np.linalg.inv(K[np.ix_(idx, idx)])
    kuo = f * 2 * (Dm[idx] * sig_t ** 2) @ u / NPUB ** 2
    kuu = f * np.sum(u ** 2 * sig_t ** 2) / NPUB ** 2
    mu = float(kuo @ Ki @ o[idx]); var = float(kuu - kuo @ Ki @ kuo)
    m2 = float(np.mean(u ** 2))
    return dict(h_post=mu, h_sd_post=float(np.sqrt(max(var, 0))), h_sd_prior=float(np.sqrt(kuu)),
                info_share=float(1 - max(var, 0) / kuu), m2=m2, lam_pub=float(-mu / m2) if m2 > 0 else np.nan,
                lam_sampling_se=float(np.sqrt(kuu) / m2) if m2 > 0 else np.nan,
                lam_pub_sd=float(np.sqrt(max(var, 0)) / m2) if m2 > 0 else np.nan)


P('\nPart 2: kriging of the public residual of v30b onto pieces (all 16 readings; prior mean 0 = null)')
P('  lam_pub = public-implied step along the piece (0 = no value; +1 = add the piece once more; for an existing offset, scale = 1 + lam_pub)')
R['krig_all'] = {}
for n, u in U.items():
    r = krig(u); R['krig_all'][n] = r
    P(f"  {n:28s} m2 {r['m2']:7.3f}  lam_pub {r['lam_pub']:+6.2f} +- {r['lam_pub_sd']:5.2f}  info {r['info_share']:.2f}  public-vs-population sampling se {r['lam_sampling_se']:.2f}")
# leave-v31s/v31f-out: would the public history before 10-11 20:00 have predicted the v31s sign?
pre = [i for i, n in enumerate(names) if n not in ('v31s', 'v31f')]
R['krig_pre_v31'] = {}
P('\n  readings before v31s/v31f only (what public said about the A3 offset scale before v31s was sent):')
for n in ('A3off_all', 'A3off_ns', 'A3off_st', 'stretch_dev', 'v31s', 'v31s_shrink_only', 'v31s_stretch_only'):
    r = krig(U[n], pre); R['krig_pre_v31'][n] = r
    P(f"  {n:28s} lam_pub {r['lam_pub']:+6.2f} +- {r['lam_pub_sd']:5.2f}  info {r['info_share']:.2f}")
# predictive check of each reading from the others (standardised)
loo = {}
for j, n in enumerate(names):
    idx = [i for i in range(len(names)) if i != j]
    Ki = np.linalg.inv(K[np.ix_(idx, idx)])
    mu = float(K[j, idx] @ Ki @ o[idx]); var = float(K[j, j] - K[j, idx] @ Ki @ K[idx, j])
    loo[n] = dict(obs=float(o[j]), pred=mu, sd=float(np.sqrt(var)), z=float((o[j] - mu) / np.sqrt(var)))
R['loo'] = loo
P('\n  leave-one-reading-out prediction of o_j = dMSE_vs_v30b - m2 (standardised):')
P('  ' + ', '.join(f"{n} {v['z']:+.2f}" for n, v in loo.items()))

