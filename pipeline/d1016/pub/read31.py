"""blend_v40b8a4 (9.9433362872) and blend_v40q8an4b (9.9572738847) readings (user 2026-10-19 01:08 KST; 10-19 files 2-3) vs their
kriging30 and full-predictive predictions; v40b8a4 - v39q8a4b public MSE; LOO z under kriging31. Main thread 2026-10-19 01:09 KST, from read30.py."""
import io, contextlib
from scipy.stats import norm
C = '/mnt/project-files/work/hardness-cache'
K29, K30 = {}, {}  # here K29 = kriging30 (predictions), K30 = kriging31 (LOO)
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(C + '/scripts/d1016/pub/kriging30.py').read(), K29)
    exec(open(C + '/scripts/d1016/pub/kriging31.py').read(), K30)
np, pd, PUB, te = K30['np'], K30['pd'], K30['PUB'], K30['te']
rd = lambda p: pd.read_csv(p, float_precision='round_trip').set_index('ID').hardness.loc[te].to_numpy(float)
t30 = rd(f'{C}/submissions/blend_v30b.csv')
f, n, N = 0.3, 300, 1000
Cs = -f * (1 - f) / (N - 1) * np.ones((N, N)); np.fill_diagonal(Cs, f * (1 - f))
Dm, o, Kr, sig_t = K29['Dm'], K29['o'], K29['K'], K29['sig_t']
def gp_full(x):
    u = x - t30; u2 = u ** 2
    kco = (u2 @ Cs @ (Dm ** 2).T + f * 4 * (u * sig_t ** 2) @ Dm.T) / n ** 2
    kcc = (u2 @ Cs @ u2 + f * 4 * np.sum(u2 * sig_t ** 2)) / n ** 2
    return float(np.mean(u2)) + float(kco @ np.linalg.inv(Kr) @ o), float(np.sqrt(max(float(kcc - kco @ np.linalg.inv(Kr) @ kco), 0.0)))
for nm in ('v40b8a4', 'v40q8an4b'):
    x = rd(f'{C}/submissions/blend_{nm}.csv')
    obs = PUB[nm] ** 2 - PUB['v30b'] ** 2
    k = K29['kpred'](x - t30)
    p, sd = gp_full(x)
    print(f"{nm}: observed public dMSE vs v30b {obs:+.3f} (RMSE {PUB[nm]:.4f}); kriging30 predicted {k['pred']:+.3f} sd {k['sd']:.3f} "
          f"(x1.34 {k['sd_x134']:.3f}) -> RMSE {k['rmse']:.4f}, z {(obs - k['pred']) / k['sd_x134']:+.2f}")
    print(f"{nm}: full predictive (kriging30) {p:+.3f} sd {sd:.3f} -> RMSE {np.sqrt(PUB['v30b'] ** 2 + p):.4f}, z {(obs - p) / sd:+.2f} "
          f"(x1.34 z {(obs - p) / (1.34 * sd):+.2f}); P(beat v39q8a4b) had been {norm.cdf((PUB['v39q8a4b'] ** 2 - PUB['v30b'] ** 2 - p) / sd):.3f} at x1.0")
print(f"v40b8a4 - v39q8a4b public MSE {PUB['v40b8a4'] ** 2 - PUB['v39q8a4b'] ** 2:+.4f}; v40q8an4b - v39q8a4b {PUB['v40q8an4b'] ** 2 - PUB['v39q8a4b'] ** 2:+.4f}")
print('LOO z (kriging31):', ', '.join(f"{nm} {v['z']:+.2f}" for nm, v in K30['loo'].items() if nm.startswith('v3') or nm.startswith('v4')))
