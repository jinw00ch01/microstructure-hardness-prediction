"""fullpred28 (main thread, 2026-10-18; from the verify-p3b-1018 pubmodel verifier gp_full). Full-predictive public model
(forensics28 LOO formula applied to a candidate: includes the candidate's own
mean_P(u^2) sampling term and its covariance with the reading noise). Label-free. Prints per option row:
kriging28 point/P(beat) as the judge shows it, and the full predictive RMSE, sd (x1.34) and P(beat best).
usage: [KRIG=.../krigingNN.py] [BEST=v39q3a6] [OUTJ=out.json] python3 -B fullpred28.py <build dir> [<build dir> ...]
Label-free: reads submitted test files, readings and candidate test.csv; fits nothing on test rows; prints summaries only."""
import io, contextlib, sys, json, os, glob
C = '/mnt/project-files/work/hardness-cache'
K = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.environ.get('KRIG', C + '/scripts/d1016/pub/kriging28.py')).read(), K)
np, pd, PUB, te, sig_t, G1t, names = K['np'], K['pd'], K['PUB'], K['te'], K['sig_t'], K['G1t'], K['names']
from scipy.stats import norm
T = K['T']; rd = lambda p: pd.read_csv(p, float_precision='round_trip').set_index('ID').hardness.loc[te].to_numpy(float)
t30 = T['v30b']; f, n, N = 0.3, 300, 1000
BEST = os.environ.get('BEST', 'v39q3a6')
obsQ = PUB[BEST] ** 2 - PUB['v30b'] ** 2
Cs = -f * (1 - f) / (N - 1) * np.ones((N, N)); np.fill_diagonal(Cs, f * (1 - f))
Dm, o, Kr = K['Dm'], K['o'], K['K']
Ki_all = np.linalg.inv(Kr)
def gp_full(x):
    u = x - t30; D = Dm; X2 = D ** 2; u2 = u ** 2
    kco = (u2 @ Cs @ X2.T + f * 4 * (u * sig_t ** 2) @ D.T) / n ** 2
    kcc = (u2 @ Cs @ u2 + f * 4 * np.sum(u ** 2 * sig_t ** 2)) / n ** 2
    mu = float(kco @ Ki_all @ o); var = float(kcc - kco @ Ki_all @ kco)
    p = float(np.mean(u2)) + mu
    return p, np.sqrt(max(var, 0))
def eimp(p, sd):  # E[(obsQ - X)+] in MSE units, X ~ N(p, sd)
    if sd <= 0: return max(obsQ - p, 0.0)
    z = (obsQ - p) / sd; return float(sd * (norm.pdf(z) + z * norm.cdf(z)))
dirs = sys.argv[1:]
rows = []
for d in dirs:
    for rj in sorted(glob.glob(d.rstrip('/') + '/*/result.json')):
        od = os.path.dirname(rj); r = json.load(open(rj))
        if r.get('kind') != 'option': continue
        x = rd(od + '/test.csv'); k = K['kpred'](x - t30); p, sd = gp_full(x)
        s = 1.34 * sd
        rows.append(dict(dir=od.replace(C + '/scripts/', ''), fams='+'.join(r.get('families', [])), k28=k['rmse'],
                         kP=float(norm.cdf((obsQ - k['pred']) / k['sd_x134'])), full=float(np.sqrt(PUB['v30b'] ** 2 + p)),
                         full_sd_rmse=float(s / (2 * np.sqrt(PUB['v30b'] ** 2 + p))), fP=float(norm.cdf((obsQ - p) / s)),
                         eimp_mse=eimp(p, s)))
rows.sort(key=lambda q: -q['fP'])
for q in rows:
    print(f"{q['dir']:48s} [{q['fams']:32s}] k28 {q['k28']:.4f} kP {q['kP']:.2f} | full {q['full']:.4f} sd {q['full_sd_rmse']:.4f} P(beat {BEST}) {q['fP']:.3f} E[gain] {q['eimp_mse']:.4f} MSE")
json.dump(rows, open(os.environ.get('OUTJ', '/dev/null'), 'w'), indent=1)
