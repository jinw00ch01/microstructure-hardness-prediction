"""d1018 BOLD judge under the PROPOSED full-predictive order (main thread, 2026-10-18, d1018/AMEND_bold_fullpred_1018.txt).
Same inputs as judge_slots1018.py (each BUILD_DIR holds <option>/result.json + test.csv). For every option it restates the
builder's flag, then adds the full-predictive public model of pub/fullpred28.py (forensics LOO formula applied to the candidate;
sd x1.34) and the proposed BOLD class FP-BOLD: not SAFE, vs-BESTfixed OOF dMSE <= +0.42, latest kriging point <= best + 0.0132,
skip rule (G1 rms >= 0.25 vs every submitted or handed-over file), full-predictive sd >= 0.002 RMSE (public score not already
fixed by the readings). Order: SAFE (lowest kriging) first, then FP-BOLD by P(beat best) (within 0.01 -> better dMSE).
Label-free: test rows enter only through the candidate files; nothing is fitted on test rows.
usage: [KRIG=.../krigingNN.py] python3 -B judge_bold_fp1018.py BUILD_DIR [BUILD_DIR ...]"""
import io, contextlib, json, os, sys, glob
C = '/mnt/project-files/work/hardness-cache'
K = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.environ.get('KRIG', C + '/scripts/d1016/pub/kriging28.py')).read(), K)
np, pd, PUB, te, sig_t, T, Dm, o, Kr = K['np'], K['pd'], K['PUB'], K['te'], K['sig_t'], K['T'], K['Dm'], K['o'], K['K']
from scipy.stats import norm
f, n, N = 0.3, 300, 1000; t30 = T['v30b']; V30B = PUB['v30b']
Cs = -f * (1 - f) / (N - 1) * np.ones((N, N)); np.fill_diagonal(Cs, f * (1 - f))
Ki = np.linalg.inv(Kr)


def gp_full(x):
    u = x - t30; u2 = u ** 2
    kco = (u2 @ Cs @ (Dm ** 2).T + f * 4 * (u * sig_t ** 2) @ Dm.T) / n ** 2
    kcc = (u2 @ Cs @ u2 + f * 4 * np.sum(u2 * sig_t ** 2)) / n ** 2
    return float(np.mean(u2)) + float(kco @ Ki @ o), float(np.sqrt(max(float(kcc - kco @ Ki @ kco), 0.0)))


rows = []
for d in sys.argv[1:]:
    for fp in sorted(glob.glob(os.path.join(d, '*', 'result.json'))):
        r = json.load(open(fp))
        if r.get('kind') != 'option':
            continue
        best, bp = r['best_name'], r['best_pub']
        assert abs(PUB[best] - bp) < 1e-9, (best, bp)
        bold_kr = round(bp + 0.0132, 4)
        VB, kL, rs = r['vs_BESTfixed'], r['kriging_latest_public'], r['rms_G1t_vs_submitted']
        skip = min(rs.values()) < 0.25
        safe = VB['oof_dmse'] < 0 and VB['folds_better'] >= 3 and VB['resplit40']['P_ge0'] <= 0.5 and kL['rmse'] < bp
        bold = VB['oof_dmse'] <= 0.42 and kL['rmse'] <= bold_kr and rs[best] >= 0.5
        flag = 'SKIP(rms<0.25)' if skip else ('SAFE' if safe else ('BOLD' if bold else 'no'))
        assert flag == r['slate_flag'], (fp, flag, r['slate_flag'])
        x = pd.read_csv(os.path.join(os.path.dirname(fp), 'test.csv'), float_precision='round_trip').set_index('ID').hardness.loc[te].to_numpy(float)
        p, sd = gp_full(x); s = 1.34 * sd; fr = float(np.sqrt(V30B ** 2 + p))
        P = float(norm.cdf((bp ** 2 - V30B ** 2 - p) / s)) if s > 0 else float(p < bp ** 2 - V30B ** 2)
        sd_r = s / (2 * fr)
        fpb = (not safe) and VB['oof_dmse'] <= 0.42 and kL['rmse'] <= bold_kr and not skip and sd_r >= 0.002
        cls = 'SAFE' if (safe and not skip) else ('FP-BOLD' if fpb else ('fixed-by-readings' if (not skip and sd_r < 0.002) else flag))
        rows.append(dict(path=os.path.relpath(os.path.dirname(fp), C + '/scripts'), fams='+'.join(r['families']), cls=cls, flag=flag,
                         dmse=VB['oof_dmse'], folds=VB['folds_better'], dt3=VB['droptop3'], krig=kL['rmse'], full=fr, sd=sd_r, P=P,
                         rms_best=rs[best], min_sub=min(rs.values()), md5=r['test_md5'], best=best))
rank = {'SAFE': 0, 'FP-BOLD': 1}
rows.sort(key=lambda q: (rank.get(q['cls'], 9), round(q['krig'], 4) if q['cls'] == 'SAFE' else -round(q['P'], 2), q['dmse']))
for q in rows:
    print(f"{q['cls']:17s} {q['path']:44s} [{q['fams']}] (builder {q['flag']}) vs {q['best']} {q['dmse']:+.3f} {q['folds']}/5 dt3 {q['dt3']:+.3f} | "
          f"k {q['krig']:.4f} full {q['full']:.4f} sd {q['sd']:.4f} P(beat) {q['P']:.3f} | rms best {q['rms_best']:.3f} min-sub {q['min_sub']:.3f} | md5 {q['md5'][:12]}")
