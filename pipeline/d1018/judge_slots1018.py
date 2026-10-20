"""d1018 slot judge (main thread, written 2026-10-18 02:16 KST, before any 3-seed option of g1mix9..g1mix12 exists).
Reads the result.json of every option in the build dirs given and lists them in the d1017/NOTE_slate1018.txt order:
SAFE, then BOLD (lowest latest kriging), then the proposed NEAR-TIE class of d1018/NOTE_neartie1018.txt with the
clarifications of d1018/DISCLOSURE_neartie1018.txt (lenient = the note's conditions; strict = also 40-split mean < 0 and
drop-top-3 < 0 vs BESTfixed). It only restates those rules; the builders' own SAFE/BOLD/SKIP flags are taken as they are and
re-derived here as a check. NEAR-TIE is reported for information until JINWOO picks it on the 10-18 card.
usage: python3 -B judge_slots1018.py BUILD_DIR [BUILD_DIR ...]   (each BUILD_DIR holds <option>/result.json)"""
import json, os, sys, glob, math

V30B = 10.0206885658


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


rows = []
for d in sys.argv[1:]:
    for fp in sorted(glob.glob(os.path.join(d, '*', 'result.json'))):
        r = json.load(open(fp))
        if r.get('kind') != 'option':
            continue
        best, bp = r['best_name'], r['best_pub']
        bold_kr = round(bp + 0.0132, 4)
        VB, kL, rs = r['vs_BESTfixed'], r['kriging_latest_public'], r['rms_G1t_vs_submitted']
        skip = min(rs.values()) < 0.25
        safe = VB['oof_dmse'] < 0 and VB['folds_better'] >= 3 and VB['resplit40']['P_ge0'] <= 0.5 and kL['rmse'] < bp
        bold = VB['oof_dmse'] <= 0.42 and kL['rmse'] <= bold_kr and rs[best] >= 0.5
        flag = 'SKIP(rms<0.25)' if skip else ('SAFE' if safe else ('BOLD' if bold else 'no'))
        assert flag == r['slate_flag'], (fp, flag, r['slate_flag'])
        near = (flag == 'no' and VB['oof_dmse'] < 0 and VB['folds_better'] >= 3 and VB['resplit40']['P_ge0'] <= 0.5
                and kL['rmse'] <= bold_kr)
        strict = near and VB['resplit40']['mean'] < 0 and VB['droptop3'] < 0
        p_beat = phi((bp ** 2 - V30B ** 2 - kL['pred']) / kL['sd_x134'])
        cls = flag if flag != 'no' else ('NEAR-TIE(strict)' if strict else ('NEAR-TIE' if near else 'no'))
        rows.append(dict(build=os.path.relpath(d), option=r['option'], fams='+'.join(r['families']), cls=cls,
                         dmse=VB['oof_dmse'], folds=VB['folds_better'], rs_mean=VB['resplit40']['mean'],
                         P=VB['resplit40']['P_ge0'], dt3=VB['droptop3'], krig=kL['rmse'], p_beat=p_beat,
                         rms_best=rs[best], min_sub=min(rs.values()), md5=r['test_md5'], best=best))
rank = {'SAFE': 0, 'BOLD': 1, 'NEAR-TIE(strict)': 2, 'NEAR-TIE': 2}
rows.sort(key=lambda x: (rank.get(x['cls'], 9), round(x['krig'], 4), x['dmse']))
for x in rows:
    print(f"{x['cls']:16s} {x['build']}/{x['option']:5s} [{x['fams']}] vs {x['best']} {x['dmse']:+.3f} {x['folds']}/5 "
          f"rs {x['rs_mean']:+.3f} P {x['P']:.2f} dt3 {x['dt3']:+.3f} | krig {x['krig']:.4f} P(beat best) {x['p_beat']:.2f} | "
          f"rms best {x['rms_best']:.3f} min-sub {x['min_sub']:.3f} | md5 {x['md5']}")
