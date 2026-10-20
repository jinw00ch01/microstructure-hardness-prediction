"""d1010 N/gate stage 3 (label-free): reliability scores, clean+LV-train calibrated cuts, pass counts, sims.

No hardness label is read. formula7 coefficients are the recorded full clean fit (L/l_test_result.json coef_full,
the coefficients v25 / v26L used on test rows); their use is checked against blend_v26L on the 42 LV test rows
(f7 = 2 * v26L - v25 there).
Gate (fixed before any number below was computed):
  W   = sqrt(SD_seg(f7)^2 + D_thr^2)  [HV]  SD over the 5 segmentations V0..V4 (ddof 0), D_thr = half the f7 change
        when the V0 phase threshold moves -0.02 -> +0.02
  gap = V0 per-image 2-GMM phase gap (gn units; NaN -> 0)
  cuts from the 288 clean + LV TRAIN rows only: W90 = p90(W), g10 = p10(gap)
  G_W: W <= W90; G_gap: gap >= g10; PRIMARY G_both: both.
Outputs: rel_train.parquet, rel_test.parquet, gate_result.json, stdout.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import json
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import glob
import hashlib

import numpy as np
import pandas as pd
from scipy import stats as st

import gate_lib as L

CACHE = '/mnt/project-files/work/hardness-cache'
LDIR = CACHE + '/scripts/d1010/L/'
SUB = CACHE + '/submissions/'
TERMS = L.TERMS


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def load_q(split):
    fs = sorted(glob.glob(f'{HERE}/q_{split}_*of*.parquet'))
    Q = pd.concat([pd.read_parquet(f) for f in fs])
    return Q


def groups(v3, ids):
    x = v3.loc[ids]
    nz, rd = x.ic_noise.values, x.ic_ridge_snr.values
    g = np.full(len(ids), '', object)
    g[nz < 9.5] = 'clean'
    g[(nz >= 9.5) & (nz < 12) & (rd >= 0.5)] = 'LV'
    g[(nz >= 9.5) & (nz < 12) & (rd < 0.5)] = 'midLF'
    g[(nz >= 12) & (rd >= 0.5)] = 'HLV'
    g[(nz >= 12) & (rd < 0.5)] = 'G1hi'
    assert (g != '').all()
    return pd.Series(g, index=ids)


def qd(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0)
    return dict(n=int(len(x)), p10=float(np.percentile(x, 10)), med=float(np.median(x)),
                p90=float(np.percentile(x, 90)), mean=float(np.mean(x)))


if __name__ == '__main__':
    out_js = HERE + '/gate_result.json'
    assert not os.path.exists(out_js), 'refusing to overwrite gate_result.json'
    REPO = L.REPO
    v3 = pd.read_parquet(REPO + '/data/features_v3.parquet').set_index('ID')
    ids_tr = pd.read_csv(REPO + '/data/train.csv', usecols=['ID']).ID.values
    ids_te = pd.read_csv(REPO + '/data/sample_submission.csv', usecols=['ID']).ID.values
    Qtr, Qte = load_q('train').loc[ids_tr], load_q('test').loc[ids_te]
    Gtr, Gte = groups(v3, ids_tr), groups(v3, ids_te)
    res = {'counts': {'train': Gtr.value_counts().to_dict(), 'test': Gte.value_counts().to_dict()}}
    print('groups', res['counts'])

    # ------------------------------------------------ 1. V0 == cached unchanged pipeline (clean, LV, HLV)
    rep = {}
    for split, Q in (('train', Qtr), ('test', Qte)):
        Fc = L.design(pd.read_parquet(L.CH + f'feats_C_{split}.parquet'))
        Fl = L.design(pd.read_parquet(LDIR + f'feats_LV_{split}.parquet'))
        for nm, F in (('clean', Fc), ('LV+HLV', Fl)):
            mine = Q.loc[F.index, [f'V0__{t}' for t in TERMS]].values
            rep[f'{split}_{nm}'] = dict(n=int(len(F)), max_abs_diff=float(np.nanmax(np.abs(mine - F.values))))
    print('V0 reproduction', rep)
    assert all(r['max_abs_diff'] < 1e-12 for r in rep.values())
    res['v0_reproduction'] = rep

    # ------------------------------------------------ 2. formula7 coefficients (recorded) + check vs blend_v26L
    lt = json.load(open(LDIR + 'l_test_result.json'))
    cf = lt['coef_full']
    b = np.array([cf['intercept']] + [cf[t] for t in TERMS])
    v26 = pd.read_csv(SUB + 'blend_v26L.csv').set_index('ID').hardness
    v25 = pd.read_csv(SUB + 'blend_v25.csv').set_index('ID').hardness
    lv_te = Gte.index[Gte == 'LV']
    f7_lv = L.f7_apply(b, Qte.loc[lv_te, [f'V0__{t}' for t in TERMS]].values)
    chk = float(np.max(np.abs(f7_lv - (2 * v26.loc[lv_te].values - v25.loc[lv_te].values))))
    print('f7 vs 2*v26L - v25 on LV test rows, max abs diff', chk)
    res['f7_check_vs_v26L'] = dict(max_abs_diff=chk, md5_v26L=md5(SUB + 'blend_v26L.csv'), md5_v25=md5(SUB + 'blend_v25.csv'))
    assert chk < 1e-6

    # ------------------------------------------------ 3. reliability scores
    Rtr, Rte = L.reliability(Qtr, b), L.reliability(Qte, b)
    for R, G, v3i in ((Rtr, Gtr, ids_tr), (Rte, Gte, ids_te)):
        R['group'] = G.values
        for c in ('ic_noise', 'ic_ridge_snr', 'ic_gmm_sep'):
            R[c] = v3.loc[v3i, c].values
    # Mahalanobis distance of the V0 term vector to the clean train cloud (leverage, report only)
    Dc = Qtr.loc[Gtr.index[Gtr == 'clean'], [f'V0__{t}' for t in TERMS]].values
    mu, Si = Dc.mean(0), np.linalg.inv(np.cov(Dc.T))
    for R, Q in ((Rtr, Qtr), (Rte, Qte)):
        X = Q[[f'V0__{t}' for t in TERMS]].values - mu
        R['mahal'] = np.sqrt(np.einsum('ij,jk,ik->i', X, Si, X))
    mahal99 = float(np.percentile(Rtr.mahal[Gtr == 'clean'], 99))
    f7lo, f7hi = float(Rtr.f7_V0[Gtr == 'clean'].min()), float(Rtr.f7_V0[Gtr == 'clean'].max())

    # ------------------------------------------------ 4. cuts from clean + LV TRAIN rows only
    cal = Rtr[Rtr.group.isin(['clean', 'LV'])]
    assert len(cal) == 288
    W90, W95 = float(np.percentile(cal.W, 90)), float(np.percentile(cal.W, 95))
    g10 = float(np.percentile(cal.gap, 10))
    res['cuts'] = dict(W90=W90, W95=W95, g10=g10, mahal_clean_p99=mahal99, f7_clean_train_range=[f7lo, f7hi],
                       n_cal=int(len(cal)))
    print('cuts', res['cuts'])
    for R in (Rtr, Rte):
        R['G_W'] = R.W <= W90
        R['G_gap'] = R.gap >= g10
        R['G_both'] = R.G_W & R.G_gap
        R['G_both95'] = (R.W <= W95) & R.G_gap
        R['in_clean_range'] = (R.mahal <= mahal99)
    Rtr.to_parquet(HERE + '/rel_train.parquet'); Rte.to_parquet(HERE + '/rel_test.parquet')

    # ------------------------------------------------ 5. pass counts by group, train vs test
    order = ['clean', 'LV', 'midLF', 'HLV', 'G1hi']
    pc = {}
    for gate in ('G_W', 'G_gap', 'G_both', 'G_both95'):
        pc[gate] = {}
        for grp in order:
            a, bb = Rtr[Rtr.group == grp][gate], Rte[Rte.group == grp][gate]
            tab = [[int(a.sum()), int((~a).sum())], [int(bb.sum()), int((~bb).sum())]]
            p = float(st.fisher_exact(tab)[1])
            pc[gate][grp] = dict(train_pass=int(a.sum()), train_n=int(len(a)), test_pass=int(bb.sum()), test_n=int(len(bb)),
                                 train_rate=float(a.mean()), test_rate=float(bb.mean()), fisher_p=p)
    res['pass_counts'] = pc
    for gate in pc:
        print(gate, {g: f"tr {v['train_pass']}/{v['train_n']} te {v['test_pass']}/{v['test_n']} p {v['fisher_p']:.2f}"
                     for g, v in pc[gate].items()})

    # ------------------------------------------------ 6. component distributions by group
    comp = {}
    for c in ('W', 'sd_seg', 'd_thr', 'gap', 'fallback', 'nfb_var', 'amb_area', 'mahal', 'ic_gmm_sep', 'ic_ridge_snr',
              'ic_noise'):
        comp[c] = {f'{grp}_{s}': qd(R[R.group == grp][c]) for grp in order for s, R in (('train', Rtr), ('test', Rte))}
    res['components'] = comp
    for c in ('W', 'sd_seg', 'd_thr', 'gap', 'nfb_var'):
        print(c, {k: round(v['med'], 3) for k, v in comp[c].items() if v.get('n', 0)})
    # train vs test shift of W and gap within groups
    res['shift_ks'] = {f'{c}_{grp}': float(st.ks_2samp(Rtr[Rtr.group == grp][c], Rte[Rte.group == grp][c]).pvalue)
                       for c in ('W', 'gap') for grp in order}
    print('KS train vs test p', {k: round(v, 3) for k, v in res['shift_ks'].items()})

    # ------------------------------------------------ 7. label-free agreement: corr(f7_V0, v24 prediction) per group x gate
    # (v24, not v26L: v26L contains 0.5 * f7 on clean / LV rows, which would inflate the agreement there)
    v24_oof = pd.read_csv(SUB + 'blend_v24_oof.csv').set_index('ID').hardness
    v24 = pd.read_csv(SUB + 'blend_v24.csv').set_index('ID').hardness
    res['md5_v24'] = dict(csv=md5(SUB + 'blend_v24.csv'), oof=md5(SUB + 'blend_v24_oof.csv'))
    agree = {}
    for grp in order:
        for s, R, base in (('train_oof', Rtr, v24_oof), ('test', Rte, v24)):
            for lab, m in (('pass', R.G_both), ('fail', ~R.G_both), ('passW', R.G_W), ('failW', ~R.G_W)):
                sel = R[(R.group == grp) & m]
                if len(sel) >= 5:
                    bb = base.loc[sel.index].values
                    agree[f'{grp}_{s}_{lab}'] = dict(n=int(len(sel)), corr=float(np.corrcoef(sel.f7_V0, bb)[0, 1]),
                                                    rms_diff=float(np.sqrt(np.mean((sel.f7_V0 - bb) ** 2))),
                                                    mean_diff=float(np.mean(sel.f7_V0 - bb)), sd_base=float(np.std(bb)))
    res['agreement_f7_vs_v24'] = agree
    for k, v in agree.items():
        print('agree', k, {kk: round(vv, 2) for kk, vv in v.items()})

    # ------------------------------------------------ 8. clean rows that fail
    cf_ = {}
    for s, R in (('train', Rtr), ('test', Rte)):
        C = R[R.group == 'clean']
        for lab, m in (('pass', C.G_both), ('fail', ~C.G_both), ('fail_W', ~C.G_W), ('fail_gap', ~C.G_gap)):
            sel = C[m]
            Qs = (Qtr if s == 'train' else Qte).loc[sel.index]
            cf_[f'{s}_{lab}'] = dict(n=int(len(sel)), fallback=float(sel.fallback.mean()) if len(sel) else None,
                                     fd_area_med=float(Qs['V0__fd_area'].median()) if len(sel) else None,
                                     N_all_med=float(Qs['V0__N_all'].median()) if len(sel) else None,
                                     noise_med=float(sel.ic_noise.median()) if len(sel) else None,
                                     ridge_med=float(sel.ic_ridge_snr.median()) if len(sel) else None,
                                     gmm_sep_med=float(sel.ic_gmm_sep.median()) if len(sel) else None,
                                     W_med=float(sel.W.median()) if len(sel) else None,
                                     f7_med=float(sel.f7_V0.median()) if len(sel) else None)
    res['clean_pass_fail'] = cf_
    for k, v in cf_.items():
        print('clean', k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})

    # ------------------------------------------------ 9. test rows a gate extension would change (label-free sizes)
    ext = {}
    for grp in ('midLF', 'HLV', 'G1hi'):
        sel = Rte[(Rte.group == grp) & Rte.G_both]
        ch = 0.5 * (sel.f7_V0 - v26.loc[sel.index])
        ext[grp] = dict(n=int(len(sel)), change_rms=float(np.sqrt(np.mean(ch ** 2))) if len(sel) else None,
                        change_max_abs=float(np.max(np.abs(ch))) if len(sel) else None,
                        n_out_of_clean_cloud=int((~sel.in_clean_range).sum()),
                        n_f7_outside_clean_train_range=int(((sel.f7_V0 < f7lo) | (sel.f7_V0 > f7hi)).sum()),
                        ids_top_change=ch.abs().sort_values(ascending=False).index[:5].tolist(),
                        top_change=[float(x) for x in ch.loc[ch.abs().sort_values(ascending=False).index[:5]]])
    res['test_change_if_extended'] = ext
    print('test change sizes', ext)

    # ------------------------------------------------ 10. sims (degraded clean TRAIN images)
    Fc_tr = L.design(pd.read_parquet(L.CH + 'feats_C_train.parquet'))
    f7_orig = pd.Series(L.f7_apply(b, Fc_tr.values), index=Fc_tr.index)
    sims = {}
    for sc in ('lv', 'mid', 'g1'):
        p = f'{HERE}/sim_{sc}.parquet'
        if not os.path.exists(p):
            continue
        S = pd.read_parquet(p)
        RS = L.reliability(S, b)
        d = RS.f7_V0 - f7_orig.loc[RS.index]
        RS['G_W'] = RS.W <= W90; RS['G_gap'] = RS.gap >= g10; RS['G_both'] = RS.G_W & RS.G_gap
        r = dict(n=int(len(RS)), delta_rms=float(np.sqrt(np.mean(d ** 2))), delta_bias=float(d.mean()),
                 corr_f7=float(np.corrcoef(RS.f7_V0, f7_orig.loc[RS.index])[0, 1]),
                 spearman_W_absd=float(st.spearmanr(RS.W, d.abs())[0]),
                 spearman_gap_absd=float(st.spearmanr(RS.gap, d.abs())[0]),
                 spearman_sdseg_absd=float(st.spearmanr(RS.sd_seg, d.abs())[0]),
                 spearman_dthr_absd=float(st.spearmanr(RS.d_thr, d.abs())[0]),
                 fallback_share=float(RS.fallback.mean()), W_med=float(RS.W.median()), gap_med=float(RS.gap.median()))
        for gate in ('G_W', 'G_gap', 'G_both'):
            m = RS[gate].values
            r[gate] = dict(pass_n=int(m.sum()),
                           rms_pass=float(np.sqrt(np.mean(d[m] ** 2))) if m.sum() else None,
                           rms_fail=float(np.sqrt(np.mean(d[~m] ** 2))) if (~m).sum() else None,
                           bias_pass=float(d[m].mean()) if m.sum() else None,
                           corr_pass=float(np.corrcoef(RS.f7_V0[m], f7_orig.loc[RS.index][m])[0, 1]) if m.sum() > 5 else None)
        sims[sc] = r
        print('sim', sc, json.dumps(r))
    res['sims'] = sims
    res['f7_orig_sd_clean_train'] = float(f7_orig.std())
    json.dump(res, open(out_js, 'w'), indent=1, default=float)
    print('wrote', out_js)
