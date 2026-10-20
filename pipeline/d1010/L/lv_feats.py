"""d1010 L stage 1 (label-free): formula7 terms for the line-visible mid-noise (LV) rows with the UNCHANGED clean pipeline.

Rows (PREREG.txt): LV  = 9.5 <= raw features_v3 ic_noise < 12 AND ic_ridge_snr >= 0.5 (train + test)
                   HLV = ic_noise >= 12 AND ic_ridge_snr >= 0.5 (report only; never part of the bar or the file)
Pipeline = d1009/J/cleanhead seg_c.py (src.het_blocks.segment sigma 1.0, regionprops grain table, shading_norm gn)
          + feats_c.py feats() imported unchanged (per-image GMM phase threshold, pores gn < 0.68 & area < 600,
          N_eff = N_in + 0.5 N_bd). The grain table and the int16 label-map round trip copy seg_c.py exactly.
Check first: 5 clean train images must reproduce cleanhead feats_C_train (all 42 columns, exact), grains_C_train
and ws_C_train exactly; the run aborts otherwise.
No label is read (train.csv is read with usecols=['ID']); test images are only segmented, nothing is fitted across
images (the 2-GMM is per image, as in the clean pipeline).
Outputs (this folder): feats_LV_{train,test}.parquet (feats_C columns + group, ic_noise, ic_ridge_snr, ic_gmm_sep,
gmm_gap, thr_fallback), fidelity.json.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import json
import sys

sys.dont_write_bytecode = True        # never write __pycache__ into the shared cleanhead folder
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy import stats as st
from skimage import measure
from sklearn.mixture import GaussianMixture

REPO = '/home/claude/microstructure-hardness-prediction'
CH = '/mnt/project-files/work/hardness-cache/scripts/d1009/J/cleanhead/'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, CH)
from src.het_blocks import segment, shading_norm  # noqa: E402
import feats_c  # noqa: E402  (cleanhead feats(), phase_thr(); module level only defines constants)

TERMS = ['fd_area', 'pore_frac', 'logasp_aw', 'S_align', 'logN_eff', 'het4', 'bd_frac']


def seg_one(split, i):
    """Copy of cleanhead seg_c.one with the split as an argument (seg_c reads it from sys.argv)."""
    im = np.array(Image.open(f'{REPO}/data/{split}/{i}.png')).astype(np.float32)
    sm, ws = segment(im)
    rows = []
    for p in measure.regionprops(ws, intensity_image=sm):
        minr, minc, maxr, maxc = p.bbox
        rows.append(dict(label=p.label, area=p.area, cy=p.centroid[0], cx=p.centroid[1], mean=p.intensity_mean,
                         maj=p.axis_major_length, mnr=p.axis_minor_length, ori=p.orientation, perim=p.perimeter,
                         border=minr == 0 or minc == 0 or maxr == 256 or maxc == 256,
                         nb=int((minr == 0) + (minc == 0) + (maxr == 256) + (maxc == 256))))
    g = pd.DataFrame(rows); g['gn'] = shading_norm(g); g['ID'] = i
    return g, ws.astype(np.int16)


def gmm_info(g):
    """The GMM of feats_c.phase_thr, re-run to expose the phase gap and whether the 0.90 fallback was used."""
    pore = ((g.gn < 0.68) & (g.area < 600)).values
    gp = g[~pore]
    big = gp[gp.area > 25]
    try:
        gm = GaussianMixture(2, random_state=0).fit(big.gn.values[:, None])
        mu = np.sort(gm.means_.ravel())
        gap = float(mu[1] - mu[0])
        return gap, float(not gap > 0.08)
    except Exception:
        return np.nan, 1.0


def run_one(split, i):
    g, ws16 = seg_one(split, i)
    f = feats_c.feats(g.reset_index(drop=True), ws16.astype(np.int32))
    gap, fb = gmm_info(g)
    return i, g, ws16, f, gap, fb


def design(F):
    D = pd.DataFrame(index=F.index)
    for c in ['fd_area', 'pore_frac', 'logasp_aw', 'S_align', 'het4']:
        D[c] = F[c].astype(float)
    D['logN_eff'] = np.log(F['N_eff'].astype(float))
    D['bd_frac'] = F['N_bd'].astype(float) / F['N_all'].astype(float)
    return D[TERMS]


def q(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return dict(n=int(len(x)), mean=float(np.mean(x)), sd=float(np.std(x)), p10=float(np.percentile(x, 10)),
                med=float(np.median(x)), p90=float(np.percentile(x, 90)), min=float(np.min(x)), max=float(np.max(x)))


if __name__ == '__main__':
    out_tr, out_te, out_js = (HERE + '/feats_LV_train.parquet', HERE + '/feats_LV_test.parquet', HERE + '/fidelity.json')
    for p in (out_tr, out_te, out_js):
        assert not os.path.exists(p), f'refusing to overwrite {p}'
    v3 = pd.read_parquet(REPO + '/data/features_v3.parquet').set_index('ID')
    ids_tr_all = pd.read_csv(REPO + '/data/train.csv', usecols=['ID']).ID.values      # IDs only, no label
    ids_te_all = pd.read_csv(REPO + '/data/sample_submission.csv', usecols=['ID']).ID.values
    assert len(ids_tr_all) == 500 and len(ids_te_all) == 1000

    def groups(ids):
        x = v3.loc[ids]
        nz, rd = x.ic_noise.values, x.ic_ridge_snr.values
        return dict(C=nz < 9.5, LV=(nz >= 9.5) & (nz < 12) & (rd >= 0.5), HLV=(nz >= 12) & (rd >= 0.5),
                    MLF=(nz >= 9.5) & (nz < 12) & (rd < 0.5))
    gtr, gte = groups(ids_tr_all), groups(ids_te_all)
    counts = {s: {k: int(v.sum()) for k, v in gg.items()} for s, gg in (('train', gtr), ('test', gte))}
    print('counts', counts)
    assert counts['train']['C'] == 267 and counts['test']['C'] == 519
    exp = dict(train=dict(LV=21, HLV=4), test=dict(LV=42, HLV=6))
    count_note = {s: {k: (counts[s][k], exp[s][k], counts[s][k] == exp[s][k]) for k in exp[s]} for s in exp}
    print('counts vs prereg expectation (found, expected, equal):', count_note)

    # ---------------------------------------------------------- 1. reproduction check on 5 clean train images
    Fc_tr = pd.read_parquet(CH + 'feats_C_train.parquet')
    Fc_te = pd.read_parquet(CH + 'feats_C_test.parquet')
    Gc_tr = pd.read_parquet(CH + 'grains_C_train.parquet')
    Wc_tr = np.load(CH + 'ws_C_train.npz')
    rng = np.random.default_rng(20261010)
    chk_ids = sorted(rng.choice(np.sort(Fc_tr.index.values), 5, replace=False).tolist())
    res = Parallel(n_jobs=3)(delayed(run_one)('train', i) for i in chk_ids)
    repro = {}
    for i, g, ws16, f, gap, fb in res:
        ref_f = Fc_tr.loc[i]
        mine = pd.Series(f).astype(float)[Fc_tr.columns]
        f_ok = bool(((mine.values == ref_f.values) | (np.isnan(mine.values) & np.isnan(ref_f.values))).all())
        ref_g = Gc_tr[Gc_tr.ID == i].reset_index(drop=True)
        g_ok = bool(g.reset_index(drop=True).drop(columns='ID').equals(ref_g.drop(columns='ID')))
        w_ok = bool(np.array_equal(ws16, Wc_tr[i]))
        repro[i] = dict(feats_exact=f_ok, grains_exact=g_ok, ws_exact=w_ok,
                        max_abs_diff=float(np.nanmax(np.abs(mine.values - ref_f.values))))
        print('repro', i, repro[i], flush=True)
    assert all(r['feats_exact'] and r['grains_exact'] and r['ws_exact'] for r in repro.values()), 'pipeline NOT exact'

    # ---------------------------------------------------------- 2. LV (+ HLV report-only) train and test images
    jobs = [('train', i, 'LV') for i in ids_tr_all[gtr['LV']]] + [('train', i, 'HLV') for i in ids_tr_all[gtr['HLV']]] + \
           [('test', i, 'LV') for i in ids_te_all[gte['LV']]] + [('test', i, 'HLV') for i in ids_te_all[gte['HLV']]]
    res = Parallel(n_jobs=3)(delayed(run_one)(s, i) for s, i, _ in jobs)
    rows = {'train': {}, 'test': {}}
    for (s, i, grp), (i2, g, ws16, f, gap, fb) in zip(jobs, res):
        assert i == i2
        r = dict(f); r.update(group=grp, gmm_gap=gap, thr_fallback=fb, n_grains_raw=len(g),
                              ic_noise=float(v3.loc[i, 'ic_noise']), ic_ridge_snr=float(v3.loc[i, 'ic_ridge_snr']),
                              ic_gmm_sep=float(v3.loc[i, 'ic_gmm_sep']))
        rows[s][i] = r
    Ftr = pd.DataFrame(rows['train']).T; Fte = pd.DataFrame(rows['test']).T
    for F in (Ftr, Fte):
        F.index.name = 'ID'
        for c in F.columns:
            if c != 'group':
                F[c] = F[c].astype(float)
    Ftr.to_parquet(out_tr); Fte.to_parquet(out_te)

    # ---------------------------------------------------------- 3. fidelity report (label-free)
    # phase gap / fallback for the clean rows from the cached clean grain tables (same GMM; consistency vs feats_C thr)
    def clean_gaps(G, Fc):
        out = {}
        for i, g in G.groupby('ID'):
            out[i] = gmm_info(g.reset_index(drop=True))
        d = pd.DataFrame(out, index=['gmm_gap', 'thr_fallback']).T
        cons = ((d.thr_fallback == 1) == (Fc.loc[d.index, 'thr'] == 0.90)).mean()
        return d, float(cons)
    cg_tr, cons_tr = clean_gaps(Gc_tr, Fc_tr)
    Gc_te = pd.read_parquet(CH + 'grains_C_test.parquet')
    cg_te, cons_te = clean_gaps(Gc_te, Fc_te)
    del Gc_te
    print('clean fallback flag consistent with feats_C thr==0.90: train %.3f test %.3f' % (cons_tr, cons_te))

    Dc_tr, Dc_te = design(Fc_tr), design(Fc_te)
    lv_tr, lv_te = Ftr[Ftr.group == 'LV'], Fte[Fte.group == 'LV']
    hv_tr, hv_te = Ftr[Ftr.group == 'HLV'], Fte[Fte.group == 'HLV']
    Dlv_tr, Dlv_te = design(lv_tr), design(lv_te)
    Dhv_tr, Dhv_te = design(hv_tr), design(hv_te)
    assert np.isfinite(Dlv_tr.values).all() and np.isfinite(Dlv_te.values).all(), 'non-finite formula7 term on LV rows'
    sep = lambda ids: v3.loc[ids, 'ic_gmm_sep'].values
    fid = {
        'prereg': 'scripts/d1010/L/PREREG.txt', 'stage': 'label-free (no label read; test images only segmented)',
        'counts': counts, 'counts_vs_prereg': {s: {k: dict(found=v[0], expected=v[1], equal=v[2]) for k, v in d.items()}
                                               for s, d in count_note.items()},
        'repro_check_5_clean_train': repro,
        'clean_fallback_consistency_with_feats_C_thr': dict(train=cons_tr, test=cons_te),
        'ic_gmm_sep': {'LV_train': q(sep(lv_tr.index)), 'LV_test': q(sep(lv_te.index)),
                       'clean_train': q(sep(Fc_tr.index)), 'clean_test': q(sep(Fc_te.index)),
                       'HLV_train': q(sep(hv_tr.index)), 'HLV_test': q(sep(hv_te.index))},
        'phase_gap': {'LV_train': q(lv_tr.gmm_gap), 'LV_test': q(lv_te.gmm_gap),
                      'clean_train': q(cg_tr.gmm_gap), 'clean_test': q(cg_te.gmm_gap),
                      'HLV_train': q(hv_tr.gmm_gap), 'HLV_test': q(hv_te.gmm_gap)},
        'gmm_fallback_share': {'LV_train': float(lv_tr.thr_fallback.mean()), 'LV_test': float(lv_te.thr_fallback.mean()),
                               'clean_train': float(cg_tr.thr_fallback.mean()), 'clean_test': float(cg_te.thr_fallback.mean()),
                               'HLV_train': float(hv_tr.thr_fallback.mean()), 'HLV_test': float(hv_te.thr_fallback.mean())},
        'N_all': {'LV_train': q(lv_tr.N_all), 'LV_test': q(lv_te.N_all), 'clean_train': q(Fc_tr.N_all), 'clean_test': q(Fc_te.N_all)},
        'terms': {}, 'n_terms_outside_clean_train_range': {},
    }
    lo, hi = Dc_tr.min(), Dc_tr.max()
    for t in TERMS:
        a, b, c = Dlv_tr[t].values, Dlv_te[t].values, Dc_tr[t].values
        ks_tt = st.ks_2samp(a, b); ks_lc = st.ks_2samp(np.r_[a, b], c)
        fid['terms'][t] = dict(LV_train=q(a), LV_test=q(b), clean_train=q(c), clean_test=q(Dc_te[t].values),
                               HLV_train=q(Dhv_tr[t].values), HLV_test=q(Dhv_te[t].values),
                               ks_LVtrain_vs_LVtest_p=float(ks_tt.pvalue), ks_LVall_vs_clean_train_p=float(ks_lc.pvalue),
                               LV_mean_z_vs_clean_train=float((np.r_[a, b].mean() - c.mean()) / c.std()))
    out_range = lambda D: {t: int(((D[t] < lo[t]) | (D[t] > hi[t])).sum()) for t in TERMS}
    fid['n_terms_outside_clean_train_range'] = dict(LV_train=out_range(Dlv_tr), LV_test=out_range(Dlv_te),
                                                    HLV_train=out_range(Dhv_tr), HLV_test=out_range(Dhv_te))
    # Mahalanobis distance of each LV row to the clean train term cloud (label-free leverage check)
    mu, S = Dc_tr.mean().values, np.cov(Dc_tr.values.T)
    Si = np.linalg.inv(S)
    md = lambda D: np.sqrt(np.einsum('ij,jk,ik->i', D.values - mu, Si, D.values - mu))
    fid['mahalanobis_to_clean_train'] = dict(clean_train=q(md(Dc_tr)), clean_test=q(md(Dc_te)), LV_train=q(md(Dlv_tr)),
                                             LV_test=q(md(Dlv_te)), HLV_train=q(md(Dhv_tr)), HLV_test=q(md(Dhv_te)),
                                             clean_train_p99=float(np.percentile(md(Dc_tr), 99)))
    json.dump(fid, open(out_js, 'w'), indent=1)

    # ---------------------------------------------------------- summary print
    f2 = lambda d: f"{d['med']:.3f} [{d['p10']:.3f}, {d['p90']:.3f}]"
    print('ic_gmm_sep med [p10,p90]:', {k: f2(v) for k, v in fid['ic_gmm_sep'].items()})
    print('phase gap  med [p10,p90]:', {k: f2(v) for k, v in fid['phase_gap'].items()})
    print('GMM fallback share:', {k: round(v, 3) for k, v in fid['gmm_fallback_share'].items()})
    for t in TERMS:
        d = fid['terms'][t]
        print(f"{t:10s} LVtr {f2(d['LV_train'])} LVte {f2(d['LV_test'])} cleantr {f2(d['clean_train'])} "
              f"KS(tr,te) p {d['ks_LVtrain_vs_LVtest_p']:.3f} KS(LV,clean) p {d['ks_LVall_vs_clean_train_p']:.3f} "
              f"z {d['LV_mean_z_vs_clean_train']:+.2f}")
    print('outside clean range:', fid['n_terms_outside_clean_train_range'])
    print('mahalanobis med:', {k: (round(v['med'], 2) if isinstance(v, dict) else round(v, 2))
                               for k, v in fid['mahalanobis_to_clean_train'].items()})
