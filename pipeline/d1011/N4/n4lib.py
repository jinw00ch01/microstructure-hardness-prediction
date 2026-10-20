"""N4 (2026-10-11 afternoon, new submission criterion): shared helpers so every candidate is scored the same way
against blend_v29g. Labels: train only (src.common.load_train via lib_l4). Test rows enter only as label-free group
membership and as the candidate's own test predictions (never fitted on, never used to choose anything).
"""
import sys, json, hashlib, os
sys.path.insert(0, '/mnt/project-files/work/hardness-cache/scripts/d1011/bold/L4_noisy')
from lib_l4 import *  # noqa  (np, pd, REPO, CACHE, SUB, STRATA, RHO, groups, load, signflip_p, strat_split, score)

N4 = CACHE + '/scripts/d1011/N4'
V29G_MD5 = '496f5cc978c2e4298775e992455dbeac'


def load29():
    D = load()
    o = pd.read_csv(f'{SUB}/blend_v29g_oof.csv', float_precision='round_trip')
    assert (o.ID.astype(str).to_numpy() == D['ids']).all()
    D['p29'] = o.hardness.to_numpy().astype(float)
    assert hashlib.md5(open(f'{SUB}/blend_v29g.csv', 'rb').read()).hexdigest() == V29G_MD5
    t = pd.read_csv(f'{SUB}/blend_v29g.csv', float_precision='round_trip')
    assert (t.ID.astype(str).to_numpy() == D['te_ids']).all()
    D['t29'] = t.hardness.to_numpy().astype(float)
    return D


def vs29(q, D, nflip=20000):
    """All-500 comparison of a candidate nested OOF q against the v29g OOF (same folds)."""
    y, fold, p = D['y'], D['fold'], D['p29']
    d = (q - y) ** 2 - (p - y) ** 2
    ch = d != 0
    per = [float(d[fold == k].sum()) for k in range(5)]
    o = np.argsort(d); keep = np.ones(500, bool); keep[o[:3]] = False
    out = dict(all500_dmse=float(d.mean()), rmse_v29g=float(np.sqrt(np.mean((p - y) ** 2))), rmse_new=float(np.sqrt(np.mean((q - y) ** 2))),
               n_changed=int(ch.sum()), folds_better=int(sum(x < 0 for x in per)), fold_dsse=[round(x, 1) for x in per],
               signflip_p=float(signflip_p(d[ch], nflip=nflip)) if ch.any() else 1.0, droptop3_all500_dmse=float(d[keep].mean()),
               by_group={g: dict(n_changed=int((ch & D['grp'][g]).sum()), dmse_group=float(d[D['grp'][g]].mean())) for g in STRATA})
    return out


def public_expect(q, tnew, D):
    """Expected public dMSE vs v29g and its noise sd, by row group (public = random 300 of the 1000 test rows).
    CV hypothesis: a changed test row in group g gains the mean per-changed-row dSE of the changed train rows of g, scaled
    by (test change rms / train change rms)^2 is NOT applied (kept simple and stated). Noise: per-row dSE variance of g.
    Null (no signal): dSE_i = delta_i^2 + 2 delta_i e_i with e_i ~ N(0, s_g^2), s_g = v29g OOF residual sd of g -> mean
    sum(delta^2)/1000 per public row, sd from 4 delta^2 s_g^2."""
    y, p = D['y'], D['p29']
    dtr = q - p; dte = tnew - D['t29']
    dse = (q - y) ** 2 - (p - y) ** 2
    E = 0.0; V = 0.0; En = 0.0; Vn = 0.0; rows = {}
    for g in STRATA:
        mtr = D['grp'][g] & (dtr != 0); mte = D['grpt'][g] & (dte != 0)
        nte = int(mte.sum())
        if nte == 0:
            continue
        s = float(np.std((p - y)[D['grp'][g]]))
        if mtr.sum() >= 2:
            m, v = float(dse[mtr].mean()), float(dse[mtr].var())
        else:
            m, v = 0.0, 0.0
        npub = 0.3 * nte
        E += npub * m; V += npub * v
        dd = dte[mte]
        En += 0.3 * float((dd ** 2).sum()); Vn += 0.3 * float((4 * dd ** 2 * s ** 2).sum())
        rows[g] = dict(n_test_changed=nte, test_change_rms=float(np.sqrt(np.mean(dd ** 2))),
                       train_change_rms=float(np.sqrt(np.mean(dtr[mtr] ** 2))) if mtr.any() else 0.0, per_row_dse_train=m)
    base = 10.0564339191 ** 2
    E /= 300; V = V ** 0.5 / 300; En /= 300; Vn = Vn ** 0.5 / 300
    return dict(cv_hyp_public_dmse=E, cv_hyp_sd=V, null_public_dmse=En, null_sd=Vn,
                cv_hyp_public_rmse=float(np.sqrt(base + E)), null_public_rmse=float(np.sqrt(base + En)), groups=rows,
                note='public dMSE vs v29g (public MSE 101.13); P(score <= 10.0) under the CV hypothesis ~ Phi((-1.13 - E)/sd)')


def write(dirname, q, tnew, D, meta):
    """Write candidate oof.csv / test.csv / result.json under N4/<dirname>/ (test.csv stays in the shared cache only)."""
    os.makedirs(f'{N4}/{dirname}', exist_ok=True)
    assert np.isfinite(q).all() and np.isfinite(tnew).all() and len(q) == 500 and len(tnew) == 1000
    pd.DataFrame({'ID': D['ids'], 'hardness': q}).to_csv(f'{N4}/{dirname}/oof.csv', index=False)
    pd.DataFrame({'ID': D['te_ids'], 'hardness': tnew}).to_csv(f'{N4}/{dirname}/test.csv', index=False)
    res = dict(meta); res['vs_v29g'] = vs29(q, D); res['public'] = public_expect(q, tnew, D)
    res['test_md5'] = hashlib.md5(open(f'{N4}/{dirname}/test.csv', 'rb').read()).hexdigest()
    json.dump(res, open(f'{N4}/{dirname}/result.json', 'w'), indent=1)
    return res
