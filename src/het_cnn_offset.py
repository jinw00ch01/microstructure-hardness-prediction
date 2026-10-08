"""Pre-registered stage-2 test: CNN-estimated grain-size heterogeneity (src.het_cnn) as an offset on the noisy images.

Rows: raw ic_noise >= 9.5 (data/features_v3.parquet) = 233 train / 481 test ("noisy"); every other row keeps the base
prediction exactly. Residual r = y - (nested OOF of the base blend, submissions/<base>_oof.csv).
PRIMARY: cross-fitted OLS (the 5 shared folds; fit on the noisy rows of the other 4 folds, apply to the held-out fold)
of r on [1, het_cnn, exp(0.25 logN_cnn)] over the noisy rows.
CONTROLS (reported only, never used to choose anything): (i) N-only [1, exp(0.25 logN_cnn)], cross-fitted the same way;
(ii) permutation p: het_cnn shuffled within the noisy rows (logN_cnn and folds fixed), primary cross-fitted noisy RMSE
recomputed, p = share of shuffles at least as good; (iii) zero-fit transfer of the clean v19 coefficients to
[1, het_cnn, exp(logN_cnn)^0.25]; (iv) breakdown by noise band and by logN_cnn tercile; (v) partial corr(het_cnn, r |
logN_cnn) on the noisy rows and, for reference, corr(het4, r | log N_eff) on the clean rows (measured het4).
PRE-REGISTERED BAR (fixed before any result existed, all four): noisy-row nested RMSE improves by >= 0.20 vs the base;
>= 4/5 folds improve on the noisy rows; permutation p < 0.05; the primary beats control (i) by >= 0.10.
If and only if the bar passes and --out is given: submissions/<out>.csv (base + offset on the noisy test images,
coefficients fitted on all noisy train rows), <out>_oof.csv (nested OOF), <out>.json, one LEADERBOARD line.
het_cnn comes from models trained on clean train images only (targets measured on the clean originals, hardness never
read); noisy train and all test images were only passed through those models. Test images are never fitted on here.

  python -m src.het_cnn_offset --base blend_v19 --het data/het_cnn/pilot_r18                    # CPU pilot, report only
  python -m src.het_cnn_offset --base blend_v19 --het data/het_cnn/ev2s_e32r4 --out blend_v22    # real run
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import DATA_DIR, EXP_DIR, ROOT, SUB_DIR, load_test, load_train, rmse

NOISE_MIN = 9.5  # src.het_blocks.NOISE_MAX: the het4 gate; rows at or above it are the noisy rows
CLEAN_COEF = [-74.19637629572568, 128.57014259832854, 12.577158859721699]  # blend_v19: [1, het4, N_eff^0.25]
BANDS = [(9.5, 12.0), (12.0, 15.0), (15.0, 21.0)]
BAR = {"min_gain": 0.20, "min_folds": 4, "max_p": 0.05, "min_vs_nonly": 0.10}
N_FOLDS = 5


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def crossfit(X, r, folds, rows):
    """Offset for `rows`: OLS on the rows of the other folds, applied to the held-out fold; 0 elsewhere."""
    off, coefs = np.zeros(len(r)), []
    for f in range(N_FOLDS):
        fit, app = rows & (folds != f), rows & (folds == f)
        c = ols(X[fit], r[fit])
        off[app] = X[app] @ c
        coefs.append(c.tolist())
    return off, coefs


def pcorr(a, b, c):
    """corr(a, b) after removing [1, c] from both by OLS."""
    Z = np.column_stack([np.ones(len(a)), c])
    ra, rb = a - Z @ ols(Z, a), b - Z @ ols(Z, b)
    return float(np.corrcoef(ra, rb)[0, 1])


def ols_se(X, y):
    c = ols(X, y)
    e = y - X @ c
    s2 = e @ e / (len(y) - X.shape[1])
    return c, np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))


def resolve_het(p):
    p = Path(p)
    for cand in (p, ROOT / p, DATA_DIR / "het_cnn" / p):
        if (cand / "het_cnn_train.parquet").exists():
            return cand.resolve()
    raise SystemExit(f"no het_cnn_train.parquet under {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="blend_v19", help="reads submissions/<base>_oof.csv (nested) and <base>.csv")
    ap.add_argument("--het", required=True, help="het_cnn run dir with het_cnn_{train,test}.parquet")
    ap.add_argument("--out", default=None, help="blend name to write, only if the pre-registered bar passes")
    ap.add_argument("--clean-ref", default="blend_v18",
                    help="pre-het-offset blend whose nested OOF gives the clean-row reference partial corr of het4")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default=None, help="report json (default <het dir>/stage2_<base>.json)")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    tr, te = load_train(), load_test()
    ss_ids = pd.read_csv(DATA_DIR / "sample_submission.csv").ID.values
    assert (te.ID.values == ss_ids).all()
    y, folds = tr.hardness.values, tr.fold.values

    def by_id(df, ids, col="hardness"):
        m = pd.DataFrame({"ID": ids}).merge(df, on="ID", how="left", validate="one_to_one")
        assert len(m) == len(ids) and m[col].notna().all(), f"missing {col}"
        return m[col].values

    oof = by_id(pd.read_csv(SUB_DIR / f"{a.base}_oof.csv"), tr.ID)
    base = by_id(pd.read_csv(SUB_DIR / f"{a.base}.csv"), te.ID)
    v3 = pd.read_parquet(DATA_DIR / "features_v3.parquet")[["ID", "ic_noise"]]
    nz_tr, nz_te = by_id(v3, tr.ID, "ic_noise"), by_id(v3, te.ID, "ic_noise")
    N, Nt = nz_tr >= NOISE_MIN, nz_te >= NOISE_MIN
    het_dir = resolve_het(a.het)
    Htr = pd.DataFrame({"ID": tr.ID}).merge(pd.read_parquet(het_dir / "het_cnn_train.parquet"), on="ID", how="left")
    Hte = pd.DataFrame({"ID": te.ID}).merge(pd.read_parquet(het_dir / "het_cnn_test.parquet"), on="ID", how="left")
    h, ln = Htr.het_cnn.values, Htr.logN_cnn.values
    ht, lnt = Hte.het_cnn.values, Hte.logN_cnn.values
    assert np.isfinite(h[N]).all() and np.isfinite(ln[N]).all(), "het_cnn missing on noisy train rows"
    assert np.isfinite(ht[Nt]).all() and np.isfinite(lnt[Nt]).all(), "het_cnn missing on noisy test rows"
    print(f"base {a.base} | het {het_dir} | noisy train {N.sum()} test {Nt.sum()} | src noisy train "
          f"{Htr.src[N].value_counts().to_dict() if 'src' in Htr else '-'}")

    r = y - oof
    X = np.zeros((len(tr), 3))
    X[N] = np.column_stack([np.ones(N.sum()), h[N], np.exp(0.25 * ln[N])])
    Xt = np.zeros((len(te), 3))
    Xt[Nt] = np.column_stack([np.ones(Nt.sum()), ht[Nt], np.exp(0.25 * lnt[Nt])])
    Xn = X[:, [0, 2]]

    def nrmse(off, rows=N):
        return rmse(r[rows] - off[rows], 0.0)

    def fold_rmse(off, rows=N):
        return [nrmse(off, rows & (folds == f)) for f in range(N_FOLDS)]

    zero = np.zeros(len(tr))
    off_p, coefs_p = crossfit(X, r, folds, N)
    off_n, coefs_n = crossfit(Xn, r, folds, N)
    off_z = X @ np.asarray(CLEAN_COEF) * N
    res = {"base_noisy": nrmse(zero), "primary_noisy": nrmse(off_p), "nonly_noisy": nrmse(off_n),
           "transfer_noisy": nrmse(off_z)}
    folds_tab = {k: fold_rmse(o) for k, o in (("base", zero), ("primary", off_p), ("nonly", off_n),
                                               ("transfer", off_z))}
    folds_tab["n"] = [int((N & (folds == f)).sum()) for f in range(N_FOLDS)]
    n_better = int(sum(p < b for p, b in zip(folds_tab["primary"], folds_tab["base"])))

    # (ii) permutation of het_cnn within the noisy rows
    rng = np.random.default_rng(a.seed)
    idx = np.where(N)[0]
    perm = np.empty(a.n_perm)
    for k in range(a.n_perm):
        Xp = X.copy()
        Xp[idx, 1] = X[rng.permutation(idx), 1]
        perm[k] = nrmse(crossfit(Xp, r, folds, N)[0])
    p_perm = float(np.mean(perm <= res["primary_noisy"]))

    # paired bootstrap over noisy rows of RMSE(primary) - RMSE(base) and - RMSE(N-only) (extra, not in the bar)
    e0, e1, e2 = r[N] ** 2, (r[N] - off_p[N]) ** 2, (r[N] - off_n[N]) ** 2
    bi = rng.integers(0, len(e0), size=(a.n_boot, len(e0)))
    d_base = np.sqrt(e1[bi].mean(1)) - np.sqrt(e0[bi].mean(1))
    d_nonly = np.sqrt(e1[bi].mean(1)) - np.sqrt(e2[bi].mean(1))

    # full-data fits (the coefficients a submission would use)
    coef_p, se_p = ols_se(X[N], r[N])
    coef_n, se_n = ols_se(Xn[N], r[N])

    # (iv) breakdown
    q = np.quantile(ln[N], [1 / 3, 2 / 3])
    subsets = {f"band {lo:g}-{hi:g}": N & (nz_tr >= lo) & (nz_tr < hi) for lo, hi in BANDS}
    terc = np.digitize(ln, q)
    for t, name in enumerate(["low", "mid", "high"]):
        subsets[f"logN_cnn {name}"] = N & (terc == t)
    breakdown = {}
    for k, S in subsets.items():
        breakdown[k] = {"n": int(S.sum()), "base": nrmse(zero, S), "primary": nrmse(off_p, S),
                        "nonly": nrmse(off_n, S), "transfer": nrmse(off_z, S),
                        "pcorr_het_r_given_logN": pcorr(h[S], r[S], ln[S]) if S.sum() > 5 else None,
                        "het_cnn_mean": float(h[S].mean()), "logN_cnn_mean": float(ln[S].mean())}

    # (v) partial correlations
    pc = {"noisy_het_cnn_r_given_logN_cnn": pcorr(h[N], r[N], ln[N]),
          "noisy_corr_het_cnn_r": float(np.corrcoef(h[N], r[N])[0, 1]),
          "noisy_corr_logN_cnn_r": float(np.corrcoef(ln[N], r[N])[0, 1]),
          "noisy_corr_het_cnn_logN_cnn": float(np.corrcoef(h[N], ln[N])[0, 1])}
    hb = pd.DataFrame({"ID": tr.ID}).merge(pd.read_parquet(DATA_DIR / "het_blocks_train.parquet")[["ID", "het4", "N_eff"]],
                                           on="ID", how="left")
    C = hb.het4.notna().values
    assert (C == ~N).all(), "het_blocks gate != ic_noise < 9.5"
    het4, lne = hb.het4.values, np.log(hb.N_eff.values)
    pc["clean_het4_r_base_given_logNeff"] = pcorr(het4[C], r[C], lne[C])
    ref_fp = SUB_DIR / f"{a.clean_ref}_oof.csv"
    if ref_fp.exists():
        r_ref = y - by_id(pd.read_csv(ref_fp), tr.ID)
        pc[f"clean_het4_r_{a.clean_ref}_given_logNeff"] = pcorr(het4[C], r_ref[C], lne[C])
        for lo, hi in ((0, 6), (6, 8), (8, 9.5)):
            S = C & (nz_tr >= lo) & (nz_tr < hi)
            pc[f"clean_het4_r_{a.clean_ref}_given_logNeff_band{lo:g}-{hi:g}"] = pcorr(het4[S], r_ref[S], lne[S])
        # extra: het_cnn on the clean rows that have an out-of-fold CNN value (same rows: het4 for comparison)
        Co = C & np.isfinite(h)
        if Co.sum() > 10:
            pc["clean_oof_n"] = int(Co.sum())
            pc[f"clean_oof_het_cnn_r_{a.clean_ref}_given_logN_cnn"] = pcorr(h[Co], r_ref[Co], ln[Co])
            pc[f"clean_oof_het_cnn_r_{a.clean_ref}_given_logNeff"] = pcorr(h[Co], r_ref[Co], lne[Co])
            pc[f"clean_oof_het4_r_{a.clean_ref}_given_logNeff"] = pcorr(het4[Co], r_ref[Co], lne[Co])
            pc["clean_oof_corr_het_cnn_het4"] = float(np.corrcoef(h[Co], het4[Co])[0, 1])

    gain = res["base_noisy"] - res["primary_noisy"]
    vs_n = res["nonly_noisy"] - res["primary_noisy"]
    bar = {"gain_vs_base": {"value": gain, "need": f">= {BAR['min_gain']}", "pass": bool(gain >= BAR["min_gain"])},
           "folds_improved": {"value": n_better, "need": f">= {BAR['min_folds']}/5",
                              "pass": bool(n_better >= BAR["min_folds"])},
           "perm_p": {"value": p_perm, "need": f"< {BAR['max_p']}", "pass": bool(p_perm < BAR["max_p"])},
           "gain_vs_nonly": {"value": vs_n, "need": f">= {BAR['min_vs_nonly']}",
                             "pass": bool(vs_n >= BAR["min_vs_nonly"])}}
    passed = all(v["pass"] for v in bar.values())

    nested = oof + off_p
    test_off = Xt @ coef_p
    rep = {
        "base": a.base, "het_run": str(het_dir), "noise_min": NOISE_MIN,
        "n": {"noisy_train": int(N.sum()), "noisy_test": int(Nt.sum()), "clean_train": int(C.sum())},
        "noisy_rmse": res,
        "fold_rmse_noisy": folds_tab, "folds_improved_primary": n_better,
        "folds_improved_nonly": int(sum(p < b for p, b in zip(folds_tab["nonly"], folds_tab["base"]))),
        "perm": {"n": a.n_perm, "seed": a.seed, "p": p_perm, "p_plus1": float((np.sum(perm <= res["primary_noisy"]) + 1)
                                                                             / (a.n_perm + 1)),
                 "null_mean": float(perm.mean()), "null_sd": float(perm.std()),
                 "null_q05": float(np.quantile(perm, 0.05)), "null_min": float(perm.min())},
        "boot_primary_minus_base_90": [float(np.quantile(d_base, 0.05)), float(np.quantile(d_base, 0.95))],
        "boot_primary_minus_nonly_90": [float(np.quantile(d_nonly, 0.05)), float(np.quantile(d_nonly, 0.95))],
        "coef_primary_full": {"cols": ["1", "het_cnn", "exp(0.25 logN_cnn)"], "coef": coef_p.tolist(),
                              "se": se_p.tolist()},
        "coef_nonly_full": {"cols": ["1", "exp(0.25 logN_cnn)"], "coef": coef_n.tolist(), "se": se_n.tolist()},
        "coef_primary_per_fold": coefs_p, "coef_nonly_per_fold": coefs_n, "clean_coef_transfer": CLEAN_COEF,
        "breakdown": breakdown, "logN_cnn_tercile_cuts": q.tolist(), "partial_corr": pc,
        "nested_overall": {"base": rmse(oof, y), "primary": rmse(nested, y),
                           "base_clean": rmse(oof[~N], y[~N]), "primary_clean": rmse(nested[~N], y[~N]),
                           "fold_rmse_base": [rmse(oof[folds == f], y[folds == f]) for f in range(N_FOLDS)],
                           "fold_rmse_primary": [rmse(nested[folds == f], y[folds == f]) for f in range(N_FOLDS)]},
        "distribution": {k: {"het_mean": float(v[0].mean()), "het_sd": float(v[0].std()),
                             "logN_mean": float(v[1].mean()), "logN_sd": float(v[1].std())}
                         for k, v in (("noisy_train", (h[N], ln[N])), ("noisy_test", (ht[Nt], lnt[Nt])))},
        "test_offset_if_applied": {"mean": float(test_off[Nt].mean()), "sd": float(test_off[Nt].std()),
                                   "train_full_fit_sd": float((X @ coef_p)[N].std())},
        "bar": bar, "bar_pass": passed, "out": a.out, "note": a.note,
        "cmd": f"python -m src.het_cnn_offset --base {a.base} --het {a.het}" + (f" --out {a.out}" if a.out else ""),
    }

    f3 = lambda v: f"{v:.3f}"
    print(f"noisy nested RMSE: base {f3(res['base_noisy'])} | primary {f3(res['primary_noisy'])} | N-only "
          f"{f3(res['nonly_noisy'])} | clean-coef transfer {f3(res['transfer_noisy'])}")
    for k in ("n", "base", "primary", "nonly", "transfer"):
        print(f"  folds {k:9s}", [v if k == "n" else round(v, 3) for v in folds_tab[k]])
    print(f"perm p {p_perm:.4f} (null mean {perm.mean():.3f} sd {perm.std():.3f} min {perm.min():.3f}) | boot 90% "
          f"primary-base {np.round(rep['boot_primary_minus_base_90'], 3).tolist()} primary-nonly "
          f"{np.round(rep['boot_primary_minus_nonly_90'], 3).tolist()}")
    print(f"coef primary {np.round(coef_p, 3).tolist()} se {np.round(se_p, 3).tolist()} | N-only "
          f"{np.round(coef_n, 3).tolist()}")
    for k, v in breakdown.items():
        pcv = v["pcorr_het_r_given_logN"]
        print(f"  {k:16s} n {v['n']:3d} base {f3(v['base'])} primary {f3(v['primary'])} nonly {f3(v['nonly'])} "
              f"transfer {f3(v['transfer'])} pcorr {'-' if pcv is None else f3(pcv)}")
    for k, v in pc.items():
        print(f"  {k}: {v if isinstance(v, int) else round(v, 3)}")
    for k, v in bar.items():
        print(f"BAR {k}: {v['value']:.4f} need {v['need']} -> {'PASS' if v['pass'] else 'FAIL'}")
    print(f"VERDICT: {'PASS' if passed else 'FAIL'}")

    if passed and a.out:
        pred = base + test_off * Nt
        sub = pd.DataFrame({"ID": te.ID.values, "hardness": pred})
        assert len(sub) == 1000 and np.isfinite(sub.hardness).all() and (sub.ID.values == ss_ids).all()
        assert np.array_equal(pred[~Nt], base[~Nt]) and np.array_equal(nested[~N], oof[~N])
        sub.to_csv(SUB_DIR / f"{a.out}.csv", index=False)
        pd.DataFrame({"ID": tr.ID, "hardness": nested}).to_csv(SUB_DIR / f"{a.out}_oof.csv", index=False)
        (SUB_DIR / f"{a.out}.json").write_text(json.dumps({
            "base": a.base, "mode": "het_cnn_offset", "het_run": str(het_dir),
            "gate": "raw ic_noise >= 9.5 (features_v3)", "coef_1_hetcnn_Ncnn": coef_p.tolist(),
            "train_n": int(N.sum()), "test_n": int(Nt.sum()), "bar": bar,
            "nested_cv_rmse": rep["nested_overall"]["primary"], "nested_noisy": res["primary_noisy"],
            "nested_clean": rep["nested_overall"]["primary_clean"],
            "fold_rmse": rep["nested_overall"]["fold_rmse_primary"], "fold_rmse_noisy": folds_tab["primary"],
            "notes": a.note}, indent=2))
        with open(EXP_DIR / "LEADERBOARD.md", "a") as fh:
            fh.write(f"| {a.out} (blend) | {rep['nested_overall']['primary']:.3f} |  | {a.base} + cross-fitted OLS "
                     f"offset on [1, het_cnn, exp(0.25 logN_cnn)] for raw ic_noise >= 9.5 ({N.sum()} train / "
                     f"{Nt.sum()} test; src/het_cnn_offset.py, het run {het_dir.name}); coef "
                     f"{np.round(coef_p, 2).tolist()}; noisy {res['base_noisy']:.3f} -> {res['primary_noisy']:.3f}, "
                     f"{n_better}/5 folds, perm p {p_perm:.3f}, vs N-only {vs_n:+.3f}. {a.note} |\n")
        print("written", SUB_DIR / f"{a.out}.csv")
    elif a.out:
        print(f"bar failed: {a.out} NOT written")

    rp = Path(a.report) if a.report else het_dir / f"stage2_{a.base}.json"
    rp.write_text(json.dumps(rep, indent=2))
    print("report", rp)


if __name__ == "__main__":
    main()
