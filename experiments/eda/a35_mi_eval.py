"""A35: is hardness an area-weighted mean of a nonlinear per-grain function of size and phase (beyond feat4)?
Cross-fitted probes (eda_cf: feat4 ridge base, inner cross-fitted residuals) for the a34 MI groups, fitted and applied
on clean images (ic_ridge_snr > 0.9, where the watershed is reliable) and on all images; permutation nulls.
Also direct clean-image models: y ~ area-weighted bins vs number-weighted bins (+ the same porosity column), and the
probe coefficients of the per-phase size bins (shape of f).  Usage: python a35_mi_eval.py CACHE_DIR"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

from eda_cf import Reporter, base, probe, table
from eda_common import rmse

cache = Path(sys.argv[1])
t = table()
oof, inner = base(t)
R = Reporter(t, oof)
MI = pd.read_parquet(cache / "mig_train.parquet")
t = t.merge(MI, on="ID", how="left")
y, fold, clean, terc = t.hardness.values, t.fold.values, R.clean, R.terc
print(f"base (feat4 ridge) CV {rmse(oof, y):.3f}; blend_v4 nested {rmse(t.pred, y):.3f}; clean n={clean.sum()} "
      f"(coarse {np.sum(clean & (terc == 'coarse'))}, mid {np.sum(clean & (terc == 'mid'))})")
groups = {
    "area bins x phase (16)": r"^mi_a_",
    "number bins x phase (16)": r"^mi_n_",
    "smooth area basis x phase (8)": r"^mi_s_",
    "area + number bins (32)": r"^mi_[an]_",
    "N_ws, N_eff, log(Neff/N), Neff by phase": r"^mi_(N|Neff|log_neff_over_n|Neff_dk|Neff_mx)$",
    "grey-level bins x phase (8)": r"^mi_g_",
    "matrix grey spread": r"^mi_mx_med_sd$|^mi_iqr_",
    "elongation x phase (8)": r"^mi_e_",
    "all MI": r"^mi_",
}
res = {}
for gname, rx in groups.items():
    cols = [c for c in MI.columns if re.search(rx, c)]
    X = t[cols].astype(float).values
    for mode in ("clean", "all"):
        kw = {"fit_mask": clean, "apply_mask": clean} if mode == "clean" else {}
        pr = probe(X, inner, **kw)
        res[(gname, mode)] = (X, kw, pr)
        print(R.line(f"{mode}: {gname}", pr, len(cols)), flush=True)
print("\npermutation nulls (20 row shuffles of the candidate block) for blend_v4 RMSE gain:")
for key in [("area bins x phase (16)", "clean"), ("number bins x phase (16)", "clean"), ("smooth area basis x phase (8)", "clean"),
            ("N_ws, N_eff, log(Neff/N), Neff by phase", "clean"), ("elongation x phase (8)", "clean"),
            ("grey-level bins x phase (8)", "clean"), ("all MI", "clean"), ("all MI", "all")]:
    X, kw, pr = res[key]
    obs = R.gain(pr)
    nul = R.perm_null(X, inner, n=20, **kw)
    print(f"  {key[1]:5s} {key[0]:42s} gain {obs:+.3f} | null mean {nul.mean():+.3f} max {nul.max():+.3f} p~{(np.sum(nul >= obs) + 1) / 21:.2f}")

print("\nDirect clean-image models (snr>0.9), CV on the shared folds; y ~ ridge(features); base feat4 OOF for reference:")
c = clean
pore = t[[x for x in t.columns if x.startswith("ic_pore")][:1]].astype(float).values
print(f"  porosity column used: {[x for x in t.columns if x.startswith('ic_pore')][:1]}")


def direct(X):
    X = np.asarray(X, float)[c]
    yy, ff = y[c], fold[c]
    o = np.zeros(len(yy))
    for k in range(5):
        a, b = ff != k, ff == k
        med = np.nanmedian(X[a], 0)
        Xa, Xb = np.where(np.isfinite(X[a]), X[a], med), np.where(np.isfinite(X[b]), X[b], med)
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
        o[b] = RidgeCV(alphas=np.logspace(-3, 4, 36)).fit((Xa - mu) / sd, yy[a]).predict((Xb - mu) / sd)
    tc = terc[c]
    return f"CV {rmse(o, yy):.2f} | " + " ".join(f"{g} {rmse(o[tc == g], yy[tc == g]):.2f}" for g in ("coarse", "mid", "fine"))


A_cols = [x for x in MI.columns if x.startswith("mi_a_")]
N_cols = [x for x in MI.columns if x.startswith("mi_n_")]
tc = terc[c]
print(f"  feat4 base OOF on these rows      CV {rmse(oof[c], y[c]):.2f} | " + " ".join(f"{g} {rmse(oof[c][tc == g], y[c][tc == g]):.2f}" for g in ("coarse", "mid", "fine")))
print(f"  blend_v4 nested on these rows     CV {rmse(t.pred.values[c], y[c]):.2f} | " + " ".join(f"{g} {rmse(t.pred.values[c][tc == g], y[c][tc == g]):.2f}" for g in ("coarse", "mid", "fine")))
for nm, X in (("area-weighted bins x phase", t[A_cols].values), ("number-weighted bins x phase", t[N_cols].values),
              ("area bins + porosity", np.column_stack([t[A_cols].values, pore])),
              ("number bins + porosity", np.column_stack([t[N_cols].values, pore])),
              ("phase area fraction + porosity", np.column_stack([t.mi_fd.values, pore])),
              ("fd + porosity + log N_ws + log N_eff", np.column_stack([t.mi_fd.values, pore, np.log(t.mi_N.values), np.log(t.mi_Neff.values)]))):
    print(f"  {nm:38s} {direct(X)}")

# shape of f: probe coefficients (area-bin probe fitted on all clean rows' cross-fitted residuals is not available
# outside folds, so use the outer base residual on clean rows; descriptive only)
Xc = t.loc[c, A_cols].values
rc = (y - oof)[c]
mu, sd = Xc.mean(0), Xc.std(0) + 1e-9
m = RidgeCV(alphas=np.logspace(-1, 6, 29)).fit((Xc - mu) / sd, rc)
coef = pd.Series(m.coef_ / sd, index=A_cols)
print(f"\nresidual ~ area bins on clean rows (descriptive, alpha {m.alpha_:.1f}); HV per unit area fraction relative to the mean:")
print("  bin edges px:", np.round(np.exp([np.log(12), *np.log([50, 100, 200, 400, 800, 1600, 3200])])).astype(int).tolist())
for ph in ("dk", "mx"):
    v = coef[[f"mi_a_{ph}_{b}" for b in range(8)]]
    print(f"  {ph}: " + " ".join(f"{x:+.0f}" for x in v.values))
occ = t.loc[c, A_cols].mean()
print("  mean area fraction per bin (clean): dk " + " ".join(f"{occ[f'mi_a_dk_{b}']:.3f}" for b in range(8)) +
      " | mx " + " ".join(f"{occ[f'mi_a_mx_{b}']:.3f}" for b in range(8)))

print("\nSpearman with blend_v4 residual on clean images (top 15 by |coarse| ; null SE coarse ~0.17, mid ~0.17):")
rows = []
for col in [x for x in MI.columns if x != "ID"]:
    x = t[col].astype(float).values
    r_ = {"feat": col, "clean": spearmanr(x[c], t.resid[c], nan_policy="omit")[0]}
    for g in ("coarse", "mid", "fine"):
        mm = c & (terc == g)
        r_[g] = spearmanr(x[mm], t.resid[mm], nan_policy="omit")[0]
    rows.append(r_)
S = pd.DataFrame(rows).set_index("feat")
print(S.reindex(S.coarse.abs().sort_values(ascending=False).index).head(15).round(2).to_string())
S.to_csv(Path(__file__).parent / "a35_mi_spearman_clean.csv")
