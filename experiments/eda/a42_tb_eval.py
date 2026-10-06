"""A42: cross-fitted probes (eda_cf, feat4 ridge base) for the a41 per-grain interior texture and boundary-contrast
groups, clean-only (snr > 0.9) and all images, with 20-shuffle permutation nulls; Spearman with the blend_v4 residual on
clean images by N tercile, against a permutation null of the max |rho|.  Usage: python a42_tb_eval.py CACHE_DIR"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from eda_cf import Reporter, base, probe, table

cache = Path(sys.argv[1])
t = table()
oof, inner = base(t)
R = Reporter(t, oof)
TB = pd.read_parquet(cache / "tb_train.parquet")
t = t.merge(TB, on="ID", how="left")
c, terc = R.clean, R.terc
print(f"clean n={c.sum()}; clean rows with texture features: {int((c & t.tx_lap_mx_wav.notna()).sum())}")
groups = {
    "texture level vs wavelet noise (per phase)": r"^tx_(lap|std|band)_(dk|mx)_wav$",
    "texture level vs pore noise (per phase)": r"^tx_(lap|std|band)_(dk|mx)_pore$",
    "texture spread across grains (sd, iqr, excess)": r"^tx_(lap|std|band)_(dk|mx)_(sd|iqr|xs)$",
    "texture dark-matrix difference": r"^tx_(lap|std|band)_dkmx$",
    "texture vs size / level (within image)": r"^tx_(lap|std|band)_(dk|mx)_rho_",
    "all texture": r"^tx_|^tb_wav$|^tb_pore_lap_rel$",
    "boundary contrast level per phase": r"^(bc|bd|bcn)_(dk|mx)_med$|^bc_dkmx$",
    "boundary contrast spread across grains": r"^(bc|bcn)_(dk|mx)_(sd|iqr)$",
    "boundary contrast vs size": r"^bc_(dk|mx)_rho_size$",
    "all boundary": r"^(bc|bd|bcn)_",
}
for gname, rx in groups.items():
    cols = [x for x in TB.columns if re.search(rx, x)]
    X = t[cols].astype(float).values
    for mode in ("clean", "all"):
        kw = {"fit_mask": c, "apply_mask": c} if mode == "clean" else {}
        pr = probe(X, inner, **kw)
        nul = R.perm_null(X, inner, n=20, **kw)
        print(R.line(f"{mode}: {gname}", pr, len(cols)) +
              f" | perm null mean {nul.mean():+.3f} max {nul.max():+.3f} p~{(np.sum(nul >= R.gain(pr)) + 1) / 21:.2f}", flush=True)

num = [x for x in TB.columns if x not in ("ID", "tb_n")]
for nm, m in (("clean coarse+mid", c & np.isin(terc, ["coarse", "mid"])), ("clean all", c)):
    d = t[m]
    r = d.resid.values
    Xr = np.column_stack([rankdata(d[x].fillna(d[x].median())) for x in num])
    Z = (Xr - Xr.mean(0)) / (Xr.std(0) + 1e-12)

    def rr(v):
        z = (v - v.mean()) / v.std()
        return Z.T @ z / len(z)

    obs = rr(rankdata(r))
    rng = np.random.default_rng(0)
    mx = np.array([np.abs(rr(rankdata(rng.permutation(r)))).max() for _ in range(300)])
    o = np.argsort(-np.abs(obs))
    print(f"\n{nm} (n={m.sum()}): max |rho| {np.abs(obs).max():.3f}; null of max |rho| median {np.median(mx):.3f}, 95% {np.quantile(mx, 0.95):.3f}")
    for j in o[:10]:
        x = t[num[j]].values
        print(f"  {num[j]:22s} rho {obs[j]:+.3f} | coarse {spearmanr(x[c & (terc == 'coarse')], R.br[c & (terc == 'coarse')], nan_policy='omit')[0]:+.2f} "
              f"mid {spearmanr(x[c & (terc == 'mid')], R.br[c & (terc == 'mid')], nan_policy='omit')[0]:+.2f} "
              f"fine {spearmanr(x[c & (terc == 'fine')], R.br[c & (terc == 'fine')], nan_policy='omit')[0]:+.2f} | "
              f"rho with y (clean) {spearmanr(x[c], t.hardness.values[c], nan_policy='omit')[0]:+.2f}")
