"""A7: Spearman of every existing feature (v1/v2/v3/cal + a4 grain stats) with blend OOF residuals, all and coarse
(top third of ic_acg_len50_gm), with a permutation null for the max |rho|.  Usage: python a7_resid_scan.py CACHE_DIR"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from eda_common import DATA_DIR, OUT, train_table

cache = Path(sys.argv[1])
t = train_table().merge(pd.read_parquet(DATA_DIR / "features.parquet"), on="ID").merge(pd.read_parquet(cache / "grainstats_train.parquet"), on="ID")
feat_cols = [c for c in t.columns if c not in ("ID", "hardness", "fold", "pred", "resid") and not c.startswith("oof_")]
coarse = (t.ic_acg_len50_gm >= t.ic_acg_len50_gm.quantile(2 / 3)).values
rng = np.random.default_rng(0)
res = []
for c in feat_cols:
    x = t[c].values.astype(float)
    if np.nanstd(x) == 0:
        continue
    res.append((c, spearmanr(x, t.resid, nan_policy="omit")[0], spearmanr(x[coarse], t.resid[coarse], nan_policy="omit")[0],
                spearmanr(x, t.resid.abs(), nan_policy="omit")[0]))
R = pd.DataFrame(res, columns=["feat", "rho_all", "rho_coarse", "rho_absr"])
for nm, m in (("coarse", coarse), ("all", np.ones(len(t), bool))):
    X = t.loc[m, R.feat].rank().values
    X = np.nan_to_num((X - np.nanmean(X, 0)) / np.nanstd(X, 0))
    mx = []
    for _ in range(200):
        rr = rng.permutation(t.resid[m].rank().values)
        rr = (rr - rr.mean()) / rr.std()
        mx.append(np.abs((X * rr[:, None]).mean(0)).max())
    print(f"null 95% max|rho| {nm}: {np.quantile(mx, 0.95):.3f}")
print(R.reindex(R.rho_coarse.abs().sort_values(ascending=False).index).head(15).round(3).to_string())
R.to_csv(OUT / "a7_resid_feature_corr.csv", index=False)
