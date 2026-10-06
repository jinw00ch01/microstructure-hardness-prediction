"""A9: do grain-level aggregates (a8) explain blend OOF residuals?  Spearman per feature (all / clean / clean-coarse),
and in-fold ridge / small LGB on the residual (shared folds; inner-CV alpha) -> RMSE change of corrected OOF."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eda_common import rmse, train_table

cache = Path(sys.argv[1])
t = train_table().merge(pd.read_parquet(cache / "gf_train.parquet"), on="ID")
gcols = [c for c in t.columns if c.startswith("gf_")]
clean = (t.ic_ridge_snr > 0.5).values
Nc = t.cal_seg_count_density.values
coarseN = Nc <= np.quantile(Nc, 1 / 3)
rows = []
for c in gcols:
    x = t[c].values.astype(float)
    rows.append({"feat": c,
                 "rho_y_clean": spearmanr(x[clean], t.hardness[clean], nan_policy="omit")[0],
                 "rho_r_all": spearmanr(x, t.resid, nan_policy="omit")[0],
                 "rho_r_clean": spearmanr(x[clean], t.resid[clean], nan_policy="omit")[0],
                 "rho_r_clean_coarse": spearmanr(x[clean & coarseN], t.resid[clean & coarseN], nan_policy="omit")[0],
                 "rho_absr_clean": spearmanr(x[clean], t.resid.abs()[clean], nan_policy="omit")[0]})
R = pd.DataFrame(rows).set_index("feat").round(3)
print(f"clean n={clean.sum()}, clean&coarseN n={(clean & coarseN).sum()}")
print(R.to_string())


def cv_resid(mask, cols, name):
    d = t[mask]
    X = d[cols].fillna(d[cols].median()).values
    r = d.resid.values
    fold = d.fold.values
    corr = np.zeros(len(r))
    for f in range(5):
        tr_, te_ = fold != f, fold == f
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 21)))
        m.fit(X[tr_], r[tr_])
        corr[te_] = m.predict(X[te_])
    print(f"{name:40s} n={len(r)}  RMSE {rmse(r, 0):.3f} -> {rmse(r - corr, 0):.3f}   (coarse part: "
          f"{rmse(r[coarseN[mask]], 0):.3f} -> {rmse((r - corr)[coarseN[mask]], 0):.3f})")


cv_resid(clean, gcols, "clean: all gf_ features")
cv_resid(np.ones(len(t), bool), gcols, "all images: all gf_ features")
for grp, pref in (("brightness bins", "gf_rb_"), ("matrix/dark levels", "gf_mx_|gf_dk_mean|gf_dk_sd|gf_lvl"),
                  ("texture", "gf_tex"), ("shape/strain", "gf_strain|gf_logasp|gf_align"),
                  ("clusters/contacts", "gf_dd|gf_dm|gf_dk_n|gf_dk_max|gf_dk_cl|gf_dk_span|gf_nb")):
    cols = [c for c in gcols if pd.Series([c]).str.contains(pref).iloc[0]]
    cv_resid(clean, cols, f"clean: {grp} ({len(cols)})")
