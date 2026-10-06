"""A14: train vs test marginal distributions of key image statistics (unsupervised; nothing is fitted on test)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from eda_common import DATA_DIR, load_feats

cache = Path(sys.argv[1])
f = load_feats()
f = f.merge(pd.concat([pd.read_parquet(cache / "grainstats_train.parquet"), pd.read_parquet(cache / "grainstats_test.parquet")]), on="ID", how="left") \
    if (cache / "grainstats_test.parquet").exists() else f
f["N_cal"] = f.cal_seg_count_density * 6.5536
f["is_test"] = f.ID.str.startswith("TEST")
cols = ["ic_noise", "ic_spec_slope", "ic_ridge_snr", "ic_bg_range", "ic_bg_level", "N_cal", "ic_acg_len50_gm", "ic_acg_len50_par",
        "ic_acg_len50_aspect", "cal_ic_seg_fd91", "ic_fd2_91", "ic_pore60_n", "ic_pore60_frac", "ic_st_coh", "ic_angle_abs",
        "seg_count_density", "q_illum_range", "lv_M"]
rows = []
for c in cols:
    a, b = f.loc[~f.is_test, c].dropna(), f.loc[f.is_test, c].dropna()
    ks = ks_2samp(a, b)
    rows.append({"feat": c, "train_med": a.median(), "test_med": b.median(), "smd": (b.mean() - a.mean()) / a.std(),
                 "ks": ks.statistic, "p": ks.pvalue})
print(pd.DataFrame(rows).round(4).to_string(index=False))
# quality groups: noise x blur terciles (cut points from train only)
tr = f[~f.is_test]
qn = np.quantile(tr.ic_noise, [1 / 3, 2 / 3])
qb = np.quantile(tr.ic_ridge_snr, [1 / 3, 2 / 3])
f["noise_g"] = np.digitize(f.ic_noise, qn)
f["snr_g"] = np.digitize(f.ic_ridge_snr, qb)
f["N_g"] = np.digitize(f.N_cal, np.quantile(tr.N_cal, [1 / 3, 2 / 3]))
for g in ("noise_g", "snr_g", "N_g"):
    print(g, "train", np.bincount(f.loc[~f.is_test, g], minlength=3) / (~f.is_test).sum(), "test", np.round(np.bincount(f.loc[f.is_test, g], minlength=3) / f.is_test.sum(), 3))
