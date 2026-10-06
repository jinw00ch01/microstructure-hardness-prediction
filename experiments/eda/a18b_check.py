"""A18b: robust phase measures vs existing ones on clean images, by N tercile (Spearman with y and with blend residual)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from eda_common import train_table

cache = Path(sys.argv[1])
t = train_table().merge(pd.read_parquet(cache / "rp_train.parquet"), on="ID")
N = t.cal_seg_count_density * 6.5536
t["terc"] = pd.qcut(N, 3, labels=["coarse", "mid", "fine"])
cols = sys.argv[2].split(",") if len(sys.argv) > 2 else ["rp_fd_area", "rp_fd90_area", "rp_fd88_area", "rp_fd92_area", "rp_fd_num", "rp_fd90_num",
        "rp_fd_area_int", "ic_seg_fd91", "cal_ic_seg_fd91", "seg_fd15", "ic_gmm_w", "rp_thr", "rp_contrast", "rp_d_dk", "rp_d_mx",
        "rp_pore_frac", "rp_border_dk_share", "rp_big0_dark", "rp_big0_share", "rp_logasp_dk", "rp_logasp_mx", "rp_ratio_mx_sd", "rp_ratio_dk_sd"]
for nm, m0 in (("clean snr>0.5", t.ic_ridge_snr > 0.5), ("clean snr>1", t.ic_ridge_snr > 1.0), ("all", t.ic_ridge_snr > -1)):
    print("==", nm, "n=%d" % m0.sum())
    for c in cols:
        out = []
        for g in ["coarse", "mid", "fine"]:
            m = m0 & (t.terc == g)
            out.append(f"{g} y{spearmanr(t[c][m], t.hardness[m], nan_policy='omit')[0]:+.2f} r{spearmanr(t[c][m], t.resid[m], nan_policy='omit')[0]:+.2f}")
        print(f"{c:18s} all y{spearmanr(t[c][m0], t.hardness[m0], nan_policy='omit')[0]:+.3f} r{spearmanr(t[c][m0], t.resid[m0], nan_policy='omit')[0]:+.3f} | " + " | ".join(out))
