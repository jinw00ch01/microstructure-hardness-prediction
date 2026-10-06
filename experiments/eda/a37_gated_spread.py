"""A37: SNR-gated raw local grain-size spread (fixed per-image transform; train and test rows of existing files).
On clean images the raw v4 map spreads (v4r_la_*) and the watershed log-area field spreads (lf*_sd_rel) track the
blend_v4 residual better than the calibrated v4c spreads that feat4 uses; on noisy images they are garbage.
gate g = clip((ic_ridge_snr - 0.7) / 0.4, 0, 1)   (0 below snr 0.7, 1 above 1.1)
gsp_g = g;  gsp_<col> = g * col  for col in v4r_la_{sd,sd4,rng4,max_m_mean,q90_m_q10}, lf32_sd_rel, lf48_sd_rel,
lf32_max_minus_w  (NaN lf values -> 0 where g = 0, else left NaN for the model's median imputation).
Usage: python a37_gated_spread.py OUT_PARQUET"""
import sys

import numpy as np
import pandas as pd

from eda_common import DATA_DIR

COLS_V4R = ["v4r_la_sd", "v4r_la_sd4", "v4r_la_rng4", "v4r_la_max_m_mean", "v4r_la_q90_m_q10"]
COLS_LF = ["lf32_sd_rel", "lf48_sd_rel", "lf32_max_minus_w"]
d = (pd.read_parquet(DATA_DIR / "features_v3.parquet", columns=["ID", "ic_ridge_snr"])
     .merge(pd.read_parquet(DATA_DIR / "features_v4.parquet", columns=["ID"] + COLS_V4R), on="ID")
     .merge(pd.read_parquet(DATA_DIR / "eda_feats_lf.parquet", columns=["ID"] + COLS_LF), on="ID"))
g = np.clip((d.ic_ridge_snr.values - 0.7) / 0.4, 0.0, 1.0)
out = pd.DataFrame({"ID": d.ID, "gsp_g": g})
for c in COLS_V4R + COLS_LF:
    x = d[c].values.astype(float)
    out[f"gsp_{c}"] = np.where(g == 0, 0.0, g * x)
out.to_parquet(sys.argv[1])
print(out.shape, "train", out.ID.str.startswith("TRAIN").sum(), "test", out.ID.str.startswith("TEST").sum(),
      "nan", float(out.isna().mean().mean()), "gate>0 train", float((g[out.ID.str.startswith("TRAIN")] > 0).mean()),
      "test", float((g[out.ID.str.startswith("TEST")] > 0).mean()))
