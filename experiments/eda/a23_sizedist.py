"""A23: grain-size distribution aggregates (Hall-Petch per grain, width / bimodality) from the cached watershed grains.
Only meaningful where segmentation works (ic_ridge_snr > ~0.9).  Usage: python a23_sizedist.py CACHE_DIR"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from eda_common import train_table, OUT

cache = Path(sys.argv[1])
g = pd.read_parquet(cache / "grains_train.parquet")
g = g[g.area >= 12]
rows = []
for i, d in g.groupby("ID"):
    A = d.area.values.astype(float)
    dk = d.med.values < 0.91
    intr = ~d.border.values
    eq = np.sqrt(A)
    w = A / A.sum()
    f = {"ID": i}
    f["sd_hp_area"] = float(np.sum(w / np.sqrt(eq)))            # area-weighted <d^-1/2>
    f["sd_hp_num"] = float(np.mean(1 / np.sqrt(eq)))             # number-weighted
    f["sd_hp_of_mean"] = float(1 / np.sqrt(np.sqrt(A.mean())))
    f["sd_hp_excess"] = f["sd_hp_area"] / f["sd_hp_of_mean"]
    Ai = A[intr] if intr.sum() >= 3 else A
    f["sd_cv_int"] = float(Ai.std() / Ai.mean())
    f["sd_logsd_int"] = float(np.log(Ai).std())
    f["sd_w_over_n"] = float((Ai ** 2).sum() / Ai.sum() / Ai.mean())
    f["sd_small_frac"] = float(A[A < 0.25 * np.median(A)].sum() / A.sum())
    f["sd_big_frac"] = float(A[A > 4 * np.median(A)].sum() / A.sum())
    f["sd_skew_log"] = float(pd.Series(np.log(Ai)).skew())
    for nm, m in (("dk", dk), ("mx", ~dk)):
        if m.sum() >= 2:
            f[f"sd_hp_area_{nm}"] = float(np.sum(A[m] / np.sqrt(eq[m])) / A[m].sum())
            f[f"sd_logsd_{nm}"] = float(np.log(A[m]).std())
        else:
            f[f"sd_hp_area_{nm}"] = np.nan
            f[f"sd_logsd_{nm}"] = np.nan
    f["sd_size_ratio_dk_mx"] = float(np.log(A[dk].mean() / A[~dk].mean())) if dk.sum() >= 2 and (~dk).sum() >= 2 else np.nan
    rows.append(f)
S = pd.DataFrame(rows)
S.to_parquet(cache / "sd_train.parquet")
t = train_table().merge(S, on="ID", how="left")
N = t.cal_seg_count_density * 6.5536
t["terc"] = pd.qcut(N, 3, labels=["coarse", "mid", "fine"])
for nm, m0 in (("snr>0.9", t.ic_ridge_snr > 0.9), ("snr>1.2", t.ic_ridge_snr > 1.2)):
    print("==", nm, m0.sum())
    for c in [c for c in S.columns if c != "ID"]:
        out = [f"{g_} y{spearmanr(t[c][m0 & (t.terc == g_)], t.hardness[m0 & (t.terc == g_)], nan_policy='omit')[0]:+.2f} r{spearmanr(t[c][m0 & (t.terc == g_)], t.resid[m0 & (t.terc == g_)], nan_policy='omit')[0]:+.2f}" for g_ in ("coarse", "mid", "fine")]
        print(f"{c:20s} all y{spearmanr(t[c][m0], t.hardness[m0], nan_policy='omit')[0]:+.3f} r{spearmanr(t[c][m0], t.resid[m0], nan_policy='omit')[0]:+.3f} | " + " | ".join(out))
