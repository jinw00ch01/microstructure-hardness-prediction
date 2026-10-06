"""A40: last cheap checks against the feat4 base (cross-fitted probes, eda_cf), clean images (snr > 0.9) and all.
(a) a29 spatial size features: Moran's I of log area / dark indicator, local size-field spread, size-dark co-location.
(b) number-weighted per-grain grey level (variance scales as 1/N, not 1/N_eff, which is what a number-weighted mean of
    per-grain values would give): per phase number-weighted mean / sd of the grain median rn, number fractions in grey
    bins, within-phase correlation of grain size and grey level.
(c) absolute direction of elongation: area-weighted doubled-angle mean of grain orientation (cos 2t, sin 2t, R, |cos 2t|).
Usage: python a40_extra_checks.py CACHE_DIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from eda_cf import Reporter, base, probe, table

cache = Path(sys.argv[1])
t = table()
oof, inner = base(t)
R = Reporter(t, oof)
c = R.clean
G = pd.read_parquet(cache / "grains2_train.parquet")
rows = []
for i, d in G.groupby("ID"):
    f = {"ID": i}
    dk = d.med.values < 0.91
    A = d.area.values.astype(float)
    for ph, m in (("dk", dk), ("mx", ~dk)):
        v = d.med.values[m]
        f[f"x_nmean_{ph}"] = float(v.mean()) if m.sum() else np.nan
        f[f"x_nsd_{ph}"] = float(v.std()) if m.sum() > 2 else np.nan
        f[f"x_sizegrey_{ph}"] = float(spearmanr(np.log(A[m]), v)[0]) if m.sum() > 5 else np.nan
    for b, (lo, hi) in enumerate([(0, 0.80), (0.80, 0.86), (0.86, 0.91), (0.91, 0.95), (0.95, 0.98), (0.98, 1.01), (1.01, 1.04), (1.04, 9)]):
        f[f"x_nfrac_g{b}"] = float(((d.med.values >= lo) & (d.med.values < hi)).mean())
    th = d.ori.values
    w = A / A.sum()
    C2, S2 = float((w * np.cos(2 * th)).sum()), float((w * np.sin(2 * th)).sum())
    f.update({"x_ori_c2": C2, "x_ori_s2": S2, "x_ori_R": float(np.hypot(C2, S2)), "x_ori_abs_c2": float(abs(C2) / max(np.hypot(C2, S2), 1e-9))})
    rows.append(f)
X = pd.DataFrame(rows)
t = t.merge(X, on="ID", how="left").merge(pd.read_parquet(cache / "mi_train.parquet"), on="ID", how="left", suffixes=("", "_a29"))
groups = {
    "a29 Moran / size field / size-dark co-location": [x for x in ("mi_la_24", "mi_dk_24", "mi_la_48", "mi_dk_48", "lf_la_sd", "lf_la_sd_rel", "lf_dk_sd", "lf_la_dk_corr", "lf_la_range") if x in t.columns],
    "size-dark co-location only": ["lf_la_dk_corr"],
    "number-weighted grey per phase": ["x_nmean_dk", "x_nmean_mx", "x_nsd_dk", "x_nsd_mx"],
    "number fractions in grey bins": [f"x_nfrac_g{b}" for b in range(8)],
    "size-grey correlation per phase": ["x_sizegrey_dk", "x_sizegrey_mx"],
    "absolute elongation direction": ["x_ori_c2", "x_ori_s2", "x_ori_R", "x_ori_abs_c2"],
}
for nm, cols in groups.items():
    Xg = t[cols].astype(float).values
    for mode in ("clean", "all"):
        kw = {"fit_mask": c, "apply_mask": c} if mode == "clean" else {}
        pr = probe(Xg, inner, **kw)
        nul = R.perm_null(Xg, inner, n=20, **kw)
        print(R.line(f"{mode}: {nm}", pr, len(cols)) + f" | perm p~{(np.sum(nul >= R.gain(pr)) + 1) / 21:.2f}", flush=True)
print("\nSpearman with the blend_v4 residual (clean coarse / clean mid / clean all / all images):")
terc = R.terc
for col in sum(groups.values(), []):
    x = t[col].astype(float).values
    print(f"  {col:18s} " + " ".join(f"{spearmanr(x[m], R.br[m], nan_policy='omit')[0]:+.2f}" for m in
                                     (c & (terc == "coarse"), c & (terc == "mid"), c, np.ones(len(x), bool))))
