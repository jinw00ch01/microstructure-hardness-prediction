"""A16: feature-hardness Spearman table (train only) and the variance-vs-1/N figure."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from eda_common import OUT, train_table

t = train_table().merge(pd.read_parquet("../../data/features.parquet"), on="ID")
clean = (t.ic_ridge_snr > 0.5).values
rows = []
for c in t.columns:
    if c in ("ID", "hardness", "fold", "pred", "resid") or c.startswith("oof_"):
        continue
    x = t[c].astype(float)
    if x.nunique() < 3:
        continue
    rows.append({"feat": c, "rho_y": spearmanr(x, t.hardness, nan_policy="omit")[0],
                 "rho_y_clean": spearmanr(x[clean], t.hardness[clean], nan_policy="omit")[0],
                 "rho_resid": spearmanr(x, t.resid, nan_policy="omit")[0]})
R = pd.DataFrame(rows).sort_values("rho_y", key=np.abs, ascending=False)
R.round(4).to_csv(OUT / "feature_hardness_spearman.csv", index=False)
print(R.head(25).round(3).to_string(index=False))
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    N = t.cal_seg_count_density.values * 6.5536
    snr = t.ic_ridge_snr.values
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for m, lab, col in ((snr > 1.0, "clean (snr>1, n=%d)" % (snr > 1.0).sum(), "C0"), (snr <= 1.0, "rest (n=%d)" % (snr <= 1.0).sum(), "C1")):
        x = 1 / N[m]
        q = pd.qcut(x, 6 if m.sum() < 200 else 10, labels=False)
        g = pd.DataFrame({"x": x, "r2": t.resid.values[m] ** 2}).groupby(q).agg(["mean", "sem", "size"])
        ax[0].errorbar(g[("x", "mean")], g[("r2", "mean")], yerr=g[("r2", "sem")], fmt="o-", color=col, label=lab)
    xx = np.linspace(0, (1 / N).max(), 50)
    ax[0].plot(xx, 28706 * xx, "k--", label="label noise 28700/N (fit on clean, a=0)")
    ax[0].set_xlabel("1 / N_cal  (calibrated grain count)")
    ax[0].set_ylabel("mean squared OOF residual (blend_v2)")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Residual variance grows ~ 1/N")
    ax[0].set_xlim(0, np.quantile(1 / N, 0.995))
    ax[0].set_ylim(0, 700)
    b, c = 28227.0, 10.86   # joint fit v = b/N + c/snr on all train residuals (a -> 0)
    snr_c = np.clip(snr, 0.05, None)
    Ng = pd.qcut(N, 3, labels=["coarse", "mid", "fine"])
    Sg = pd.qcut(snr, 3, labels=["noisy", "mid", "clean"])
    labs, obs, flo = [], [], []
    for ng in ["coarse", "mid", "fine"]:
        for sg in ["noisy", "mid", "clean"]:
            m = np.asarray((Ng == ng) & (Sg == sg))
            labs.append(f"{ng}\n{sg}")
            obs.append(np.sqrt(np.mean(t.resid.values[m] ** 2)))
            flo.append(np.sqrt(np.mean(b / N[m])))
    xi = np.arange(len(labs))
    ax[1].bar(xi - 0.2, obs, 0.4, label="blend_v2 OOF RMSE")
    ax[1].bar(xi + 0.2, flo, 0.4, label="label-noise floor sqrt(mean b/N)")
    ax[1].set_xticks(xi, labs)
    ax[1].set_ylabel("RMSE")
    ax[1].legend(fontsize=8)
    ax[1].set_title("Clean images already sit at the floor; excess is in noisy ones")
    ax[1].tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_var_vs_invN.png", dpi=90)
    print("saved figure")
except ImportError as e:
    print("no matplotlib", e)
