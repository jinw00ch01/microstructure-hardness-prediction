"""A1: heteroscedasticity of blend OOF residuals; label structure checks (ID order, discreteness)."""
import numpy as np
import pandas as pd
from scipy import stats

from eda_common import OUT, rmse, train_table

t = train_table()
y, r, p = t.hardness.values, t.resid.values, t.pred.values
print(f"blend nested OOF RMSE {rmse(t.pred, y):.3f}; corr(pred, y) {np.corrcoef(p, y)[0, 1]:.3f}")
print(f"calibration: slope of y on pred {np.polyfit(p, y, 1)[0]:.3f}")

# ---------------------------------------------------------------- terciles by structure length
for col in ["ic_acg_len50_gm", "cal_ic_acg_len50_par", "seg_count_density", "cal_seg_count_density"]:
    q = pd.qcut(t[col], 3, labels=["lo", "mid", "hi"])
    g = t.groupby(q, observed=True)
    print(f"\n{col}")
    print(g.apply(lambda d: pd.Series({
        "n": len(d), "col_med": d[col].median(), "rmse": rmse(d.pred, d.hardness), "y_std": d.hardness.std(),
        "pred_std": d.pred.std(), "y_mean": d.hardness.mean(), "resid_mean": d.resid.mean(),
        "corr": np.corrcoef(d.pred, d.hardness)[0, 1]}), include_groups=False).round(3).to_string())

# ---------------------------------------------------------------- residual vs ID order
idn = t.ID.str[-6:].astype(int).values
o = np.argsort(idn)
ro = r[o]
print("\nID order: Spearman(resid, ID)", stats.spearmanr(idn, r)[0].round(3),
      " Spearman(y, ID)", stats.spearmanr(idn, y)[0].round(3),
      " Spearman(|r|, ID)", stats.spearmanr(idn, np.abs(r))[0].round(3))
ac = [np.corrcoef(ro[:-k], ro[k:])[0, 1] for k in (1, 2, 3, 5, 10)]
print("lag autocorr of residuals in ID order (1,2,3,5,10):", np.round(ac, 3))
yo = y[o]
print("lag autocorr of y in ID order (1,2,3,5,10):", np.round([np.corrcoef(yo[:-k], yo[k:])[0, 1] for k in (1, 2, 3, 5, 10)], 3))
# block means of residual by ID decile
print("resid mean by ID decile:", np.round(pd.Series(r).groupby(pd.qcut(idn, 10, labels=False)).mean().values, 2))
print("missing IDs in 1..max:", sorted(set(range(1, idn.max() + 1)) - set(idn))[:20], "max", idn.max())

# ---------------------------------------------------------------- discreteness / clusters in y
ys = np.sort(y)
for dec in (0, 1):
    vc = pd.Series(np.round(ys, dec)).value_counts()
    print(f"rounded to {dec} dp: max multiplicity {vc.max()}, n distinct {len(vc)}")
# kernel density peaks (bandwidth 1 HV)
grid = np.linspace(150, 266, 1161)
kde = stats.gaussian_kde(y, bw_method=1.0 / y.std())
dens = kde(grid)
pk = (dens[1:-1] > dens[:-2]) & (dens[1:-1] > dens[2:])
print("KDE(bw=1HV) local maxima:", np.round(grid[1:-1][pk], 1))
# test for lattice: power of exp(2 pi i y / d) for candidate spacings
for d in (0.5, 1, 2, 2.5, 5, 10):
    R = np.abs(np.mean(np.exp(2j * np.pi * y / d)))
    print(f"lattice spacing {d}: R={R:.3f} (uniform expectation ~{1 / np.sqrt(len(y)):.3f})")
print("residual normality: skew %.3f kurt %.3f; Shapiro p=%.2g" % (stats.skew(r), stats.kurtosis(r), stats.shapiro(r)[1]))
print("y log-normal check: skew(log y) %.3f vs skew(y) %.3f" % (stats.skew(np.log(y)), stats.skew(y)))
t[["ID", "hardness", "fold", "pred", "resid"]].to_csv(OUT / "blend_v2_nested_oof.csv", index=False)
