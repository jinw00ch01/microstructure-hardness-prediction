"""A17b: leakage-free residual probes.  Naive probes on blend OOF residuals are biased negative (the OOF residuals of
training folds were produced by models that saw the held-out fold).  Here, per outer fold k:
  base   = RidgeCV on v3+cal features (the feat2_v3cal_ridge setup) fitted on the 4 training folds -> prediction on fold k
  inner  = 4-fold cross-fitted base residuals inside the training folds (never touch fold k)
  probe  = RidgeCV(representation -> inner residual), applied to fold k
Reports OOF corr(probe, base residual) and RMSE(base) -> RMSE(base + probe), overall / per N tercile, all and clean.
Usage: python a17b_probes_crossfit.py CACHE_DIR"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from eda_common import DATA_DIR, OUT, rmse, train_table

cache = Path(sys.argv[1])
t = train_table()
y = t.hardness.values
fold = t.fold.values
N = t.cal_seg_count_density.values * 6.5536
terc = np.asarray(pd.qcut(N, 3, labels=["coarse", "mid", "fine"]))
clean = t.ic_ridge_snr.values > 0.5
DROP = ["cal_ic_seg_L_", "cal_ic_seg_mx_area_mean", "cal_seg_area_cv", "cal_segdk_area_cv"]
v3 = pd.read_parquet(DATA_DIR / "features_v3.parquet")
cal = pd.read_parquet(DATA_DIR / "features_cal.parquet")
base_cols = [c for c in list(v3.columns[1:]) + list(cal.columns[1:]) if not any(re.search(p, c) for p in DROP)]
XB = t[base_cols].astype(float).replace([np.inf, -np.inf], np.nan)
XB = XB.fillna(XB.median()).values


def ridge_fit_predict(Xtr, ytr, Xte, alphas=np.logspace(-2, 4, 40)):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    m = RidgeCV(alphas=alphas).fit((Xtr - mu) / sd, ytr)
    return m.predict((Xte - mu) / sd)


# base OOF and inner cross-fitted residuals (fixed for all representations)
base_oof = np.zeros(len(y))
inner_res = {}
for k in range(5):
    tr_ = np.where(fold != k)[0]
    te_ = np.where(fold == k)[0]
    base_oof[te_] = ridge_fit_predict(XB[tr_], y[tr_], XB[te_])
    ir = np.zeros(len(tr_))
    for j in sorted(set(fold[tr_])):
        a = tr_[fold[tr_] != j]
        b = fold[tr_] == j
        ir[b] = y[tr_][b] - ridge_fit_predict(XB[a], y[a], XB[tr_][b])
    inner_res[k] = (tr_, te_, ir)
base_res = y - base_oof
print(f"base ridge v3+cal ({len(base_cols)} cols) CV RMSE {rmse(base_oof, y):.3f}; corr with blend residual {np.corrcoef(base_res, t.resid)[0, 1]:.3f}")


def load_emb(name):
    E = np.load(DATA_DIR / "emb" / f"{name}.npy", mmap_mode="r")
    ids = pd.read_csv(DATA_DIR / "emb" / f"{name}_ids.csv").iloc[:, 0].values
    pos = pd.Series(np.arange(len(ids)), index=ids)[t.ID].values
    return np.asarray(E[pos], dtype=np.float32)


sl = lambda X: np.sign(X) * np.log1p(np.abs(X))
reps = {}
for nm in ("tf_efficientnetv2_s.in21k_ft_in1k_256", "resnet18.a1_in1k_256", "resnet18.a1_in1k_256_nlm"):
    reps[nm.replace("tf_efficientnetv2_s.in21k_ft_in1k", "effv2s").replace("resnet18.a1_in1k", "r18")] = sl(load_emb(nm).mean(1))
for nm in ("tf_efficientnetv2_s.in21k_ft_in1k_256_g2a", "resnet18.a1_in1k_256_g2a", "convnext_nano.d1h_in1k_256_g2a"):
    E = load_emb(nm).reshape(len(t), 6, 5, -1)[:, :4]
    short = nm.replace("tf_efficientnetv2_s.in21k_ft_in1k", "effv2s").replace("resnet18.a1_in1k", "r18").replace("convnext_nano.d1h_in1k", "cnxn")
    reps[short + ":global"] = sl(E[:, :, 0].mean(1))
    reps[short + ":cellstd"] = sl(E[:, :, 1:].mean(1).std(1))
f1 = pd.read_parquet(DATA_DIR / "features.parquet")
v2 = pd.read_parquet(DATA_DIR / "features_v2.parquet")
gf = pd.read_parquet(cache / "gf_train.parquet")
for nm, df in (("feat_v1", f1), ("feat_v2", v2), ("grain_aggr(a8)", gf)):
    X = t[["ID"]].merge(df, on="ID", how="left").drop(columns="ID").astype(float).replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().mean() > 0.5]
    reps[nm] = X.fillna(X.median()).values

rows = []
for nm, X in reps.items():
    pr = np.zeros(len(y))
    for k, (tr_, te_, ir) in inner_res.items():
        pr[te_] = ridge_fit_predict(X[tr_], ir, X[te_], alphas=np.logspace(-1, 6, 29))
    row = {"rep": nm, "dim": X.shape[1]}
    for sub, m0 in (("all", np.ones(len(y), bool)), ("clean", clean)):
        row[f"corr_{sub}"] = np.corrcoef(pr[m0], base_res[m0])[0, 1]
        row[f"gain_{sub}"] = rmse(base_res[m0], 0) - rmse(base_res[m0] - pr[m0], 0)
        for g in ("coarse", "mid", "fine"):
            m = m0 & (terc == g)
            row[f"corr_{sub}_{g}"] = np.corrcoef(pr[m], base_res[m])[0, 1]
            row[f"gain_{sub}_{g}"] = rmse(base_res[m], 0) - rmse(base_res[m] - pr[m], 0)
    rows.append(row)
    print(nm, {k: round(v, 3) for k, v in row.items() if k.startswith(("corr_all", "gain_all"))}, flush=True)
R = pd.DataFrame(rows)
R.to_csv(OUT / "a17b_probes_crossfit.csv", index=False)
pd.set_option("display.width", 300)
cols = ["rep", "dim", "corr_all", "gain_all", "corr_all_coarse", "corr_all_mid", "corr_all_fine", "gain_all_coarse",
        "corr_clean", "gain_clean", "corr_clean_coarse", "corr_clean_mid", "corr_clean_fine"]
print(R[cols].round(3).to_string(index=False))
for g in ("coarse", "mid", "fine"):
    print(f"base RMSE {g}: all {rmse(base_res[terc == g], 0):.2f}  clean {rmse(base_res[(terc == g) & clean], 0):.2f}")
