"""A17c: dissect the effv2s grid-cell-std probe: which stage/pool blocks carry it, which views, and what it tracks."""
import re, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from eda_common import DATA_DIR, OUT, rmse, train_table

cache = Path(sys.argv[1])
t = train_table()
y = t.hardness.values; fold = t.fold.values
N = t.cal_seg_count_density.values * 6.5536
terc = np.asarray(pd.qcut(N, 3, labels=["coarse", "mid", "fine"]))
clean = t.ic_ridge_snr.values > 0.5
DROP = ["cal_ic_seg_L_", "cal_ic_seg_mx_area_mean", "cal_seg_area_cv", "cal_segdk_area_cv"]
v3 = pd.read_parquet(DATA_DIR / "features_v3.parquet"); cal = pd.read_parquet(DATA_DIR / "features_cal.parquet")
base_cols = [c for c in list(v3.columns[1:]) + list(cal.columns[1:]) if not any(re.search(p, c) for p in DROP)]
XB = t[base_cols].astype(float).replace([np.inf, -np.inf], np.nan); XB = XB.fillna(XB.median()).values

def rfp(Xtr, ytr, Xte, alphas=np.logspace(-2, 4, 40)):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    return RidgeCV(alphas=alphas).fit((Xtr - mu) / sd, ytr).predict((Xte - mu) / sd)

base_oof = np.zeros(len(y)); inner = {}
for k in range(5):
    tr_, te_ = np.where(fold != k)[0], np.where(fold == k)[0]
    base_oof[te_] = rfp(XB[tr_], y[tr_], XB[te_])
    ir = np.zeros(len(tr_))
    for j in sorted(set(fold[tr_])):
        a = tr_[fold[tr_] != j]; b = fold[tr_] == j
        ir[b] = y[tr_][b] - rfp(XB[a], y[a], XB[tr_][b])
    inner[k] = (tr_, te_, ir)
base_res = y - base_oof

def probe(X):
    pr = np.zeros(len(y))
    for k, (tr_, te_, ir) in inner.items():
        pr[te_] = rfp(X[tr_], ir, X[te_], alphas=np.logspace(-1, 6, 29))
    return pr

def report(nm, pr):
    out = f"{nm:34s} corr {np.corrcoef(pr, base_res)[0,1]:+.3f} gain {rmse(base_res,0)-rmse(base_res-pr,0):+.3f} |"
    for g in ("coarse", "mid", "fine"):
        m = terc == g
        out += f" {g} {np.corrcoef(pr[m], base_res[m])[0,1]:+.3f}/{rmse(base_res[m],0)-rmse(base_res[m]-pr[m],0):+.2f}"
    out += f" | clean {np.corrcoef(pr[clean], base_res[clean])[0,1]:+.3f}"
    print(out, flush=True)

name = "tf_efficientnetv2_s.in21k_ft_in1k_256_g2a"
meta = json.load(open(DATA_DIR / "emb" / f"{name}_meta.json"))
E = np.load(DATA_DIR / "emb" / f"{name}.npy", mmap_mode="r")
ids = pd.read_csv(DATA_DIR / "emb" / f"{name}_ids.csv").iloc[:, 0].values
pos = pd.Series(np.arange(len(ids)), index=ids)[t.ID].values
E = np.asarray(E[pos], np.float32).reshape(len(t), 6, 5, -1)
sl = lambda X: np.sign(X) * np.log1p(np.abs(X))
cs4 = sl(E[:, :4, 1:].mean(1).std(1))          # geometric views averaged, std over 4 cells
report("cellstd views0-3 (ref)", probe(cs4))
report("cellstd view id only", probe(sl(E[:, 0, 1:].std(1))))
report("cellstd views noise/blur aug", probe(sl(E[:, 4:, 1:].mean(1).std(1))))
report("cellstd per-view then mean", probe(sl(E[:, :4, 1:].std(2).mean(1))))
report("cell range (max-min)", probe(sl(E[:, :4, 1:].mean(1).max(1) - E[:, :4, 1:].mean(1).min(1))))
report("cellstd raw (no log)", probe(E[:, :4, 1:].mean(1).std(1)))
for b in meta["blocks"]:
    report(f"cellstd stage{b['stage']} {b['pool']}", probe(cs4[:, b["start"]:b["end"]]))
pr = probe(cs4)
np.save(OUT / "a17c_cellstd_probe_oof.npy", pr)
# what does the probe track?  Spearman with handcrafted features / quality
f1 = pd.read_parquet(DATA_DIR / "features.parquet"); v2 = pd.read_parquet(DATA_DIR / "features_v2.parquet")
gf = pd.read_parquet(cache / "gf_train.parquet")
T = t.merge(f1, on="ID").merge(gf, on="ID", how="left")
cols = [c for c in T.columns if c not in ("ID", "hardness", "fold", "pred", "resid") and not c.startswith("oof_")]
rho = pd.Series({c: spearmanr(T[c], pr, nan_policy="omit")[0] for c in cols}).dropna()
print("probe vs features (top |rho|):"); print(rho.reindex(rho.abs().sort_values(ascending=False).index).head(25).round(3).to_string())
