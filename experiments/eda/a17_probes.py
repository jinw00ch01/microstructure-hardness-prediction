"""A17: linear probes for the blend_v2 nested OOF residual (shared folds, in-fold RidgeCV, standardised inputs).
Representations: cached CNN embeddings (view-averaged; g2a: global cell, mean / std over the 4 grid cells) and the full
handcrafted feature table.  Reports OOF corr(probe, residual) and the RMSE after subtracting the probe, overall and per
N tercile (N = calibrated grain count), all images and clean (ic_ridge_snr > 0.5).
Usage: python a17_probes.py CACHE_DIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from eda_common import DATA_DIR, OUT, rmse, train_table

cache = Path(sys.argv[1])
t = train_table()
t = t.merge(pd.read_parquet(DATA_DIR / "features.parquet"), on="ID")
t = t.merge(pd.read_parquet(cache / "gf_train.parquet"), on="ID", how="left")
r = t.resid.values
fold = t.fold.values
N = t.cal_seg_count_density.values * 6.5536
terc = np.asarray(pd.qcut(N, 3, labels=["coarse", "mid", "fine"]))
clean = t.ic_ridge_snr.values > 0.5


def load_emb(name):
    E = np.load(DATA_DIR / "emb" / f"{name}.npy", mmap_mode="r")
    ids = pd.read_csv(DATA_DIR / "emb" / f"{name}_ids.csv").iloc[:, 0].values
    pos = pd.Series(np.arange(len(ids)), index=ids)[t.ID].values
    return np.asarray(E[pos], dtype=np.float32)


def signed_log(X):
    return np.sign(X) * np.log1p(np.abs(X))


reps = {}
for nm in ("tf_efficientnetv2_s.in21k_ft_in1k_256", "resnet18.a1_in1k_256", "resnet18.a1_in1k_256_q", "resnet18.a1_in1k_256_nlm"):
    reps[nm.replace("tf_efficientnetv2_s.in21k_ft_in1k", "effv2s").replace("resnet18.a1_in1k", "r18")] = signed_log(load_emb(nm).mean(1))
for nm in ("tf_efficientnetv2_s.in21k_ft_in1k_256_g2a", "resnet18.a1_in1k_256_g2a", "convnext_nano.d1h_in1k_256_g2a"):
    E = load_emb(nm)  # (n, 6 views x 5 cells, d), view-major, cell 0 = global
    E = E.reshape(len(E), 6, 5, -1)[:, :4]  # geometric views only
    short = nm.replace("tf_efficientnetv2_s.in21k_ft_in1k", "effv2s").replace("resnet18.a1_in1k", "r18").replace("convnext_nano.d1h_in1k", "cnxn")
    glob_ = E[:, :, 0].mean(1)
    cells = E[:, :, 1:].mean(1)  # (n, 4 cells, d)
    reps[short + ":global"] = signed_log(glob_)
    reps[short + ":cellstd"] = signed_log(cells.std(1))
    reps[short + ":global+cellstd"] = np.hstack([signed_log(glob_), signed_log(cells.std(1))])
fcols = [c for c in t.columns if c not in ("ID", "hardness", "fold", "pred", "resid") and not c.startswith("oof_")]
F = t[fcols].astype(float).replace([np.inf, -np.inf], np.nan)
reps["features_all(%d)" % len(fcols)] = F.fillna(F.median()).values


def probe(X):
    oof = np.zeros(len(r))
    for k in range(5):
        tr_, te_ = fold != k, fold == k
        mu, sd = X[tr_].mean(0), X[tr_].std(0) + 1e-6
        A, B = (X[tr_] - mu) / sd, (X[te_] - mu) / sd
        m = RidgeCV(alphas=np.logspace(-1, 6, 29)).fit(A, r[tr_])
        oof[te_] = m.predict(B)
    return oof


rows = []
for nm, X in reps.items():
    p = probe(X)
    row = {"rep": nm, "dim": X.shape[1]}
    for sub, m0 in (("all", np.ones(len(r), bool)), ("clean", clean)):
        row[f"corr_{sub}"] = np.corrcoef(p[m0], r[m0])[0, 1]
        row[f"rmse_{sub}"] = f"{rmse(r[m0], 0):.2f}->{rmse(r[m0] - p[m0], 0):.2f}"
        for g in ("coarse", "mid", "fine"):
            m = m0 & (terc == g)
            row[f"corr_{sub}_{g}"] = np.corrcoef(p[m], r[m])[0, 1]
    rows.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
R = pd.DataFrame(rows)
R.to_csv(OUT / "a17_probes.csv", index=False)
pd.set_option("display.width", 250)
print(R.round(3).to_string(index=False))
print("SE of a null corr: all %.3f, tercile(all) %.3f, clean %.3f, tercile(clean) ~%.3f" % (
    1 / np.sqrt(len(r)), 1 / np.sqrt(len(r) / 3), 1 / np.sqrt(clean.sum()), 1 / np.sqrt(clean.sum() / 3)))
