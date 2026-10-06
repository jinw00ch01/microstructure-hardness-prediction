"""A10: are blend residuals locally predictable in any representation?  kNN mean of neighbours' OOF residuals vs own
residual (Moran-type test), permutation null.  Representations: handcrafted features (PCA), CNN embeddings (PCA)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from eda_common import DATA_DIR, train_table

t = train_table()
r = t.resid.values
coarse = (t.cal_seg_count_density <= t.cal_seg_count_density.quantile(1 / 3)).values
rng = np.random.default_rng(0)
reps = {}
fc = [c for c in t.columns if c.startswith(("ic_", "seg", "cal_", "q_", "lv_", "ph_", "dk_", "pore", "ac", "st", "hp_"))]
X = t[fc].astype(float)
X = X.fillna(X.median()).values
reps["feats_pca20"] = PCA(20, random_state=0).fit_transform(StandardScaler().fit_transform(X))
ids_tr = pd.read_csv(DATA_DIR / "train.csv").ID
for nm in ("tf_efficientnetv2_s.in21k_ft_in1k_256", "resnet18.a1_in1k_256"):
    E = np.load(DATA_DIR / "emb" / f"{nm}.npy", mmap_mode="r")
    eids = pd.read_csv(DATA_DIR / "emb" / f"{nm}_ids.csv").iloc[:, 0]
    pos = pd.Series(np.arange(len(eids)), index=eids.values)
    Et = np.asarray(E[pos[t.ID].values]).mean(1)
    reps[nm[:12] + "_pca50"] = PCA(50, random_state=0).fit_transform(StandardScaler().fit_transform(np.log1p(np.abs(Et)) * np.sign(Et)))


def knn_stat(Z, rr, k):
    D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, 1)[:, :k]
    return np.corrcoef(rr, rr[nn].mean(1))[0, 1], nn


for nm, Z in reps.items():
    Z = Z / Z.std(0)
    for sub, m in (("all", np.ones(len(r), bool)), ("coarseN", coarse)):
        for k in (5, 15):
            s, nn = knn_stat(Z[m], r[m], k)
            null = [np.corrcoef(p_, p_[nn].mean(1))[0, 1] for p_ in (rng.permutation(r[m]) for _ in range(300))]
            print(f"{nm:28s} {sub:8s} k={k:2d}: corr(r, kNN mean r) {s:+.3f}  null95 {np.quantile(null, .95):+.3f}  p={np.mean(np.array(null) >= s):.3f}")
