"""Ridge/SVR head on cached embeddings (+ optional handcrafted features).
python -m src.train_head --emb tf_efficientnetv2_s.in21k_ft_in1k_256 --head svr [--with-feats]"""
import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .common import DATA_DIR, load_test, load_train, save_experiment

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--head", default="svr", choices=["svr", "ridge"])
    ap.add_argument("--with-feats", action="store_true")
    ap.add_argument("--pca", type=int, default=256)
    a = ap.parse_args()
    tr, te = load_train(), load_test()
    E = pd.DataFrame(np.load(DATA_DIR / "emb" / f"{a.emb}.npy")).add_prefix("e")
    E.insert(0, "ID", pd.read_csv(DATA_DIR / "emb" / f"{a.emb}_ids.csv").ID)
    if a.with_feats:
        E = E.merge(pd.read_parquet(DATA_DIR / "features.parquet"), on="ID")
    X = tr[["ID"]].merge(E, on="ID").drop(columns="ID").values
    Xt = te[["ID"]].merge(E, on="ID").drop(columns="ID").values
    y = tr.hardness.values
    oof, pred = np.zeros(len(tr)), np.zeros(len(te))
    for f in range(5):
        trn, val = (tr.fold != f).values, (tr.fold == f).values
        steps = [StandardScaler(), PCA(min(a.pca, trn.sum() - 1), whiten=True)]
        head = SVR(C=50.0, epsilon=1.0) if a.head == "svr" else RidgeCV(alphas=np.logspace(-2, 4, 30))
        m = make_pipeline(*steps, head)
        mu, sd = y[trn].mean(), y[trn].std()
        m.fit(X[trn], (y[trn] - mu) / sd if a.head == "svr" else y[trn])
        p, pt = m.predict(X[val]), m.predict(Xt)
        if a.head == "svr":
            p, pt = p * sd + mu, pt * sd + mu
        oof[val] = p
        pred += pt / 5
    name = f"emb_{a.emb}_{a.head}{'_feats' if a.with_feats else ''}"
    save_experiment(name, tr, oof, te, pred, notes=f"{a.head} head on {a.emb}")
