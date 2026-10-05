"""GBM / linear models on handcrafted features. python -m src.train_gbm --model lgb --name feat_lgb"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .common import DATA_DIR, SEED, load_test, load_train, save_experiment


def make_model(kind):
    if kind == "lgb":
        return lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=15, min_child_samples=10,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
                                 random_state=SEED, verbose=-1)
    if kind == "cat":
        return CatBoostRegressor(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                                 random_seed=SEED, verbose=0)
    if kind == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
    if kind == "svr":
        return make_pipeline(StandardScaler(), SVR(C=30.0, epsilon=1.0, gamma="scale"))
    raise ValueError(kind)


def run(model, name, feat_file="features.parquet"):
    tr, te = load_train(), load_test()
    feats = pd.read_parquet(DATA_DIR / feat_file)
    X = tr[["ID"]].merge(feats, on="ID").drop(columns="ID")
    Xt = te[["ID"]].merge(feats, on="ID").drop(columns="ID")
    y = tr.hardness.values
    oof, pred = np.zeros(len(tr)), np.zeros(len(te))
    imp = np.zeros(X.shape[1])
    for f in range(5):
        trn, val = tr.fold != f, tr.fold == f
        m = make_model(model)
        if model == "lgb":
            m.fit(X[trn], y[trn], eval_set=[(X[val], y[val])], callbacks=[lgb.early_stopping(200, verbose=False)])
            imp += m.feature_importances_
        elif model == "cat":
            m.fit(X[trn], y[trn], eval_set=(X[val], y[val]), early_stopping_rounds=300)
        else:
            m.fit(X[trn], y[trn])
        oof[val] = m.predict(X[val])
        pred += m.predict(Xt) / 5
    if imp.any():
        print(pd.Series(imp, X.columns).sort_values(ascending=False).head(20))
    return save_experiment(name, tr, oof, te, pred, notes=f"{model} on {feat_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgb", choices=["lgb", "cat", "ridge", "svr"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--feat", default="features.parquet")
    a = ap.parse_args()
    run(a.model, a.name or f"feat_{a.model}", a.feat)
