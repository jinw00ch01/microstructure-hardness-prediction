"""GBM / linear models on handcrafted features.

python -m src.train_gbm --model lgb --name feat_lgb                               (v1 baseline)
python -m src.train_gbm --feat features_v2.parquet --model lgb --name feat2_lgb
python -m src.train_gbm --feat features_v2.parquet,features.parquet --model cat --name feat2_cat_v1v2
python -m src.train_gbm --feat features_v2.parquet --model ridge --cols physics --name feat2_ridge_phys

Honest CV: hyper-parameters are fixed a priori; the validation fold is never used for early stopping
(unless --es is passed, which is flagged in the notes). Any feature selection happens inside the fold
on training rows only. Threads are limited to 1 (shared machine).
"""
import argparse
import os
import re

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .common import DATA_DIR, EXP_DIR, SEED, load_test, load_train, save_experiment

N_THREADS = int(os.environ.get("N_THREADS", "1"))

# compact physics-motivated column set (v2 features); regexes matched against column names
PHYSICS_COLS = [
    r"^seg_fd(12|15|20)$", r"^dk_frac$", r"^ph_rel(90|92)$", r"^ph_deficit2?$",
    r"^pore(60|70)_(frac|n|n_round|n_elong)$",
    r"^seg_inv_sqrt_d$", r"^seg_L_inv_sqrt$", r"^seg_count_density$", r"^ac50_gm$",
    r"^seg_asp_wmean$", r"^seg_ori_R$", r"^seg_elong_align$", r"^segdk_asp_wmean$", r"^dk_asp_wmean$",
    r"^dk_ori_R$", r"^ac50_aspect$", r"^st[12]_coh$", r"^seg_L_aniso$", r"^seg_area_cv$",
    r"^q_noise$", r"^q_ridge_snr$", r"^q_spec_slope_hi$",
]


def make_model(kind, seed=SEED):
    if kind == "lgb":
        return lgb.LGBMRegressor(n_estimators=700, learning_rate=0.02, num_leaves=15, min_child_samples=15,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=2.0,
                                 random_state=seed, verbose=-1, n_jobs=N_THREADS)
    if kind == "lgbs":  # small trees, heavy subsampling: better for noisy targets / many correlated features
        return lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.01, num_leaves=4, min_child_samples=20,
                                 subsample=0.7, subsample_freq=1, colsample_bytree=0.3, reg_lambda=5.0,
                                 random_state=seed, verbose=-1, n_jobs=N_THREADS)
    if kind == "lgb_es":  # legacy: early stopping on the validation fold (optimistic)
        return lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=15, min_child_samples=10,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
                                 random_state=seed, verbose=-1, n_jobs=N_THREADS)
    if kind == "cat":
        return CatBoostRegressor(iterations=1500, learning_rate=0.03, depth=5, l2_leaf_reg=5, rsm=0.5,
                                 random_seed=seed, verbose=0, thread_count=N_THREADS)
    if kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 40)))
    if kind == "svr":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             SVR(C=30.0, epsilon=1.0, gamma="scale"))
    raise ValueError(kind)


def load_features(feat_files):
    out = None
    for k, ff in enumerate(feat_files.split(",")):
        d = pd.read_parquet(DATA_DIR / ff)
        if out is not None:
            dup = [c for c in d.columns if c != "ID" and c in out.columns]
            d = d.rename(columns={c: f"f{k}_{c}" for c in dup})
            out = out.merge(d, on="ID")
        else:
            out = d
    return out


def select_columns(cols, spec):
    if not spec:
        return list(cols)
    pats = PHYSICS_COLS if spec == "physics" else spec.split(",")
    keep = [c for c in cols if any(re.search(p, c) for p in pats)]
    return keep


def corr_select(X, y, k):
    """Univariate |spearman| ranking on training rows only."""
    r = X.rank().corrwith(pd.Series(y, index=X.index).rank()).abs().fillna(0)
    return list(r.sort_values(ascending=False).index[:k])


def _std_matrix(Xtr, *others):
    """Median-impute + standardize using training-fold statistics only."""
    med = Xtr.median()
    A = Xtr.fillna(med)
    mu, sd = A.mean(), A.std().replace(0, 1.0)
    out = [((A - mu) / sd).values]
    for o in others:
        out.append(((o.fillna(med) - mu) / sd).values)
    return out


def forward_select(X, y, max_k=20, inner_folds=5, alpha=3.0, tol=0.01, seed=SEED):
    """Greedy forward selection with ridge, scored by inner K-fold CV on the given (training) rows only."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    (A,) = _std_matrix(X)
    cols = list(X.columns)
    splits = list(KFold(inner_folds, shuffle=True, random_state=seed).split(A))
    def score(idx):
        err = 0.0
        for a, b in splits:
            m = Ridge(alpha=alpha).fit(A[a][:, idx], y[a])
            err += ((m.predict(A[b][:, idx]) - y[b]) ** 2).sum()
        return np.sqrt(err / len(y))
    sel, best = [], np.inf
    for _ in range(max_k):
        cand = [(score(sel + [j]), j) for j in range(len(cols)) if j not in sel]
        sc, j = min(cand)
        if sc > best - tol:
            break
        sel.append(j)
        best = sc
    return [cols[j] for j in sel]


def hetero_weights(Xtr, ytr, z, inner=5, clip=(0.25, 4.0)):
    """Inverse-variance sample weights, estimated on training rows only.

    Inner-CV ridge residuals r give log(r^2 + 1) ~ a + Z b by OLS, where Z holds the variance columns
    (log-transformed when strictly positive), e.g. correlation length and noise level.
    w = 1 / exp(fit), normalised to mean 1 and clipped.
    """
    from sklearn.model_selection import KFold
    res = np.zeros(len(ytr))
    for a, b in KFold(inner, shuffle=True, random_state=SEED).split(Xtr):
        A, B = _std_matrix(Xtr.iloc[a], Xtr.iloc[b])
        m = RidgeCV(alphas=np.logspace(-2, 4, 40)).fit(A, ytr[a])
        res[b] = ytr[b] - m.predict(B)
    Z = pd.DataFrame(np.asarray(z, float).reshape(len(ytr), -1))
    Z = Z.fillna(Z.median())
    Z = Z.apply(lambda c: np.log(c) if (c > 0).all() else c)
    D = np.column_stack([np.ones(len(ytr)), Z.values])
    coef, *_ = np.linalg.lstsq(D, np.log(res ** 2 + 1.0), rcond=None)
    w = 1.0 / np.exp(D @ coef)
    w = np.clip(w / w.mean(), *clip)
    return w / w.mean(), coef[1:]


def hetero_weights_add(Xtr, ytr, z, inner=5, clip=(0.25, 4.0)):
    """Additive variance model v = b0 + sum_k b_k / z_k (NNLS on inner-CV squared residuals, training rows
    only); e.g. z = (calibrated grain count density, ridge SNR) gives the label-noise b/N + c/snr form."""
    from scipy.optimize import nnls
    from sklearn.model_selection import KFold
    res = np.zeros(len(ytr))
    for a, b in KFold(inner, shuffle=True, random_state=SEED).split(Xtr):
        A, B = _std_matrix(Xtr.iloc[a], Xtr.iloc[b])
        m = RidgeCV(alphas=np.logspace(-2, 4, 40)).fit(A, ytr[a])
        res[b] = ytr[b] - m.predict(B)
    Z = pd.DataFrame(np.asarray(z, float).reshape(len(ytr), -1))
    Z = Z.fillna(Z.median())
    G = np.column_stack([np.ones(len(ytr))] + [1.0 / np.clip(Z[c].values, 1e-3, None) for c in Z.columns])
    scale = G.mean(0)
    coef, _ = nnls(G / scale, res ** 2)
    v = np.maximum((G / scale) @ coef, 1e-3 * np.mean(res ** 2))
    w = np.clip((1.0 / v) / np.mean(1.0 / v), *clip)
    return w / w.mean(), coef / scale


def tercile_rmse(tr, oof):
    """RMSE by image-quality tercile (ic_ridge_snr cut on train): T1 = noisiest, T3 = cleanest."""
    from .common import rmse
    q = tr[["ID"]].merge(pd.read_parquet(DATA_DIR / "features_v3.parquet")[["ID", "ic_ridge_snr"]], on="ID")
    t = pd.qcut(q.ic_ridge_snr.values, 3, labels=["T1_noisy", "T2", "T3_clean"])
    y = tr.hardness.values
    return {str(k): round(rmse(oof[t == k], y[t == k]), 3) for k in ["T1_noisy", "T2", "T3_clean"]}


def mono_vector(cols, spec):
    """'+' constraints for columns matching regexes in spec; prefix a regex with '-' for a decreasing one."""
    v = np.zeros(len(cols), int)
    for pat in spec.split(","):
        sign = -1 if pat.startswith("-") else 1
        pat = pat.lstrip("+-")
        for j, c in enumerate(cols):
            if re.search(pat, c):
                v[j] = sign
    return v


def run(model, name, feat_file="features.parquet", cols=None, seeds=1, select_k=0, es=False, drop=None, fwd=0,
        hetero=None, mono=None, save=True, hetero_add=None):
    tr, te = load_train(), load_test()
    feats = load_features(feat_file)
    use = select_columns([c for c in feats.columns if c != "ID"], cols)
    if drop:
        use = [c for c in use if not any(re.search(p, c) for p in drop.split(","))]
    X = tr[["ID"]].merge(feats, on="ID")[use]
    Xt = te[["ID"]].merge(feats, on="ID")[use]
    X = X.replace([np.inf, -np.inf], np.nan)
    Xt = Xt.replace([np.inf, -np.inf], np.nan)
    y = tr.hardness.values
    hz = hetero or hetero_add
    feats_tr_z = tr[["ID"]].merge(feats, on="ID")[hz.split(",")].values if hz else None
    oof, pred = np.zeros(len(tr)), np.zeros(len(te))
    imp = pd.Series(0.0, index=use)
    kind = "lgb_es" if (model == "lgb" and es) else model
    sel_log = []
    for f in range(5):
        trn, val = (tr.fold != f).values, (tr.fold == f).values
        cols_f = corr_select(X[trn], y[trn], select_k) if select_k else use
        if fwd:
            cols_f = forward_select(X.loc[trn, cols_f], y[trn], max_k=fwd)
            print(f"fold {f}: forward-selected {cols_f}", flush=True)
            sel_log.append(cols_f)
        w = None
        if hetero_add:
            w, coef = hetero_weights_add(X.loc[trn, cols_f], y[trn], feats_tr_z[trn])
            print(f"fold {f}: additive var coefs {np.round(coef, 4).tolist()}; weight range {w.min():.2f}-{w.max():.2f}", flush=True)
        elif hetero:
            w, coef = hetero_weights(X.loc[trn, cols_f], y[trn], feats_tr_z[trn])
            print(f"fold {f}: hetero log-var slopes {np.round(coef, 3).tolist()}; weight range {w.min():.2f}-{w.max():.2f}", flush=True)
        if kind in ("ridge_fs", "fwd"):
            from sklearn.linear_model import RidgeCV
            A, Av, At = _std_matrix(X.loc[trn, cols_f], X.loc[val, cols_f], Xt[cols_f])
            m = RidgeCV(alphas=np.logspace(-2, 3, 30)).fit(A, y[trn], sample_weight=w)
            oof[val] = m.predict(Av)
            pred += m.predict(At) / 5
            imp[cols_f] += np.abs(m.coef_)
            continue
        for s in range(seeds):
            m = make_model(kind, SEED + s)
            if mono and kind in ("lgb", "lgbs"):
                m.set_params(monotone_constraints=list(mono_vector(cols_f, mono)),
                             monotone_constraints_method="intermediate")
            if kind == "lgb_es":
                m.fit(X.loc[trn, cols_f], y[trn], eval_set=[(X.loc[val, cols_f], y[val])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
            elif kind == "cat" and es:
                m.fit(X.loc[trn, cols_f], y[trn], eval_set=(X.loc[val, cols_f], y[val]), early_stopping_rounds=300)
            elif kind in ("ridge", "svr"):
                step = m.steps[-1][0]
                m.fit(X.loc[trn, cols_f], y[trn], **({f"{step}__sample_weight": w} if w is not None else {}))
            else:
                m.fit(X.loc[trn, cols_f], y[trn], sample_weight=w)
            if hasattr(m, "feature_importances_"):
                imp[cols_f] += np.asarray(m.feature_importances_, float) / seeds
            elif hasattr(m, "get_feature_importance"):
                imp[cols_f] += m.get_feature_importance() / seeds
            oof[val] += m.predict(X.loc[val, cols_f]) / seeds
            pred += m.predict(Xt[cols_f]) / (5 * seeds)
    if imp.any():
        print((imp / imp.sum()).sort_values(ascending=False).head(30).round(4).to_string())
    notes = f"{model} on {feat_file}; cols={cols or 'all'}({len(use)})"
    if select_k:
        notes += f"; in-fold spearman top{select_k}"
    if seeds > 1:
        notes += f"; {seeds} seeds"
    if es:
        notes += "; EARLY-STOP ON VAL FOLD (optimistic)"
    if fwd:
        notes += f"; in-fold forward selection (ridge inner-CV, max {fwd})"
    if hetero_add:
        notes += f"; hetero weights var = b0 + sum b/({hetero_add}) in-fold"
    elif hetero:
        notes += f"; hetero weights ~ log-var(log {hetero}) in-fold"
    if mono:
        notes += f"; monotone({mono})"
    terc = tercile_rmse(tr, oof)
    if not save:
        from .common import rmse
        fr = [rmse(oof[tr.fold == k], y[tr.fold == k]) for k in range(5)]
        print(f"[{name} NOT SAVED] CV RMSE {rmse(oof, y):.4f} folds {np.round(fr, 3).tolist()} terciles {terc} | {notes}")
        return oof
    out = save_experiment(name, tr, oof, te, pred, notes=notes)
    print(f"[{name}] snr terciles {terc}")
    import json
    (EXP_DIR / name / "tercile_rmse.json").write_text(json.dumps(terc))
    imp.sort_values(ascending=False).to_csv(EXP_DIR / name / "importance.csv", header=["importance"])
    if sel_log:
        (EXP_DIR / name / "selected.txt").write_text("\n".join(",".join(c) for c in sel_log))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgb", choices=["lgb", "lgbs", "cat", "ridge", "svr", "fwd"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--feat", default="features.parquet", help="comma-separated parquet files in DATA_DIR")
    ap.add_argument("--cols", default=None, help="'physics' or comma-separated regexes")
    ap.add_argument("--drop", default=None, help="comma-separated regexes of columns to drop")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--select_k", type=int, default=0)
    ap.add_argument("--es", action="store_true", help="early stopping on validation fold (optimistic CV)")
    ap.add_argument("--fwd", type=int, default=0, help="in-fold greedy forward selection (max features)")
    ap.add_argument("--hetero", default=None, help="column used to model residual variance -> sample weights")
    ap.add_argument("--mono", default=None, help="regexes of +monotone columns for lgb ('-regex' = decreasing)")
    ap.add_argument("--no_save", action="store_true", help="screening run: print CV only, write nothing")
    ap.add_argument("--hetero_add", default=None, help="columns z_k for additive variance b0 + sum b_k/z_k")
    a = ap.parse_args()
    fwd = a.fwd or (20 if a.model == "fwd" else 0)
    run(a.model, a.name or f"feat_{a.model}", a.feat, a.cols, a.seeds, a.select_k, a.es, a.drop, fwd,
        a.hetero, a.mono, not a.no_save, a.hetero_add)
