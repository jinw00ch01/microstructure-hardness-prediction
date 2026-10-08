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
from sklearn.base import BaseEstimator, TransformerMixin
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


# ---- gated grain-size heterogeneity columns ("member route", --het_member; default off)
# het4 / N_eff come from src/het_blocks.py (per-image extraction only, images with raw ic_noise < 9.5; train and test
# files are built the same way, nothing is fitted across images). Ridge: gate, gate*het4, gate*N_eff^0.25 (0 for
# ungated rows), standardized in-fold by the pipeline's StandardScaler and then multiplied by HET_FACTOR so RidgeCV's
# common alpha penalises them 25x less (fixed, pre-registered; not tuned). Trees: het4 and N_eff^0.25 with NaN for
# ungated rows. The hetero sample weights are computed on the member's design WITHOUT these columns, so they are
# identical to the original recipe.
HET_FACTOR = 5.0
HET_TREE_KINDS = ("lgb", "lgbs", "lgb_es", "cat")


def het_member_cols(ids, split, kind):
    F = pd.read_parquet(DATA_DIR / f"het_blocks_{split}.parquet").set_index("ID")
    ids = pd.Index(ids)
    g = ids.isin(F.index)
    h = F.het4.reindex(ids).values.astype(float)
    n = F.N_eff.reindex(ids).values.astype(float) ** 0.25
    if kind == "ridge":
        return pd.DataFrame({"hetm_gate": g.astype(float), "hetm_gate_het4": np.where(g, h, 0.0),
                             "hetm_gate_n025": np.where(g, n, 0.0)})
    if kind in HET_TREE_KINDS:
        return pd.DataFrame({"hetm_het4": np.where(g, h, np.nan), "hetm_n025": np.where(g, n, np.nan)})
    raise ValueError(f"--het_member not implemented for model {kind}")


class ColScale(BaseEstimator, TransformerMixin):
    """Multiplies the columns idx (negative = from the end) by factor; placed after StandardScaler."""

    def __init__(self, idx=(), factor=1.0):
        self.idx, self.factor = idx, factor

    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self):   # stateless
        return True

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        X[:, list(self.idx)] *= self.factor
        return X


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
        hetero=None, mono=None, save=True, hetero_add=None, fold_extra=None, extra_note="", het_member=False):
    """fold_extra: optional {fold: (extra_train, extra_val, extra_test)} DataFrames of fold-specific columns
    (e.g. cross-fitted stacked features), row-aligned with the training rows / validation rows / test rows.
    het_member: append the gated het columns (het_member_cols) after the hetero weights are computed."""
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
    extra_cols = list(fold_extra[0][0].columns) if fold_extra else []
    kind = "lgb_es" if (model == "lgb" and es) else model
    het_cols = []
    if het_member:
        H, Ht = het_member_cols(tr.ID.values, "train", kind), het_member_cols(te.ID.values, "test", kind)
        het_cols = list(H.columns)
        het_n = [int((D[het_cols[-1]].notna() & (D[het_cols[-1]] != 0)).sum()) for D in (H, Ht)]
    imp = pd.Series(0.0, index=use + extra_cols + het_cols)
    sel_log = []
    for f in range(5):
        trn, val = (tr.fold != f).values, (tr.fold == f).values
        cols_f = corr_select(X[trn], y[trn], select_k) if select_k else use
        if fwd:
            cols_f = forward_select(X.loc[trn, cols_f], y[trn], max_k=fwd)
            print(f"fold {f}: forward-selected {cols_f}", flush=True)
            sel_log.append(cols_f)
        A_tr, A_va, A_te = X.loc[trn, cols_f], X.loc[val, cols_f], Xt[cols_f]
        if fold_extra:
            e_tr, e_va, e_te = fold_extra[f]
            A_tr = pd.concat([A_tr, e_tr.set_axis(A_tr.index)], axis=1)
            A_va = pd.concat([A_va, e_va.set_axis(A_va.index)], axis=1)
            A_te = pd.concat([A_te, e_te.set_axis(A_te.index)], axis=1)
            cols_f = list(A_tr.columns)
        w = None
        if hetero_add:
            w, coef = hetero_weights_add(A_tr, y[trn], feats_tr_z[trn])
            print(f"fold {f}: additive var coefs {np.round(coef, 4).tolist()}; weight range {w.min():.2f}-{w.max():.2f}", flush=True)
        elif hetero:
            w, coef = hetero_weights(A_tr, y[trn], feats_tr_z[trn])
            print(f"fold {f}: hetero log-var slopes {np.round(coef, 3).tolist()}; weight range {w.min():.2f}-{w.max():.2f}", flush=True)
        if het_member:   # after the weights, so they stay those of the original recipe
            A_tr = pd.concat([A_tr, H[trn].set_axis(A_tr.index)], axis=1)
            A_va = pd.concat([A_va, H[val].set_axis(A_va.index)], axis=1)
            A_te = pd.concat([A_te, Ht.set_axis(A_te.index)], axis=1)
            cols_f = list(A_tr.columns)
        if kind in ("ridge_fs", "fwd"):
            if het_member:
                raise ValueError("--het_member not implemented for ridge_fs / fwd")
            from sklearn.linear_model import RidgeCV
            A, Av, At = _std_matrix(A_tr, A_va, A_te)
            m = RidgeCV(alphas=np.logspace(-2, 3, 30)).fit(A, y[trn], sample_weight=w)
            oof[val] = m.predict(Av)
            pred += m.predict(At) / 5
            imp[cols_f] += np.abs(m.coef_)
            continue
        for s in range(seeds):
            m = make_model(kind, SEED + s)
            if het_member and kind == "ridge":   # het columns are the last len(het_cols) after imputer + scaler
                m.steps.insert(2, ("hetscale", ColScale(tuple(range(-len(het_cols), 0)), HET_FACTOR)))
            if mono and kind in ("lgb", "lgbs"):
                m.set_params(monotone_constraints=list(mono_vector(cols_f, mono)),
                             monotone_constraints_method="intermediate")
            if kind == "lgb_es":
                m.fit(A_tr, y[trn], eval_set=[(A_va, y[val])], callbacks=[lgb.early_stopping(200, verbose=False)])
            elif kind == "cat" and es:
                m.fit(A_tr, y[trn], eval_set=(A_va, y[val]), early_stopping_rounds=300)
            elif kind in ("ridge", "svr"):
                step = m.steps[-1][0]
                m.fit(A_tr, y[trn], **({f"{step}__sample_weight": w} if w is not None else {}))
            else:
                m.fit(A_tr, y[trn], sample_weight=w)
            if hasattr(m, "feature_importances_"):
                imp[cols_f] += np.asarray(m.feature_importances_, float) / seeds
            elif hasattr(m, "get_feature_importance"):
                imp[cols_f] += m.get_feature_importance() / seeds
            oof[val] += m.predict(A_va) / seeds
            pred += m.predict(A_te) / (5 * seeds)
    if imp.any():
        print((imp / imp.sum()).sort_values(ascending=False).head(30).round(4).to_string())
    notes = f"{model} on {feat_file}; cols={cols or 'all'}({len(use)})"
    if drop:
        notes += "; drop=/" + drop.replace("|", "\\|") + "/"  # escaped: notes also go into the markdown LEADERBOARD
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
    if extra_note:
        notes += f"; {extra_note}"
    if het_member:
        import shlex
        import sys
        form = (f"gate, gate*het4, gate*N_eff^0.25 standardized in-fold x{HET_FACTOR:g}" if kind == "ridge" else
                "het4, N_eff^0.25 with NaN for ungated rows")
        notes += (f"; + gated het member columns ({form}; src/het_blocks.py, raw ic_noise < 9.5, "
                  f"{het_n[0]} train / {het_n[1]} test gated); "
                  "hetero weights from the design without them; cmd (| escaped as \\|): python -W ignore -m "
                  "src.train_gbm " + shlex.join(sys.argv[1:]).replace("|", "\\|"))
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


# =====================================================================================================
# Block-level multiple-instance model (MIL). Tests whether hardness behaves like an area average of a
# nonlinear LOCAL function of grain size / phase / pores (the Jensen term that global features miss).
# Blocks: data/mil_blocks.parquet (7x7 grid of 64 px blocks per image, raw + calibrated local measures and
# image quality context; built by `python -m src.features --mil_blocks`). Folds are the shared image folds,
# so no image is ever split between training and validation.
#   kind "lgb":     LightGBM on blocks, every block's target = its image's hardness; block predictions are
#                   aggregated per image (mean, valid-area-weighted mean, q10/q50/q90, max, min, sd).
#   kind "spl":     additive MIL model fitted on the aggregated loss: per-block B-spline bases (+ dark-fraction x
#                   size-spline products), valid-area-weighted mean over the image's blocks, RidgeCV at image level.
#                   Its block prediction is phi(x_b).beta, so the weighted mean of block predictions is the image fit.
#   kind "splmean": Jensen control: the same splines applied to the image-mean block measures (f(mean x), not
#                   mean f(x)).
# Stacking: mil_crossfit gives fold-specific aggregates for run(..., fold_extra=...): training rows get inner
# out-of-fold aggregates (inner folds = the other shared folds), validation / test rows the outer fold model.
#   python -m src.train_gbm --mil lgb --name mil_lgb [--no_save]
#   python -m src.train_gbm --mil lgb --mil_stack --feat ... --model ridge ... --name ... [--no_save]
# =====================================================================================================
MIL_RAW = ["bd_ws", "la", "sfd93", "acr_len50", "acr_r4", "acr_r8", "sm_std", "grad_mean", "rline_mean", "dk_frac",
           "fd2_93", "fdo_93", "deficit", "pore68_frac", "pore60_frac", "pore_deficit", "pore60_n", "valid_frac"]
MIL_CAL = ["c_bd", "c_la", "c_acd", "c_sfd93", "c_sfd91", "c_fdo93", "c_deficit", "c_pore60"]
MIL_CTX = ["ic_noise", "ic_spec_4_8", "ic_spec_8_16", "ic_spec_16_32", "ic_spec_32_48", "ic_spec_slope",
           "v4_ridge_snr", "v4_M"]
MIL_SPLINE_VARS = ["c_la", "c_sfd93", "c_pore60", "c_acd", "acr_len50"]
MIL_NB = 49


def mil_table(path=None):
    b = pd.read_parquet(DATA_DIR / (path or "mil_blocks.parquet")).sort_values(["ID", "blk"])
    assert (b.groupby("ID").size() == MIL_NB).all()
    return b.set_index("ID")


def _mil_lgb(seed):
    return lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.02, num_leaves=8, min_child_samples=300,
                             subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0,
                             random_state=seed, verbose=-1, n_jobs=N_THREADS)


def _mil_wts(B):
    return np.clip(B["valid_frac"].values.astype(float), 0.05, None)


def mil_aggregate(pred, wv, prefix="mil"):
    P = np.asarray(pred, float).reshape(-1, MIL_NB)
    W = np.asarray(wv, float).reshape(-1, MIL_NB)
    return pd.DataFrame({f"{prefix}_mean": P.mean(1), f"{prefix}_wmean": (P * W).sum(1) / W.sum(1),
                         f"{prefix}_q10": np.quantile(P, 0.1, axis=1), f"{prefix}_q50": np.median(P, 1),
                         f"{prefix}_q90": np.quantile(P, 0.9, axis=1), f"{prefix}_max": P.max(1),
                         f"{prefix}_min": P.min(1), f"{prefix}_sd": P.std(1)})


class MilModel:
    def __init__(self, kind, seeds=1):
        self.kind, self.seeds = kind, seeds

    # ---- spline design (fitted on training blocks / images only)
    def _basis(self, B, fit):
        from sklearn.preprocessing import SplineTransformer
        V = B[MIL_SPLINE_VARS].astype(float)
        if fit:
            self.med_ = V.median()
        V = V.fillna(self.med_)
        if fit:
            self.st_ = {c: SplineTransformer(n_knots=5, degree=3, knots="quantile", extrapolation="linear",
                                             include_bias=False).fit(V[[c]].values) for c in MIL_SPLINE_VARS}
        parts = [self.st_[c].transform(V[[c]].values) for c in MIL_SPLINE_VARS]
        size = self.st_["c_la"].transform(V[["c_la"]].values)
        parts.append(size * V[["c_sfd93"]].values)                 # per-phase grain-size effect
        parts.append(self.st_["c_acd"].transform(V[["c_acd"]].values) * V[["c_sfd93"]].values)
        return np.hstack(parts)

    def _img_design(self, B, fit):
        w = _mil_wts(B).reshape(-1, MIL_NB)
        if self.kind == "spl":
            Phi = self._basis(B, fit)
            k = Phi.shape[1]
            Phi = Phi.reshape(-1, MIL_NB, k)
            D = (Phi * w[..., None]).sum(1) / w.sum(1, keepdims=True)
        else:  # splmean: splines of the image-mean measures
            V = B[MIL_SPLINE_VARS].astype(float)
            if fit:
                self.vmed_ = V.median()
            V = V.fillna(self.vmed_).values.reshape(-1, MIL_NB, len(MIL_SPLINE_VARS))
            M = pd.DataFrame((V * w[..., None]).sum(1) / w.sum(1, keepdims=True), columns=MIL_SPLINE_VARS)
            D = self._basis(M, fit)
        ctx = B[MIL_CTX].astype(float).values.reshape(-1, MIL_NB, len(MIL_CTX))[:, 0, :]
        return np.hstack([D, ctx])

    def fit(self, B, y_img, w_img=None):
        if self.kind == "lgb":
            self.cols_ = MIL_CAL + MIL_RAW + MIL_CTX
            yb = np.repeat(y_img, MIL_NB)
            wb = np.repeat(w_img, MIL_NB) if w_img is not None else None
            self.models_ = [_mil_lgb(SEED + s).fit(B[self.cols_], yb, sample_weight=wb) for s in range(self.seeds)]
            return self
        D = self._img_design(B, fit=True)
        self.mu_, self.sd_ = np.nanmean(D, 0), np.nanstd(D, 0)
        self.sd_[self.sd_ == 0] = 1.0
        self.ridge_ = RidgeCV(alphas=np.logspace(-2, 4, 40)).fit((D - self.mu_) / self.sd_, y_img, sample_weight=w_img)
        return self

    def predict_blocks(self, B):
        if self.kind == "lgb":
            return np.mean([m.predict(B[self.cols_]) for m in self.models_], 0)
        if self.kind == "spl":   # block prediction phi(x_b).beta (+ context part); weighted mean = image fit
            Phi = self._basis(B, fit=False)
            ctx = B[MIL_CTX].astype(float).values
            D = np.hstack([Phi, ctx])
            return self.ridge_.predict((D - self.mu_) / self.sd_)
        D = self._img_design(B, fit=False)   # splmean: one prediction per image, repeated over its blocks
        return np.repeat(self.ridge_.predict((D - self.mu_) / self.sd_), MIL_NB)


def mil_crossfit(kind, seeds=1, inner=True, w_fn=None):
    """Outer OOF aggregates (train), test aggregates (mean of the 5 outer models) and, if inner, fold_extra
    for run(): {f: (inner-OOF aggregates of fold f's training images, outer aggregates val, outer aggregates test)}."""
    tr, te = load_train(), load_test()
    B = mil_table()
    y, fold = tr.hardness.values, tr.fold.values
    ids, te_ids = tr.ID.values, te.ID.values
    Bte = B.loc[te_ids]
    oof = None
    test = 0.0
    fold_extra = {}

    def fit_pred(fit_ids, fit_y, app):
        m = MilModel(kind, seeds).fit(B.loc[fit_ids], fit_y, w_fn(fit_ids, fit_y) if w_fn else None)
        return [mil_aggregate(m.predict_blocks(Ba), _mil_wts(Ba)) for Ba in app]

    for f in range(5):
        trn, val = fold != f, fold == f
        a_val, a_te = fit_pred(ids[trn], y[trn], [B.loc[ids[val]], Bte])
        if oof is None:
            oof = pd.DataFrame(np.nan, index=np.arange(len(tr)), columns=a_val.columns)
        oof.loc[np.where(val)[0]] = a_val.values
        test = test + a_te / 5.0
        if inner:
            inn = pd.DataFrame(np.nan, index=np.arange(trn.sum()), columns=a_val.columns)
            pos = np.where(trn)[0]
            for j in [j for j in range(5) if j != f]:
                fit_m, app_m = trn & (fold != j), fold == j
                (a_j,) = fit_pred(ids[fit_m], y[fit_m], [B.loc[ids[app_m]]])
                inn.loc[np.searchsorted(pos, np.where(app_m)[0])] = a_j.values
            fold_extra[f] = (inn.reset_index(drop=True), a_val.reset_index(drop=True), a_te.reset_index(drop=True))
        print(f"mil {kind} fold {f} done", flush=True)
    return oof, test, fold_extra


def mil_design_fold_extra(B=None, kind="spl"):
    """Per-fold MIL spline design (no labels involved; knots fitted on the training split only), context excluded.
    kind "spl": valid-area-weighted block mean of the B-spline bases (mean f(x_b), local / Jensen form);
    kind "splmean": B-splines of the valid-area-weighted image means of the same block measures (f(mean x))."""
    tr, te = load_train(), load_test()
    B = mil_table() if B is None else B
    fe = {}
    for f in range(5):
        trn, val = (tr.fold != f).values, (tr.fold == f).values
        m = MilModel(kind)
        Dtr = m._img_design(B.loc[tr.ID.values[trn]], fit=True)
        k = Dtr.shape[1] - len(MIL_CTX)
        cols = [f"mil_phi{j}" for j in range(k)]
        mk = lambda D: pd.DataFrame(D[:, :k], columns=cols)
        fe[f] = (mk(Dtr), mk(m._img_design(B.loc[tr.ID.values[val]], fit=False)),
                 mk(m._img_design(B.loc[te.ID.values], fit=False)))
    return fe


def mil_run(kind, name, save=True, mil_seeds=1, stack=False, member_pred="mil_wmean", w_fn=None, design=False,
            blocks_file=None, **run_kw):
    from .common import rmse
    if design:   # additive MIL terms fitted jointly with the member's global features
        form = ("valid-weighted block means of B-splines" if kind == "spl" else
                "B-splines of the valid-weighted image means (Jensen control)")
        B = mil_table(blocks_file) if blocks_file else None
        return run(save=save, name=name, fold_extra=mil_design_fold_extra(B, kind=kind),
                   extra_note=f"+ MIL {kind} design ({form} of block c_la, c_sfd93, c_pore60, c_acd, acr_len50 + "
                              "c_sfd93 x spline(c_la), c_sfd93 x spline(c_acd); knots in-fold; blocks "
                              f"{blocks_file or 'mil_blocks.parquet'})", **run_kw)
    tr, te = load_train(), load_test()
    y = tr.hardness.values
    seeds = mil_seeds
    oof_a, test_a, fold_extra = mil_crossfit(kind, seeds=mil_seeds, inner=stack, w_fn=w_fn)
    fr = lambda p: [round(rmse(p[tr.fold == k], y[tr.fold == k]), 3) for k in range(5)]
    for c in ("mil_mean", "mil_wmean"):
        print(f"[MIL {kind} alone, {c}] CV {rmse(oof_a[c].values, y):.4f} folds {fr(oof_a[c].values)} "
              f"terciles {tercile_rmse(tr, oof_a[c].values)}", flush=True)
    note = f"block MIL {kind} ({seeds} seeds) on mil_blocks.parquet"
    if not stack:
        oof, pred = oof_a[member_pred].values, test_a[member_pred].values
        if save:
            save_experiment(name, tr, oof, te, pred, notes=f"{note}; member = {member_pred} of block predictions")
            import json
            (EXP_DIR / name / "tercile_rmse.json").write_text(json.dumps(tercile_rmse(tr, oof)))
        return oof_a
    return run(save=save, name=name, fold_extra=fold_extra,
               extra_note=f"+ cross-fitted {note} aggregates (inner folds = other shared folds)", **run_kw)


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
    ap.add_argument("--mil", default=None, choices=["lgb", "spl", "splmean"], help="block-level MIL model")
    ap.add_argument("--mil_stack", action="store_true", help="add cross-fitted MIL aggregates to --model/--feat")
    ap.add_argument("--mil_seeds", type=int, default=1)
    ap.add_argument("--mil_design", action="store_true", help="add the spline-mean MIL design itself to --model/--feat")
    ap.add_argument("--mil_blocks_file", default=None, help="block table for the MIL design (default mil_blocks.parquet)")
    ap.add_argument("--het_member", action="store_true",
                    help="add gated het columns from data/het_blocks_{train,test}.parquet (ridge: gate, gate*het4, "
                         "gate*N_eff^0.25, standardized x5; trees: het4, N_eff^0.25 with NaN for ungated rows)")
    a = ap.parse_args()
    fwd = a.fwd or (20 if a.model == "fwd" else 0)
    if a.mil:
        kw = dict(model=a.model, feat_file=a.feat, cols=a.cols, seeds=a.seeds, drop=a.drop, hetero=a.hetero,
                  hetero_add=a.hetero_add) if (a.mil_stack or a.mil_design) else {}
        if a.het_member:
            if not kw:
                raise SystemExit("--het_member needs --mil_stack or --mil_design (a member model)")
            kw["het_member"] = True
        mil_run(a.mil, a.name or f"mil_{a.mil}", save=not a.no_save, mil_seeds=a.mil_seeds, stack=a.mil_stack,
                design=a.mil_design, blocks_file=a.mil_blocks_file, **kw)
        raise SystemExit
    run(a.model, a.name or f"feat_{a.model}", a.feat, a.cols, a.seeds, a.select_k, a.es, a.drop, fwd,
        a.hetero, a.mono, not a.no_save, a.hetero_add, het_member=a.het_member)
