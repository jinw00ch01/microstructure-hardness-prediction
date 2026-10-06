"""Leakage-free residual probes against the feat4 ridge base (train labels only).

Base = the feat4_v3cal_ridge_het_spat_v4 setup (same 187 columns, median impute + standardise + RidgeCV, in-fold
hetero weights from src.train_gbm.hetero_weights on ic_acg_len50_gm).  Per outer fold k the base is fitted on the
other 4 folds; inner 4-fold cross-fitted base residuals are computed inside those folds; a probe
RidgeCV(candidate -> inner residual) is fitted on the training folds (optionally on a row mask) and applied to fold k.
The probe output is reported against the outer base residual and added to the blend_v4 nested OOF."""
import re
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from eda_common import DATA_DIR, OUT, ROOT, blend_oof, rmse

sys.path.insert(0, str(ROOT))
from src.train_gbm import hetero_weights, load_features, make_model  # noqa: E402

D1 = "cal_ic_seg_L_,cal_ic_seg_mx_area_mean,cal_seg_area_cv,cal_segdk_area_cv"
F4_FILES = "features_v3.parquet,features_cal.parquet,eda_feats_lledge.parquet,eda_feats_ecs.parquet,features_v4.parquet"
F4_DROP = D1 + r",^v4(?!c_(bd|la|acd)_(mean|sd|q90|max_m_mean)$|c_(bd|la)_hp$)"
F3L_FILES = ("features_v3.parquet,features_v2.parquet,features_cal.parquet,eda_feats_lledge.parquet,"
             "eda_feats_ecs.parquet,eda_feats_lf.parquet")
CACHE = OUT / "_cf_cache.npz"


def table():
    """blend_v4 nested OOF + v3/cal/v4 columns, rows in load_train() order."""
    b = blend_oof("blend_v4")
    f = load_features("features_v3.parquet,features_cal.parquet,features_v4.parquet")
    t = b.merge(f, on="ID", how="left")
    assert (t.ID.values == b.ID.values).all()
    return t


def f4_matrix(ids):
    feats = load_features(F4_FILES)
    use = [c for c in feats.columns if c != "ID" and not any(re.search(p, c) for p in F4_DROP.split(","))]
    X = pd.DataFrame({"ID": ids}).merge(feats, on="ID", how="left")[use].replace([np.inf, -np.inf], np.nan)
    return X.reset_index(drop=True)


def _fit_pred(X, y, z, a, b):
    w, _ = hetero_weights(X.iloc[a], y[a], z[a])
    m = make_model("ridge")
    m.fit(X.iloc[a], y[a], ridgecv__sample_weight=w)
    return m.predict(X.iloc[b])


def base(t, use_cache=True):
    """outer OOF of the base and, per outer fold, (train idx, test idx, inner cross-fitted residuals)."""
    y, fold = t.hardness.values, t.fold.values
    if use_cache and CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        if (d["ids"] == t.ID.values).all():
            return d["oof"], d["inner"].item()
    X = f4_matrix(t.ID.values)
    assert X.shape[1] == 187, X.shape
    z = t[["ic_acg_len50_gm"]].values
    oof = np.zeros(len(y))
    inner = {}
    for k in range(5):
        tr_, te_ = np.where(fold != k)[0], np.where(fold == k)[0]
        oof[te_] = _fit_pred(X, y, z, tr_, te_)
        ir = np.zeros(len(tr_))
        for j in sorted(set(fold[tr_])):
            a = tr_[fold[tr_] != j]
            bm = fold[tr_] == j
            ir[bm] = y[tr_][bm] - _fit_pred(X, y, z, a, tr_[bm])
        inner[k] = (tr_, te_, ir)
    np.savez(CACHE, ids=t.ID.values, oof=oof, inner=np.array(inner, dtype=object))
    return oof, inner


def _std_fit_pred(Xtr, ytr, Xte, alphas):
    med = np.nanmedian(Xtr, 0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, med)
    Xte = np.where(np.isfinite(Xte), Xte, med)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    return RidgeCV(alphas=alphas).fit((Xtr - mu) / sd, ytr).predict((Xte - mu) / sd)


def probe(X, inner, fit_mask=None, apply_mask=None, alphas=np.logspace(-1, 6, 29)):
    """cross-fitted probe output for every row (0 where apply_mask is False)."""
    X = np.asarray(X, float)
    pr = np.zeros(len(X))
    for k, (tr_, te_, ir) in inner.items():
        m = np.ones(len(tr_), bool) if fit_mask is None else fit_mask[tr_]
        pr[te_] = _std_fit_pred(X[tr_][m], ir[m], X[te_], alphas)
    if apply_mask is not None:
        pr = np.where(apply_mask, pr, 0.0)
    return pr


class Reporter:
    def __init__(self, t, base_oof):
        self.y = t.hardness.values
        self.fold = t.fold.values
        self.br = t.resid.values  # blend_v4 nested residual
        self.base_res = self.y - base_oof
        N = t.cal_seg_count_density.values * 6.5536
        self.terc = np.asarray(pd.qcut(N, 3, labels=["coarse", "mid", "fine"]))
        self.clean = t.ic_ridge_snr.values > 0.9

    def gain(self, pr):
        return rmse(self.br, 0) - rmse(self.br - pr, 0)

    def line(self, name, pr, ncols=None):
        br, terc, clean = self.br, self.terc, self.clean
        d = [rmse(br[self.fold == k] - pr[self.fold == k], 0) - rmse(br[self.fold == k], 0) for k in range(5)]
        s = (f"[{name}]" + (f" {ncols} cols" if ncols is not None else "") +
             f" | corr(base res) {np.corrcoef(pr, self.base_res)[0, 1]:+.3f} base {rmse(self.base_res, 0):.3f}->"
             f"{rmse(self.base_res - pr, 0):.3f} | blend_v4 {rmse(br, 0):.3f}->{rmse(br - pr, 0):.3f} "
             f"({sum(x < 0 for x in d)}/5 folds) |")
        for g in ("coarse", "mid", "fine"):
            m = terc == g
            s += f" {g} {rmse(br[m], 0):.2f}->{rmse(br[m] - pr[m], 0):.2f}"
        for g in ("coarse", "mid"):
            m = clean & (terc == g)
            s += f" | clean-{g} {rmse(br[m], 0):.2f}->{rmse(br[m] - pr[m], 0):.2f} ({np.corrcoef(pr[m], br[m])[0, 1]:+.2f})"
        return s

    def perm_null(self, X, inner, n=20, seed=0, **kw):
        rng = np.random.default_rng(seed)
        X = np.asarray(X, float)
        return np.array([self.gain(probe(X[rng.permutation(len(X))], inner, **kw)) for _ in range(n)])
