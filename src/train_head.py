"""Regression heads on cached frozen embeddings (data/emb/<tag>.npy from src.extract_embeddings), optionally
concatenated with handcrafted features (data/features.parquet).

Honest CV protocol: outer folds = data/folds.csv. Inside each outer training split everything is fitted on
training rows only: elementwise transform -> StandardScaler -> optional block weighting -> optional PCA ->
head, with head hyperparameters chosen by an inner leave-one-fold-out CV over the 4 remaining folds.
Test embeddings are only transformed / predicted; test prediction = mean of the 5 outer-fold models.

Views: `--view-mode mean` averages the stored TTA views into one embedding; `--view-mode aug` uses every
view as a training row (same label) and averages the per-view predictions at val/test time.
`--head gridge` (generalized ridge) instead fits on the view-averaged embedding and penalises the head's
response to within-image deviations (other views = orientation, `--cells all` grid cells = spatial position,
`--use-augs` noise/blur copies = imaging nuisance), with the penalty weight lam chosen by inner CV.

python -m src.train_head --emb resnet18.a1_in1k_256 --stages 1,2,3,4 --pools mean,std --head ridge
python -m src.train_head --emb resnet18.a1_in1k_256 --head krr --with-feats --name emb_r18_krr_feats
"""
import os
import sys


def _early_threads(default=1):
    n = default
    for i, a in enumerate(sys.argv):
        if a == "--threads" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
        elif a.startswith("--threads="):
            n = int(a.split("=", 1)[1])
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(n)
    return n


_N_THREADS = _early_threads()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.linalg import eigh  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

from .common import DATA_DIR, load_test, load_train, rmse, save_experiment  # noqa: E402

# ----------------------------------------------------------------------------------------- data


def load_emb(tag, stages="all", pools=("mean", "std"), n_views=0, cells="global", use_augs=False, aug_filter=""):
    """Returns ids, X (N, R, d), blocks, meta, is_orig (R,). The stored row axis is [view][cell] (cell 0 =
    global pooling, 1.. = grid cells when extracted with --grid); `cells` picks global / cells / all.
    Views named 'aug:*' are nuisance augmentations (noise/blur/...): only used when use_augs, and then
    only as training rows / deviation rows, never for prediction."""
    meta = json.loads((DATA_DIR / "emb" / f"{tag}_meta.json").read_text())
    E = np.load(DATA_DIR / "emb" / f"{tag}.npy", mmap_mode="r")
    ids = pd.read_csv(DATA_DIR / "emb" / f"{tag}_ids.csv").ID.values
    nst = len(meta["channels"])
    if stages == "none":  # no global columns from this tag (e.g. only its --cellstats are wanted)
        st = set()
    else:
        st = set(range(nst)) if stages in (None, "all") else {int(s) % nst for s in str(stages).split(",")}
    sel = [b for b in meta["blocks"] if b["stage"] in st and b["pool"] in pools]
    if not sel and stages != "none":
        raise ValueError(f"no blocks selected in {tag} for stages={stages} pools={pools}")
    cols = np.concatenate([np.arange(b["start"], b["end"]) for b in sel]) if sel else np.zeros(0, int)
    nc = meta.get("cells", 1)
    names = meta["views"]
    orig = [k for k, n in enumerate(names) if not n.startswith("aug:")]
    orig = orig[:n_views] if n_views else orig
    augs = [k for k, n in enumerate(names) if n.startswith("aug:")] if use_augs else []
    if aug_filter:  # keep only aug views whose name contains one of the comma-separated substrings
        augs = [k for k in augs if any(f in names[k] for f in aug_filter.split(","))]
    keep = {"global": [0], "cells": list(range(1, nc)) or [0], "all": list(range(nc))}[cells]
    vidx = [v * nc + c for v in orig + augs for c in keep]
    is_orig = np.array([v in orig for v in orig + augs for c in keep])
    rows = [(names[v], c) for v in orig + augs for c in keep]
    X = np.asarray(E[:, vidx][:, :, cols], dtype=np.float64)
    blocks = [(f"{tag}|s{b['stage']}|{b['pool']}", b["end"] - b["start"]) for b in sel]
    return ids, X, blocks, meta, is_orig, rows


def cell_stats(tag, rows, stages, pools, stats, reduce="none", transform="log", grids="2", row_mode="all"):
    """Cross-cell (spatial heterogeneity) statistics for a --grid extraction. For every head row (view v, cell 0)
    take the grid cells 1..G^2 of the same view and compute std / range / max / min over the cells per channel of
    the selected stage/pool blocks (sign*log1p if transform='log'; reduce='block' averages over the channels of each
    block, like the EDA ecs features). Rows whose view has no stored cells (grid-1 extra files) reuse the identity
    view (zero nuisance deviation). Pure per-image transform; returns ids, CS (N, R, d), blocks."""
    meta = json.loads((DATA_DIR / "emb" / f"{tag}_meta.json").read_text())
    ids = pd.read_csv(DATA_DIR / "emb" / f"{tag}_ids.csv").ID.values
    names = meta["views"]
    nst = len(meta["channels"])
    st = set(range(nst)) if stages in (None, "all") else {int(x) % nst for x in str(stages).split(",")}
    if "cell_layout" in meta:  # compact store <tag>_cells.npy (N, views, cells of all grids, Dc)
        L = meta["cell_layout"]
        Ec = np.load(DATA_DIR / "emb" / f"{tag}_cells.npy", mmap_mode="r")
        offs = np.cumsum([0] + [g * g for g in L["grids"]])
        cell_sets = {g: np.arange(offs[k], offs[k + 1]) for k, g in enumerate(L["grids"])}
        blocks_src = L["blocks"]

        def get_cells(vi, idx, cols):
            return np.asarray(Ec[:, vi][:, idx][:, :, cols], dtype=np.float64)
    else:  # cells stored inside the main array (--grid G): row = view * cells + cell, cell 0 = global
        nc = meta.get("cells", 1)
        if nc < 2:
            raise ValueError(f"{tag} has no grid cells (extract with --grid 2+ or --cell-grids)")
        E = np.load(DATA_DIR / "emb" / f"{tag}.npy", mmap_mode="r")
        cell_sets = {meta["grid"]: np.arange(1, nc)}
        blocks_src = meta["blocks"]

        def get_cells(vi, idx, cols):
            return np.asarray(E[:, vi * nc + idx][:, :, cols], dtype=np.float64)
    gl = [int(g) for g in str(grids).split(",")]
    missing = [g for g in gl if g not in cell_sets]
    if missing:
        raise ValueError(f"{tag}: grids {missing} not stored (have {list(cell_sets)})")
    sel = [b for b in blocks_src if b["stage"] in st and b["pool"] in pools]
    cols = np.concatenate([np.arange(b["start"], b["end"]) for b in sel])
    def cell_stats_view(vi):
        Fs = []
        for g in gl:
            C = get_cells(vi, cell_sets[g], cols)  # (N, cells, d)
            fs = {"std": lambda: C.std(1), "range": lambda: C.max(1) - C.min(1), "max": lambda: C.max(1),
                  "min": lambda: C.min(1)}
            F = np.concatenate([fs[x]() for x in stats], 1)
            if transform == "log":
                F = np.sign(F) * np.log1p(np.abs(F))
            if reduce == "block":  # mean over the channels of every stat/stage/pool block
                edges = np.cumsum([0] + [b["end"] - b["start"] for b in sel] * len(stats))
                F = np.stack([F[:, a:b].mean(1) for a, b in zip(edges[:-1], edges[1:])], 1)
            Fs.append(F)
        return np.concatenate(Fs, 1)

    # row_mode: all  = every row uses its own view's cells (aug rows -> noise penalty also acts on these columns)
    #           orig = 'aug:' rows reuse the identity view (no noise penalty on the heterogeneity columns)
    #           mean = average over the original views for every row (no view / aug penalty on them)
    # Rows whose view has no stored cells (grid-1 extra files) always reuse the identity view.
    cache, out = {}, []
    orig_views = [v for v, _ in rows if not v.startswith("aug:") and v in names]
    for v, _ in rows:
        if row_mode == "mean":
            key = "__mean__"
        elif v in names and (row_mode == "all" or not v.startswith("aug:")):
            key = v
        else:
            key = "id"
        if key not in cache:
            cache[key] = (np.mean([cell_stats_view(names.index(u)) for u in orig_views], 0) if key == "__mean__"
                          else cell_stats_view(names.index(key)))
        out.append(cache[key])
    CS = np.stack(out, 1)
    blocks = [(f"{tag}|cs-g{g}-{x}|s{b['stage']}|{b['pool']}", 1 if reduce == "block" else b["end"] - b["start"])
              for g in gl for x in stats for b in sel]
    return ids, CS, blocks


def elementwise(X, kind):
    if kind == "none":
        return X
    if kind == "sqrt":
        return np.sign(X) * np.sqrt(np.abs(X))
    if kind == "log":
        return np.sign(X) * np.log1p(np.abs(X))
    raise ValueError(kind)


class Prep:
    """Fitted on training rows only: impute -> standardize -> per-block weights -> optional PCA.
    The last `n_pass` columns (handcrafted features) bypass the PCA and are appended after it."""

    def __init__(self, col_w, pca=0, whiten=False, n_pass=0):
        self.col_w, self.pca, self.whiten, self.n_pass = col_w, pca, whiten, n_pass

    def fit(self, X):
        self.mu = np.nanmean(X, 0)
        sd = np.nanstd(X, 0)
        self.keep = sd > 1e-9 * (np.abs(self.mu) + 1e-12)
        self.keep[len(self.keep) - self.n_pass:] = True
        self.sd = np.where(sd > 0, sd, 1.0)
        self.is_pass = np.zeros(X.shape[1], bool)
        if self.n_pass:
            self.is_pass[-self.n_pass:] = True
        if self.pca:
            Z = self._scale(X, ~self.is_pass)
            k = min(self.pca, Z.shape[0] - 1, Z.shape[1])
            self.zmu = Z.mean(0)
            U, s, Vt = np.linalg.svd(Z - self.zmu, full_matrices=False)
            self.Vt = Vt[:k]
            self.ps = s[:k] / np.sqrt(Z.shape[0] - 1) if self.whiten else np.ones(k)
        return self

    def _scale(self, X, cols):
        m = self.keep & cols
        Z = (np.where(np.isnan(X[:, m]), self.mu[m], X[:, m]) - self.mu[m]) / self.sd[m]
        return Z * self.col_w[m]

    def transform(self, X):
        if not self.pca:
            return self._scale(X, np.ones(X.shape[1], bool))
        Z = (self._scale(X, ~self.is_pass) - self.zmu) @ self.Vt.T / self.ps
        return np.concatenate([Z, self._scale(X, self.is_pass)], 1) if self.n_pass else Z


# ----------------------------------------------------------------------------------------- heads
# Each path function fits on (Ztr, ytr) and returns predictions (n_grid, n_eval) for every grid point.


def sqdist(A, B):
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T
    return np.maximum(d, 0)


def ridge_path(Ztr, ytr, Zev, grid):
    mu = ytr.mean()
    zc = Ztr.mean(0)
    U, s, Vt = np.linalg.svd(Ztr - zc, full_matrices=False)
    uty = U.T @ (ytr - mu)
    P = (Zev - zc) @ Vt.T
    return np.stack([mu + P @ (s / (s ** 2 + a) * uty) for a in grid["alpha"]]), [
        {"alpha": a} for a in grid["alpha"]]


def krr_path(Ztr, ytr, Zev, grid):
    mu = ytr.mean()
    Dtr, Dev = sqdist(Ztr, Ztr), sqdist(Zev, Ztr)
    med = np.median(Dtr[np.triu_indices_from(Dtr, 1)])
    out, hps = [], []
    for c in grid["gamma_mult"]:
        g = c / med
        w, Q = eigh(np.exp(-g * Dtr))
        qty = Q.T @ (ytr - mu)
        Kev = np.exp(-g * Dev)
        KQ = Kev @ Q
        for a in grid["alpha"]:
            out.append(mu + KQ @ (qty / (np.maximum(w, 0) + a)))
            hps.append({"gamma_mult": c, "alpha": a})
    return np.stack(out), hps


def svr_path(Ztr, ytr, Zev, grid):
    Dtr, Dev = sqdist(Ztr, Ztr), sqdist(Zev, Ztr)
    med = np.median(Dtr[np.triu_indices_from(Dtr, 1)])
    out, hps = [], []
    for c in grid["gamma_mult"]:
        Ktr, Kev = np.exp(-c / med * Dtr), np.exp(-c / med * Dev)
        for C in grid["C"]:
            for e in grid["epsilon"]:
                m = SVR(kernel="precomputed", C=C, epsilon=e, cache_size=500).fit(Ktr, ytr)
                out.append(m.predict(Kev))
                hps.append({"gamma_mult": c, "C": C, "epsilon": e})
    return np.stack(out), hps


def lgb_path(Ztr, ytr, Zev, grid):
    import lightgbm as lgb

    out, hps = [], []
    for nl in grid["num_leaves"]:
        m = lgb.LGBMRegressor(n_estimators=max(grid["n_estimators"]), learning_rate=0.02, num_leaves=nl,
                              min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                              reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=_N_THREADS).fit(Ztr, ytr)
        for n in grid["n_estimators"]:
            out.append(m.predict(Zev, num_iteration=n))
            hps.append({"num_leaves": nl, "n_estimators": n})
    return np.stack(out), hps


HEADS = {
    "lgb": (lgb_path, {"num_leaves": [4, 15], "n_estimators": [150, 300, 600, 1200, 2400]}),
    "ridge": (ridge_path, {"alpha": np.logspace(-1, 7, 33)}),
    "gridge": (None, {"lam": [0.0, 2.0, 8.0, 32.0], "alpha": np.logspace(-1, 7, 33)}),
    "gridge3": (None, {"lam_view": [0.0, 4.0, 16.0], "lam_cell": [0.0, 1.0, 4.0, 16.0],
                       "lam_aug": [0.0, 4.0, 16.0, 64.0], "alpha": np.logspace(-1, 7, 17)}),
    "krr": (krr_path, {"gamma_mult": [0.0078, 0.0156, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0],
                       "alpha": np.logspace(-4, 1, 16)}),
    "svr": (svr_path, {"gamma_mult": [0.125, 0.25, 0.5, 1.0], "C": [1.0, 3.0, 10.0, 30.0],
                       "epsilon": [0.05, 0.2]}),
}

# ----------------------------------------------------------------------------------------- CV


def gridge_fit_eval(X3tr, ytr, X3evs, cfg, grid):
    """Generalized ridge: min |y - Zbar w|^2 + lam/V * sum_r |D_r w|^2 + alpha |w|^2, where Zbar is the mean
    over the original rows (views/cells) and D_r = Z_r - Zbar are within-image deviations of every row (views =
    orientation, grid cells = spatial position, 'aug:' rows = noise/blur). lam=1 ~ plain view augmentation,
    lam=0 ~ ridge on the mean. Prep (scaler/PCA) is fitted on the training images' Zbar only."""
    n, V, d = X3tr.shape
    io = cfg.get("is_orig", np.ones(V, bool))
    Xbar = X3tr[:, io].mean(1)
    prep = Prep(cfg["col_w"], cfg["pca"], cfg["whiten"], cfg.get("n_pass", 0)).fit(Xbar)
    Zbar = prep.transform(Xbar)
    Dev = (prep.transform(X3tr.reshape(n * V, d)).reshape(n, V, -1) - Zbar[:, None]).reshape(n * V, -1)
    ymu, ysd = ytr.mean(), ytr.std()
    yc = (ytr - ymu) / ysd
    zc = Zbar.mean(0)
    sizes = [len(X) for X in X3evs]
    P_ev = (prep.transform(np.concatenate([X[:, io].mean(1) for X in X3evs])) - zc)
    out, hps = [], []
    Zc, k = Zbar - zc, Zbar.shape[1]
    normal_eq = n * (V + 1) > 1.5 * k  # many rows: solve via k x k normal equations, else via SVD
    if normal_eq:
        G, S, g = Zc.T @ Zc, Dev.T @ Dev / V, Zc.T @ (yc - yc.mean())
    for lam in grid["lam"]:
        if normal_eq:
            w_, Q = eigh(G + lam * S)
            s2, P, uty = np.maximum(w_, 0), P_ev @ Q, Q.T @ g
        else:
            A = np.concatenate([Zc, np.sqrt(lam / V) * Dev]) if lam > 0 else Zc
            b = np.concatenate([yc - yc.mean(), np.zeros(n * V)]) if lam > 0 else yc - yc.mean()
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            s2, P, uty = s ** 2, P_ev @ Vt.T, s * (U.T @ b)
        for a in grid["alpha"]:
            out.append(yc.mean() + P @ (uty / (s2 + a)))
            hps.append({"lam": lam, "alpha": a})
    P = np.stack(out) * ysd + ymu
    return np.split(P, np.cumsum(sizes)[:-1], axis=1), hps


def gridge3_fit_eval(X3tr, ytr, X3evs, cfg, grid, sw=None):
    """Generalized ridge with one penalty per nuisance type:
    min sum_i sw_i (y_i - Zbar_i w)^2 + lv*mean_v|w.(Z_v - Zbar)|^2 + lc*mean_vc|w.(Z_vc - mean_c Z_vc)|^2
                       + la*mean_a|w.(Z_aug_a - Z_id)|^2 + alpha|w|^2
    Zbar = mean of the original views' global rows (also used for prediction); view deviations = orientation,
    cell deviations (within each view, centred over its cells) = spatial position, aug deviations (noise/blur
    copy minus the identity view) = imaging nuisance. Prep is fitted on the training images' Zbar only.
    sw = optional per-image sample weights (mean 1) for the data term (heteroscedastic labels)."""
    rows = cfg["rows"]
    pos = {r: k for k, r in enumerate(rows)}
    gidx = [k for k, (v, c) in enumerate(rows) if c == 0 and not v.startswith("aug:")]
    n, R, d = X3tr.shape
    Xbar = X3tr[:, gidx].mean(1)
    prep = Prep(cfg["col_w"], cfg["pca"], cfg["whiten"], cfg.get("n_pass", 0)).fit(Xbar)
    # own columns (--own-col): NaN outside their group -> Prep imputes the in-fold mean of the group's training
    # images -> exactly 0 for every other image (all rows, so no view/aug deviation either). grid['own_w'] = column
    # weight of the own blocks, i.e. their own ridge group (penalty alpha / own_w^2), chosen by inner CV.
    own = cfg.get("own_grp")
    own_k = own[prep.keep] if own is not None and (own > 0).any() and not cfg["pca"] else None
    own_ws = grid.get("own_w", [1.0]) if own_k is not None else [None]
    Zbar = prep.transform(Xbar)
    Zall = prep.transform(X3tr.reshape(n * R, d)).reshape(n, R, -1)
    S = {}
    Dv = [Zall[:, k] - Zbar for k in gidx]
    if len(Dv) > 1:
        S["view"] = sum(D.T @ D for D in Dv) / len(Dv)
    cells = {}
    for k, (v, c) in enumerate(rows):
        if c > 0 and not v.startswith("aug:"):
            cells.setdefault(v, []).append(k)
    if cells:
        Dc = [Zall[:, ks] - Zall[:, ks].mean(1, keepdims=True) for ks in cells.values()]
        S["cell"] = sum(np.einsum("nci,ncj->ij", D, D) for D in Dc) / sum(D.shape[1] for D in Dc)
    aug = [k for k, (v, c) in enumerate(rows) if c == 0 and v.startswith("aug:")]
    if aug and ("id", 0) in pos:
        Da = [Zall[:, k] - Zall[:, pos[("id", 0)]] for k in aug]
        S["aug"] = sum(D.T @ D for D in Da) / len(Da)
    ymu, ysd = ytr.mean(), ytr.std()
    yc = (ytr - ymu) / ysd
    sw = np.ones(n) if sw is None else np.asarray(sw, np.float64) / np.mean(sw)
    zc = (sw[:, None] * Zbar).sum(0) / sw.sum()  # weighted centring = unpenalised intercept under weights
    ybar = float((sw * yc).sum() / sw.sum())
    Zc = Zbar - zc
    G, gvec = Zc.T @ (sw[:, None] * Zc), Zc.T @ (sw * (yc - ybar))
    sizes = [len(X) for X in X3evs]
    P_ev = prep.transform(np.concatenate([X[:, gidx].mean(1) for X in X3evs])) - zc
    out, hps = [], []
    for ow in own_ws:
        if ow is None:  # no own columns: unchanged path
            G_, gv_, S_, Pev_, ext = G, gvec, S, P_ev, {}
        else:  # column scaling commutes with the (weighted) centring, so rescale the unit-weight Gram matrices
            dv = np.where(own_k > 0, float(ow), 1.0)
            DD = np.outer(dv, dv)
            G_, gv_, S_, Pev_, ext = G * DD, gvec * dv, {k: M * DD for k, M in S.items()}, P_ev * dv, {"own_w": ow}
        for lv in (grid["lam_view"] if "view" in S_ else [0.0]):
            for lc in (grid["lam_cell"] if "cell" in S_ else [0.0]):
                for la in (grid["lam_aug"] if "aug" in S_ else [0.0]):
                    M = G_ + lv * S_.get("view", 0.0) + lc * S_.get("cell", 0.0) + la * S_.get("aug", 0.0)
                    w_, Q = eigh(M)
                    s2, P, uty = np.maximum(w_, 0), Pev_ @ Q, Q.T @ gv_
                    for a in grid["alpha"]:
                        out.append(ybar + P @ (uty / (s2 + a)))
                        hps.append({"lam_view": lv, "lam_cell": lc, "lam_aug": la, "alpha": a, **ext})
    P = np.stack(out) * ysd + ymu
    return np.split(P, np.cumsum(sizes)[:-1], axis=1), hps


def fit_eval(X3tr, ytr, X3evs, cfg, grid, sw=None):
    """Fit prep + head path on training images; return list of (n_grid, n_eval) view-averaged predictions."""
    if cfg["head"] == "gridge3":
        return gridge3_fit_eval(X3tr, ytr, X3evs, cfg, grid, sw=sw)
    if sw is not None:
        raise ValueError("sample weights (--hetero) are implemented for --head gridge3 only")
    if cfg["head"] == "gridge":
        return gridge_fit_eval(X3tr, ytr, X3evs, cfg, grid)
    n, V, d = X3tr.shape
    io = cfg.get("is_orig", np.ones(V, bool))
    Vo = int(io.sum())
    prep = Prep(cfg["col_w"], cfg["pca"], cfg["whiten"], cfg.get("n_pass", 0)).fit(X3tr.reshape(n * V, d))
    ymu, ysd = ytr.mean(), ytr.std()
    Ztr = prep.transform(X3tr.reshape(n * V, d))
    sizes = [len(X) for X in X3evs]
    # training uses every row (views, cells, nuisance augs); evaluation averages over original rows only
    Zev = prep.transform(np.concatenate([X[:, io].reshape(len(X) * Vo, d) for X in X3evs]))
    P, hps = HEADS[cfg["head"]][0](Ztr, np.repeat((ytr - ymu) / ysd, V), Zev, grid)
    P = P.reshape(len(P), -1, Vo).mean(-1) * ysd + ymu
    return np.split(P, np.cumsum(sizes)[:-1], axis=1), hps


def hetero_weights(res, z, clip=(0.25, 4.0)):
    """Inverse-variance sample weights from training rows only: OLS of log(res^2 + 1) on [1, Z] (Z columns
    log-transformed when strictly positive), w = 1 / exp(fit), normalised to mean 1 and clipped.
    res = inner-CV residuals of the unweighted head; z = variance covariates (e.g. correlation length)."""
    Z = pd.DataFrame(np.asarray(z, np.float64).reshape(len(res), -1))
    Z = Z.fillna(Z.median())
    Z = Z.apply(lambda c: np.log(c) if (c > 0).all() else c)
    D = np.column_stack([np.ones(len(res)), Z.values])
    coef, *_ = np.linalg.lstsq(D, np.log(res ** 2 + 1.0), rcond=None)
    w = 1.0 / np.exp(D @ coef)
    w = np.clip(w / w.mean(), *clip)
    return w / w.mean(), coef[1:]


def inner_cv(X3tr, ytr, ftr, cfg, grid, sw=None):
    """Leave-one-fold-out CV on an outer training split; returns (sse per grid point, inner OOF, hps)."""
    ioof = None
    for g in np.unique(ftr):
        itr, iva = ftr != g, ftr == g
        (P,), hps = fit_eval(X3tr[itr], ytr[itr], [X3tr[iva]], cfg, grid, sw=None if sw is None else sw[itr])
        if ioof is None:
            ioof = np.zeros((len(P), len(ytr)))
        ioof[:, iva] = P
    return ((ioof - ytr) ** 2).sum(1), ioof, hps


def run_cv(X3, y, folds, X3te, cfg, grid, verbose=True):
    """Outer CV. cfg['hetero_z'] (train-aligned covariates) -> two-pass weighted fit: unweighted inner CV
    residuals -> hetero_weights on the outer training rows -> weighted inner CV + final fit.
    cfg['hp_avg'] = K averages the predictions of the K best grid points by inner CV (default 1 = argmin)."""
    oof, pte = np.zeros(len(y)), np.zeros(len(X3te))
    chosen, inner_best, oracle = [], [], []
    hz, K = cfg.get("hetero_z"), int(cfg.get("hp_avg", 1))
    for f in sorted(np.unique(folds)):
        tr, va = folds != f, folds == f
        sse, ioof, hps = inner_cv(X3[tr], y[tr], folds[tr], cfg, grid)
        sw = None
        if hz is not None:
            sw, coef = hetero_weights(y[tr] - ioof[int(np.argmin(sse))], hz[tr])
            if verbose:
                print(f"  fold {f}: hetero log-var slopes {np.round(coef, 3).tolist()}; "
                      f"weights {sw.min():.2f}-{sw.max():.2f}", flush=True)
            sse, ioof, hps = inner_cv(X3[tr], y[tr], folds[tr], cfg, grid, sw=sw)
        bs = np.argsort(sse, kind="stable")[:K]
        b = int(bs[0])
        (Pva, Pte), hps = fit_eval(X3[tr], y[tr], [X3[va], X3te], cfg, grid, sw=sw)
        oof[va] = Pva[bs].mean(0)
        pte += Pte[bs].mean(0) / len(np.unique(folds))
        chosen.append(hps[b])
        inner_best.append(float(np.sqrt(sse[b] / tr.sum())))
        oracle.append(Pva)
        if verbose:
            print(f"  fold {f}: inner {inner_best[-1]:.3f} hp {hps[b]} -> outer {rmse(oof[va], y[va]):.3f}",
                  flush=True)
    # diagnostic only (not used for selection): best fixed grid point judged on the outer folds
    O = np.zeros((len(oracle[0]), len(y)))
    for f, Pva in zip(sorted(np.unique(folds)), oracle):
        O[:, folds == f] = Pva
    orc = np.sqrt(((O - y) ** 2).mean(1))
    return oof, pte, chosen, inner_best, float(orc.min()), hps[int(orc.argmin())]


def build(args, tr, te):
    blocks, Xs, lic, base_ids, is_orig, rows = [], [], [], None, None, None
    tags = args.emb.split(",")
    extra = args.extra_rows.split(",") if args.extra_rows else [""] * len(tags)
    assert len(extra) == len(tags), "--extra-rows needs one tag per --emb tag"
    stage_specs = args.stages.split(";") if ";" in args.stages else [args.stages] * len(tags)
    assert len(stage_specs) == len(tags), "--stages: give one ';'-separated spec per --emb tag"
    for tag, xtag, st in zip(tags, extra, stage_specs):
        ids, X, bl, meta, io, rw = load_emb(tag, st, args.pools.split(","), args.views, args.cells,
                                            args.use_augs, args.aug_filter)
        for xt in [t for t in xtag.split("+") if t]:  # append rows of extra extractions of the same backbone
            ids2, X2, bl2, _, io2, rw2 = load_emb(xt, st, args.pools.split(","), 0, "global", True,
                                                 args.aug_filter)
            assert [b[1] for b in bl2] == [b[1] for b in bl], f"{xt}: column layout differs from {tag}"
            sel = [k for k, r in enumerate(rw2) if r not in rw]  # new orientation views and/or 'aug:' rows
            X = np.concatenate([X, X2[pd.Series(np.arange(len(ids2)), index=ids2)[ids].values][:, sel]], 1)
            io = np.concatenate([io, io2[sel]])
            rw = rw + [rw2[k] for k in sel]
        X = elementwise(X, args.transform)
        if args.cellstats:  # spatial heterogeneity across grid cells, appended per view row
            ids3, CS, csb = cell_stats(tag, rw, args.cs_stages, args.cs_pools.split(","), args.cellstats.split(","),
                                       args.cs_reduce, args.cs_transform, args.cs_grid, args.cs_rows)
            assert (ids3 == ids).all()
            X = np.concatenate([X, CS], 2)
            bl = bl + csb
        if base_ids is None:
            base_ids, is_orig, rows = ids, io, rw
        else:  # align rows to the first tag's id order
            X = X[pd.Series(np.arange(len(ids)), index=ids)[base_ids].values]
            if rw != rows:
                raise ValueError(f"{tag}: view/cell row layout differs from the first embedding")
        Xs.append(X)
        blocks += bl
        n_orig = len(meta["views"]) - sum(n.startswith("aug:") for n in meta["views"])
        lic.append(f"{meta['backbone']} ({meta['license']}, {meta['size']}px, {n_orig} views)")
    X = np.concatenate(Xs, 2)
    V = X.shape[1]
    if args.view_std:  # per-channel spread across views = orientation dependence (anisotropy) of the texture
        X = np.concatenate([X, np.repeat(X[:, is_orig].std(1, keepdims=True), V, 1)], 2)
        blocks += [(n + "|viewstd", s) for n, s in blocks]
    col_w = np.concatenate([np.full(s, ((1 / np.sqrt(s)) if args.block_norm else 1.0)
                                    * (args.cs_weight if "|cs-" in n else 1.0)) for n, s in blocks])
    own_grp, args.own_info = np.zeros(X.shape[2], int), ""
    if args.own_col:  # own columns: separate slopes per group of a label-free per-image statistic (e.g. ic_noise)
        g_all, args.own_info = own_groups(args, base_ids, tr, te)
        d0, base_blocks = X.shape[2], list(blocks)
        for g in range(1, int(g_all.max()) + 1):  # group 0 (at or below the first cut) uses the shared slopes only
            X = np.concatenate([X, np.where((g_all == g)[:, None, None], X[:, :, :d0], np.nan)], 2)
            col_w = np.concatenate([col_w, col_w[:d0]])
            blocks += [(n + f"|own{g}", s) for n, s in base_blocks]
            own_grp = np.concatenate([own_grp, np.full(d0, g)])
    if args.with_feats:
        F = pd.concat([pd.read_parquet(DATA_DIR / f).set_index("ID") for f in args.feats.split(",")], axis=1)
        F = F.loc[base_ids, ~F.columns.duplicated()]
        X = np.concatenate([X, np.repeat(F.values.astype(np.float64)[:, None], V, 1)], 2)
        w = args.feat_weight / (np.sqrt(F.shape[1]) if args.block_norm else 1.0)
        col_w = np.concatenate([col_w, np.full(F.shape[1], w)])
        blocks.append((args.feats, F.shape[1]))
        own_grp = np.concatenate([own_grp, np.zeros(F.shape[1], int)])
    if args.view_mode == "mean":
        X, is_orig, rows = X[:, is_orig].mean(1, keepdims=True), np.array([True]), [("mean", 0)]
    pos = pd.Series(np.arange(len(base_ids)), index=base_ids)
    n_pass = blocks[-1][1] if args.with_feats else 0
    return X[pos[tr.ID].values], X[pos[te.ID].values], col_w, blocks, lic, n_pass, is_orig, rows, own_grp


def own_groups(args, ids, tr, te):
    """Group index per image for --own-col: cuts = quantiles (--own-q, e.g. '2/3' or '1/3,2/3') of the column over
    the TRAIN images only (label-free per-image statistic, fixed before any CV); group = number of cuts the value
    exceeds, so '2/3' -> 1 for the top tercile (value > cut), else 0. --own-random SEED is a control: the same
    number of images per group, drawn at random from all train + test images."""
    from fractions import Fraction

    O = pd.read_parquet(DATA_DIR / args.own_file).set_index("ID")[args.own_col]
    assert O.loc[ids].notna().all(), f"{args.own_col} has missing values"
    qs = [float(Fraction(q)) for q in args.own_q.split(",")]
    cuts = [float(np.quantile(O.loc[tr.ID].values, q)) for q in qs]
    g = (O.loc[ids].values[:, None] > np.array(cuts)[None]).sum(1)
    if args.own_random >= 0:
        g = np.random.default_rng(args.own_random).permutation(g)
    gs = pd.Series(g, index=ids)
    cnt = lambda idx: "/".join(str(int((gs.loc[idx] == k).sum())) for k in range(len(cuts) + 1))  # noqa: E731
    info = (f"own={args.own_col}@{args.own_file} q={args.own_q} cuts={','.join(f'{c:.4f}' for c in cuts)} "
            f"groups train {cnt(tr.ID)} test {cnt(te.ID)}{' RANDOM seed ' + str(args.own_random) if args.own_random >= 0 else ''}")
    print(info, flush=True)
    return g, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="comma-separated embedding tags in data/emb/")
    ap.add_argument("--stages", default="all", help="'all' or comma list of stage indices (negatives ok); "
                    "use ';' to give one spec per --emb tag, e.g. '1,2,3;1,2'")
    ap.add_argument("--pools", default="mean,std")
    ap.add_argument("--views", type=int, default=0, help="use the first N stored views (0 = all)")
    ap.add_argument("--view-mode", default="mean", choices=["mean", "aug"])
    ap.add_argument("--cells", default="global", choices=["global", "cells", "all"],
                    help="for --grid extractions: rows from global pooling, grid cells, or both")
    ap.add_argument("--extra-rows", default="", help="comma list (one per --emb tag; join several with '+') of "
                    "extra extractions of the same backbone whose new rows (extra orientation views and/or 'aug:' "
                    "copies) are appended; implies --use-augs")
    ap.add_argument("--aug-filter", default="", help="only aug views containing one of these substrings")
    ap.add_argument("--use-augs", action="store_true",
                    help="include stored 'aug:' nuisance views as training/deviation rows (never predicted on)")
    ap.add_argument("--transform", default="none", choices=["none", "sqrt", "log"])
    ap.add_argument("--block-norm", action="store_true", help="each stage/pool block gets unit total weight")
    ap.add_argument("--view-std", action="store_true", help="append per-channel std across views")
    ap.add_argument("--cellstats", default="", help="cross-cell stats from --grid extractions, e.g. std,range,max")
    ap.add_argument("--cs-stages", default="0,1,2", help="stages for --cellstats (same index for every tag)")
    ap.add_argument("--cs-pools", default="mean,std", help="pools for --cellstats")
    ap.add_argument("--cs-reduce", default="none", choices=["none", "block"], help="block = mean over channels")
    ap.add_argument("--cs-transform", default="log", choices=["none", "log"])
    ap.add_argument("--cs-weight", type=float, default=1.0, help="column weight of the cellstat block(s)")
    ap.add_argument("--cs-grid", default="2", help="grid(s) for --cellstats, e.g. 2 | 4 | 2,4 (stored grids only)")
    ap.add_argument("--cs-grid-bag", default="", help="';'-separated cs-grid specs, e.g. '2;4;8;2,4;2,8;4,8;2,4,8': "
                    "equal-weight average of one full CV run per spec (fixed weights, honest)")
    ap.add_argument("--cs-rows", default="all", choices=["all", "orig", "mean"],
                    help="which rows carry their own cellstats (see cell_stats); orig/mean exempt them from penalties")
    ap.add_argument("--hetero", default="", help="comma list of --hetero-file columns modelling the residual "
                    "variance -> in-fold inverse-variance sample weights (gridge3 only), e.g. ic_acg_len50_gm")
    ap.add_argument("--hetero-file", default="features_v3.parquet")
    ap.add_argument("--hp-avg", type=int, default=1, help="average the K best grid points by inner CV")
    ap.add_argument("--own-col", default="", help="own columns: the whole head input is duplicated per group of this "
                    "--own-file column (NaN outside the group -> in-fold group mean -> 0), giving each group above the "
                    "first cut its own slopes; gridge3 grid key own_w = their column weight (own ridge group)")
    ap.add_argument("--own-file", default="features_v3.parquet")
    ap.add_argument("--own-q", default="2/3", help="train-image quantile cut(s) of --own-col, e.g. '2/3' (top "
                    "tercile gets own slopes) or '1/3,2/3' (mid and top terciles each get own slopes)")
    ap.add_argument("--own-random", type=int, default=-1, help="control: permute the groups at random (seed)")
    ap.add_argument("--with-feats", action="store_true")
    ap.add_argument("--feat-weight", type=float, default=1.0)
    ap.add_argument("--feats", default="features.parquet", help="comma list of data/*.parquet feature files")
    ap.add_argument("--pca", type=int, default=0)
    ap.add_argument("--whiten", action="store_true")
    ap.add_argument("--head", default="ridge", choices=list(HEADS))
    ap.add_argument("--grid-json", default=None, help='override head grid keys, e.g. {"lam_aug": [0, 16, 64]}')
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--oof-out", default="", help="also write the OOF predictions to this CSV path")
    ap.add_argument("--test-out", default="", help="also write the test predictions to this CSV path (scratch)")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    if a.extra_rows:
        a.use_augs = True
    if a.head in ("gridge", "gridge3"):  # need the individual view/cell rows
        a.view_mode = "aug"
    t0 = time.time()
    tr, te = load_train(), load_test()
    y, folds = tr.hardness.values.astype(np.float64), tr.fold.values
    grid = dict(HEADS[a.head][1], **(json.loads(a.grid_json) if a.grid_json else {}))
    # --cs-grid-bag: fixed equal-weight average of full honest CV runs, one per cellstats grid spec
    specs = [g for g in a.cs_grid_bag.split(";") if g] if a.cs_grid_bag else [a.cs_grid]
    hz = None
    if a.hetero:  # variance covariates for training images only (weights never touch test rows)
        H = pd.read_parquet(DATA_DIR / a.hetero_file).set_index("ID")
        hz = H.loc[tr.ID, a.hetero.split(",")].values.astype(np.float64)
    oofs, ptes, chosen = [], [], []
    for spec in specs:
        a.cs_grid = spec
        Xtr, Xte, col_w, blocks, lic, n_pass, is_orig, rows, own_grp = build(a, tr, te)
        cfg = {"head": a.head, "col_w": col_w, "pca": a.pca, "whiten": a.whiten, "n_pass": n_pass,
               "is_orig": is_orig, "rows": rows, "hetero_z": hz, "hp_avg": a.hp_avg}
        if (own_grp > 0).any():
            cfg["own_grp"] = own_grp
        print(f"X {Xtr.shape} test {Xte.shape} blocks {len(blocks)} head {a.head}"
              f"{' cs_grid=' + spec if a.cellstats else ''}", flush=True)
        oof_k, pte_k, chosen_k, inner, orc, orc_hp = run_cv(Xtr, y, folds, Xte, cfg, grid, verbose=len(specs) == 1)
        fr = [rmse(oof_k[folds == f], y[folds == f]) for f in sorted(np.unique(folds))]
        print(f"CV RMSE {rmse(oof_k, y):.4f} folds {np.round(fr, 3).tolist()} | best-fixed-hp (diagnostic, "
              f"optimistic) {orc:.4f} {orc_hp} | {time.time() - t0:.0f}s", flush=True)
        oofs.append(oof_k)
        ptes.append(pte_k)
        chosen += chosen_k
    oof, pte = np.mean(oofs, 0), np.mean(ptes, 0)
    if len(specs) > 1:
        a.cs_grid = "bag[" + ";".join(specs) + "]"
        fr = [rmse(oof[folds == f], y[folds == f]) for f in sorted(np.unique(folds))]
        print(f"BAG CV RMSE {rmse(oof, y):.4f} folds {np.round(fr, 3).tolist()} ({len(specs)} members)")
    if a.oof_out:  # scratch OOF dump for paired comparisons (does not register an experiment)
        pd.DataFrame({"ID": tr.ID, "hardness": oof}).to_csv(a.oof_out, index=False)
    if a.test_out:  # scratch test dump (mean of the fold models; keep it out of the repo)
        pd.DataFrame({"ID": te.ID, "hardness": pte}).to_csv(a.test_out, index=False)
    if not a.no_save:
        name = a.name or f"emb_{a.emb.split('.')[0]}_{a.head}"
        hp_s = [",".join(f"{k}={float(v):.4g}" for k, v in h.items()) for h in chosen]
        hp_txt = " ".join(f"{h} x{hp_s.count(h)}" for h in dict.fromkeys(hp_s))
        data_note = "" if DATA_DIR.name == "data" else f" [DATA_DIR={DATA_DIR.name}]"
        notes = (f"{a.head} head (inner-CV hp: {hp_txt}) on {' + '.join(lic)}; emb={a.emb}{data_note}; stages={a.stages} "
                 f"pools={a.pools} view_mode={a.view_mode} cells={a.cells} augs={a.use_augs}{':' + a.aug_filter if a.aug_filter else ''} "
                 f"extra_rows={a.extra_rows or '-'} "
                 f"view_std={a.view_std} "
                 f"{'cellstats=' + a.cellstats + '@g' + a.cs_grid + ':' + a.cs_rows + ':s' + a.cs_stages + ':' + a.cs_pools + ':' + a.cs_reduce + ':' + a.cs_transform + ':w' + str(a.cs_weight) + ' ' if a.cellstats else ''}"
                 f"transform={a.transform} "
                 f"block_norm={a.block_norm} "
                 f"pca={a.pca}{'w' if a.whiten else ''} feats={a.feats + ' x' + str(a.feat_weight) if a.with_feats else 'no'}"
                 f"{' hetero=' + a.hetero + '@' + a.hetero_file + ' (in-fold inverse-variance weights)' if a.hetero else ''}"
                 f"{' hp_avg=' + str(a.hp_avg) if a.hp_avg > 1 else ''}"
                 f"{' ' + a.own_info + ' (own slopes per group; NaN outside -> in-fold group mean -> 0)' if a.own_info else ''}; "
                 f"{'grid=' + a.grid_json + '; ' if a.grid_json else ''}"
                 f"licenses: {', '.join(sorted(set(s.split('(')[1].split(',')[0] for s in lic)))}")
        save_experiment(name, tr, oof, te, pte, notes=notes)
