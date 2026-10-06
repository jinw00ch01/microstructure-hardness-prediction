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
    st = set(range(nst)) if stages in (None, "all") else {int(s) % nst for s in str(stages).split(",")}
    sel = [b for b in meta["blocks"] if b["stage"] in st and b["pool"] in pools]
    if not sel:
        raise ValueError(f"no blocks selected in {tag} for stages={stages} pools={pools}")
    cols = np.concatenate([np.arange(b["start"], b["end"]) for b in sel])
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


def gridge3_fit_eval(X3tr, ytr, X3evs, cfg, grid):
    """Generalized ridge with one penalty per nuisance type:
    min |y - Zbar w|^2 + lv*mean_v|w.(Z_v - Zbar)|^2 + lc*mean_vc|w.(Z_vc - mean_c Z_vc)|^2
                       + la*mean_a|w.(Z_aug_a - Z_id)|^2 + alpha|w|^2
    Zbar = mean of the original views' global rows (also used for prediction); view deviations = orientation,
    cell deviations (within each view, centred over its cells) = spatial position, aug deviations (noise/blur
    copy minus the identity view) = imaging nuisance. Prep is fitted on the training images' Zbar only."""
    rows = cfg["rows"]
    pos = {r: k for k, r in enumerate(rows)}
    gidx = [k for k, (v, c) in enumerate(rows) if c == 0 and not v.startswith("aug:")]
    n, R, d = X3tr.shape
    Xbar = X3tr[:, gidx].mean(1)
    prep = Prep(cfg["col_w"], cfg["pca"], cfg["whiten"], cfg.get("n_pass", 0)).fit(Xbar)
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
    zc = Zbar.mean(0)
    Zc = Zbar - zc
    G, gvec = Zc.T @ Zc, Zc.T @ (yc - yc.mean())
    sizes = [len(X) for X in X3evs]
    P_ev = prep.transform(np.concatenate([X[:, gidx].mean(1) for X in X3evs])) - zc
    out, hps = [], []
    for lv in (grid["lam_view"] if "view" in S else [0.0]):
        for lc in (grid["lam_cell"] if "cell" in S else [0.0]):
            for la in (grid["lam_aug"] if "aug" in S else [0.0]):
                M = G + lv * S.get("view", 0.0) + lc * S.get("cell", 0.0) + la * S.get("aug", 0.0)
                w_, Q = eigh(M)
                s2, P, uty = np.maximum(w_, 0), P_ev @ Q, Q.T @ gvec
                for a in grid["alpha"]:
                    out.append(yc.mean() + P @ (uty / (s2 + a)))
                    hps.append({"lam_view": lv, "lam_cell": lc, "lam_aug": la, "alpha": a})
    P = np.stack(out) * ysd + ymu
    return np.split(P, np.cumsum(sizes)[:-1], axis=1), hps


def fit_eval(X3tr, ytr, X3evs, cfg, grid):
    """Fit prep + head path on training images; return list of (n_grid, n_eval) view-averaged predictions."""
    if cfg["head"] == "gridge":
        return gridge_fit_eval(X3tr, ytr, X3evs, cfg, grid)
    if cfg["head"] == "gridge3":
        return gridge3_fit_eval(X3tr, ytr, X3evs, cfg, grid)
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


def run_cv(X3, y, folds, X3te, cfg, grid, verbose=True):
    oof, pte = np.zeros(len(y)), np.zeros(len(X3te))
    chosen, inner_best, oracle = [], [], []
    for f in sorted(np.unique(folds)):
        tr, va = folds != f, folds == f
        # inner leave-one-fold-out CV on the outer training split
        ftr = folds[tr]
        sse = 0.0
        for g in np.unique(ftr):
            itr, iva = ftr != g, ftr == g
            (P,), hps = fit_eval(X3[tr][itr], y[tr][itr], [X3[tr][iva]], cfg, grid)
            sse = sse + ((P - y[tr][iva]) ** 2).sum(1)
        b = int(np.argmin(sse))
        (Pva, Pte), hps = fit_eval(X3[tr], y[tr], [X3[va], X3te], cfg, grid)
        oof[va] = Pva[b]
        pte += Pte[b] / len(np.unique(folds))
        chosen.append(hps[b])
        inner_best.append(float(np.sqrt(sse[b] / tr.sum())))
        oracle.append(Pva)
        if verbose:
            print(f"  fold {f}: inner {inner_best[-1]:.3f} hp {hps[b]} -> outer {rmse(Pva[b], y[va]):.3f}",
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
        if xtag:  # append the 'aug:' rows of a second extraction of the same backbone (same columns)
            ids2, X2, bl2, _, _, rw2 = load_emb(xtag, st, args.pools.split(","), 0, "global", True,
                                                args.aug_filter)
            assert [b[1] for b in bl2] == [b[1] for b in bl], f"{xtag}: column layout differs from {tag}"
            sel = [k for k, (v, c) in enumerate(rw2) if v.startswith("aug:") and (v, c) not in rw]
            X = np.concatenate([X, X2[pd.Series(np.arange(len(ids2)), index=ids2)[ids].values][:, sel]], 1)
            io = np.concatenate([io, np.zeros(len(sel), bool)])
            rw = rw + [rw2[k] for k in sel]
        if base_ids is None:
            base_ids, is_orig, rows = ids, io, rw
        else:  # align rows to the first tag's id order
            X = X[pd.Series(np.arange(len(ids)), index=ids)[base_ids].values]
            if rw != rows:
                raise ValueError(f"{tag}: view/cell row layout differs from the first embedding")
        Xs.append(elementwise(X, args.transform))
        blocks += bl
        n_orig = len(meta["views"]) - sum(n.startswith("aug:") for n in meta["views"])
        lic.append(f"{meta['backbone']} ({meta['license']}, {meta['size']}px, {n_orig} views)")
    X = np.concatenate(Xs, 2)
    V = X.shape[1]
    if args.view_std:  # per-channel spread across views = orientation dependence (anisotropy) of the texture
        X = np.concatenate([X, np.repeat(X[:, is_orig].std(1, keepdims=True), V, 1)], 2)
        blocks += [(n + "|viewstd", s) for n, s in blocks]
    col_w = np.concatenate([np.full(s, (1 / np.sqrt(s)) if args.block_norm else 1.0) for _, s in blocks])
    if args.with_feats:
        F = pd.concat([pd.read_parquet(DATA_DIR / f).set_index("ID") for f in args.feats.split(",")], axis=1)
        F = F.loc[base_ids, ~F.columns.duplicated()]
        X = np.concatenate([X, np.repeat(F.values.astype(np.float64)[:, None], V, 1)], 2)
        w = args.feat_weight / (np.sqrt(F.shape[1]) if args.block_norm else 1.0)
        col_w = np.concatenate([col_w, np.full(F.shape[1], w)])
        blocks.append((args.feats, F.shape[1]))
    if args.view_mode == "mean":
        X, is_orig, rows = X[:, is_orig].mean(1, keepdims=True), np.array([True]), [("mean", 0)]
    pos = pd.Series(np.arange(len(base_ids)), index=base_ids)
    n_pass = blocks[-1][1] if args.with_feats else 0
    return X[pos[tr.ID].values], X[pos[te.ID].values], col_w, blocks, lic, n_pass, is_orig, rows


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
    ap.add_argument("--extra-rows", default="", help="comma list (one per --emb tag) of augs-only extractions "
                    "whose 'aug:' rows are appended (needs --use-augs semantics; implies it)")
    ap.add_argument("--aug-filter", default="", help="only aug views containing one of these substrings")
    ap.add_argument("--use-augs", action="store_true",
                    help="include stored 'aug:' nuisance views as training/deviation rows (never predicted on)")
    ap.add_argument("--transform", default="none", choices=["none", "sqrt", "log"])
    ap.add_argument("--block-norm", action="store_true", help="each stage/pool block gets unit total weight")
    ap.add_argument("--view-std", action="store_true", help="append per-channel std across views")
    ap.add_argument("--with-feats", action="store_true")
    ap.add_argument("--feat-weight", type=float, default=1.0)
    ap.add_argument("--feats", default="features.parquet", help="comma list of data/*.parquet feature files")
    ap.add_argument("--pca", type=int, default=0)
    ap.add_argument("--whiten", action="store_true")
    ap.add_argument("--head", default="ridge", choices=list(HEADS))
    ap.add_argument("--grid-json", default=None, help='override head grid keys, e.g. {"lam_aug": [0, 16, 64]}')
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    if a.extra_rows:
        a.use_augs = True
    if a.head in ("gridge", "gridge3"):  # need the individual view/cell rows
        a.view_mode = "aug"
    t0 = time.time()
    tr, te = load_train(), load_test()
    Xtr, Xte, col_w, blocks, lic, n_pass, is_orig, rows = build(a, tr, te)
    y, folds = tr.hardness.values.astype(np.float64), tr.fold.values
    cfg = {"head": a.head, "col_w": col_w, "pca": a.pca, "whiten": a.whiten, "n_pass": n_pass,
           "is_orig": is_orig, "rows": rows}
    print(f"X {Xtr.shape} test {Xte.shape} blocks {len(blocks)} head {a.head}", flush=True)
    grid = dict(HEADS[a.head][1], **(json.loads(a.grid_json) if a.grid_json else {}))
    oof, pte, chosen, inner, orc, orc_hp = run_cv(Xtr, y, folds, Xte, cfg, grid)
    fr = [rmse(oof[folds == f], y[folds == f]) for f in sorted(np.unique(folds))]
    print(f"CV RMSE {rmse(oof, y):.4f} folds {np.round(fr, 3).tolist()} | best-fixed-hp (diagnostic, optimistic) "
          f"{orc:.4f} {orc_hp} | {time.time() - t0:.0f}s")
    if not a.no_save:
        name = a.name or f"emb_{a.emb.split('.')[0]}_{a.head}"
        hp_s = [",".join(f"{k}={float(v):.4g}" for k, v in h.items()) for h in chosen]
        hp_txt = " ".join(f"{h} x{hp_s.count(h)}" for h in dict.fromkeys(hp_s))
        notes = (f"{a.head} head (inner-CV hp: {hp_txt}) on {' + '.join(lic)}; stages={a.stages} "
                 f"pools={a.pools} view_mode={a.view_mode} cells={a.cells} augs={a.use_augs}{':' + a.aug_filter if a.aug_filter else ''} "
                 f"extra_rows={a.extra_rows or '-'} "
                 f"view_std={a.view_std} "
                 f"transform={a.transform} "
                 f"block_norm={a.block_norm} "
                 f"pca={a.pca}{'w' if a.whiten else ''} feats={a.feats + ' x' + str(a.feat_weight) if a.with_feats else 'no'}; "
                 f"{'grid=' + a.grid_json + '; ' if a.grid_json else ''}"
                 f"licenses: {', '.join(sorted(set(s.split('(')[1].split(',')[0] for s in lic)))}")
        save_experiment(name, tr, oof, te, pte, notes=notes)
