"""Non-negative OOF blend of experiments. python -m src.ensemble [exp1 exp2 ...] --out blend_v1

Covariate-aware stacker (opt-in, the default NNLS path above is unchanged):
    python -m src.ensemble EXP1 EXP2 ... --stacker linear --cov snr [--dry-run] --out blend_v5
Member weights vary with label-free per-image covariates (log ic_ridge_snr, log cal_seg_count_density),
stay non-negative and sum to 1 at every covariate value. Covariate quantiles/edges are cut on the rows
a stacker is fit on (never on test). Shrinkage toward the global blend is picked by inner CV inside each
outer fold; the report compares fold-paired against plain NNLS (fit_w) over the same members.
"""
import argparse
import itertools
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

from .common import DATA_DIR, EXP_DIR, SUB_DIR, load_test, load_train, rmse


def fit_w(P, y):
    w, _ = nnls(P, y)
    return w / w.sum() if w.sum() > 0 else np.full(P.shape[1], 1 / P.shape[1])


# ---------------------------------------------------------------- covariate-aware stacker
COV_SRC = {"snr": ("features_v3.parquet", "ic_ridge_snr"), "n": ("features_cal.parquet", "cal_seg_count_density")}


def load_covs(ids, names):
    """log of per-image covariates (fixed image transforms, train and test alike), aligned with ids."""
    out = pd.DataFrame({"ID": np.asarray(ids)})
    for nm in names:
        fn, col = COV_SRC[nm]
        out = out.merge(pd.read_parquet(DATA_DIR / fn, columns=["ID", col]), on="ID", how="left")
    Z = np.log(out[[COV_SRC[nm][1] for nm in names]].to_numpy(float))
    assert np.isfinite(Z).all(), "missing or non-positive covariate"
    return Z


class CovStacker:
    """Blend whose member weights depend on covariates Z (n x J).

    kind="linear":  w(t) = w0 + sum_j v_j * t_j with t_j = Z_j mapped linearly to [0, 1] between its
                    fit-row quantiles q (clipped outside). Constrained >= 0 at every corner of [0, 1]^J,
                    hence everywhere; sum(w0) = 1, sum(v_j) = 0.
    kind="tercile": one weight vector per fit-row tercile of Z[:, 0] (J must be 1), each on the simplex.
    alpha shrinks toward the global blend: every corner/cell blend also pays alpha/B times its squared
    error on all fit rows (alpha = 1: as much weight as the data term; alpha = inf: global fit_w weights).
    """

    def __init__(self, kind="linear", alpha=0.0, q=(0.1, 0.9)):
        self.kind, self.alpha, self.q = kind, float(alpha), q

    def _t(self, Z):
        if self.kind == "tercile":
            return np.digitize(Z[:, 0], self.edges)
        return np.clip((Z - self.lo) / (self.hi - self.lo), 0.0, 1.0)

    def _maps(self):
        """K x D matrices mapping theta to each corner (linear) or cell (tercile) blend."""
        K, J, I = self.K, self.J, np.eye(self.K)
        if self.kind == "tercile":
            return [np.hstack([I if b == c else 0 * I for b in range(3)]) for c in range(3)]
        return [np.hstack([I] + [c[j] * I for j in range(J)]) for c in itertools.product([0, 1], repeat=J)]

    def _design(self, P, t):
        if self.kind == "tercile":
            return np.hstack([P * (t == c)[:, None] for c in range(3)])
        return np.hstack([P] + [P * t[:, [j]] for j in range(self.J)])

    def fit(self, P, y, Z):
        n, self.K = P.shape
        self.J = Z.shape[1]
        if self.kind == "tercile":
            assert self.J == 1, "tercile stacker takes one covariate"
            self.edges = np.quantile(Z[:, 0], [1 / 3, 2 / 3])
        else:
            self.lo, self.hi = np.quantile(Z, self.q[0], axis=0), np.quantile(Z, self.q[1], axis=0)
        self.wg = fit_w(P, y)
        maps = self._maps()
        if np.isinf(self.alpha):  # no covariate effect: exactly the plain NNLS blend
            self.theta = np.linalg.lstsq(np.vstack(maps), np.tile(self.wg, len(maps)), rcond=None)[0]
            return self
        # every blend sums to 1, so P w = m + (P - m) w: centre on the row mean for conditioning
        m = P.mean(1)
        Pc, yc = P - m[:, None], y - m
        X = self._design(Pc, self._t(Z))
        H, f = X.T @ X, X.T @ yc
        if self.alpha > 0:
            G0, g0 = Pc.T @ Pc, Pc.T @ yc
            for C in maps:
                H += self.alpha / len(maps) * C.T @ G0 @ C
                f += self.alpha / len(maps) * C.T @ g0
        s = np.trace(H) / len(H)
        H, f = H / s, f / s
        D, K = X.shape[1], self.K
        if self.kind == "tercile":
            A = np.kron(np.eye(3), np.ones((1, K)))
            b = np.ones(3)
        else:
            A = np.kron(np.eye(1 + self.J), np.ones((1, K)))
            b = np.r_[1.0, np.zeros(self.J)]
        G = np.vstack(maps)
        x0 = np.linalg.lstsq(G, np.tile(self.wg, len(maps)), rcond=None)[0]  # feasible start: global weights
        r = minimize(lambda x: 0.5 * x @ H @ x - f @ x, x0, jac=lambda x: H @ x - f, method="SLSQP",
                     constraints=[{"type": "eq", "fun": lambda x: A @ x - b, "jac": lambda x: A},
                                  {"type": "ineq", "fun": lambda x: G @ x, "jac": lambda x: G}],
                     options={"ftol": 1e-14, "maxiter": 2000})
        self.theta = r.x
        self.converged = bool(r.success)
        return self

    def weights(self, Z):
        """Per-row member weights (n x K)."""
        t, th, K = self._t(Z), self.theta, self.K
        if self.kind == "tercile":
            W = th.reshape(3, K)[t]
        else:
            W = th[:K][None, :] + sum(t[:, [j]] * th[(j + 1) * K:(j + 2) * K][None, :] for j in range(self.J))
        W = np.clip(W, 0, None)  # remove solver round-off below zero
        return W / W.sum(1, keepdims=True)

    def predict(self, P, Z):
        return (P * self.weights(Z)).sum(1)

    def corner_weights(self):
        return [C @ self.theta for C in self._maps()]


def select_alpha(P, y, Z, folds, kind, q, grid):
    """Leave-one-fold-out CV over the folds present in the given rows; returns (best alpha, rmse per alpha)."""
    errs = []
    for al in grid:
        pred = np.zeros(len(y))
        for g in np.unique(folds):
            m = folds == g
            pred[m] = CovStacker(kind, al, q).fit(P[~m], y[~m], Z[~m]).predict(P[m], Z[m])
        errs.append(rmse(pred, y))
    return grid[int(np.argmin(errs))], errs


def tercile_rmse(pred, y, z):
    idx = np.digitize(z, np.quantile(z, [1 / 3, 2 / 3]))
    return [rmse(pred[idx == k], y[idx == k]) for k in range(3)]


def run_cov(a, tr, te, exps, P, T, y):
    covs = a.cov.split(",")
    Z, Zt = load_covs(tr.ID, covs), load_covs(te.ID, covs)
    folds = tr.fold.values
    q = tuple(float(v) for v in a.q.split(","))
    grid = [float(v) for v in a.alphas.split(",")] if a.alpha is None else [float(a.alpha)]
    # reference: plain NNLS over the same members, same nested protocol as the default path
    base = np.zeros(len(y))
    for f in range(5):
        m = folds == f
        base[m] = P[m] @ fit_w(P[~m], y[~m])
    w_g = fit_w(P, y)
    # covariate stacker: everything (quantiles, edges, alpha by inner CV) refit inside each outer fold
    nested, chosen = np.zeros(len(y)), []
    for f in range(5):
        m = folds == f
        al = select_alpha(P[~m], y[~m], Z[~m], folds[~m], a.stacker, q, grid)[0] if len(grid) > 1 else grid[0]
        chosen.append(al)
        nested[m] = CovStacker(a.stacker, al, q).fit(P[~m], y[~m], Z[~m]).predict(P[m], Z[m])
    al_final, cv_errs = select_alpha(P, y, Z, folds, a.stacker, q, grid) if len(grid) > 1 else (grid[0], [])
    st = CovStacker(a.stacker, al_final, q).fit(P, y, Z)
    ins, test_pred = st.predict(P, Z), st.predict(T, Zt)

    fr = lambda p: [rmse(p[folds == f], y[folds == f]) for f in range(5)]  # noqa: E731
    fb, fs = fr(base), fr(nested)
    d = np.array(fs) - np.array(fb)
    nb = int((d < -1e-9).sum())  # strictly better folds (alpha=inf folds tie exactly up to round-off)
    rep = {"nnls": (rmse(base, y), rmse(P @ w_g, y)), "stacker": (rmse(nested, y), rmse(ins, y))}
    print(f"\n== {a.stacker} stacker on {covs}, q={q}, alpha grid={grid}; {len(exps)} members ==")
    print(f"{'':10s} {'nested':>8s} {'in-samp':>8s} {'gap':>6s}  fold RMSE")
    for k, p in [("nnls", fb), ("stacker", fs)]:
        nv, iv = rep[k]
        print(f"{k:10s} {nv:8.4f} {iv:8.4f} {nv - iv:6.3f}  {np.round(p, 3).tolist()}")
    print(f"delta nested {rep['stacker'][0] - rep['nnls'][0]:+.4f}; per fold {np.round(d, 3).tolist()}; "
          f"folds improved {nb}/5")
    Zr = load_covs(tr.ID, ["snr", "n"])
    for lab, z, names in [("snr", Zr[:, 0], "noisy/mid/clean"), ("N", Zr[:, 1], "coarse/mid/fine")]:
        print(f"{lab} tercile RMSE ({names}): nnls {np.round(tercile_rmse(base, y, z), 2).tolist()}"
              f"  stacker {np.round(tercile_rmse(nested, y, z), 2).tolist()}")
    print(f"alpha per outer fold {chosen}; final alpha {al_final} (5-fold CV rmse {np.round(cv_errs, 4).tolist()})")
    print("global nnls weights", dict(zip(exps, np.round(w_g, 3))))
    labels = (["noisy", "mid", "clean"] if a.stacker == "tercile" else
              ["".join(f"{c}{'lo' if b == 0 else 'hi'}" for c, b in zip(covs, bits))
               for bits in itertools.product([0, 1], repeat=len(covs))])
    cw = {lab: np.round(w, 3).tolist() for lab, w in zip(labels, st.corner_weights())}
    for lab, w in cw.items():
        print(f"weights @{lab:12s} {w}")
    if a.dry_run:
        return
    nv = rep["stacker"][0]
    SUB_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"ID": te.ID, "hardness": test_pred}).to_csv(SUB_DIR / f"{a.out}.csv", index=False)
    meta = {"exps": exps, "stacker": a.stacker, "covs": covs, "cov_columns": [COV_SRC[c] for c in covs],
            "q": q, "alpha": str(al_final), "alpha_per_outer_fold": [str(v) for v in chosen],
            "edges_or_quantiles": (st.edges.tolist() if a.stacker == "tercile" else
                                   {"lo": st.lo.tolist(), "hi": st.hi.tolist()}),
            "corner_weights": cw, "nnls_weights": w_g.tolist(), "nested_cv_rmse": nv,
            "nested_cv_rmse_nnls_same_members": rep["nnls"][0], "in_sample_rmse": rep["stacker"][1],
            "fold_rmse": fs, "fold_rmse_nnls": fb}
    (SUB_DIR / f"{a.out}.json").write_text(json.dumps(meta, indent=2))
    with open(EXP_DIR / "LEADERBOARD.md", "a") as fh:
        fh.write(f"| {a.out} (blend) | {nv:.3f} |  | nested-CV, {len(exps)} models, {a.stacker} covariate stacker "
                 f"on log {'+'.join(covs)} (alpha by inner CV); plain NNLS same members {rep['nnls'][0]:.3f}, "
                 f"{nb}/5 folds better |\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="*")
    ap.add_argument("--out", default="blend")
    ap.add_argument("--stacker", default="nnls", choices=["nnls", "linear", "tercile"])
    ap.add_argument("--cov", default="snr", help="covariate stacker: comma list from " + ",".join(COV_SRC))
    ap.add_argument("--q", default="0.1,0.9", help="linear stacker: train quantiles mapped to t=0 and t=1")
    ap.add_argument("--alphas", default="0,0.1,0.3,1,3,10,inf", help="shrinkage grid for inner CV")
    ap.add_argument("--alpha", type=float, default=None, help="fixed shrinkage (skips inner CV)")
    ap.add_argument("--dry-run", action="store_true", help="covariate stacker: report only, write nothing")
    a = ap.parse_args()
    tr, te = load_train(), load_test()
    exps = a.exps or sorted(p.parent.name for p in EXP_DIR.glob("*/score.json"))
    P = np.column_stack([tr[["ID"]].merge(pd.read_csv(EXP_DIR / e / "oof.csv"), on="ID").hardness for e in exps])
    T = np.column_stack([te[["ID"]].merge(pd.read_csv(EXP_DIR / e / "test.csv"), on="ID").hardness for e in exps])
    y = tr.hardness.values
    for e, col in zip(exps, P.T):
        print(f"{e:50s} {rmse(col, y):.4f}")
    if a.stacker != "nnls":
        run_cov(a, tr, te, exps, P, T, y)
        raise SystemExit
    # honest estimate: weights fit on 4 folds, applied to the 5th
    nested = np.zeros(len(y))
    for f in range(5):
        m = (tr.fold == f).values
        nested[m] = P[m] @ fit_w(P[~m], y[~m])
    w = fit_w(P, y)
    print("weights", dict(zip(exps, np.round(w, 3))))
    print(f"blend in-sample {rmse(P @ w, y):.4f}  nested-CV {rmse(nested, y):.4f}")
    SUB_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"ID": te.ID, "hardness": T @ w}).to_csv(SUB_DIR / f"{a.out}.csv", index=False)
    (SUB_DIR / f"{a.out}.json").write_text(json.dumps({"exps": exps, "weights": w.tolist(),
                                                       "nested_cv_rmse": rmse(nested, y)}, indent=2))
    with open(EXP_DIR / "LEADERBOARD.md", "a") as fh:
        fh.write(f"| {a.out} (blend) | {rmse(nested, y):.3f} |  | nested-CV, {len(exps)} models |\n")
