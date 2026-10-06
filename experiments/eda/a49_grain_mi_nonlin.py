"""A49: label = mean over grains of g(x_i) with g nonlinear and multi-dimensional?  Per-image pooled bases of the a48
per-grain vectors x_i (12 features), number-weighted mean and area-weighted mean:
  SPL: 1-D cubic B-splines (5 quantile knots over train grains) of each continuous feature + dark flag, plus tensor
       products of quadratic splines (4 knots): size x grey, size x dark, grey x neighbour dark share
  RFF: 300 random Fourier features cos(W z + b) of the standardised x_i (150 at length scale 2.5, 150 at 5)
Scaling constants, knots and W come from train grains only (label-free).  Cross-fitted ridge probes of the blend_v5
nested residual against the feat5 splmean base (eda_cf, kind feat5s): clean-only (snr > 0.9) and SNR-gated on all
images; 20-shuffle permutation nulls.  Usage: python a49_grain_mi_nonlin.py CACHE_DIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer

from eda_cf import Reporter, base, probe, table

cache = Path(sys.argv[1])
t = table("blend_v5")
oof, inner = base(t, kind="feat5s")
R = Reporter(t, oof)
c = R.clean
g = np.clip((t.ic_ridge_snr.values - 0.7) / 0.4, 0, 1)
G = pd.read_parquet(cache / "g3_train.parquet")
CONT = ["la", "asp", "rel_ang", "grey", "gstd", "bc", "nb_n", "nb_la", "nb_dk", "d_pore", "zone"]
X = G[CONT].astype(float)
X = X.fillna(X.median())
mu, sd = X.mean(), X.std()
Z = ((X - mu) / sd).values
dark = G.dark.values[:, None]


def splines(col, k, deg):
    v = X[[col]].values
    return SplineTransformer(n_knots=k, degree=deg, knots="quantile", extrapolation="constant", include_bias=False).fit(v).transform(v)


S1 = np.hstack([splines(cn, 5, 3) for cn in CONT] + [dark])
la2, gr2, nd2 = splines("la", 4, 2), splines("grey", 4, 2), splines("nb_dk", 4, 2)
T = np.hstack([np.einsum("ij,ik->ijk", la2, gr2).reshape(len(G), -1), la2 * dark, la2 * (1 - dark),
               np.einsum("ij,ik->ijk", gr2, nd2).reshape(len(G), -1)])
SPL = np.hstack([S1, T])
rng = np.random.default_rng(0)
Zf = np.hstack([Z, dark * 2 - 1])
W = np.vstack([rng.normal(0, 1 / 2.5, (150, Zf.shape[1])), rng.normal(0, 1 / 5.0, (150, Zf.shape[1]))])
b = rng.uniform(0, 2 * np.pi, 300)
RFF = np.sqrt(2 / 300) * np.cos(Zf @ W.T + b)
gid = pd.factorize(G.ID)[0]
img = pd.factorize(G.ID)[1]
A = G.area.values.astype(float)


def pool(M, wts):
    s = np.zeros((len(img), M.shape[1]))
    np.add.at(s, gid, M * wts[:, None])
    ws = np.bincount(gid, weights=wts)
    P = pd.DataFrame(s / ws[:, None])
    P.insert(0, "ID", img)
    return t[["ID"]].merge(P, on="ID", how="left").drop(columns="ID").values


bases = {f"{nm} {wn}": pool(M, wv) for nm, M in (("SPL", SPL), ("RFF", RFF)) for wn, wv in (("number", np.ones(len(G))), ("area", A))}
print(f"{len(G)} grains in {len(img)} images; SPL {SPL.shape[1]} cols, RFF {RFF.shape[1]} cols; clean n={c.sum()}")


def gated(M):
    med = np.nanmedian(M[g > 0], 0)
    M = np.where(np.isfinite(M), M, med)
    return np.column_stack([g[:, None] * M, g])


for nm, M in bases.items():
    for mode in ("clean", "gated"):
        Mu, kw = (M, {"fit_mask": c, "apply_mask": c}) if mode == "clean" else (gated(M), {})
        pr = probe(Mu, inner, **kw)
        nul = R.perm_null(Mu, inner, n=20, **kw)
        print(R.line(f"{mode}: {nm}", pr, Mu.shape[1]) +
              f" | null mean {nul.mean():+.3f} max {nul.max():+.3f} p~{(np.sum(nul >= R.gain(pr)) + 1) / 21:.2f}", flush=True)
