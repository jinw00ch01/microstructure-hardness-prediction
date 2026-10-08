"""Pre-registered refinements of the blend_v19 clean-image heterogeneity offset, judged by nested selection.

Base: blend_v18 nested OOF (train) / blend_v18 (test). Gated rows = the IDs of data/het_blocks_{train,test}.parquet
(raw ic_noise < 9.5: 267 train / 519 test). Reference = v19's offset: OLS of y - v18_oof on [1, het4, N_eff^0.25].

Each gated image is segmented ONCE (src.het_blocks.segment + the pore rule of src.het_blocks.one); every column below is
computed from that single image only (feature extraction; nothing is fitted across images, test images included).
Variant features, fixed before any result was seen (a = visible pixel area of the watershed grain of the pixel,
pores excluded, blocks aligned to the image origin):
  het4_pXXX  F1  sd over the 4x4 grid (64 px, blocks >= 50 non-pore px) of b_k = -(1/p) log(mean a^-p),
                 p in {0.25, 0.5, 0.75, 1.0}; p = 0.5 is v19's het4 (asserted bit-identical to het_blocks_*.parquet)
  het8       F2  as het4 (p = 0.5) on an 8x8 grid of 32-px blocks, blocks with >= 20 non-pore px
  het2       F3  as het4 (p = 0.5) on a 2x2 grid of 128-px blocks, blocks with >= 50 non-pore px
  (F4 = het4 * log N_eff, derived)
  het4_mad   F5  MAD * 1.4826 instead of sd over the 16 het4 block values (p = 0.5)
  (F6 = N form N_eff^0.25 / log N_eff / N_eff^0.5, derived)
  het4_gm    F7  grain-mean: b_k = -2 log(mean over the grains present among the block's non-pore pixels of a_g^-1/2),
                 each grain weighted equally, border grains included; same blocks as het4 (>= 50 non-pore px)
Designs (all with an intercept): see VARIANTS.

Decision (pre-registered): for each outer fold k, every variant (v19 included) is scored by 4-fold CV (the shared
folds) on the gated rows of the other four outer folds; the best is refit on those rows and applied to fold k.
Adopt only if the nested-selection gated RMSE <= 9.617 (v19 9.717 - 0.10), >= 4/5 outer folds better than v19's
cross-fitted offset, and the same non-v19 variant is picked in >= 3/5 outer folds (that variant, refit on all 267
gated train rows, would go to test).

  python -m src.het_refine extract      # -> data/het_refine_{train,test}.parquet
  python -m src.het_refine evaluate     # -> /mnt/project-files/work/hardness-cache/scripts/het-1008/refine/
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

from .common import DATA_DIR, SUB_DIR, load_test, load_train, rmse
from .het_blocks import grain_table, segment, shading_norm

OUT_DIR = Path("/mnt/project-files/work/hardness-cache/scripts/het-1008/refine")
P_LIST = (0.25, 0.5, 0.75, 1.0)
BAR_RMSE, BAR_FOLDS, BAR_PICKS = 9.617, 4, 3


def grid(size):
    n = 256 // size
    return np.add.outer(np.arange(256) // size * n, np.arange(256) // size), n * n


G4, G8, G2 = grid(64), grid(32), grid(128)


def block_vals(a, m, g, minpx, p=0.5):
    gid, nb = g
    # same expression as src.het_blocks.one, so p = 0.5 on the 4x4 grid is bit-identical
    return [-(1 / p) * np.log((a[s] ** -p).mean()) for b in range(nb) if (s := (gid == b) & m).sum() >= minpx]


def feats(path):
    im = np.array(Image.open(path)).astype(np.float32)
    sm, ws = segment(im)
    g = grain_table(sm, ws)
    pore = ((shading_norm(g) < 0.68) & (g.area < 600)).values
    gp = g[~pore]
    n_eff = (~gp.border).sum() + 0.5 * gp.border.sum()
    is_pore = np.zeros(ws.max() + 1, bool)
    is_pore[g.label.values[pore]] = True
    area = np.maximum(np.bincount(ws.ravel()).astype(float), 1)
    a = area[ws]
    m = ~is_pore[ws]
    out = {"N_eff": float(n_eff)}
    for p in P_LIST:
        out[f"het4_p{int(round(p * 100)):03d}"] = float(np.std(block_vals(a, m, G4, 50, p)))
    v4 = np.array(block_vals(a, m, G4, 50, 0.5))
    out["het4"] = float(np.std(v4))
    out["het8"] = float(np.std(block_vals(a, m, G8, 20, 0.5)))
    out["het2"] = float(np.std(block_vals(a, m, G2, 50, 0.5)))
    out["het4_mad"] = float(np.median(np.abs(v4 - np.median(v4))) * 1.4826)
    gm = []
    for b in range(16):
        s = (G4[0] == b) & m
        if s.sum() >= 50:
            gm.append(-2 * np.log((area[np.unique(ws[s])] ** -0.5).mean()))
    out["het4_gm"] = float(np.std(gm))
    out["n_blk4"], out["n_blk8"] = len(v4), len(block_vals(a, m, G8, 20, 0.5))
    return out


def extract(n_jobs):
    for split in ("train", "test"):
        ref = pd.read_parquet(DATA_DIR / f"het_blocks_{split}.parquet")
        res = Parallel(n_jobs=n_jobs)(delayed(feats)(DATA_DIR / split / f"{i}.png") for i in ref.ID)
        out = pd.DataFrame(res)
        out.insert(0, "ID", ref.ID.values)
        assert np.array_equal(out.het4.values, ref.het4.values), "het4 (p=0.5) must reproduce het_blocks exactly"
        assert np.array_equal(out.het4_p050.values, ref.het4.values)
        assert np.array_equal(out.N_eff.values, ref.N_eff.values)
        out.to_parquet(DATA_DIR / f"het_refine_{split}.parquet", index=False)
        print(split, out.shape, out.drop(columns="ID").describe().loc[["mean", "std", "min", "max"]].round(3).to_string())


# name -> columns after the intercept (pre-registered; v19 first so ties keep v19)
VARIANTS = {
    "v19": lambda F: [F.het4, F.N_eff ** 0.25],
    "F1_p025": lambda F: [F.het4_p025, F.N_eff ** 0.25],
    "F1_p075": lambda F: [F.het4_p075, F.N_eff ** 0.25],
    "F1_p100": lambda F: [F.het4_p100, F.N_eff ** 0.25],
    "F2_het8": lambda F: [F.het4, F.N_eff ** 0.25, F.het8],
    "F3_het2": lambda F: [F.het4, F.N_eff ** 0.25, F.het2],
    "F4_het4logN": lambda F: [F.het4, F.N_eff ** 0.25, F.het4 * np.log(F.N_eff)],
    "F5_mad": lambda F: [F.het4_mad, F.N_eff ** 0.25],
    "F6_logN": lambda F: [F.het4, np.log(F.N_eff)],
    "F6_sqrtN": lambda F: [F.het4, F.N_eff ** 0.5],
    "F7_gm": lambda F: [F.het4_gm, F.N_eff ** 0.25],
}


def design(F, name):
    cols = VARIANTS[name](F)
    return np.column_stack([np.ones(len(F))] + [np.asarray(c, float) for c in cols])


def ols(X, r):
    return np.linalg.lstsq(X, r, rcond=None)[0]


def crossfit(X, r, fold, rows):
    """Offset for `rows`, each fold predicted by OLS on the other folds of `rows`."""
    off = np.full(len(r), np.nan)
    for f in np.unique(fold[rows]):
        fit, app = rows & (fold != f), rows & (fold == f)
        off[app] = X[app] @ ols(X[fit], r[fit])
    return off


def evaluate():
    tr, te = load_train(), load_test()
    y, folds = tr.hardness.values, tr.fold.values
    oof = tr[["ID"]].merge(pd.read_csv(SUB_DIR / "blend_v18_oof.csv"), on="ID").hardness.values
    base = te[["ID"]].merge(pd.read_csv(SUB_DIR / "blend_v18.csv"), on="ID").hardness.values
    Ftr = pd.read_parquet(DATA_DIR / "het_refine_train.parquet")
    Fte = pd.read_parquet(DATA_DIR / "het_refine_test.parquet")
    g_tr, g_te = tr.ID.isin(Ftr.ID).values, te.ID.isin(Fte.ID).values
    assert g_tr.sum() == 267 and g_te.sum() == 519
    Fg = Ftr.set_index("ID").loc[tr.ID[g_tr]]
    Ft = Fte.set_index("ID").loc[te.ID[g_te]]
    yg, og, fg = y[g_tr], oof[g_tr], folds[g_tr]
    rg = yg - og
    allrows = np.ones(len(yg), bool)
    X = {v: design(Fg, v) for v in VARIANTS}
    e_out = (y - oof)[~g_tr]   # non-gated rows keep v18

    def all500(off_g):
        return float(np.sqrt((np.sum((rg - off_g) ** 2) + np.sum(e_out ** 2)) / len(y)))

    def per_fold(off_g):
        return np.array([rmse((rg - off_g)[fg == k], 0) for k in range(5)])

    terc = np.quantile(Fg.N_eff.values, [1 / 3, 2 / 3])
    tbin = np.digitize(Fg.N_eff.values, terc)   # 0 = coarse (fewest grains)

    # informational: each variant cross-fitted on the 5 shared folds
    res = {}
    for v in VARIANTS:
        off = crossfit(X[v], rg, fg, allrows)
        res[v] = dict(gated=rmse(rg - off, 0), all500=all500(off), folds=per_fold(off),
                      terciles=[rmse((rg - off)[tbin == t], 0) for t in range(3)], coef_all=ols(X[v], rg).tolist())
    ref = res["v19"]
    print(f"v18 gated {rmse(rg, 0):.4f} all500 {rmse(y - oof, 0):.4f}")
    print(f"v19 reproduced: gated {ref['gated']:.4f} all500 {ref['all500']:.4f}")
    assert abs(ref["gated"] - 9.717) < 5e-4 and abs(ref["all500"] - 11.754) < 5e-4
    for v, d in res.items():
        d["folds_better"] = int((d["folds"] < ref["folds"]).sum())
        print(f"{v:12s} gated {d['gated']:.4f} ({d['gated'] - ref['gated']:+.4f}) all500 {d['all500']:.4f} "
              f"folds better {d['folds_better']}/5 terciles {np.round(d['terciles'], 2).tolist()} "
              f"coef {np.round(d['coef_all'], 2).tolist()}")

    # decision: nested selection
    sel_off, picks, inner_tab = np.full(len(yg), np.nan), [], {}
    for k in range(5):
        inner = fg != k
        sc = {v: rmse((rg - crossfit(X[v], rg, fg, inner))[inner], 0) for v in VARIANTS}
        best = min(sc, key=sc.get)   # dict order: v19 first, strict min keeps v19 on ties
        picks.append(best)
        inner_tab[k] = sc
        app = fg == k
        sel_off[app] = X[best][app] @ ols(X[best][inner], rg[inner])
        print(f"outer {k}: pick {best:12s} inner {sc[best]:.4f} (v19 {sc['v19']:.4f})")
    sel = dict(gated=rmse(rg - sel_off, 0), all500=all500(sel_off), folds=per_fold(sel_off),
               terciles=[rmse((rg - sel_off)[tbin == t], 0) for t in range(3)])
    sel["folds_better"] = int((sel["folds"] < ref["folds"]).sum())
    counts = pd.Series(picks).value_counts()
    top, top_n = counts.index[0], int(counts.iloc[0])
    adopt = bool(sel["gated"] <= BAR_RMSE and sel["folds_better"] >= BAR_FOLDS and top != "v19" and top_n >= BAR_PICKS)
    print(f"nested selection: gated {sel['gated']:.4f} (v19 {ref['gated']:.4f}, {sel['gated'] - ref['gated']:+.4f}) "
          f"all500 {sel['all500']:.4f} folds {np.round(sel['folds'], 3).tolist()} vs v19 "
          f"{np.round(ref['folds'], 3).tolist()} better {sel['folds_better']}/5 picks {picks} -> ADOPT {adopt}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "base": "blend_v18", "gate_train": int(g_tr.sum()), "gate_test": int(g_te.sum()),
        "v18_gated": rmse(rg, 0), "v18_all500": rmse(y - oof, 0), "neff_tercile_cuts": terc.tolist(),
        "variants": {v: {**d, "folds": d["folds"].tolist()} for v, d in res.items()},
        "nested_selection": {**sel, "folds": sel["folds"].tolist(), "picks": picks,
                             "inner_rmse": {str(k): s for k, s in inner_tab.items()}},
        "bar": {"gated_rmse_max": BAR_RMSE, "folds_better_min": BAR_FOLDS, "same_variant_picks_min": BAR_PICKS},
        "most_picked": top, "most_picked_n": top_n, "adopt": adopt,
    }
    if adopt:
        Xt = design(Ft, top)
        coef = ols(X[top], rg)
        off_t = Xt @ coef
        pd.DataFrame({"ID": te.ID[g_te].values, "base_v18": base[g_te], "offset": off_t,
                      "hardness": base[g_te] + off_t}).to_csv(OUT_DIR / f"offset_{top}_test.csv", index=False)
        summary["adopted"] = {"variant": top, "coef": coef.tolist(), "test_offset_sd": float(off_t.std()),
                              "test_offset_mean": float(off_t.mean())}
        print("adopted", top, np.round(coef, 3).tolist(), "test offset sd", round(float(off_t.std()), 3))
    (OUT_DIR / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["extract", "evaluate"])
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args()
    extract(a.n_jobs) if a.step == "extract" else evaluate()


if __name__ == "__main__":
    main()
