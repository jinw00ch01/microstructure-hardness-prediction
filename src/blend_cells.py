"""Cell-dependent blends (orchestrator; reproduces blend_v14, blend_v15 and blend_v16).

Cell "fine_noisy": log(cal_seg_count_density) > train median AND raw ic_noise > train median (no labels, no test
statistics). The CNN mix is a fixed weighted mean of CNN experiments (--cnn name:w,...).
--mode share (v14, v15): stage 1 = NNLS over a member pool (src.ensemble.fit_w; the pool of an earlier blend json),
  stage 2 = the CNN mix at share s_cell in the cell and s_other elsewhere (fixed by the caller).
--mode nnls (v16): separate NNLS weights in the cell and outside it, over --members plus the CNN mix as one member.
--mode nnls --other share (v17): the cell as in nnls, every other image as in share mode (pool NNLS + CNN mix at
  --share-other), i.e. v16 inside the cell and v14 outside it.
Nested CV refits every fitted weight per outer fold on the other four folds.

  C=cnn_ev2s_rawnlm_degcons_gpu_s6:0.7,cnn_cnxt_rawnlm_degcons_gpu_s3:0.3
  python -m src.blend_cells --cnn $C --share-other 0.2 --share-cell 0.2 --out blend_v14
  python -m src.blend_cells --cnn $C --share-other 0.2 --share-cell 0.7 --out blend_v15
  python -m src.blend_cells --cnn $C --mode nnls --members feat6_rest_ridge_all,feat5_v3cal_ridge_het_spat_v4_milspl,\
feat3_v23cal_lgbs_het_spat,feat2_v23cal_lgbs_hetN,emb_effv2s_256_gridge3_noise_cs24 --out blend_v16
  python -m src.blend_cells --cnn $C --mode nnls --other share --share-other 0.2 --members <the v16 members> --out blend_v17
  python -m src.blend_cells --cnn $C --cnn-nnls cnn_ev2s_rawnlm_degcons_gpu_s6:1 --mode nnls --other share --share-other 0.2 \
--members <the v16 members> --out blend_v18      # v17 with effnetv2-s alone as the cell's CNN member
  python -m src.blend_cells --cnn $C --cnn-nnls cnn_ev2s_rawnlm_degcons_gpu_s6:1 --mode nnls --other share --share-other 0.2 \
--members <the v16 members> --cell-share 0.8 --out blend_v20   # cell: members-only NNLS + effnetv2-s at a fixed 0.8
"""
import argparse
import json

import numpy as np
import pandas as pd

from .common import DATA_DIR, EXP_DIR, SUB_DIR, load_test, load_train, rmse
from .ensemble import fit_w


def cell_masks(tr_ids, te_ids, cell="fine_noisy"):
    assert cell == "fine_noisy", cell
    cal = pd.read_parquet(DATA_DIR / "features_cal.parquet").set_index("ID")
    v3 = pd.read_parquet(DATA_DIR / "features_v3.parquet").set_index("ID")
    g = lambda ids: np.log(cal.loc[ids, "cal_seg_count_density"].values)
    n = lambda ids: v3.loc[ids, "ic_noise"].values
    mg, mn = float(np.median(g(tr_ids))), float(np.median(n(tr_ids)))
    rule = {"log_cal_seg_count_density_gt": mg, "ic_noise_gt": mn}
    return (g(tr_ids) > mg) & (n(tr_ids) > mn), (g(te_ids) > mg) & (n(te_ids) > mn), rule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-from", default="blend_v7")
    ap.add_argument("--cnn", required=True, help="name:weight,... (weights of the fixed CNN mix)")
    ap.add_argument("--share-other", type=float, default=0.2)
    ap.add_argument("--share-cell", type=float, default=0.2)
    ap.add_argument("--cell", default="fine_noisy")
    ap.add_argument("--mode", default="share", choices=["share", "nnls"])
    ap.add_argument("--members", default="", help="--mode nnls: comma list of members (the CNN mix is added)")
    ap.add_argument("--other", default="nnls", choices=["nnls", "share"], help="--mode nnls: images outside the cell")
    ap.add_argument("--cnn-nnls", default=None, help="--mode nnls: CNN mix used as the NNLS member (default: --cnn)")
    ap.add_argument("--cell-share", type=float, default=None,
                    help="--mode nnls: in the cell, NNLS over --members only, then the --cnn-nnls mix at this fixed share")
    ap.add_argument("--out", default=None)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    tr, te = load_train(), load_test()
    y, folds = tr.hardness.values, tr.fold.values
    cnn = [(s.split(":")[0], float(s.split(":")[1])) for s in a.cnn.split(",")]
    ld = lambda e, k: (tr if k == "oof" else te)[["ID"]].merge(pd.read_csv(EXP_DIR / e / f"{k}.csv"), on="ID").hardness.values
    c = sum(w * ld(e, "oof") for e, w in cnn)
    ct = sum(w * ld(e, "test") for e, w in cnn)
    cnn_n = [(s.split(":")[0], float(s.split(":")[1])) for s in a.cnn_nnls.split(",")] if a.cnn_nnls else cnn
    in_tr, in_te, rule = cell_masks(tr.ID, te.ID, a.cell)
    nested = np.zeros(len(y))
    if a.mode == "share" or a.other == "share":
        sh_pool = [e for e in json.load(open(SUB_DIR / f"{a.pool_from}.json"))["exps"] if e not in dict(cnn)]
        P = np.column_stack([ld(e, "oof") for e in sh_pool])
        T = np.column_stack([ld(e, "test") for e in sh_pool])
        for f in range(5):
            m = folds == f
            b = P[m] @ fit_w(P[~m], y[~m])
            nested[m] = b + np.where(in_tr[m], a.share_cell, a.share_other) * (c[m] - b)
        w = fit_w(P, y)
        bt = T @ w
        pred = bt + np.where(in_te, a.share_cell, a.share_other) * (ct - bt)
        weights = {"base_weights": w.tolist(), "cnn_share": {"cell": a.share_cell, "other": a.share_other}}
        pool = sh_pool
    if a.mode == "nnls":
        cells = (True, False) if a.other == "nnls" else (True,)
        pool_n = [e for e in a.members.split(",") if e]
        P = np.column_stack([ld(e, "oof") for e in pool_n] + [sum(w * ld(e, "oof") for e, w in cnn_n)])
        T = np.column_stack([ld(e, "test") for e in pool_n] + [sum(w * ld(e, "test") for e, w in cnn_n)])
        cs = a.cell_share
        for f in range(5):
            m = folds == f
            for cell in cells:
                fit_rows, app = (~m) & (in_tr == cell), m & (in_tr == cell)
                if cell and cs is not None:
                    b = P[app, :-1] @ fit_w(P[fit_rows, :-1], y[fit_rows])
                    nested[app] = b + cs * (P[app, -1] - b)
                else:
                    nested[app] = P[app] @ fit_w(P[fit_rows], y[fit_rows])
        if cs is None:
            w_cell = fit_w(P[in_tr], y[in_tr])
            cell_te = T @ w_cell
        else:
            w_cell = np.append(fit_w(P[in_tr, :-1], y[in_tr]) * (1 - cs), cs)   # equivalent fixed weights
            cell_te = T @ w_cell
        if a.other == "nnls":
            w_other = fit_w(P[~in_tr], y[~in_tr])
            pred = np.where(in_te, cell_te, T @ w_other)
            weights = {"weights_cell": dict(zip(pool_n + ["cnn_mix"], w_cell.tolist())),
                       "weights_other": dict(zip(pool_n + ["cnn_mix"], w_other.tolist()))}
            pool = pool_n
        else:
            pred = np.where(in_te, cell_te, pred)
            weights = {"weights_cell": dict(zip(pool_n + ["cnn_mix"], w_cell.tolist())),
                       "other_share_mode": weights}
            pool = pool_n + [e for e in pool if e not in pool_n]
    fr = [rmse(nested[folds == f], y[folds == f]) for f in range(5)]
    print(f"nested CV {rmse(nested, y):.4f} folds {np.round(fr, 3).tolist()} | cell train {in_tr.sum()} test {in_te.sum()}")
    if a.out:
        sub = pd.DataFrame({"ID": te.ID, "hardness": pred})
        assert len(sub) == 1000 and np.isfinite(sub.hardness).all()
        sub.to_csv(SUB_DIR / f"{a.out}.csv", index=False)
        pd.DataFrame({"ID": tr.ID, "hardness": nested}).to_csv(SUB_DIR / f"{a.out}_oof.csv", index=False)  # git-ignored
        rule.update(train_n=int(in_tr.sum()), test_n=int(in_te.sum()))
        (SUB_DIR / f"{a.out}.json").write_text(json.dumps({
            "exps": pool + list(dict.fromkeys(e for e, _ in cnn + cnn_n)),
            "mode": a.mode + ("+other_share" if a.other == "share" else ""),
            **weights, "cnn_mix": dict(cnn), **({"cnn_mix_nnls": dict(cnn_n)} if a.cnn_nnls else {}),
            **({"cell_share": cs} if a.mode == "nnls" and cs is not None else {}), "cell": a.cell,
            "cell_rule": rule, "nested_cv_rmse": rmse(nested, y), "fold_rmse": fr, "notes": a.note}, indent=2))
        print("written", SUB_DIR / f"{a.out}.csv")


if __name__ == "__main__":
    main()
