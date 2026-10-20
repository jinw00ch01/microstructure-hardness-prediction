"""Copy U1 (bold/impl_C4) into submissions/ as blend_v29g, after format and identity checks.

Run from the repo root:  python /mnt/project-files/work/hardness-cache/scripts/d1011/bold/plan_bold/build_v29g.py
- Refuses to overwrite any existing blend_v29g* file.
- Copies bytes exactly (md5 preserved). Writes blend_v29g.json (md5s, nested numbers, what changed, reading lines).
- Train labels are read only to recompute the nested RMSE; nothing label-derived is written except aggregate RMSEs.
"""
import hashlib, json, os, shutil, sys
import numpy as np, pandas as pd

HC = "/mnt/project-files/work/hardness-cache"
SUB = f"{HC}/submissions"
BOLD = f"{HC}/scripts/d1011/bold"
SRC_TEST = f"{BOLD}/impl_C4/test.csv"
SRC_OOF = f"{BOLD}/impl_C4/oof.csv"
REB_TEST = f"{BOLD}/verify_C4/rebuild/test.csv"
REB_OOF = f"{BOLD}/verify_C4/rebuild/oof.csv"
BASE = f"{SUB}/blend_v28n.csv"
BASE_OOF = f"{SUB}/blend_v28n_oof.csv"
DST = f"{SUB}/blend_v29g.csv"
DST_OOF = f"{SUB}/blend_v29g_oof.csv"
DST_JSON = f"{SUB}/blend_v29g.json"
EXP = dict(test="496f5cc978c2e4298775e992455dbeac", oof="f0b889eb9ba7a7de6ea11c868e08e2c6",
           base="396a52f2263c4ec857b8dbee1316a3ae", base_oof="a7b357b872a80f780804c30df432fd07")


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    for p in (DST, DST_OOF, DST_JSON):
        if os.path.exists(p):
            sys.exit(f"REFUSE: {p} exists; not overwriting")
    m = dict(test=md5(SRC_TEST), oof=md5(SRC_OOF), base=md5(BASE), base_oof=md5(BASE_OOF),
             rebuild_test=md5(REB_TEST), rebuild_oof=md5(REB_OOF))
    assert m["test"] == EXP["test"] and m["rebuild_test"] == EXP["test"], m
    assert m["oof"] == EXP["oof"] and m["rebuild_oof"] == EXP["oof"], m
    assert m["base"] == EXP["base"] and m["base_oof"] == EXP["base_oof"], m

    ss = pd.read_csv("data/sample_submission.csv")
    t = pd.read_csv(SRC_TEST, float_precision="round_trip")
    b = pd.read_csv(BASE, float_precision="round_trip")
    assert list(t.columns) == ["ID", "hardness"], t.columns
    assert len(t) == 1000 and (t.ID.values == ss.ID.values).all()
    assert np.isfinite(t.hardness.values).all()
    assert (b.ID.values == t.ID.values).all()
    chg = t.hardness.values != b.hardness.values
    n_chg = int(chg.sum())
    assert n_chg == 580, n_chg
    # unchanged rows identical as text too
    tl = open(SRC_TEST).read().splitlines()
    bl = open(BASE).read().splitlines()
    assert tl[0] == bl[0] and len(tl) == len(bl) == 1001
    txt_diff = [i for i in range(1000) if tl[i + 1] != bl[i + 1]]
    assert set(txt_diff) == set(np.where(chg)[0].tolist()), "text diff set != value diff set"
    d = t.hardness.values - b.hardness.values

    tr = pd.read_csv("data/train.csv")
    o = pd.read_csv(SRC_OOF, float_precision="round_trip")
    ob = pd.read_csv(BASE_OOF, float_precision="round_trip")
    assert len(o) == 500 and (o.ID.values == tr.ID.values).all() and (ob.ID.values == tr.ID.values).all()
    assert np.isfinite(o.hardness.values).all()
    y = tr.hardness.values
    rm = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))
    r_new, r_base = rm(o.hardness.values), rm(ob.hardness.values)
    folds = pd.read_csv("data/folds.csv")
    fmap = dict(zip(folds.ID, folds.fold))
    fk = np.array([fmap[i] for i in tr.ID])
    fd = []
    for k in sorted(set(fk)):
        s = fk == k
        fd.append(round(float(np.sqrt(np.mean((o.hardness.values[s] - y[s]) ** 2))
                              - np.sqrt(np.mean((ob.hardness.values[s] - y[s]) ** 2))), 4))
    assert abs(r_new - 10.617579795116168) < 1e-9 and abs(r_base - 10.991311049864239) < 1e-9, (r_new, r_base)

    shutil.copyfile(SRC_TEST, DST)
    shutil.copyfile(SRC_OOF, DST_OOF)
    assert md5(DST) == EXP["test"] and md5(DST_OOF) == EXP["oof"]

    info = {
        "name": "blend_v29g",
        "alias": "U1 (d1011 bold C4)",
        "base": "blend_v28n",
        "base_md5": EXP["base"],
        "md5": EXP["test"],
        "oof_md5": EXP["oof"],
        "source": SRC_TEST,
        "builder": f"{BOLD}/impl_C4/build_u1.py (run from repo root: flock SLOT timeout 1500 python <builder> --nj 3)",
        "independent_rebuild": f"{BOLD}/verify_C4/rebuild/ (second agent; test and oof md5 identical)",
        "copied_by": f"{BOLD}/plan_bold/build_v29g.py",
        "rule": ("v28n with 580 test rows changed. Clean rows (raw features_v3 ic_noise < 9.5; 519): "
                 "0.5*(v21base + off_V4) + 0.5*f8_C2; off_V4 = OLS of (y - v21base OOF) on [1, V4, N_eff^0.25] "
                 "(full-train coefs [-91.86, 161.21, 15.61]); f8_C2 = log-y OLS f8 with het4 := V4 (6-map exact "
                 "4x4-grid het), fd_area := FD6t, logN := N_eff^0.25, predict exp(p + s2/2). LV rows (9.5 <= "
                 "ic_noise < 12 and ic_ridge_snr >= 0.5; 42): 0.5*v25 + 0.5*f7V4 (L1 V4_lv rule). T-CP rows "
                 "(midLF and m3b gate_cp; 19): 0.5*v28n + 0.5*f7 (blend_v26L coef_full, old het). Other 420 rows "
                 "bit-identical to v28n (values and text). All coefficients fitted on the 267 clean train rows only."),
        "rows_changed_test": {"total": n_chg, "clean": 519, "LV": 42, "TCP": 19},
        "test_change": {"rms_all_rows": float(np.sqrt(np.mean(d ** 2))), "max_abs": float(np.abs(d).max()),
                        "mean_changed": float(d[chg].mean()), "sd_changed": float(d[chg].std())},
        "nested_cv_all": r_new,
        "nested_cv_all_v28n": r_base,
        "nested_all500_dmse": r_new ** 2 - r_base ** 2,
        "nested_fold_drmse": fd,
        "nested_folds_better": int(sum(x < 0 for x in fd)),
        "nested_signflip_p": 0.0005,
        "nested_droptop3_all500_dmse": -5.24,
        "resplits_100": {"mean_dmse": -8.04, "P_ge0": 0.0},
        "by_group_rmse_v28n_to_v29g": {"clean": [9.101, 8.361, 267], "LV": [10.522, 9.496, 21],
                                       "midLF": [10.498, 10.341, 47], "HLV": [7.631, 7.631, 4],
                                       "G1hi": [13.775, 13.775, 161]},
        "by_grain_tercile_rmse": {"coarse": [13.248, 12.541], "mid": [10.638, 10.309], "fine": [8.586, 8.638]},
        "selection_honest": {"note": "verify_C4/nested_select.py: 31 options chosen per outer fold by inner CV",
                             "all500": 10.676, "dmse": -6.8, "keep_v28n_ever_picked": False},
        "qualify_M36": {"lane": "DANGER", "ratio": 2.34, "delta": -0.367, "sd_tr": 0.119, "sd_pub": 0.153,
                        "bias": 0.170, "strict_gate_pass": True},
        "expected_public": {"central": 9.82, "range": [9.60, 9.95], "v28n": 10.0712},
        "reading_lines": {"carried_le": 9.90, "refuted_ge": 10.24,
                          "prereg": f"{BOLD}/impl_C4/READING.txt"},
        "caveats": [
            "selection (V4, riders, W1 form) made on all clean labels: honest gain about -6.8 not -8.08 MSE",
            "folds 3 and 4 slightly worse (+0.016, +0.099 RMSE)",
            "riders and T-CP are below one reading's resolution; the reading tests the W1 core",
            "repo port of V4 / FD6t / m3b pipelines owed before the 10/18 code check",
            "C2 guard reads a test-prediction SD (label-free sanity gate); make it report-only in the port",
        ],
        "public_score": None,
    }
    with open(DST_JSON, "x") as f:
        json.dump(info, f, indent=1)
    print(json.dumps({k: info[k] for k in ("md5", "oof_md5", "nested_cv_all", "nested_cv_all_v28n",
                                           "nested_all500_dmse", "nested_fold_drmse", "rows_changed_test",
                                           "test_change")}, indent=1))
    print("json md5", md5(DST_JSON))


if __name__ == "__main__":
    main()
