"""N6-G1 (notes/handoff/laptop-gpu.md section 17): label-free gate lines of a src.train_cnn --init-het run.

No label is read: train.csv is opened for its ID column only. Test images enter only through the run's own test
predictions, for the KS shift check (d), which is a stop rule and never fits anything.
G1 = raw features_v3 ic_noise >= 9.5 and ic_ridge_snr < 0.5 (src.het_cnn.g1_train_ids), train and test rows.

  python -W ignore tools/n6/gate17.py --name RUN [--folds 0 1] [--a3 ev2s_locK_real5] [--yhat]
    Reads data/cnn_cache/RUN/fold{f}_seed{s}.npz of --folds (every seed s cached for all of them, averaged per fold:
    y-hat, b-hat per block, logN-hat) and the A3 reader data/het_cnn/<a3>/het_cnn_train.parquet.
    (a) on the G1 train rows held out in --folds: corr(sd over the 16 blocks of b-hat, het_A3) >= 0.85 and
        corr(logN-hat, logN_A3) >= 0.97 -> PASS / FAIL. Report line: the same two correlations for the A3 fold
        models themselves (data/het_cnn/<a3>/fold{f}.npz, the 0-step values) when those files exist.
    (d) KS of y-hat, G1 test rows (mean of the --folds models and seeds) vs the G1 train rows held out in --folds,
        p > 0.01 -> PASS / FAIL (stop only).
    --yhat: a '# yhat' block with the y-hat of every held-out row of --folds (train.csv order, 2 decimals, n, sum)
        for the cloud's gates (b) and (c) (they need the cloud-only V4 het).
  python -W ignore tools/n6/gate17.py --stage3 NAME6 NAME3 NAMEFULL [--kf-chunk K [K ...]]
    experiments/NAME6, NAME3 (fold runs, oof.csv + test.csv) and NAMEFULL (the --full run, test.csv):
    corr(OOF NAME6, OOF NAME3) >= 0.98; rms(test NAMEFULL - test NAME6) <= 1.5 (incumbent s6 vs full6: 1.30);
    KS of y-hat, G1 test of kf = 0.5 * (test NAME6 + test NAMEFULL) vs G1 train OOF of NAME6, p > 0.01 (stop only).
    --kf-chunk K: print kf test chunk K (0-3, 250 rows, sample_submission order) in the GPU-B chunk format
    ('# NAME6kf test chunk K IDs first last n 250 sum S' + the values with 2 decimals).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
EXP_DIR = Path(os.environ.get("GATE17_EXP_DIR", ROOT / "experiments"))  # override: tests only
NOISE_MAX, G1_RIDGE_MAX = 9.5, 0.5  # src.het_cnn


def g1_ids():
    v3 = pd.read_parquet(DATA_DIR / "features_v3.parquet", columns=["ID", "ic_noise", "ic_ridge_snr"])
    m = (v3.ic_noise >= NOISE_MAX) & (v3.ic_ridge_snr < G1_RIDGE_MAX)
    return set(v3.ID[m])


def train_frame():
    """train.csv IDs (no label column) merged with data/folds.csv: the row order of src.train_cnn (load_train)."""
    return pd.read_csv(DATA_DIR / "train.csv", usecols=["ID"]).merge(pd.read_csv(DATA_DIR / "folds.csv"), on="ID")


def ks(a, b):
    from scipy.stats import ks_2samp
    r = ks_2samp(a, b)
    return float(r.statistic), float(r.pvalue)


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def run_gate(a):
    tr = train_frame()
    te_ids = pd.read_csv(DATA_DIR / "sample_submission.csv", usecols=["ID"]).ID.tolist()
    g1 = g1_ids()
    cache = DATA_DIR / "cnn_cache" / a.name
    seeds = []
    for s in range(64):
        if all((cache / f"fold{f}_seed{s}.npz").exists() for f in a.folds):
            seeds.append(s)
        elif seeds or s > 8:
            break
    if not seeds:
        raise SystemExit(f"{cache}: no seed cached for all of folds {a.folds}")
    n = len(tr)
    y, b, ln = np.full(n, np.nan), np.full((n, 16), np.nan), np.full(n, np.nan)
    yt = []
    for f in a.folds:
        zs = [dict(np.load(cache / f"fold{f}_seed{s}.npz")) for s in seeds]
        if any("val_b" not in z for z in zs):
            raise SystemExit(f"{cache}/fold{f}: no val_b in the cache (not an --init-het --head hetfpn run)")
        vi = zs[0]["val_idx"]
        assert all(np.array_equal(z["val_idx"], vi) for z in zs) and (tr.fold.values[vi] == f).all()
        y[vi] = np.mean([z["val"] for z in zs], 0)
        b[vi] = np.mean([z["val_b"] for z in zs], 0)
        ln[vi] = np.mean([z["val_logn"] for z in zs], 0)
        if all(len(z["test"]) == len(te_ids) for z in zs):
            yt.append(np.mean([z["test"] for z in zs], 0))
    held = tr.fold.isin(a.folds).values
    gm = held & tr.ID.isin(g1).values
    a3 = pd.read_parquet(DATA_DIR / "het_cnn" / a.a3 / "het_cnn_train.parquet").set_index("ID").loc[tr.ID]
    het_hat = b.std(1)
    c_het, c_n = corr(het_hat[gm], a3.het_cnn.values[gm]), corr(ln[gm], a3.logN_cnn.values[gm])
    ok_a = c_het >= 0.85 and c_n >= 0.97
    print(f"GATE17 {a.name} folds {a.folds} seeds {len(seeds)} | (a) G1 train held-out n {int(gm.sum())}: corr(sd b-hat, "
          f"het_A3) {c_het:.4f} (>= 0.85), corr(logN-hat, logN_A3) {c_n:.4f} (>= 0.97) -> {'PASS' if ok_a else 'FAIL'}")
    rh, rn = [], []
    for f in a.folds:
        fp = DATA_DIR / "het_cnn" / a.a3 / f"fold{f}.npz"
        if not fp.exists():
            break
        with np.load(fp) as z:
            pos = {i: k for k, i in enumerate(z["pred_ids"].tolist())}
            rows = [i for i in tr.ID[gm & (tr.fold.values == f)]]
            if not all(i in pos for i in rows):
                break
            pr = [pos[i] for i in rows]
            rh.append(pd.Series(z["b"][pr].std(1), index=rows))
            rn.append(pd.Series(z["logn"][pr], index=rows))
    else:
        if rh:
            ids_g = tr.ID[gm].tolist()
            h0, n0 = pd.concat(rh).loc[ids_g].values, pd.concat(rn).loc[ids_g].values
            print(f"GATE17 {a.name} (a) ref, the {a.a3} fold models themselves on these rows (0 steps): corr het "
                  f"{corr(h0, a3.het_cnn.values[gm]):.4f}, corr logN {corr(n0, a3.logN_cnn.values[gm]):.4f}")
    if yt:
        ytm = np.mean(yt, 0)
        gt = np.array([i in g1 for i in te_ids])
        D, p = ks(ytm[gt], y[gm])
        print(f"GATE17 {a.name} (d) KS y-hat G1 test (n {int(gt.sum())}, mean of folds {a.folds}) vs G1 train held-out "
              f"(n {int(gm.sum())}): D {D:.4f} p {p:.4f} (> 0.01) -> {'PASS' if p > 0.01 else 'FAIL'} (stop only)")
    else:
        print(f"GATE17 {a.name} (d) n/a: no test predictions in the cache")
    v = np.round(y[held], 2)
    print(f"GATE17 {a.name} y-hat held-out n {int(held.sum())}: mean {v.mean():.2f} sd {v.std():.2f}; G1 rows mean "
          f"{y[gm].mean():.2f}; het-hat mean {het_hat[gm].mean():.4f} (A3 {a3.het_cnn.values[gm].mean():.4f})")
    if a.yhat:
        print(f"# yhat {a.name} folds {' '.join(map(str, a.folds))} seeds {len(seeds)}: n {len(v)} sum {v.sum():.2f} "
              f"(held-out rows, train.csv order)")
        for k in range(0, len(v), 25):
            print(" ".join(f"{x:.2f}" for x in v[k:k + 25]))


def run_stage3(a):
    n6, n3, nf = a.stage3
    rd = lambda p: pd.read_csv(p).set_index("ID").hardness  # noqa: E731
    tr = train_frame()
    te_ids = pd.read_csv(DATA_DIR / "sample_submission.csv", usecols=["ID"]).ID.tolist()
    o6, o3 = rd(EXP_DIR / n6 / "oof.csv").loc[tr.ID], rd(EXP_DIR / n3 / "oof.csv").loc[tr.ID]
    t6, tf = rd(EXP_DIR / n6 / "test.csv").loc[te_ids], rd(EXP_DIR / nf / "test.csv").loc[te_ids]
    kf = 0.5 * (t6.values + tf.values)
    g1 = g1_ids()
    c = corr(o6.values, o3.values)
    r = float(np.sqrt(np.mean((tf.values - t6.values) ** 2)))
    gt, gm = np.array([i in g1 for i in te_ids]), tr.ID.isin(g1).values
    D, p = ks(kf[gt], o6.values[gm])
    if a.kf_chunk is None:
        print(f"GATE17 stage3 {n6}: corr(OOF {n6}, OOF {n3}) {c:.4f} (>= 0.98) -> {'PASS' if c >= 0.98 else 'FAIL'} | "
              f"rms(test {nf} - test {n6}) {r:.3f} (<= 1.5) -> {'PASS' if r <= 1.5 else 'FAIL'} | KS y-hat G1 test kf "
              f"(n {int(gt.sum())}) vs G1 train OOF (n {int(gm.sum())}): D {D:.4f} p {p:.4f} -> "
              f"{'PASS' if p > 0.01 else 'FAIL'} (stop only)")
        print(f"GATE17 stage3 sums: test {n6} {np.round(t6.values, 2).sum():.2f}, test {nf} "
              f"{np.round(tf.values, 2).sum():.2f}, kf {np.round(kf, 2).sum():.2f}")
        return
    for k in a.kf_chunk:
        v = np.round(kf, 2)[250 * k:250 * k + 250]
        print("#", n6 + "kf", "test", "chunk", k, "IDs", te_ids[250 * k], te_ids[250 * k + len(v) - 1], "n", len(v),
              "sum %.2f" % v.sum())
        print(" ".join("%.2f" % x for x in v))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default=None)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--a3", default="ev2s_locK_real5")
    ap.add_argument("--yhat", action="store_true")
    ap.add_argument("--stage3", nargs=3, default=None, metavar=("NAME6", "NAME3", "NAMEFULL"))
    ap.add_argument("--kf-chunk", type=int, nargs="+", default=None, choices=[0, 1, 2, 3])
    a = ap.parse_args(argv)
    if a.stage3:
        run_stage3(a)
    elif a.name:
        run_gate(a)
    else:
        ap.error("--name RUN or --stage3 NAME6 NAME3 NAMEFULL")


if __name__ == "__main__":
    sys.exit(main())
