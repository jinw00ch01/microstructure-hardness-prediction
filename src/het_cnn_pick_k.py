"""Label-free stage-1 pick for route K (laptop-gpu.md section 13; pre-registration d1010/K in the hardness-cache).

  python -m src.het_cnn_pick_k --base ev2s_loc_e32r4 ev2s_loc_e32r4_s1 --d1 scrK_d1              # after the D1 screen
  python -m src.het_cnn_pick_k --base ev2s_loc_e32r4 ev2s_loc_e32r4_s1 --d1 scrK_d1 --d2 scrK_m2  # after the D2 screen
  python -m src.het_cnn_pick_k --stage1b ev2sK_d1 --design d1        # 5-fold run of the pick: stage-1b gate

Inputs (data/het_cnn/<run>/; no labels, no images): the base runs' score.json, score_eval.json (src.het_cnn --eval-only
--eval-presets noisy real) and score_eval_fold{f}.npz; the D1 screen's score.json ('stage1_eval', from
--eval-presets noisy real) and fold{f}_eval.npz; the D2 screen's score.json; the base runs' het_cnn_train.parquet (the
v24 estimator on the real G1 train images, for the agreement check); features_v3.parquet (G1 membership) and the prep
(logN of the clean originals). Every metric is a mean over folds 0 and 1.

D1 (--render-preset real, ev2s_loc flags) qualifies when, on folds 0-1:
  R (real held-out clean pcorr_het_given_logN, score.json) >= 0.7358;
  W (corr_block_within on the fixed current-prior renders, eval preset noisy) >= 0.6865;
  P_cur (pcorr_het_given_logN on the fixed current-prior renders) >= base P_cur - 0.02;
  P_real (pcorr on the fixed 'real'-prior renders) >= base P_real + 0.05, base = the base runs' fold models scored by
    --eval-only on the identical renders (md5 checked);
  real G1 train logN-hat (fold models 0-1 averaged): |mean - prep mean| <= 0.10 and |sd - prep sd| <= 0.10
    (--logn-ref base: against the base runs' own folds 0-1 logN-hat on the same images instead; the v24 estimator
    itself sits at mean -0.106 / sd +0.018 vs the prep on these 208 images, so the prep form is a tight bar).
  Sanity only (reported, never decides): partial corr (each residualised on its own logN) of the D1 het-hat with the
  v24 estimator (ev2s_loc_e32r4 + _s1 parquets, seed mean) on the 208 real G1 train images >= 0.75.
D2 (tf_efficientnetv2_m, current prior; run only if D1 does not qualify) qualifies when P (stage1_renders) >= 0.5713,
  W >= 0.6865 and R >= 0.7358 (the d1009 H bars).
Last line: 'PICK <run>', 'NEXT d2' (D1 does not qualify and no --d2 given) or 'PICK none'.
--stage1b RUN --design d1|d2: the 5-fold gate (D1: pooled P_cur >= 0.48 and pooled R >= 0.70; D2: pooled P >= 0.54);
last line 'STAGE1B PASS' or 'STAGE1B FAIL'."""
import argparse
import json
import sys

import numpy as np
import pandas as pd

from .common import DATA_DIR as DATA

ROOT = DATA / "het_cnn"
FOLDS = ("0", "1")
D1_BARS = dict(R=0.7358, W=0.6865, dP_cur=-0.02, dP_real=0.05, logn=0.10, agree=0.75)
D2_BARS = dict(P=0.5713, W=0.6865, R=0.7358)
S1B = dict(d1=dict(P_cur=0.48, R=0.70), d2=dict(P=0.54))


def _js(run, name):
    fp = ROOT / run / name
    if not fp.exists():
        raise SystemExit(f"{fp} missing")
    return json.loads(fp.read_text())


def _resid(v, z):
    Z = np.column_stack([np.ones(len(z)), z])
    return v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]


def _g1(run, pattern):
    """Real G1 train predictions of folds 0-1 (blocks averaged over the folds, het = sd of the 16 blocks)."""
    Z = [np.load(ROOT / run / pattern.format(f=f)) for f in FOLDS]
    ids = [list(z["g1_ids"]) for z in Z]
    assert all(i == ids[0] for i in ids), f"{run}: folds predicted different G1 lists"
    b = np.mean([z["g1_b"] for z in Z], 0)
    return ids[0], b.std(1), np.mean([z["g1_logn"] for z in Z], 0)


def base_metrics(run):
    s, e = _js(run, "score.json"), _js(run, "score_eval.json")
    per, ev = s["stage1_per_fold"], e["per_fold"]
    miss = [f for f in FOLDS if f not in per or f not in ev]
    assert not miss, f"{run}: folds {miss} missing in score.json / score_eval.json"
    m = dict(R=np.mean([per[f]["heldout_real_clean"]["pcorr_het_given_logN"] for f in FOLDS]),
             W_score=np.mean([per[f]["stage1_renders"]["corr_block_within"] for f in FOLDS]),
             P_score=np.mean([per[f]["stage1_renders"]["pcorr_het_given_logN"] for f in FOLDS]))
    for pr, tag in (("noisy", "cur"), ("real", "real")):
        m[f"P_{tag}"] = np.mean([ev[f][pr]["P"] for f in FOLDS])
        m[f"W_{tag}"] = np.mean([ev[f][pr]["W"] for f in FOLDS])
    md5 = {(f, pr): ev[f][pr]["md5"] for f in FOLDS for pr in ("noisy", "real")}
    return {k: float(v) for k, v in m.items()}, md5, _g1(run, "score_eval_fold{f}.npz")


def d1_metrics(run):
    s = _js(run, "score.json")
    per, se = s["stage1_per_fold"], s.get("stage1_eval")
    assert se, f"{run}: score.json has no stage1_eval (run it with --eval-presets noisy real)"
    ev = se["per_fold"]
    miss = [f for f in FOLDS if f not in per or f not in ev]
    assert not miss, f"{run}: folds {miss} not done (or without --eval-presets)"
    m = dict(R=np.mean([per[f]["heldout_real_clean"]["pcorr_het_given_logN"] for f in FOLDS]))
    for pr, tag in (("noisy", "cur"), ("real", "real")):
        m[f"P_{tag}"] = np.mean([ev[f][pr]["P"] for f in FOLDS])
        m[f"W_{tag}"] = np.mean([ev[f][pr]["W"] for f in FOLDS])
    md5 = {(f, pr): ev[f][pr]["md5"] for f in FOLDS for pr in ("noisy", "real")}
    train_preset = s.get("render_preset", "noisy")
    return {k: float(v) for k, v in m.items()}, md5, _g1(run, "fold{f}_eval.npz"), train_preset


def d2_metrics(run):
    per = _js(run, "score.json")["stage1_per_fold"]
    miss = [f for f in FOLDS if f not in per]
    assert not miss, f"{run}: folds {miss} not done"
    return dict(P=float(np.mean([per[f]["stage1_renders"]["pcorr_het_given_logN"] for f in FOLDS])),
                W=float(np.mean([per[f]["stage1_renders"]["corr_block_within"] for f in FOLDS])),
                R=float(np.mean([per[f]["heldout_real_clean"]["pcorr_het_given_logN"] for f in FOLDS])))


def v24_estimator(base, ids):
    H = [pd.read_parquet(ROOT / r / "het_cnn_train.parquet").set_index("ID").loc[ids] for r in base]
    return np.mean([h.het_cnn.values for h in H], 0), np.mean([h.logN_cnn.values for h in H], 0)


def g1_ids_v3():
    v3 = pd.read_parquet(DATA / "features_v3.parquet", columns=["ID", "ic_noise", "ic_ridge_snr"]).set_index("ID")
    ids = pd.read_csv(DATA / "train.csv", usecols=["ID"]).ID
    return [i for i in ids if v3.loc[i, "ic_noise"] >= 9.5 and v3.loc[i, "ic_ridge_snr"] < 0.5]


def stage1b(a):
    s = _js(a.stage1b, "score.json")
    done = s["folds_done"]
    per, se = s["stage1_per_fold"], s.get("stage1_eval") or {}
    for f in map(str, done):  # per-fold summary (P, W on the training-preset renders; eval presets; real clean R)
        ev, s1, hr = se.get("per_fold", {}).get(f, {}), per[f]["stage1_renders"], per[f]["heldout_real_clean"]
        print(f"fold {f}: stage1_renders P {s1['pcorr_het_given_logN']:.4f} W {s1['corr_block_within']:.4f} | R "
              + (f"{hr['pcorr_het_given_logN']:.4f}" if hr else "n/a")
              + "".join(f" | eval {pr} P {ev[pr]['P']:.4f} W {ev[pr]['W']:.4f}"
                        for pr in ("noisy", "real") if pr in ev))
    if s.get("stage1_pooled"):
        print(f"pooled stage1_renders: P {s['stage1_pooled']['pcorr_het_given_logN']:.4f} W "
              f"{s['stage1_pooled']['corr_block_within']:.4f}")
    if a.design == "d1":
        se = s.get("stage1_eval") or {}
        pc = (se.get("pooled") or {}).get("noisy")
        hr = se.get("heldout_real_clean_pooled")
        if pc is None or hr is None or len(done) < 5:
            why = (f"only folds {done} done" if len(done) < 5 else
                   "score.json stage1_eval has no pooled noisy metrics or pooled real-clean R (run with --eval-presets"
                   " noisy real)")
            print(f"{a.stage1b}: {why}")
            print("STAGE1B FAIL")
            return
        P, R = pc["pcorr_het_given_logN"], hr["pcorr_het_given_logN"]
        ok = P >= S1B["d1"]["P_cur"] and R >= S1B["d1"]["R"]
        print(f"{a.stage1b} (D1, 5 folds {done}): pooled P_cur {P:.4f} (bar >= 0.48), pooled R {R:.4f} (bar >= 0.70); "
              f"pooled P_real {se['pooled']['real']['pcorr_het_given_logN']:.4f} (report); real G1 logN-hat "
              f"{se['real_g1_train']['logN_mean']:.3f} / {se['real_g1_train']['logN_sd']:.3f} (prep "
              f"{se['real_g1_train']['prep_logN_mean']:.3f} / {se['real_g1_train']['prep_logN_sd']:.3f}; report)")
    else:
        pl = s.get("stage1_pooled")
        P = pl["pcorr_het_given_logN"] if pl else float("nan")
        ok = len(done) == 5 and P >= S1B["d2"]["P"]
        print(f"{a.stage1b} (D2, 5 folds {done}): pooled P {P:.4f} (bar >= 0.54)")
    print("STAGE1B PASS" if ok else "STAGE1B FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", default=["ev2s_loc_e32r4", "ev2s_loc_e32r4_s1"])
    ap.add_argument("--d1", default=None)
    ap.add_argument("--d2", default=None)
    ap.add_argument("--stage1b", default=None)
    ap.add_argument("--design", choices=["d1", "d2"], default=None)
    ap.add_argument("--logn-ref", choices=["prep", "base"], default="prep",
                    help="D1 logN bar reference: prep (the clean originals' log N_eff, as pre-registered) or base "
                         "(the base runs' folds 0-1 logN-hat on the same real G1 images)")
    a = ap.parse_args()
    if a.stage1b:
        if a.design is None:
            ap.error("--stage1b needs --design d1|d2")
        stage1b(a)
        return
    if not a.d1:
        ap.error("--d1 RUN is required")
    B = [base_metrics(r) for r in a.base]
    bm = {k: float(np.mean([b[0][k] for b in B])) for k in B[0][0]}
    print(f"base {'+'.join(a.base)} (folds 0-1): R {bm['R']:.4f} | eval noisy P_cur {bm['P_cur']:.4f} W_cur "
          f"{bm['W_cur']:.4f} (score.json stage1_renders P {bm['P_score']:.4f} W {bm['W_score']:.4f}) | eval real "
          f"P_real {bm['P_real']:.4f} W_real {bm['W_real']:.4f}")
    g1 = g1_ids_v3()
    ok_d1 = None
    out = dict(base=bm)
    try:
        dm, dmd5, (gid, gh, gl), tp = d1_metrics(a.d1)
    except (AssertionError, SystemExit) as e:
        print(f"D1 {a.d1}: not available ({e})")
        dm = None
    if dm is not None:
        same = all(dmd5[k] == b[1][k] for b in B for k in dmd5)
        assert same, "the D1 and base eval renders differ (md5): not the same fixed render sets"
        assert list(gid) == g1 and len(g1) == 208, "G1 list differs from features_v3 (expected the 208 G1 train)"
        P = np.load(ROOT / "prep.npz")
        ln = np.log(P["n_eff"])
        hv, lv = v24_estimator(a.base, gid)
        agree = float(np.corrcoef(_resid(gh, gl), _resid(hv, lv))[0, 1])
        bg = [b[2] for b in B]
        agree_b = [float(np.corrcoef(_resid(gh, gl), _resid(x[1], x[2]))[0, 1]) for x in bg]
        # logN-hat of the real G1 train images vs the reference (prep: the clean originals' log N_eff; base: the base
        # runs' own folds 0-1 logN-hat on the same images)
        ref = ((float(ln.mean()), float(ln.std())) if a.logn_ref == "prep" else
               (float(np.mean([x[2].mean() for x in bg])), float(np.mean([x[2].std() for x in bg]))))
        dmean, dsd = float(gl.mean() - ref[0]), float(gl.std() - ref[1])
        bdm = [(float(x[2].mean() - ln.mean()), float(x[2].std() - ln.std())) for x in bg]
        bars = dict(R=dm["R"] >= D1_BARS["R"], W_cur=dm["W_cur"] >= D1_BARS["W"],
                    P_cur=dm["P_cur"] >= bm["P_cur"] + D1_BARS["dP_cur"],
                    P_real=dm["P_real"] >= bm["P_real"] + D1_BARS["dP_real"],
                    logN=abs(dmean) <= D1_BARS["logn"] and abs(dsd) <= D1_BARS["logn"])
        ok_d1 = all(bars.values()) and tp == "real"
        print(f"D1 {a.d1} (training preset {tp}): R {dm['R']:.4f} (>= 0.7358) | W_cur {dm['W_cur']:.4f} (>= 0.6865) | "
              f"P_cur {dm['P_cur']:.4f} (>= {bm['P_cur'] + D1_BARS['dP_cur']:.4f}) | P_real {dm['P_real']:.4f} (>= "
              f"{bm['P_real'] + D1_BARS['dP_real']:.4f}) | W_real {dm['W_real']:.4f} (report)")
        print(f"D1 real G1 train logN-hat (folds 0-1): mean {gl.mean():.3f} sd {gl.std():.3f} vs {a.logn_ref} "
              f"{ref[0]:.3f} / {ref[1]:.3f} (d {dmean:+.3f} / {dsd:+.3f}, bar within +-0.10) | prep {ln.mean():.3f} / "
              f"{ln.std():.3f}; D1 - prep {gl.mean() - ln.mean():+.3f} / {gl.std() - ln.std():+.3f}; each base run - "
              "prep " + ", ".join(f"{u:+.3f} / {v:+.3f}" for u, v in bdm))
        print(f"D1 sanity (never decides): agreement with the v24 estimator on the 208 real G1 train {agree:.3f} "
              f"(>= 0.75: {'ok' if agree >= D1_BARS['agree'] else 'LOW'}); with each base run's folds 0-1 "
              f"{', '.join(f'{v:.3f}' for v in agree_b)}")
        print(f"D1 bars: {json.dumps(bars)} -> {'qualifies' if ok_d1 else 'does not qualify'}")
        out.update(d1=dict(run=a.d1, **dm, agree_v24=agree, d_logN_mean=dmean, d_logN_sd=dsd, bars=bars,
                           qualifies=ok_d1))
    if ok_d1:
        print(json.dumps(out, default=float))
        print(f"PICK {a.d1}")
        return
    if not a.d2:
        print(json.dumps(out, default=float))
        print("NEXT d2" if ok_d1 is False else "PICK none")
        return
    if not (ROOT / a.d2 / "score.json").exists():
        print(f"D2 {a.d2}: no score.json (not run)")
        print("PICK none")
        return
    try:
        m2 = d2_metrics(a.d2)
    except AssertionError as e:
        print(f"D2 {e} (incomplete)")
        print(json.dumps(out, default=float))
        print("PICK none")
        return
    ok2 =m2["P"] >= D2_BARS["P"] and m2["W"] >= D2_BARS["W"] and m2["R"] >= D2_BARS["R"]
    print(f"D2 {a.d2}: P {m2['P']:.4f} (>= 0.5713) W {m2['W']:.4f} (>= 0.6865) R {m2['R']:.4f} (>= 0.7358) -> "
          f"{'qualifies' if ok2 else 'does not qualify'}")
    out.update(d2=dict(run=a.d2, **m2, qualifies=ok2))
    print(json.dumps(out, default=float))
    print(f"PICK {a.d2 if ok2 else 'none'}")


if __name__ == "__main__":
    sys.exit(main())
