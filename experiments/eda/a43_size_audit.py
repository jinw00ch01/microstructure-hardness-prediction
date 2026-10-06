"""A43: audit features for the official factor list (fixed per-image transform, train or test), from the a34 per-grain
table (CACHE/grains2_{split}.parquet: watershed grains, area >= 12, pore share <= 0.5) and the cached watershed.
G = all such grains, I = interior grains (not touching the border, untruncated).  Dark = interior median rn < 0.91.
Grain-size estimators (log of a length in px; d = sqrt(area)):
  gs_cnt_jeff    0.5 log(A_G / N_J),  N_J = |I| + 0.5 |G \\ I|  (Jeffries planimetric count)
  gs_cnt_raw     0.5 log(A_G / |G|)
  gs_med         0.5 median(log a_I)            gs_mode   0.5 argmax KDE(log a_I)
  gs_trim10/20   0.5 trimmed mean (10/20% per tail) of log a_I      gs_trimA10  0.5 log trimmed mean of a_I
  gs_cnt_ex10/20 count density after removing the largest grains holding the top 10/20% of grain area (Jeffries)
  gs_icpt        log mean intercept length along rows and columns (watershed-line runs per line length)
  gs_icpt_par/perp  the same along / across the dominant grain orientation (nearest of the 4 axes 0/45/90/135 deg)
  gs_area_w      0.5 log(sum a^2 / sum a)  (area-weighted mean area, G)      gs_num_mean  0.5 log(mean a_I)
Alignment (never the absolute angle): order parameter S = |sum w exp(2i theta)| / sum w per phase (all/dk/mx),
  weights number (al_S_n_*), area (al_S_a_*), elongation 1 - min/maj (al_S_e_*); mean aspect maj/min number- and
  area-weighted (al_asp_n_*, al_asp_a_*).
Size-distribution width (I): sd / IQR / skew of log a, Gini and CV of a, top-10% (by count) area share, Sarle bimodality
  coefficient, 2-GMM Ashman D and minor weight on log a; area-weighted sd / IQR of log a; area share in grains
  > 2x / > 4x the median area (sw_*).
Writes CACHE/ga_{split}.parquet.  Usage: python a43_size_audit.py CACHE_DIR split"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, skew, kurtosis
from sklearn.mixture import GaussianMixture

from eda_common import DATA_DIR

cache, split = Path(sys.argv[1]), sys.argv[2]
ids = pd.read_csv(DATA_DIR / ("train.csv" if split == "train" else "sample_submission.csv")).ID.tolist()
ws_all = np.load(cache / f"ws_{split}.npy", mmap_mode="r")
G2 = pd.read_parquet(cache / f"grains2_{split}.parquet")
groups = dict(tuple(G2.groupby("ID")))


def wq(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return float(np.interp(q, c, x[o]))


def crossings(line):
    z = (line == 0).astype(np.int8)
    return int(np.sum(np.diff(np.concatenate([[0], z])) == 1))


def gini(a):
    a = np.sort(a)
    n = len(a)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(a) / (n * a.sum()))


def diag_lines(w, anti=False):
    W = np.fliplr(w) if anti else w
    return [np.diagonal(W, o) for o in range(-200, 201)]


rows = []
for k, i in enumerate(ids):
    f = {"ID": i}
    d = groups.get(i)
    w = np.asarray(ws_all[k])
    if d is None or len(d) < 8:
        rows.append(f)
        continue
    a = d.area.values.astype(float)
    inter = ~d.border.values.astype(bool)
    dk = d.med.values < 0.91
    aI = a[inter] if inter.sum() >= 5 else a
    la = np.log(aI)
    AG = a.sum()
    NJ = inter.sum() + 0.5 * (~inter).sum()
    f["gs_cnt_jeff"] = 0.5 * np.log(AG / NJ)
    f["gs_cnt_raw"] = 0.5 * np.log(AG / len(a))
    f["gs_med"] = 0.5 * float(np.median(la))
    try:
        xs = np.linspace(la.min(), la.max(), 200)
        f["gs_mode"] = 0.5 * float(xs[np.argmax(gaussian_kde(la)(xs))])
    except Exception:
        f["gs_mode"] = np.nan
    for p in (10, 20):
        lo, hi = np.percentile(la, [p, 100 - p])
        f[f"gs_trim{p}"] = 0.5 * float(la[(la >= lo) & (la <= hi)].mean())
    lo, hi = np.percentile(aI, [10, 90])
    f["gs_trimA10"] = 0.5 * float(np.log(aI[(aI >= lo) & (aI <= hi)].mean()))
    o = np.argsort(-a)
    cs = np.cumsum(a[o]) / AG
    for p in (10, 20):
        drop = cs <= p / 100.0
        drop[0] = True if not drop.any() else drop[0]
        keep = o[~drop]
        if len(keep) >= 5:
            f[f"gs_cnt_ex{p}"] = 0.5 * np.log(a[keep].sum() / (inter[keep].sum() + 0.5 * (~inter[keep]).sum()))
    cr = sum(crossings(w[r]) for r in range(256)) + sum(crossings(w[:, c]) for c in range(256))
    f["gs_icpt"] = float(np.log(2 * 256 * 256 / max(cr, 1)))
    # dominant orientation (area-weighted doubled angle), nearest of 4 axes for directional intercepts
    th = d.ori.values
    C2, S2 = np.sum(a * np.cos(2 * th)), np.sum(a * np.sin(2 * th))
    th0 = 0.5 * np.arctan2(S2, C2)  # skimage: angle of the major axis w.r.t. the row (0th) axis
    ax = int(np.round((np.degrees(th0) % 180) / 45.0)) % 4  # 0: along rows-axis (vertical), 1/3: diagonals, 2: horizontal
    lines = {0: [w[:, c] for c in range(256)], 2: [w[r] for r in range(256)], 1: diag_lines(w, False), 3: diag_lines(w, True)}
    par, perp = lines[ax], lines[(ax + 2) % 4]
    Lp = sum(len(x) for x in par) / max(sum(crossings(x) for x in par), 1)
    Lq = sum(len(x) for x in perp) / max(sum(crossings(x) for x in perp), 1)
    f["gs_icpt_par"], f["gs_icpt_perp"] = float(np.log(Lp)), float(np.log(Lq))
    f["gs_area_w"] = 0.5 * float(np.log((a ** 2).sum() / AG))
    f["gs_num_mean"] = 0.5 * float(np.log(aI.mean()))
    # alignment and aspect per phase
    asp = d["maj"].values / np.maximum(d["min"].values, 1.0)
    el = 1.0 - 1.0 / asp
    for ph, m in (("all", np.ones(len(a), bool)), ("dk", dk), ("mx", ~dk)):
        if m.sum() < 3:
            continue
        e2 = np.exp(2j * th[m])
        f[f"al_S_n_{ph}"] = float(np.abs(e2.mean()))
        f[f"al_S_a_{ph}"] = float(np.abs((a[m] * e2).sum()) / a[m].sum())
        f[f"al_S_e_{ph}"] = float(np.abs((el[m] * e2).sum()) / max(el[m].sum(), 1e-9))
        f[f"al_asp_n_{ph}"] = float(np.log(asp[m]).mean())
        f[f"al_asp_a_{ph}"] = float((a[m] * np.log(asp[m])).sum() / a[m].sum())
    # size-distribution width
    f["sw_sd"] = float(la.std())
    f["sw_iqr"] = float(np.subtract(*np.percentile(la, [75, 25])))
    f["sw_skew"] = float(skew(la))
    f["sw_gini"] = gini(aI)
    f["sw_cv"] = float(aI.std() / aI.mean())
    n10 = max(1, int(round(0.1 * len(aI))))
    f["sw_top10_share"] = float(np.sort(aI)[::-1][:n10].sum() / aI.sum())
    n = len(la)
    if n > 3:
        g_, kt = skew(la), kurtosis(la)
        f["sw_bimod"] = float((g_ ** 2 + 1) / (kt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))
    try:
        gm = GaussianMixture(2, random_state=0).fit(la[:, None])
        mu, sd = gm.means_.ravel(), np.sqrt(gm.covariances_.ravel())
        f["sw_ashD"] = float(np.sqrt(2) * abs(mu[0] - mu[1]) / np.sqrt(sd[0] ** 2 + sd[1] ** 2))
        f["sw_gmm_wmin"] = float(gm.weights_.min())
    except Exception:
        pass
    laG, wG = np.log(a), a / AG
    mu_w = float((wG * laG).sum())
    f["sw_sd_aw"] = float(np.sqrt((wG * (laG - mu_w) ** 2).sum()))
    f["sw_iqr_aw"] = wq(laG, a, 0.75) - wq(laG, a, 0.25)
    med_a = np.median(aI)
    f["sw_share_gt2med"] = float(a[a > 2 * med_a].sum() / AG)
    f["sw_share_gt4med"] = float(a[a > 4 * med_a].sum() / AG)
    f["ga_n"] = int(len(a))
    rows.append(f)
    if k % 100 == 0:
        print(k, flush=True)
pd.DataFrame(rows).to_parquet(cache / f"ga_{split}.parquet")
print("done", split, len(rows))
