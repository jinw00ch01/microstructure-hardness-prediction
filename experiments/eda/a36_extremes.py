"""A36: clean-image extremes of the blend_v4 nested residual (ic_ridge_snr > 0.9, coarse and mid N terciles).
1. Montages: the 6 most under-predicted (resid > 0) and 6 most over-predicted images per tercile, raw + rn with the
   dark-phase / pore outline, annotated with y, pred, resid, N, fd, elongation of dark and matrix grains.
2. Feature scan on the 96 clean coarse+mid images: Spearman of every available per-image feature with the residual,
   compared with a permutation null of the maximum |rho| (multiple testing), and the elongation-by-phase block.
Usage: python a36_extremes.py CACHE_DIR"""
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

from eda_common import DATA_DIR, OUT, load_feats  # noqa: E402
from eda_cf import Reporter, base, table  # noqa: E402

cache = Path(sys.argv[1])
t = table()
oof, inner = base(t)
R = Reporter(t, oof)
t["terc"] = R.terc
t["base_res"] = R.base_res
F = load_feats(("features_v2.parquet", "eda_feats_lledge.parquet", "eda_feats_ecs.parquet", "eda_feats_lf.parquet"))
F = F.drop(columns=[c for c in F.columns if c in t.columns and c != "ID"])
t = t.merge(F, on="ID", how="left")
for fn in ("mig_train", "gf_train", "rp_train", "sp_train", "sd_train"):
    d = pd.read_parquet(cache / f"{fn}.parquet")
    t = t.merge(d.drop(columns=[c for c in d.columns if c in t.columns and c != "ID"]), on="ID", how="left")
idx_train = {i: k for k, i in enumerate(pd.read_csv(DATA_DIR / "train.csv").ID)}
rn_all = np.load(cache / "rn_train.npy", mmap_mode="r")
ws_all = np.load(cache / "ws_train.npy", mmap_mode="r")
sel = (t.ic_ridge_snr > 0.9) & t.terc.isin(["coarse", "mid"])
c = t[sel].reset_index(drop=True)
print(f"clean coarse+mid n={len(c)}; blend_v4 residual sd {c.resid.std():.2f}")

# ---------------- montages
for g in ("coarse", "mid"):
    d = c[c.terc == g].sort_values("resid")
    picks = [("under-predicted (y > pred)", d.tail(6).iloc[::-1]), ("over-predicted (y < pred)", d.head(6))]
    fig, ax = plt.subplots(4, 6, figsize=(24, 17))
    for r_, (lab, dd) in enumerate(picks):
        for j, (_, row) in enumerate(dd.iterrows()):
            raw = cv2.imread(str(DATA_DIR / "train" / f"{row.ID}.png"), cv2.IMREAD_GRAYSCALE)
            k = idx_train[row.ID]
            rn = np.asarray(rn_all[k], np.float32)
            ws = np.asarray(ws_all[k])
            a0, a1 = ax[2 * r_, j], ax[2 * r_ + 1, j]
            a0.imshow(raw, cmap="gray", vmin=0, vmax=255)
            a0.set_title(f"{lab[:5]} {row.ID}\ny {row.hardness:.0f} pred {row.pred:.0f} r {row.resid:+.0f}", fontsize=10)
            ov = np.dstack([np.clip(rn, 0.6, 1.1)] * 3)
            ov = (ov - 0.6) / 0.5
            dk = cv2.GaussianBlur(rn, (0, 0), 1.5) < 0.91
            ov[dk] = ov[dk] * np.array([1.0, 0.55, 0.55])
            ov[ws == 0] = [0.2, 0.6, 1.0]
            a1.imshow(np.clip(ov, 0, 1))
            a1.set_title(f"N {row.cal_seg_count_density * 6.5536:.0f} fd {row.cal_ic_seg_fd91:.2f} snr {row.ic_ridge_snr:.2f}\n"
                         f"asp dk {np.exp(row.get('mi_e_dk_logasp', np.nan)):.2f} mx {np.exp(row.get('mi_e_mx_logasp', np.nan)):.2f} "
                         f"pore {row.ic_pore68_frac:.3f}", fontsize=9)
            for a in (a0, a1):
                a.axis("off")
    fig.suptitle(f"clean {g} tercile: blend_v4 nested residual extremes (rows 1-2 under-predicted, rows 3-4 over-predicted; "
                 f"red = dark phase, blue = watershed lines)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / f"m_v4_extremes_clean_{g}.png", dpi=55)
    plt.close(fig)

# ---------------- feature scan
num = [x for x in c.columns if x not in ("ID", "hardness", "fold", "pred", "resid", "terc", "base_res") and not x.startswith("oof_")
       and pd.api.types.is_numeric_dtype(c[x]) and c[x].notna().mean() > 0.8 and c[x].nunique() > 5]
r = c.resid.values
Xr = np.column_stack([rankdata(c[x].fillna(c[x].median())) for x in num])
rr = rankdata(r)


def rho_all(target_rank):
    Z = (Xr - Xr.mean(0)) / (Xr.std(0) + 1e-12)
    z = (target_rank - target_rank.mean()) / target_rank.std()
    return Z.T @ z / len(z)


obs = rho_all(rr)
rng = np.random.default_rng(0)
mx = np.array([np.abs(rho_all(rng.permutation(rr))).max() for _ in range(300)])
order = np.argsort(-np.abs(obs))
print(f"{len(num)} features; max |rho| observed {np.abs(obs).max():.3f}; permutation null of max |rho|: "
      f"median {np.median(mx):.3f}, 95% {np.quantile(mx, 0.95):.3f}")
print("top 20 by |rho| with the blend_v4 residual (clean coarse+mid):")
for j in order[:20]:
    x = c[num[j]]
    print(f"  {num[j]:32s} rho {obs[j]:+.3f} | coarse {spearmanr(x[c.terc == 'coarse'], r[c.terc == 'coarse'], nan_policy='omit')[0]:+.2f} "
          f"mid {spearmanr(x[c.terc == 'mid'], r[c.terc == 'mid'], nan_policy='omit')[0]:+.2f} | vs base res {spearmanr(x, c.base_res, nan_policy='omit')[0]:+.2f}")
pd.DataFrame({"feat": num, "rho_resid": obs}).sort_values("rho_resid", key=np.abs, ascending=False).to_csv(OUT / "a36_scan_clean_coarse_mid.csv", index=False)

print("\nelongation by phase (Spearman with residual; clean coarse / mid / pooled) and group medians (top vs bottom third):")
q1, q2 = np.quantile(r, [1 / 3, 2 / 3])
for x in [v for v in num if v.startswith("mi_e_") or "asp" in v or "elong" in v or v.startswith("gf_logasp") or v in ("ic_st_coh", "seg_ori_R")]:
    s = c[x]
    print(f"  {x:28s} coarse {spearmanr(s[c.terc == 'coarse'], r[c.terc == 'coarse'], nan_policy='omit')[0]:+.2f} "
          f"mid {spearmanr(s[c.terc == 'mid'], r[c.terc == 'mid'], nan_policy='omit')[0]:+.2f} pooled {spearmanr(s, r, nan_policy='omit')[0]:+.2f} | "
          f"under-pred median {s[r > q2].median():.3f} over-pred median {s[r < q1].median():.3f}")
