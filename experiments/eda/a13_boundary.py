"""A13: grain-boundary appearance (depth / width) on clean images vs residuals."""
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from eda_common import DATA_DIR, train_table

cache = Path(sys.argv[1])
t = train_table()
ids = pd.read_csv(DATA_DIR / "train.csv").ID.tolist()
pos = {i: k for k, i in enumerate(ids)}
rn = np.load(cache / "rn_train.npy", mmap_mode="r")
ws = np.load(cache / "ws_train.npy", mmap_mode="r")
rows = []
for i in t.ID[t.ic_ridge_snr > 0.5]:
    k = pos[i]
    x = rn[k].astype(np.float32)
    raw = cv2.imread(str(DATA_DIR / "train" / f"{i}.png"), 0).astype(np.float32)
    w = np.asarray(ws[k])
    line = w == 0
    s1 = cv2.GaussianBlur(x, (0, 0), 0.7)
    # matrix-matrix boundaries: line pixels whose 5x5 neighbourhood interior median is matrix-like
    loc_max = ndi.grey_dilation(s1, size=5)
    bd = line & (loc_max > 0.95)
    depth = 1 - np.median(s1[bd] / np.clip(loc_max[bd], 0.5, None)) if bd.sum() > 50 else np.nan
    # width proxy: fraction of pixels within 2 px of a line that are darker than 0.97*local max
    near = ndi.binary_dilation(line, iterations=2) & (loc_max > 0.95)
    dark_near = np.mean((s1[near] / loc_max[near]) < 0.97) if near.sum() > 50 else np.nan
    rows.append({"ID": i, "bd_depth": depth, "bd_width_proxy": dark_near, "bd_line_frac": line.mean(),
                 "bd_raw_depth": 1 - np.median(raw[bd]) / np.median(raw[~line]) if bd.sum() > 50 else np.nan})
B = pd.DataFrame(rows).merge(t, on="ID")
Nc = B.cal_seg_count_density
for c in ["bd_depth", "bd_width_proxy", "bd_raw_depth", "bd_line_frac"]:
    co = Nc <= Nc.quantile(0.4)
    print(f"{c:16s} rho(y) {spearmanr(B[c], B.hardness, nan_policy='omit')[0]:+.3f}  rho(r) {spearmanr(B[c], B.resid, nan_policy='omit')[0]:+.3f}  "
          f"rho(r|coarse) {spearmanr(B[c][co], B.resid[co], nan_policy='omit')[0]:+.3f}  rho(|r|) {spearmanr(B[c], B.resid.abs(), nan_policy='omit')[0]:+.3f}  n={len(B)}")
