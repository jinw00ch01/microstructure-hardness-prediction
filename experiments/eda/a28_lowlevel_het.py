"""A28: low-level heterogeneity across image blocks (raw image; fixed per-image transform, train or test).
Per block (2x2 of 128 px, 4x4 of 64 px): mean, std, p5, p95 of raw; noise (MAD of a Laplacian-like residual); edge energy
(mean |grad| of Gauss_1); then the std / range across blocks relative to the image mean, and a plane-fit gradient of
the block means (shading direction / strength).  Usage: python a28_lowlevel_het.py OUT_PARQUET split"""
import sys

import cv2
import numpy as np
import pandas as pd

from eda_common import DATA_DIR

out, split = sys.argv[1], sys.argv[2]
ids = pd.read_csv(DATA_DIR / ("train.csv" if split == "train" else "sample_submission.csv")).ID.tolist()
rows = []
for i in ids:
    raw = cv2.imread(str(DATA_DIR / split / f"{i}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    lap = raw - cv2.blur(raw, (3, 3))
    g1 = cv2.GaussianBlur(raw, (0, 0), 1.0)
    gy, gx = np.gradient(g1)
    gm = np.hypot(gx, gy)
    M = raw.mean()
    f = {"ID": i}
    for nb in (2, 4):
        bs = 256 // nb
        st = {k: [] for k in ("mean", "std", "p5", "p95", "noise", "edge")}
        for by in range(nb):
            for bx in range(nb):
                sl = (slice(by * bs, (by + 1) * bs), slice(bx * bs, (bx + 1) * bs))
                b = raw[sl]
                st["mean"].append(b.mean())
                st["std"].append(b.std())
                p5, p95 = np.percentile(b, [5, 95])
                st["p5"].append(p5)
                st["p95"].append(p95)
                st["noise"].append(1.4826 * np.median(np.abs(lap[sl] - np.median(lap[sl]))))
                st["edge"].append(gm[sl].mean())
        for k, v in st.items():
            v = np.array(v)
            f[f"ll{nb}_{k}_sd"] = float(v.std() / M)
            f[f"ll{nb}_{k}_rng"] = float((v.max() - v.min()) / M)
        if nb == 4:
            yy, xx = np.mgrid[:4, :4]
            A = np.column_stack([np.ones(16), yy.ravel(), xx.ravel()])
            c = np.linalg.lstsq(A, np.array(st["mean"]), rcond=None)[0]
            f["ll4_grad_mag"] = float(np.hypot(c[1], c[2]) / M)
            f["ll4_plane_resid_sd"] = float((np.array(st["mean"]) - A @ c).std() / M)
    f["ll_mean"] = float(M)
    rows.append(f)
pd.DataFrame(rows).to_parquet(out)
print("done", split, len(rows))
