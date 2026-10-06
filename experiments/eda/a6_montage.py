"""A6: montages of coarse images: pairs with similar blend prediction but large +/- residuals.
Usage: python a6_montage.py CACHE_DIR [snr_min] [tag]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from eda_common import DATA_DIR, OUT, train_table

cache = Path(sys.argv[1])
snr_min = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
tag = sys.argv[3] if len(sys.argv) > 3 else "clean"
t = train_table().merge(pd.read_parquet(cache / "grainstats_train.parquet"), on="ID")
c = t[(t.ic_acg_len50_gm >= t.ic_acg_len50_gm.quantile(2 / 3)) & (t.ic_ridge_snr >= snr_min)].copy()
print("coarse & snr>=", snr_min, len(c))
pos = c[c.resid > 12].sort_values("resid", ascending=False)
neg = c[c.resid < -12].copy()
pairs = []
used = set()
for _, rp in pos.iterrows():
    cand = neg[~neg.ID.isin(used)]
    if cand.empty:
        break
    j = (cand.pred - rp.pred).abs().idxmin()
    rn_ = cand.loc[j]
    if abs(rn_.pred - rp.pred) > 6:
        continue
    used.add(rn_.ID)
    pairs.append((rp, rn_))
    if len(pairs) == 6:
        break


def tile(row):
    a = np.asarray(Image.open(DATA_DIR / "train" / f"{row.ID}.png").convert("RGB"))
    im = Image.fromarray(a.copy())
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 255, 12], fill=(0, 0, 0))
    d.text((2, 1), f"{row.ID[-4:]} y{row.hardness:.0f} p{row.pred:.0f} r{row.resid:+.0f} fd{row.cal_ic_seg_fd91:.2f} n{row.g_n}",
           fill=(255, 255, 0))
    return im


W = Image.new("RGB", (len(pairs) * 260, 2 * 260), (255, 255, 255))
for k, (a, b) in enumerate(pairs):
    W.paste(tile(a), (k * 260, 0))
    W.paste(tile(b), (k * 260, 260))
W.save(OUT / f"m_coarse_pairs_{tag}.png")
for a, b in pairs:
    print(f"{a.ID} y={a.hardness:.1f} pred={a.pred:.1f} | {b.ID} y={b.hardness:.1f} pred={b.pred:.1f}")
