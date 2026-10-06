"""A27: compact effv2s grid-cell heterogeneity features for train + test (fixed per-image transforms).
cellstd = std over the 4 grid cells (2x2 quadrants) of the tf_efficientnetv2_s g2a embedding, per geometric view (id,
hflip, transpose, rot90), averaged over views, signed-log.  Exports
  ecs_blk{stage}_{pool}: mean over channels of each stage/pool block (10 cols)
  ecs_pc{j}: PCA(16) of stages 2-3 cellstd, PCA fitted on TRAIN rows only and applied to every image as a fixed linear map
Usage: python a27_cellstd_parquet.py OUT_PARQUET"""
import json
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from eda_common import DATA_DIR

name = "tf_efficientnetv2_s.in21k_ft_in1k_256_g2a"
meta = json.load(open(DATA_DIR / "emb" / f"{name}_meta.json"))
E = np.load(DATA_DIR / "emb" / f"{name}.npy", mmap_mode="r")
ids = pd.read_csv(DATA_DIR / "emb" / f"{name}_ids.csv").iloc[:, 0].values
X = np.asarray(E, np.float32).reshape(len(ids), 6, 5, -1)
cs = X[:, :4, 1:].std(2).mean(1)
cs = np.sign(cs) * np.log1p(np.abs(cs))
out = pd.DataFrame({"ID": ids})
for b in meta["blocks"]:
    out[f"ecs_blk{b['stage']}_{b['pool']}"] = cs[:, b["start"]:b["end"]].mean(1)
sel = np.concatenate([np.arange(b["start"], b["end"]) for b in meta["blocks"] if b["stage"] in (2, 3)])
is_tr = pd.Series(ids).str.startswith("TRAIN").values
Z = cs[:, sel]
mu, sd = Z[is_tr].mean(0), Z[is_tr].std(0) + 1e-6
pca = PCA(16, random_state=0).fit((Z[is_tr] - mu) / sd)
P = pca.transform((Z - mu) / sd)
for j in range(16):
    out[f"ecs_pc{j}"] = P[:, j]
out.to_parquet(sys.argv[1])
print(out.shape, "explained var (train) %.3f" % pca.explained_variance_ratio_.sum())
