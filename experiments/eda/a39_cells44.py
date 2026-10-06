"""A39: does 4x4 cross-cell spread of CNN feature maps carry signal beyond feat4 and the blend_v4 members (which
include the 2x2 cell-std head emb_effv2s_256_gridge3_noise_cs2)?  Cross-fitted probes (eda_cf) on
data/emb/tf_efficientnetv2_s.in21k_ft_in1k_256_c24n_cells.npy (views id/hflip/transpose/rot90 used; cells 0-3 = 2x2,
4-19 = 4x4; dims = stage 0-2 mean/std blocks).  Per image: std (and max-min) over cells per dim, mean over the 4
views, sign*log1p; block means (6 per grid) and full vectors.  Usage: python a39_cells44.py"""
import json

import numpy as np
import pandas as pd

from eda_cf import Reporter, base, probe, table
from eda_common import DATA_DIR

t = table()
oof, inner = base(t)
R = Reporter(t, oof)
stem = DATA_DIR / "emb" / "tf_efficientnetv2_s.in21k_ft_in1k_256_c24n"
E = np.load(f"{stem}_cells.npy", mmap_mode="r")
ids = pd.read_csv(f"{stem}_ids.csv").iloc[:, 0].tolist()
pos = {i: k for k, i in enumerate(ids)}
rows = np.array([pos[i] for i in t.ID])
X = np.asarray(E[rows][:, :4], np.float64)  # 500 x 4 views x 20 cells x 272
meta = json.loads(open(f"{stem}_meta.json").read())
lay = eval(meta["cell_layout"]) if isinstance(meta["cell_layout"], str) else meta["cell_layout"]  # noqa: S307 (own file)
blocks = [(b["stage"], b["pool"], b["start"], b["end"]) for b in lay["blocks"]]
print("cell layout grids", lay["grids"], "stages", lay["stages"], "blocks", [(s, p, a, e) for s, p, a, e in blocks])


def slog(a):
    return np.sign(a) * np.log1p(np.abs(a))


feats = {}
for gname, sl in (("2x2", slice(0, 4)), ("4x4", slice(4, 20))):
    sd = slog(X[:, :, sl].std(2).mean(1))
    rg = slog((X[:, :, sl].max(2) - X[:, :, sl].min(2)).mean(1))
    feats[f"{gname} std blockmeans"] = np.column_stack([sd[:, a:e].mean(1) for _, _, a, e in blocks])
    feats[f"{gname} range blockmeans"] = np.column_stack([rg[:, a:e].mean(1) for _, _, a, e in blocks])
    feats[f"{gname} std full"] = sd
    feats[f"{gname} std blockmeans s0-1"] = np.column_stack([sd[:, a:e].mean(1) for s, _, a, e in blocks if s <= 1])
feats["4x4 + 2x2 std blockmeans"] = np.column_stack([feats["2x2 std blockmeans"], feats["4x4 std blockmeans"]])
for nm, F in feats.items():
    pr = probe(F, inner)
    nul = R.perm_null(F, inner, n=20)
    print(R.line(nm, pr, F.shape[1]) + f" | perm null mean {nul.mean():+.3f} max {nul.max():+.3f} p~{(np.sum(nul >= R.gain(pr)) + 1) / 21:.2f}", flush=True)
