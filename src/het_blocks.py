"""Within-image grain-size heterogeneity on a 4x4 grid of 64-px blocks (het4) and the effective grain count.

Only for images with raw ic_noise < 9.5 (features_v3), where a sigma-1.0 ridge segmentation still finds the grains.
Per block: -2 log(mean over its non-pore pixels of a^-1/2), a = visible area of the pixel's grain (border-cut
grains are not corrected); het4 = sd over the 16 blocks. N_eff = interior grains + 0.5 * border grains (pores
excluded). Feature extraction only: every value is computed from one image, nothing is fitted across images.
The label-free GMM phase split of the original exploration script is not needed for these two columns.

  python -m src.het_blocks            # -> data/het_blocks_{train,test}.parquet (ID, het4, N_eff, ic_noise)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation

from .common import DATA_DIR

NOISE_MAX = 9.5
GRID = np.add.outer(np.arange(256) // 64 * 4, np.arange(256) // 64)   # block id 0..15


def segment(im, sig=1.0):
    sm = ndi.gaussian_filter(im, sig)
    r = filters.sato(sm, sigmas=(sig, 1.5 * sig), black_ridges=True, mode="reflect")
    interior = ndi.binary_opening(~(r > filters.threshold_otsu(r) * 0.6), structure=morphology.disk(1))
    lab, _ = ndi.label(interior)
    small = np.bincount(lab.ravel()) < 6
    small[0] = False
    lab[small[lab]] = 0
    lab, _, _ = segmentation.relabel_sequential(lab)
    return sm, segmentation.watershed(r, markers=lab, connectivity=1)


def grain_table(sm, ws):
    rows = []
    for p in measure.regionprops(ws, intensity_image=sm):
        minr, minc, maxr, maxc = p.bbox
        rows.append(dict(label=p.label, area=p.area, cy=p.centroid[0], cx=p.centroid[1], mean=p.intensity_mean,
                         border=minr == 0 or minc == 0 or maxr == ws.shape[0] or maxc == ws.shape[1]))
    return pd.DataFrame(rows)


def shading_norm(g):
    """Grain mean grey over a robust quadratic shading surface fitted to the bright (matrix) grains of this image."""
    x, y = g.cx.values / 256 - 0.5, g.cy.values / 256 - 0.5
    X = np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], 1)
    w, v = g.area.values.astype(float), g["mean"].values.astype(float)
    keep = v > 0.9 * np.median(v[g.area.values > 30])
    for _ in range(4):
        W = np.sqrt(w[keep])
        beta, *_ = np.linalg.lstsq(X[keep] * W[:, None], v[keep] * W, rcond=None)
        fit = X @ beta
        keep = v / fit > 0.93
    return v / fit


def one(path):
    im = np.array(Image.open(path)).astype(np.float32)
    sm, ws = segment(im)
    g = grain_table(sm, ws)
    pore = ((shading_norm(g) < 0.68) & (g.area < 600)).values
    gp = g[~pore]
    n_eff = (~gp.border).sum() + 0.5 * gp.border.sum()
    is_pore = np.zeros(ws.max() + 1, bool)
    is_pore[g.label.values[pore]] = True
    a = np.maximum(np.bincount(ws.ravel()).astype(float)[ws], 1)
    m = ~is_pore[ws]
    vals = [-2 * np.log((a[s] ** -0.5).mean()) for b in range(16) if (s := (GRID == b) & m).sum() >= 50]
    return float(np.std(vals)), float(n_eff)


def main():
    v3 = pd.read_parquet(DATA_DIR / "features_v3.parquet").set_index("ID")
    for split, csv in (("train", "train.csv"), ("test", "sample_submission.csv")):
        ids = [i for i in pd.read_csv(DATA_DIR / csv).ID if v3.loc[i, "ic_noise"] < NOISE_MAX]
        res = Parallel(n_jobs=4)(delayed(one)(DATA_DIR / split / f"{i}.png") for i in ids)
        out = pd.DataFrame(res, columns=["het4", "N_eff"], index=pd.Index(ids, name="ID")).reset_index()
        out["ic_noise"] = v3.loc[ids, "ic_noise"].values
        out.to_parquet(DATA_DIR / f"het_blocks_{split}.parquet", index=False)
        print(split, out.shape, out[["het4", "N_eff"]].describe().loc[["mean", "std"]].round(3).to_dict())


if __name__ == "__main__":
    main()
