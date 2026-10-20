"""s14 (LABEL-FREE feature build, train + test clean rows, one image at a time): the origin-grid het term and the size
block re-measured on a pre-declared family of segmentations.
Segmentation: src.het_blocks-style sato ridge + otsu*tf + watershed (as d1010/C segalt and d1011/H/B b13), sigma in
SIGMAS, tf 0.6; pores (gn < 0.68 & area < 600, shading-normalised) are refilled by the nearest grain (geo.fill_pores),
because pores are rendered on top of grains. Per image and sigma:
  het4  = sd over the origin 4x4 grid of -2 log(mean over block px of a^-1/2), a = full area of the pixel's grain
  het4q = same with a^-1/4 (p = 0.25)
  N_eff, bd_frac, N_all (geo.geo_feats definitions)
Writes s14_{split}.parquet (long: ID, sig, ...). Nothing is fitted; test images only pass through this function."""
import os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation, measure
REPO = '/home/claude/microstructure-hardness-prediction'
HB = '/mnt/project-files/work/hardness-cache/scripts/d1011/H/B_oneoverN'
sys.path.insert(0, HB); sys.path.insert(0, REPO)
from geo import fill_pores  # noqa
from src.het_blocks import shading_norm  # noqa
CH = '/mnt/project-files/work/hardness-cache/scripts/d1009/J/cleanhead/'
HERE = '/mnt/project-files/work/hardness-cache/scripts/d1011/bold/L1_formula/'
SIGMAS = (0.6, 0.7, 0.8, 0.9, 1.0)
GRID = np.add.outer(np.arange(256) // 64 * 4, np.arange(256) // 64)


def seg(im, sig, tf=0.6):
    sm = ndi.gaussian_filter(im, sig); r = filters.sato(sm, sigmas=(sig, 1.5 * sig), black_ridges=True, mode="reflect")
    interior = ndi.binary_opening(~(r > filters.threshold_otsu(r) * tf), structure=morphology.disk(1))
    lab, _ = ndi.label(interior); small = np.bincount(lab.ravel()) < 6; small[0] = False; lab[small[lab]] = 0
    lab, _, _ = segmentation.relabel_sequential(lab); return sm, segmentation.watershed(r, markers=lab, connectivity=1)


def pores_of(sm, ws):
    rows = [dict(label=p.label, area=p.area, cy=p.centroid[0], cx=p.centroid[1], mean=p.intensity_mean)
            for p in measure.regionprops(ws, intensity_image=sm)]
    g = pd.DataFrame(rows); g['gn'] = shading_norm(g)
    return g.label.values[((g.gn < 0.68) & (g.area < 600)).values]


def feats(ws):
    labs, inv = np.unique(ws, return_inverse=True); ws = inv.reshape(ws.shape)
    n = np.bincount(ws.ravel()).astype(float)
    border = np.zeros(len(n), bool)
    for e in (ws[0], ws[-1], ws[:, 0], ws[:, -1]):
        border[np.unique(e)] = True
    a = n[ws]
    v2 = np.array([-2 * np.log((a[GRID == b] ** -0.5).mean()) for b in range(16)])
    v4 = np.array([-4 * np.log((a[GRID == b] ** -0.25).mean()) for b in range(16)])
    return dict(het4=v2.std(), het4q=v4.std(), N_all=len(n), N_bd=int(border.sum()),
                N_eff=(~border).sum() + 0.5 * border.sum(), bd_frac=border.sum() / len(n))


def one(split, i):
    im = np.array(Image.open(f'{REPO}/data/{split}/{i}.png')).astype(np.float32)
    out = []
    for sig in SIGMAS:
        sm, ws = seg(im, sig)
        f = feats(fill_pores(ws.astype(np.int64), pores_of(sm, ws)))
        f.update(ID=i, sig=sig); out.append(f)
    return out


if __name__ == '__main__':
    split = sys.argv[1]
    ids = sorted(pd.read_parquet(CH + f'feats_C_{split}.parquet').index)
    nj = int(os.environ.get('NJ', 2))
    res = Parallel(n_jobs=nj)(delayed(one)(split, i) for i in ids)
    L = pd.DataFrame([r for rr in res for r in rr])
    L.to_parquet(HERE + f's14_{split}.parquet')
    print(split, L.shape, L.groupby('sig')[['het4', 'het4q', 'N_eff', 'bd_frac']].mean().round(4).to_string())
