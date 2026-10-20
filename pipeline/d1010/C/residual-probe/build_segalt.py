"""d1010 C residual-probe: alternative-segmentation contrasts for the clean train (267) and clean test (519) rows.

Segmentations exactly as cleanhead seg_alt.py (sato ridge, otsu * tf, watershed; pores gn < 0.68 & area < 600 removed)
for (sigma, tf) in (0.8, 0.6), (1.3, 0.6), (1.0, 0.45), (1.0, 0.8). Label-free, one image at a time.
Candidates (PREREG addendum):
  seg_dbdf  = mean(bdf[s0.8,t0.6], bdf[s1.0,t0.45]) - mean(bdf[s1.3,t0.6], bdf[s1.0,t0.8])   bdf = N_bd / N_all
  seg_dlN   = same contrast of log N_all
  seg_dring = same contrast of log ring
  python build_segalt.py   -> segalt_{train,test}.parquet (long, raw) and feats_segc_{train,test}.parquet (ID index)
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation

REPO = '/home/claude/microstructure-hardness-prediction'
sys.path.insert(0, REPO)
from src.het_blocks import shading_norm  # noqa: E402

CH = '/mnt/project-files/work/hardness-cache/scripts/d1009/J/cleanhead/'
OUT = os.path.dirname(os.path.abspath(__file__)) + '/'
VARIANTS = [(0.8, 0.6), (1.3, 0.6), (1.0, 0.45), (1.0, 0.8)]
FINE, COARSE = [(0.8, 0.6), (1.0, 0.45)], [(1.3, 0.6), (1.0, 0.8)]


def seg(im, sig, tf):
    sm = ndi.gaussian_filter(im, sig); r = filters.sato(sm, sigmas=(sig, 1.5 * sig), black_ridges=True, mode="reflect")
    interior = ndi.binary_opening(~(r > filters.threshold_otsu(r) * tf), structure=morphology.disk(1))
    lab, _ = ndi.label(interior); small = np.bincount(lab.ravel()) < 6; small[0] = False; lab[small[lab]] = 0
    lab, _, _ = segmentation.relabel_sequential(lab); return sm, segmentation.watershed(r, markers=lab, connectivity=1)


def one(split, i):
    im = np.array(Image.open(f'{REPO}/data/{split}/{i}.png')).astype(np.float32)
    out = []
    for sig, tf in VARIANTS:
        sm, ws = seg(im, sig, tf); rows = []
        for p in measure.regionprops(ws, intensity_image=sm):
            a, b, c, e = p.bbox
            rows.append(dict(area=p.area, cy=p.centroid[0], cx=p.centroid[1], mean=p.intensity_mean,
                             border=a == 0 or b == 0 or c == 256 or e == 256))
        g = pd.DataFrame(rows); g['gn'] = shading_norm(g); g = g[~((g.gn < 0.68) & (g.area < 600))]
        R = 1 + (ws[:, 1:] != ws[:, :-1]).sum(1); C = 1 + (ws[1:, :] != ws[:-1, :]).sum(0)
        ring = np.mean([(R[k] + R[255 - k] + C[k] + C[255 - k]) / 4 for k in range(1, 5)]); allm = (R.mean() + C.mean()) / 2
        out.append(dict(ID=i, N_all=len(g), N_bd=int(g.border.sum()), N_in=int((~g.border).sum()), ring=ring, allm=allm,
                        sig=sig, tf=tf))
    return out


def contrasts(L):
    L = L.copy()
    L['bdf'] = L.N_bd / L.N_all; L['lN'] = np.log(L.N_all); L['lring'] = np.log(L.ring)
    def m(var, vs):
        parts = [L[(L.sig == s) & (L.tf == t)].set_index('ID')[var] for s, t in vs]
        return sum(parts) / len(parts)
    D = pd.DataFrame({'seg_dbdf': m('bdf', FINE) - m('bdf', COARSE), 'seg_dlN': m('lN', FINE) - m('lN', COARSE),
                      'seg_dring': m('lring', FINE) - m('lring', COARSE)})
    D.index.name = 'ID'
    return D


if __name__ == '__main__':
    for split in ('train', 'test'):
        ids = list(pd.read_parquet(CH + f'feats_C_{split}.parquet').index)
        res = Parallel(n_jobs=3)(delayed(one)(split, i) for i in ids)
        L = pd.DataFrame([r for rr in res for r in rr])
        L.to_parquet(OUT + f'segalt_{split}.parquet')
        if split == 'train':   # reproduction check against the cleanhead table
            ref = pd.read_parquet(CH + 'segalt_train.parquet')
            mg = L.merge(ref, on=['ID', 'sig', 'tf'], suffixes=('', '_ref'))
            assert len(mg) == len(ref) == 1068, (len(mg), len(ref))
            for c in ('N_all', 'N_bd', 'N_in', 'ring', 'allm'):
                print('repro', c, 'max abs diff', float(np.abs(mg[c] - mg[c + '_ref']).max()))
        D = contrasts(L).loc[ids]
        assert np.isfinite(D.values).all()
        D.to_parquet(OUT + f'feats_segc_{split}.parquet')
        print(split, D.shape, D.describe().loc[['mean', 'std', 'min', 'max']].round(4).to_dict())
