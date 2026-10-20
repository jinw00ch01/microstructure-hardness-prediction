"""Label-free robustness: N_bd/N_all from alternative segmentations (sigma 0.8, 1.3; threshold factor 0.5/0.7)."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, '/home/claude/microstructure-hardness-prediction')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation
from src.het_blocks import shading_norm
def seg(im, sig, tf):
    sm = ndi.gaussian_filter(im, sig); r = filters.sato(sm, sigmas=(sig, 1.5 * sig), black_ridges=True, mode="reflect")
    interior = ndi.binary_opening(~(r > filters.threshold_otsu(r) * tf), structure=morphology.disk(1))
    lab, _ = ndi.label(interior); small = np.bincount(lab.ravel()) < 6; small[0] = False; lab[small[lab]] = 0
    lab, _, _ = segmentation.relabel_sequential(lab); return sm, segmentation.watershed(r, markers=lab, connectivity=1)
def one(i, sig, tf):
    im = np.array(Image.open(f'/home/claude/microstructure-hardness-prediction/data/train/{i}.png')).astype(np.float32)
    sm, ws = seg(im, sig, tf); rows = []
    for p in measure.regionprops(ws, intensity_image=sm):
        a, b, c, e = p.bbox; rows.append(dict(area=p.area, cy=p.centroid[0], cx=p.centroid[1], mean=p.intensity_mean, border=a == 0 or b == 0 or c == 256 or e == 256))
    g = pd.DataFrame(rows); g['gn'] = shading_norm(g); g = g[~((g.gn < 0.68) & (g.area < 600))]
    R = 1 + (ws[:, 1:] != ws[:, :-1]).sum(1); C = 1 + (ws[1:, :] != ws[:-1, :]).sum(0)
    ring = np.mean([(R[k] + R[255 - k] + C[k] + C[255 - k]) / 4 for k in range(1, 5)]); allm = (R.mean() + C.mean()) / 2
    return dict(ID=i, N_all=len(g), N_bd=g.border.sum(), N_in=(~g.border).sum(), ring=ring, allm=allm)
tr = pd.read_parquet('/tmp/claude/cleanhead/tr0.parquet'); ids = tr.ID[tr.g == 'C'].tolist()
out = []
for sig, tf in [(0.8, 0.6), (1.3, 0.6), (1.0, 0.45), (1.0, 0.8)]:
    r = pd.DataFrame(Parallel(n_jobs=3)(delayed(one)(i, sig, tf) for i in ids)); r['sig'] = sig; r['tf'] = tf; out.append(r)
pd.concat(out).to_parquet('/tmp/claude/cleanhead/segalt_train.parquet'); print('done')
