"""Segment clean train images (raw ic_noise<9.5) with the src.het_blocks pipeline (sigma 1.0); save per-grain tables + label maps.
Train images only; labels not read here."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, '/home/claude/microstructure-hardness-prediction')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from skimage import measure
from src.het_blocks import segment, shading_norm
OUT = '/tmp/claude/cleanhead/'
split = sys.argv[1] if len(sys.argv) > 1 else 'train'
def one(i):
    im = np.array(Image.open(f'/home/claude/microstructure-hardness-prediction/data/{split}/{i}.png')).astype(np.float32)
    sm, ws = segment(im)
    rows = []
    for p in measure.regionprops(ws, intensity_image=sm):
        minr, minc, maxr, maxc = p.bbox
        rows.append(dict(label=p.label, area=p.area, cy=p.centroid[0], cx=p.centroid[1], mean=p.intensity_mean,
                         maj=p.axis_major_length, mnr=p.axis_minor_length, ori=p.orientation, perim=p.perimeter,
                         border=minr == 0 or minc == 0 or maxr == 256 or maxc == 256,
                         nb=int((minr == 0) + (minc == 0) + (maxr == 256) + (maxc == 256))))
    g = pd.DataFrame(rows); g['gn'] = shading_norm(g); g['ID'] = i
    return g, ws.astype(np.int16)
if __name__ == '__main__':
    t = pd.read_parquet(OUT + ('tr0.parquet' if split == 'train' else 'te0.parquet'))
    ids = t.ID[t.g == 'C'].tolist()
    res = Parallel(n_jobs=int(os.environ.get('NJ', 2)))(delayed(one)(i) for i in ids)
    G = pd.concat([r[0] for r in res], ignore_index=True)
    G.to_parquet(OUT + f'grains_C_{split}.parquet')
    np.savez_compressed(OUT + f'ws_C_{split}.npz', **{i: r[1] for i, r in zip(ids, res)})
    print(G.shape, G.ID.nunique())
