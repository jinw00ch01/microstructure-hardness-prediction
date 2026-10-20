"""Per-image label-free statistics from the clean segmentation (grain tables + label maps)."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture
OUT = '/tmp/claude/cleanhead/'
split = sys.argv[1] if len(sys.argv) > 1 else 'train'
GRID = np.add.outer(np.arange(256) // 64 * 4, np.arange(256) // 64)

def phase_thr(gp):
    big = gp[gp.area > 25]; thr = 0.90
    try:
        gm = GaussianMixture(2, random_state=0).fit(big.gn.values[:, None])
        mu = np.sort(gm.means_.ravel())
        if mu[1] - mu[0] > 0.08:
            xs = np.linspace(mu[0], mu[1], 200); p = gm.predict_proba(xs[:, None]); lo = np.argmin(gm.means_.ravel())
            thr = xs[np.argmin(np.abs(p[:, lo] - 0.5))]
    except Exception:
        pass
    return thr

def feats(g, ws):
    f = {}
    pore = ((g.gn < 0.68) & (g.area < 600)).values
    gp = g[~pore].copy()
    thr = phase_thr(gp); dark = (gp.gn < thr).values
    A = 65536.0
    area = gp.area.values.astype(float); bd = gp.border.values; inner = ~bd
    f['thr'] = thr
    f['N_all'] = len(gp); f['N_in'] = inner.sum(); f['N_eff'] = inner.sum() + 0.5 * bd.sum()
    f['N_bd'] = bd.sum()
    f['Anp'] = area.sum()
    f['pore_frac'] = g.area.values[pore].sum() / A; f['pore_n'] = pore.sum()
    f['fd_area'] = area[dark].sum() / area.sum(); f['fd_num'] = dark.mean(); f['fd_num_in'] = dark[inner].mean()
    # size definitions
    f['amean_all'] = area.mean(); f['amean_in'] = area[inner].mean(); f['amed_in'] = np.median(area[inner])
    d_all = np.sqrt(4 * area / np.pi); d_in = d_all[inner]
    f['d_num_all'] = d_all.mean(); f['d_num_in'] = d_in.mean()
    f['d_aw_in'] = (d_in * area[inner]).sum() / area[inner].sum()
    f['hp_num_all'] = (d_all ** -0.5).mean(); f['hp_num_in'] = (d_in ** -0.5).mean()
    f['hp_aw_all'] = ((d_all ** -0.5) * area).sum() / area.sum(); f['hp_aw_in'] = ((d_in ** -0.5) * area[inner]).sum() / area[inner].sum()
    # lineal intercepts on the label map (pores count as grains here; boundaries = label changes)
    h = (ws[:, 1:] != ws[:, :-1]).sum(); v = (ws[1:, :] != ws[:-1, :]).sum()
    f['Lint_x'] = 256 * 256 / (h + 256); f['Lint_y'] = 256 * 256 / (v + 256); f['Lint_gm'] = np.sqrt(f['Lint_x'] * f['Lint_y'])
    # per phase
    f['N_dark_in'] = (dark & inner).sum(); f['N_mat_in'] = (~dark & inner).sum()
    f['d_dark'] = d_in[dark[inner]].mean() if (dark & inner).sum() else np.nan
    f['d_mat'] = d_in[~dark[inner]].mean()
    f['a_dark_all'] = area[dark].mean() if dark.sum() else np.nan; f['a_mat_all'] = area[~dark].mean()
    # shape
    asp = gp.maj.values / np.maximum(gp.mnr.values, 1)
    ai = area[inner]
    f['logasp_aw'] = (np.log(asp[inner]) * ai).sum() / ai.sum(); f['logasp_num'] = np.log(asp[inner]).mean()
    th = gp.ori.values; w = area * (asp - 1); z = (w * np.exp(2j * th)).sum() / w.sum(); f['S_align'] = np.abs(z)
    # size spread
    f['la_sd_in'] = np.log(ai).std(); f['la_sd_all'] = np.log(area).std()
    f['acv_in'] = ai.std() / ai.mean(); f['acv_all'] = area.std() / area.mean()
    f['dcv_in'] = d_in.std() / d_in.mean()
    # het4 (src.het_blocks definition) + variants
    lab_pore = np.zeros(ws.max() + 1, bool); lab_pore[g.label.values[pore]] = True
    a = np.maximum(np.bincount(ws.ravel()).astype(float)[ws], 1); m = ~lab_pore[ws]
    vals, cnt = [], []
    for b in range(16):
        s = (GRID == b) & m
        if s.sum() >= 50: vals.append(-2 * np.log((a[s] ** -0.5).mean()))
    f['het4'] = float(np.std(vals)); f['blkmean4'] = float(np.mean(vals))
    # centroid count CV on 4x4
    c4 = np.histogram2d(gp.cy, gp.cx, bins=4, range=[[0, 256], [0, 256]])[0]; f['cnt_cv4'] = c4.std() / c4.mean()
    # per-phase blockwise dark fraction sd
    dk_lab = np.zeros(ws.max() + 1, bool); dk_lab[gp.label.values[dark]] = True
    fdb = [dk_lab[ws][(GRID == b) & m].mean() for b in range(16)]
    f['fd_blk_sd'] = float(np.std(fdb))
    return f

if __name__ == '__main__':
    G = pd.read_parquet(OUT + f'grains_C_{split}.parquet'); W = np.load(OUT + f'ws_C_{split}.npz')
    rows = {}
    for i, g in G.groupby('ID'):
        rows[i] = feats(g.reset_index(drop=True), W[i].astype(np.int32))
    F = pd.DataFrame(rows).T.astype(float); F.index.name = 'ID'
    F.to_parquet(OUT + f'feats_C_{split}.parquet'); print(F.shape)
    if split == 'train':
        hb = pd.read_parquet('/home/claude/microstructure-hardness-prediction/data/het_blocks_train.parquet').set_index('ID')
        print('het4 max abs diff', (F.het4 - hb.loc[F.index, 'het4']).abs().max(), 'N_eff', (F.N_eff - hb.loc[F.index, 'N_eff']).abs().max())
    print(F.describe().T[['mean', 'std', 'min', 'max']].round(3).to_string())
