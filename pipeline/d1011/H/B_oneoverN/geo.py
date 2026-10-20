"""Shared label-free geometry functions for lens B (d1011/H/B_oneoverN). No label is read anywhere in this module.

geo_feats(ws, pore_labels) -> dict of per-image tessellation statistics computed identically on real clean label maps
(cleanhead ws_C_*) and on simulated in-box anisotropic Poisson-Voronoi maps rendered and re-segmented with the same
src.het_blocks.segment pipeline.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

GRID = np.add.outer(np.arange(256) // 64 * 4, np.arange(256) // 64)
KMAX = 16


def fill_pores(ws, pore_labels):
    """Reassign pore pixels to the nearest non-pore label (pores are rendered on top of grains)."""
    if pore_labels is None or len(pore_labels) == 0:
        return ws
    lp = np.zeros(ws.max() + 1, bool); lp[np.asarray(pore_labels, int)] = True
    m = lp[ws]
    if not m.any():
        return ws
    _, (iy, ix) = ndi.distance_transform_edt(m, return_indices=True)
    out = ws.copy(); out[m] = ws[iy[m], ix[m]]
    return out


def runs_per_line(ws):
    """pieces along each row (axis 1 changes) and each column."""
    rows = 1 + (ws[:, 1:] != ws[:, :-1]).sum(1)
    cols = 1 + (ws[1:, :] != ws[:-1, :]).sum(0)
    return rows.astype(float), cols.astype(float)


def shape_tensor(ws, labels_in):
    """mean of per-cell normalised second-moment tensors (det -> 1) over interior cells; returns r_hat, theta, S."""
    yy, xx = np.indices(ws.shape)
    lab = ws.ravel(); n = np.bincount(lab)
    sx = np.bincount(lab, xx.ravel()); sy = np.bincount(lab, yy.ravel())
    sxx = np.bincount(lab, (xx * xx).ravel()); syy = np.bincount(lab, (yy * yy).ravel()); sxy = np.bincount(lab, (xx * yy).ravel())
    L = np.asarray(labels_in)
    nn = n[L]; mx = sx[L] / nn; my = sy[L] / nn
    cxx = sxx[L] / nn - mx ** 2 + 1 / 12; cyy = syy[L] / nn - my ** 2 + 1 / 12; cxy = sxy[L] / nn - mx * my
    det = np.sqrt(np.maximum(cxx * cyy - cxy ** 2, 1e-9))
    T = np.array([[np.mean(cxx / det), np.mean(cxy / det)], [np.mean(cxy / det), np.mean(cyy / det)]])
    ev, evec = np.linalg.eigh(T)
    r_hat = float(np.sqrt(ev[1] / ev[0]))
    th = float(np.arctan2(evec[1, 1], evec[0, 1]))   # direction of the major axis (x, y) image coords
    # per-cell orientation coherence
    ang = 0.5 * np.arctan2(2 * cxy, cxx - cyy); ecc = np.sqrt((cxx - cyy) ** 2 + 4 * cxy ** 2) / (cxx + cyy)
    S = float(np.abs(np.sum(ecc * np.exp(2j * ang))) / max(np.sum(ecc), 1e-9))
    return r_hat, th, S


def geo_feats(ws):
    ws = np.asarray(ws, np.int64)
    labs, inv = np.unique(ws, return_inverse=True)
    ws = inv.reshape(ws.shape)
    n = np.bincount(ws.ravel()).astype(float)
    nl = len(n)
    border = np.zeros(nl, bool)
    for e in (ws[0], ws[-1], ws[:, 0], ws[:, -1]):
        border[np.unique(e)] = True
    f = {}
    f['N_all'] = nl; f['N_bd'] = int(border.sum()); f['N_in'] = int((~border).sum())
    f['N_eff'] = f['N_in'] + 0.5 * f['N_bd']; f['bd_frac'] = f['N_bd'] / nl
    ai = n[~border]
    f['acv_in'] = float(ai.std() / ai.mean()) if len(ai) > 2 else np.nan
    f['la_sd_in'] = float(np.log(ai).std()) if len(ai) > 2 else np.nan
    f['amean_bd_over_in'] = float(n[border].mean() / ai.mean()) if len(ai) else np.nan
    # het4 as src.het_blocks (no pores here)
    a = n[ws]
    vals = np.array([-2 * np.log((a[GRID == b] ** -0.5).mean()) for b in range(16)])
    f['het4'] = float(vals.std()); f['blkmean4'] = float(vals.mean())
    outer = np.array([b // 4 in (0, 3) or b % 4 in (0, 3) for b in range(16)])
    f['edge_ctr'] = float(vals[outer].mean() - vals[~outer].mean())
    # interior-only het (border-cut cells excluded)
    mi = ~border[ws]
    vi = np.array([-2 * np.log((a[(GRID == b) & mi] ** -0.5).mean()) if ((GRID == b) & mi).sum() >= 50 else np.nan for b in range(16)])
    f['het4_int'] = float(np.nanstd(vi))
    # line piece profiles
    r, c = runs_per_line(ws)
    allm = (r.mean() + c.mean()) / 2
    f['line_allmean'] = float(allm)
    for side, prof in (('top', r[:KMAX]), ('bot', r[::-1][:KMAX]), ('left', c[:KMAX]), ('right', c[::-1][:KMAX])):
        for k in range(KMAX):
            f[f'lp_{side}_{k}'] = float(prof[k] / allm)
    f['ring1_6'] = float(np.mean([r[1:7].mean(), r[-7:-1].mean(), c[1:7].mean(), c[-7:-1].mean()]) / allm)
    f['interior_lines'] = float((r[64:192].mean() + c[64:192].mean()) / 2 / allm)
    r_hat, th, S = shape_tensor(ws, np.where(~border & (n >= 30))[0])
    f['r_hat'] = r_hat; f['theta'] = th; f['S_cell'] = S
    return f


def sim_voronoi(N, r, theta, rng, margin=0.0, size=256, dens=None):
    """N seeds uniform in [-margin, size+margin]^2 (or thinned by a density field), anisotropic metric: cells elongated
    by factor r along direction theta (image x, y coords). Returns the label map of the size x size box and the
    number of seeds that own at least one pixel."""
    lo, hi = -margin, size + margin
    if dens is None:
        pts = rng.uniform(lo, hi, size=(N, 2))
    else:
        pts = []
        while len(pts) < N:
            p = rng.uniform(lo, hi, size=(4 * N, 2))
            keep = rng.uniform(size=len(p)) < dens(p)
            pts.extend(p[keep].tolist())
        pts = np.array(pts[:N])
    u = np.array([np.cos(theta), np.sin(theta)]); v = np.array([-np.sin(theta), np.cos(theta)])
    def tr(P):
        return np.stack([P @ u / r, P @ v], 1)
    yy, xx = np.indices((size, size))
    pix = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], 1)
    tree = cKDTree(tr(pts))
    _, idx = tree.query(tr(pix), k=1)
    ws = idx.reshape(size, size)
    return ws


def render(ws, rng, dark_p=0.2, noise=7.0, bdrop=40.0, blur=1.0):
    """flat-shaded cells (matrix 150, dark 112, per-cell jitter sd 4.5), 1-px boundary lines bdrop below the cell
    grey, Gaussian blur, white noise. Defaults give about -20 grey on the boundary pixel and pixel sd about 7-8,
    as measured on real clean images. Only meant to give the segmentation the same kind of input."""
    nl = ws.max() + 1
    dark = rng.uniform(size=nl) < dark_p
    g = np.where(dark, 112.0, 150.0) + rng.normal(0, 4.5, nl)
    im = g[ws]
    b = np.zeros(ws.shape, bool)
    b[:, 1:] |= ws[:, 1:] != ws[:, :-1]; b[1:, :] |= ws[1:, :] != ws[:-1, :]
    im[b] -= bdrop
    im = ndi.gaussian_filter(im, blur) + rng.normal(0, noise, ws.shape)
    return np.clip(im, 0, 255).astype(np.float32)
