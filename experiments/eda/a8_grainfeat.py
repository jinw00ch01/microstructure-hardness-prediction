"""A8: per-grain descriptors aggregated to image level (candidate hidden signals).
Per grain (cached watershed, clean images segment well): relative brightness (median rn), absolute level,
interior texture (std of raw minus image noise), aspect ratio, orientation vs the image axis, neighbours,
dark-dark contacts, dark clusters (connected through shared boundaries), largest grains, pore contacts.
Usage: python a8_grainfeat.py CACHE_DIR split"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage import measure

from eda_common import DATA_DIR

cache = Path(sys.argv[1])
split = sys.argv[2]
ids = pd.read_csv(DATA_DIR / ("train.csv" if split == "train" else "sample_submission.csv")).ID.tolist()
rn_all = np.load(cache / f"rn_{split}.npy", mmap_mode="r")
ws_all = np.load(cache / f"ws_{split}.npy", mmap_mode="r")
sig = pd.read_csv(cache / f"sig_{split}.csv").sigma.values
RB = [0.0, 0.75, 0.80, 0.85, 0.88, 0.91, 0.94, 0.97, 1.00, 1.03, 1.06, 9.0]


def wm(v, w):
    return float(np.sum(v * w) / np.sum(w)) if np.sum(w) > 0 else np.nan


rows = []
for k, i in enumerate(ids):
    raw = cv2.imread(str(DATA_DIR / split / f"{i}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    x = rn_all[k].astype(np.float32)
    w = np.asarray(ws_all[k]).astype(np.int32)
    s1 = cv2.GaussianBlur(x, (0, 0), 1.0)
    pm = ndi.binary_opening(s1 < 0.62, structure=np.ones((3, 3)))
    line = w == 0
    inner = w.copy()
    inner[ndi.binary_dilation(line, iterations=2)] = 0
    nl = int(w.max())
    lab = np.arange(1, nl + 1)
    area = np.asarray(ndi.sum(np.ones_like(x), w, lab))
    a_in = np.asarray(ndi.sum(np.ones_like(x), inner, lab))
    med = np.asarray(ndi.median(x, inner, lab))
    lvl = np.asarray(ndi.median(raw, inner, lab))
    # interior texture: std of raw minus local mean (3x3 removes nothing of grain level), compare with noise
    hp = raw - cv2.blur(raw, (5, 5))
    tex = np.sqrt(np.asarray(ndi.mean(hp ** 2, inner, lab)))
    pf = np.asarray(ndi.mean(pm.astype(np.float32), w, lab))
    pore_touch = np.asarray(ndi.maximum(ndi.binary_dilation(pm, iterations=2).astype(np.float32), w, lab)) > 0
    ok = (a_in >= 8) & (pf <= 0.5)
    f = {"ID": i}
    if ok.sum() < 3:
        rows.append(f)
        continue
    props = {p.label: p for p in measure.regionprops(w)}
    maj = np.array([props[l].axis_major_length if l in props else 0 for l in lab])
    mi = np.array([max(props[l].axis_minor_length, 1.0) if l in props else 1 for l in lab])
    ori = np.array([props[l].orientation if l in props else 0 for l in lab])
    asp = maj / mi
    A = area[ok]
    # dominant axis from area-weighted second-moment tensors (normalised per grain)
    th = ori[ok]
    R = np.sum(A * np.exp(2j * th)) / A.sum()
    th0 = 0.5 * np.angle(R)
    align = np.cos(2 * (th - th0))
    # tensor-average strain estimate (Shimamoto-Ikeda style): mean of normalised shape tensors
    la = np.log(np.clip(asp[ok], 1, 50))
    Txx = np.mean(np.cosh(la) + np.sinh(la) * np.cos(2 * (th - th0)))
    Tyy = np.mean(np.cosh(la) - np.sinh(la) * np.cos(2 * (th - th0)))
    Txy = np.mean(np.sinh(la) * np.sin(2 * (th - th0)))
    ev = np.linalg.eigvalsh(np.array([[Txx, Txy], [Txy, Tyy]]))
    f["gf_strain_tensor"] = float(0.5 * np.log(ev[1] / max(ev[0], 1e-6)))
    f["gf_logasp_mean"] = float(la.mean())
    f["gf_logasp_wmean"] = wm(la, A)
    f["gf_align_R"] = float(np.abs(R))
    f["gf_align_wmean"] = wm(align, A)
    m = med[ok]
    dk = m < 0.91
    f["gf_n"] = int(ok.sum())
    for a_, b_ in zip(RB[:-1], RB[1:]):
        f[f"gf_rb_{a_:.2f}"] = float(A[(m >= a_) & (m < b_)].sum() / A.sum())
    f["gf_mx_mean"] = wm(m[~dk], A[~dk])
    f["gf_mx_sd"] = float(np.sqrt(wm((m[~dk] - f["gf_mx_mean"]) ** 2, A[~dk]))) if (~dk).sum() > 1 else np.nan
    f["gf_mx_skew"] = float(wm((m[~dk] - f["gf_mx_mean"]) ** 3, A[~dk]) / (f["gf_mx_sd"] ** 3 + 1e-9)) if (~dk).sum() > 2 else np.nan
    f["gf_dk_mean"] = wm(m[dk], A[dk]) if dk.any() else np.nan
    f["gf_dk_sd"] = float(np.sqrt(wm((m[dk] - f["gf_dk_mean"]) ** 2, A[dk]))) if dk.sum() > 1 else np.nan
    f["gf_lvl_mx"] = wm(lvl[ok][~dk], A[~dk])
    f["gf_lvl_dk"] = wm(lvl[ok][dk], A[dk]) if dk.any() else np.nan
    f["gf_lvl_contrast"] = f["gf_lvl_dk"] / f["gf_lvl_mx"] if dk.any() else np.nan
    tx = tex[ok] / max(sig[k], 1e-3)
    f["gf_tex_mx"] = wm(tx[~dk], A[~dk])
    f["gf_tex_dk"] = wm(tx[dk], A[dk]) if dk.any() else np.nan
    f["gf_tex_sd"] = float(np.std(tx))
    # size stats
    f["gf_amax_frac"] = float(A.max() / A.sum())
    f["gf_amax_dk_frac"] = float(A[dk].max() / A.sum()) if dk.any() else 0.0
    f["gf_area_cv"] = float(A.std() / A.mean())
    f["gf_dk_area_ratio"] = float(A[dk].mean() / A[~dk].mean()) if dk.any() and (~dk).any() else np.nan
    f["gf_dk_asp_ratio"] = float(np.exp(la[dk].mean() - la[~dk].mean())) if dk.any() and (~dk).any() else np.nan
    # pores touching dark vs matrix
    pt = pore_touch[ok]
    f["gf_pore_touch_dk"] = float(A[pt & dk].sum() / A.sum())
    f["gf_pore_touch_mx"] = float(A[pt & ~dk].sum() / A.sum())
    # adjacency (labels across watershed lines, 2-px offsets)
    pairs = []
    for a1, a2 in ((w[:, :-2], w[:, 2:]), (w[:-2, :], w[2:, :]), (w[:-2, :-2], w[2:, 2:]), (w[:-2, 2:], w[2:, :-2])):
        mm = (a1 > 0) & (a2 > 0) & (a1 != a2)
        pairs.append(np.stack([a1[mm], a2[mm]], 1))
    P = np.concatenate(pairs)
    P = np.unique(np.sort(P, 1), axis=0) - 1
    okl = ok.copy()
    dkl = np.zeros(nl, bool)
    dkl[np.where(ok)[0][dk]] = True
    P = P[okl[P[:, 0]] & okl[P[:, 1]]]
    nb = np.bincount(P.ravel(), minlength=nl)[ok]
    f["gf_nb_mean"] = float(nb.mean())
    both = dkl[P[:, 0]] & dkl[P[:, 1]]
    one = dkl[P[:, 0]] ^ dkl[P[:, 1]]
    pdk = dk.mean()
    f["gf_dd_contact"] = float(both.sum() / max(len(P), 1))
    f["gf_dd_excess"] = float(both.sum() / max(len(P), 1) - pdk ** 2)
    f["gf_dm_contact"] = float(one.sum() / max(len(P), 1))
    if dk.sum() >= 1:
        Pd = P[both]
        G = coo_matrix((np.ones(len(Pd)), (Pd[:, 0], Pd[:, 1])), shape=(nl, nl))
        nc, cl = connected_components(G, directed=False)
        cl_dk = cl[dkl]
        a_full = area.copy()
        csz = np.bincount(cl_dk, weights=a_full[dkl])
        csz = csz[csz > 0]
        f["gf_dk_ncluster"] = int(len(csz))
        f["gf_dk_maxcluster_frac"] = float(csz.max() / A.sum())
        f["gf_dk_maxcluster_share"] = float(csz.max() / csz.sum())
        f["gf_dk_cluster_wmean"] = float((csz ** 2).sum() / csz.sum() / A.sum())
        # spanning: does any dark cluster touch two opposite borders?
        span = 0
        bt = {"t": np.unique(w[0]), "b": np.unique(w[-1]), "l": np.unique(w[:, 0]), "r": np.unique(w[:, -1])}
        for c_ in np.unique(cl_dk):
            members = set((np.where(dkl & (cl == c_))[0] + 1).tolist())
            tb = members & set(bt["t"].tolist()) and members & set(bt["b"].tolist())
            lr = members & set(bt["l"].tolist()) and members & set(bt["r"].tolist())
            span = max(span, int(bool(tb) or bool(lr)))
        f["gf_dk_span"] = span
    else:
        f["gf_dk_ncluster"] = 0
        f["gf_dk_maxcluster_frac"] = f["gf_dk_maxcluster_share"] = f["gf_dk_cluster_wmean"] = 0.0
        f["gf_dk_span"] = 0
    rows.append(f)
pd.DataFrame(rows).to_parquet(cache / f"gf_{split}.parquet")
print("done", split, len(rows))
