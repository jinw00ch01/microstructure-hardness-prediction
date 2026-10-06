"""A3: is the label tied to a fixed location (indent-like local sample)?
Correlation map between blend OOF residual and the local dark-phase / pore fraction at each position,
after removing the image-global fraction. Permutation null for the max |corr| over the map."""
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from eda_common import OUT, train_table

cache = Path(sys.argv[1])
t = train_table()
rn = np.load(cache / "rn_train.npy").astype(np.float32)
n = len(t)
r = t.resid.values
coarse = (t.ic_acg_len50_gm >= t.ic_acg_len50_gm.quantile(2 / 3)).values

dark = np.zeros((n, 256, 256), np.float32)
pore = np.zeros((n, 256, 256), np.float32)
for k in range(n):
    s1 = cv2.GaussianBlur(rn[k], (0, 0), 1.0)
    s2 = cv2.GaussianBlur(rn[k], (0, 0), 2.0)
    pm = ndi.binary_opening(s1 < 0.68, structure=np.ones((3, 3)))
    pm_ex = ndi.binary_dilation(pm, iterations=3)
    pore[k] = pm
    dark[k] = ((s2 < 0.91) & ~pm_ex).astype(np.float32)


def block(x, b):
    m = x.shape[1] // b
    return x.reshape(len(x), m, b, m, b).mean(axis=(2, 4))


def corr_map(M, rr):
    Mc = M - M.mean(0)
    rc = rr - rr.mean()
    num = np.tensordot(rc, Mc, axes=(0, 0))
    den = np.sqrt((rc ** 2).sum() * (Mc ** 2).sum(0)) + 1e-12
    return num / den


rng = np.random.default_rng(0)
lines = []
for name, M in (("dark", dark), ("pore", pore)):
    for b in (16, 32, 64):
        B = block(M, b)
        Bl = B - B.mean(axis=(1, 2), keepdims=True)  # local minus global
        for sub, msk in (("all", np.ones(n, bool)), ("coarse", coarse)):
            cm = corr_map(Bl[msk], r[msk])
            null = [np.abs(corr_map(Bl[msk], rng.permutation(r[msk]))).max() for _ in range(200)]
            iy, ix = np.unravel_index(np.abs(cm).argmax(), cm.shape)
            lines.append(f"{name:5s} block{b:3d} {sub:6s}: max|corr| {np.abs(cm).max():.3f} at cell ({iy},{ix}) "
                         f"null95 {np.quantile(null, 0.95):.3f}  p={np.mean(np.array(null) >= np.abs(cm).max()):.3f}; "
                         f"center cells corr {cm[cm.shape[0] // 2 - 1:cm.shape[0] // 2 + 1, cm.shape[1] // 2 - 1:cm.shape[1] // 2 + 1].mean():+.3f}")
            print(lines[-1], flush=True)
            if b == 32:
                np.save(OUT / f"corrmap_{name}_{sub}_b32.npy", cm)

# radial windows around the centre
yy, xx = np.mgrid[:256, :256]
rad = np.hypot(yy - 127.5, xx - 127.5)
gl_dark = dark.mean(axis=(1, 2))
for R in (4, 8, 16, 32, 64):
    w = (rad <= R).astype(np.float32)
    loc = (dark * w).sum(axis=(1, 2)) / w.sum()
    res_loc = loc - np.polyval(np.polyfit(gl_dark, loc, 1), gl_dark)
    for sub, msk in (("all", np.ones(n, bool)), ("coarse", coarse)):
        c = np.corrcoef(res_loc[msk], r[msk])[0, 1]
        print(f"centre disk R={R:2d} {sub:6s}: corr(resid, local dark - fit(global)) = {c:+.3f}")
np.save(cache / "dark_train.npy", dark.astype(np.uint8))
np.save(cache / "pore_train.npy", pore.astype(np.uint8))
