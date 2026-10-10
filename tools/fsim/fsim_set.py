"""fsim_set: training items for the localized het CNN drawn from the degrade_sim ('fsim') world.

FsimRenderSet(src.het_cnn.RenderSet): with probability P_FSIM an item is a degrade_sim copy of the clean prep image j
(prior 'G1hi' with probability P_G1HI, else 'midLF'), with the UNCHANGED prep targets of j (16 block values b, het4,
log N_eff, 16x16 cell targets, extra targets) and the same D4 op handling as RenderSet's non-augmented path. Otherwise
the item is exactly src.het_cnn.RenderSet's (cartoon / degrade_m render, zoom / mosaic as configured).
The fsim draw uses its own generator default_rng([seed, fold, epoch, i, FSIM_TAG]); the parent's generators are not
touched, so with P_FSIM = 0 the items are bit-identical to RenderSet's.
Label-free: targets come from the clean originals' label maps (prep.npz); no hardness value, no noisy image and no test
image enters training. degrade_sim's priors were fitted on statistics of TRAIN images only.
Rule (d1011 PLAN_G / PLAN_H): degrade_sim is used here for a HET reader only, never for an fd or hardness reader.
Setup: degrade_sim.py and fx_sim.py must sit in the same folder as this file (copies of hardness-cache
scripts/d1011/G/forensics/{degrade_sim,fx_sim}.py); install(repo_root, p_fsim, p_g1hi) patches fx_sim.REPO and
src.het_cnn.RenderSet. Module-level state is re-created in spawn workers from the class attributes (pickled)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
FSIM_TAG = 7311
_REPO = os.path.abspath(os.path.join(HERE, '..', '..'))   # default: <repo>/tools/fsim -> <repo>


def _sim():
    import fx_sim as X
    import degrade_sim as D
    if X.REPO != FsimRenderSetBase.repo:
        X.REPO = FsimRenderSetBase.repo
    return D


class FsimRenderSetBase:
    repo = _REPO
    p_fsim = 0.5
    p_g1hi = 0.77


def make_class(H):
    class FsimRenderSet(H.RenderSet):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.p_fsim, self.p_g1hi, self.repo = FsimRenderSetBase.p_fsim, FsimRenderSetBase.p_g1hi, FsimRenderSetBase.repo

        def __getitem__(self, idx):
            import torch
            ep, i = divmod(int(idx), self.n_items)
            ra = np.random.default_rng([self.seed, self.fold, ep, i, FSIM_TAG])
            if not (ra.random() < self.p_fsim):
                return super().__getitem__(idx)
            if self._P is None:
                self._P = H.load_prep(self.fp)
            P = self._P
            j = int(self.rows[i % len(self.rows)])
            FsimRenderSetBase.repo = self.repo
            D = _sim()
            prior = 'G1hi' if ra.random() < self.p_g1hi else 'midLF'
            img, _ = D.degrade_train(j, prior, seed=[self.seed, self.fold, ep, i, FSIM_TAG, 1])
            k = int(ra.integers(8))
            img = np.ascontiguousarray(H.d4_np(img, k))
            b = P["b"][j][H.GRID_PERM[k]]
            valid = ~np.isnan(b)
            item = (torch.from_numpy(img)[None], torch.from_numpy(np.nan_to_num(b).astype(np.float32)),
                    torch.from_numpy(valid.astype(np.float32)), torch.tensor(float(P["het4"][j])),
                    torch.tensor(float(np.log(P["n_eff"][j]))))
            if self.xt is not None:
                item = item + (torch.from_numpy(self.xt[j].copy()),)
            if self.cells:
                c = H.cell_targets(P["ws"][j].astype(np.int64), P["pore"][j])
                c = np.ascontiguousarray(H.d4_np(c, k))
                item = item + (torch.from_numpy(np.nan_to_num(c).astype(np.float32)),
                               torch.from_numpy((~np.isnan(c)).astype(np.float32)))
            return item
    FsimRenderSet.__module__ = __name__
    FsimRenderSet.__qualname__ = 'FsimRenderSet'
    return FsimRenderSet


_CLS = {}


def __getattr__(name):            # lets pickle resolve fsim_set.FsimRenderSet in spawn workers
    if name == 'FsimRenderSet':
        if 'c' not in _CLS:
            from src import het_cnn as H
            _CLS['c'] = make_class(H)
        return _CLS['c']
    raise AttributeError(name)


def install(repo_root, p_fsim, p_g1hi):
    """Patch src.het_cnn.RenderSet -> FsimRenderSet (training items only) and fx_sim.REPO -> repo_root."""
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from src import het_cnn as H
    FsimRenderSetBase.repo, FsimRenderSetBase.p_fsim, FsimRenderSetBase.p_g1hi = repo_root, p_fsim, p_g1hi
    cls = __getattr__('FsimRenderSet')
    H.RenderSet = cls
    _sim()
    return H, cls
