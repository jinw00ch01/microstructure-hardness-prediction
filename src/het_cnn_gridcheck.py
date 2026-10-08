"""Independent checks of the 4x4-grid handling in src.het_cnn (reviewer, 2026-10-08).

Independent of het_cnn's own selftest: block ids are derived by tracking pixel coordinates (no GRID, no d4_np on a
4x4 array), targets are recomputed from a full re-segmentation of the transformed IMAGE, the training dataset item is
checked against targets recomputed from the transformed label map, the real HetNet head's cell order is probed, and
predict()'s TTA un-permutation is checked with a randomly initialised network in eval mode.

  python -m src.het_cnn_gridcheck        # needs data/het_cnn/prep.npz (python -m src.het_cnn --prep)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import het_cnn as H


def op_coords(k):
    """For op k (rot90 by k%4 counter-clockwise as np.rot90, then a left-right flip if k >= 4), the source pixel
    (y, x) of every output pixel, computed from coordinates."""
    n = 256
    yy, xx = np.indices((n, n))
    if k >= 4:  # output = flip_lr(rot(src)): out[y, x] = rot[y, n-1-x]
        xx = n - 1 - xx
    for _ in range(k % 4):  # rot90 ccw: rot[y, x] = prev[x, n-1-y]
        yy, xx = xx, n - 1 - yy
    return yy, xx


def apply_op(img, k):
    yy, xx = op_coords(k)
    return img[yy, xx]


def perm_from_coords(k):
    """perm[j] = original block id of output block j (block = 64-px cell, row-major), from the cell centres."""
    yy, xx = op_coords(k)
    return np.array([(yy[64 * r + 32, 64 * c + 32] // 64) * 4 + xx[64 * r + 32, 64 * c + 32] // 64
                     for r in range(4) for c in range(4)])


def blocks_indep(ws, pore):
    a = np.maximum(np.bincount(ws.ravel()).astype(float)[ws], 1) ** -0.5
    m = ~pore[ws]
    out = np.full(16, np.nan)
    for r in range(4):
        for c in range(4):
            sl = (slice(64 * r, 64 * r + 64), slice(64 * c, 64 * c + 64))
            if m[sl].sum() >= 50:
                out[4 * r + c] = -2 * np.log(a[sl][m[sl]].mean())
    return out


def het_blocks_on_array(im):
    """src.het_blocks.one on an array (re-segmentation) -> per-block b via blocks_indep, het4, N_eff."""
    from . import het_blocks as hb
    sm, ws = hb.segment(im.astype(np.float32))
    g = hb.grain_table(sm, ws)
    pore_g = ((hb.shading_norm(g) < 0.68) & (g.area < 600)).values
    gp = g[~pore_g]
    n_eff = float((~gp.border).sum() + 0.5 * gp.border.sum())
    is_pore = np.zeros(ws.max() + 1, bool)
    is_pore[g.label.values[pore_g]] = True
    b = blocks_indep(ws, is_pore)
    return b, float(np.std(b[~np.isnan(b)])), n_eff


def main():
    P = H.load_prep()
    # 1. ops: het_cnn's d4_np / d4_t equal the coordinate-tracked ops; GRID_PERM equals the coordinate-derived perm
    x = np.random.default_rng(0).integers(0, 255, (256, 256)).astype(np.uint8)
    for k in range(8):
        assert np.array_equal(H.d4_np(x, k), apply_op(x, k)), k
        assert np.array_equal(H.d4_t(torch.from_numpy(x), k).numpy(), apply_op(x, k)), k
        assert np.array_equal(H.GRID_PERM[k], perm_from_coords(k)), (k, H.GRID_PERM[k], perm_from_coords(k))
        assert np.array_equal(perm_from_coords(k)[H.GRID_INV[k]], np.arange(16))
    print("ok A: d4 ops == coordinate-tracked ops; GRID_PERM == perm from cell centres; GRID_INV inverts it")

    # 2. full re-segmentation of the transformed IMAGE (independent of the stored label map)
    worst_b, worst_h, worst_n, n_exact = 0.0, 0.0, 0.0, 0
    for j in (0, 50, 100, 200):
        b0 = P["b"][j]
        for k in range(1, 8):
            bk, hk, nk = het_blocks_on_array(apply_op(P["im8"][j], k))
            want = b0[perm_from_coords(k)]
            worst_b = max(worst_b, float(np.nanmax(np.abs(bk - want))))
            worst_h = max(worst_h, abs(hk - P["het4"][j]))
            worst_n = max(worst_n, abs(nk - P["n_eff"][j]))
            n_exact += int(np.allclose(bk, want, atol=1e-9, equal_nan=True))
            # the right permutation must beat every other one by a wide margin
            errs = [np.nanmean(np.abs(bk - b0[perm_from_coords(q)])) for q in range(8)]
            assert int(np.argmin(errs)) == k or np.allclose(b0[perm_from_coords(k)], b0[perm_from_coords(int(np.argmin(errs)))]), (j, k, errs)
    print(f"ok B: re-segmented transformed images: block targets == permuted targets (max |db| {worst_b:.2e}, "
          f"|dhet4| {worst_h:.2e}, |dN_eff| {worst_n:.2e}; {n_exact}/28 exact)")

    # 3. training items: the returned image is a D4 op of the render, the returned targets are the blocks of that op's
    #    label map (independent block code), het4 and logN untouched
    ds = H.RenderSet(H.PREP_FP, np.arange(len(P["ids"])), 2, 0, 0)
    seen = set()
    for idx in list(range(0, 40)) + [len(ds) * 3 + 7, len(ds) * 11 + 101]:
        img, b, v, het, logn = ds[idx]
        ep, i = divmod(idx, ds.n_items)
        j = int(ds.rows[i % len(ds.rows)])
        rng = np.random.default_rng([ds.seed, ds.fold, ep, i])
        r0, _ = H.render_noisy(P, j, rng)
        ks = [k for k in range(8) if np.array_equal(apply_op(r0, k), img[0].numpy())]
        assert len(ks) == 1, (idx, ks)
        k = ks[0]
        seen.add(k)
        ws_k = apply_op(P["ws"][j].astype(np.int64), k)
        want = blocks_indep(ws_k, P["pore"][j])
        assert np.allclose(b.numpy(), np.nan_to_num(want), atol=1e-5) and np.array_equal(v.numpy() > 0, ~np.isnan(want))
        assert abs(float(het) - P["het4"][j]) < 1e-6 and abs(float(logn) - np.log(P["n_eff"][j])) < 1e-6
    print(f"ok C: 42 RenderSet items: targets == blocks of the transformed label map (ops seen {sorted(seen)})")

    # 4. HetNet cell order: with a translation-equivariant body that only sees local brightness, lighting up 64-px
    #    block (r, c) must light up output cell 4r+c
    net = H.HetNet("resnet18", pretrained=False)

    class LocalBody(nn.Module):
        num_features = net.body.num_features

        def forward_features(self, x):
            return F.avg_pool2d(x, 32).repeat(1, self.num_features, 1, 1)

    net.body = LocalBody()
    with torch.no_grad():
        net.cell.weight.fill_(1.0)
        net.cell.bias.zero_()
    for blk in range(16):
        im = torch.zeros(1, 1, 256, 256)
        r, c = divmod(blk, 4)
        im[..., 64 * r:64 * r + 64, 64 * c:64 * c + 64] = 1
        out, _ = net(im)
        assert int(out.argmax()) == blk and float(out.max()) > 0 and (out > 0).sum() == 1
    print("ok D: HetNet output cell j covers 64-px block (j // 4, j % 4) (row-major, as GRID)")

    # 5. predict(): TTA un-permutation with a randomly initialised resnet18 HetNet in eval mode, against a manual
    #    loop with the coordinate-derived inverse; and with the equivariant body every single view agrees
    torch.manual_seed(0)
    net = H.HetNet("resnet18", pretrained=False).eval()
    X8 = np.stack([P["im8"][j] for j in (0, 1, 2)])
    mu, sd = torch.tensor(0.45), torch.tensor(0.2)
    bz, nz = H.predict(net, X8, mu, sd, torch.device("cpu"), None, tta=8)
    x = (torch.from_numpy(X8).float()[:, None] / 255.0 - mu) / sd
    acc, accn = 0, 0
    with torch.no_grad():
        for k in range(8):
            xk = torch.from_numpy(np.ascontiguousarray(np.stack([apply_op(t[0].numpy(), k) for t in x])))[:, None]
            bk, nk = net(xk)
            inv = np.argsort(perm_from_coords(k))  # original block i sits at output position inv[i]
            acc = acc + bk.numpy()[:, inv]
            accn = accn + nk.numpy()
    assert np.allclose(bz, acc / 8, atol=1e-4) and np.allclose(nz, accn / 8, atol=1e-4), np.abs(bz - acc / 8).max()
    net2 = H.HetNet("resnet18", pretrained=False).eval()
    net2.body = LocalBody()
    b1, _ = H.predict(net2, X8, mu, sd, torch.device("cpu"), None, tta=1)
    for k in range(8):
        xk = np.stack([apply_op(t, k) for t in X8])
        bk1, _ = H.predict(net2, xk, mu, sd, torch.device("cpu"), None, tta=1)  # cells in the transformed frame
        assert np.allclose(bk1[:, np.argsort(perm_from_coords(k))], b1, atol=1e-5)
        bk8, _ = H.predict(net2, xk, mu, sd, torch.device("cpu"), None, tta=8)
        assert np.allclose(bk8, b1[:, perm_from_coords(k)], atol=1e-5)
    print("ok E: predict() TTA == manual 8-view loop with the coordinate-derived inverse (random resnet18, eval); "
          "equivariant body: rotated inputs give permuted cells")
    print("gridcheck passed")


if __name__ == "__main__":
    main()
