"""Frozen multi-stage embeddings from openly licensed timm backbones (GitHub-hosted weights through
common.create_timm). Every stage of a `features_only` model is pooled (mean / std / max / GeM) and
concatenated; each TTA view is stored separately so heads can average views or use them as augmentation.

Output (pure inference, nothing is fitted here):
  data/emb/<tag>.npy        float32 (N, V, D), rows = train ids then test ids
  data/emb/<tag>_ids.csv    row order
  data/emb/<tag>_meta.json  column blocks [{stage, pool, start, end}], views, license, timing
  tag = <backbone>_<size>[_<prep>]

python -m src.extract_embeddings --backbone resnet18.a1_in1k --size 256 --views 4 --threads 1
"""
import os
import sys


def _early_threads(default=1):
    """Limit BLAS/OpenMP pools before numpy/torch are imported (argparse runs too late for that)."""
    n = default
    for i, a in enumerate(sys.argv):
        if a == "--threads" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
        elif a.startswith("--threads="):
            n = int(a.split("=", 1)[1])
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(n)
    return n


_N_THREADS = _early_threads()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .common import DATA_DIR, create_timm, load_test, read_img  # noqa: E402

# The first four views see an oriented texture at t, -t, 90-t, 90+t (mod 180 deg), the most
# orientation-diverse 4-subset of the dihedral group; views 5-8 complete D4.
VIEWS = {
    "id": lambda x: x,
    "hflip": lambda x: x.flip(-1),
    "transpose": lambda x: x.transpose(-1, -2),
    "rot90": lambda x: torch.rot90(x, 1, (-2, -1)),
    "vflip": lambda x: x.flip(-2),
    "rot180": lambda x: torch.rot90(x, 2, (-2, -1)),
    "rot270": lambda x: torch.rot90(x, 3, (-2, -1)),
    "antitranspose": lambda x: torch.rot90(x, 2, (-2, -1)).transpose(-1, -2),
}
VIEW_ORDER = list(VIEWS)

LICENSE_HINT = {"swin": "MIT", "default": "Apache-2.0"}


def gaussian_blur(x, sigma):
    r = max(1, int(round(3 * sigma)))
    t = torch.arange(-r, r + 1, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, 1, -1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.transpose(-1, -2))


def denoise_np(img, step):
    """Per-image denoising on a single (H,W) float image in [0,1]; uses only that image (no fitting)."""
    import cv2

    cv2.setNumThreads(1)
    u8 = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    if step == "nlm":  # same recipe as src.features: strength from the image's own noise estimate
        from skimage import restoration

        h = float(np.clip(restoration.estimate_sigma(img) * 255 * 1.2, 3, 40))
        return cv2.fastNlMeansDenoising(u8, None, h=h, templateWindowSize=7, searchWindowSize=21) / 255.0
    if step.startswith("median"):
        return cv2.medianBlur(u8, int(step[6:] or 3)) / 255.0
    if step == "ic":  # illumination correction: divide by the local matrix level (70th pct, ~80 px window)
        from scipy import ndimage as ndi

        h, w = img.shape
        small = img.reshape(h // 4, 4, w // 4, 4).mean((1, 3))  # 4x4 block means -> 64x64
        bg = ndi.percentile_filter(small, 70, size=21, mode="reflect")
        bg = ndi.gaussian_filter(bg, 2, mode="reflect")
        bg = cv2.resize(bg.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(img / np.maximum(bg, 1e-3) * 0.6, 0, 1)
    raise ValueError(step)


NP_STEPS = ("nlm", "median", "ic")


def np_cache(ids, prep):
    """Per-image numpy preprocessing is deterministic; cache it (uint8) so every backbone can reuse it."""
    steps = [s for s in prep.split("+") if s.startswith(NP_STEPS)]
    if not steps:
        return None
    path = DATA_DIR / "emb" / f"_img_{'-'.join(steps)}_{len(ids)}.npy"
    idp = path.with_suffix(".ids.csv")
    if path.exists() and list(pd.read_csv(idp).ID) == list(ids):
        return dict(zip(ids, np.load(path, mmap_mode="r")))
    t0, arr = time.time(), np.zeros((len(ids), 256, 256), np.uint8)
    for k, i in enumerate(ids):
        im = read_img(i)
        for s in steps:
            im = denoise_np(im, s)
        arr[k] = np.clip(im * 255 + 0.5, 0, 255).astype(np.uint8)
        if k % 100 == 0:
            print(f"prep {'-'.join(steps)} {k}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
    path.parent.mkdir(exist_ok=True)
    np.save(path, arr)
    pd.Series(ids).to_csv(idp, index=False, header=["ID"])
    return dict(zip(ids, arr))


def load_batch(ids, cache=None):
    if cache is None:
        imgs = [read_img(i) for i in ids]
    else:
        imgs = [cache[i].astype(np.float32) / 255.0 for i in ids]
    return torch.from_numpy(np.stack(imgs).astype(np.float32))[:, None]


def preprocess(x, prep):
    """x: (B,1,H,W) in [0,1]. prep: '+'-joined steps, e.g. 'raw', 'imgnorm', 'blur1', 'nlm+imgnorm'
    (numpy-level steps nlm/median are applied in load_batch)."""
    for step in prep.split("+"):
        if step in ("raw", "") or step.startswith(NP_STEPS):
            continue
        if step.startswith("blur"):
            x = gaussian_blur(x, float(step[4:] or 1.0))
        elif step == "imgnorm":  # per-image z-score mapped onto ImageNet grey statistics
            mu = x.mean((2, 3), keepdim=True)
            sd = x.std((2, 3), keepdim=True).clamp_min(1e-4)
            x = (x - mu) / sd * 0.226 + 0.449
        else:
            raise ValueError(step)
    return x


def pool_stage(f, pools, gem_p=3.0):
    f = f.float()
    var, mean = torch.var_mean(f, dim=(2, 3), correction=0)
    srt = f.flatten(2).sort(dim=2).values if any(p[0] == "q" for p in pools) else None
    out = []
    for p in pools:
        if p == "mean":
            out.append(mean)
        elif p == "std":
            out.append(var.clamp_min(0).sqrt())
        elif p == "max":
            out.append(f.amax((2, 3)))
        elif p == "gem":
            out.append(f.clamp_min(1e-6).pow(gem_p).mean((2, 3)).pow(1.0 / gem_p))
        elif p[0] == "q" and p[1:].isdigit():  # spatial quantile, e.g. q10 / q50 / q90
            out.append(srt[:, :, int(round(int(p[1:]) / 100.0 * (srt.shape[2] - 1)))])
        else:
            raise ValueError(p)
    return out


def pool_cells(f, pools, grid):
    """Global pooling plus pooling inside each cell of a grid x grid partition of the feature map."""
    outs = [pool_stage(f, pools)]
    if grid > 1:
        H, W = f.shape[-2:]
        hs = [round(i * H / grid) for i in range(grid + 1)]
        ws = [round(i * W / grid) for i in range(grid + 1)]
        for i in range(grid):
            for j in range(grid):
                outs.append(pool_stage(f[:, :, hs[i]:hs[i + 1], ws[j]:ws[j + 1]], pools))
    return outs


def id_seed(i):
    return int("".join(ch for ch in i if ch.isdigit())) + (0 if i.startswith("TRAIN") else 10 ** 7)


def apply_aug(x, aug, ids):
    """Deterministic per-image nuisance augmentation on (B,1,H,W) images in [0,1] (identity orientation).
    Used as extra 'aug:' views so a head can penalise its sensitivity to noise / blur / contrast."""
    if aug.startswith("noise"):
        s = float(aug[5:])
        nz = torch.stack([torch.randn(x.shape[1:], generator=torch.Generator().manual_seed(id_seed(i)))
                          for i in ids])
        return (x + s * nz).clamp(0, 1)
    if aug.startswith("blur"):
        return gaussian_blur(x, float(aug[4:]))
    if aug.startswith("gamma"):
        return x.clamp_min(1e-6) ** float(aug[5:])
    if aug.startswith("shade"):  # smooth multiplicative illumination field, +-s relative (bicubic 4x4 grid)
        s = float(aug[5:])
        fld = torch.stack([torch.rand((1, 4, 4), generator=torch.Generator().manual_seed(id_seed(i) + 7)) * 2 - 1
                           for i in ids])
        fld = F.interpolate(fld, size=x.shape[-2:], mode="bicubic", align_corners=True)
        return (x * (1 + s * fld)).clamp(0, 1)
    if aug.startswith("contrast"):
        mu = x.mean((2, 3), keepdim=True)
        return ((x - mu) * float(aug[8:]) + mu).clamp(0, 1)
    raise ValueError(aug)


@torch.no_grad()
def extract(backbone, ids, size=256, n_views=4, pools=("mean", "std", "max", "gem"), prep="raw", bs=16,
            out_indices=None, log_every=10, grid=1, augs=(), view_names=None, cell_grids=(), cell_stages=None):
    kw = dict(pretrained=True, features_only=True)
    if out_indices is not None:
        kw["out_indices"] = out_indices
    m = create_timm(backbone, **kw).eval()
    cfg = m.pretrained_cfg
    chans, reds = m.feature_info.channels(), m.feature_info.reduction()
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    views = list(view_names) if view_names else VIEW_ORDER[:n_views]
    assert all(v in VIEWS for v in views), views
    blocks, col = [], 0
    for si, c in enumerate(chans):
        for p in pools:
            blocks.append({"stage": si, "pool": p, "start": col, "end": col + c, "channels": c, "reduction": reds[si]})
            col += c
    n_cells = 1 + grid * grid if grid > 1 else 1
    names = list(views) + [f"aug:{a}" for a in augs]
    E = np.zeros((len(ids), len(names) * n_cells, col), np.float32)
    # compact per-cell store (only cell_stages, any number of grids) -> <tag>_cells.npy (N, rows, cells, Dc)
    cell_sel = [si for si in range(len(chans)) if cell_stages is None or si in cell_stages]
    cblocks, ccol = [], 0
    for si in cell_sel:
        for p in pools:
            cblocks.append({"stage": si, "pool": p, "start": ccol, "end": ccol + chans[si], "channels": chans[si],
                            "reduction": reds[si]})
            ccol += chans[si]
    n_cc = sum(g * g for g in cell_grids)
    Ec = np.zeros((len(ids), len(names), n_cc, ccol), np.float32) if n_cc else None
    cache = np_cache(ids, prep)

    def to_input(z):
        if size != z.shape[-1]:
            z = F.interpolate(z, size=(size, size), mode="bilinear", align_corners=False)
        return ((z.repeat(1, 3, 1, 1) - mean) / std).contiguous()

    t0 = time.time()
    for bi, s in enumerate(range(0, len(ids), bs)):
        bids = ids[s:s + bs]
        x01 = preprocess(load_batch(bids, cache), prep)
        x = to_input(x01)
        for vi, v in enumerate(names):
            xin = VIEWS[v](x) if v in VIEWS else to_input(apply_aug(x01, v[4:], bids))
            feats = m(xin.contiguous())
            cells = [[] for _ in range(n_cells)]
            for f, c in zip(feats, chans):
                if f.shape[1] != c and f.shape[-1] == c:  # NHWC (swin) -> NCHW
                    f = f.permute(0, 3, 1, 2)
                for k, pooled in enumerate(pool_cells(f, pools, grid)):
                    cells[k] += pooled
            for k in range(n_cells):
                E[s:s + len(x), vi * n_cells + k] = torch.cat(cells[k], 1).numpy()
            if n_cc:
                cc = [[] for _ in range(n_cc)]
                for si in cell_sel:
                    f = feats[si]
                    if f.shape[1] != chans[si] and f.shape[-1] == chans[si]:
                        f = f.permute(0, 3, 1, 2)
                    H, W, k = f.shape[-2], f.shape[-1], 0
                    for g in cell_grids:
                        hs = [round(i * H / g) for i in range(g + 1)]
                        ws = [round(i * W / g) for i in range(g + 1)]
                        for i in range(g):
                            for j in range(g):
                                cc[k] += pool_stage(f[:, :, hs[i]:hs[i + 1], ws[j]:ws[j + 1]], pools)
                                k += 1
                for k in range(n_cc):
                    Ec[s:s + len(x), vi, k] = torch.cat(cc[k], 1).numpy()
        if bi % log_every == 0:
            done = s + len(x)
            el = time.time() - t0
            print(f"{done}/{len(ids)} {el:.0f}s eta {el / done * (len(ids) - done):.0f}s", flush=True)
    meta = {"backbone": backbone, "size": size, "prep": prep, "views": names, "pools": list(pools),
            "grid": grid, "cells": n_cells,  # stored view axis = view-major [view][cell], cell 0 = global
            **({"cell_layout": {"grids": list(cell_grids), "stages": cell_sel, "blocks": cblocks, "dim": ccol}}
               if n_cc else {}),
            "channels": chans, "reduction": reds, "blocks": blocks, "dim": col,
            "license": LICENSE_HINT["swin"] if "swin" in backbone else (cfg.get("license") or "?"),
            "weights_url": cfg.get("url"), "seconds": round(time.time() - t0, 1), "threads": _N_THREADS}
    return (E, meta, Ec) if n_cc else (E, meta)


def tag_of(backbone, size, prep="raw"):
    return f"{backbone.replace('/', '_')}_{size}" + ("" if prep == "raw" else "_" + prep.replace("+", "-"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--views", type=int, default=4, help="number of TTA views (1-8), stored separately")
    ap.add_argument("--pools", default="mean,std,max,gem")
    ap.add_argument("--prep", default="raw", help="raw | imgnorm | blur<sigma> | combos joined by '+'")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N train+test ids")
    ap.add_argument("--grid", type=int, default=1, help=">1: also pool each cell of a grid x grid partition")
    ap.add_argument("--augs", default="", help="extra nuisance views, e.g. noise0.03,blur1.0,contrast0.7")
    ap.add_argument("--view-names", default="", help="explicit orientation views (overrides --views), e.g. "
                    "vflip,rot180,rot270,antitranspose")
    ap.add_argument("--cell-grids", default="", help="compact per-cell store, e.g. 2,4 -> <tag>_cells.npy")
    ap.add_argument("--cell-stages", default="0,1,2", help="stages kept in the per-cell store")
    ap.add_argument("--shard", default="", help="k/n: process only the k-th of n contiguous id chunks (0-based)")
    ap.add_argument("--merge-shards", type=int, default=0, help="n: merge <tag>.shard{k}of{n} files into <tag>")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    ids = list(pd.read_csv(DATA_DIR / "train.csv").ID) + list(load_test().ID)
    if a.limit:
        ids = ids[:a.limit]
    out = DATA_DIR / "emb"
    if a.merge_shards:  # concatenate shard outputs (same args) in id order and remove the shard files
        n, tag = a.merge_shards, a.tag
        parts = [f"{tag}.shard{k}of{n}" for k in range(n)]
        np.save(out / f"{tag}.npy", np.concatenate([np.load(out / f"{q}.npy") for q in parts]))
        if (out / f"{parts[0]}_cells.npy").exists():
            np.save(out / f"{tag}_cells.npy", np.concatenate([np.load(out / f"{q}_cells.npy") for q in parts]))
        mid = pd.concat([pd.read_csv(out / f"{q}_ids.csv") for q in parts])
        assert list(mid.ID) == ids, "shard ids do not reproduce the full id order"
        mid.to_csv(out / f"{tag}_ids.csv", index=False)
        metas = [json.loads((out / f"{q}_meta.json").read_text()) for q in parts]
        metas[0]["seconds"] = round(sum(m["seconds"] for m in metas), 1)
        metas[0]["shards"] = n
        (out / f"{tag}_meta.json").write_text(json.dumps(metas[0], indent=1))
        for q in parts:
            for suf in (".npy", "_cells.npy", "_ids.csv", "_meta.json"):
                (out / f"{q}{suf}").unlink(missing_ok=True)
        print(f"merged {n} shards -> {tag}")
        sys.exit(0)
    shard_sfx = ""
    if a.shard:
        k, n = map(int, a.shard.split("/"))
        ids = [str(i) for i in np.array_split(np.array(ids), n)[k]]
        shard_sfx = f".shard{k}of{n}"
    res = extract(a.backbone, ids, a.size, a.views, a.pools.split(","), a.prep, a.bs, grid=a.grid,
                  augs=[s for s in a.augs.split(",") if s], view_names=[s for s in a.view_names.split(",") if s],
                  cell_grids=[int(g) for g in a.cell_grids.split(",") if g],
                  cell_stages=[int(g) for g in a.cell_stages.split(",") if g])
    E, meta, Ec = res if len(res) == 3 else (*res, None)
    out.mkdir(exist_ok=True)
    tag = (a.tag or tag_of(a.backbone, a.size, a.prep) + (f"_lim{a.limit}" if a.limit else "")) + shard_sfx
    np.save(out / f"{tag}.npy", E)
    if Ec is not None:
        np.save(out / f"{tag}_cells.npy", Ec)
    pd.Series(ids).to_csv(out / f"{tag}_ids.csv", index=False, header=["ID"])
    (out / f"{tag}_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"saved {tag} {E.shape} in {meta['seconds']}s")
