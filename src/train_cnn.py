"""End-to-end CNN regression on the shared 5 folds (data/folds.csv).

Honest CV: a fixed epoch schedule is used and the *final* (or EMA) weights predict the
validation fold -- no checkpoint is ever selected on the validation fold.
Training uses random crops (no resizing -> grain/pore scale is preserved) with label-preserving
augmentation only: D4 flips/rot90, mild brightness/contrast jitter, gaussian noise, slight blur.
Inference runs on the full 256x256 image with D4 TTA. Target is standardized per fold.
--input raw|nlm|raw+nlm: optional non-local-means denoised channel (a fixed per-image transform,
nothing is fitted on test). Input normalization stats come from the training fold only.
--deg-p P: with probability P a training image from the cleaner half of train (ic_ridge_snr > --deg-snr-min)
is replaced by one of K synthetic degradations of itself (same label), made with the feature-engineer's
calibration-v2 degradation model (src.features._degrade_v2: noise/blur/dark-contrast/shading matched to the
real images per noise band). Build the bank once with --build-deg. Train images only; validation-fold
copies are never used in that fold. At assembly the OOF RMSE per train ic_ridge_snr tercile is reported.
--input raw+rest | raw+nlm+rest: adds the restored image of src.restore's original restorer (a fixed per-image
transform of train and test alike, trained on synthetic pairs of clean train images only, nothing fitted on test),
read from --rest-dir (default data_restored/, written by `python -m src.restore apply`). Like nlm it is a denoised
channel: crop/D4/brightness/contrast are shared, blur and noise touch only the raw channel. With --deg-p every
degraded bank copy gets its own restored channel: the same restorer (data/restore_cache/restore_unet.pt) applied to
the degraded raw copy, cached in data/cnn_cache/_pre/deg2_k{K}_snr{S}_rest.npz by
  python -m src.train_cnn --build-deg --input raw+nlm+rest --deg-k 8 --deg-snr-min 0.5 [--device cuda]
That file also keeps the restorer's output for 8 real noisy train images, which must match --rest-dir (checked when
it is built and whenever it is loaded), so a bank and a --rest-dir made by different restorers are never mixed.
Old --input values run exactly as before.
--full: one model per seed on ALL train images (same fixed schedule and epochs, no validation fold, nothing
selected), predicting the test images with the same TTA; the degradation bank uses every train image that passes the
SNR rule. Writes only test predictions (experiments/<name>/test.csv + full.json, no OOF, no score.json), so it needs
its own --name; pair it with the fold run of the same recipe for the OOF.
--scale S: bilinear upsampling (on the device) of each augmented training crop and of every TTA view by S; --crop
stays in native pixels (S=2, crop 224: 448 px crops and 512 px views, about 4x compute; crop 112: same training
cost as now). Default 1.0 skips it entirely.
  python -m src.train_cnn <recipe> --full --seeds 6 --no-save --train-seeds 0 --name X_full6   # ... seeds 1..5
  python -m src.train_cnn <recipe> --full --seeds 6 --name X_full6                             # assemble + save
--pool grid4 (needs --crop 256, i.e. the full image, so the grid stays aligned to the image origin): the final feature
map is average-pooled onto a fixed 4x4 grid of 64-px cells (F.adaptive_avg_pool2d(f, 4); a stride-32 backbone's 8x8
map splits exactly into 2x2 blocks) and the head sees concat(mean over the 16 cells, sd over the 16 cells) -- avgstd
on the aligned cell map, so it can represent a grid-aligned spread such as het4 (sd of block log grain size over the
same 4x4 grid, src/het_blocks.py). D4 flips/rot90 map the aligned grid onto itself, so D4 aug and TTA stay.
CPU-friendly: bf16 autocast + channels_last (uses AMX on Sapphire/Emerald Rapids), threads capped
at 2 on CPU, no DataLoader workers. --device auto|cpu|cuda: on a GPU, autocast uses bf16 when the card
supports it, else fp16 with a GradScaler. Per-fold predictions are cached in data/cnn_cache/<name>/ so a run
can be resumed; experiments/<name>/ is written only when all 5 folds exist.

  python -m src.train_cnn --backbone resnet18.a1_in1k --epochs 30 --name cnn_r18
  python -m src.train_cnn --folds 0 --epochs 15 --no-test --name dbg_r18   # sanity check only
  # seeds in parallel 1-thread processes, then an assembly-only call (everything cached):
  python -m src.train_cnn --threads 1 --seeds 2 --train-seeds 0 --no-save --name X &
  python -m src.train_cnn --threads 1 --seeds 2 --train-seeds 1 --no-save --name X; wait
  python -m src.train_cnn --seeds 2 --name X
"""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .common import DATA_DIR, EXP_DIR, ROOT, SEED, create_timm, load_test, load_train, read_img, rmse, save_experiment

CACHE = DATA_DIR / "cnn_cache"
RT = {"dev": torch.device("cpu"), "dtype": torch.bfloat16}  # runtime device / autocast dtype


def amp(enabled):
    return torch.autocast(RT["dev"].type, dtype=RT["dtype"], enabled=enabled)


def d4(x, k):  # x: (..., H, W); k in 0..7 -> the 8 dihedral transforms
    x = torch.rot90(x, k % 4, dims=(-2, -1))
    return x.flip(-1) if k >= 4 else x


def _raw_u8(i):
    return np.round(read_img(i) * 255).astype(np.uint8)


def _nlm_arr(u8):
    """Non-local-means denoising with h tied to the estimated noise level (as in src/features.py)."""
    import cv2
    from skimage import restoration
    h = float(np.clip(restoration.estimate_sigma(u8.astype(np.float32) / 255.0) * 255 * 1.2, 3, 40))
    return cv2.fastNlMeansDenoising(u8, None, h=h, templateWindowSize=7, searchWindowSize=21)


def _nlm_u8(i):
    return _nlm_arr(_raw_u8(i))


DEG_CHANNELS = {"raw": [0], "nlm": [1], "raw+nlm": [0, 1]}  # bank channels are (raw, nlm)


def deg_bank(k=8, snr_min=0.5, build=False):
    """Synthetic degradations of the cleaner TRAIN images -> {ID: uint8 (k, 2, 256, 256)} (raw, nlm).
    Uses src.features._degrade_v2 read-only (feature-engineer's model matched to the real images)."""
    import pandas as pd
    fp = CACHE / "_pre" / f"deg2_k{k}_snr{snr_min}.npz"
    if fp.exists():
        z = np.load(fp)
        return dict(zip(z["ids"].tolist(), z["img"]))
    if not build:
        raise SystemExit(f"{fp} missing: build it first with --build-deg --deg-k {k} --deg-snr-min {snr_min}")
    import cv2
    from .features import _degrade_v2
    cv2.setNumThreads(1)
    v3 = pd.read_parquet(DATA_DIR / "features_v3.parquet", columns=["ID", "ic_ridge_snr"]).set_index("ID")
    src = [i for i in load_train().ID if v3.loc[i, "ic_ridge_snr"] > snr_min]  # train images only
    out = np.zeros((len(src), k, 2, 256, 256), np.uint8)
    t = time.time()
    for n, i in enumerate(src):
        raw8 = _raw_u8(i)
        for j in range(k):
            d, _ = _degrade_v2(raw8, np.random.default_rng([SEED, int(i.split("_")[-1]), j]))
            out[n, j, 0], out[n, j, 1] = d, _nlm_arr(d)
        if n % 50 == 0:
            print(f"deg bank {n}/{len(src)} {time.time() - t:.0f}s", flush=True)
    fp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(fp, ids=np.array(src), img=out)
    print(f"deg bank: {len(src)} sources x {k} copies -> {fp} in {time.time() - t:.0f}s", flush=True)
    return dict(zip(src, out))


REST_CHECK_N = 8  # real train images whose restorer output is stored with the restored bank (consistency check)


def _rest_dir(a):
    p = Path(a.rest_dir)
    return p if p.is_absolute() else ROOT / p


def _rest_u8(i, rest_dir):
    fp = rest_dir / ("train" if i.startswith("TRAIN") else "test") / f"{i}.png"
    if not fp.exists():
        raise SystemExit(f"{fp} missing: write the restored images first (python -m src.restore apply; "
                         "notes/handoff/laptop-gpu.md section 8)")
    return np.asarray(Image.open(fp).convert("L"), dtype=np.uint8)


def _rest_check_ids(n=REST_CHECK_N):
    """Fixed real train images for the --rest-dir consistency check, spread over the upper two thirds of the noise
    range (clean images pass through any restorer almost unchanged, so they cannot tell restorers apart)."""
    import pandas as pd
    ids = load_train().ID.tolist()
    nz = pd.read_parquet(DATA_DIR / "features_v3.parquet", columns=["ID", "ic_noise"]).set_index("ID")
    nz = nz.loc[ids, "ic_noise"]
    order = nz.sort_values(kind="stable").index.tolist()
    return [order[j] for j in np.linspace(len(order) // 3, len(order) - 1, n).round().astype(int)]


def _rest_match(ids, imgs, rest_dir):
    """Does --rest-dir hold this restorer output for `ids`? Allows the rare 1-grey-level rounding flips of a rerun of
    the same model (other batch layout or GPU kernels), not another restorer or another precision."""
    d = np.abs(np.stack([_rest_u8(i, rest_dir) for i in ids]).astype(np.int16) - imgs.astype(np.int16))
    mean, frac, mx = float(d.mean()), float((d > 1).mean()), int(d.max())
    return (mean < 0.05 and frac < 1e-3), (f"vs {rest_dir.name}/ on {len(ids)} noisy train images: mean |diff| "
                                           f"{mean:.4f} grey levels, {100 * frac:.3f}% of pixels off by >1, max {mx}")


def _restore_u8(R, model, imgs8):
    """uint8 (N, H, W) -> restored uint8, as `python -m src.restore apply` writes its PNGs (8-view TTA, rounded)."""
    imgs8 = np.ascontiguousarray(imgs8)
    r = R.restore(model, torch.from_numpy(imgs8), [R._sigma(x) for x in imgs8], tta=8).numpy()
    return np.clip(np.round(r), 0, 255).astype(np.uint8)


def rest_bank(k=8, snr_min=0.5, rest_dir=None, build=False):
    """Restored channel of the degradation bank -> ({ID: uint8 (k, 256, 256)}, restorer sha). Each degraded raw copy
    is restored by the original restorer (src.restore, data/restore_cache/restore_unet.pt) exactly as `src.restore
    apply` made --rest-dir, so a degraded training copy gets its own restored channel. Train images only."""
    fp = CACHE / "_pre" / f"deg2_k{k}_snr{snr_min}_rest.npz"
    hint = f"--build-deg --input raw+nlm+rest --deg-k {k} --deg-snr-min {snr_min} --device <the device used for apply>"
    if fp.exists():
        with np.load(fp) as z:
            ids, img, sha, info = z["ids"].tolist(), z["img"], str(z["model_sha"]), f"{z['device']} {z['precision']}"
            ok, msg = _rest_match(z["chk_ids"].tolist(), z["chk_img"], rest_dir)
        if ok:
            print(f"restored deg bank {fp.name} (restorer {sha}, {info}): {msg}", flush=True)
            return dict(zip(ids, img)), sha
        if not build:
            raise SystemExit(f"{fp} was made by another restorer than {rest_dir} ({msg}). Rebuild it with {hint}")
        print(f"{fp.name} does not match {rest_dir} ({msg}): rebuilding", flush=True)
    elif not build:
        raise SystemExit(f"{fp} missing: build it first with {hint} (needs data/restore_cache/restore_unet.pt)")
    from . import restore as R
    mp = R.paths("")["model"]
    if not mp.exists():
        raise SystemExit(f"{mp} missing: train the restorer first (python -m src.restore bank / train / apply)")
    R.set_device(RT["dev"].type)  # strict fp32 on a GPU; on the CPU the restorer's own precision (original: bf16)
    model = R.load_model("")
    sha = hashlib.sha256(mp.read_bytes()).hexdigest()[:12]
    prec = "fp32" if (RT["dev"].type != "cpu" or not model.amp) else "bf16"
    chk_ids = _rest_check_ids()
    chk_img = _restore_u8(R, model, np.stack([_raw_u8(i) for i in chk_ids]))
    ok, msg = _rest_match(chk_ids, chk_img, rest_dir)
    print(f"restorer {mp.name} ({sha}, {RT['dev'].type} {prec}) {msg}", flush=True)
    if not ok:
        raise SystemExit(f"{rest_dir} was not written by this restorer at this device/precision: rewrite it with "
                         f"`python -m src.restore apply --device {RT['dev'].type}` or build with the --device "
                         "used there")
    b = deg_bank(k, snr_min)
    ids = list(b)
    out = np.zeros((len(ids), k, 256, 256), np.uint8)
    t = time.time()
    for n, i in enumerate(ids):
        out[n] = _restore_u8(R, model, b[i][:, 0])  # restored from the degraded raw copy (bank channel 0)
        if n % 50 == 0:
            print(f"restored deg bank {n}/{len(ids)} {time.time() - t:.0f}s", flush=True)
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = fp.with_name(fp.stem + "_tmp.npz")
    np.savez(tmp, ids=np.array(ids), img=out, chk_ids=np.array(chk_ids), chk_img=chk_img, model_sha=sha,
             device=RT["dev"].type, precision=prec)
    os.replace(tmp, fp)
    print(f"restored deg bank: {len(ids)} sources x {k} copies -> {fp} in {time.time() - t:.0f}s", flush=True)
    return dict(zip(ids, out)), sha


def snr_tercile_rmse(tr, oof):
    """OOF RMSE per tercile of train ic_ridge_snr (noisy, mid, clean); None if features_v3 is missing."""
    import pandas as pd
    fp = DATA_DIR / "features_v3.parquet"
    if not fp.exists():
        return None
    s = pd.read_parquet(fp, columns=["ID", "ic_ridge_snr"]).set_index("ID").loc[tr.ID, "ic_ridge_snr"].values
    t = np.digitize(s, np.quantile(s, [1 / 3, 2 / 3]))
    y = tr.hardness.values
    return [rmse(oof[t == k], y[t == k]) for k in range(3)]


def cell_rmse(tr, te, oof):
    """OOF RMSE inside / outside the blend's 'fine_noisy' cell (src.blend_cells.cell_masks: log cal_seg_count_density
    and raw ic_noise above their train medians; no labels), over the images that have an OOF prediction.
    -> ((rmse_in, n_in, rmse_out, n_out, {fold: rmse_in}), None), or (None, reason) when the cell cannot be computed
    here (e.g. data/features_cal.parquet missing)."""
    try:
        from .blend_cells import cell_masks
        m = np.asarray(cell_masks(tr.ID, te.ID)[0], bool)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    y, ok, fo = tr.hardness.values, ~np.isnan(oof), tr.fold.values
    ci, co = m & ok, ~m & ok
    if not ci.any() or not co.any():
        return None, "no OOF images in the cell"
    by_fold = {int(f): round(rmse(oof[ci & (fo == f)], y[ci & (fo == f)]), 3) for f in np.unique(fo[ci])}
    return (rmse(oof[ci], y[ci]), int(ci.sum()), rmse(oof[co], y[co]), int(co.sum()), by_fold), None


def _cached(kind, ids):
    fp = CACHE / "_pre" / f"{kind}.npz"
    have = {}
    if fp.exists():
        z = np.load(fp)
        have = dict(zip(z["ids"].tolist(), z["img"]))
    miss = [i for i in ids if i not in have]
    if miss:
        import cv2
        cv2.setNumThreads(1)
        t = time.time()
        for i in miss:
            have[i] = _nlm_u8(i)
        print(f"computed {kind} for {len(miss)} images in {time.time() - t:.0f}s", flush=True)
        fp.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(have)
        np.savez(fp, ids=np.array(keys), img=np.stack([have[k] for k in keys]))
    return np.stack([have[i] for i in ids])


def load_u8(ids, mode="raw", rest_dir=None):
    """-> uint8 tensor (N, C, 256, 256); channel 0 is raw when mode contains 'raw'; 'rest' is read from rest_dir."""
    ids = list(ids)
    chans = []
    for kind in mode.split("+"):
        if kind == "rest":
            chans.append(np.stack([_rest_u8(i, rest_dir) for i in ids]))
            continue
        chans.append(np.stack([_raw_u8(i) for i in ids]) if kind == "raw" else _cached(kind, ids))
    return torch.from_numpy(np.stack(chans, 1))


def gauss_blur(x, sigma):  # x: (1, H, W) float
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=torch.float32) / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, 1, -1)
    y = F.pad(x[None], (r, r, r, r), mode="reflect")
    y = F.conv2d(F.conv2d(y, k), k.transpose(-1, -2))
    return y[0]


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p, self.eps = nn.Parameter(torch.tensor(p)), eps

    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean((-2, -1)).pow(1.0 / self.p)


class Net(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.body = create_timm(a.backbone, pretrained=not a.scratch, num_classes=0,
                                in_chans=len(a.input.split("+")),
                                drop_path_rate=a.drop_path)
        nf = self.body.num_features
        self.mode = a.pool
        self.pool = GeM() if a.pool == "gem" else None
        nh = 2 * nf if a.pool in ("avgstd", "avgstd2", "grid4") else nf
        self._mid = None
        if a.pool == "avgstd2":  # + spatial std of the stride-8 stage map (block-averaged to the final grid)
            info = [d for d in self.body.feature_info if d["reduction"] == 8][-1]
            dict(self.body.named_modules())[info["module"]].register_forward_hook(
                lambda m, i, o: setattr(self, "_mid", o))
            nh += info["num_chs"]
        self.drop = nn.Dropout(a.drop)
        self.fc = nn.Linear(nh, 1)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        f = self.body.forward_features(x)
        with torch.autocast(x.device.type, enabled=False):  # pooling + regression head in fp32
            f = f.float()
            if self.mode in ("avgstd", "avgstd2"):
                # mean + spatial std: keeps within-image heterogeneity (e.g. zones of coarse vs fine grains)
                z = [f.mean((-2, -1)), f.std((-2, -1))]
                if self.mode == "avgstd2":
                    z.append(F.adaptive_avg_pool2d(self._mid.float(), f.shape[-2:]).std((-2, -1)))
                f = torch.cat(z, 1)
            elif self.mode == "grid4":
                # mean + sd over the 16 cells of a fixed 4x4 grid of 64-px cells aligned to the image origin
                assert f.shape[-2] % 4 == 0 and f.shape[-1] % 4 == 0, f"grid4: map {tuple(f.shape[-2:])} not divisible"
                c = F.adaptive_avg_pool2d(f, 4).flatten(2)
                f = torch.cat([c.mean(-1), c.std(-1)], 1)
            else:
                f = self.pool(f) if self.pool is not None else f.mean((-2, -1))
            return self.fc(self.drop(f)).squeeze(-1)


class Aug:
    """Label-preserving augmentation on a (C, H, W) float image in [0, 1] (before normalization).
    Geometry / brightness / contrast are shared by all channels; blur and noise touch only the
    raw channel (channel 0), never a denoised one (nlm, rest)."""

    def __init__(self, a, seed):
        self.a, self.rng = a, np.random.default_rng(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.raw = a.input.split("+")[0] == "raw"

    def __call__(self, x):
        a, rng = self.a, self.rng
        if a.crop < x.shape[-1]:
            r, c = rng.integers(0, x.shape[-1] - a.crop + 1, 2)
            x = x[:, r:r + a.crop, c:c + a.crop]
        x = d4(x, int(rng.integers(8)))
        if self.raw and a.blur_p > 0 and rng.random() < a.blur_p:
            x = torch.cat([gauss_blur(x[:1], float(rng.uniform(0.3, a.blur))), x[1:]])
        if a.contrast > 0:
            m = x.mean((-2, -1), keepdim=True)
            x = (x - m) * float(rng.uniform(1 - a.contrast, 1 + a.contrast)) + m
        if a.bright > 0:
            x = x + float(rng.uniform(-a.bright, a.bright))
        if self.raw and a.noise > 0 and rng.random() < a.noise_p:
            n = float(rng.uniform(0, a.noise)) * torch.randn(x[:1].shape, generator=self.gen)
            x = torch.cat([x[:1] + n, x[1:]])
        return x

    # replayable version (same geometry/jitter for an image and its degraded twin; used by --cons)
    def sample(self, size):
        a, rng = self.a, self.rng
        return {"rc": rng.integers(0, size - a.crop + 1, 2) if a.crop < size else None,
                "k": int(rng.integers(8)),
                "blur": float(rng.uniform(0.3, a.blur)) if (self.raw and a.blur_p > 0 and rng.random() < a.blur_p) else 0.0,
                "c": float(rng.uniform(1 - a.contrast, 1 + a.contrast)) if a.contrast > 0 else 1.0,
                "b": float(rng.uniform(-a.bright, a.bright)) if a.bright > 0 else 0.0,
                "n": float(rng.uniform(0, a.noise)) if (self.raw and a.noise > 0 and rng.random() < a.noise_p) else 0.0}

    def apply(self, x, p):
        if p["rc"] is not None:
            r, c = p["rc"]
            x = x[:, r:r + self.a.crop, c:c + self.a.crop]
        x = d4(x, p["k"])
        if p["blur"] > 0:
            x = torch.cat([gauss_blur(x[:1], p["blur"]), x[1:]])
        m = x.mean((-2, -1), keepdim=True)
        x = (x - m) * p["c"] + m + p["b"]
        if p["n"] > 0:
            x = torch.cat([x[:1] + p["n"] * torch.randn(x[:1].shape, generator=self.gen), x[1:]])
        return x


def to_input(x, mu, sd):  # x: (B, C, H, W) float in [0,1]; mu, sd: (1, C, 1, 1) or None
    if mu is None:  # per-image standardization (--norm image)
        mu, sd = x.mean((2, 3), keepdim=True), x.std((2, 3), keepdim=True) + 1e-3
    return ((x - mu) / sd).contiguous(memory_format=torch.channels_last)


def upscale(x, s):
    """--scale: bilinear upsampling of a normalised batch on its device (thin 1-2 px boundaries survive the early
    downsampling of pretrained stems). s == 1.0 returns x itself, so the default path is unchanged."""
    if s == 1.0:
        return x
    h, w = x.shape[-2:]
    return F.interpolate(x, size=(round(h * s), round(w * s)), mode="bilinear",
                         align_corners=False).contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def predict(model, X, mu, sd, tta=8, bs=32, bf16=True, views=False, scale=1.0):
    """Mean over the first `tta` D4 views; views=True returns all views (tta, N). scale: --scale."""
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        x = upscale(to_input(X[i:i + bs].float() / 255.0, mu, sd).to(RT["dev"], non_blocking=True), scale)
        ps = []
        for k in range(tta):
            with amp(bf16):
                ps.append(model(d4(x, k).contiguous(memory_format=torch.channels_last)).float())
        out.append(torch.stack(ps).cpu())
    out = torch.cat(out, 1).numpy().astype(np.float64)
    return out if views else out.mean(0)


def param_groups(model, a):
    groups = {}
    for n, p in model.named_parameters():
        head = not n.startswith("body.")
        decay = (p.ndim > 1 and not head) or n.endswith("fc.weight")  # no wd on bias/BN/GeM-p
        key = (head, decay)
        groups.setdefault(key, []).append(p)
    out = []
    for (head, decay), ps in groups.items():
        out.append({"params": ps, "lr": a.lr * (a.head_lr_mult if head else 1.0),
                    "weight_decay": a.wd if decay else 0.0})
    return out


def train_one(a, f, s, Xtr, ytr, Xva, yva, Xte, bank=None, bidx=None, wtr=None):
    """Train one model with a fixed schedule; return predictions of the chosen weights.
    bank: uint8 tensor (Nb, K, C, H, W) of degraded copies; bidx[i] = bank row of training image i or -1.
    Xva None (--full, f = FULL_F): no validation fold; only the test predictions are returned."""
    seed = SEED + 1000 * s + f
    lab = f"fold {f}" if Xva is not None else "full"
    torch.manual_seed(seed)
    aug = Aug(a, seed)
    rng = np.random.default_rng(seed + 7)  # batch order
    drng = np.random.default_rng(seed + 13)  # degraded-copy sampling (separate stream)
    ymu, ysd = float(ytr.mean()), float(ytr.std())
    yt = torch.tensor((ytr - ymu) / ysd, dtype=torch.float32, device=RT["dev"])
    wt = None if wtr is None else torch.tensor(wtr, dtype=torch.float32, device=RT["dev"])  # --wN row weights (mean 1)
    pix = Xtr.float() / 255.0  # input normalization from this fold's training images only
    mu, sd = pix.mean((0, 2, 3)).view(1, -1, 1, 1), pix.std((0, 2, 3)).view(1, -1, 1, 1)
    del pix
    if a.norm == "image":
        mu = sd = None

    model = Net(a).to(RT["dev"]).to(memory_format=torch.channels_last)
    scaler = torch.amp.GradScaler("cuda", enabled=RT["dev"].type == "cuda" and a.bf16 and RT["dtype"] == torch.float16)
    groups = param_groups(model, a)
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.wd)
    n = len(Xtr)
    spe = n // a.bs
    total = a.epochs * spe
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=total,
                                              pct_start=a.warmup, div_factor=a.div, final_div_factor=a.final_div)
    ema = None
    if a.ema > 0:
        from timm.utils import ModelEmaV3
        ema = ModelEmaV3(model, decay=a.ema, use_warmup=True)
    loss_fn = (lambda p, t: F.mse_loss(p, t)) if a.loss == "mse" else \
        (lambda p, t: F.smooth_l1_loss(p, t, beta=a.huber_beta))
    loss_el = (lambda p, t: F.mse_loss(p, t, reduction="none")) if a.loss == "mse" else \
        (lambda p, t: F.smooth_l1_loss(p, t, beta=a.huber_beta, reduction="none"))  # per-row, for --wN

    t0, step = time.time(), 0
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(n)
        tl = 0.0
        for b in range(spe):
            idx = perm[b * a.bs:(b + 1) * a.bs]
            xs, xd, di = [], [], []
            for j, i in enumerate(idx):
                if a.cons > 0:  # original + (with prob deg_p) its degraded twin under identical augmentation
                    prm = aug.sample(Xtr.shape[-1])
                    xs.append(aug.apply(Xtr[i].float() / 255.0, prm))
                    if bank is not None and bidx[i] >= 0 and drng.random() < a.deg_p:
                        d = bank[bidx[i], int(drng.integers(bank.shape[1]))]
                        xd.append(aug.apply(d.float() / 255.0, prm))
                        di.append(j)
                    continue
                xi = Xtr[i]
                if bank is not None and bidx[i] >= 0 and drng.random() < a.deg_p:
                    xi = bank[bidx[i], int(drng.integers(bank.shape[1]))]
                xs.append(aug(xi.float() / 255.0))
            x = torch.stack(xs + xd)
            x = upscale(to_input(x, mu, sd).to(RT["dev"], non_blocking=True), a.scale)
            with amp(a.bf16):
                p = model(x)
            p = p.float()
            tb = yt[torch.as_tensor(idx, device=RT["dev"])]
            wb = None if wt is None else wt[torch.as_tensor(idx, device=RT["dev"])]
            loss = loss_fn(p[:len(idx)], tb) if wb is None else (wb * loss_el(p[:len(idx)], tb)).mean()
            if di:  # degraded twins: supervised + consistency with the (stop-grad) original prediction
                dj = torch.as_tensor(di, device=RT["dev"])
                pdg = p[len(idx):]
                lab_dg = loss_fn(pdg, tb[dj]) if wb is None else (wb[dj] * loss_el(pdg, tb[dj])).mean()
                loss = loss + lab_dg + a.cons * F.mse_loss(pdg, p[:len(idx)][dj].detach())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if a.clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            scaler.step(opt)
            scaler.update()
            sch.step()
            step += 1
            if ema is not None:
                ema.update(model, step=step)
            tl += loss.item()
        msg = f"{lab} seed {s} ep {ep + 1:3d}/{a.epochs} loss {tl / spe:.4f} lr {sch.get_last_lr()[0]:.2e}"
        if Xva is not None and ((a.monitor and (ep + 1) % a.monitor == 0) or ep + 1 == a.epochs):  # monitoring only
            r = rmse(predict(model, Xva, mu, sd, tta=1, bf16=a.bf16, scale=a.scale) * ysd + ymu, yva)
            msg += f" | val(tta1) {r:.3f}"
            if ema is not None:
                r = rmse(predict(ema.module, Xva, mu, sd, tta=1, bf16=a.bf16, scale=a.scale) * ysd + ymu, yva)
                msg += f" ema {r:.3f}"
        print(msg + f" | {time.time() - t0:.0f}s", flush=True)

    res = {}
    final_model = ema.module if (ema is not None and a.use == "ema") else model
    if Xva is None:  # --full: final weights predict the test images only
        res["test"] = predict(final_model, Xte, mu, sd, tta=a.tta, bf16=a.bf16, scale=a.scale) * ysd + ymu
        print(f"full seed {s}: trained on {len(Xtr)} images ({total} steps), test predicted (tta{a.tta}) "
              f"| {time.time() - t0:.0f}s", flush=True)
        return res
    pv = predict(final_model, Xva, mu, sd, tta=a.tta, bf16=a.bf16, views=True, scale=a.scale) * ysd + ymu
    res["val"] = pv.mean(0)
    print(f"fold {f} seed {s} per-view val rmse {[round(rmse(v, yva), 2) for v in pv]} "
          f"view-pred sd {pv.std(0).mean():.2f}", flush=True)
    if ema is not None:  # diagnostics only: the other weights' val score (not used for OOF)
        other = model if a.use == "ema" else ema.module
        res["val_other"] = predict(other, Xva, mu, sd, tta=a.tta, bf16=a.bf16, scale=a.scale) * ysd + ymu
    res["test"] = (predict(final_model, Xte, mu, sd, tta=a.tta, bf16=a.bf16, scale=a.scale) * ysd + ymu
                   if Xte is not None else None)
    print(f"fold {f} seed {s} final val(tta{a.tta}) {rmse(res['val'], yva):.3f}"
          + (f" | other {rmse(res['val_other'], yva):.3f}" if "val_other" in res else "")
          + f" | {time.time() - t0:.0f}s", flush=True)
    return res


FULL_F = 5  # seed offset of --full models (fold models use 0..4)
WN_SRC = ["het_cnn/ev2s_loc_e32r4", "het_cnn/ev2s_loc_e32r4_s1"]  # localized het CNN outputs (v24's offset input)


def grain_count(ids):
    """--wN: label-free grain count N = exp(mean logN_cnn over WN_SRC) per train image (no label is read)."""
    import pandas as pd
    ls = []
    for d in WN_SRC:
        fp = DATA_DIR / d / "het_cnn_train.parquet"
        if not fp.exists():
            raise SystemExit(f"--wN needs {fp} (label-free output of the localized het CNN run)")
        ls.append(pd.read_parquet(fp).set_index("ID").loc[list(ids), "logN_cnn"].values)
    N = np.exp(np.mean(ls, 0))
    if not np.isfinite(N).all():
        raise SystemExit(f"--wN: {int((~np.isfinite(N)).sum())} train images have no finite logN_cnn")
    print(f"--wN: N of {len(N)} train images from {', '.join(WN_SRC)}: sum logN {float(np.log(N).sum()):.2f}, "
          f"range {N.min():.1f}-{N.max():.1f}", flush=True)
    return N


def _check_name(a):
    """Fold runs and --full runs never share a --name (cache files and experiments/<name>/ would mix)."""
    c, e = CACHE / a.name, EXP_DIR / a.name
    fold_art = any(c.glob("fold*_seed*.npz")) or (e / "oof.csv").exists() or (e / "score.json").exists()
    full_art = any(c.glob("full_seed*.npz")) or (e / "full.json").exists()
    if a.full and fold_art:
        raise SystemExit(f"--name {a.name} holds a fold run; give the --full run its own name (e.g. {a.name}_full)")
    if not a.full and full_art:
        raise SystemExit(f"--name {a.name} holds a --full run; give the fold run another name")
    if a.full and (a.folds or a.no_test):
        raise SystemExit("--full trains on all train images and predicts the test images: drop --folds / --no-test")


def recipe_notes(a, rest_dir=None, rest_sha=None):
    """Recipe part of the experiment notes (fold and --full runs)."""
    notes = (f"{a.backbone} in={a.input} norm={a.norm} ep{a.epochs} bs{a.bs} crop{a.crop} lr{a.lr} hlr{a.head_lr_mult} wd{a.wd} "
             f"{a.loss} pool={a.pool} drop{a.drop} dp{a.drop_path} ema{a.ema}->{a.use} seeds{a.seeds} "
             f"aug(b{a.bright} c{a.contrast} n{a.noise}@{a.noise_p} blur{a.blur}@{a.blur_p}) tta{a.tta}; "
             f"fixed schedule, no val checkpoint selection")
    if rest_dir is not None:
        notes += f"; restored channel from {rest_dir.name}/" + (
            f" (deg copies restored by the same restorer, sha {rest_sha})" if rest_sha else "")
    if a.deg_p > 0:
        notes += f"; deg_v2 aug p{a.deg_p} k{a.deg_k} src snr>{a.deg_snr_min}" + (f" cons{a.cons}" if a.cons > 0 else "")
    if a.scale != 1.0:
        notes += f"; input x{a.scale} bilinear ({round(a.crop * a.scale)} px crops, {round(256 * a.scale)} px views)"
    if a.wN > 0:
        notes += f"; 1/N label-noise row weights (N/mean N)^{a.wN}, N = exp(logN_cnn) of {', '.join(WN_SRC)}"
    return notes


def assemble_full(a, te, cache, rest_dir=None, rest_sha=None):
    """--full: mean test prediction of seeds 0..--seeds-1 -> experiments/<name>/test.csv (git-ignored) + full.json + a
    LEADERBOARD line. No oof.csv and deliberately no score.json: src.ensemble discovers experiments by */score.json
    and reads their oof.csv. The OOF of the matching fold run stays the member's OOF."""
    import pandas as pd
    fs = [cache / f"full_seed{s}.npz" for s in range(a.seeds)]
    have = [p.name for p in fs if p.exists()]
    if len(have) < a.seeds or a.no_save:
        print(f"full-data run: {len(have)}/{a.seeds} seeds cached ({', '.join(have) or 'none'})"
              f"{' --no-save' if len(have) == a.seeds else ''}; experiments/ not written", flush=True)
        return
    zs = []
    for p in fs:
        with np.load(p) as z:
            zs.append(dict(z))
    preds = np.stack([z["test"] for z in zs])
    if preds.shape[1] != len(te):
        raise SystemExit(f"cached full-data test predictions have {preds.shape[1]} rows, expected {len(te)}")
    n = int(zs[0]["n_train"])
    pred = preds.mean(0)
    sd = float(preds.std(0).mean()) if len(preds) > 1 else float("nan")
    notes = (f"FULL-DATA: all {n} train images, no validation fold, no OOF (CV n/a); test = mean of {a.seeds} seeds "
             f"(between-seed sd {sd:.2f}); " + recipe_notes(a, rest_dir, rest_sha))
    out = EXP_DIR / a.name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": te.ID, "hardness": pred}).to_csv(out / "test.csv", index=False)
    (out / "full.json").write_text(json.dumps({"mode": "full-data", "cv_rmse": None, "n_train": n, "seeds": a.seeds,
                                               "test_seed_sd": None if np.isnan(sd) else sd,
                                               "test_mean": float(pred.mean()), "notes": notes}, indent=2))
    with open(EXP_DIR / "LEADERBOARD.md", "a") as fh:
        fh.write(f"| {a.name} | full-data, no CV |  | {notes} |\n")
    print(f"[{a.name}] full-data test predictions ({a.seeds} seeds, {len(pred)} images, mean {pred.mean():.3f}, "
          f"between-seed sd {sd:.2f}) -> {out / 'test.csv'}; no OOF", flush=True)


def main(a):
    use_cuda = a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available())
    RT["dev"] = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        RT["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        torch.backends.cudnn.benchmark = True
    else:
        a.threads = min(a.threads, 2)  # shared cloud machine: never more than 2 threads
    if a.pool == "grid4" and a.crop != 256:
        raise SystemExit("--pool grid4 needs --crop 256 (full images keep the 4x4 grid of 64-px cells aligned)")
    torch.set_num_threads(a.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    kinds = a.input.split("+")
    rest_dir = _rest_dir(a) if "rest" in kinds else None
    if a.build_deg:
        deg_bank(a.deg_k, a.deg_snr_min, build=True)
        if rest_dir is not None:  # + the restored channel of every degraded copy
            rest_bank(a.deg_k, a.deg_snr_min, rest_dir, build=True)
        return
    _check_name(a)
    if a.wN < 0:
        raise SystemExit("--wN must be >= 0")
    tr, te = load_train(), load_test()
    y = tr.hardness.values.astype(np.float64)
    folds = a.folds if a.folds else list(range(5))
    bank_t, row, rest_sha = None, {}, None
    if a.deg_p > 0:
        b = deg_bank(a.deg_k, a.deg_snr_min)
        bank_ids = list(b)
        if rest_dir is None:
            ch = DEG_CHANNELS[a.input]
            bank_t = torch.from_numpy(np.stack([b[i][:, ch] for i in bank_ids]))
        else:  # bank channels (raw, nlm) + each degraded copy's own restored channel, in --input order
            br, rest_sha = rest_bank(a.deg_k, a.deg_snr_min, rest_dir)
            miss = [i for i in bank_ids if i not in br or br[i].shape[0] != b[i].shape[0]]
            if miss:
                raise SystemExit(f"restored deg bank misses {len(miss)} bank sources: rebuild it with --build-deg")
            pick = {"raw": lambda i: b[i][:, 0], "nlm": lambda i: b[i][:, 1], "rest": lambda i: br[i]}
            bank_t = torch.from_numpy(np.stack([np.stack([pick[c](i) for c in kinds], 1) for i in bank_ids]))
            del br
        row = {i: r for r, i in enumerate(bank_ids)}
        del b
        print(f"degradation bank: {len(bank_ids)} train sources x {bank_t.shape[1]} copies, p={a.deg_p}", flush=True)
    cache = CACHE / a.name
    cache.mkdir(parents=True, exist_ok=True)
    used = {"rest_dir": rest_dir is not None, "full": a.full, "scale": a.scale != 1.0, "wN": a.wN > 0}  # new options only when used
    Nrow = grain_count(tr.ID) if a.wN > 0 else None  # label-free per-train-image grain count (--wN)

    def wts(rows):  # --wN weights of these training rows, normalised to mean 1 over them (None = unweighted)
        if Nrow is None:
            return None
        w = Nrow[rows] ** a.wN
        return w / w.mean()
    args = {k: v for k, v in vars(a).items() if used.get(k, True)}
    (cache / "args.json").write_text(json.dumps(args, indent=2))
    t_all = time.time()
    Xall = load_u8(tr.ID, a.input, rest_dir)
    Xte = None if a.no_test else load_u8(te.ID, a.input, rest_dir)
    print(f"loaded {len(Xall)} train / {0 if Xte is None else len(Xte)} test images "
          f"in {time.time() - t_all:.0f}s; device={RT['dev']} amp={RT['dtype'] if a.bf16 else 'off'} "
          f"threads={a.threads}" + (f"; restored channel from {rest_dir}" if rest_dir is not None else ""), flush=True)

    if a.full:  # all train images, no validation fold; bank sources = every train image passing the SNR rule
        bidx = np.array([row.get(i, -1) for i in tr.ID.values]) if bank_t is not None else None
        for s in (a.train_seeds if a.train_seeds is not None else range(a.seeds)):
            fp = cache / f"full_seed{s}.npz"
            if fp.exists() and not a.overwrite:
                print(f"full seed {s}: cached", flush=True)
                continue
            r = train_one(a, FULL_F, s, Xall, y, None, None, Xte, bank=bank_t, bidx=bidx, wtr=wts(np.arange(len(y))))
            np.savez(fp, test=r["test"], n_train=len(y))
        assemble_full(a, te, cache, rest_dir, rest_sha)
        print(f"total wall time {time.time() - t_all:.0f}s", flush=True)
        return

    for f in folds:
        trn, val = np.where(tr.fold.values != f)[0], np.where(tr.fold.values == f)[0]
        for s in (a.train_seeds if a.train_seeds is not None else range(a.seeds)):
            fp = cache / f"fold{f}_seed{s}.npz"
            if fp.exists() and not a.overwrite:
                print(f"fold {f} seed {s}: cached", flush=True)
                continue
            bidx = np.array([row.get(i, -1) for i in tr.ID.values[trn]]) if bank_t is not None else None
            r = train_one(a, f, s, Xall[trn], y[trn], Xall[val], y[val], Xte, bank=bank_t, bidx=bidx, wtr=wts(trn))
            np.savez(fp, val_idx=val, val=r["val"], val_other=r.get("val_other", np.full(len(val), np.nan)),
                     test=r["test"] if r["test"] is not None else np.zeros(0))

    # assemble whatever is cached
    oof, oof_other, pred = np.full(len(tr), np.nan), np.full(len(tr), np.nan), np.zeros(len(te))
    done, have_test = [], True
    for f in range(5):
        fs = [cache / f"fold{f}_seed{s}.npz" for s in range(a.seeds)]
        if not all(p.exists() for p in fs):
            continue
        zs = [np.load(p) for p in fs]
        val = zs[0]["val_idx"]
        oof[val] = np.mean([z["val"] for z in zs], 0)
        oof_other[val] = np.mean([z["val_other"] for z in zs], 0)
        if any(len(z["test"]) == 0 for z in zs):
            have_test = False
        else:
            pred += np.mean([z["test"] for z in zs], 0) / 5
        done.append(f)
    for f in done:
        m = tr.fold.values == f
        print(f"fold {f}: rmse {rmse(oof[m], y[m]):.3f}" +
              (f" (other weights {rmse(oof_other[m], y[m]):.3f})" if not np.isnan(oof_other[m]).any() else ""))
    terc = snr_tercile_rmse(tr, oof) if len(done) == 5 else None
    if terc is not None:
        print(f"OOF rmse by train ic_ridge_snr tercile (noisy/mid/clean): {terc[0]:.3f} / {terc[1]:.3f} / {terc[2]:.3f}")
    cell, why = cell_rmse(tr, te, oof) if done else (None, None)
    if cell is not None:  # the blend uses the CNN only in this cell (orchestrator, blend_v16)
        print(f"OOF rmse in the fine_noisy blend cell: {cell[0]:.3f} on {cell[1]} images (outside: {cell[2]:.3f} on "
              f"{cell[3]}); cell by fold {cell[4]}" + ("" if len(done) == 5 else f"; done folds {done} only"),
              flush=True)
    elif done:
        print(f"fine_noisy cell rmse n/a ({why})", flush=True)
    if len(done) == 5 and have_test and not a.no_save:
        notes = recipe_notes(a, rest_dir, rest_sha)
        if terc is not None:
            notes += f"; snr-tercile rmse noisy/mid/clean {terc[0]:.2f}/{terc[1]:.2f}/{terc[2]:.2f}"
        if cell is not None:
            notes += f"; fine_noisy cell rmse {cell[0]:.2f} (n={cell[1]})"
        save_experiment(a.name, tr, oof, te, pred, notes=notes)
    else:
        m = ~np.isnan(oof)
        print(f"partial run ({len(done)}/5 folds, test={'yes' if have_test else 'no'}): "
              f"rmse on done folds {rmse(oof[m], y[m]):.3f}; experiments/ not written")
    print(f"total wall time {time.time() - t_all:.0f}s", flush=True)


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet18.a1_in1k")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--head-lr-mult", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1, help="OneCycle pct_start")
    ap.add_argument("--div", type=float, default=25.0)
    ap.add_argument("--final-div", type=float, default=1e3)
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--loss", choices=["mse", "huber"], default="mse")
    ap.add_argument("--huber-beta", type=float, default=1.0)
    ap.add_argument("--pool", choices=["avg", "gem", "avgstd", "avgstd2", "grid4"], default="avg",
                    help="avgstd: concat spatial mean+std of the final map; avgstd2: + std of the stride-8 stage map; "
                         "grid4 (needs --crop 256): concat mean+sd over a fixed 4x4 grid of 64-px cells of the final "
                         "map, aligned to the image origin")
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--drop-path", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay (0 = off)")
    ap.add_argument("--use", choices=["final", "ema"], default="final", help="weights used for OOF/test")
    ap.add_argument("--input", default="raw", choices=["raw", "nlm", "raw+nlm", "raw+rest", "raw+nlm+rest"])
    ap.add_argument("--rest-dir", default="data_restored",
                    help="raw+rest / raw+nlm+rest: folder with {train,test}/<ID>.png written by `python -m src.restore "
                         "apply` (relative to the repo root)")
    ap.add_argument("--norm", default="global", choices=["global", "image"],
                    help="global: fold-level pixel mean/std; image: standardize each image/channel")
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="bilinear upsampling of the network input after augmentation and of every TTA view; "
                         "--crop stays in native pixels, so the network sees crop*scale px crops and 256*scale px "
                         "test views (1.0 = off, bit-identical to before)")
    ap.add_argument("--deg-p", type=float, default=0.0, help="prob. of using a synthetic degraded copy (0 = off)")
    ap.add_argument("--deg-k", type=int, default=8, help="degraded copies per source image in the bank")
    ap.add_argument("--deg-snr-min", type=float, default=0.5, help="bank sources: train ic_ridge_snr above this")
    ap.add_argument("--build-deg", action="store_true",
                    help="build the degradation bank and exit; with a 'rest' input also its restored channel "
                         "(needs data/restore_cache/restore_unet.pt and --rest-dir from the same restorer)")
    ap.add_argument("--cons", type=float, default=0.0,
                    help="consistency weight: degraded twin (prob --deg-p) is trained on the label and pulled "
                         "towards the stop-grad prediction of its original under identical augmentation")
    ap.add_argument("--bright", type=float, default=0.03)
    ap.add_argument("--contrast", type=float, default=0.1)
    ap.add_argument("--noise", type=float, default=0.03, help="max extra gaussian noise sigma (pixel units)")
    ap.add_argument("--noise-p", type=float, default=0.5)
    ap.add_argument("--blur", type=float, default=1.0, help="max blur sigma")
    ap.add_argument("--blur-p", type=float, default=0.0)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=1, help="models per fold (averaged)")
    ap.add_argument("--wN", type=float, default=0.0,
                    help="1/N label-noise weights: each training image's loss is weighted by (N / mean N)^wN, N = its "
                         "label-free grain count (exp logN_cnn of the localized het CNN); 0 = off (default, unchanged)")
    ap.add_argument("--train-seeds", type=int, nargs="*",
                    help="train only these seed indices (run seeds in parallel processes; assemble later)")
    ap.add_argument("--no-save", action="store_true", help="do not write experiments/ even if complete")
    ap.add_argument("--full", action="store_true",
                    help="train on all train images (no validation fold, same fixed schedule) and write only test "
                         "predictions: experiments/<name>/test.csv + full.json, no OOF. Seeds are cached as "
                         "data/cnn_cache/<name>/full_seed{s}.npz (--seeds/--train-seeds as for fold runs). Needs its "
                         "own --name")
    ap.add_argument("--monitor", type=int, default=1, help="log val rmse every N epochs (no selection)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-bf16", dest="bf16", action="store_false", help="disable mixed precision")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--no-test", action="store_true", help="skip test inference (sanity checks)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--name", default="cnn")
    return ap.parse_args(argv)


if __name__ == "__main__":
    main(parse())
