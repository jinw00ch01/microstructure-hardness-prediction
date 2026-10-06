"""End-to-end CNN regression on the shared 5 folds (data/folds.csv).

Honest CV: a fixed epoch schedule is used and the *final* (or EMA) weights predict the
validation fold -- no checkpoint is ever selected on the validation fold.
Training uses random crops (no resizing -> grain/pore scale is preserved) with label-preserving
augmentation only: D4 flips/rot90, mild brightness/contrast jitter, gaussian noise, slight blur.
Inference runs on the full 256x256 image with D4 TTA. Target is standardized per fold.
--input raw|nlm|raw+nlm: optional non-local-means denoised channel (a fixed per-image transform,
nothing is fitted on test). Input normalization stats come from the training fold only.
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
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DATA_DIR, SEED, create_timm, load_test, load_train, read_img, rmse, save_experiment

CACHE = DATA_DIR / "cnn_cache"
RT = {"dev": torch.device("cpu"), "dtype": torch.bfloat16}  # runtime device / autocast dtype


def amp(enabled):
    return torch.autocast(RT["dev"].type, dtype=RT["dtype"], enabled=enabled)


def d4(x, k):  # x: (..., H, W); k in 0..7 -> the 8 dihedral transforms
    x = torch.rot90(x, k % 4, dims=(-2, -1))
    return x.flip(-1) if k >= 4 else x


def _raw_u8(i):
    return np.round(read_img(i) * 255).astype(np.uint8)


def _nlm_u8(i):
    """Non-local-means denoising with h tied to the estimated noise level (as in src/features.py)."""
    import cv2
    from skimage import restoration
    raw = read_img(i)
    h = float(np.clip(restoration.estimate_sigma(raw) * 255 * 1.2, 3, 40))
    return cv2.fastNlMeansDenoising(_raw_u8(i), None, h=h, templateWindowSize=7, searchWindowSize=21)


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


def load_u8(ids, mode="raw"):
    """-> uint8 tensor (N, C, 256, 256); channel 0 is raw when mode contains 'raw'."""
    ids = list(ids)
    chans = []
    for kind in mode.split("+"):
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
        self.pool = GeM() if a.pool == "gem" else None
        self.drop = nn.Dropout(a.drop)
        self.fc = nn.Linear(nf, 1)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        f = self.body.forward_features(x)
        with torch.autocast(x.device.type, enabled=False):  # pooling + regression head in fp32
            f = f.float()
            f = self.pool(f) if self.pool is not None else f.mean((-2, -1))
            return self.fc(self.drop(f)).squeeze(-1)


class Aug:
    """Label-preserving augmentation on a (C, H, W) float image in [0, 1] (before normalization).
    Geometry / brightness / contrast are shared by all channels; blur and noise touch only the
    raw channel (channel 0), never a denoised one."""

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


def to_input(x, mu, sd):  # x: (B, C, H, W) float in [0,1]; mu, sd: (1, C, 1, 1) or None
    if mu is None:  # per-image standardization (--norm image)
        mu, sd = x.mean((2, 3), keepdim=True), x.std((2, 3), keepdim=True) + 1e-3
    return ((x - mu) / sd).contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def predict(model, X, mu, sd, tta=8, bs=32, bf16=True, views=False):
    """Mean over the first `tta` D4 views; views=True returns all views (tta, N)."""
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        x = to_input(X[i:i + bs].float() / 255.0, mu, sd).to(RT["dev"], non_blocking=True)
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


def train_one(a, f, s, Xtr, ytr, Xva, yva, Xte):
    """Train one model with a fixed schedule; return predictions of the chosen weights."""
    seed = SEED + 1000 * s + f
    torch.manual_seed(seed)
    aug = Aug(a, seed)
    rng = np.random.default_rng(seed + 7)  # batch order
    ymu, ysd = float(ytr.mean()), float(ytr.std())
    yt = torch.tensor((ytr - ymu) / ysd, dtype=torch.float32, device=RT["dev"])
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

    t0, step = time.time(), 0
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(n)
        tl = 0.0
        for b in range(spe):
            idx = perm[b * a.bs:(b + 1) * a.bs]
            x = torch.stack([aug(Xtr[i].float() / 255.0) for i in idx])
            x = to_input(x, mu, sd).to(RT["dev"], non_blocking=True)
            with amp(a.bf16):
                p = model(x)
            loss = loss_fn(p.float(), yt[torch.as_tensor(idx, device=RT["dev"])])
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
        msg = f"fold {f} seed {s} ep {ep + 1:3d}/{a.epochs} loss {tl / spe:.4f} lr {sch.get_last_lr()[0]:.2e}"
        if (a.monitor and (ep + 1) % a.monitor == 0) or ep + 1 == a.epochs:  # monitoring only, never selection
            r = rmse(predict(model, Xva, mu, sd, tta=1, bf16=a.bf16) * ysd + ymu, yva)
            msg += f" | val(tta1) {r:.3f}"
            if ema is not None:
                r = rmse(predict(ema.module, Xva, mu, sd, tta=1, bf16=a.bf16) * ysd + ymu, yva)
                msg += f" ema {r:.3f}"
        print(msg + f" | {time.time() - t0:.0f}s", flush=True)

    res = {}
    final_model = ema.module if (ema is not None and a.use == "ema") else model
    pv = predict(final_model, Xva, mu, sd, tta=a.tta, bf16=a.bf16, views=True) * ysd + ymu
    res["val"] = pv.mean(0)
    print(f"fold {f} seed {s} per-view val rmse {[round(rmse(v, yva), 2) for v in pv]} "
          f"view-pred sd {pv.std(0).mean():.2f}", flush=True)
    if ema is not None:  # diagnostics only: the other weights' val score (not used for OOF)
        other = model if a.use == "ema" else ema.module
        res["val_other"] = predict(other, Xva, mu, sd, tta=a.tta, bf16=a.bf16) * ysd + ymu
    res["test"] = predict(final_model, Xte, mu, sd, tta=a.tta, bf16=a.bf16) * ysd + ymu if Xte is not None else None
    print(f"fold {f} seed {s} final val(tta{a.tta}) {rmse(res['val'], yva):.3f}"
          + (f" | other {rmse(res['val_other'], yva):.3f}" if "val_other" in res else "")
          + f" | {time.time() - t0:.0f}s", flush=True)
    return res


def main(a):
    use_cuda = a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available())
    RT["dev"] = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        RT["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        torch.backends.cudnn.benchmark = True
    else:
        a.threads = min(a.threads, 2)  # shared cloud machine: never more than 2 threads
    torch.set_num_threads(a.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    tr, te = load_train(), load_test()
    y = tr.hardness.values.astype(np.float64)
    folds = a.folds if a.folds else list(range(5))
    cache = CACHE / a.name
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "args.json").write_text(json.dumps(vars(a), indent=2))
    t_all = time.time()
    Xall = load_u8(tr.ID, a.input)
    Xte = None if a.no_test else load_u8(te.ID, a.input)
    print(f"loaded {len(Xall)} train / {0 if Xte is None else len(Xte)} test images "
          f"in {time.time() - t_all:.0f}s; device={RT['dev']} amp={RT['dtype'] if a.bf16 else 'off'} "
          f"threads={a.threads}", flush=True)

    for f in folds:
        trn, val = np.where(tr.fold.values != f)[0], np.where(tr.fold.values == f)[0]
        for s in (a.train_seeds if a.train_seeds is not None else range(a.seeds)):
            fp = cache / f"fold{f}_seed{s}.npz"
            if fp.exists() and not a.overwrite:
                print(f"fold {f} seed {s}: cached", flush=True)
                continue
            r = train_one(a, f, s, Xall[trn], y[trn], Xall[val], y[val], Xte)
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
    if len(done) == 5 and have_test and not a.no_save:
        notes = (f"{a.backbone} in={a.input} norm={a.norm} ep{a.epochs} bs{a.bs} crop{a.crop} lr{a.lr} hlr{a.head_lr_mult} wd{a.wd} "
                 f"{a.loss} pool={a.pool} drop{a.drop} dp{a.drop_path} ema{a.ema}->{a.use} seeds{a.seeds} "
                 f"aug(b{a.bright} c{a.contrast} n{a.noise}@{a.noise_p} blur{a.blur}@{a.blur_p}) tta{a.tta}; "
                 f"fixed schedule, no val checkpoint selection")
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
    ap.add_argument("--pool", choices=["avg", "gem"], default="avg")
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--drop-path", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay (0 = off)")
    ap.add_argument("--use", choices=["final", "ema"], default="final", help="weights used for OOF/test")
    ap.add_argument("--input", default="raw", choices=["raw", "nlm", "raw+nlm"])
    ap.add_argument("--norm", default="global", choices=["global", "image"],
                    help="global: fold-level pixel mean/std; image: standardize each image/channel")
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--bright", type=float, default=0.03)
    ap.add_argument("--contrast", type=float, default=0.1)
    ap.add_argument("--noise", type=float, default=0.03, help="max extra gaussian noise sigma (pixel units)")
    ap.add_argument("--noise-p", type=float, default=0.5)
    ap.add_argument("--blur", type=float, default=1.0, help="max blur sigma")
    ap.add_argument("--blur-p", type=float, default=0.0)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=1, help="models per fold (averaged)")
    ap.add_argument("--train-seeds", type=int, nargs="*",
                    help="train only these seed indices (run seeds in parallel processes; assemble later)")
    ap.add_argument("--no-save", action="store_true", help="do not write experiments/ even if complete")
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
