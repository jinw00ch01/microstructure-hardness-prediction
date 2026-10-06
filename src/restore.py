"""Image restoration (denoise + deblur + dark-contrast recovery) for degraded microstructure images (cnn-trainer).

A compact U-Net is trained on synthetic pairs  _degrade_v2(clean) -> clean  built only from the cleanest TRAIN images
(ic_ridge_snr > 0.9; src.features._degrade_v2 is imported read-only: the feature-engineer's degradation model matched to
the real noise / blur / dark-contrast per noise band). No labels and no test images are used for training.

Photometry: _degrade_v2 also applies a random global gain/offset and a smooth shading field, which cannot be undone
uniquely. The training target is therefore the clean image re-expressed in the degraded image's photometry. The
deterministic part of the degradation (NLM layer, matrix background, pores, dark-side compression with the returned
`contrast`, blur with the returned `sb`) is reconstructed noise-free, then a smooth cubic gain surface + offset is fitted
from that reconstruction to the degraded image on all pixels (well identified), and applied to the clean image.
The network only removes noise / blur / dark-contrast loss and never re-normalises grey levels. About 15% of training samples are identity pairs (clean -> clean) so clean images pass
through. Input channels: image + its estimated noise level (constant map). Output = input + predicted residual.

Applied as a fixed per-image transform (8-view D4 average) to all 1500 images -> data_restored/{train,test}/<ID>.png
(8-bit, same names) with train.csv / sample_submission.csv / folds.csv copied, so DATA_DIR=data_restored works.

  python -m src.restore bank --part 0 --nparts 2 &  python -m src.restore bank --part 1 --nparts 2   # pairs, 1 core each
  python -m src.restore train --threads 2                                                          # -> model + val report
  python -m src.restore apply --part 0 --nparts 2 &  python -m src.restore apply --part 1 --nparts 2
  python -m src.restore check                                                                      # pass-through + montages

High-noise specialist (--tag; untagged commands behave exactly as before). The same U-Net fine-tuned from
restore_unet.pt on high-noise pairs only (q >= 0.4, i.e. _degrade_v2 noise_t >= 10.8): the original bank's pairs plus a
dedicated bank drawn by exact rejection sampling of _degrade_v2 on its latent q (same 120 train / 11 held-out sources).
A tag writes model + checkpoint to data/restore_cache/<tag>/, reports to experiments/restore/<tag>/ and images to
data_restored_<tag>/. `compare` evaluates restorers on the same held-out pairs (paired, by noise band) and optionally
the fidelity of v3 features on restored high-noise copies. `apply --apply-noise-min` restores only images whose raw
ic_noise (data/features_v3.parquet) passes a fixed cut and copies the rest from --fill-from.

  python -m src.restore bank --bank-name hn --q-min 0.4 --k-train 16 --k-val 24 --seed-base 9 --part P --nparts 4  # P=0..3
  python -m src.restore train --tag hn --banks main,hn --train-q-min 0.4 --init data/restore_cache/restore_unet.pt \
      --fp32 --threads 4 --patch 128 --bs 16 --steps 3000 --lr 5e-4 --identity 0 --mon-n 48 --ckpt-every 250 --no-validate
  python -m src.restore compare --tags base,hn --banks main,hn --threads 4 --jobs 4 --features --report-tag hn
  python -m src.restore apply --tag hn --apply-noise-min 11.92 --fill-from data_restored --part P --nparts 4  # P=0..3
  python -m src.restore check --tag hn --fill-from data_restored
The specialist is much worse than the original on low-noise inputs (never trained there): use it only above the cut.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DATA_DIR, EXP_DIR, ROOT, load_test, load_train

CACHE = DATA_DIR / "restore_cache"
OUT_DIR = ROOT / "data_restored"
REPORT_DIR = EXP_DIR / "restore"
SRC_SNR, N_VAL, K_TRAIN, K_VAL = 0.9, 11, 32, 6
MU, SD, SIG_SCALE = 0.55, 0.10, 20.0  # input normalisation (pixel/255) and noise-channel scale (grey levels)
BASE_CH = (32, 64, 96, 128)
CHECK_FEATS = ["ic_seg_fd93", "ic_fdo_93", "ic_gmm_w", "ic_acg_len50_perp", "ic_seg_nfrac91"]


def paths(tag=""):
    """'' = the original restorer's locations (unchanged); a tag gets its own model, checkpoint, report and images."""
    if not tag:
        return {"cache": CACHE, "model": CACHE / "restore_unet.pt", "ckpt": CACHE / "restore_ckpt.pt",
                "report": REPORT_DIR, "out": OUT_DIR}
    c = CACHE / tag
    return {"cache": c, "model": c / "restore_unet.pt", "ckpt": c / "restore_ckpt.pt", "report": REPORT_DIR / tag,
            "out": ROOT / f"data_restored_{tag}"}


def bank_dir(name=""):
    return CACHE if name in ("", "main") else CACHE / f"bank_{name}"


def _bank_names(s):
    return ["" if n == "main" else n for n in s.split(",") if n]


def _root_path(p):
    return Path(p) if os.path.isabs(p) else ROOT / p


def _threads(a):
    return max(1, min(a.threads, os.cpu_count() or 1))


# ----------------------------------------------------------------------------------------------- data
def _read8(i, root=DATA_DIR):
    split = "train" if i.startswith("TRAIN") else "test"
    return cv2.imread(str(root / split / f"{i}.png"), cv2.IMREAD_GRAYSCALE)


def _sigma(img8):
    from skimage import restoration
    return float(restoration.estimate_sigma(img8.astype(np.float32)))


def sources():
    v3 = pd.read_parquet(DATA_DIR / "features_v3.parquet", columns=["ID", "ic_ridge_snr"])
    tr_ids = set(load_train().ID)
    src = sorted(v3[v3.ID.isin(tr_ids) & (v3.ic_ridge_snr > SRC_SNR)].ID)
    val = sorted(np.random.default_rng(0).choice(src, N_VAL, replace=False).tolist())
    return [i for i in src if i not in val], val


def _src_layers(clean8):
    """Deterministic per-source layers exactly as computed inside src.features._degrade_v2."""
    from scipy import ndimage as ndi
    from .features import _bg_matrix
    img = clean8.astype(np.float32)
    den = cv2.fastNlMeansDenoising(clean8, None, h=float(np.clip(_sigma(clean8), 2.0, 30.0)), templateWindowSize=5,
                                   searchWindowSize=21).astype(np.float32)
    bg = _bg_matrix(den)
    pore = ndi.binary_dilation(cv2.GaussianBlur(den / bg, (0, 0), 1.0) < 0.68, iterations=1)
    return img, den, bg, pore


def _reconstruct(layers, p):
    """Noise-free, pre-photometric degraded image from the parameters _degrade_v2 returns (contrast, sb)."""
    img, den, bg, pore = layers
    d = den - bg
    grain = bg + np.where(d < 0, p["contrast"] * d, d) + (img - den)
    if p["sb"] > 0.3:
        grain = cv2.GaussianBlur(grain, (0, 0), p["sb"])
    return np.where(pore, img, grain)


_YY, _XX = np.mgrid[:256, :256].astype(np.float32) / 127.5 - 1.0
_BASIS = [np.ones_like(_XX), _XX, _YY, _XX ** 2, _YY ** 2, _XX * _YY, _XX ** 3, _YY ** 3, _XX ** 2 * _YY, _XX * _YY ** 2]


def photometric_fit(recon, deg8):
    """deg ~= gain(x, y) * recon + offset over all pixels (both smoothed, sigma 2); gain = cubic surface -> 11 coefs."""
    r = cv2.GaussianBlur(recon.astype(np.float32), (0, 0), 2.0)[::2, ::2]
    d = cv2.GaussianBlur(deg8.astype(np.float32), (0, 0), 2.0)[::2, ::2]
    A = np.stack([r * b[::2, ::2] for b in _BASIS] + [np.ones_like(r)], -1).reshape(-1, len(_BASIS) + 1)
    coef, *_ = np.linalg.lstsq(A.astype(np.float64), d.ravel().astype(np.float64), rcond=None)
    return coef.astype(np.float32)


def photometric_target(clean8, coef):
    gain = sum(k * b for k, b in zip(coef[:-1], _BASIS))
    return np.clip(gain * clean8.astype(np.float32) + coef[-1], 0, 255)


def _accepted(prefix, num, n, q_min):
    """First n degradation indices k whose _degrade_v2 latent q is >= q_min. q is the first draw of
    default_rng([prefix, num, k]) inside _degrade_v2 (everything before it is deterministic), so this is exact
    rejection sampling of _degrade_v2 conditioned on q >= q_min. q_min <= 0 gives range(n) (the original bank)."""
    if q_min <= 0:
        return list(range(n))
    ks, k = [], 0
    while len(ks) < n:
        if np.random.default_rng([prefix, num, k]).uniform() >= q_min:
            ks.append(k)
        k += 1
    return ks


def build_bank(part, nparts, name="", q_min=0.0, k_train=K_TRAIN, k_val=K_VAL, seed_base=7):
    """Defaults = the original bank (rng prefixes 7 train / 8 val, all q). name -> data/restore_cache/bank_<name>/."""
    from .features import _degrade_v2
    cv2.setNumThreads(1)
    trn, val = sources()
    num = lambda i: int(i.split("_")[-1])
    jobs = ([(i, k, "train") for i in trn for k in _accepted(seed_base, num(i), k_train, q_min)] +
            [(i, k, "val") for i in val for k in _accepted(seed_base + 1, num(i), k_val, q_min)])
    jobs = sorted(jobs[part::nparts])  # group by source so per-source layers are computed once
    layers, out = {}, {"id": [], "kind": [], "deg": [], "coef": [], "sigma": [], "noise_t": [], "q": [], "sb": [],
                       "contrast": [], "fit_rms": [], "gain0": []}
    if name:
        out["k"] = []
    t = time.time()
    for n, (i, k, kind) in enumerate(jobs):
        clean = _read8(i)
        if i not in layers:
            layers = {i: _src_layers(clean)}  # sources are contiguous within a part, keep one
        rng = np.random.default_rng([seed_base if kind == "train" else seed_base + 1, num(i), k])
        deg, p = _degrade_v2(clean, rng)
        assert p["q"] >= q_min, (i, k, p["q"])
        if name:
            out["k"].append(k)
        recon = _reconstruct(layers[i], p)
        coef = photometric_fit(recon, deg)
        gain = sum(c_ * b for c_, b in zip(coef[:-1], _BASIS))
        r = cv2.GaussianBlur(deg.astype(np.float32), (0, 0), 2.0) - cv2.GaussianBlur(gain * recon + coef[-1], (0, 0), 2.0)
        for key, v in (("id", i), ("kind", kind), ("deg", deg), ("coef", coef), ("sigma", _sigma(deg)),
                       ("noise_t", p["noise_t"]), ("q", p["q"]), ("sb", p["sb"]), ("contrast", p["contrast"]),
                       ("fit_rms", float(np.sqrt(np.mean(r ** 2)))), ("gain0", float(coef[0]))):
            out[key].append(v)
        if n % 200 == 0:
            print(f"part {part}: {n}/{len(jobs)} {time.time() - t:.0f}s", flush=True)
    d = bank_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"bank_part{part}.npz", **{k: np.array(v) for k, v in out.items()})
    print(f"part {part}: {len(jobs)} pairs in {time.time() - t:.0f}s; median fit rms {np.median(out['fit_rms']):.2f}",
          flush=True)


def load_bank(names=("",)):
    """Concatenate the parts of one or more banks ('' = the original); b['bank'] tells them apart."""
    zs, tags = [], []
    for nm in names:
        parts = sorted(bank_dir(nm).glob("bank_part*.npz"))
        assert parts, f"no bank_part*.npz in {bank_dir(nm)}"
        zs += [np.load(p) for p in parts]
        tags += [nm or "main"] * len(parts)
    keys = [k for k in zs[0].files if all(k in z.files for z in zs)]
    b = {k: np.concatenate([z[k] for z in zs]) for k in keys}
    b["bank"] = np.concatenate([np.full(len(z["id"]), t) for z, t in zip(zs, tags)])
    ids = sorted(set(b["id"].tolist()))
    clean = {i: _read8(i) for i in ids}
    return b, clean


# ----------------------------------------------------------------------------------------------- model
def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, ch=BASE_CH, cin=2):
        super().__init__()
        self.ch = tuple(ch)
        self.e1, self.e2, self.e3, self.b = _block(cin, ch[0]), _block(ch[0], ch[1]), _block(ch[1], ch[2]), _block(ch[2], ch[3])
        self.u3, self.u2, self.u1 = _block(ch[3] + ch[2], ch[2]), _block(ch[2] + ch[1], ch[1]), _block(ch[1] + ch[0], ch[0])
        self.out = nn.Conv2d(ch[0], 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        b = self.b(F.avg_pool2d(e3, 2))
        d3 = self.u3(torch.cat([up(b), e3], 1))
        d2 = self.u2(torch.cat([up(d3), e2], 1))
        d1 = self.u1(torch.cat([up(d2), e1], 1))
        with torch.autocast(x.device.type, enabled=False):
            return x[:, :1].float() + self.out(d1.float())


def _ch_from_state(sd):
    return tuple(int(sd[f"{k}.0.weight"].shape[0]) for k in ("e1", "e2", "e3", "b"))


def widen_init(model, old_sd):
    """Zero-expanded init of a wider UNet from a narrower state dict: the old weights fill the leading channels and
    every weight that reads a new channel is zero, so the wider network starts as exactly the old function."""
    o, n = _ch_from_state(old_sd), model.ch
    segs = {"e1.0": [(2, 2)], "e1.2": [(o[0], n[0])], "e2.0": [(o[0], n[0])], "e2.2": [(o[1], n[1])],
            "e3.0": [(o[1], n[1])], "e3.2": [(o[2], n[2])], "b.0": [(o[2], n[2])], "b.2": [(o[3], n[3])],
            "u3.0": [(o[3], n[3]), (o[2], n[2])], "u3.2": [(o[2], n[2])],  # decoder input = cat(upsampled, skip)
            "u2.0": [(o[2], n[2]), (o[1], n[1])], "u2.2": [(o[1], n[1])],
            "u1.0": [(o[1], n[1]), (o[0], n[0])], "u1.2": [(o[0], n[0])], "out": [(o[0], n[0])]}
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    for name, seg in segs.items():
        w_old, w = old_sd[f"{name}.weight"], sd[f"{name}.weight"]
        r, oi, ni = w_old.shape[0], 0, 0
        for lo, ln in seg:
            w[:, ni + lo:ni + ln] = 0
            w[:r, ni:ni + lo] = w_old[:, oi:oi + lo]
            oi, ni = oi + lo, ni + ln
        sd[f"{name}.bias"][:r] = old_sd[f"{name}.bias"]
    model.load_state_dict(sd)


def to_input(img8, sigma):
    """uint8 (B, H, W) tensor + per-image sigma (B,) -> network input (B, 2, H, W), channels_last."""
    x = (img8.float() / 255.0 - MU) / SD
    s = (torch.as_tensor(sigma, dtype=torch.float32) / SIG_SCALE).view(-1, 1, 1, 1).expand(-1, 1, *x.shape[-2:])
    return torch.cat([x[:, None], s], 1).contiguous(memory_format=torch.channels_last)


def from_output(y):
    return ((y[:, 0] * SD + MU) * 255.0).clamp(0, 255)


def d4(x, k):
    x = torch.rot90(x, k % 4, dims=(-2, -1))
    return x.flip(-1) if k >= 4 else x


def d4_inv(x, k):
    if k >= 4:
        x = x.flip(-1)
    return torch.rot90(x, -(k % 4), dims=(-2, -1))


@torch.no_grad()
def restore(model, imgs8, sigmas, tta=8, bs=8, amp=None):
    """imgs8: uint8 tensor (N, H, W) -> float restored (N, H, W) in grey levels (mean over D4 views).
    amp: bf16 autocast; None = the model's own setting (model.amp, True for the original restorer)."""
    amp = getattr(model, "amp", True) if amp is None else amp
    model.eval()
    out = []
    for s in range(0, len(imgs8), bs):
        x = to_input(imgs8[s:s + bs], sigmas[s:s + bs])
        acc = 0
        for k in range(tta):
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
                y = model(d4(x, k).contiguous(memory_format=torch.channels_last))
            acc = acc + d4_inv(y.float(), k)
        out.append(from_output(acc / tta))
    return torch.cat(out)


def psnr(a, b):
    return float(10 * np.log10(255.0 ** 2 / max(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2), 1e-9)))


# ----------------------------------------------------------------------------------------------- training
def train(a):
    pth = paths(a.tag)
    torch.set_num_threads(_threads(a))
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    names = _bank_names(a.banks)
    b, clean = load_bank(names)
    tr = np.where((b["kind"] == "train") & (b["q"] >= a.train_q_min))[0]
    va = np.where((b["kind"] == "val") & (b["q"] >= a.train_q_min))[0]
    va_mon = va if not a.mon_n or a.mon_n >= len(va) else va[np.linspace(0, len(va) - 1, a.mon_n).round().astype(int)]
    print(f"bank: {len(tr)} train pairs from {len(set(b['id'][tr]))} sources, {len(va)} val pairs", flush=True)
    clean_sig = {i: _sigma(c) for i, c in clean.items()} if a.identity > 0 else {}
    deg = torch.from_numpy(b["deg"])
    tgt_cache = {}

    def target(j):  # cached as float16 (0.06 grey-level resolution at 128-255) to halve memory
        if j not in tgt_cache:
            tgt_cache[j] = torch.from_numpy(photometric_target(clean[b["id"][j]], b["coef"][j])).half()
        return tgt_cache[j].float()

    amp = not a.fp32
    ch = tuple(int(round(c * a.width)) for c in BASE_CH)
    model = UNet(ch=ch)
    if a.init:
        sd0 = torch.load(_root_path(a.init), map_location="cpu")
        if _ch_from_state(sd0) == ch:
            model.load_state_dict(sd0)
        else:
            widen_init(model, sd0)
        print(f"init from {a.init} (channels {_ch_from_state(sd0)} -> {ch})", flush=True)
    model = model.to(memory_format=torch.channels_last)
    model.amp = amp
    print(f"UNet params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=a.pct_start)
    P = a.patch
    t0 = time.time()
    run = 0.0
    # checkpoint / resume (a container suspend kills background jobs): weights + optimizer + scheduler + RNG + step
    ckpt, start = pth["ckpt"], 1
    cfg = {k: getattr(a, k) for k in ("steps", "bs", "patch", "lr", "grad_w", "identity")}
    extra = {"tag": a.tag, "banks": ",".join(n or "main" for n in names), "train_q_min": a.train_q_min,
             "init": a.init, "width": a.width, "fp32": a.fp32, "pct_start": a.pct_start}
    dflt = {"tag": "", "banks": "main", "train_q_min": 0.0, "init": "", "width": 1.0, "fp32": False, "pct_start": 0.05}
    cfg.update({k: v for k, v in extra.items() if v != dflt[k]})  # default runs keep the original cfg (and checkpoint)
    if ckpt.exists() and not a.fresh:
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        if st["cfg"] == cfg:
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            sch.load_state_dict(st["sch"])
            rng.bit_generator.state = st["rng"]
            torch.set_rng_state(st["torch_rng"])
            start = st["step"] + 1
            print(f"resumed from {ckpt} at step {st['step']}", flush=True)
        else:
            print(f"checkpoint cfg {st['cfg']} != {cfg}: starting fresh", flush=True)

    def save_ckpt(step):
        pth["cache"].mkdir(parents=True, exist_ok=True)
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"cfg": cfg, "step": step, "model": model.state_dict(), "opt": opt.state_dict(),
                    "sch": sch.state_dict(), "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state()}, tmp)
        os.replace(tmp, ckpt)

    if a.init and start == 1:  # starting point of a fine-tune on the monitoring pairs
        rest = restore(model, deg[va_mon], b["sigma"][va_mon], tta=1).numpy()
        pa = np.mean([psnr(rest[n], target(j)) for n, j in enumerate(va_mon)])
        print(f"  step 0 val PSNR (tta1, {len(va_mon)} pairs) restored {pa:.2f} dB", flush=True)
    for step in range(start, a.steps + 1):
        model.train()
        xs, ys, sg = [], [], []
        for _ in range(a.bs):
            if rng.random() < a.identity:
                i = b["id"][tr[rng.integers(len(tr))]]
                x = y = torch.from_numpy(clean[i]).float()
                s = clean_sig[i]
            else:
                j = tr[rng.integers(len(tr))]
                x, y, s = deg[j].float(), target(j), float(b["sigma"][j])
            r, c = rng.integers(0, 256 - P + 1, 2)
            k = int(rng.integers(8))
            xs.append(d4(x[r:r + P, c:c + P], k))
            ys.append(d4(y[r:r + P, c:c + P], k))
            sg.append(s)
        xin = to_input(torch.stack(xs), sg)
        yt = (torch.stack(ys)[:, None] / 255.0 - MU) / SD
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
            out = model(xin)
        out = out.float()
        loss = F.l1_loss(out, yt)
        if a.grad_w > 0:
            gx = lambda t: t[..., :, 1:] - t[..., :, :-1]
            gy = lambda t: t[..., 1:, :] - t[..., :-1, :]
            loss = loss + a.grad_w * (F.l1_loss(gx(out), gx(yt)) + F.l1_loss(gy(out), gy(yt)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
        run += loss.item()
        if step % 100 == 0:
            print(f"step {step}/{a.steps} loss {run / 100:.4f} lr {sch.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s",
                  flush=True)
            run = 0.0
        if step % a.eval_every == 0 or step == a.steps:  # monitoring only (fixed schedule, last weights are kept)
            rest = restore(model, deg[va_mon], b["sigma"][va_mon], tta=1).numpy()
            pa = np.mean([psnr(rest[n], target(j)) for n, j in enumerate(va_mon)])
            pb = np.mean([psnr(deg[j].numpy(), target(j)) for j in va_mon])
            print(f"  val PSNR (tta1, {len(va_mon)} pairs) degraded {pb:.2f} -> restored {pa:.2f} dB", flush=True)
        if step % a.ckpt_every == 0 or step == a.steps:
            save_ckpt(step)
    pth["cache"].mkdir(parents=True, exist_ok=True)
    tmp = pth["model"].with_suffix(".tmp")
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, pth["model"])
    if a.tag:
        (pth["cache"] / "meta.json").write_text(json.dumps({"ch": list(ch), "amp": amp, "cfg": cfg}, indent=2))
    print(f"saved {pth['model']} after {time.time() - t0:.0f}s", flush=True)
    if not a.no_validate:
        validate(model, b, va, target, pth["report"])


def _target_info(t):
    """Reference maps of a (float) target image used by the held-out metrics."""
    from skimage import filters
    rid = filters.sato(t, sigmas=[1.0, 1.5], black_ridges=True)
    den = cv2.GaussianBlur(t, (0, 0), 2.0)
    return {"t": t, "rid": rid, "bmask": rid > np.percentile(rid, 85), "den": den,
            "dark": den < np.percentile(den, 20), "mat": den > np.percentile(den, 50)}


def _img_metrics(img, ti, tag):
    """PSNR, PSNR on boundary pixels (top 15% target ridge), ridge-map correlation, dark-phase contrast ratio."""
    from skimage import filters
    t, den, dark, mat, bmask = ti["t"], ti["den"], ti["dark"], ti["mat"], ti["bmask"]
    r2 = filters.sato(img, sigmas=[1.0, 1.5], black_ridges=True)
    sm = cv2.GaussianBlur(img, (0, 0), 2.0)
    return {f"psnr_{tag}": psnr(img, t), f"psnr_bd_{tag}": psnr(img[bmask], t[bmask]),
            f"ridge_corr_{tag}": float(np.corrcoef(r2.ravel(), ti["rid"].ravel())[0, 1]),
            f"dark_contrast_{tag}": float((sm[mat].mean() - sm[dark].mean()) /
                                          max(den[mat].mean() - den[dark].mean(), 1e-3))}


def validate(model, b, va, target, report_dir=REPORT_DIR):
    """PSNR before/after by noise band, boundary PSNR / ridge correlation and dark-phase contrast on held-out sources."""
    deg = torch.from_numpy(b["deg"][va])
    rest = restore(model, deg, b["sigma"][va], tta=8).numpy()
    rows = []
    for n, j in enumerate(va):
        ti = _target_info(target(j).numpy())
        row = {"noise_t": float(b["noise_t"][j]), "sb": float(b["sb"][j]), "contrast": float(b["contrast"][j])}
        row.update(_img_metrics(deg[n].numpy().astype(np.float32), ti, "deg"))
        row.update(_img_metrics(rest[n], ti, "rest"))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["band"] = pd.cut(df.noise_t, [0, 9, 15, 99], labels=["low noise (<9)", "mid (9-15)", "high (>15)"])
    cols = ["psnr_deg", "psnr_rest", "psnr_bd_deg", "psnr_bd_rest", "ridge_corr_deg", "ridge_corr_rest",
            "dark_contrast_deg", "dark_contrast_rest"]
    summ = df.groupby("band", observed=True)[cols].mean().round(3)
    summ["n"] = df.groupby("band", observed=True).size()
    allm = df[cols].mean().round(3)
    print(f"held-out validation ({len(va)} pairs of the 11 held-out sources; targets = clean in degraded photometry):")
    print(summ.to_string())
    print("all:", allm.to_dict())
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "val_metrics.json").write_text(json.dumps(
        {"by_band": summ.reset_index().astype({"band": str}).to_dict(orient="records"), "all": allm.to_dict()}, indent=2))
    df.to_csv(report_dir / "val_pairs.csv", index=False)
    # held-out synthetic montage: degraded | restored | target (128 px crops), 6 pairs spanning the noise range
    order = np.argsort(b["noise_t"][va])
    tiles, c = [], 128
    u8 = lambda z: np.clip(np.round(z), 0, 255).astype(np.uint8)
    for n in order[np.linspace(0, len(order) - 1, 6).round().astype(int)]:
        j, sep = va[n], np.full((c, 3), 255, np.uint8)
        row = np.concatenate([deg[n].numpy()[:c, :c], sep, u8(rest[n][:c, :c]), sep, u8(target(j).numpy()[:c, :c])], 1)
        cv2.putText(row, f"nt{b['noise_t'][j]:.0f} sb{b['sb'][j]:.1f}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255, 1)
        tiles.append(np.pad(row, ((0, 5), (0, 5)), constant_values=255))
    cv2.imwrite(str(report_dir / "montage_val_synthetic.png"),
                np.concatenate([np.concatenate(tiles[k:k + 2], 1) for k in range(0, len(tiles), 2)], 0))


# ----------------------------------------------------------------------------------------------- apply / checks
def load_model(tag=""):
    """The original restorer ('') or a tagged specialist (channels from the weights, bf16 setting from meta.json)."""
    pth = paths(tag)
    sd = torch.load(pth["model"], map_location="cpu")
    m = UNet(ch=_ch_from_state(sd))
    m.load_state_dict(sd)
    meta = pth["cache"] / "meta.json"
    m.amp = bool(json.loads(meta.read_text()).get("amp", True)) if tag and meta.exists() else True
    return m.to(memory_format=torch.channels_last).eval()


def validate_saved(a):
    torch.set_num_threads(_threads(a))
    b, clean = load_bank(_bank_names(a.banks))
    va = np.where((b["kind"] == "val") & (b["q"] >= a.train_q_min))[0]
    validate(load_model(a.tag), b, va, lambda j: torch.from_numpy(photometric_target(clean[b["id"][j]], b["coef"][j])),
             paths(a.tag)["report"])


def apply(a):
    """Restore every image (default) or, with --apply-noise-min, only those whose raw ic_noise >= the cut, copying the
    rest byte-for-byte from --fill-from. Writes are atomic; --resume skips images already written."""
    pth = paths(a.tag)
    out_dir = pth["out"]
    assert DATA_DIR.resolve() != out_dir.resolve(), f"run apply with the original DATA_DIR (unset), not {out_dir.name}"
    torch.set_num_threads(_threads(a))
    cv2.setNumThreads(1)
    ids = load_train().ID.tolist() + load_test().ID.tolist()
    ids = ids[a.part::a.nparts]
    use, fill = set(ids), None
    if a.apply_noise_min > 0:  # fixed per-image rule on the raw image's noise estimate (nothing fitted)
        assert a.fill_from, "--apply-noise-min needs --fill-from (the set the other images are copied from)"
        fill = _root_path(a.fill_from)
        nz = pd.read_parquet(DATA_DIR / "features_v3.parquet", columns=["ID", "ic_noise"]).set_index("ID").ic_noise
        use = {i for i in ids if nz[i] >= a.apply_noise_min}
        pth["cache"].mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ID": ids, "ic_noise": [float(nz[i]) for i in ids], "specialist": [int(i in use) for i in ids]}
                     ).to_csv(pth["cache"] / f"applied_part{a.part}.csv", index=False)
        print(f"part {a.part}: {len(use)} of {len(ids)} images have ic_noise >= {a.apply_noise_min}", flush=True)
    model = load_model(a.tag) if use else None
    for split in ("train", "test"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)
    tmpdir = out_dir.parent / f".{out_dir.name}_tmp{a.part}"  # outside the output dir so no stray files appear there
    tmpdir.mkdir(exist_ok=True)
    rel = lambda i: Path("train" if i.startswith("TRAIN") else "test") / f"{i}.png"
    out = lambda i: out_dir / rel(i)
    if a.resume:
        done = [i for i in ids if out(i).exists()]
        ids = [i for i in ids if not out(i).exists()]
        print(f"part {a.part}: resume, {len(done)} already written, {len(ids)} to go", flush=True)
    t = time.time()
    run_ids = [i for i in ids if i in use]
    for s in range(0, len(run_ids), 16):
        chunk = run_ids[s:s + 16]
        imgs = [_read8(i) for i in chunk]
        rest = restore(model, torch.from_numpy(np.stack(imgs)), [_sigma(x) for x in imgs], tta=a.tta).numpy()
        for i, r in zip(chunk, rest):
            cv2.imwrite(str(tmpdir / f"{i}.png"), np.clip(np.round(r), 0, 255).astype(np.uint8))
            os.replace(tmpdir / f"{i}.png", out(i))
        if s % 160 == 0:
            print(f"part {a.part}: {s + len(chunk)}/{len(run_ids)} {time.time() - t:.0f}s", flush=True)
    copy_ids = [i for i in ids if i not in use]
    for i in copy_ids:
        shutil.copyfile(fill / rel(i), tmpdir / f"{i}.png")
        os.replace(tmpdir / f"{i}.png", out(i))
    for f in ("train.csv", "sample_submission.csv", "folds.csv"):
        shutil.copy(DATA_DIR / f, out_dir / f)
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"part {a.part}: restored {len(run_ids)}, copied {len(copy_ids)} images in {time.time() - t:.0f}s", flush=True)


def check(a):
    """Pass-through on clean real images, change by SNR tercile, and before/after montages."""
    ids = load_train().ID.tolist() + load_test().ID.tolist()
    miss = [i for i in ids if not (OUT_DIR / ("train" if i.startswith("TRAIN") else "test") / f"{i}.png").exists()]
    assert not miss, f"{len(miss)} restored images missing"
    v3 = pd.read_parquet(DATA_DIR / "features_v3.parquet", columns=["ID", "ic_ridge_snr", "ic_noise"]).set_index("ID")
    trn_src, val_src = sources()
    rows = []
    for i in ids:
        o, r = _read8(i).astype(np.float32), _read8(i, OUT_DIR).astype(np.float32)
        rows.append({"ID": i, "snr": v3.loc[i, "ic_ridge_snr"], "noise": v3.loc[i, "ic_noise"], "mad": float(np.abs(o - r).mean()),
                     "mean_shift": float(r.mean() - o.mean()), "psnr": psnr(o, r),
                     "lowpass_mad": float(np.abs(cv2.GaussianBlur(o, (0, 0), 3) - cv2.GaussianBlur(r, (0, 0), 3)).mean()),
                     "role": "train_source" if i in trn_src else ("val_source" if i in val_src else "other")})
    df = pd.DataFrame(rows)
    df["snr_band"] = pd.cut(df.snr, [-1, 0.283, 0.774, 0.9, 99], labels=["noisy <0.28", "mid", "0.77-0.9", "clean >0.9"])
    summ = df.groupby("snr_band", observed=True)[["mad", "lowpass_mad", "mean_shift", "psnr", "noise"]].median().round(2)
    summ["n"] = df.groupby("snr_band", observed=True).size()
    print("real images, original vs restored (medians):")
    print(summ.to_string())
    cl = df[df.snr > 0.9]
    print("clean (snr > 0.9) by role:", cl.groupby("role")[["mad", "lowpass_mad", "psnr"]].median().round(2).to_dict())
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_DIR / "apply_change.csv", index=False)
    # montages: real noisy (snr < 0.3) and real clean
    def montage(sel, fn, crop=160):
        tiles = []
        for i in sel:
            o, r = _read8(i)[:crop, :crop], _read8(i, OUT_DIR)[:crop, :crop]
            row = np.concatenate([o, np.full((crop, 4), 255, np.uint8), r], 1)
            cv2.putText(row, f"{i[-4:]} snr{v3.loc[i, 'ic_ridge_snr']:.2f}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255, 1)
            tiles.append(np.pad(row, ((0, 6), (0, 6)), constant_values=255))
        rows_ = [np.concatenate(tiles[k:k + 2], 1) for k in range(0, len(tiles), 2)]
        cv2.imwrite(str(REPORT_DIR / fn), np.concatenate(rows_, 0))
    noisy = df[df.snr < 0.3].sort_values("snr")
    pick = noisy.iloc[np.linspace(0, len(noisy) - 1, 8).round().astype(int)].ID.tolist()
    montage(pick, "montage_noisy_before_after.png")
    clean_pick = df[(df.snr > 0.9) & (df.role == "other")].sort_values("snr").ID.tolist()
    clean_pick = [clean_pick[k] for k in np.linspace(0, len(clean_pick) - 1, 4).round().astype(int)]
    mid = df[(df.snr > 0.3) & (df.snr < 0.77)].sort_values("snr")
    mid_pick = mid.iloc[np.linspace(0, len(mid) - 1, 4).round().astype(int)].ID.tolist()
    montage(mid_pick + clean_pick, "montage_mid_clean_before_after.png")
    # method noise (original - restored, x3 + 128) on 4 noisy + 2 clean images: structure in it = removed signal
    tiles, c = [], 160
    for i in pick[::2] + clean_pick[1::2]:
        o, r = _read8(i)[:c, :c], _read8(i, OUT_DIR)[:c, :c]
        res = np.clip(3.0 * (o.astype(np.float32) - r) + 128, 0, 255).astype(np.uint8)
        sep = np.full((c, 3), 255, np.uint8)
        row = np.concatenate([o, sep, r, sep, res], 1)
        cv2.putText(row, f"{i[-4:]} snr{v3.loc[i, 'ic_ridge_snr']:.2f}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255, 1)
        tiles.append(np.pad(row, ((0, 5), (0, 0)), constant_values=255))
    cv2.imwrite(str(REPORT_DIR / "montage_method_noise.png"), np.concatenate(tiles, 0))
    print("montages:", REPORT_DIR / "montage_noisy_before_after.png", REPORT_DIR / "montage_mid_clean_before_after.png")


def _boot_ci(stat, groups, n_boot=2000, seed=0):
    """90% interval of stat(index array) under a bootstrap over sources (pairs of one source are correlated)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    vals = [stat(np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])) for _ in range(n_boot)]
    return [round(float(np.percentile(vals, 5)), 4), round(float(np.percentile(vals, 95)), 4)]


def _pair_row(meta, t, imgs):
    cv2.setNumThreads(1)
    ti = _target_info(t)
    row = dict(meta)
    for tag, img in imgs:
        row.update(_img_metrics(img, ti, tag))
    return row


def _v3_feats(key, img):
    from .features import extract_v3
    cv2.setNumThreads(1)
    f = extract_v3(key, img=img)
    return {"key": key, **{c: float(f[c]) for c in CHECK_FEATS}}


def compare(a):
    """Paired held-out comparison of restorers (--tags, 'base' = the original) on the same synthetic pairs: every val
    pair of the given banks (all from the 11 held-out sources), by noise band, with source-bootstrap 90% intervals of
    the paired differences. --features: R^2 / bias of a few v3 measures on the restored high-noise copies
    (estimated sigma >= --rule-noise) against the clean source's value."""
    from joblib import Parallel, delayed
    torch.set_num_threads(_threads(a))
    tags = ["" if t == "base" else t for t in a.tags.split(",")]
    tn = lambda t: t or "base"
    vers = ["deg"] + [tn(t) for t in tags]
    rdir = paths(a.report_tag)["report"]
    ccache = CACHE / "compare"
    ccache.mkdir(parents=True, exist_ok=True)
    b, clean = load_bank(_bank_names(a.banks))
    va = np.where(b["kind"] == "val")[0]
    deg = b["deg"][va]
    rest = {}
    for t in tags:  # restored val pairs, cached per restorer (invalidated when the model file changes)
        f = ccache / f"val_restored_{tn(t)}{'_fp32' if a.force_fp32 else ''}.npz"
        mt = paths(t)["model"].stat().st_mtime
        if f.exists():
            z = np.load(f)
            if (np.array_equal(z["id"], b["id"][va]) and np.array_equal(z["noise_t"], b["noise_t"][va])
                    and int(z["tta"]) == a.tta and float(z["mtime"]) == mt):
                rest[t] = z["rest"]
                print(f"{tn(t)}: cached restorations {f}", flush=True)
                continue
        m = load_model(t)
        t0 = time.time()
        rest[t] = restore(m, torch.from_numpy(deg), b["sigma"][va], tta=a.tta, amp=False if a.force_fp32 else None).numpy()
        print(f"{tn(t)}: restored {len(va)} pairs in {time.time() - t0:.0f}s (bf16 {bool(m.amp) and not a.force_fp32})",
              flush=True)
        np.savez(f, rest=rest[t], id=b["id"][va], noise_t=b["noise_t"][va], tta=a.tta, mtime=mt)
    metas = [{"bank": str(b["bank"][j]), "id": str(b["id"][j]), "noise_t": float(b["noise_t"][j]),
              "sigma": float(b["sigma"][j]), "q": float(b["q"][j]), "sb": float(b["sb"][j]),
              "contrast": float(b["contrast"][j])} for j in va]
    tgt = lambda n: photometric_target(clean[b["id"][va[n]]], b["coef"][va[n]]).astype(np.float32)
    t0 = time.time()
    rows = Parallel(n_jobs=a.jobs)(delayed(_pair_row)(metas[n], tgt(n), [("deg", deg[n].astype(np.float32))] +
                                                      [(tn(t), rest[t][n]) for t in tags]) for n in range(len(va)))
    df = pd.DataFrame(rows)
    print(f"metrics for {len(df)} pairs in {time.time() - t0:.0f}s", flush=True)
    for v in vers:
        df[f"dark_err_{v}"] = (df[f"dark_contrast_{v}"] - 1.0).abs()
    bands = [("low (<=9)", df.noise_t <= 9), ("mid (9-15]", (df.noise_t > 9) & (df.noise_t <= 15)),
             ("high (>15)", df.noise_t > 15), (f"rule: sigma_est >= {a.rule_noise}", df.sigma >= a.rule_noise),
             ("all", df.noise_t > 0)]
    mets = ["psnr", "psnr_bd", "ridge_corr", "dark_contrast", "dark_err"]
    grp = df.id.values
    summary = []
    for bn, m in bands:
        m = m.values
        if not m.any():
            continue
        row = {"band": bn, "n": int(m.sum()), "n_sources": int(df.id[m].nunique())}
        for me in mets:
            for v in vers:
                row[f"{me}_{v}"] = round(float(df[f"{me}_{v}"][m].mean()), 4)
        if "" in tags:
            for t in tags:
                if not t:
                    continue
                for me in mets:
                    d = (df[f"{me}_{tn(t)}"] - df[f"{me}_base"]).values[m]
                    row[f"d_{me}_{tn(t)}"] = round(float(d.mean()), 4)
                    row[f"d_{me}_{tn(t)}_ci90"] = _boot_ci(lambda ii: d[ii].mean(), grp[m])
        summary.append(row)
    print("held-out pairs (targets = clean source in the degraded photometry), means; d_* = restorer - base:")
    hdr = ["band", "n"] + [f"{me} " + " / ".join(vers) for me in ("psnr", "psnr_bd", "ridge_corr", "dark_contrast")]
    print("| " + " | ".join(hdr) + " |")
    for r in summary:
        cells = [r["band"], str(r["n"])] + [" / ".join(f"{r[f'{me}_{v}']:.3f}" if "psnr" not in me else f"{r[f'{me}_{v}']:.2f}"
                                                     for v in vers) for me in ("psnr", "psnr_bd", "ridge_corr", "dark_contrast")]
        print("| " + " | ".join(cells) + " |")
    for r in summary:
        ds = {k: v for k, v in r.items() if k.startswith("d_")}
        if ds:
            print(r["band"], {k: v for k, v in ds.items()})
    out = {"tags": [tn(t) for t in tags], "banks": a.banks, "tta": a.tta, "force_fp32": a.force_fp32,
           "n_pairs": len(df), "by_band": summary}
    # ---------------------------------------------------------------- feature-level check on high-noise copies
    rdir.mkdir(parents=True, exist_ok=True)
    if a.features:
        sel = np.where(df.sigma.values >= a.rule_noise)[0]
        u8 = lambda z: np.clip(np.round(z), 0, 255).astype(np.uint8)
        pos = {f"{df.bank[n]}:{df.id[n]}:{df.noise_t[n]:.6f}": n for n in sel}
        keys = {v: list(pos) for v in vers}
        keys["clean"] = sorted(set(df.id.values[sel]))

        def img_of(v, k):
            if v == "clean":
                return clean[k]
            return deg[pos[k]] if v == "deg" else u8(rest["" if v == "base" else v][pos[k]])

        have, jobs = {}, []
        for v in keys:  # deg / base / clean features never change: cached; specialists are always recomputed
            fc = ccache / f"v3feats_{v}.parquet"
            have[v] = pd.read_parquet(fc).set_index("key") if (v in ("deg", "base", "clean") and fc.exists()) else None
            jobs += [(v, k) for k in keys[v] if have[v] is None or k not in have[v].index]
        t0 = time.time()
        res = Parallel(n_jobs=a.jobs)(delayed(_v3_feats)(k, img_of(v, k)) for v, k in jobs)
        print(f"v3 features for {len(jobs)} images in {time.time() - t0:.0f}s", flush=True)
        F = {}
        for v in keys:
            got = [r for (vv, _), r in zip(jobs, res) if vv == v]
            parts = ([have[v]] if have[v] is not None else []) + ([pd.DataFrame(got).set_index("key")] if got else [])
            F[v] = pd.concat(parts)
            F[v] = F[v][~F[v].index.duplicated(keep="last")]
            if v in ("deg", "base", "clean"):
                F[v].reset_index().to_parquet(ccache / f"v3feats_{v}.parquet", index=False)
            F[v] = F[v].loc[keys[v]]
        sub = df.iloc[sel].reset_index(drop=True)
        r2 = lambda x, c: 1 - np.sum((x - c) ** 2) / max(np.sum((c - c.mean()) ** 2), 1e-12)
        fb = []
        for bn, m in ((f"rule: sigma_est >= {a.rule_noise}", np.ones(len(sub), bool)), ("high (>15)", sub.noise_t.values > 15)):
            g = sub.id.values[m]
            for fe in CHECK_FEATS:
                c = F["clean"].loc[g, fe].values
                xs = {v: F[v][fe].values[m] for v in vers}
                row = {"band": bn, "feature": fe, "n": int(m.sum()), "clean_sd": round(float(c.std()), 4)}
                for v in vers:
                    row[f"r2_{v}"] = round(float(r2(xs[v], c)), 4)
                    row[f"bias_{v}"] = round(float(np.mean(xs[v] - c)), 4)
                if "" in tags:
                    for t in tags:
                        if t:
                            row[f"d_r2_{tn(t)}_ci90"] = _boot_ci(lambda ii: r2(xs[tn(t)][ii], c[ii]) - r2(xs["base"][ii], c[ii]), g)
                fb.append(row)
        print("v3 feature fidelity vs the clean source (R^2 over pairs; bias = mean(restored - clean)):")
        print(pd.DataFrame(fb).to_string(index=False))
        out["features"] = fb
        pd.concat([F[v].assign(version=v) for v in list(vers) + ["clean"]]).reset_index().to_csv(
            rdir / "compare_features.csv", index=False)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "compare_metrics.json").write_text(json.dumps(out, indent=2))
    df.to_csv(rdir / "compare_pairs.csv", index=False)
    # held-out synthetic montage (high band): degraded | each restorer | target, 128 px crops
    hi = np.where(df.noise_t.values > 15)[0]
    if len(hi):
        order = hi[np.argsort(df.noise_t.values[hi])]
        tiles, c = [], 128
        u8 = lambda z: np.clip(np.round(z), 0, 255).astype(np.uint8)
        sep = np.full((c, 3), 255, np.uint8)
        for n in order[np.linspace(0, len(order) - 1, 6).round().astype(int)]:
            cols = [deg[n][:c, :c]] + [u8(rest[t][n][:c, :c]) for t in tags] + [u8(tgt(n)[:c, :c])]
            row = np.concatenate(sum([[x, sep] for x in cols], [])[:-1], 1)
            cv2.putText(row, f"nt{df.noise_t[n]:.0f} sb{df.sb[n]:.1f}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255, 1)
            tiles.append(np.pad(row, ((0, 5), (0, 0)), constant_values=255))
        cv2.imwrite(str(rdir / "montage_compare_synthetic.png"), np.concatenate(tiles, 0))
    print(f"wrote {rdir / 'compare_metrics.json'}, compare_pairs.csv, montage_compare_synthetic.png "
          f"(columns: degraded | {' | '.join(tn(t) for t in tags)} | target)")


def check_tag(a):
    """Tagged set: every image present, CSVs copied, non-specialist images byte-identical to --fill-from; change of
    the specialist images vs the base restorer. Per-image table and a real-image montage (raw | base | specialist) go to
    the git-ignored data/restore_cache/<tag>/ (they show / measure competition images, incl. test)."""
    pth = paths(a.tag)
    out_dir, fill = pth["out"], _root_path(a.fill_from) if a.fill_from else OUT_DIR
    ids = load_train().ID.tolist() + load_test().ID.tolist()
    rel = lambda i: Path("train" if i.startswith("TRAIN") else "test") / f"{i}.png"
    miss = [i for i in ids if not (out_dir / rel(i)).exists()]
    assert not miss, f"{len(miss)} images missing in {out_dir}"
    for f in ("train.csv", "sample_submission.csv", "folds.csv"):
        assert (out_dir / f).read_bytes() == (DATA_DIR / f).read_bytes(), f
    man = pd.concat([pd.read_csv(f) for f in sorted(pth["cache"].glob("applied_part*.csv"))]).set_index("ID")
    assert sorted(man.index) == sorted(ids), "applied_part*.csv do not cover all images"
    spec = set(man.index[man.specialist == 1])
    bad = [i for i in ids if i not in spec and (out_dir / rel(i)).read_bytes() != (fill / rel(i)).read_bytes()]
    assert not bad, f"{len(bad)} non-specialist images differ from {fill}"
    rows = []
    for i in sorted(spec):
        raw, base, sp = (_read8(i).astype(np.float32), _read8(i, fill).astype(np.float32),
                         _read8(i, out_dir).astype(np.float32))
        lp = lambda z: cv2.GaussianBlur(z, (0, 0), 3)
        rows.append({"ID": i, "split": "train" if i.startswith("TRAIN") else "test", "ic_noise": man.ic_noise[i],
                     "mad_spec_base": float(np.abs(sp - base).mean()), "mad_spec_raw": float(np.abs(sp - raw).mean()),
                     "mad_base_raw": float(np.abs(base - raw).mean()), "psnr_spec_base": psnr(sp, base),
                     "lowpass_mad_spec_base": float(np.abs(lp(sp) - lp(base)).mean()),
                     "mean_shift_spec_base": float(sp.mean() - base.mean())})
    df = pd.DataFrame(rows)
    df.to_csv(pth["cache"] / "check_specialist.csv", index=False)
    print(f"{out_dir}: all {len(ids)} images present; {len(spec)} from the specialist "
          f"(train {int((df.split == 'train').sum())}, test {int((df.split == 'test').sum())}); "
          f"the other {len(ids) - len(spec)} are byte-identical to {fill.name}/")
    df["noise_bin"] = pd.cut(df.ic_noise, [0, 15, 18, 99], labels=["<15", "15-18", ">18"])
    cols = ["mad_spec_base", "mad_spec_raw", "mad_base_raw", "psnr_spec_base", "lowpass_mad_spec_base", "mean_shift_spec_base"]
    summ = df.groupby("noise_bin", observed=True)[cols].median().round(3)
    summ["n"] = df.groupby("noise_bin", observed=True).size()
    print("specialist images, medians (all splits):")
    print(summ.to_string())
    tr = df[df.split == "train"].sort_values("ic_noise")
    pick = tr.iloc[np.linspace(0, len(tr) - 1, 8).round().astype(int)].ID.tolist()
    tiles, c = [], 160
    sep = np.full((c, 3), 255, np.uint8)
    for i in pick:
        row = np.concatenate([_read8(i)[:c, :c], sep, _read8(i, fill)[:c, :c], sep, _read8(i, out_dir)[:c, :c]], 1)
        cv2.putText(row, f"{i[-4:]} n{man.ic_noise[i]:.1f}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255, 1)
        tiles.append(np.pad(row, ((0, 5), (0, 0)), constant_values=255))
    cv2.imwrite(str(pth["cache"] / "montage_real_raw_base_spec.png"), np.concatenate(tiles, 0))
    print(f"montage (raw | base | {a.tag}, 8 noisy train images): {pth['cache'] / 'montage_real_raw_base_spec.png'}")


def bench(a):
    torch.set_num_threads(_threads(a))
    m = UNet(ch=tuple(int(round(c * a.width)) for c in BASE_CH)).to(memory_format=torch.channels_last)
    m.amp = not a.fp32
    print(f"params {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    x = to_input(torch.randint(0, 255, (a.bs, a.patch, a.patch), dtype=torch.uint8), [10.0] * a.bs)
    y = torch.randn(a.bs, 1, a.patch, a.patch)
    for n in range(4):
        if n == 1:
            t = time.time()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=m.amp):
            out = m(x)
        loss = F.l1_loss(out.float(), y)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"train step bs{a.bs} patch{a.patch}: {(time.time() - t) / 3:.3f}s")
    t = time.time()
    restore(m, torch.randint(0, 255, (8, 256, 256), dtype=torch.uint8), [10.0] * 8, tta=8)
    print(f"inference 256px x8 TTA: {(time.time() - t) / 8:.3f}s per image")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bank", "train", "validate", "compare", "apply", "check", "bench"])
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-w", type=float, default=0.5)
    ap.add_argument("--identity", type=float, default=0.15)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--fresh", action="store_true", help="ignore an existing training checkpoint")
    ap.add_argument("--resume", action="store_true", help="apply: skip images already written")
    # high-noise specialist options (defaults = the original restorer)
    ap.add_argument("--tag", default="", help="model/ckpt in data/restore_cache/<tag>/, report in "
                    "experiments/restore/<tag>/, images in data_restored_<tag>/ ('' = the original restorer)")
    ap.add_argument("--bank-name", default="", help="bank: write to data/restore_cache/bank_<name>/ ('' = original)")
    ap.add_argument("--q-min", type=float, default=0.0, help="bank: only _degrade_v2 draws with q >= this (rejection)")
    ap.add_argument("--k-train", type=int, default=K_TRAIN, help="bank: degradations per train source")
    ap.add_argument("--k-val", type=int, default=K_VAL, help="bank: degradations per held-out source")
    ap.add_argument("--seed-base", type=int, default=7, help="bank: rng prefix of train pairs, val = +1 (original 7/8)")
    ap.add_argument("--banks", default="main", help="train/validate/compare: comma list of banks ('main' = original)")
    ap.add_argument("--train-q-min", type=float, default=0.0, help="train/validate: only pairs with q >= this")
    ap.add_argument("--init", default="", help="train: initial state dict (a narrower one is zero-expanded)")
    ap.add_argument("--width", type=float, default=1.0, help="train/bench: channel multiplier of the U-Net")
    ap.add_argument("--fp32", action="store_true", help="train/bench without bf16 autocast (faster on CPUs without AMX)")
    ap.add_argument("--pct-start", type=float, default=0.05, help="train: OneCycle warm-up fraction")
    ap.add_argument("--mon-n", type=int, default=0, help="train: monitor on at most this many val pairs (0 = all)")
    ap.add_argument("--no-validate", action="store_true", help="train: skip the final held-out report (use compare)")
    ap.add_argument("--tags", default="base", help="compare: comma list of restorers ('base' = the original)")
    ap.add_argument("--report-tag", default="", help="compare: report goes to experiments/restore/<report-tag>/")
    ap.add_argument("--features", action="store_true", help="compare: v3 feature fidelity on high-noise copies")
    ap.add_argument("--rule-noise", type=float, default=11.92, help="compare: sigma cut of the 'rule' band / features")
    ap.add_argument("--force-fp32", action="store_true", help="compare: run every restorer without bf16 autocast")
    ap.add_argument("--jobs", type=int, default=1, help="compare: processes for metrics / feature extraction")
    ap.add_argument("--apply-noise-min", type=float, default=0.0, help="apply: restore only raw ic_noise >= this")
    ap.add_argument("--fill-from", default="", help="apply/check: directory the other images are copied from")
    a = ap.parse_args()
    torch.set_num_threads(_threads(a))
    {"bank": lambda: build_bank(a.part, a.nparts, a.bank_name, a.q_min, a.k_train, a.k_val, a.seed_base),
     "train": lambda: train(a), "validate": lambda: validate_saved(a), "compare": lambda: compare(a),
     "apply": lambda: apply(a), "check": lambda: check_tag(a) if a.tag else check(a), "bench": lambda: bench(a)}[a.cmd]()
