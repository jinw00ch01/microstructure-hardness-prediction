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


def build_bank(part, nparts):
    from .features import _degrade_v2
    cv2.setNumThreads(1)
    trn, val = sources()
    jobs = [(i, k, "train") for i in trn for k in range(K_TRAIN)] + [(i, k, "val") for i in val for k in range(K_VAL)]
    jobs = sorted(jobs[part::nparts])  # group by source so per-source layers are computed once
    layers, out = {}, {"id": [], "kind": [], "deg": [], "coef": [], "sigma": [], "noise_t": [], "q": [], "sb": [],
                       "contrast": [], "fit_rms": [], "gain0": []}
    t = time.time()
    for n, (i, k, kind) in enumerate(jobs):
        clean = _read8(i)
        if i not in layers:
            layers = {i: _src_layers(clean)}  # sources are contiguous within a part, keep one
        rng = np.random.default_rng([7 if kind == "train" else 8, int(i.split("_")[-1]), k])
        deg, p = _degrade_v2(clean, rng)
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
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / f"bank_part{part}.npz", **{k: np.array(v) for k, v in out.items()})
    print(f"part {part}: {len(jobs)} pairs in {time.time() - t:.0f}s; median fit rms {np.median(out['fit_rms']):.2f}",
          flush=True)


def load_bank():
    parts = sorted(CACHE.glob("bank_part*.npz"))
    zs = [np.load(p) for p in parts]
    b = {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}
    ids = sorted(set(b["id"].tolist()))
    clean = {i: _read8(i) for i in ids}
    return b, clean


# ----------------------------------------------------------------------------------------------- model
def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, ch=(32, 64, 96, 128), cin=2):
        super().__init__()
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
def restore(model, imgs8, sigmas, tta=8, bs=8):
    """imgs8: uint8 tensor (N, H, W) -> float restored (N, H, W) in grey levels (mean over D4 views)."""
    model.eval()
    out = []
    for s in range(0, len(imgs8), bs):
        x = to_input(imgs8[s:s + bs], sigmas[s:s + bs])
        acc = 0
        for k in range(tta):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                y = model(d4(x, k).contiguous(memory_format=torch.channels_last))
            acc = acc + d4_inv(y.float(), k)
        out.append(from_output(acc / tta))
    return torch.cat(out)


def psnr(a, b):
    return float(10 * np.log10(255.0 ** 2 / max(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2), 1e-9)))


# ----------------------------------------------------------------------------------------------- training
def train(a):
    torch.set_num_threads(min(a.threads, 2))
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    b, clean = load_bank()
    tr = np.where(b["kind"] == "train")[0]
    va = np.where(b["kind"] == "val")[0]
    print(f"bank: {len(tr)} train pairs from {len(set(b['id'][tr]))} sources, {len(va)} val pairs", flush=True)
    clean_sig = {i: _sigma(c) for i, c in clean.items()}
    deg = torch.from_numpy(b["deg"])
    tgt_cache = {}

    def target(j):  # cached as float16 (0.06 grey-level resolution at 128-255) to halve memory
        if j not in tgt_cache:
            tgt_cache[j] = torch.from_numpy(photometric_target(clean[b["id"][j]], b["coef"][j])).half()
        return tgt_cache[j].float()

    model = UNet().to(memory_format=torch.channels_last)
    print(f"UNet params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    P = a.patch
    t0 = time.time()
    run = 0.0
    # checkpoint / resume (a container suspend kills background jobs): weights + optimizer + scheduler + RNG + step
    ckpt, start = CACHE / "restore_ckpt.pt", 1
    cfg = {k: getattr(a, k) for k in ("steps", "bs", "patch", "lr", "grad_w", "identity")}
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
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"cfg": cfg, "step": step, "model": model.state_dict(), "opt": opt.state_dict(),
                    "sch": sch.state_dict(), "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state()}, tmp)
        os.replace(tmp, ckpt)

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
        with torch.autocast("cpu", dtype=torch.bfloat16):
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
            rest = restore(model, deg[va], b["sigma"][va], tta=1).numpy()
            pa = np.mean([psnr(rest[n], target(j)) for n, j in enumerate(va)])
            pb = np.mean([psnr(deg[j].numpy(), target(j)) for j in va])
            print(f"  val PSNR (tta1) degraded {pb:.2f} -> restored {pa:.2f} dB", flush=True)
        if step % a.ckpt_every == 0 or step == a.steps:
            save_ckpt(step)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / "restore_unet.tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, CACHE / "restore_unet.pt")
    print(f"saved {CACHE / 'restore_unet.pt'} after {time.time() - t0:.0f}s", flush=True)
    validate(model, b, va, target)


def validate(model, b, va, target):
    """PSNR before/after by noise band, boundary PSNR / ridge correlation and dark-phase contrast on held-out sources."""
    from skimage import filters
    deg = torch.from_numpy(b["deg"][va])
    rest = restore(model, deg, b["sigma"][va], tta=8).numpy()
    rows = []
    for n, j in enumerate(va):
        t = target(j).numpy()
        clean8 = np.clip(np.round(t), 0, 255).astype(np.uint8)
        rid = filters.sato(t, sigmas=[1.0, 1.5], black_ridges=True)
        bmask = rid > np.percentile(rid, 85)
        den = cv2.GaussianBlur(t, (0, 0), 2.0)
        dark = den < np.percentile(den, 20)
        mat = den > np.percentile(den, 50)
        row = {"noise_t": float(b["noise_t"][j]), "sb": float(b["sb"][j]), "contrast": float(b["contrast"][j])}
        for tag, img in (("deg", deg[n].numpy().astype(np.float32)), ("rest", rest[n])):
            row[f"psnr_{tag}"] = psnr(img, t)
            row[f"psnr_bd_{tag}"] = psnr(img[bmask], t[bmask])
            r2 = filters.sato(img, sigmas=[1.0, 1.5], black_ridges=True)
            row[f"ridge_corr_{tag}"] = float(np.corrcoef(r2.ravel(), rid.ravel())[0, 1])
            sm = cv2.GaussianBlur(img, (0, 0), 2.0)
            row[f"dark_contrast_{tag}"] = float((sm[mat].mean() - sm[dark].mean()) / max(den[mat].mean() - den[dark].mean(), 1e-3))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["band"] = pd.cut(df.noise_t, [0, 9, 15, 99], labels=["low noise (<9)", "mid (9-15)", "high (>15)"])
    cols = ["psnr_deg", "psnr_rest", "psnr_bd_deg", "psnr_bd_rest", "ridge_corr_deg", "ridge_corr_rest",
            "dark_contrast_deg", "dark_contrast_rest"]
    summ = df.groupby("band", observed=True)[cols].mean().round(3)
    summ["n"] = df.groupby("band", observed=True).size()
    allm = df[cols].mean().round(3)
    print("held-out validation (11 clean sources x 6 degradations; targets = clean in degraded photometry):")
    print(summ.to_string())
    print("all:", allm.to_dict())
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "val_metrics.json").write_text(json.dumps(
        {"by_band": summ.reset_index().astype({"band": str}).to_dict(orient="records"), "all": allm.to_dict()}, indent=2))
    df.to_csv(REPORT_DIR / "val_pairs.csv", index=False)
    # held-out synthetic montage: degraded | restored | target (128 px crops), 6 pairs spanning the noise range
    order = np.argsort(b["noise_t"][va])
    tiles, c = [], 128
    u8 = lambda z: np.clip(np.round(z), 0, 255).astype(np.uint8)
    for n in order[np.linspace(0, len(order) - 1, 6).round().astype(int)]:
        j, sep = va[n], np.full((c, 3), 255, np.uint8)
        row = np.concatenate([deg[n].numpy()[:c, :c], sep, u8(rest[n][:c, :c]), sep, u8(target(j).numpy()[:c, :c])], 1)
        cv2.putText(row, f"nt{b['noise_t'][j]:.0f} sb{b['sb'][j]:.1f}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255, 1)
        tiles.append(np.pad(row, ((0, 5), (0, 5)), constant_values=255))
    cv2.imwrite(str(REPORT_DIR / "montage_val_synthetic.png"),
                np.concatenate([np.concatenate(tiles[k:k + 2], 1) for k in range(0, len(tiles), 2)], 0))


# ----------------------------------------------------------------------------------------------- apply / checks
def load_model():
    m = UNet().to(memory_format=torch.channels_last)
    m.load_state_dict(torch.load(CACHE / "restore_unet.pt", map_location="cpu"))
    return m.eval()


def validate_saved(a):
    torch.set_num_threads(min(a.threads, 2))
    b, clean = load_bank()
    va = np.where(b["kind"] == "val")[0]
    validate(load_model(), b, va, lambda j: torch.from_numpy(photometric_target(clean[b["id"][j]], b["coef"][j])))


def apply(a):
    assert DATA_DIR.resolve() != OUT_DIR.resolve(), "run apply with the original DATA_DIR (unset), not data_restored"
    torch.set_num_threads(min(a.threads, 2))
    cv2.setNumThreads(1)
    ids = load_train().ID.tolist() + load_test().ID.tolist()
    ids = ids[a.part::a.nparts]
    model = load_model()
    for split in ("train", "test"):
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)
    tmpdir = OUT_DIR.parent / f".data_restored_tmp{a.part}"  # outside data_restored/ so no stray files appear there
    tmpdir.mkdir(exist_ok=True)
    out = lambda i: OUT_DIR / ("train" if i.startswith("TRAIN") else "test") / f"{i}.png"
    if a.resume:
        done = [i for i in ids if out(i).exists()]
        ids = [i for i in ids if not out(i).exists()]
        print(f"part {a.part}: resume, {len(done)} already written, {len(ids)} to go", flush=True)
    t = time.time()
    for s in range(0, len(ids), 16):
        chunk = ids[s:s + 16]
        imgs = [_read8(i) for i in chunk]
        rest = restore(model, torch.from_numpy(np.stack(imgs)), [_sigma(x) for x in imgs], tta=a.tta).numpy()
        for i, r in zip(chunk, rest):
            cv2.imwrite(str(tmpdir / f"{i}.png"), np.clip(np.round(r), 0, 255).astype(np.uint8))
            os.replace(tmpdir / f"{i}.png", out(i))
        if s % 160 == 0:
            print(f"part {a.part}: {s + len(chunk)}/{len(ids)} {time.time() - t:.0f}s", flush=True)
    for f in ("train.csv", "sample_submission.csv", "folds.csv"):
        shutil.copy(DATA_DIR / f, OUT_DIR / f)
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"part {a.part}: wrote {len(ids)} images in {time.time() - t:.0f}s", flush=True)


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


def bench(a):
    torch.set_num_threads(min(a.threads, 2))
    m = UNet().to(memory_format=torch.channels_last)
    print(f"params {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    x = to_input(torch.randint(0, 255, (a.bs, a.patch, a.patch), dtype=torch.uint8), [10.0] * a.bs)
    y = torch.randn(a.bs, 1, a.patch, a.patch)
    for n in range(4):
        if n == 1:
            t = time.time()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = m(x)
        loss = F.l1_loss(out.float(), y)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"train step bs{a.bs} patch{a.patch}: {(time.time() - t) / 3:.3f}s")
    t = time.time()
    restore(m, torch.randint(0, 255, (8, 256, 256), dtype=torch.uint8), [10.0] * 8, tta=8)
    print(f"inference 256px x8 TTA: {(time.time() - t) / 8:.3f}s per image")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bank", "train", "validate", "apply", "check", "bench"])
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
    a = ap.parse_args()
    torch.set_num_threads(min(a.threads, 2))
    {"bank": lambda: build_bank(a.part, a.nparts), "train": lambda: train(a), "validate": lambda: validate_saved(a),
     "apply": lambda: apply(a),
     "check": lambda: check(a), "bench": lambda: bench(a)}[a.cmd]()
