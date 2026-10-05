"""Frozen pretrained backbone embeddings with flip/rot TTA. Needs internet for timm weights (run on a
machine that can reach huggingface.co). python -m src.extract_embeddings --backbone convnext_small.fb_in22k_ft_in1k"""
import argparse

import numpy as np
import pandas as pd
import timm
import torch

from .common import DATA_DIR, load_test, read_img


@torch.no_grad()
def embed(backbone, ids, size=256, bs=32, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = timm.create_model(backbone, pretrained=True, num_classes=0).eval().to(device)
    cfg = m.pretrained_cfg
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
    out = []
    for s in range(0, len(ids), bs):
        x = torch.from_numpy(np.stack([read_img(i) for i in ids[s:s + bs]]))[:, None].repeat(1, 3, 1, 1).to(device)
        if size != 256:
            x = torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)
        x = (x - mean) / std
        views = [x, x.flip(-1), x.flip(-2), x.transpose(-1, -2)]
        out.append(torch.stack([m(v) for v in views]).mean(0).float().cpu().numpy())
        print(s, end=" ", flush=True)
    return np.concatenate(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()
    ids = list(pd.read_csv(DATA_DIR / "train.csv").ID) + list(load_test().ID)
    E = embed(a.backbone, ids, a.size)
    (DATA_DIR / "emb").mkdir(exist_ok=True)
    tag = f"{a.backbone.replace('/', '_')}_{a.size}"
    np.save(DATA_DIR / "emb" / f"{tag}.npy", E)
    pd.Series(ids).to_csv(DATA_DIR / "emb" / f"{tag}_ids.csv", index=False, header=["ID"])
    print("\nsaved", E.shape)
