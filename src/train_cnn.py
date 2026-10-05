"""End-to-end CNN regression, 5-fold, D4 augmentation + TTA.
python -m src.train_cnn --backbone efficientnet_b0.ra_in1k --epochs 40 --name cnn_effb0"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .common import SEED, create_timm, load_test, load_train, read_img, save_experiment


def d4(x, k):  # x: (..., H, W); 8 dihedral transforms
    x = torch.rot90(x, k % 4, dims=(-2, -1))
    return x.flip(-1) if k >= 4 else x


class DS(Dataset):
    def __init__(self, ids, y=None, train=False, crop=224):
        self.imgs = [torch.from_numpy(read_img(i)) for i in ids]
        self.y, self.train, self.crop = y, train, crop

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x = self.imgs[i][None]
        if self.train:
            x = d4(x, np.random.randint(8))
            if self.crop < 256:
                r, c = np.random.randint(0, 256 - self.crop + 1, 2)
                x = x[:, r:r + self.crop, c:c + self.crop]
            x = x * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05)
        x = (x - 0.5) / 0.25
        return (x, np.float32(self.y[i])) if self.y is not None else x


@torch.no_grad()
def predict(m, dl, dev, tta=8):
    m.eval()
    out = []
    for x in dl:
        x = x.to(dev)
        out.append(torch.stack([m(d4(x, k)).squeeze(-1) for k in range(tta)]).mean(0).float().cpu())
    return torch.cat(out).numpy()


def main(a):
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, te = load_train(), load_test()
    y = tr.hardness.values.astype(np.float32)
    oof, pred = np.zeros(len(tr)), np.zeros(len(te))
    dl_te = DataLoader(DS(te.ID), batch_size=32, num_workers=a.workers)
    for f in range(5):
        if a.folds and f not in a.folds:
            continue
        trn, val = (tr.fold != f).values, (tr.fold == f).values
        mu, sd = y[trn].mean(), y[trn].std()
        m = create_timm(a.backbone, pretrained=not a.scratch, num_classes=1, in_chans=1,
                              drop_path_rate=a.drop_path).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.05)
        dl = DataLoader(DS(tr.ID[trn].tolist(), (y[trn] - mu) / sd, train=True, crop=a.crop),
                        batch_size=a.bs, shuffle=True, drop_last=True, num_workers=a.workers)
        dl_va = DataLoader(DS(tr.ID[val].tolist()), batch_size=32, num_workers=a.workers)
        steps = a.epochs * len(dl)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
        scaler = torch.amp.GradScaler(enabled=dev == "cuda")
        best, best_state = 1e9, None
        for ep in range(a.epochs):
            m.train()
            for x, t in dl:
                x, t = x.to(dev), t.to(dev)
                with torch.autocast(dev, enabled=dev == "cuda"):
                    loss = nn.functional.smooth_l1_loss(m(x).squeeze(-1).float(), t, beta=0.5)
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
            if ep >= a.epochs // 2 or ep == a.epochs - 1:
                p = predict(m, dl_va, dev, tta=2) * sd + mu
                r = float(np.sqrt(np.mean((p - y[val]) ** 2)))
                print(f"fold {f} ep {ep} val {r:.3f}", flush=True)
                if r < best:
                    best, best_state = r, {k: v.detach().clone() for k, v in m.state_dict().items()}
        m.load_state_dict(best_state)  # NOTE: checkpoint picked on val fold -> CV slightly optimistic
        oof[val] = predict(m, dl_va, dev) * sd + mu
        pred += predict(m, dl_te, dev) * sd + mu
        print(f"fold {f} best {best:.3f}")
    pred /= len(a.folds) if a.folds else 5
    save_experiment(a.name, tr, oof, te, pred, notes=f"{a.backbone} ep{a.epochs} crop{a.crop} lr{a.lr}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="efficientnet_b0.ra_in1k")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--name", default="cnn")
    main(ap.parse_args())
