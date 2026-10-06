"""A2: per-image maps cache (train + test) -- illumination-corrected denoised image (rn), watershed grain labels.
Same preprocessing as src.features.extract_v3 (NLM, 70th-pct local matrix level, sato ridge watershed).
Test images are processed only for unsupervised statistics (no fitting).
Usage: python a2_maps.py CACHE_DIR [train|test]"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, restoration, segmentation

from eda_common import DATA_DIR


def _bg_matrix(den, pct=70, win=20, sub=4):
    small = cv2.resize(den, (den.shape[1] // sub, den.shape[0] // sub), interpolation=cv2.INTER_AREA)
    b = ndi.percentile_filter(small, pct, size=win, mode="reflect")
    b = cv2.GaussianBlur(b, (0, 0), win / 3)
    return cv2.resize(b, den.shape[::-1], interpolation=cv2.INTER_CUBIC)


def one(i, split):
    raw8 = cv2.imread(str(DATA_DIR / split / f"{i}.png"), cv2.IMREAD_GRAYSCALE)
    raw = raw8.astype(np.float32)
    sig = float(restoration.estimate_sigma(raw))
    den = cv2.fastNlMeansDenoising(raw8, None, h=float(np.clip(sig, 2.0, 30.0)), templateWindowSize=5,
                                   searchWindowSize=21).astype(np.float32)
    bg = _bg_matrix(den)
    rn = den / np.maximum(bg, 1.0)
    rid = filters.sato(den, sigmas=[1.0, 1.5], black_ridges=True)
    rs = cv2.GaussianBlur(rid.astype(np.float32), (0, 0), 1.0)
    mk = measure.label(morphology.h_minima(rs, 0.15 * float(np.percentile(rs, 99))))
    ws = segmentation.watershed(rs, mk, watershed_line=True)
    return rn.astype(np.float16), ws.astype(np.int32), sig


if __name__ == "__main__":
    cache = Path(sys.argv[1])
    split = sys.argv[2]
    cache.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(DATA_DIR / ("train.csv" if split == "train" else "sample_submission.csv")).ID.tolist()
    res = Parallel(n_jobs=1)(delayed(one)(i, split) for i in ids)
    np.save(cache / f"rn_{split}.npy", np.stack([a for a, _, _ in res]))
    np.save(cache / f"ws_{split}.npy", np.stack([b for _, b, _ in res]))
    pd.DataFrame({"ID": ids, "sigma": [c for _, _, c in res]}).to_csv(cache / f"sig_{split}.csv", index=False)
    print("done", split, len(ids))
