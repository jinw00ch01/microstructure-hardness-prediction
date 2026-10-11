"""A4 (het CNN with ensemble-measured block targets): the label-free block-target file for src.het_cnn --target-file.

What it measures, per clean train image of the het_cnn prep (data/het_cnn/prep.npz 'ids', in prep order): the 16 grid
block values of src.het_cnn.block_targets (the prep's target definition, unchanged):
  b_k = -2 log(mean over the block's non-pore pixels of a^-1/2), a = pixel area of the watershed grain containing the
  pixel, NaN when the block has < 50 non-pore pixels; pore grains = the src.het_blocks rule on the same map
  (grey / shading surface < 0.68 and area < 600),
on 6 segmentations of the image instead of 1: src.het_blocks.segment with Gaussian sigma in (0.7, 0.8, 0.9) x Otsu factor
tf in (0.5, 0.6) (the prep uses sigma 1.0, tf 0.6). These are the 6 maps of d1011/bold/L1_formula s26/s28 'V4'.
Output b = per-block mean over the 6 maps (blocks NaN in a map are left out of that block's mean; none on the cloud
prep). Same functional form as the prep targets; only the source maps change. A 7th map, (1.0, 0.6) = the prep's own
segmentation, must reproduce the prep's b (b_ref, max |diff| < 1e-9; src.het_cnn --target-file checks it again), which
ties the file to the prep it is used with.

Reads only the train PNGs of the prep ids and the prep 'ids' / 'b' arrays (plus the ID column of train.csv and
features_v3 ic_noise for the clean-id cross-check when present). The hardness label is never read; test images are
never read. Nothing is fitted.

  python -W ignore tools/a4/v4_block_targets.py [--jobs 4] [--out data/het_cnn/a4_v4_block_targets.npz]
  (--repo PATH: repo root for 'src', default two levels above this file; DATA_DIR as in src.common)
Prints the checksums of b (sum, sum of squares, md5 of the float64 bytes) to compare with the cloud build."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

MAPS = [(0.7, 0.5), (0.7, 0.6), (0.8, 0.5), (0.8, 0.6), (0.9, 0.5), (0.9, 0.6)]  # L1 s26 order (sigma, tf)
REF_MAP = (1.0, 0.6)  # src.het_blocks.segment / the prep


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out", default=None, help="default: DATA_DIR/het_cnn/a4_v4_block_targets.npz")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="first N images only (timing test; the file is not valid)")
    return ap.parse_args(argv)


A = parse() if __name__ == "__main__" else None
if A is not None:
    sys.path.insert(0, A.repo)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage import filters, morphology, segmentation  # noqa: E402


def segment(im, sig, tf):
    """src.het_blocks.segment with the Otsu factor tf as a parameter (0.6 there); identical operations."""
    sm = ndi.gaussian_filter(im, sig)
    r = filters.sato(sm, sigmas=(sig, 1.5 * sig), black_ridges=True, mode="reflect")
    interior = ndi.binary_opening(~(r > filters.threshold_otsu(r) * tf), structure=morphology.disk(1))
    lab, _ = ndi.label(interior)
    small = np.bincount(lab.ravel()) < 6
    small[0] = False
    lab[small[lab]] = 0
    lab, _, _ = segmentation.relabel_sequential(lab)
    return sm, segmentation.watershed(r, markers=lab, connectivity=1)


def one(repo, path):
    """(7, 16) block values of one image: the 6 MAPS, then REF_MAP; pores and b exactly as src.het_cnn.prep_one."""
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from src import het_blocks as hb
    from src.het_cnn import block_targets
    im = np.array(Image.open(path)).astype(np.float32)  # exactly as src.het_blocks.one / src.het_cnn.prep_one
    out = np.full((len(MAPS) + 1, 16), np.nan)
    for m, (s, tf) in enumerate(MAPS + [REF_MAP]):
        sm, ws = segment(im, s, tf)
        g = hb.grain_table(sm, ws)
        pore_g = ((hb.shading_norm(g) < 0.68) & (g.area < 600)).values  # pore rule of het_blocks.one
        is_pore = np.zeros(int(ws.max()) + 1, bool)
        is_pore[g.label.values[pore_g]] = True
        out[m] = block_targets(ws, is_pore)
    return out


def checksums(b):
    return dict(sum=float(np.nansum(b)), sumsq=float(np.nansum(b * b)),
                md5=hashlib.md5(np.ascontiguousarray(b, dtype="<f8").tobytes()).hexdigest(), nan=int(np.isnan(b).sum()))


def main(a):
    from joblib import Parallel, delayed
    from src.common import DATA_DIR
    from src import het_blocks as hb
    from src.het_cnn import PREP_FP, NOISE_MAX
    t0 = time.time()
    with np.load(PREP_FP) as z:
        ids = [str(i) for i in z["ids"]]
        b_prep = z["b"].astype(np.float64)
    # clean-id cross-check (label-free: ID column and ic_noise only)
    v3_fp = DATA_DIR / "features_v3.parquet"
    if v3_fp.exists():
        v3 = pd.read_parquet(v3_fp, columns=["ID", "ic_noise"]).set_index("ID")
        tr = pd.read_csv(DATA_DIR / "train.csv", usecols=["ID"]).ID
        clean = [i for i in tr if v3.loc[i, "ic_noise"] < NOISE_MAX]
        assert clean == ids, "prep ids != the clean train ids (features_v3 ic_noise < 9.5, train.csv order)"
        print(f"prep ids == clean train ids ({len(ids)}, train.csv order)", flush=True)
    n = len(ids) if not a.limit else a.limit
    # the tf-parameterised segment == src.het_blocks.segment at (1.0, 0.6), bit for bit (first 2 images)
    for i in ids[:2]:
        im = np.array(Image.open(DATA_DIR / "train" / f"{i}.png")).astype(np.float32)
        s0, w0 = hb.segment(im)
        s1, w1 = segment(im, *REF_MAP)
        assert np.array_equal(w0, w1) and np.array_equal(s0, s1), "segment(1.0, 0.6) != src.het_blocks.segment"
    res = Parallel(n_jobs=a.jobs)(delayed(one)(a.repo, str(DATA_DIR / "train" / f"{i}.png")) for i in ids[:n])
    R = np.stack(res)  # (n, 7, 16)
    b_maps, b_ref = R[:, :len(MAPS)], R[:, len(MAPS)]
    d_ref = float(np.nanmax(np.abs(b_ref - b_prep[:n])))
    nan_pat = bool(np.array_equal(np.isnan(b_ref), np.isnan(b_prep[:n])))
    print(f"reference map {REF_MAP} vs prep b: max |diff| {d_ref:.2e}, NaN pattern equal {nan_pat}", flush=True)
    if not (d_ref < 1e-9 and nan_pat):
        raise SystemExit("the (1.0, 0.6) map does not reproduce the prep's b: this prep was built with another "
                         "segmentation (library versions?); not writing the file")
    nvalid = (~np.isnan(b_maps)).sum(1)  # (n, 16) maps with >= 50 non-pore px in that block
    with np.errstate(invalid="ignore"):
        b = np.nanmean(b_maps, 1) if (nvalid > 0).all() else np.where(nvalid > 0, np.nanmean(b_maps, 1), np.nan)
    het4 = np.array([np.std(r[~np.isnan(r)]) for r in b])
    cs = checksums(b)
    print(f"b (per-block mean over the {len(MAPS)} maps): {b.shape}, blocks with < 6 valid maps "
          f"{int((nvalid < len(MAPS)).sum())}, with 0 {int((nvalid == 0).sum())}; checksums sum {cs['sum']:.6f} "
          f"sumsq {cs['sumsq']:.6f} md5 {cs['md5']} nan {cs['nan']}", flush=True)
    for m, (s, tf) in enumerate(MAPS):
        print(f"  map sigma {s} tf {tf}: mean b {np.nanmean(b_maps[:, m]):.6f}, NaN blocks "
              f"{int(np.isnan(b_maps[:, m]).sum())}, mean het {np.mean([np.nanstd(r) for r in b_maps[:, m]]):.6f}")
    if a.limit:
        print(f"--limit {a.limit}: timing test only, nothing written ({time.time() - t0:.0f}s)")
        return
    out = Path(a.out) if a.out else DATA_DIR / "het_cnn" / "a4_v4_block_targets.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    import numpy, scipy, skimage  # noqa: E401
    prov = dict(what="src.het_cnn block targets (het_blocks form) averaged per block over 6 segmentations",
                maps=MAPS, ref_map=REF_MAP, n=len(ids), b_checksums=cs, ref_max_abs_diff=d_ref,
                script_md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
                versions=dict(numpy=numpy.__version__, scipy=scipy.__version__, skimage=skimage.__version__),
                label_free="reads train PNGs of the prep ids and prep ids/b only; hardness never read; no test image")
    tmp = out.with_name(out.stem + "_tmp.npz")
    np.savez(tmp, ids=np.array(ids), b=b, het4=het4, b_maps=b_maps, b_ref=b_ref, nvalid=nvalid,
             maps=np.array(MAPS), provenance=np.array(json.dumps(prov)))
    os.replace(tmp, out)
    print(f"wrote {out} md5 {hashlib.md5(out.read_bytes()).hexdigest()} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(A)
