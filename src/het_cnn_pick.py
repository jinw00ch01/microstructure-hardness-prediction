"""Label-free pick among het CNN --screen runs (laptop-gpu.md section 12, pre-registration d1009/H in the hardness-cache).

usage: python -m src.het_cnn_pick --base ev2s_loc_e32r4 ev2s_loc_e32r4_s1 --runs scr12_e64 scr12_r8 scr12_lw2 scr12_lw4

For each run (data/het_cnn/<run>/score.json) it averages, over folds 0 and 1, the held-out render metrics P (partial
corr of het with het4 given log N), W (within-image block corr) and the real held-out clean P (R). The baseline is the
same average over the --base runs. Pick = the run with the highest P; it qualifies only if P >= base + 0.05,
W >= base - 0.03 and R >= base - 0.05. Prints one line per run and a final 'PICK <run>' or 'PICK none'.
Reads only score.json files (no labels, no images)."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "data" / "het_cnn"
FOLDS = ("0", "1")


def metrics(run):
    s = json.loads((ROOT / run / "score.json").read_text())
    per = s["stage1_per_fold"]
    miss = [f for f in FOLDS if f not in per]
    assert not miss, f"{run}: folds {miss} not done"
    P = np.mean([per[f]["stage1_renders"]["pcorr_het_given_logN"] for f in FOLDS])
    W = np.mean([per[f]["stage1_renders"]["corr_block_within"] for f in FOLDS])
    R = np.mean([per[f]["heldout_real_clean"]["pcorr_het_given_logN"] for f in FOLDS])
    return float(P), float(W), float(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    B = np.mean([metrics(r) for r in a.base], 0)
    bars = (B[0] + 0.05, B[1] - 0.03, B[2] - 0.05)
    print(f"base {'+'.join(a.base)}: P {B[0]:.4f} W {B[1]:.4f} R {B[2]:.4f} | bars P >= {bars[0]:.4f}, "
          f"W >= {bars[1]:.4f}, R >= {bars[2]:.4f}")
    rows = []
    for r in a.runs:
        if not (ROOT / r / "score.json").exists():
            print(f"{r}: no score.json (not run)")
            continue
        try:
            P, W, R = metrics(r)
        except AssertionError as e:
            print(f"{e} (incomplete, skipped)")
            continue
        ok = P >= bars[0] and W >= bars[1] and R >= bars[2]
        rows.append((P, r, ok))
        print(f"{r}: P {P:.4f} W {W:.4f} R {R:.4f} {'qualifies' if ok else 'does not qualify'}")
    best = max(rows) if rows else None
    print(f"PICK {best[1] if best and best[2] else 'none'}")


if __name__ == "__main__":
    main()
