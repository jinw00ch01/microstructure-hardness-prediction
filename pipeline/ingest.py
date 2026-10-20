"""Write a laptop experiment sent by message into experiments/<name>/ after checking it.
usage (repo root): PYTHONPATH=. python $S/ingest.py NAME oof.txt test.txt score.txt"""
import io, json, sys
import numpy as np, pandas as pd
from src.common import EXP_DIR

name, f_oof, f_te, f_sc = sys.argv[1:5]
rd = lambda p: pd.read_csv(io.StringIO(open(p).read().strip() + "\n"))
oof, te, sc = rd(f_oof), rd(f_te), json.load(open(f_sc))
tr = pd.read_csv("data/train.csv").merge(pd.read_csv("data/folds.csv"), on="ID")
ss = pd.read_csv("data/sample_submission.csv")
assert list(oof.columns) == ["ID", "hardness"] and list(te.columns) == ["ID", "hardness"], (oof.columns, te.columns)
assert len(oof) == 500 and set(oof.ID) == set(tr.ID) and oof.ID.is_unique, "oof IDs"
assert len(te) == 1000 and set(te.ID) == set(ss.ID) and te.ID.is_unique, "test IDs"
assert np.isfinite(oof.hardness).all() and np.isfinite(te.hardness).all(), "non-finite"
m = tr.merge(oof, on="ID", suffixes=("", "_p"))
r = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
cv = r(m.hardness_p, m.hardness)
fr = [r(m.hardness_p[m.fold == f], m.hardness[m.fold == f]) for f in range(5)]
print(f"{name}: CV {cv:.4f} (score.json {sc.get('cv_rmse')}) folds {np.round(fr, 3).tolist()}")
assert abs(cv - sc["cv_rmse"]) < 1e-3, "CV does not match score.json"
snr = m[["ID"]].merge(pd.read_parquet("data/features_v3.parquet", columns=["ID", "ic_ridge_snr"]), on="ID").ic_ridge_snr.values
t = np.digitize(snr, np.quantile(snr, [1 / 3, 2 / 3]))
print("snr terciles noisy/mid/clean", [round(r(m.hardness_p[t == k], m.hardness[t == k]), 3) for k in range(3)])
print("test pred mean/std", round(te.hardness.mean(), 2), round(te.hardness.std(), 2),
      "| oof mean/std", round(oof.hardness.mean(), 2), round(oof.hardness.std(), 2))
out = EXP_DIR / name
out.mkdir(parents=True, exist_ok=True)
oof.to_csv(out / "oof.csv", index=False)
te.to_csv(out / "test.csv", index=False)
(out / "score.json").write_text(json.dumps(sc, indent=2))
lb = EXP_DIR / "LEADERBOARD.md"
if f"| {name} |" not in lb.read_text():
    with open(lb, "a") as f:
        f.write(f"| {name} | {cv:.3f} |  | {sc.get('notes', '')} (laptop RTX 5060) |\n")
print("written", out)
