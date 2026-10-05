import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .common import DATA_DIR, SEED

if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "train.csv")
    bins = pd.qcut(df.hardness, 10, labels=False, duplicates="drop")
    df["fold"] = -1
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(df, bins)):
        df.loc[va, "fold"] = k
    df[["ID", "fold"]].to_csv(DATA_DIR / "folds.csv", index=False)
    print(df.fold.value_counts().sort_index().to_dict())
