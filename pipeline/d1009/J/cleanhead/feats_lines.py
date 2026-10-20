"""Label-free: number of grain pieces crossed along every row and column of the label map (pores excluded from counts as boundaries? no: all labels)."""
import numpy as np, pandas as pd, sys
OUT = '/tmp/claude/cleanhead/'
split = sys.argv[1] if len(sys.argv) > 1 else 'train'
W = np.load(OUT + f'ws_C_{split}.npz'); G = pd.read_parquet(OUT + f'grains_C_{split}.parquet')
R, C = {}, {}
for i in W.files:
    ws = W[i].astype(np.int32)
    R[i] = 1 + (ws[:, 1:] != ws[:, :-1]).sum(1)   # pieces along row r
    C[i] = 1 + (ws[1:, :] != ws[:-1, :]).sum(0)   # pieces along column c
pd.DataFrame(R).T.to_parquet(OUT + f'rowcnt_{split}.parquet'); pd.DataFrame(C).T.to_parquet(OUT + f'colcnt_{split}.parquet'); print('ok')
