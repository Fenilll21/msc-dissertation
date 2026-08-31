import json, itertools
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent.parent
MAT, SEQ, SPL = BASE/"data"/"matrices", BASE/"data"/"sequences", BASE/"data"/"splits"
SEED, TEST_SIZE, VAL_SIZE = 42, 0.20, 0.10

rel = pd.read_csv(BASE/"sw_matrix_relabelled.csv", low_memory=False)
base_cols = pd.read_csv(MAT/"sw_feature_matrix_v2.csv", nrows=0).columns.tolist()
for suffix, src in [("corr", "sw_left_shift_corrected"),
                    ("corrpacu", "sw_left_shift_corrected_pacu_pos")]:
    out = rel[base_cols].copy()
    out["sw_left_shift"] = rel[src].values
    p = MAT/f"sw_feature_matrix_v2_{suffix}.csv"
    out.to_csv(p, index=False)
    print(f"{p.name}: {out.shape}, positive rate {out.sw_left_shift.mean():.4f}")