"""
route-corrected and collapsed.

`collapse` merges consecutive transfer rows in the SAME care unit before any
left-shift logic runs. 28,083 SW rows are immediately preceded by a row in the
identical care unit across 20,184 admissions — bed/room moves recorded as
separate transfers, not a discharge from the ward and a return to it.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent.parent
MAT, SEQ, SPL = BASE/"data"/"matrices", BASE/"data"/"sequences", BASE/"data"/"splits"
SEED, TEST_SIZE, VAL_SIZE = 42, 0.20, 0.10

src = (BASE/"python"/"SW_pipeline_test.py").read_text(encoding="utf-8").split("if __name__")[0]
mod = type(sys)("swp"); exec(src, mod.__dict__)

raw = mod.load(str(BASE/"data-1785450752282.csv"))


sw = raw[raw.careunit.isin(mod.SW)]
t0 = sw.groupby('hadm_id').outtime.min().rename('t0'); j = sw.join(t0, on='hadm_id')
old = t0.to_frame(); old['old'] = old.index.isin(j[j.intime > j.t0].hadm_id).astype(int)
truth = pd.read_csv(BASE/"data"/"labels"/"sw_left_shift_labels.csv").set_index('hadm_id')
assert (old.join(truth).old == old.join(truth).sw_left_shift).all(), "GATE FAILED"
print(f"gate passed: reproduced all {len(old):,} ward-name labels ({int(old.old.sum()):,} positives)")

L = mod.build(raw, 0.0)
base = pd.read_csv(MAT/"sw_feature_matrix_v2.csv")
lab = L['left_shift'].reindex(base.hadm_id).fillna(0).astype(int)
out = base.copy(); out['sw_left_shift'] = lab.values
p = MAT/"sw_feature_matrix_v2_collapse.csv"; out.to_csv(p, index=False)
print(f"{p.name}: {out.shape}, positives {lab.sum():,} ({100*lab.mean():.3f}%)")


