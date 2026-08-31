import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE       = Path("C:/Users/Lenovo/Desktop/Msc Project/data")
EVENT_LOG  = BASE / "event_log" / "mimic_transfers_v4_clean.csv"
ICU_PP     = BASE / "features"  / "icu_prediction_points.csv"
SW_PP      = BASE / "features"  / "sw_prediction_points.csv"
OUT_DIR    = BASE / "sequences"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = 30   # covers 100th percentile for both cohorts (ICU max=20, SW max=29)
PAD_TOKEN = 0

ACTIVITY_VOCAB = {
    'Hospital Admission':                 1,
    'Admission to Emergency Department':  2,
    'Admission to Medical Ward':          3,
    'Admission to ICU':                   4,
    'Admission to Surgery Ward':          5,
    'Admission to Stepdown':              6,
    'Transfer to Medical Ward':           7,
    'Transfer to ICU':                    8,
    'Transfer to Surgery Ward':           9,
    'Transfer to Stepdown':              10,
    'Transfer to Emergency Department':  11,
}

with open(OUT_DIR / "activity_vocab.json", "w") as f:
    json.dump(ACTIVITY_VOCAB, f, indent=2)

print("Loading event log...")
events = pd.read_csv(EVENT_LOG)
events['Timestamp'] = pd.to_datetime(events['Timestamp'])
# Sanity check the vocab covers every activity present
unknown = set(events['Activity'].unique()) - set(ACTIVITY_VOCAB.keys())
assert not unknown, f"Unknown activities not in vocab: {unknown}"
events['activity_id'] = events['Activity'].map(ACTIVITY_VOCAB).astype(np.int32)
print(f"  {len(events):,} events across {events['Case ID'].nunique():,} cases")


def build_sequences(cohort_name: str, pred_pts_path: Path):
    """
    Build padded integer sequences for one cohort.
    Returns (X, hadm_id_order, dropped_count).
    """
    print(f"\n=== {cohort_name} ===")
    pp = pd.read_csv(pred_pts_path)
    pp['prediction_point'] = pd.to_datetime(pp['prediction_point'])
    pp = pp.rename(columns={'hadm_id': 'Case ID'})
    print(f"  {len(pp):,} admissions in cohort")

    merged = events.merge(pp, on='Case ID', how='inner')
    merged = merged[merged['Timestamp'] < merged['prediction_point']]
    merged = merged.sort_values(['Case ID', 'Timestamp'])

    seqs = merged.groupby('Case ID', sort=False)['activity_id'].apply(list)

    dropped = len(pp) - len(seqs)
    print(f"  {len(seqs):,} admissions with ≥1 event before prediction point")
    print(f"  {dropped} admissions dropped (0 events before pp)")

    lens = seqs.apply(len)
    print(f"  sequence lengths: min={lens.min()}, mean={lens.mean():.2f}, "
          f"max={lens.max()}, 95th pct={np.percentile(lens,95):.0f}")
    truncated = (lens > MAX_LEN).sum()
    print(f"  truncated (len > {MAX_LEN}): {truncated}")
    hadm_id_order = seqs.index.to_numpy(dtype=np.int64)
    X = np.zeros((len(seqs), MAX_LEN), dtype=np.int32)
    for i, s in enumerate(seqs.values):
        s = s[-MAX_LEN:]     
        X[i, MAX_LEN - len(s):] = s           
    return X, hadm_id_order, dropped

X_icu, ids_icu, drop_icu = build_sequences("ICU", ICU_PP)
X_sw,  ids_sw,  drop_sw  = build_sequences("SW",  SW_PP)

np.save(OUT_DIR / "icu_sequences.npy",     X_icu)
np.save(OUT_DIR / "icu_hadm_id_order.npy", ids_icu)
np.save(OUT_DIR / "sw_sequences.npy",      X_sw)
np.save(OUT_DIR / "sw_hadm_id_order.npy",  ids_sw)

print("\n=== SUMMARY ===")
print(f"  ICU: X shape {X_icu.shape}, dropped {drop_icu} admissions")
print(f"  SW:  X shape {X_sw.shape},  dropped {drop_sw} admissions")
print(f"\nSaved to {OUT_DIR}/")
print("  icu_sequences.npy, icu_hadm_id_order.npy")
print("  sw_sequences.npy,  sw_hadm_id_order.npy")
print("  activity_vocab.json")

print("\n=== First 3 ICU sequences (hadm_id → padded sequence) ===")
for i in range(3):
    nonzero = X_icu[i][X_icu[i] != 0]
    print(f"  {ids_icu[i]}: {X_icu[i].tolist()} (actual len={len(nonzero)})")