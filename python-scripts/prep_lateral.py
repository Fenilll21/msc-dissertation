"""
'lat' label variant: lateral transfers relabelled negative.

A positive in the published labels only survives if the patient was in a unit
BELOW the target tier at some point between leaving it and returning. Movements
within the tier (CVICU -> TSICU, Surgery -> Surgery/Trauma) are not returns.

Applied symmetrically to BOTH cohorts so ICU-vs-SW comparisons stay like-for-like.
Cases are RELABELLED 0, not dropped, so cohort sizes are unchanged.

"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent.parent
MAT, SEQ, SPL = BASE/"data"/"matrices", BASE/"data"/"sequences", BASE/"data"/"splits"
SEED, TEST_SIZE, VAL_SIZE = 42, 0.20, 0.10

ICU = {'Medical Intensive Care Unit (MICU)','Cardiac Vascular Intensive Care Unit (CVICU)',
       'Medical/Surgical Intensive Care Unit (MICU/SICU)','Surgical Intensive Care Unit (SICU)',
       'Trauma SICU (TSICU)','Coronary Care Unit (CCU)',
       'Neuro Surgical Intensive Care Unit (Neuro SICU)','Intensive Care Unit (ICU)'}
SW  = {'Surgery','Surgery/Trauma','Cardiac Surgery','Thoracic Surgery',
       'Surgery/Pancreatic/Biliary/Bariatric','Med/Surg/Trauma'}

d = pd.read_csv(BASE/"data"/"transfers_all.csv", parse_dates=['intime','outtime'])
d = d[d.eventtype != 'discharge'].sort_values(['hadm_id','intime'])

def corrected(units, cohort, stored_file, stored_col):
    u  = d[d.careunit.isin(units)]
    t0 = u.groupby('hadm_id').outtime.min().rename('t0')
    j  = u.join(t0, on='hadm_id')
    ret = j[j.intime > j.t0].groupby('hadm_id').intime.min().rename('t_ret')
    L = t0.to_frame().join(ret)
    pos = set(L[L.t_ret.notna()].index)
    g = d.join(L, on='hadm_id'); g = g[(g.intime >= g.t0) & (g.intime < g.t_ret)]
    lower = set(g[~g.careunit.isin(units)].hadm_id)   # was OUTSIDE the tier in between
    keep  = pos & lower
    st = pd.read_csv(stored_file).set_index('hadm_id')[stored_col]
    recomputed = pd.Series(0, index=L.index); recomputed.loc[sorted(pos)] = 1
    common = st.index.intersection(L.index)
    assert (st.loc[common] == recomputed.loc[common]).all(), f"GATE FAILED for {cohort}"
    print(f"gate passed [{cohort}]: recomputation matches the stored label on {len(common):,} admissions")
    print(f"  published {len(pos):,} ({100*len(pos)/len(L):.2f}%) -> corrected {len(keep):,} "
          f"({100*len(keep)/len(L):.2f}%)   lateral removed {len(pos)-len(keep):,}")
    return keep

for cohort, units, mfile, lfile in [
        ("icu", ICU, "icu_feature_matrix_v2.csv", "icu_left_shift_labels.csv"),
        ("sw",  SW,  "sw_feature_matrix_v2.csv",  "sw_left_shift_labels.csv")]:
    lab = f"{cohort}_left_shift"
    keep = corrected(units, cohort.upper(), BASE/"data"/"labels"/lfile, lab)
    m = pd.read_csv(MAT/mfile)
    m[lab] = m.hadm_id.isin(keep).astype(int)
    p = MAT/f"{cohort}_feature_matrix_v2_lat.csv"; m.to_csv(p, index=False)
    print(f"  wrote {p.name}: {m.shape}, positive rate {m[lab].mean():.4f}\n")