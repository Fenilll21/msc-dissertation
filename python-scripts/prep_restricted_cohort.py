"""
Restricted cohort = admissions whose first contiguous block in the target
care-unit type ENDS IN A TRANSFER OUT of that type. Excludes admissions where
the patient died in the unit or was discharged directly from it, which are
structurally incapable of a positive outcome.

  ICU  85,248 -> 73,405   (11,843 structurally negative, 13.9%)
  SW   73,567 -> 14,012   (59,555 structurally negative, 81.0%)

"""
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

BASE = Path(__file__).resolve().parent.parent
MAT, SEQ, LAB, FEAT = (BASE/"data"/"matrices", BASE/"data"/"sequences",
                       BASE/"data"/"labels", BASE/"data"/"features")
OUT = BASE/"data"/"splits"/"restricted"
SEED, N_TEST_FOLDS, N_VAL_FOLDS = 42, 5, 9

ICU = {'Medical Intensive Care Unit (MICU)','Cardiac Vascular Intensive Care Unit (CVICU)',
       'Medical/Surgical Intensive Care Unit (MICU/SICU)','Surgical Intensive Care Unit (SICU)',
       'Trauma SICU (TSICU)','Coronary Care Unit (CCU)',
       'Neuro Surgical Intensive Care Unit (Neuro SICU)','Intensive Care Unit (ICU)'}
SW  = {'Surgery','Surgery/Trauma','Cardiac Surgery','Thoracic Surgery',
       'Surgery/Pancreatic/Biliary/Bariatric','Med/Surg/Trauma'}
UNITS = {"icu": ICU, "sw": SW}
SRC   = {"icu": "icu_feature_matrix_v2_lat.csv", "sw": "sw_feature_matrix_v2_lat.csv"}


def digest(tr, va, te):
    h = hashlib.sha256()
    for part in (sorted(map(int, tr)), sorted(map(int, va)), sorted(map(int, te))):
        h.update(b"|"); h.update(",".join(map(str, part)).encode())
    return h.hexdigest()[:16]


d = pd.read_csv(BASE/"data"/"transfers_all.csv", parse_dates=["intime", "outtime"])
d = d[d.eventtype != "discharge"].sort_values(["hadm_id", "intime"])
OUT.mkdir(parents=True, exist_ok=True); FEAT.mkdir(parents=True, exist_ok=True)

for cohort in ("icu", "sw"):
    units, lab = UNITS[cohort], f"{cohort}_left_shift"
    x = d.copy(); x["in"] = x.careunit.isin(units)
    blk = (x.hadm_id != x.hadm_id.shift()) | (x["in"] != x["in"].shift())
    x["_b"] = blk.cumsum()
    b = (x.groupby("_b").agg(hadm_id=("hadm_id","first"), intier=("in","first"),
                             intime=("intime","min"), outtime=("outtime","max"))
           .reset_index(drop=True))
    full = set(b[b.intier].hadm_id)
    first_end = b[b.intier].groupby("hadm_id").outtime.min().rename("t0")
    j = b.join(first_end, on="hadm_id")
    onward = set(j[(~j.intier) & (j.intime >= j.t0)].hadm_id) 
    struct_neg = full - onward

    pd.DataFrame({"hadm_id": sorted(struct_neg)}).to_csv(
        FEAT/f"{cohort}_structurally_negative_hadm_ids.csv", index=False)

    m = pd.read_csv(MAT/SRC[cohort])
    r = m[m.hadm_id.isin(onward)].copy()
    p = MAT/f"{cohort}_feature_matrix_restricted.csv"
    r.to_csv(p, index=False)

    order = np.load(SEQ/f"{cohort}_hadm_id_order.npy", allow_pickle=True)
    common = np.array(sorted(set(r.hadm_id) & set(order.tolist())))
    sub = r.set_index("hadm_id").reindex(common).reset_index()
    sid = pd.read_csv(LAB/f"{cohort}_left_shift_labels.csv", usecols=["hadm_id","subject_id"])
    sub = sub.merge(sid, on="hadm_id", how="left", suffixes=("", "_y"))
    assert sub.subject_id.notna().all(), f"{cohort}: hadm_id without subject_id"

    y, g, hid, n = sub[lab].values.astype(int), sub.subject_id.values, sub.hadm_id.values, len(sub)
    tr_i, te_i = next(StratifiedGroupKFold(N_TEST_FOLDS, shuffle=True, random_state=SEED)
                      .split(np.zeros(n), y, groups=g))
    tr2, va2 = next(StratifiedGroupKFold(N_VAL_FOLDS, shuffle=True, random_state=SEED)
                    .split(np.zeros(len(tr_i)), y[tr_i], groups=g[tr_i]))
    tr_i, va_i = tr_i[tr2], tr_i[va2]
    train, val, test = (sorted(int(v) for v in hid[i]) for i in (tr_i, va_i, te_i))
    G = {k: set(int(s) for s in g[i]) for k, i in [("train",tr_i),("val",va_i),("test",te_i)]}
    for a, bb in [("train","val"),("train","test"),("val","test")]:
        assert G[a].isdisjoint(G[bb]), f"{cohort}: PATIENT overlap {a}/{bb}"
    assert len(train)+len(val)+len(test) == n, f"{cohort}: sizes do not sum"

    rec = {"cohort": cohort, "variant": "restricted", "grouped_on": "subject_id",
           "stratified_on": "lat", "seed": SEED, "n_common": int(n),
           "split_id": digest(train, val, test),
           "train": train, "val": val, "test": test}
    (OUT/f"{cohort}_canonical_split.json").write_text(json.dumps(rec))

    pr = sub[lab].mean()
    print(f"[{cohort.upper()}] full {len(full):,} -> restricted {len(r):,} "
          f"(structurally negative {len(struct_neg):,}, {100*len(struct_neg)/len(full):.1f}%)")
    print(f"          split n={n:,}  train {len(train):,} val {len(val):,} test {len(test):,}"
          f"  |  prevalence {100*pr:.2f}%  |  split_id {rec['split_id']}")
    for k, idx in [("train",tr_i),("val",va_i),("test",te_i)]:
        yy = y[idx]
        print(f"            {k:5s} n={len(yy):>6,} pos={int(yy.sum()):>5,} ({100*yy.mean():5.2f}%)")

print(f"\nwritten to {OUT.relative_to(BASE)}/ and data/matrices/*_feature_matrix_restricted.csv")
