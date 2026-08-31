"""
One split per cohort. Grouped on subject_id so no patient appears in more than
one partition. Stratified on the PRIMARY label (`lat`) only, then reused
unchanged by every label variant — so a difference between variants is a label
effect, never a split effect.

Route: StratifiedGroupKFold(n_splits=5) for test, then n_splits=9 on the
remainder for validation (Part 1 section 1.3).

ICU and SW are separate splits. A patient in ICU-train and SW-test is not
leakage; those models share no data.

Each file carries `grouped_on`, `stratified_on` and a `split_id` digest.
"""
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

BASE = Path(__file__).resolve().parent.parent
MAT, SEQ, LAB = BASE/"data"/"matrices", BASE/"data"/"sequences", BASE/"data"/"labels"
OUT = BASE/"data"/"splits"/"canonical"
SEED, N_TEST_FOLDS, N_VAL_FOLDS = 42, 5, 9

PRIMARY = {"icu": "icu_feature_matrix_v2_lat.csv", "sw": "sw_feature_matrix_v2_lat.csv"}


def split_digest(tr, va, te):
    h = hashlib.sha256()
    for part in (sorted(map(int, tr)), sorted(map(int, va)), sorted(map(int, te))):
        h.update(b"|"); h.update(",".join(map(str, part)).encode())
    return h.hexdigest()[:16]


def build(cohort):
    lab = f"{cohort}_left_shift"
    m = pd.read_csv(MAT/PRIMARY[cohort], usecols=["hadm_id", lab])
    order = np.load(SEQ/f"{cohort}_hadm_id_order.npy", allow_pickle=True)
    common = np.array(sorted(set(m.hadm_id) & set(order.tolist())))
    m = m.set_index("hadm_id").reindex(common).reset_index()

    sid = pd.read_csv(LAB/f"{cohort}_left_shift_labels.csv", usecols=["hadm_id", "subject_id"])
    m = m.merge(sid, on="hadm_id", how="left")
    assert m.subject_id.notna().all(), f"{cohort}: hadm_id without subject_id"

    y = m[lab].values.astype(int)
    g = m.subject_id.values
    hid = m.hadm_id.values
    n = len(m)

    tr_i, te_i = next(StratifiedGroupKFold(N_TEST_FOLDS, shuffle=True, random_state=SEED)
                      .split(np.zeros(n), y, groups=g))
    tr2, va2 = next(StratifiedGroupKFold(N_VAL_FOLDS, shuffle=True, random_state=SEED)
                    .split(np.zeros(len(tr_i)), y[tr_i], groups=g[tr_i]))
    tr_i, va_i = tr_i[tr2], tr_i[va2]

    train, val, test = (sorted(int(x) for x in hid[i]) for i in (tr_i, va_i, te_i))
    S = {"train": set(train), "val": set(val), "test": set(test)}
    G = {k: set(int(s) for s in g[i]) for k, i in [("train", tr_i), ("val", va_i), ("test", te_i)]}

    # assertions 1-3 and 7, at the point of construction
    assert S["train"].isdisjoint(S["val"]) and S["train"].isdisjoint(S["test"]) \
        and S["val"].isdisjoint(S["test"]), f"{cohort}: admission overlap"
    assert S["train"] | S["val"] | S["test"] == set(int(x) for x in hid), f"{cohort}: partition gap"
    assert len(train) + len(val) + len(test) == n, f"{cohort}: sizes do not sum"
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert G[a].isdisjoint(G[b]), f"{cohort}: PATIENT overlap {a}/{b}: {len(G[a] & G[b])}"

    rec = {"cohort": cohort, "variant": "canonical",
           "grouped_on": "subject_id", "stratified_on": "lat",
           "seed": SEED, "n_common": int(n),
           "split_id": split_digest(train, val, test),
           "train": train, "val": val, "test": test}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/f"{cohort}_canonical_split.json").write_text(json.dumps(rec))

    print(f"[{cohort.upper()}] n={n:,}  train {len(train):,} ({100*len(train)/n:.1f}%) "
          f"val {len(val):,} ({100*len(val)/n:.1f}%) test {len(test):,} ({100*len(test)/n:.1f}%)")
    print(f"          patients  train {len(G['train']):,} val {len(G['val']):,} test {len(G['test']):,}"
          f"  |  overlap 0/0/0  |  split_id {rec['split_id']}")
    return rec


if __name__ == "__main__":
    for c in ("icu", "sw"):
        build(c)
    print(f"\nwritten to {OUT.relative_to(BASE)}/")
