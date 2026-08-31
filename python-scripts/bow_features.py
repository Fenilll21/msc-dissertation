"""
bag-of-words control-flow features over the activity prefix.

Two corrections applied since the first version:

  P-9 (12 Aug 2026): the prefix is cut STRICTLY BEFORE the prediction point.
      It was '<=', which admitted the transfer being predicted -- a near
      deterministic label proxy.

  D2  (21 Aug 2026): the vectoriser is fitted on the canonical TRAIN partition
      only. It was fitted over every admission in the cohort, so `min_df=2` was
      computed across train, validation and test. 15 n-grams per cohort were
      admitted solely because val/test rows were visible. The fitted columns are
      baked into the output file, so every downstream consumer inherited the
      leak invisibly.

Trace construction is vectorised (groupby rather than a per-admission filter).

"""
import os, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

BASE = Path(__file__).resolve().parent.parent
FEAT, MAT = BASE/"data"/"features", BASE/"data"/"matrices"
EVENT = BASE/"data"/"event_log"/"mimic_transfers_v4_clean.csv"
SPLIT_DIR = Path(os.environ.get("MSC_SPLIT_DIR", BASE/"data"/"splits"/"canonical"))
NGRAM, MIN_DF = (1, 3), 2


def validated_split(cohort):
    p = SPLIT_DIR/f"{cohort}_canonical_split.json"
    s = json.loads(p.read_text())
    assert s.get("grouped_on") == "subject_id", f"{p}: not patient-grouped (stale split file?)"
    assert s.get("stratified_on") == "lat", f"{p}: not stratified on the primary label"
    tr, va, te = (sorted(int(x) for x in s[k]) for k in ("train", "val", "test"))
    h = hashlib.sha256()
    for part in (tr, va, te):
        h.update(b"|"); h.update(",".join(map(str, part)).encode())
    assert h.hexdigest()[:16] == s["split_id"], f"{p}: contents do not match split_id"
    return s


print("Loading event log...")
el = pd.read_csv(EVENT, parse_dates=["Timestamp"])
el.columns = ["hadm_id", "timestamp", "activity"]
el["tok"] = el["activity"].str.replace(" ", "_", regex=False)

for cohort, matrix_file in [("icu", "icu_feature_matrix_v2.csv"), ("sw", "sw_feature_matrix_v2.csv")]:
    sp = validated_split(cohort)
    pp = pd.read_csv(FEAT/f"{cohort}_prediction_points.csv", parse_dates=["prediction_point"])
    ids = pd.read_csv(MAT/matrix_file, usecols=["hadm_id"])["hadm_id"]

    # prefix strictly before the prediction point (P-9), vectorised
    m = el.merge(pp, on="hadm_id", how="inner")
    m = m[m.timestamp < m.prediction_point].sort_values(["hadm_id", "timestamp"], kind="mergesort")
    traces = m.groupby("hadm_id", sort=False)["tok"].apply(" ".join).reindex(ids).fillna("").values

    train_mask = np.isin(ids.values, np.array(sp["train"], dtype=np.int64))
    assert train_mask.sum() > 0, f"{cohort}: no training rows matched the split"

    # D2: fit on TRAIN ONLY, then transform every row
    cv = CountVectorizer(ngram_range=NGRAM, token_pattern=r"\S+", min_df=MIN_DF)
    cv.fit(traces[train_mask])
    M = cv.transform(traces)

    bow = pd.DataFrame(M.toarray(), columns=["bow_" + f for f in cv.get_feature_names_out()])
    bow.insert(0, "hadm_id", ids.values)
    bow.to_csv(FEAT/f"{cohort}_bow_features.csv", index=False)

    base = pd.read_csv(MAT/matrix_file)
    full = base.merge(bow, on="hadm_id", how="left")
    full.to_csv(MAT/f"{cohort}_feature_matrix_v3_bow.csv", index=False)

    print(f"  {cohort.upper()}: fitted on {train_mask.sum():,} train rows | "
          f"{M.shape[1]} n-gram features | matrix {full.shape}")
