#!/usr/bin/env python
"""
split_audit.py -- make the ML notebook and the DL scripts evaluate on the SAME
patients.

Two modes:

  verify  Reproduces the ML test split (positional stratified split on the CSV
          row order the notebook uses) and the DL test split (align matrix to
          sequence order, then positional split) and reports how much their test
          sets overlap. Run this FIRST -- if the sets already coincide, you only
          need to document the split and no code changes are required.

  build   Creates ONE canonical split per cohort, keyed on hadm_id over the
          common cohort (admissions present in BOTH the matrix and the sequence
          file), deterministic and stratified. Writes {cohort}_canonical_split.json
          with train/val/test hadm_id lists. Both pipelines then select rows by
          membership in these lists instead of re-deriving a positional split, so
          every ML-vs-DL-vs-DS comparison is on identical patients.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.10
COHORTS = {
    "icu": dict(matrix="icu_feature_matrix_v2.csv", ml_matrix="icu_feature_matrix_v3_bow.csv",
                seqs="icu_sequences.npy", order="icu_hadm_id_order.npy", label="icu_left_shift"),
    "sw":  dict(matrix="sw_feature_matrix_v2.csv", ml_matrix="sw_feature_matrix_v3_bow.csv",
                seqs="sw_sequences.npy", order="sw_hadm_id_order.npy", label="sw_left_shift"),
}


def _stratified_test_ids(hadm_ids, y, seed=SEED):
    """Positional stratified 80/20; return the test hadm_ids (as a set)."""
    idx = np.arange(len(y))
    _, te = train_test_split(idx, test_size=TEST_SIZE, random_state=seed, stratify=y)
    return set(hadm_ids[te])


def ml_test_ids(mat_dir: Path, cfg, use_ml_matrix=True):
    """Reproduce the notebook's split: positional split on the CSV row order."""
    fname = cfg["ml_matrix"] if use_ml_matrix else cfg["matrix"]
    path = mat_dir / fname
    if not path.exists():                     
        path = mat_dir / cfg["matrix"]
    df = pd.read_csv(path)
    return _stratified_test_ids(df["hadm_id"].values, df[cfg["label"]].values), len(df), path.name


def dl_test_ids(mat_dir: Path, seq_dir: Path, cfg):
    """Reproduce the DL split: align matrix to sequence order, then positional split."""
    df = pd.read_csv(mat_dir / cfg["matrix"])
    order = np.load(seq_dir / cfg["order"], allow_pickle=True)
    ids = set(df["hadm_id"])
    keep = np.array([h in ids for h in order])
    kept = order[keep]
    m = df.set_index("hadm_id").reindex(kept).reset_index()
    return _stratified_test_ids(m["hadm_id"].values, m[cfg["label"]].values), len(m)


def verify(mat_dir: Path, seq_dir: Path):
    print("=" * 74)
    print("VERIFY: does the ML notebook evaluate on the same test patients as DL?")
    print("=" * 74)
    for ck, cfg in COHORTS.items():
        ml_ids, n_ml, ml_src = ml_test_ids(mat_dir, cfg)
        dl_ids, n_dl = dl_test_ids(mat_dir, seq_dir, cfg)
        inter = ml_ids & dl_ids
        union = ml_ids | dl_ids
        jac = len(inter) / len(union) if union else float("nan")
        print(f"\n[{ck.upper()}]  ML cohort n={n_ml} (from {ml_src})  |  DL cohort n={n_dl}")
        print(f"  ML test={len(ml_ids)}  DL test={len(dl_ids)}  shared={len(inter)}  "
              f"ML-only={len(ml_ids - dl_ids)}  DL-only={len(dl_ids - ml_ids)}")
        print(f"  Jaccard(test sets) = {jac:.4f}")
        if jac == 1.0:
            print("  -> IDENTICAL test sets. No change needed; just document the split.")
        elif n_ml != n_dl:
            print("  -> Cohorts differ in size AND test sets differ. Use `build` for a "
                  "canonical split on the common cohort.")
        else:
            print("  -> Same cohort size but DIFFERENT test patients (row-order effect). "
                  "Use `build` so ML and DL share one split.")


def build(mat_dir: Path, seq_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("BUILD: canonical hadm_id-keyed split on the common cohort")
    print("=" * 74)
    for ck, cfg in COHORTS.items():
        df = pd.read_csv(mat_dir / cfg["matrix"])
        order = np.load(seq_dir / cfg["order"], allow_pickle=True)
        common = np.array(sorted(set(df["hadm_id"]) & set(order.tolist())))  # deterministic
        sub = df.set_index("hadm_id").reindex(common).reset_index()
        y = sub[cfg["label"]].values.astype(int)

        tr_ids, te_ids = train_test_split(common, test_size=TEST_SIZE, random_state=SEED, stratify=y)
        y_tr = sub.set_index("hadm_id").loc[tr_ids, cfg["label"]].values.astype(int)
        tr2_ids, va_ids = train_test_split(tr_ids, test_size=VAL_SIZE, random_state=SEED, stratify=y_tr)

        split = {"cohort": ck, "seed": SEED, "test_size": TEST_SIZE, "val_size": VAL_SIZE,
                 "n_common": int(len(common)),
                 "train": [int(x) for x in tr2_ids],
                 "val":   [int(x) for x in va_ids],
                 "test":  [int(x) for x in te_ids]}
        # integrity: partition covers the common cohort with no overlap
        s_tr, s_va, s_te = set(split["train"]), set(split["val"]), set(split["test"])
        assert s_tr.isdisjoint(s_va) and s_tr.isdisjoint(s_te) and s_va.isdisjoint(s_te)
        assert s_tr | s_va | s_te == set(int(x) for x in common)

        p = out_dir / f"{ck}_canonical_split.json"
        p.write_text(json.dumps(split))
        pos = sub[sub["hadm_id"].isin(s_te)][cfg["label"]].mean()
        print(f"[{ck.upper()}] common={len(common)}  train={len(s_tr)} val={len(s_va)} "
              f"test={len(s_te)}  test pos-rate={pos:.3f}  -> {p.name}")
    print(f"\nWrote canonical splits to {out_dir}. See the module docstring for the "
          f"two small edits to wire them into preprocess() and the DL loader.")


def load_canonical_split(split_dir, cohort):
    """Helper for both pipelines: returns dict of train/val/test hadm_id SETS."""
    s = json.loads((Path(split_dir) / f"{cohort}_canonical_split.json").read_text())
    return {"train": set(s["train"]), "val": set(s["val"]), "test": set(s["test"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["verify", "build"])
    ap.add_argument("--matrices", required=True)
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--out", default="./splits")
    a = ap.parse_args()
    mat, seq = Path(a.matrices), Path(a.sequences)
    if a.mode == "verify":
        verify(mat, seq)
    else:
        build(mat, seq, Path(a.out))


if __name__ == "__main__":
    main()
