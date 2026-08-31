"""
merge deep results (Aire) with local results, verifying
that both halves came from the same partition.

Chapter 3 section 3.4: every result row carries the `split_id` of the partition it
was produced under. The merge REFUSES to proceed if the digests disagree, which
is the only reliable way to catch a stale Aire copy after the fact.
"""
import re, sys
from pathlib import Path
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"
PAT = re.compile(r"^(.*?)_results_(old|corr|corrpacu|collapse|lat|restricted)"
                 r"(?:_s(\d+))?\.csv$")

frames = []
for f in sorted(RES.glob("*_results_*.csv")):
    m = PAT.match(f.name)
    if not m:
        continue
    d = pd.read_csv(f)
    d["Variant"] = m.group(2)
    if "Seed" not in d.columns:
        d["Seed"] = int(m.group(3) or 42)
    if "Model" not in d.columns:
        d["Model"] = m.group(1)
    d["_source_file"] = f.name
    frames.append(d)

if not frames:
    sys.exit("no tagged result files found - run the jobs first")

allr = pd.concat(frames, ignore_index=True)

#split_id check
if "split_id" not in allr.columns:
    sys.exit("FATAL: no split_id column. Results predate Part 3 and cannot be "
             "verified as coming from one partition. Re-run.")

missing = allr[allr.split_id.isna() | (allr.split_id.astype(str).str.strip() == "")]
if len(missing):
    print(f"WARNING: {len(missing)} rows carry no split_id, from: "
          f"{sorted(missing._source_file.unique())}")

ok = True
for (coh, var), g in allr.dropna(subset=["split_id"]).groupby(["Cohort", "Variant"]):
    ids = sorted(g.split_id.astype(str).unique())
    if len(ids) > 1:
        ok = False
        print(f"FATAL: {coh}/{var} mixes {len(ids)} split_ids: {ids}")
        for i in ids:
            print(f"        {i}: {sorted(g[g.split_id == i]._source_file.unique())[:4]}")
if not ok:
    sys.exit("\nRefusing to merge. Results came from different partitions - most "
             "likely a stale script or split file on one machine. Identify the "
             "offending files above, re-run those, then collect again.")

for coh, g in allr.dropna(subset=["split_id"]).groupby("Cohort"):
    print(f"  {coh}: split_id {g.split_id.iloc[0]}  ({len(g):,} rows, "
          f"{g._source_file.nunique()} files)  OK")

allr = allr.drop(columns=["_source_file"])
allr.to_csv(RES / "all_results_long.csv", index=False)

keys = [k for k in ["Model", "Cohort", "Condition", "Variant"] if k in allr.columns]
agg = allr.groupby(keys)[["AUROC", "AUPRC", "F1"]].agg(["mean", "std", "count"]).round(4)
agg.to_csv(RES / "all_results_summary.csv")

print(f"\n{len(allr)} rows from {len(frames)} files -> results/all_results_long.csv")
print(agg.to_string())
