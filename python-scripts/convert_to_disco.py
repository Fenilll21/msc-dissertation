import pandas as pd
import argparse
import os
import sys

DEFAULT_INPUT  = r"E:/mimic_transfers_v3.csv"
DEFAULT_OUTPUT = r"E:/mimic_disco_ready.csv"


COLUMN_MAP = {
    "case_id"   : "Case ID",
    "activity"  : "Activity",
    "event_time": "Timestamp",
}


EXCLUDE_ACTIVITIES = {"UNKNOWN", "Unknown", ""}


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV with basic validation."""
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    print(f"[INFO] Loading: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"[INFO] Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"[INFO] Columns found: {list(df.columns)}")
    return df


def validate_columns(df: pd.DataFrame, col_map: dict) -> None:
    """Check that all required source columns exist."""
    missing = [col for col in col_map if col not in df.columns]
    if missing:
        print(f"[ERROR] Missing required columns: {missing}")
        print(f"        Available columns: {list(df.columns)}")
        print("        Update COLUMN_MAP at the top of the script to match your CSV.")
        sys.exit(1)


def convert(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:

    df = df[[col for col in col_map.keys()]].copy()


    df = df.dropna(subset=["activity"])
    df = df[~df["activity"].isin(EXCLUDE_ACTIVITIES)]

    df["event_time"] = pd.to_datetime(df["event_time"], infer_datetime_format=True, errors="coerce")

    invalid_ts = df["event_time"].isna().sum()
    if invalid_ts > 0:
        print(f"[WARN] Dropped {invalid_ts:,} rows with unparseable timestamps.")
    df = df.dropna(subset=["event_time"])

    df["event_time"] = df["event_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    df = df.rename(columns=col_map)

    df = df.sort_values(["Case ID", "Timestamp"]).reset_index(drop=True)

    return df


def print_summary(df: pd.DataFrame) -> None:
    print(f"  Total events : {len(df):,}")
    print(f"  Unique cases : {df['Case ID'].nunique():,}")
    print(f"  Activities   : {df['Activity'].nunique():,} unique")
    print("\n  Activity breakdown:")
    for activity, count in df["Activity"].value_counts().items():
        print(f"    {activity:<50} {count:>10,}")


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8")
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"[INFO] Saved Disco-ready file: {path}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Convert MIMIC-IV CSV to Disco format.")
    parser.add_argument("--input",  "-i", default=DEFAULT_INPUT,  help="Path to input CSV")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="Path for output CSV")
    args = parser.parse_args()

    df = load_csv(args.input)
    validate_columns(df, COLUMN_MAP)
    df = convert(df, COLUMN_MAP)
    print_summary(df)
    save_csv(df, args.output)
    print("[DONE] Ready to import into Disco.")


if __name__ == "__main__":
    main()