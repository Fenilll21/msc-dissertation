import pandas as pd

df = pd.read_csv("E:/mimic_transfers_v3.csv")

# Remove rows where activity is NULL (excluded categories)
df = df.dropna(subset=["activity"])

# Convert timestamp
df["event_time"] = pd.to_datetime(df["event_time"])

# Sort by case and time
df = df.sort_values(["case_id", "event_time"])

# Rename for Disco
df = df.rename(columns={
    "case_id": "Case ID",
    "activity": "Activity",
    "event_time": "Timestamp"
})

print(df["Activity"].value_counts())
print(f"Total cases: {df['Case ID'].nunique()}")
print(f"Total events: {len(df)}")

df.to_csv("E:/mimic_transfers_v3_clean.csv", index=False)
print("Done.")