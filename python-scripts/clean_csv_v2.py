import pandas as pd

df = pd.read_csv("C:/Users/Lenovo/Desktop/Msc Project/data/event_log/mimic_transfers_v4.csv")
df = df.dropna(subset=['activity'])
df = df.rename(columns={'case_id': 'Case ID', 'event_time': 'Timestamp', 'activity': 'Activity'})
df = df.sort_values(['Case ID', 'Timestamp'])

print(f"Rows: {len(df)}")
print(f"Unique activities: {df['Activity'].nunique()}")
print(df['Activity'].value_counts())

df.to_csv("C:/Users/Lenovo/Desktop/Msc Project/data/event_log/mimic_transfers_v4_clean.csv", index=False)
print("Done.")