import pandas as pd

df = pd.read_csv("E:/mimic_transfers_v3_clean.csv")

# Check first few events for a sample case
sample = df[df['Case ID'] == df['Case ID'].iloc[0]]
print(sample)

# Check what the first activity is for first 10 cases
first_events = df.groupby('Case ID').first().reset_index()
print(first_events['Activity'].value_counts())