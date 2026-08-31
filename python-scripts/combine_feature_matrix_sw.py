import pandas as pd

labels = pd.read_csv("E:/sw_left_shift_labels.csv")
demographic = pd.read_csv("E:/sw_demographic_features.csv")
clinical = pd.read_csv("E:/sw_clinical_features.csv")
process = pd.read_csv("E:/sw_process_features.csv")

df = labels.merge(demographic, on=['hadm_id', 'subject_id'], how='inner')
df = df.merge(clinical, on='hadm_id', how='inner')
df = df.merge(process, on='hadm_id', how='inner')

print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"Columns: {list(df.columns)}")
print(f"\nLeft shift distribution:")
print(df['sw_left_shift'].value_counts())
print(f"\nMissing values:")
print(df.isnull().sum())


df.to_csv("E:/sw_feature_matrix.csv", index=False)
print("\nSW feature matrix saved.")