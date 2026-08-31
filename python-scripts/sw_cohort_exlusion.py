# SW Cohort Exclusion
# Removes 6501 Surgery-to-Discharge only patients from SW feature matrix
# Left shift rate: 13.3% -> 14.3% after exclusion
# Final SW cohort: 67,066 admissions

import pandas as pd

sw = pd.read_csv("C:/Users/Lenovo/Desktop/Msc Project/data/matrices/sw_feature_matrix_v2.csv")
exclude = pd.read_csv("C:/Users/Lenovo/Desktop/Msc Project/data/features/sw_exclude_ids.csv")

print(f"SW matrix before exclusion: {sw.shape}")
print(f"Left shift rate before: {sw['sw_left_shift'].mean():.3f}")
print(f"IDs to exclude: {len(exclude)}")

# Filter
exclude_ids = set(exclude['hadm_id'])
sw_filtered = sw[~sw['hadm_id'].isin(exclude_ids)].copy()
sw_filtered = sw_filtered.reset_index(drop=True)

print(f"\nSW matrix after exclusion: {sw_filtered.shape}")
print(f"Left shift rate after: {sw_filtered['sw_left_shift'].mean():.3f}")
print(f"Removed: {len(sw) - len(sw_filtered)} patients")
print(f"Missing values: {sw_filtered.isnull().sum().sum()}")

sw_filtered.to_csv("C:/Users/Lenovo/Desktop/Msc Project/data/matrices/sw_feature_matrix_v2.csv", index=False)
print("\nSaved.")
