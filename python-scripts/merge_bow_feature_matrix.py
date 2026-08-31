import pandas as pd

BASE = "C:/Users/Lenovo/Desktop/Msc Project/data/"

# ICU
icu = pd.read_csv(BASE + "matrices/icu_feature_matrix_v2.csv")
icu_bow = pd.read_csv(BASE + "features/icu_bow_features.csv")
icu_full = icu.merge(icu_bow, on='hadm_id', how='left')
icu_full.to_csv(BASE + "matrices/icu_feature_matrix_v3_bow.csv", index=False)
print(f"ICU with BoW: {icu_full.shape}")

# SW
sw = pd.read_csv(BASE + "matrices/sw_feature_matrix_v2.csv")
sw_bow = pd.read_csv(BASE + "features/sw_bow_features.csv")
sw_full = sw.merge(sw_bow, on='hadm_id', how='left')
sw_full.to_csv(BASE + "matrices/sw_feature_matrix_v3_bow.csv", index=False)
print(f"SW with BoW: {sw_full.shape}")

print("Done.")