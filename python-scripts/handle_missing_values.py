import pandas as pd

# ICU feature matrix
icu = pd.read_csv("E:/icu_feature_matrix.csv")
icu['insurance'] = icu['insurance'].fillna('UNKNOWN')
icu['marital_status'] = icu['marital_status'].fillna('UNKNOWN')
print("ICU missing values after fix:")
print(icu.isnull().sum()[icu.isnull().sum() > 0])
icu.to_csv("E:/icu_feature_matrix.csv", index=False)
print(f"ICU saved. Shape: {icu.shape}")

#SW feature matrix
sw = pd.read_csv("E:/sw_feature_matrix.csv")
sw['insurance'] = sw['insurance'].fillna('UNKNOWN')
sw['marital_status'] = sw['marital_status'].fillna('UNKNOWN')
print("\nSW missing values after fix:")
print(sw.isnull().sum()[sw.isnull().sum() > 0])
sw.to_csv("E:/sw_feature_matrix.csv", index=False)
print(f"SW saved. Shape: {sw.shape}")

print("\nBoth feature matrices clean and saved.")