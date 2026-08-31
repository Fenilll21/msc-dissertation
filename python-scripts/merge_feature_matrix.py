import pandas as pd

BASE   = "C:/Users/Lenovo/Desktop/Msc Project/data/"
FEAT   = BASE + "features/"
LABELS = BASE + "labels/"
EVENT  = BASE + "event_log/mimic_transfers_v3_clean.csv"
OUT    = BASE + "matrices/"


print("Loading files...")
icu_labels  = pd.read_csv(LABELS + "icu_left_shift_labels.csv")
sw_labels   = pd.read_csv(LABELS + "sw_left_shift_labels.csv")

demographics_icu = pd.read_csv(FEAT + "icu_demographic_features_v2.csv")
demographics_sw  = pd.read_csv(FEAT + "sw_demographic_features_v2.csv")

uhr          = pd.read_csv(FEAT + "uhr_admission_type.csv")
bmi          = pd.read_csv(FEAT + "bmi_features.csv")
chronic      = pd.read_csv(FEAT + "chronic_disease_count.csv")
icd          = pd.read_csv(FEAT + "icd_chapter_counts.csv")
charlson     = pd.read_csv(FEAT + "charlson_scores.csv")
icu_clinical = pd.read_csv(FEAT + "icu_clinical_features.csv")
sw_clinical  = pd.read_csv(FEAT + "sw_clinical_features.csv")
icu_process = pd.read_csv(FEAT + "icu_process_features_v2.csv")
sw_process  = pd.read_csv(FEAT + "sw_process_features_v2.csv")

print("Deriving ED admission flag from event log...")
el = pd.read_csv(EVENT, usecols=["Case ID", "Activity"])
el.columns = ["hadm_id", "activity"]
ed_flag = (
    el[el["activity"] == "Emergency Department"]
    .groupby("hadm_id")
    .size()
    .reset_index(name="ed_count")
    .assign(ed_admission=1)[["hadm_id", "ed_admission"]]
)

icu_clinical = icu_clinical[["hadm_id", "pct_abnormal_labs",
                               "num_medications", "polypharmacy"]]
sw_clinical  = sw_clinical[["hadm_id", "pct_abnormal_labs",
                              "num_medications", "polypharmacy"]]

for df in [uhr, bmi, chronic, icd, charlson, icu_process, sw_process]:
    if "subject_id" in df.columns:
        df.drop(columns=["subject_id"], inplace=True)

def build_matrix(labels, demographics, clinical, process, label_col):
    df = labels.copy()

    # Demographics
    df = df.merge(demographics.drop(columns=["subject_id"]),
                  on="hadm_id", how="left")

    # UHR + admission type
    df = df.merge(uhr, on="hadm_id", how="left")

    # ED admission flag
    df = df.merge(ed_flag, on="hadm_id", how="left")
    df["ed_admission"] = df["ed_admission"].fillna(0).astype(int)

    # BMI
    df = df.merge(bmi, on="hadm_id", how="left")

    # Reused clinical (pct_abnormal_labs, num_medications, polypharmacy)
    df = df.merge(clinical, on="hadm_id", how="left")

    # Chronic disease count
    df = df.merge(chronic, on="hadm_id", how="left")

    # 18 ICD chapters
    df = df.merge(icd, on="hadm_id", how="left")

    # Charlson
    df = df.merge(charlson, on="hadm_id", how="left")

    # Process features
    df = df.merge(process, on="hadm_id", how="left")

    return df

#ICU matrix 
print("Building ICU feature matrix...")
icu_matrix = build_matrix(icu_labels, demographics_icu,
                           icu_clinical,icu_process, "icu_left_shift")
print(f"ICU matrix shape: {icu_matrix.shape}")
print(f"ICU missing values:\n{icu_matrix.isnull().sum()[icu_matrix.isnull().sum() > 0]}")
print(f"ICU left shift rate: {icu_matrix['icu_left_shift'].mean():.3f}")

#SW matrix
print("\nBuilding SW feature matrix...")
sw_matrix = build_matrix(sw_labels, demographics_sw,
                          sw_clinical,sw_process, "sw_left_shift")
print(f"SW matrix shape: {sw_matrix.shape}")
print(f"SW missing values:\n{sw_matrix.isnull().sum()[sw_matrix.isnull().sum() > 0]}")
print(f"SW left shift rate: {sw_matrix['sw_left_shift'].mean():.3f}")


print(f"\nColumns ({len(icu_matrix.columns)}):")
for i, col in enumerate(icu_matrix.columns):
    print(f"  {i+1:2}. {col}")

icu_matrix.to_csv(OUT + "icu_feature_matrix_v2.csv", index=False)
sw_matrix.to_csv(OUT + "sw_feature_matrix_v2.csv", index=False)
print("\nBoth matrices saved to data/matrices/")