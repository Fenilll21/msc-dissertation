import os, json
import pathlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

BASE=Path(__file__).resolve().parent.parent; MAT=BASE/"data"/"matrices"; RES=BASE/"results"; RES.mkdir(exist_ok=True)
SEED=int(os.environ.get("MSC_SEED",42)); SW_MATRIX=os.environ.get("MSC_SW_MATRIX","sw_feature_matrix_v2.csv")
ICU_MATRIX=os.environ.get("MSC_ICU_MATRIX","icu_feature_matrix_v2.csv")
SPLIT_DIR=Path(os.environ.get("MSC_SPLIT_DIR",BASE/"data"/"splits"/"old")); TAG=os.environ.get("MSC_TAG","")
COHORTS   = [c.strip().lower() for c in os.environ.get("MSC_COHORTS", "icu,sw").split(",")]
np.random.seed(SEED)

DEMOGRAPHICS=["age_band","gender","insurance","ethnicity","marital_status"]
CLINICAL_CORE=["uhr_count","ed_admission","admission_type","bmi_category","pct_abnormal_labs","num_medications","polypharmacy"]
CHARLSON=["charlson_score"]; CHRONIC=["chronic_disease_count"]
PROCESS=["admission_location","accumulated_duration_hrs","icu_duration_hrs","sw_duration_hrs","num_icu_stays","num_sw_stays","total_events"]
def _validated_split(split_dir, cohort):
    import hashlib as _hl
    p = pathlib.Path(split_dir)/f"{cohort}_canonical_split.json"
    s = json.loads(p.read_text())
    assert s.get("grouped_on") == "subject_id", f"{p}: not grouped on subject_id (stale split file?)"
    assert s.get("stratified_on") == "lat", f"{p}: not stratified on the primary label"
    assert "split_id" in s, f"{p}: no split_id (pre-fix split file)"
    tr, va, te = (sorted(int(x) for x in s[k]) for k in ("train", "val", "test"))
    h = _hl.sha256()
    for part in (tr, va, te):
        h.update(b"|"); h.update(",".join(map(str, part)).encode())
    assert h.hexdigest()[:16] == s["split_id"], f"{p}: contents do not match split_id"
    return {"train": tr, "val": va, "test": te, "split_id": s["split_id"]}

CATEGORICAL=["age_band","gender","insurance","ethnicity","marital_status","admission_type","bmi_category","admission_location"]
LR_PARAMS=dict(class_weight="balanced",random_state=SEED,max_iter=2000,solver="lbfgs")

def load(cohort):
    df=pd.read_csv(MAT/(ICU_MATRIX if cohort=="icu" else SW_MATRIX))
    assert df.shape[1]==42 and not df.isnull().any().any()
    icd=[c for c in df.columns if c.startswith("icd_ch")]
    groups={"demographics":DEMOGRAPHICS,"clinical_core":CLINICAL_CORE,"icd_chapters":icd,
            "charlson":CHARLSON,"chronic_count":CHRONIC,"process":PROCESS}
    flat=[f for g in groups.values() for f in g]
    assert len(flat)==len(set(flat))==39, f"groups must partition 39 features, got {len(flat)}"
    return df, groups, f"{cohort}_left_shift"

def evaluate(df,label,cols,cohort):
    if not cols: return dict(AUROC=float("nan"),AUPRC=float("nan"),n_features=0)
    _sp=_validated_split(SPLIT_DIR, cohort)
    hid=df["hadm_id"].values
    is_tr=np.isin(hid,np.array(_sp["train"])); is_te=np.isin(hid,np.array(_sp["test"]))
    assert not (is_tr & is_te).any(), f"{cohort}: train/test overlap"
    X=pd.get_dummies(df[cols],columns=[c for c in CATEGORICAL if c in cols],drop_first=False)
    Xtr=X[is_tr].astype(np.float32).values; Xte=X[is_te].astype(np.float32).values
    ytr=df[label].values[is_tr].astype(int); yte=df[label].values[is_te].astype(int)
    sc=StandardScaler().fit(Xtr); p=LogisticRegression(**LR_PARAMS).fit(sc.transform(Xtr),ytr).predict_proba(sc.transform(Xte))[:,1]
    return dict(AUROC=round(roc_auc_score(yte,p),4),AUPRC=round(average_precision_score(yte,p),4),n_features=Xtr.shape[1])

logo=[];inc=[];solo=[];tv=[]
for cohort in COHORTS:
    df,G,label=load(cohort); allc=[f for g in G.values() for f in g]
    full=evaluate(df,label,allc,cohort)
    logo.append(dict(cohort=cohort.upper(),removed_group="(none) FULL",**full,AUROC_drop=0.0))
    for gn,gc in G.items():
        r=evaluate(df,label,[c for c in allc if c not in gc],cohort)
        logo.append(dict(cohort=cohort.upper(),removed_group=gn,**r,AUROC_drop=round(full["AUROC"]-r["AUROC"],4)))
        solo.append(dict(cohort=cohort.upper(),group_alone=gn,**evaluate(df,label,gc,cohort)))
    cum=[]; prev=None
    for gn in ["demographics","clinical_core","chronic_count","charlson","icd_chapters","process"]:
        cum+=G[gn]; r=evaluate(df,label,cum,cohort)
        inc.append(dict(cohort=cohort.upper(),added_group=gn,cumulative_AUROC=r["AUROC"],
                        cumulative_AUPRC=r["AUPRC"],AUROC_gain=None if prev is None else round(r["AUROC"]-prev,4),
                        n_features=r["n_features"])); prev=r["AUROC"]
    # ── P-4: temporal validity ──────────────────────────────────────────────
    disch=G["icd_chapters"]+G["charlson"]+G["chronic_count"]
    r=evaluate(df,label,[c for c in allc if c not in disch],cohort)
    tv.append(dict(cohort=cohort.upper(),model="FULL (39 features)",**full,AUROC_cost=0.0))
    tv.append(dict(cohort=cohort.upper(),model=f"minus {len(disch)} discharge-assigned features",**r,
                   AUROC_cost=round(full["AUROC"]-r["AUROC"],4)))

for name,rows in [("leave_one_out",logo),("incremental",inc),("single_group",solo),("temporal_validity",tv)]:
    p=RES/f"ablation_{name}{TAG}.csv"; pd.DataFrame(rows).to_csv(p,index=False); print(f"wrote {p.name}")
print("\n=== TEMPORAL VALIDITY (P-4) ==="); print(pd.DataFrame(tv).to_string(index=False))
