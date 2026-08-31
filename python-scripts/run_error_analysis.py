import os, json
import pathlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score

BASE=Path(__file__).resolve().parent.parent; MAT=BASE/"data"/"matrices"; RES=BASE/"results"; RES.mkdir(exist_ok=True)
SEED=int(os.environ.get("MSC_SEED",42)); SW_MATRIX=os.environ.get("MSC_SW_MATRIX","sw_feature_matrix_v2.csv")
ICU_MATRIX=os.environ.get("MSC_ICU_MATRIX","icu_feature_matrix_v2.csv")
SPLIT_DIR=Path(os.environ.get("MSC_SPLIT_DIR",BASE/"data"/"splits"/"old")); TAG=os.environ.get("MSC_TAG","")
COHORTS   = [c.strip().lower() for c in os.environ.get("MSC_COHORTS", "icu,sw").split(",")]
np.random.seed(SEED)
def _validated_split(split_dir, cohort):
    """Read the canonical partition, asserting it is the patient-grouped one."""
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

IDS=["hadm_id","subject_id"]
CATEGORICAL=["age_band","gender","insurance","ethnicity","marital_status","admission_type","bmi_category","admission_location"]
PROFILE_CATS=["age_band","gender","insurance","admission_type","bmi_category","ethnicity"]
RF_PARAMS=dict(n_estimators=200,max_depth=20,min_samples_split=5,min_samples_leaf=2,
               max_features="sqrt",class_weight="balanced",random_state=SEED,n_jobs=-1)

def fit_and_tag(cohort):
    label=f"{cohort}_left_shift"
    df=pd.read_csv(MAT/(ICU_MATRIX if cohort=="icu" else SW_MATRIX))
    assert df.shape[1]==42
    cols=[c for c in df.columns if c not in IDS+[label]]; assert len(cols)==39
    sp=_validated_split(SPLIT_DIR, cohort)
    hid=df["hadm_id"].values
    te=np.isin(hid,np.array(sp["test"])); va=np.isin(hid,np.array(sp["val"])); tr=~(te|va)
    X=pd.get_dummies(df[cols],columns=CATEGORICAL,drop_first=False).astype(np.float32)
    sc=StandardScaler().fit(X[tr].values)
    y=df[label].values.astype(int)
    mdl=RandomForestClassifier(**RF_PARAMS).fit(sc.transform(X[tr].values),y[tr])
    p_va=mdl.predict_proba(sc.transform(X[va].values))[:,1]
    p_te=mdl.predict_proba(sc.transform(X[te].values))[:,1]
    grid=np.linspace(0.05,0.95,91)
    thr=float(grid[np.argmax([f1_score(y[va],(p_va>=t).astype(int),zero_division=0) for t in grid])])
    pred=(p_te>=thr).astype(int); yt=y[te]; auroc=roc_auc_score(yt,p_te)
    cat=np.full(len(yt),"",dtype=object)
    cat[(yt==1)&(pred==1)]="TP"; cat[(yt==0)&(pred==0)]="TN"
    cat[(yt==0)&(pred==1)]="FP"; cat[(yt==1)&(pred==0)]="FN"
    out=df.loc[te,:].copy()
    out["_true"],out["_pred"],out["_proba"],out["_error_group"]=yt,pred,p_te,cat
    print(f"[{cohort.upper()}] AUROC={auroc:.4f} thr={thr:.2f} (val-tuned, F1={f1_score(yt,pred,zero_division=0):.4f}) "
          f"TP={np.sum(cat=='TP')} FN={np.sum(cat=='FN')} FP={np.sum(cat=='FP')} TN={np.sum(cat=='TN')}",flush=True)
    return out,auroc,thr

NUMERIC=["uhr_count","ed_admission","pct_abnormal_labs","num_medications","polypharmacy",
         "chronic_disease_count","charlson_score","accumulated_duration_hrs","icu_duration_hrs",
         "sw_duration_hrs","num_icu_stays","num_sw_stays","total_events"]

profs,fntps,cats,meta=[],[],[],[]
for cohort in COHORTS:
    t,auroc,thr=fit_and_tag(cohort); C=cohort.upper()
    meta.append(dict(cohort=C,AUROC=round(auroc,4),threshold=thr,
                     n_test=len(t),**t["_error_group"].value_counts().to_dict()))
    num=NUMERIC+[c for c in t.columns if c.startswith("icd_ch")]
    pr=t.groupby("_error_group")[num].mean().T; pr.columns=[f"{C}_{c}" for c in pr.columns]
    pr.to_csv(RES/f"error_analysis_{cohort}_group_means{TAG}.csv")
    fn,tp=t[t._error_group=="FN"],t[t._error_group=="TP"]
    rows=[dict(cohort=C,feature=f,FN_mean=round(fn[f].mean(),3),TP_mean=round(tp[f].mean(),3),
               abs_diff=round(fn[f].mean()-tp[f].mean(),3),
               rel_diff_pct=round(100*(fn[f].mean()-tp[f].mean())/(abs(tp[f].mean()) if abs(tp[f].mean())>1e-9 else 1),1))
          for f in num]
    fntps.append(pd.DataFrame(rows).reindex(pd.DataFrame(rows)["abs_diff"].abs().sort_values(ascending=False).index))
    pos=t[t._true==1]
    frames=[]
    for col in PROFILE_CATS:
        g=pos.groupby(col)["_error_group"].apply(lambda s:(s=="FN").mean()).rename("miss_rate")
        n=pos.groupby(col).size().rename("n_positives")
        f=pd.concat([g,n],axis=1).reset_index().rename(columns={col:"value"}); f.insert(0,"feature",col); frames.append(f)
    c=pd.concat(frames,ignore_index=True); c.insert(0,"cohort",C); c["reliable"]=c["n_positives"]>=20; cats.append(c)
    t.to_csv(RES/f"error_analysis_{cohort}_tagged{TAG}.csv",index=False)

pd.concat(fntps,ignore_index=True).to_csv(RES/f"error_analysis_fn_vs_tp{TAG}.csv",index=False)
pd.concat(cats,ignore_index=True).to_csv(RES/f"error_analysis_categorical_missrate{TAG}.csv",index=False)
pd.DataFrame(meta).to_csv(RES/f"error_analysis_meta{TAG}.csv",index=False)
print("\n=== headline: FN vs TP, top 6 by absolute difference ===")
for d in fntps: print(d.head(6).to_string(index=False)); print()
