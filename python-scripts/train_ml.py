import os, time, json
import pathlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score)

BASE = Path(__file__).resolve().parent.parent
MAT  = BASE/"data"/"matrices"; RES = BASE/"results"; RES.mkdir(exist_ok=True)
SEEDS     = [int(x) for x in os.environ.get("MSC_SEEDS", os.environ.get("MSC_SEED", "42")).split(",")]
SEED      = SEEDS[0]   # split selection is seed-independent; kept for the split helper
SW_MATRIX = os.environ.get("MSC_SW_MATRIX", "sw_feature_matrix_v2.csv")
ICU_MATRIX = os.environ.get("MSC_ICU_MATRIX", "icu_feature_matrix_v2.csv")
SPLIT_DIR = os.environ.get("MSC_SPLIT_DIR", str(BASE/"data"/"splits"/"old"))
TAG       = os.environ.get("MSC_TAG", "")
COHORTS   = [c.strip().lower() for c in os.environ.get("MSC_COHORTS", "icu,sw").split(",")]
np.random.seed(SEED)

CATEGORICAL = ['age_band','gender','insurance','ethnicity','marital_status',
               'admission_type','bmi_category','admission_location']
PROCESS     = ['admission_location','accumulated_duration_hrs','icu_duration_hrs',
               'sw_duration_hrs','num_icu_stays','num_sw_stays','total_events']
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

IDS         = ['hadm_id','subject_id']

SPLIT_ID = {} 

def cohort_frames(cohort):
    lab = f"{cohort}_left_shift"
    base = pd.read_csv(MAT/(ICU_MATRIX if cohort=="icu" else SW_MATRIX))
    bow  = pd.read_csv(MAT/f"{cohort}_feature_matrix_v3_bow.csv")
    bow  = bow.drop(columns=[lab]).merge(base[['hadm_id',lab]], on='hadm_id', how='inner')
    return base, bow, lab

def split_idx(df, cohort):
    _sp = _validated_split(SPLIT_DIR, cohort)
    SPLIT_ID[cohort] = _sp["split_id"] 
    hid = df['hadm_id'].values
    tr = np.where(np.isin(hid, np.array(_sp["train"])))[0]
    te = np.where(np.isin(hid, np.array(_sp["test"])))[0]
    assert len(set(tr) & set(te)) == 0, f"{cohort}: train/test index overlap"
    return tr, te

def prep(df, cols, lab, cohort):
    itr, ite = split_idx(df, cohort)
    X = pd.get_dummies(df[cols], columns=[c for c in CATEGORICAL if c in cols], drop_first=False)
    Xtr = X.iloc[itr].astype(np.float32).values; Xte = X.iloc[ite].astype(np.float32).values
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), df[lab].values[itr].astype(int), df[lab].values[ite].astype(int)

def models(seed):
    return [("LR",  LogisticRegression(class_weight='balanced', random_state=seed, max_iter=1000), False),
            ("RF",  RandomForestClassifier(class_weight='balanced', random_state=seed, n_jobs=-1), False),
            ("GB",  GradientBoostingClassifier(random_state=seed), True),
            ("KNN", KNeighborsClassifier(n_jobs=-1), False)]

rows = []
for cohort in COHORTS:
    base, bow, lab = cohort_frames(cohort)
    feats = [c for c in base.columns if c not in IDS + [lab]]
    conds = {"No Process":   (base, [c for c in feats if c not in PROCESS]),
             "With Process": (base, feats),
             "With BoW":     (bow,  [c for c in bow.columns if c not in IDS + [lab]])}
    for cname, (df, cols) in conds.items():
        Xtr, Xte, ytr, yte = prep(df, cols, lab, cohort)
        for seed in SEEDS:
          for mname, mdl, needs_w in models(seed):
              t0 = time.perf_counter()
              mdl.fit(Xtr, ytr, sample_weight=compute_sample_weight('balanced', ytr)) if needs_w else mdl.fit(Xtr, ytr)
              ttrain = time.perf_counter() - t0
              t0 = time.perf_counter(); p = mdl.predict_proba(Xte)[:, 1]; ttest = time.perf_counter() - t0
              pr = (p >= .5).astype(int)
              rows.append(dict(Model=mname, Cohort=cohort.upper(), Condition=cname,
                  Accuracy=round(accuracy_score(yte,pr),4), AUROC=round(roc_auc_score(yte,p),4),
                  AUPRC=round(average_precision_score(yte,p),4),
                  Precision=round(precision_score(yte,pr,zero_division=0),4),
                  Recall=round(recall_score(yte,pr),4), F1=round(f1_score(yte,pr),4),
                  Train_Time_s=round(ttrain,3), Test_Time_s=round(ttest,3),
                  n_features=Xtr.shape[1], pos_rate=round(float(ytr.mean()),4), Seed=seed,
                  split_id=SPLIT_ID[cohort]))
              print(rows[-1], flush=True)
out = RES/f"ml_results{TAG}.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nSaved {out}  ({len(rows)} experiments)")
