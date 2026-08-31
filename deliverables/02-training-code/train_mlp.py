import os, json, time, copy, random
import pathlib
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score)

BASE = Path(__file__).resolve().parent.parent
MAT, RES = BASE/"data"/"matrices", BASE/"results"; RES.mkdir(exist_ok=True)
SEEDS      = [int(x) for x in os.environ.get("MSC_SEEDS", os.environ.get("MSC_SEED", "42")).split(",")]
SEED       = SEEDS[0]
ICU_MATRIX = os.environ.get("MSC_ICU_MATRIX", "icu_feature_matrix_v2.csv")
SW_MATRIX  = os.environ.get("MSC_SW_MATRIX",  "sw_feature_matrix_v2.csv")
SPLIT_DIR  = Path(os.environ.get("MSC_SPLIT_DIR", BASE/"data"/"splits"/"old"))
TAG        = os.environ.get("MSC_TAG", "")
COHORTS    = [c.strip().lower() for c in os.environ.get("MSC_COHORTS", "icu,sw").split(",")]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[setup] device={DEVICE} seed={SEED} icu={ICU_MATRIX} sw={SW_MATRIX} split={SPLIT_DIR.name}", flush=True)

SPLIT_ID = {} 

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

IDS = ['hadm_id','subject_id']
CATEGORICAL = ['age_band','gender','insurance','ethnicity','marital_status',
               'admission_type','bmi_category','admission_location']
PROCESS = ['admission_location','accumulated_duration_hrs','icu_duration_hrs',
           'sw_duration_hrs','num_icu_stays','num_sw_stays','total_events']
HIDDEN, DROPOUT, BATCH, MAX_EPOCHS, PATIENCE, LR = 64, 0.3, 256, 100, 10, 1e-3


class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def load(cohort):
    lab = f"{cohort}_left_shift"
    df = pd.read_csv(MAT/(ICU_MATRIX if cohort == "icu" else SW_MATRIX))
    assert df.shape[1] == 42, f"expected 42 cols, got {df.shape[1]}"
    cols = [c for c in df.columns if c not in IDS + [lab]]
    assert len(cols) == 39, f"expected 39 features, got {len(cols)}"
    sp = _validated_split(SPLIT_DIR, cohort)
    SPLIT_ID[cohort] = sp["split_id"]      #
    return df, lab, cols, sp


def run(cohort, condition):
    df, lab, cols, sp = load(cohort)
    use = [c for c in cols if c not in PROCESS] if condition == "No Process" else cols
    X = pd.get_dummies(df[use], columns=[c for c in CATEGORICAL if c in use], drop_first=False)
    hid = df["hadm_id"].values
    te = np.isin(hid, np.array(sp["test"])); va = np.isin(hid, np.array(sp["val"])); tr = ~(te | va)
    y = df[lab].values.astype(np.float32)
    sc = StandardScaler().fit(X[tr].values.astype(np.float32))
    Xtr, Xva, Xte = (sc.transform(X[m].values.astype(np.float32)) for m in (tr, va, te))
    ytr, yva, yte = y[tr], y[va], y[te]
    print(f"[{cohort.upper()} | {condition}] train={Xtr.shape} val={Xva.shape} test={Xte.shape} "
          f"pos={ytr.mean():.4f}", flush=True)

    g = torch.Generator(); g.manual_seed(SEED)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=BATCH, shuffle=True, generator=g, drop_last=False)
    model = MLP(Xtr.shape[1]).to(DEVICE)
    pw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32, device=DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def proba(A):
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(torch.from_numpy(A).to(DEVICE))).cpu().numpy()

    best, best_state, bad, t0 = -1.0, None, 0, time.perf_counter()
    for ep in range(MAX_EPOCHS):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        auc = roc_auc_score(yva, proba(Xva))
        if auc > best + 1e-6:
            best, best_state, bad = auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    model.load_state_dict(best_state)
    elapsed, p = time.perf_counter() - t0, proba(Xte)
    pr = (p >= 0.5).astype(int)
    return dict(Model="MLP", Cohort=cohort.upper(), Condition=condition,
                Accuracy=round(accuracy_score(yte, pr), 4), AUROC=round(roc_auc_score(yte, p), 4),
                AUPRC=round(average_precision_score(yte, p), 4),
                Precision=round(precision_score(yte, pr, zero_division=0), 4),
                Recall=round(recall_score(yte, pr), 4), F1=round(f1_score(yte, pr), 4),
                Epochs=ep + 1, Train_Time_s=round(elapsed, 1),
                n_features=Xtr.shape[1], pos_rate=round(float(ytr.mean()), 4), Seed=SEED,
                split_id=SPLIT_ID[cohort])
rows = []
for _seed in SEEDS:
    random.seed(_seed); np.random.seed(_seed)
    torch.manual_seed(_seed); torch.cuda.manual_seed_all(_seed)
    globals()["SEED"] = _seed
    for c in COHORTS:
        for cond in ["No Process", "With Process"]:
            rows.append(run(c, cond))
for r in rows: print(r, flush=True)
out = RES/f"mlp_results{TAG}.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nSaved {out}  ({len(rows)} experiments)")
