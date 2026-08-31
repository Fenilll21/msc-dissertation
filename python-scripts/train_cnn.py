import pathlib
import copy, time, random, warnings
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score)
warnings.filterwarnings("ignore", category=UserWarning)

import os, json as _json, hashlib as _hashlib

MSC_ROOT = os.environ.get("MSC_ROOT", "/users/dtjs0367/msc_project")
SEED = int(os.environ.get("MSC_SEED", 42))
SW_MATRIX_FILE = os.environ.get("MSC_SW_MATRIX", "sw_feature_matrix_v2.csv")
ICU_MATRIX_FILE = os.environ.get("MSC_ICU_MATRIX", "icu_feature_matrix_v2.csv")
SPLIT_DIR = os.environ.get("MSC_SPLIT_DIR", "")
TAG = os.environ.get("MSC_TAG", "")
COHORTS = [c.strip().upper() for c in os.environ.get("MSC_COHORTS", "ICU,SW").split(",")]

def _load_split(cohort):
    """Load THE canonical partition. Fails loudly on a stale or ungrouped file."""
    if not SPLIT_DIR:
        raise RuntimeError("MSC_SPLIT_DIR is unset. Refusing to redraw a split at run time.")
    p = pathlib.Path(SPLIT_DIR) / f"{cohort.lower()}_canonical_split.json"
    if not p.exists():
        raise FileNotFoundError(p)
    s = _json.loads(p.read_text())
    assert s.get("grouped_on") == "subject_id", f"{p}: not grouped on subject_id (stale split file?)"
    assert s.get("stratified_on") == "lat", f"{p}: not stratified on the primary label"
    assert "split_id" in s, f"{p}: no split_id (pre-fix split file)"
    tr, va, te = (sorted(int(x) for x in s[k]) for k in ("train", "val", "test"))
    h = _hashlib.sha256()
    for part in (tr, va, te):
        h.update(b"|"); h.update(",".join(map(str, part)).encode())
    assert h.hexdigest()[:16] == s["split_id"], f"{p}: contents do not match split_id"
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te), \
        f"{p}: partitions overlap"
    return {"train": np.array(tr, dtype=np.int64), "val": np.array(va, dtype=np.int64),
            "test": np.array(te, dtype=np.int64), "split_id": s["split_id"]}

SPLIT_ID = {}
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[setup] Using device: {DEVICE}", flush=True)
if DEVICE.type == "cuda":
    print(f"[setup] GPU: {torch.cuda.get_device_name(0)}", flush=True)

BASE    = Path(MSC_ROOT) / "data"
SEQ_DIR = BASE / "sequences"; MAT_DIR = BASE / "matrices"
RESULTS = Path(MSC_ROOT) / "results"; RESULTS.mkdir(parents=True, exist_ok=True)

ICU_LABEL, SW_LABEL = "icu_left_shift", "sw_left_shift"
IDENTIFIER_COLS = ['hadm_id', 'subject_id']
CATEGORICAL_COLS = ['age_band','gender','insurance','ethnicity','marital_status',
                    'admission_type','bmi_category','admission_location']
TEST_SIZE = 0.20
VOCAB_SIZE, EMBED_DIM = 12, 16
NUM_FILTERS = 64
KERNEL_SIZES = (2, 3, 4)
DROPOUT, BATCH_SIZE, MAX_EPOCHS, PATIENCE, LR = 0.3, 256, 100, 10, 1e-3


def to_post_padded(seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Pre-padded (zeros at start) -> post-padded (zeros at end)."""
    B, L = seq.shape
    out = torch.zeros_like(seq)
    idx = torch.arange(L, device=seq.device).unsqueeze(0)
    src_mask = idx >= (L - lengths).unsqueeze(1)
    dst_mask = idx <  lengths.unsqueeze(1)
    out[dst_mask] = seq[src_mask]
    return out


def load_cohort(cohort: str):
    if cohort == "ICU":
        m = pd.read_csv(MAT_DIR/ICU_MATRIX_FILE)
        s = np.load(SEQ_DIR/"icu_sequences.npy"); o = np.load(SEQ_DIR/"icu_hadm_id_order.npy")
        return m, s, o, ICU_LABEL
    if cohort == "SW":
        m = pd.read_csv(MAT_DIR/SW_MATRIX_FILE)
        s = np.load(SEQ_DIR/"sw_sequences.npy"); o = np.load(SEQ_DIR/"sw_hadm_id_order.npy")
        return m, s, o, SW_LABEL
    raise ValueError(cohort)


def align_matrix_and_sequences(matrix, sequences, hadm_order, cohort):
    matrix_ids = set(matrix["hadm_id"])
    keep = np.array([h in matrix_ids for h in hadm_order])
    kept_ids = hadm_order[keep]; seq_al = sequences[keep]
    mat_al = matrix.set_index("hadm_id").reindex(kept_ids).reset_index()
    assert not mat_al["hadm_id"].isna().any()
    assert len(mat_al) == len(seq_al)
    assert (mat_al["hadm_id"].values == kept_ids).all()
    print(f"[{cohort}] Aligned: matrix {len(matrix)} & seq {len(sequences)} "
          f"-> {len(mat_al)} common.", flush=True)
    return mat_al, seq_al


def get_tabular_feature_cols(matrix, label_col):
    excl = set(IDENTIFIER_COLS + [label_col])
    cols = [c for c in matrix.columns if c not in excl]
    assert len(cols) == 39, f"Expected 39 features, got {len(cols)}"
    return cols


def preprocess_tabular(matrix, feature_cols, label_col, cohort, condition):
    X = pd.get_dummies(matrix[feature_cols],
                       columns=[c for c in CATEGORICAL_COLS if c in feature_cols],
                       drop_first=False)
    y = matrix[label_col].values.astype(np.float32)
    idx = np.arange(len(matrix))
    sp = _load_split(cohort); SPLIT_ID[cohort] = sp["split_id"]
    hid = matrix["hadm_id"].values
    idx_tr = idx[np.isin(hid, sp["train"])]
    idx_va = idx[np.isin(hid, sp["val"])]
    idx_te = idx[np.isin(hid, sp["test"])]
    assert len(idx_tr) + len(idx_va) + len(idx_te) == len(matrix), (
        f"{cohort}: partition sizes do not cover the cohort")
    assert len(set(idx_tr) & set(idx_va)) == 0 and len(set(idx_tr) & set(idx_te)) == 0 \
        and len(set(idx_va) & set(idx_te)) == 0, f"{cohort}: index overlap"
    print(f"[{cohort} | {condition}] canonical split {sp['split_id']}: "
          f"train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)}", flush=True)
    Xtr = X.iloc[idx_tr].values.astype(np.float32)
    Xva = X.iloc[idx_va].values.astype(np.float32)
    Xte = X.iloc[idx_te].values.astype(np.float32)
    ytr, yva, yte = y[idx_tr], y[idx_va], y[idx_te]
    # scaler fitted on TRAIN ONLY (previously train+val, since val was carved from train)
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xva = sc.transform(Xva); Xte = sc.transform(Xte)
    print(f"[{cohort} | {condition}] Train {Xtr.shape} pos {ytr.mean():.3f} | "
          f"Val {Xva.shape} pos {yva.mean():.3f} | Test {Xte.shape} pos {yte.mean():.3f}", flush=True)
    return Xtr, Xva, Xte, ytr, yva, yte, idx_tr, idx_va, idx_te


class SequenceDataset(Dataset):
    def __init__(self, seq, y, X=None):
        self.seq = torch.from_numpy(seq).long()
        self.y = torch.from_numpy(y).float()
        self.X = torch.from_numpy(X).float() if X is not None else None
    def __len__(self): return len(self.seq)
    def __getitem__(self, i):
        if self.X is None: return self.seq[i], self.y[i]
        return self.seq[i], self.X[i], self.y[i]


class TextCNN(nn.Module):
    def __init__(self, n_tabular=0):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(EMBED_DIM, NUM_FILTERS, kernel_size=k)
                                    for k in KERNEL_SIZES])
        head_in = NUM_FILTERS * len(KERNEL_SIZES) + n_tabular
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(head_in, 64),
                                  nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(64, 1))
        self.n_tabular = n_tabular

    def forward(self, seq, tabular=None):
        lengths = (seq != 0).sum(dim=1)
        seq = to_post_padded(seq, lengths)
        max_k = max(KERNEL_SIZES)
        if seq.size(1) < max_k:
            pad = torch.zeros(seq.size(0), max_k - seq.size(1),
                              dtype=seq.dtype, device=seq.device)
            seq = torch.cat([seq, pad], dim=1)
        x = self.embed(seq).transpose(1, 2)            # (B,E,L)
        NEG = -1e9
        pooled = []
        for conv, k in zip(self.convs, KERNEL_SIZES):
            h = torch.relu(conv(x))                    # (B,F,L_out)
            L_out = h.size(2)
            valid = (lengths - k + 1).clamp(min=0)
            pos = torch.arange(L_out, device=h.device).unsqueeze(0)
            keep = pos < valid.unsqueeze(1)
            h = h.masked_fill(~keep.unsqueeze(1), NEG)
            pk = torch.max(h, dim=2).values
            pk = torch.where((valid == 0).unsqueeze(1), torch.zeros_like(pk), pk)
            pooled.append(pk)
        out = torch.cat(pooled, dim=1)
        if self.n_tabular > 0:
            out = torch.cat([out, tabular], dim=1)
        return self.head(out).squeeze(1)


@torch.no_grad()
def evaluate(model, loader, has_tab):
    model.eval(); P, Y = [], []
    for b in loader:
        if has_tab:
            s, t, y = b; s,t,y = s.to(DEVICE),t.to(DEVICE),y.to(DEVICE); lo = model(s,t)
        else:
            s, y = b; s,y = s.to(DEVICE),y.to(DEVICE); lo = model(s)
        P.append(torch.sigmoid(lo).cpu().numpy()); Y.append(y.cpu().numpy())
    P, Y = np.concatenate(P), np.concatenate(Y); pr = (P>=0.5).astype(int)
    return {'Accuracy':round(accuracy_score(Y,pr),4),'AUROC':round(roc_auc_score(Y,P),4),
            'AUPRC':round(average_precision_score(Y,P),4),
            'Precision':round(precision_score(Y,pr,zero_division=0),4),
            'Recall':round(recall_score(Y,pr,zero_division=0),4),
            'F1':round(f1_score(Y,pr,zero_division=0),4)}


def train_model(tr, va, te, n_tab, pos_weight, cohort, condition):
    has = n_tab > 0
    model = TextCNN(n_tab).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))
    trl = DataLoader(tr, BATCH_SIZE, shuffle=True); val = DataLoader(va, BATCH_SIZE)
    tel = DataLoader(te, BATCH_SIZE)
    best, best_state, no_imp, t0, last = -1.0, None, 0, time.perf_counter(), 0
    for ep in range(MAX_EPOCHS):
        last = ep; model.train()
        for b in trl:
            if has: s,t,y=b; s,t,y=s.to(DEVICE),t.to(DEVICE),y.to(DEVICE); lo=model(s,t)
            else:   s,y=b;   s,y=s.to(DEVICE),y.to(DEVICE); lo=model(s)
            loss=lf(lo,y); opt.zero_grad(); loss.backward(); opt.step()
        va_auroc = evaluate(model, val, has)['AUROC']
        if va_auroc > best: best, best_state, no_imp = va_auroc, copy.deepcopy(model.state_dict()), 0
        else: no_imp += 1
        if ep % 5 == 0 or ep < 3:
            print(f"    Epoch {ep:3d} | Val AUROC {va_auroc:.4f} | Best {best:.4f} | NoImp {no_imp}", flush=True)
        if no_imp >= PATIENCE:
            print(f"    Early stopping at epoch {ep}", flush=True); break
    model.load_state_dict(best_state)
    return evaluate(model, tel, has), last+1, round(time.perf_counter()-t0,1)


def run_experiment(cohort, condition):
    print(f"\n{'='*60}\n  CNN | {cohort} | {condition}\n{'='*60}", flush=True)
    m, s, o, lc = load_cohort(cohort)
    m, s = align_matrix_and_sequences(m, s, o, cohort)
    fc = get_tabular_feature_cols(m, lc)
    Xtr, Xva, Xte, ytr, yva, yte, itr, iva, ite = preprocess_tabular(m, fc, lc, cohort, condition)
    # validation membership comes from the canonical file; the former internal
    # train_test_split has been REMOVED (ungrouped, label-stratified, variant-specific)
    s_tr, s_va, s_te = s[itr], s[iva], s[ite]
    y_tr, y_va = ytr, yva
    pos = float(y_tr.sum()); neg = float(len(y_tr)-pos); pw = neg/max(pos,1.0)
    if condition == "Sequence Only":
        tr, va, te, nt = SequenceDataset(s_tr,y_tr), SequenceDataset(s_va,y_va), SequenceDataset(s_te,yte), 0
    else:
        tr = SequenceDataset(s_tr, y_tr, Xtr); va = SequenceDataset(s_va, y_va, Xva)
        te = SequenceDataset(s_te, yte, Xte); nt = Xtr.shape[1]
    mtr, ep, el = train_model(tr, va, te, nt, pw, cohort, condition)
    print(f"\n  RESULT: AUROC={mtr['AUROC']} | AUPRC={mtr['AUPRC']} | F1={mtr['F1']} | epochs={ep} | {el}s", flush=True)
    return {'Model':'CNN','Cohort':cohort,'Condition':condition, **mtr, 'Epochs':ep, 'Train_Time_s':el, 'split_id': SPLIT_ID[cohort]}


def masking_invariance_test(model_cls, n_tabular=0):
    model = model_cls(n_tabular=n_tabular).to(DEVICE).eval()
    cases = {
        "1-token": ([[0,1]],                 [[0,0,0,0,0,1]]),
        "2-token": ([[0,0,1,4]],             [[0,0,0,0,0,0,1,4]]),
        "3-token": ([[0,0,1,4,8]],           [[0,0,0,0,0,0,1,4,8]]),
        "5-token": ([[0,1,4,8,7,3]],         [[0,0,0,0,0,1,4,8,7,3]]),
    }
    tab = None
    if n_tabular > 0:
        g = torch.Generator(device='cpu').manual_seed(0)
        tab = torch.randn(1, n_tabular, generator=g).to(DEVICE)
    worst = 0.0
    tag = f"n_tab={n_tabular}"
    with torch.no_grad():
        for name, (a, b) in cases.items():
            oa = model(torch.tensor(a, device=DEVICE), tab)
            ob = model(torch.tensor(b, device=DEVICE), tab)
            d = (oa - ob).abs().max().item(); worst = max(worst, d)
            print(f"[invariance {tag}] {name}: max|diff|={d:.3e} "
                  f"({'PASS' if d < 1e-5 else 'FAIL'})", flush=True)
    assert worst < 1e-5, f"Masking invariance FAILED ({tag}): worst={worst:.3e}"
    return worst


def main():
    print("\n[GATE] Running masking-invariance test before any training...", flush=True)
    masking_invariance_test(TextCNN, n_tabular=0)
    masking_invariance_test(TextCNN, n_tabular=76)
    print("[GATE] Invariance passed. Proceeding to training.\n", flush=True)

    experiments = [(c, cond) for c in COHORTS
                   for cond in ("Sequence Only", "Sequence + Tabular")]
    results, t0 = [], time.perf_counter()
    for c, cond in experiments:
        results.append(run_experiment(c, cond))
    print(f"\n{'#'*60}\n# ALL CNN EXPERIMENTS COMPLETE in {time.perf_counter()-t0:.1f}s\n{'#'*60}", flush=True)
    df = pd.DataFrame(results); out = RESULTS/f"cnn_results{TAG}.csv"; df.to_csv(out, index=False)
    print(f"\nSaved: {out}\n", flush=True); print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()