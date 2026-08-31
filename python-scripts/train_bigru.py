import pathlib
import copy
import time
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
)

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
    p = pathlib.Path(SPLIT_DIR) / f"{cohort.lower()}_canonical_split.json"
    if not SPLIT_DIR:
        raise RuntimeError("MSC_SPLIT_DIR is unset. Refusing to redraw a split at run time.")
    if not p.exists():
        raise FileNotFoundError(p)
    s = _json.loads(p.read_text())
    # assertion 4/5: this must be the patient-grouped canonical split, not a per-variant redraw
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
    return {"train": np.array(tr, dtype=np.int64),
            "val":   np.array(va, dtype=np.int64),
            "test":  np.array(te, dtype=np.int64),
            "split_id": s["split_id"]}

SPLIT_ID = {} 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[setup] Using device: {DEVICE}", flush=True)
if DEVICE.type == "cuda":
    print(f"[setup] GPU: {torch.cuda.get_device_name(0)}", flush=True)

BASE     = Path(MSC_ROOT) / "data"
SEQ_DIR  = BASE / "sequences"
MAT_DIR  = BASE / "matrices"
RESULTS  = Path(MSC_ROOT) / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


ICU_LABEL = "icu_left_shift"
SW_LABEL  = "sw_left_shift"

IDENTIFIER_COLS: List[str] = ['hadm_id', 'subject_id']

CATEGORICAL_COLS: List[str] = [
    'age_band', 'gender', 'insurance', 'ethnicity',
    'marital_status', 'admission_type', 'bmi_category',
    'admission_location',
]

PROCESS_FEATURE_COLS: List[str] = [
    'admission_location', 'accumulated_duration_hrs',
    'icu_duration_hrs', 'sw_duration_hrs',
    'num_icu_stays', 'num_sw_stays', 'total_events',
]

TEST_SIZE: float = 0.20

VOCAB_SIZE   = 12       
EMBED_DIM    = 16
RNN_HIDDEN   = 64
DROPOUT      = 0.3
BATCH_SIZE   = 256
MAX_EPOCHS   = 100
PATIENCE     = 10
LR           = 1e-3


def load_cohort(cohort: str):
    if cohort == "ICU":
        matrix    = pd.read_csv(MAT_DIR / ICU_MATRIX_FILE)
        sequences = np.load(SEQ_DIR / "icu_sequences.npy")
        hadm_ord  = np.load(SEQ_DIR / "icu_hadm_id_order.npy")
        label_col = ICU_LABEL
    elif cohort == "SW":
        matrix    = pd.read_csv(MAT_DIR / SW_MATRIX_FILE)
        sequences = np.load(SEQ_DIR / "sw_sequences.npy")
        hadm_ord  = np.load(SEQ_DIR / "sw_hadm_id_order.npy")
        label_col = SW_LABEL
    else:
        raise ValueError(cohort)
    return matrix, sequences, hadm_ord, label_col


def align_matrix_and_sequences(matrix: pd.DataFrame, sequences: np.ndarray,
                               hadm_order: np.ndarray, cohort: str):
    matrix_ids = set(matrix["hadm_id"])
    keep_mask = np.array([hid in matrix_ids for hid in hadm_order])
    kept_ids  = hadm_order[keep_mask]
    sequences_aligned = sequences[keep_mask]

    matrix_aligned = (matrix.set_index("hadm_id")
                            .reindex(kept_ids)
                            .reset_index())

    assert not matrix_aligned["hadm_id"].isna().any(),
    assert len(matrix_aligned) == len(sequences_aligned), 
    assert (matrix_aligned["hadm_id"].values == kept_ids).all(), 

    print(f"[{cohort}] Aligned to intersection: "
          f"matrix {len(matrix)} & sequences {len(sequences)} "
          f"-> {len(matrix_aligned)} common admissions.", flush=True)
    return matrix_aligned, sequences_aligned


def get_tabular_feature_cols(matrix: pd.DataFrame, label_col: str) -> List[str]:
    """All 39 features (matches MLP notebook 'With Process' condition)."""
    excluded = set(IDENTIFIER_COLS + [label_col])
    cols = [c for c in matrix.columns if c not in excluded]
    assert len(cols) == 39, f"Expected 39 features, got {len(cols)}"
    return cols


def preprocess_tabular(matrix: pd.DataFrame, feature_cols: List[str],
                       label_col: str, cohort: str, condition: str):
    """One-hot encode categoricals, split, scale (scaler fit on train only)."""
    X_full = pd.get_dummies(
        matrix[feature_cols],
        columns=[c for c in CATEGORICAL_COLS if c in feature_cols],
        drop_first=False,
    )
    y_full = matrix[label_col].values.astype(np.float32)

    idx_all = np.arange(len(matrix))
    sp  = _load_split(cohort)
    SPLIT_ID[cohort] = sp["split_id"]
    hid = matrix["hadm_id"].values
    idx_train = idx_all[np.isin(hid, sp["train"])]
    idx_val   = idx_all[np.isin(hid, sp["val"])]
    idx_test  = idx_all[np.isin(hid, sp["test"])]

    assert len(idx_train) + len(idx_val) + len(idx_test) == len(matrix), (
        f"{cohort}: partition sizes {len(idx_train)}+{len(idx_val)}+{len(idx_test)} "
        f"!= cohort {len(matrix)}")
    assert len(set(idx_train) & set(idx_val)) == 0 and len(set(idx_train) & set(idx_test)) == 0 \
        and len(set(idx_val) & set(idx_test)) == 0, f"{cohort}: index overlap"

    print(f"[{cohort} | {condition}] canonical split {sp['split_id']}: "
          f"train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}", flush=True)

    X_train = X_full.iloc[idx_train].values.astype(np.float32)
    X_val   = X_full.iloc[idx_val].values.astype(np.float32)
    X_test  = X_full.iloc[idx_test].values.astype(np.float32)
    y_train = y_full[idx_train]
    y_val   = y_full[idx_val]
    y_test  = y_full[idx_test]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"[{cohort} | {condition}] Train: {X_train.shape} "
          f"| pos rate: {y_train.mean():.3f}", flush=True)
    print(f"[{cohort} | {condition}] Val:   {X_val.shape} "
          f"| pos rate: {y_val.mean():.3f}", flush=True)
    print(f"[{cohort} | {condition}] Test:  {X_test.shape} "
          f"| pos rate: {y_test.mean():.3f}", flush=True)
    return X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, y: np.ndarray,
                 X_tabular: np.ndarray = None):
        self.seq = torch.from_numpy(sequences).long()
        self.y   = torch.from_numpy(y).float()
        self.X   = torch.from_numpy(X_tabular).float() if X_tabular is not None else None

    def __len__(self) -> int:
        return len(self.seq)

    def __getitem__(self, i: int):
        if self.X is None:
            return self.seq[i], self.y[i]
        return self.seq[i], self.X[i], self.y[i]



class BiGRUNet(nn.Module):
    def __init__(self, n_tabular: int = 0):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.gru = nn.GRU(input_size=EMBED_DIM, hidden_size=RNN_HIDDEN,
                          batch_first=True, bidirectional=True)
        head_in = RNN_HIDDEN * 2 + n_tabular
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(head_in, 64), nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1),
        )
        self.n_tabular = n_tabular

    def forward(self, seq: torch.Tensor, tabular: torch.Tensor = None) -> torch.Tensor:
        assert seq.dim() == 2, f"seq must be (B, L), got {tuple(seq.shape)}"
        lengths = sequence_lengths(seq)
        seq = to_post_padded(seq, lengths)
        x = self.embed(seq)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)                            
        final = torch.cat([h_n[-2], h_n[-1]], dim=1)     
        if self.n_tabular > 0:
            assert tabular is not None, "tabular expected but None"
            final = torch.cat([final, tabular], dim=1)
        out = self.head(final).squeeze(1)
        assert out.dim() == 1, f"out must be (B,), got {tuple(out.shape)}"
        return out

MODEL_CLASS = BiGRUNet
MODEL_NAME = "BiGRU"


def to_post_padded(seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    B, L = seq.shape
    out = torch.zeros_like(seq)
    for i in range(B):
        n = int(lengths[i])
        out[i, :n] = seq[i, L - n:]   # move the trailing real tokens to the front
    return out


def sequence_lengths(seq: torch.Tensor) -> torch.Tensor:

    assert seq.dim() == 2, f"expected (B, L), got {tuple(seq.shape)}"
    lengths = (seq != 0).sum(dim=1)
    lengths = torch.clamp(lengths, min=1)
    return lengths.cpu()


@torch.no_grad()
def evaluate(model, loader, device, has_tabular: bool) -> Tuple[float, Dict[str, float]]:
    model.eval()
    all_probs, all_y = [], []
    for batch in loader:
        if has_tabular:
            seq, tab, y = batch
            seq, tab, y = seq.to(device), tab.to(device), y.to(device)
            logits = model(seq, tab)
        else:
            seq, y = batch
            seq, y = seq.to(device), y.to(device)
            logits = model(seq)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_y.append(y.cpu().numpy())
    probs = np.concatenate(all_probs)
    y     = np.concatenate(all_y)
    preds = (probs >= 0.5).astype(int)
    return roc_auc_score(y, probs), {
        'Accuracy' : round(accuracy_score(y, preds), 4),
        'AUROC'    : round(roc_auc_score(y, probs), 4),
        'AUPRC'    : round(average_precision_score(y, probs), 4),
        'Precision': round(precision_score(y, preds, zero_division=0), 4),
        'Recall'   : round(recall_score(y, preds, zero_division=0), 4),
        'F1'       : round(f1_score(y, preds, zero_division=0), 4),
    }

def train_model(train_ds, val_ds, test_ds, n_tabular: int, pos_weight: float,
                cohort: str, condition: str) -> Tuple[Dict[str, float], int, float]:
    has_tab = n_tabular > 0
    model = MODEL_CLASS(n_tabular=n_tabular).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_val_auroc = -1.0
    best_state = None
    no_improve = 0
    t0 = time.perf_counter()
    last_epoch = 0

    for epoch in range(MAX_EPOCHS):
        last_epoch = epoch
        model.train()
        for batch in train_loader:
            if has_tab:
                seq, tab, y = batch
                seq, tab, y = seq.to(DEVICE), tab.to(DEVICE), y.to(DEVICE)
                logits = model(seq, tab)
            else:
                seq, y = batch
                seq, y = seq.to(DEVICE), y.to(DEVICE)
                logits = model(seq)
            loss = loss_fn(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

        val_auroc, _ = evaluate(model, val_loader, DEVICE, has_tab)
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch < 3:
            print(f"    Epoch {epoch:3d} | Val AUROC: {val_auroc:.4f} "
                  f"| Best: {best_val_auroc:.4f} | No improve: {no_improve}",
                  flush=True)

        if no_improve >= PATIENCE:
            print(f"    Early stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)", flush=True)
            break

    model.load_state_dict(best_state)
    _, test_metrics = evaluate(model, test_loader, DEVICE, has_tab)
    elapsed = round(time.perf_counter() - t0, 1)
    return test_metrics, last_epoch + 1, elapsed


def run_experiment(cohort: str, condition: str) -> Dict:
    print(f"\n{'='*60}", flush=True)
    print(f"  BiGRU | {cohort} | {condition}", flush=True)
    print(f"{'='*60}", flush=True)

    matrix, sequences, hadm_order, label_col = load_cohort(cohort)
    matrix, sequences = align_matrix_and_sequences(matrix, sequences, hadm_order, cohort)

    feature_cols = get_tabular_feature_cols(matrix, label_col)
    X_train_tab, X_val_tab, X_test_tab, y_tr, y_val, y_test, idx_train, idx_val, idx_test = \
        preprocess_tabular(matrix, feature_cols, label_col, cohort, condition)
    seq_train = sequences[idx_train]
    seq_val   = sequences[idx_val]
    seq_test  = sequences[idx_test]

    pos = float(y_tr.sum())
    neg = float(len(y_tr) - pos)
    pos_weight = neg / max(pos, 1.0)

    if condition == "Sequence Only":
        train_ds = SequenceDataset(seq_train, y_tr)
        val_ds   = SequenceDataset(seq_val,   y_val)
        test_ds  = SequenceDataset(seq_test,  y_test)
        n_tab    = 0
    elif condition == "Sequence + Tabular":
        train_ds = SequenceDataset(seq_train, y_tr,   X_train_tab)
        val_ds   = SequenceDataset(seq_val,   y_val,  X_val_tab)
        test_ds  = SequenceDataset(seq_test,  y_test, X_test_tab)
        n_tab    = X_train_tab.shape[1]
    else:
        raise ValueError(condition)

    metrics, epochs, elapsed = train_model(
        train_ds, val_ds, test_ds, n_tab, pos_weight, cohort, condition
    )
    print(f"\n  RESULT: AUROC={metrics['AUROC']} | AUPRC={metrics['AUPRC']} "
          f"| F1={metrics['F1']} | epochs={epochs} | {elapsed}s", flush=True)

    return {
        'Model': MODEL_NAME,
        'Cohort': cohort,
        'Condition': condition,
        **metrics,
        'Epochs': epochs,
        'Train_Time_s': elapsed,
        'split_id': SPLIT_ID[cohort],
    }


def main():
    experiments = [(c, cond) for c in COHORTS
                   for cond in ("Sequence Only", "Sequence + Tabular")]
    results = []
    overall_start = time.perf_counter()
    for cohort, condition in experiments:
        results.append(run_experiment(cohort, condition))

    overall = time.perf_counter() - overall_start
    print(f"\n{'#'*60}", flush=True)
    print(f"# ALL BiGRU EXPERIMENTS COMPLETE in {overall:.1f}s", flush=True)
    print(f"{'#'*60}", flush=True)

    df = pd.DataFrame(results)
    out_path = RESULTS / f"bigru_results{TAG}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}", flush=True)
    print(f"\nBiGRU Results:", flush=True)
    print(df.to_string(index=False), flush=True)


def _smoke_test():
    print("[smoke] running forward-pass shape check...", flush=True)
    B, L = 4, 30
    dummy_seq = torch.randint(0, VOCAB_SIZE, (B, L)).to(DEVICE)
    # Sequence-only
    m0 = MODEL_CLASS(n_tabular=0).to(DEVICE)
    out0 = m0(dummy_seq)
    assert out0.shape == torch.Size([B]), f"seq-only out shape {out0.shape}"
    # Sequence + tabular
    n_tab = 76
    dummy_tab = torch.randn(B, n_tab).to(DEVICE)
    m1 = MODEL_CLASS(n_tabular=n_tab).to(DEVICE)
    out1 = m1(dummy_seq, dummy_tab)
    assert out1.shape == torch.Size([B]), f"seq+tab out shape {out1.shape}"
    print("[smoke] OK — both output shapes correct (B,).", flush=True)


if __name__ == "__main__":
    _smoke_test()
    main()
