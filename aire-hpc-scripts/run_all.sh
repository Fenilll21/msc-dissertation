#!/bin/bash
#SBATCH --job-name=msc_all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/msc_all_%A_%a.log

module load miniforge
source activate msc_project
export PYTHONUNBUFFERED=1
MSC_ROOT="${MSC_ROOT:-/users/dtjs0367/msc_project}"
export MSC_ROOT
cd "$MSC_ROOT"

SRC=python; [ -d scripts ] && SRC=scripts  

MODELS=(train_rnn train_lstm train_gru train_bilstm train_bigru train_cnn train_cnn_k23 train_cnn_lstm)
VARIANTS=(old corr corrpacu collapse lat restricted)
SEEDS=(42 1 2 3 4)

M=${MODELS[$(( SLURM_ARRAY_TASK_ID % 8 ))]}
V=${VARIANTS[$(( SLURM_ARRAY_TASK_ID / 8 ))]}

ICU_MAT="icu_feature_matrix_v2.csv"
COHORTS="ICU,SW"
case $V in
  old)      SW_MAT="sw_feature_matrix_v2.csv" ;;
  corr)     SW_MAT="sw_feature_matrix_v2_corr.csv";     COHORTS="SW" ;;
  corrpacu) SW_MAT="sw_feature_matrix_v2_corrpacu.csv"; COHORTS="SW" ;;
  collapse) SW_MAT="sw_feature_matrix_v2_collapse.csv"; COHORTS="SW" ;;
  lat)      SW_MAT="sw_feature_matrix_v2_lat.csv"; ICU_MAT="icu_feature_matrix_v2_lat.csv" ;;
  restricted) SW_MAT="sw_feature_matrix_restricted.csv"; ICU_MAT="icu_feature_matrix_restricted.csv" ;;
  *) echo "FATAL: unknown variant '$V' (array index $SLURM_ARRAY_TASK_ID)"; exit 1 ;;
esac

if [ "$V" = "restricted" ]; then
  SPLIT="$MSC_ROOT/data/splits/restricted"
else
  SPLIT="$MSC_ROOT/data/splits/canonical"
fi

for f in "data/matrices/$ICU_MAT" "data/matrices/$SW_MAT" \
         "$SPLIT/icu_canonical_split.json" "$SPLIT/sw_canonical_split.json" "$SRC/$M.py"; do
  [ -f "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done
grep -q MSC_ICU_MATRIX "$SRC/$M.py" || { echo "FATAL: $SRC/$M.py is an OLD unpatched copy"; exit 1; }

COMMIT=$(git -C "$MSC_ROOT" rev-parse --short HEAD 2>/dev/null || echo NA)
echo "=== $M | variant=$V | cohorts=$COHORTS | ICU=$ICU_MAT | SW=$SW_MAT | split=$(basename "$SPLIT") | commit $COMMIT ==="
for S in "${SEEDS[@]}"; do
  echo "--- seed $S ---"
  MSC_SEED=$S MSC_ICU_MATRIX=$ICU_MAT MSC_SW_MATRIX=$SW_MAT \
  MSC_SPLIT_DIR=$SPLIT MSC_COHORTS=$COHORTS MSC_TAG="_${V}_s${S}" \
  python -u "$SRC/$M.py" || echo "FAILED: $M $V seed $S"
done
echo "=== done $M $V ==="
