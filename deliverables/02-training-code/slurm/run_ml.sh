#!/bin/bash
#SBATCH --job-name=msc_ml
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/msc_ml_%A_%a.log
# Submit: sbatch --array=0-2 jobs/run_ml.sh
module load miniforge; source activate msc_project
export PYTHONUNBUFFERED=1; cd /users/dtjs0367/msc_project
VARIANTS=(old corr corrpacu); V=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
case $V in
  old)      MAT="sw_feature_matrix_v2.csv" ;;
  corr)     MAT="sw_feature_matrix_v2_corr.csv" ;;
  corrpacu) MAT="sw_feature_matrix_v2_corrpacu.csv" ;;
esac
MSC_SW_MATRIX=$MAT MSC_SPLIT_DIR=/users/dtjs0367/msc_project/data/splits/$V \
SRC=python; [ -d scripts ] && SRC=scripts
MSC_TAG="_${V}" python -u "$SRC/train_ml.py"
