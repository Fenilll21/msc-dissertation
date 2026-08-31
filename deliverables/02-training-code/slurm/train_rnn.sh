#!/bin/bash
#SBATCH --job-name=ds_rnn
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/users/dtjs0367/msc_project/logs/ds_rnn_%j.log

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

module load miniforge
conda activate msc_project

cd ~/msc_project
python scripts/train_rnn.py

echo "Job finished at $(date)"
