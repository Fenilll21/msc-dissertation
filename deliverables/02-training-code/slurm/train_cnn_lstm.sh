#!/bin/bash
#SBATCH --job-name=cnn_lstm_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cnn_lstm_train_%j.log

module load miniforge
source activate msc_project

export PYTHONUNBUFFERED=1

python -u scripts/train_cnn_lstm.py