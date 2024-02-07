#!/bin/bash
#SBATCH --job-name={}
#SBATCH --output=slurm_output.txt
#SBATCH --error=slurm_error.txt
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=dgx[1-39]
#SBATCH --gres=gpu:1

source ~/venv/bin/activate

cd {}

python run.py "$1" "$SLURM_ARRAY_TASK_ID"
