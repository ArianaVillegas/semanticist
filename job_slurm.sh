#!/bin/bash
#SBATCH --job-name=slotformer_balanced
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slotformer_balanced_%j.out
#SBATCH --error=logs/slotformer_balanced_%j.err
#SBATCH --account=<YOUR_ACCOUNT>

set -euo pipefail

module purge
module load cuda/12.1

source "$HOME/.bashrc"
conda activate semanticist

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TRANSFORMERS_VERBOSE=0

srun python train.py \
    --dataset coco \
    --data-root /path/to/datasets \
    --batch-size 32 \
    --num-slots 64 \
    --text-slots 4 \
    --epochs 8 \
    --precision bf16-mixed \
    --importance-weight 0.22 \
    --importance-margin 0.04 \
    --importance-warmup-epochs 2 \
    --importance-prior-weight 0.12 \
    --importance-prior-decay 0.18 \
    --importance-temperature 0.05 \
    --importance-prior-target 1.2 \
    --progressive-text-weight 0.25 \
    --progressive-max-prefix 28 \
    --progressive-decay 0.18 \
    --curriculum-min-fraction 0.35 \
    --curriculum-warmup-epochs 4 \
    --pca-warmstart-batches 4 \
    --checkpoint-interval 1
