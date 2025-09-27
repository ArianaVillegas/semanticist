#!/bin/bash
#SBATCH --job-name=slotformer_clevr
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2 # Using 2 GPUs
#SBATCH --time=24:00:00 # 24-hour time limit
#SBATCH --output=logs/train_clevr_%j.out
#SBATCH --error=logs/train_clevr_%j.err

echo "=== Submitting SlotFormer Training on CLEVR ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"

# Create logs and checkpoints directories
mkdir -p logs
mkdir -p checkpoints_clevr

# Environment setup
module load miniconda/3.0
eval "$(conda shell.bash hook)"
source activate /home/avillegas/miniconda3/envs/semanticist

# Validate environment
echo "=== Environment Validation ==="
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Change to project directory
cd /home/avillegas/semanticist

# Check for CLEVR dataset
CLEVR_PATH="./datasets/clevr/CLEVR_v1.0"
if [ ! -d "$CLEVR_PATH/images/train" ]; then
    echo "⚠️ CLEVR dataset not found!"
    echo "Please run 'bash scripts/download_clevr.sh' on the login node first."
    exit 1
fi

# Run the training script
echo "🚀 Launching 100-epoch training on CLEVR..."
python train_clevr.py

if [ $? -eq 0 ]; then
    echo "✅ CLEVR training job completed successfully."
else
    echo "❌ Training job failed. Check logs/train_clevr_${SLURM_JOB_ID}.err"
fi

echo "=== Training Complete ==="
echo "Finished: $(date)"
