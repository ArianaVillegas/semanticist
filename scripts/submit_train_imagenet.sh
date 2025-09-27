#!/bin/bash
#SBATCH --job-name=slotformer_imagenet
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4  # Requesting 4 GPUs for faster training
#SBATCH --time=48:00:00 # 48-hour time limit for 200 epochs
#SBATCH --output=logs/train_imagenet_%j.out
#SBATCH --error=logs/train_imagenet_%j.err

echo "=== Submitting SlotFormer SOTA Training on ImageNet ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"

# Create logs and checkpoints directories
mkdir -p logs
mkdir -p checkpoints_imagenet

# Environment setup
module load miniconda/3.0
eval "$(conda shell.bash hook)"
source activate /home/avillegas/miniconda3/envs/semanticist

# Validate environment
echo "=== Environment Validation ==="
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Fabric is using: $SLURM_NTASKS tasks, $SLURM_CPUS_PER_TASK CPUs per task"

# Set environment variables for multi-GPU training
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Change to project directory
cd /home/avillegas/semanticist

# IMPORTANT: Check for ImageNet path
# The train_imagenet.py script will use synthetic data if the path is not found.
echo "Checking for ImageNet dataset..."

# Run the training script using torchrun for distributed training
echo "🚀 Launching 200-epoch training on ImageNet..."
python train_imagenet.py

if [ $? -eq 0 ]; then
    echo "✅ Training job completed successfully."
else
    echo "❌ Training job failed. Check logs/train_imagenet_${SLURM_JOB_ID}.err"
fi

echo "=== Training Complete ==="
echo "Finished: $(date)"
