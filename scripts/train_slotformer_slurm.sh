#!/bin/bash
#SBATCH --job-name=slotformer_train
#SBATCH --partition=gpu
#SBATCH --nodelist=ag001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/slotformer_train_%j.out
#SBATCH --error=logs/slotformer_train_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Load modules (adjust based on your cluster setup)
# module load cuda/11.8
# module load python/3.9

# Activate conda environment
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate /home/avillegas/miniconda3/envs/semanticist

# Set environment variables for optimal A100 performance
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=16

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Print system info
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"

# Run training
echo "Starting SlotFormer training..."
python train.py

echo "Job completed at: $(date)"
