#!/bin/bash
#SBATCH --job-name=slotformer_a100
#SBATCH --partition=gpu
#SBATCH --nodelist=ag001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/slotformer_a100_%j.out
#SBATCH --error=logs/slotformer_a100_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@domain.com

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints

echo "=== SlotFormer Training on A100 ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "GPUs: $SLURM_GPUS_ON_NODE"

# Environment setup
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate /home/avillegas/miniconda3/envs/semanticist

# Validate environment activation
echo "=== Environment Validation ==="
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
if [ "$CONDA_DEFAULT_ENV" != "semanticist" ]; then
    echo "ERROR: Conda environment not activated properly!"
    echo "Expected: semanticist, Got: $CONDA_DEFAULT_ENV"
    exit 1
fi

# Check required packages
python -c "import torch, torchvision, timm, lightning; print('✓ All packages available')" || {
    echo "ERROR: Required packages not found!"
    exit 1
}

# A100-optimized environment variables
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024,expandable_segments:True
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NCCL_DEBUG=INFO
export TORCH_CUDNN_V8_API_ENABLED=1

# Navigate to project
cd $SLURM_SUBMIT_DIR

# System diagnostics
echo "=== System Info ==="
nvidia-smi
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "cuDNN: $(python -c 'import torch; print(torch.backends.cudnn.version())')"

# Run optimized training
echo "=== Starting Training ==="
python train_a100_optimized.py

echo "=== Job Completed: $(date) ==="
