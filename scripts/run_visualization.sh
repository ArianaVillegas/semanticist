#!/bin/bash
#SBATCH --job-name=slotformer_viz
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/visualization_%j.out
#SBATCH --error=logs/visualization_%j.err

echo "=== SlotFormer Visualization ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"

# Create logs directory
mkdir -p logs

# Environment setup
module load miniconda/3.0
eval "$(conda shell.bash hook)"
source activate /home/avillegas/miniconda3/envs/semanticist

# Validate environment
echo "=== Environment Validation ==="
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "GPU available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Ensure we're using conda python
export PATH="$CONDA_PREFIX/bin:$PATH"

# Set up for visualization
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# Change to project directory
cd /home/avillegas/semanticist

echo "=== Running SlotFormer Visualizations ==="

# Find the latest trained model
latest_model=$(ls -t model*.ckpt 2>/dev/null | head -n1)

if [ -n "$latest_model" ]; then
    echo "Using model: $latest_model"
    python visualize_slots.py --model "$latest_model" --num_images 8
else
    echo "No trained models found, using default"
    python visualize_slots.py --num_images 8
fi

echo "=== Visualization Complete ==="
echo "Finished: $(date)"

# List generated files
echo "Generated visualization files:"
ls -la slot_visualizations/
