#!/bin/bash
#SBATCH --job-name=slotformer_storage_analysis
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --output=logs/storage_analysis_%j.out
#SBATCH --error=logs/storage_analysis_%j.err

echo "=== SlotFormer Storage Analysis & High-Quality Reconstruction ==="
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
export OMP_NUM_THREADS=16

# Change to project directory
cd /home/avillegas/semanticist

echo "=== Running High-Quality Storage Analysis ==="

# Find the latest trained model
latest_model=$(ls -t model*.ckpt 2>/dev/null | head -n1)

if [ -n "$latest_model" ]; then
    echo "Using model: $latest_model"
    model_arg="--model $latest_model"
else
    echo "No trained models found, using default"
    model_arg=""
fi

# Run high-quality storage analysis
echo "Creating high-quality reconstructions with storage analysis..."
python high_quality_reconstruction_viz.py $model_arg --num_images 5

echo "=== Storage Analysis Complete ==="
echo "Finished: $(date)"

# Copy results to timestamped directory
timestamp=$(date +%Y%m%d_%H%M%S)
results_dir="storage_analysis_${timestamp}"
mkdir -p "$results_dir"

# Copy all visualization results
if [ -d "storage_analysis_viz" ]; then
    cp -r storage_analysis_viz/* "$results_dir/"
    echo "Results copied to: $results_dir"
fi

echo "Generated files:"
ls -la "$results_dir"

echo "=== Summary ==="
echo "This analysis shows:"
echo "  - Original RGB images (150 KB)"
echo "  - DINOv3 features (588 KB)"
echo "  - SlotFormer representations (3-384 KB depending on slot count)"
echo "  - High-quality image reconstructions from features"
echo "  - Storage vs Quality trade-off analysis"
echo "  - Compression ratios for different slot counts"
echo "  - Progressive visual improvement with more slots"
