#!/bin/bash
#SBATCH --job-name=slotformer_img_recon
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/image_reconstruction_%j.out
#SBATCH --error=logs/image_reconstruction_%j.err

echo "=== SlotFormer Image Reconstruction Visualization ==="
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

echo "=== Running Image Reconstruction Visualization ==="

# Find the latest trained model
latest_model=$(ls -t model*.ckpt 2>/dev/null | head -n1)

if [ -n "$latest_model" ]; then
    echo "Using model: $latest_model"
    model_arg="--model $latest_model"
else
    echo "No trained models found, using default"
    model_arg=""
fi

# Run image reconstruction visualization
echo "Creating image reconstructions from DINOv3 features..."
python image_reconstruction_viz.py $model_arg --num_images 6

echo "=== Image Reconstruction Complete ==="
echo "Finished: $(date)"

# Copy results to timestamped directory
timestamp=$(date +%Y%m%d_%H%M%S)
results_dir="image_reconstruction_${timestamp}"
mkdir -p "$results_dir"

# Copy all visualization results
if [ -d "image_reconstruction_viz" ]; then
    cp -r image_reconstruction_viz/* "$results_dir/"
    echo "Results copied to: $results_dir"
fi

echo "Generated files:"
ls -la "$results_dir"

echo "=== Summary ==="
echo "This visualization shows:"
echo "  - Original images from Imagenette dataset"
echo "  - Ground truth reconstructions (upper bound from perfect features)"
echo "  - SlotFormer reconstructions with 1, 2, 4, 8, 16, 32, 64, 128 slots"
echo "  - Progressive improvement in visual quality"
echo "  - Quantitative loss curves showing monotonic improvement"
