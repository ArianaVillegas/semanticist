#!/bin/bash
#SBATCH --job-name=slotformer_fixed_viz
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/fixed_visualization_%j.out
#SBATCH --error=logs/fixed_visualization_%j.err

echo "=== SlotFormer Fixed Visualization (Cluster) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"

# Create logs directory
mkdir -p logs

# Environment setup (same as training)
module load miniconda/3.0
eval "$(conda shell.bash hook)"
source activate /home/avillegas/miniconda3/envs/semanticist

# Validate environment
echo "=== Environment Validation ==="
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "GPU available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Check for required packages
echo "Checking sklearn..."
python -c "import sklearn; print('sklearn version:', sklearn.__version__)" || {
    echo "Installing scikit-learn..."
    pip install scikit-learn
}

# Ensure we're using conda python
export PATH="$CONDA_PREFIX/bin:$PATH"

# Set up for visualization
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16

# Change to project directory
cd /home/avillegas/semanticist

echo "=== Running Fixed SlotFormer Visualizations ==="

# Find the latest trained model
latest_model=$(ls -t model*.ckpt 2>/dev/null | head -n1)

if [ -n "$latest_model" ]; then
    echo "Using model: $latest_model"
    model_arg="--model $latest_model"
else
    echo "No trained models found, using default"
    model_arg=""
fi

# Run fixed visualization
echo "1. Running comprehensive fixed analysis..."
python fixed_visualize_slots.py $model_arg --num_images 8

# Also run the cluster-optimized version
echo "2. Running cluster-optimized analysis..."
python cluster_fixed_visualize.py $model_arg --num_images 10

echo "=== Fixed Visualization Complete ==="
echo "Finished: $(date)"

# Copy results to timestamped directory
timestamp=$(date +%Y%m%d_%H%M%S)
results_dir="fixed_visualizations_${timestamp}"
mkdir -p "$results_dir"

# Copy all visualization results
if [ -d "fixed_slot_visualizations" ]; then
    cp -r fixed_slot_visualizations/* "$results_dir/"
    echo "Results copied to: $results_dir"
fi

if [ -d "cluster_fixed_visualizations" ]; then
    cp -r cluster_fixed_visualizations/* "$results_dir/"
    echo "Cluster results also copied to: $results_dir"
fi

echo "Generated files:"
ls -la "$results_dir"

echo "=== Summary ==="
echo "Fixed visualizations show:"
echo "  - Proper feature space reconstruction (768D -> PCA -> RGB)"
echo "  - Feature similarity maps (cosine similarity)"
echo "  - Error maps showing reconstruction quality"
echo "  - True representation of SlotFormer's hierarchical learning"
