#!/bin/bash
#SBATCH --job-name=slotformer_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/evaluation_%j.out
#SBATCH --error=logs/evaluation_%j.err

echo "=== SlotFormer Cluster Evaluation ==="
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

# Ensure we're using conda python
export PATH="$CONDA_PREFIX/bin:$PATH"

# Set up for evaluation
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16

# Change to project directory
cd /home/avillegas/semanticist

echo "=== Running Comprehensive Evaluation ==="

# Test baseline models first
echo "1. Testing baseline models..."
python baseline_models.py

# Run comprehensive evaluation
echo "2. Running comprehensive evaluation..."
python comprehensive_evaluation.py

# Check for any trained models and evaluate them
echo "3. Looking for trained models..."
if ls model*.ckpt 1> /dev/null 2>&1; then
    echo "Found trained models, running detailed evaluation..."
    for model in model*.ckpt; do
        echo "Evaluating $model..."
        python evaluate_slotformer.py --model "$model" --device cuda
    done
else
    echo "No trained models found, evaluation used random weights"
fi

echo "=== Evaluation Complete ==="
echo "Finished: $(date)"

# Copy results to a timestamped directory
timestamp=$(date +%Y%m%d_%H%M%S)
results_dir="evaluation_results_${timestamp}"
mkdir -p "$results_dir"

# Copy all result files
cp -r results/* "$results_dir/" 2>/dev/null || echo "No results directory found"
cp -r comprehensive_results/* "$results_dir/" 2>/dev/null || echo "No comprehensive_results directory found"
cp -r quick_test_results/* "$results_dir/" 2>/dev/null || echo "No quick_test_results directory found"

echo "Results saved to: $results_dir"
ls -la "$results_dir"
