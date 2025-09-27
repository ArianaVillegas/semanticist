#!/bin/bash
#SBATCH --job-name=feature_space_analysis
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=feature_space_analysis/logs/analysis_%j.out
#SBATCH --error=feature_space_analysis/logs/analysis_%j.err

echo "=== SlotFormer Feature Space Analysis ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"

# Create log directory
mkdir -p feature_space_analysis/logs

# --- Environment Setup ---
# This assumes your environment is set up correctly on the cluster node.
# If you use modules or conda, add the setup commands here.
# Example for conda:
# module load miniconda/3.0
# eval "$(conda shell.bash hook)"
# source activate /path/to/your/env

# --- Validation ---
echo "Validating environment..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')"

# --- Execution ---
cd /media/ariana/exp/Wonderland/semanticist # Adjust if your project root is different

MODEL_CHECKPOINT="model-49.ckpt"

# --- Stage 1: Train Model (if necessary) ---
echo "\n=== Stage 1: Checking for Model Checkpoint ==="
if [ -f "$MODEL_CHECKPOINT" ]; then
    echo "✓ Found existing model checkpoint: $MODEL_CHECKPOINT"
else
    echo "⚠️ Model checkpoint not found. Starting pre-training..."
    echo "This will take some time."
    
    python feature_space_analysis/train_model.py
    
    if [ -f "$MODEL_CHECKPOINT" ]; then
        echo "✓ Pre-training complete. Model saved to $MODEL_CHECKPOINT"
    else
        echo "❌ Pre-training failed. Aborting analysis."
        exit 1
    fi
fi

# --- Stage 2: Run Analysis ---
echo "\n=== Stage 2: Running Feature Space Analysis ==="

MODEL_ARG="--model_path $MODEL_CHECKPOINT"

echo "\n🚀 Running analysis with command:"
echo "python run_feature_analysis.py $MODEL_ARG --num_images 30"

python run_feature_analysis.py $MODEL_ARG --num_images 30

echo "\n=== Analysis Complete ==="
echo "Finished: $(date)"

# --- Summary ---
echo "\n📊 Summary of generated files in feature_space_analysis/results:"
ls -lh feature_space_analysis/results/

echo "\nTo view results, check the 'feature_space_analysis/results' directory."
