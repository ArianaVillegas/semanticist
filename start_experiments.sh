#!/bin/bash
# Start SlotFormer experiments

echo "=== SlotFormer Experiment Suite ==="

# Check if training is running
echo "Checking training status..."
squeue --me | grep slotform && echo "✓ Training job is running" || echo "⚠️  No training job found"

# Download models and data if needed
echo -e "\n1. Ensuring models and data are downloaded..."
if [ ! -d "./datasets/imagenette2" ]; then
    echo "Downloading Imagenette dataset..."
    python scripts/download_models.py
else
    echo "✓ Dataset already available"
fi

# Test baseline models
echo -e "\n2. Testing baseline models..."
python baseline_models.py

# Run quick validation to test framework
echo -e "\n3. Running quick validation test..."
python run_experiments.py --mode quick

# Start comprehensive evaluation (will use untrained model first)
echo -e "\n4. Starting comprehensive evaluation..."
python comprehensive_evaluation.py

# Start monitoring for trained models
echo -e "\n5. Starting experiment monitor..."
echo "This will continuously monitor for new checkpoints and evaluate them."
echo "Press Ctrl+C to stop monitoring."

python run_experiments.py --mode monitor
