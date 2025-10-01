#!/bin/bash
# Quick evaluation script for Slot-CoCa on Imagenette
# Run all three evaluations in sequence

# Configuration
MODEL_PATH="/home/avillegas/semanticist/multimodal_training/checkpoints/slot_coca_real_captions/best_model.pt"
DATA_DIR="/home/avillegas/semanticist/datasets/imagenette2"
DEVICE="cpu"  # Change to "cuda" if using GPU

echo "============================================================"
echo "SLOT-COCA EVALUATION ON IMAGENETTE"
echo "============================================================"
echo ""
echo "Model: $MODEL_PATH"
echo "Data: $DATA_DIR"
echo "Device: $DEVICE"
echo ""

cd evaluation

# 1. Zero-Shot Classification
echo ""
echo "1️⃣  ZERO-SHOT CLASSIFICATION"
echo "============================================================"
python eval_zero_shot.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --use_imagenette \
    --device "$DEVICE" \
    --batch_size 64

# 2. Image-Text Retrieval
echo ""
echo ""
echo "2️⃣  IMAGE-TEXT RETRIEVAL"
echo "============================================================"
python eval_retrieval.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --use_imagenette \
    --device "$DEVICE" \
    --num_samples 3925 \
    --batch_size 128

# 3. Image Captioning
echo ""
echo ""
echo "3️⃣  IMAGE CAPTIONING"
echo "============================================================"
python eval_captioning.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --use_imagenette \
    --device "$DEVICE" \
    --num_samples 1000 \
    --batch_size 32

echo ""
echo "============================================================"
echo "✅ ALL EVALUATIONS COMPLETE!"
echo "============================================================"
