#!/bin/bash
# Quick start script for COCO training
# This runs all steps automatically

set -e  # Exit on error

echo "============================================================"
echo "SLOT-COCA COCO TRAINING - QUICK START"
echo "============================================================"

# Configuration
COCO_DIR="/home/avillegas/semanticist/datasets/coco"
CHECKPOINT_DIR="checkpoints/slot_coca_coco"

# Step 1: Check if COCO is downloaded
echo ""
echo "Step 1: Checking COCO dataset..."
if [ ! -d "$COCO_DIR/train2017" ]; then
    echo "❌ COCO dataset not found at $COCO_DIR"
    echo ""
    echo "Please download COCO first:"
    echo "  cd /home/avillegas/semanticist/datasets"
    echo "  mkdir -p coco && cd coco"
    echo "  wget http://images.cocodataset.org/zips/train2017.zip"
    echo "  wget http://images.cocodataset.org/zips/val2017.zip"
    echo "  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    echo "  unzip train2017.zip"
    echo "  unzip val2017.zip"
    echo "  unzip annotations_trainval2017.zip"
    exit 1
fi

echo "✅ COCO dataset found"
echo "   Train: $(ls $COCO_DIR/train2017 | wc -l) images"
echo "   Val: $(ls $COCO_DIR/val2017 | wc -l) images"

# Step 2: Check dependencies
echo ""
echo "Step 2: Checking dependencies..."
python -c "from pycocotools.coco import COCO" 2>/dev/null || {
    echo "❌ pycocotools not installed"
    echo "Installing..."
    pip install pycocotools
}
echo "✅ Dependencies ready"

# Step 3: Test dataset loading
echo ""
echo "Step 3: Testing COCO dataset..."
python coco_dataset.py || {
    echo "❌ Dataset test failed"
    exit 1
}

# Step 4: Ask user which training mode
echo ""
echo "============================================================"
echo "Choose training mode:"
echo "============================================================"
echo "1. Quick test (100 samples, 2 epochs, ~5 minutes)"
echo "2. Small proof-of-concept (10K samples, 10 epochs, ~2 hours)"
echo "3. Full training (591K samples, 30 epochs, ~15 hours CPU)"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Running QUICK TEST..."
        python train_multimodal.py \
            --use_coco \
            --data_dir "$COCO_DIR" \
            --num_samples 100 \
            --batch_size 16 \
            --epochs 2 \
            --device cpu \
            --checkpoint_dir "${CHECKPOINT_DIR}_test" \
            --log_dir "logs/coco_test"
        ;;
    2)
        echo ""
        echo "Running SMALL PROOF-OF-CONCEPT..."
        python train_multimodal.py \
            --use_coco \
            --data_dir "$COCO_DIR" \
            --num_samples 10000 \
            --batch_size 64 \
            --epochs 10 \
            --device cuda \
            --checkpoint_dir "${CHECKPOINT_DIR}_small" \
            --log_dir "logs/coco_small"
        ;;
    3)
        echo ""
        echo "Running FULL TRAINING..."
        echo "This will take ~15 hours on CPU or ~4 hours on GPU"
        read -p "Continue? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            python train_multimodal.py \
                --use_coco \
                --data_dir "$COCO_DIR" \
                --batch_size 128 \
                --epochs 30 \
                --device cuda \
                --checkpoint_dir "$CHECKPOINT_DIR" \
                --log_dir "logs/coco_full"
        else
            echo "Cancelled"
            exit 0
        fi
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "✅ TRAINING COMPLETE!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Evaluate with: cd evaluation && python eval_retrieval.py --use_coco ..."
echo "2. Generate captions: python debug_model.py --checkpoint $CHECKPOINT_DIR/best_model.pt"
echo "3. Visualize results: python generate_samples.py"
echo ""
