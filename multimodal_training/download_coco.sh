#!/bin/bash
# Download COCO Captions dataset
# Total size: ~20GB (13GB images + 1GB val + annotations)
# Time: ~30 minutes on fast connection

set -e  # Exit on error

# Configuration
DOWNLOAD_DIR="/home/avillegas/semanticist/datasets/coco"

echo "============================================================"
echo "COCO CAPTIONS DATASET DOWNLOADER"
echo "============================================================"
echo ""
echo "This will download:"
echo "  - Training images (118K images, 13GB)"
echo "  - Validation images (5K images, 1GB)"
echo "  - Annotations (241MB)"
echo ""
echo "Total size: ~20GB"
echo "Time: ~30 minutes"
echo ""
echo "Download location: $DOWNLOAD_DIR"
echo ""
read -p "Continue? (y/n): " confirm

if [ "$confirm" != "y" ]; then
    echo "Cancelled"
    exit 0
fi

# Create directory
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

echo ""
echo "============================================================"
echo "Step 1: Downloading Training Images (13GB)..."
echo "============================================================"
if [ -f "train2017.zip" ]; then
    echo "✅ Already downloaded, skipping..."
else
    wget http://images.cocodataset.org/zips/train2017.zip
fi

echo ""
echo "============================================================"
echo "Step 2: Downloading Validation Images (1GB)..."
echo "============================================================"
if [ -f "val2017.zip" ]; then
    echo "✅ Already downloaded, skipping..."
else
    wget http://images.cocodataset.org/zips/val2017.zip
fi

echo ""
echo "============================================================"
echo "Step 3: Downloading Annotations (241MB)..."
echo "============================================================"
if [ -f "annotations_trainval2017.zip" ]; then
    echo "✅ Already downloaded, skipping..."
else
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
fi

echo ""
echo "============================================================"
echo "Step 4: Extracting files..."
echo "============================================================"

if [ -d "train2017" ]; then
    echo "✅ train2017/ already exists, skipping..."
else
    echo "Extracting train2017.zip..."
    unzip -q train2017.zip
fi

if [ -d "val2017" ]; then
    echo "✅ val2017/ already exists, skipping..."
else
    echo "Extracting val2017.zip..."
    unzip -q val2017.zip
fi

if [ -d "annotations" ]; then
    echo "✅ annotations/ already exists, skipping..."
else
    echo "Extracting annotations..."
    unzip -q annotations_trainval2017.zip
fi

echo ""
echo "============================================================"
echo "Step 5: Verifying dataset..."
echo "============================================================"

TRAIN_COUNT=$(ls train2017/*.jpg 2>/dev/null | wc -l)
VAL_COUNT=$(ls val2017/*.jpg 2>/dev/null | wc -l)

echo "Training images: $TRAIN_COUNT (expected: 118,287)"
echo "Validation images: $VAL_COUNT (expected: 5,000)"

if [ -f "annotations/captions_train2017.json" ]; then
    echo "✅ Training captions found"
else
    echo "❌ Training captions missing!"
    exit 1
fi

if [ -f "annotations/captions_val2017.json" ]; then
    echo "✅ Validation captions found"
else
    echo "❌ Validation captions missing!"
    exit 1
fi

echo ""
echo "============================================================"
echo "✅ COCO DATASET READY!"
echo "============================================================"
echo ""
echo "Dataset location: $DOWNLOAD_DIR"
echo "Structure:"
echo "  $DOWNLOAD_DIR/"
echo "  ├── train2017/       ($TRAIN_COUNT images)"
echo "  ├── val2017/         ($VAL_COUNT images)"
echo "  └── annotations/"
echo "      ├── captions_train2017.json"
echo "      └── captions_val2017.json"
echo ""
echo "Next steps:"
echo "1. Test dataset: cd /home/avillegas/semanticist/multimodal_training && python coco_dataset.py"
echo "2. Start training: ./coco_quick_start.sh"
echo ""

# Optional: Clean up zip files to save space
read -p "Delete zip files to save space? (y/n): " cleanup
if [ "$cleanup" = "y" ]; then
    echo "Removing zip files..."
    rm -f train2017.zip val2017.zip annotations_trainval2017.zip
    echo "✅ Cleaned up!"
fi

echo ""
echo "============================================================"
echo "DONE!"
echo "============================================================"
