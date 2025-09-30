#!/bin/bash
#
# Quick Start Script for Slot-CoCa Multimodal Training
# This script sets up the environment and starts proof-of-concept training
#

set -e

echo "============================================================"
echo "SLOT-COCA MULTIMODAL TRAINING - QUICK START"
echo "============================================================"

# 1. Install dependencies
echo -e "\n📦 Installing dependencies..."
pip install -q transformers sentence-transformers pycocotools
pip install -q git+https://github.com/openai/CLIP.git

# 2. Check for dataset
echo -e "\n📁 Checking dataset..."
if [ ! -d "./data/imagenet_captions" ]; then
    echo "⚠️  ImageNet-Captions not found at ./data/imagenet_captions"
    echo ""
    echo "Please download the dataset:"
    echo "  1. Download ImageNet from: https://www.image-net.org/download.php"
    echo "  2. Download captions from: https://github.com/mlfoundations/imagenet-captions"
    echo ""
    echo "Or we can create synthetic captions from class names for testing."
    read -p "Continue with synthetic captions? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 3. Create directories
echo -e "\n📂 Creating directories..."
mkdir -p logs/slot_coca
mkdir -p checkpoints/slot_coca
mkdir -p data/imagenet_captions

# 4. Test dataset loading
echo -e "\n🧪 Testing dataset loading..."
python -c "from imagenet_captions_dataset import ImageNetCaptionsDataset; print('✅ Dataset module loaded successfully')"

# 5. Check GPUs
echo -e "\n🎮 Checking GPUs..."
python -c "import torch; print(f'✅ {torch.cuda.device_count()} GPU(s) available')"

# 6. Start training (proof-of-concept)
echo -e "\n🚀 Starting training..."
echo "Configuration:"
echo "  - Subset: 100K image-caption pairs"
echo "  - Batch size: 128 per GPU (256 total)"
echo "  - Epochs: 30"
echo "  - GPUs: 2x A100"
echo ""

read -p "Start training? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python train_multimodal.py \
        --data_dir ./data/imagenet_captions \
        --subset_size 100000 \
        --batch_size 128 \
        --epochs 30 \
        --world_size 2 \
        --lr 1e-4 \
        --lambda_recon 1.0 \
        --lambda_contrast 1.0 \
        --lambda_caption 1.0 \
        --log_dir ./logs/slot_coca \
        --checkpoint_dir ./checkpoints/slot_coca \
        --log_interval 100 \
        --save_interval 5
fi

echo -e "\n✅ Setup complete!"
echo ""
echo "Monitor training with:"
echo "  tensorboard --logdir ./logs/slot_coca"
echo ""
echo "Evaluate after training:"
echo "  # Zero-shot classification"
echo "  python evaluation/eval_zero_shot.py --model_path ./checkpoints/slot_coca/best_model.pt --data_dir ./data/imagenet/val"
echo ""
echo "  # Image-text retrieval"
echo "  python evaluation/eval_retrieval.py --model_path ./checkpoints/slot_coca/best_model.pt --data_dir ./data/imagenet_captions"
echo ""
echo "  # Image captioning"
echo "  python evaluation/eval_captioning.py --model_path ./checkpoints/slot_coca/best_model.pt --data_dir ./data/imagenet_captions"
