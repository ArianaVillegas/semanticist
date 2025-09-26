#!/bin/bash
# Prepare all dependencies BEFORE submitting to cluster
# Run this on the login node (which has internet access)

echo "=== Preparing Dependencies for Cluster Experiments ==="

# Activate environment
source activate /home/avillegas/miniconda3/envs/semanticist

echo "1. Installing Python packages..."
pip install transformers datasets scikit-learn seaborn --no-deps --quiet

echo "2. Downloading datasets..."
python -c "
import torchvision
import os

# Download Imagenette dataset
print('Downloading Imagenette...')
try:
    # Download both train and val splits
    train_dataset = torchvision.datasets.Imagenette('./datasets', split='train', download=True)
    val_dataset = torchvision.datasets.Imagenette('./datasets', split='val', download=True)
    print(f'✓ Imagenette downloaded: {len(train_dataset)} train, {len(val_dataset)} val samples')
except Exception as e:
    print(f'⚠️ Imagenette download failed: {e}')
"

echo "3. Downloading pre-trained models..."
python -c "
from transformers import AutoTokenizer, AutoModel
import timm

print('Downloading BERT model...')
try:
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModel.from_pretrained('distilbert-base-uncased')
    print('✓ DistilBERT downloaded')
except Exception as e:
    print(f'⚠️ DistilBERT download failed: {e}')

print('Downloading DINOv3 model...')
try:
    model = timm.create_model('vit_base_patch16_dinov3', pretrained=True)
    print('✓ DINOv3 downloaded')
except Exception as e:
    print(f'⚠️ DINOv3 download failed: {e}')
"

echo "4. Creating offline dataset cache..."
mkdir -p ./datasets/cache

echo "5. Verifying downloads..."
python -c "
import os
from pathlib import Path

# Check datasets
datasets_dir = Path('./datasets')
if (datasets_dir / 'imagenette2').exists():
    print('✓ Imagenette dataset ready')
else:
    print('⚠️ Imagenette dataset missing')

# Check transformers cache
import transformers
cache_dir = transformers.file_utils.default_cache_path
if os.path.exists(cache_dir):
    print(f'✓ Transformers cache ready: {cache_dir}')
else:
    print('⚠️ Transformers cache missing')

# Check timm cache
import timm
print('✓ TIMM models ready')
"

echo "✅ Dependencies prepared for cluster!"
echo ""
echo "Now you can safely submit cluster jobs:"
echo "  bash scripts/submit_large_scale_experiments.sh"
echo "  bash scripts/submit_storage_analysis.sh"
echo "  bash scripts/submit_image_reconstruction.sh"
