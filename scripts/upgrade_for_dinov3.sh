#!/bin/bash
# Script to upgrade timm for DINOv3 support

echo "=== Upgrading timm for DINOv3 Support ==="

# Check current version
echo "Current timm version:"
python -c "import timm; print(timm.__version__)"

# Upgrade timm
echo "Upgrading timm..."
pip install --upgrade timm>=0.9.0

# Check new version
echo "New timm version:"
python -c "import timm; print(timm.__version__)"

# Test DINOv3 availability
echo "Testing DINOv3 models..."
python -c "
import timm
models = timm.list_models()
dinov3_models = [m for m in models if 'dinov3' in m.lower()]
if dinov3_models:
    print('✓ DINOv3 models found:')
    for model in sorted(dinov3_models):
        print(f'  - {model}')
else:
    print('✗ No DINOv3 models found')
"

# Test specific model creation
echo "Testing vit_base_patch16_dinov3..."
python -c "
import timm
try:
    model = timm.create_model('vit_base_patch16_dinov3', pretrained=False)
    print('✓ vit_base_patch16_dinov3 works!')
except Exception as e:
    print(f'✗ vit_base_patch16_dinov3 failed: {e}')
"
