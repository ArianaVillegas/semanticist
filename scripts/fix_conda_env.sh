#!/bin/bash
# Fix conda environment activation issues

echo "=== Conda Environment Fix ==="

# Show current environment
echo "Current environment: $CONDA_DEFAULT_ENV"
echo "Current Python: $(which python)"
echo "Expected Python: /home/avillegas/miniconda3/envs/semanticist/bin/python"

# Method 1: Proper conda activation
echo -e "\n=== Method 1: Proper Conda Activation ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate semanticist

echo "After conda activate:"
echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"

# Method 2: Force PATH
echo -e "\n=== Method 2: Force PATH ==="
export PATH="/home/avillegas/miniconda3/envs/semanticist/bin:$PATH"
echo "After PATH fix:"
echo "Python: $(which python)"

# Method 3: Direct activation
echo -e "\n=== Method 3: Direct Activation ==="
source activate /home/avillegas/miniconda3/envs/semanticist
echo "After source activate:"
echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"

# Test timm import
echo -e "\n=== Testing timm Import ==="
python -c "
import sys
print('Python executable:', sys.executable)
try:
    import timm
    print('✓ timm imported successfully')
    print('timm version:', timm.__version__)
    print('timm location:', timm.__file__)
except ImportError as e:
    print('✗ timm import failed:', e)
    print('Available packages:')
    import os
    site_packages = os.path.join(os.path.dirname(sys.executable), '..', 'lib', 'python3.13', 'site-packages')
    if os.path.exists(site_packages):
        packages = [p for p in os.listdir(site_packages) if 'timm' in p.lower()]
        print('timm-related packages:', packages)
"

echo -e "\n=== Recommended Fix ==="
echo "Add this to your ~/.bashrc:"
echo "source ~/miniconda3/etc/profile.d/conda.sh"
echo ""
echo "Then always use:"
echo "conda activate semanticist"
echo "# NOT: source activate"
