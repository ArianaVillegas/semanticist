#!/bin/bash
# Setup script for Semanticist conda environment

echo "Setting up Semanticist conda environment..."

# Create conda environment
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

echo "Environment created! To activate it, run:"
echo "conda activate semanticist"

echo ""
echo "After activation, you can test the setup with:"
echo "python simple_test.py"
echo ""
echo "Or run the full test with:"
echo "python test_semanticist.py --mode both"
