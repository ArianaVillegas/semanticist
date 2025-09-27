#!/bin/bash
# This script downloads and prepares the CLEVR dataset.
#
# INSTRUCTIONS:
# 1. Make sure you have 'gdown' installed (`pip install gdown`).
# 2. Run this script. It will download and extract the dataset (~18 GB).

set -e

# --- Configuration ---
TARGET_DIR="./datasets/clevr"
FILE_ID="1O_22A03V5nHe2s23oE6t5sT6m3A4n2c_"
ZIP_FILE="CLEVR_v1.0.zip"

# --- Check for dependencies ---
if ! command -v wget &> /dev/null
then
    echo "'wget' could not be found. Please install it first." >&2
    exit 1
fi

# --- Create Target Directory ---
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "Downloading CLEVR_v1.0 from official S3 bucket... ~18 GB"

# --- Download File ---
# Use wget to download from the official S3 bucket (more reliable)
S3_URL="https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
wget -c "$S3_URL" -O "$ZIP_FILE"

echo "Download complete. Now extracting files..."

# --- Extract Data ---
unzip "$ZIP_FILE"

# --- Cleanup ---
rm "$ZIP_FILE"

echo "✅ CLEVR dataset is ready at $(pwd)/CLEVR_v1.0"
echo "Please update the training script to use this path."
