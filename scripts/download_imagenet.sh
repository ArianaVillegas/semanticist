#!/bin/bash
# This script downloads and prepares the ImageNet 2012 dataset.
#
# INSTRUCTIONS:
# 1. Register on https://image-net.org/download.php to get a username and access key.
# 2. Run this script on a machine with at least 200GB of free space and a fast internet connection.
# 3. Enter your username and access key when prompted.

set -e

# --- Configuration ---
TARGET_DIR="./datasets/imagenet"

# --- Check for dependencies ---
if ! command -v wget &> /dev/null
then
    echo "'wget' could not be found. Please install it first." >&2
    exit 1
fi

# --- Get User Credentials ---
read -p "Enter your ImageNet username: " USERNAME
read -p "Enter your ImageNet access key: " ACCESS_KEY

if [ -z "$USERNAME" ] || [ -z "$ACCESS_KEY" ]; then
    echo "Username and access key cannot be empty." >&2
    exit 1
fi

# --- Create Target Directory ---
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "Downloading ImageNet to $(pwd)... This will take a long time."

# --- Download Files ---
TRAIN_URL="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
VAL_URL="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
DEVKIT_URL="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"

# Download training set (approx. 138 GB)
echo "Downloading training set..."
wget -c --post-data="username=$USERNAME&accesskey=$ACCESS_KEY" "$TRAIN_URL" -O ILSVRC2012_img_train.tar

# Download validation set (approx. 6.3 GB)
echo "Downloading validation set..."
wget -c --post-data="username=$USERNAME&accesskey=$ACCESS_KEY" "$VAL_URL" -O ILSVRC2012_img_val.tar

# Download devkit (for validation script)
echo "Downloading devkit..."
wget -c --post-data="username=$USERNAME&accesskey=$ACCESS_KEY" "$DEVKIT_URL" -O ILSVRC2012_devkit_t12.tar.gz


echo "Downloads complete. Now extracting files..."

# --- Extract Training Data ---
# The training data is a tar of tars. Each class is its own tar file.
echo "Extracting training data..."
mkdir -p train

# Extract the main tar file
tar -xf ILSVRC2012_img_train.tar -C train

# Go into the train directory and extract all the class-specific tar files
cd train
for f in *.tar; do
  d=`basename $f .tar`
  mkdir $d
  tar -xf $f -C $d
  rm $f
  echo "  Extracted $f"
done
cd ..

# --- Extract Validation Data ---
# The validation data is flat and needs to be moved into class-labeled subdirectories.
echo "Extracting validation data..."
mkdir -p val
tar -xf ILSVRC2012_img_val.tar -C val

# Download the helper script to organize the validation set
VAL_SCRIPT_URL="https://raw.githubusercontent.com/soumith/imagenet-torch/master/valprep.sh"
wget -c "$VAL_SCRIPT_URL" -O valprep.sh
chmod +x valprep.sh

# Run the script
echo "Organizing validation set..."
./valprep.sh

# --- Cleanup ---
rm ILSVRC2012_img_train.tar
rm ILSVRC2012_img_val.tar
rm ILSVRC2012_devkit_t12.tar.gz
rm valprep.sh

echo "✅ ImageNet dataset is ready at $(pwd)"
echo "Please update the 'IMAGENET_PATH' in 'train_imagenet.py' to: $(pwd)"
