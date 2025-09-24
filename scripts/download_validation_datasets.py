#!/usr/bin/env python3
"""
Download datasets for SlotFormer validation experiments.
"""

import os
import torchvision
from pathlib import Path

def download_datasets(data_dir="./datasets"):
    """Download required datasets for validation."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading validation datasets...")
    
    # 1. ImageNette (already used in train.py)
    print("1. Downloading ImageNette...")
    try:
        train_data = torchvision.datasets.Imagenette(
            str(data_dir), split="train", download=True
        )
        val_data = torchvision.datasets.Imagenette(
            str(data_dir), split="val", download=True
        )
        print(f"   ✓ ImageNette: {len(train_data)} train, {len(val_data)} val samples")
    except Exception as e:
        print(f"   ✗ ImageNette failed: {e}")
    
    # 2. CIFAR-10
    print("2. Downloading CIFAR-10...")
    try:
        train_data = torchvision.datasets.CIFAR10(
            str(data_dir), train=True, download=True
        )
        test_data = torchvision.datasets.CIFAR10(
            str(data_dir), train=False, download=True
        )
        print(f"   ✓ CIFAR-10: {len(train_data)} train, {len(test_data)} test samples")
    except Exception as e:
        print(f"   ✗ CIFAR-10 failed: {e}")
    
    # 3. STL-10
    print("3. Downloading STL-10...")
    try:
        train_data = torchvision.datasets.STL10(
            str(data_dir), split="train", download=True
        )
        test_data = torchvision.datasets.STL10(
            str(data_dir), split="test", download=True
        )
        print(f"   ✓ STL-10: {len(train_data)} train, {len(test_data)} test samples")
    except Exception as e:
        print(f"   ✗ STL-10 failed: {e}")
    
    print(f"\nDatasets downloaded to: {data_dir}")
    return data_dir

if __name__ == "__main__":
    download_datasets()
