#!/usr/bin/env python3
"""
Extended training script for SlotFormer on ImageNet-1k.

This script is designed for long-duration, multi-GPU training to achieve SOTA feature representations.
"""

import torch
import torch.nn.functional as F
import torchvision
import lightning as L
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from pathlib import Path

# Assuming train.py is in the same directory or accessible
from train import SlotFormer

# --- Hyperparameters ---
BATCH_SIZE = 256  # Per device
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200
PRECISION = "16-mixed"
NUM_SLOTS = 128
TRANSFORMER_LAYERS = 8
ENCODER_NAME = "vit_base_patch16_dinov3"
IMAGENET_PATH = "/path/to/your/imagenet"  # IMPORTANT: Update this path

# --- Main Training Function ---
def train_imagenet():
    """Main function to run long-duration ImageNet training."""
    
    # 1. Setup Fabric for multi-GPU training
    fabric = L.Fabric(accelerator="auto", devices="auto", strategy="ddp", precision=PRECISION)
    fabric.launch()

    fabric.print(f"=== Starting SlotFormer Training on ImageNet ===")
    fabric.print(f"Epochs: {NUM_EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}")

    # 2. Create Model and Optimizer
    model = SlotFormer(NUM_SLOTS, TRANSFORMER_LAYERS, ENCODER_NAME)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Setup model and optimizer with Fabric
    model, optimizer = fabric.setup(model, optimizer)

    # 3. Create DataLoaders
    # Define transforms for ImageNet
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # Check if ImageNet path is valid
    if not os.path.isdir(IMAGENET_PATH) or not os.path.exists(os.path.join(IMAGENET_PATH, 'train')):
        fabric.print(f"\n⚠️ ImageNet path not found at '{IMAGENET_PATH}'!")
        fabric.print("Please update the IMAGENET_PATH variable in this script.")
        fabric.print("Using synthetic data for demonstration purposes.")
        # Create a dummy dataset for testing the script logic
        train_data = [(torch.randn(3, 224, 224), i % 1000) for i in range(10000)]
        val_data = [(torch.randn(3, 224, 224), i % 1000) for i in range(1000)]
    else:
        fabric.print(f"Loading ImageNet from: {IMAGENET_PATH}")
        train_data = torchvision.datasets.ImageFolder(
            root=os.path.join(IMAGENET_PATH, 'train'), transform=transform
        )
        val_data = torchvision.datasets.ImageFolder(
            root=os.path.join(IMAGENET_PATH, 'val'), transform=transform
        )

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)

    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    # 4. Learning Rate Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 5. Training Loop
    best_val_loss = float('inf')
    checkpoint_dir = Path("./checkpoints_imagenet")
    fabric.barrier()
    if fabric.global_rank == 0:
        checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        # --- Training Step ---
        model.train()
        total_loss = 0
        for i, (images, _) in enumerate(train_dataloader):
            # Randomly mask slots for training
            B, _, _, _ = images.shape
            num_slots_to_use = torch.randint(1, NUM_SLOTS + 1, (1,)).item()

            # Forward pass
            reconstructed_patches, original_patches = model(images, num_slots_to_use)
            loss = F.mse_loss(reconstructed_patches, original_patches)
            
            # Backward pass
            optimizer.zero_grad()
            fabric.backward(loss)
            optimizer.step()

            total_loss += loss.item()

            if i % 100 == 0 and fabric.global_rank == 0:
                fabric.print(f"Epoch {epoch}/{NUM_EPOCHS} | Batch {i}/{len(train_dataloader)} | Loss: {loss.item():.4f}")
        
        avg_train_loss = total_loss / len(train_dataloader)
        scheduler.step()

        # --- Validation Step ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, _ in val_dataloader:
                reconstructed_patches, original_patches = model(images, NUM_SLOTS) # Use all slots for validation
                val_loss = F.mse_loss(reconstructed_patches, original_patches)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)

        fabric.print(f"Epoch {epoch} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # --- Checkpointing ---
        if avg_val_loss < best_val_loss and fabric.global_rank == 0:
            best_val_loss = avg_val_loss
            model_path = checkpoint_dir / "best_model.ckpt"
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
            fabric.save(model_path, state)
            fabric.print(f"Saved new best model to {model_path} with val loss {avg_val_loss:.4f}")

if __name__ == "__main__":
    train_imagenet()
