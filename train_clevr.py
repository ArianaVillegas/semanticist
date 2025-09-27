#!/usr/bin/env python3
"""
Training script for SlotFormer on the CLEVR dataset.

This script is designed to learn object-centric representations from compositional scenes,
which will be used for intervention and manipulation experiments.
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
BATCH_SIZE = 128  # Per device
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100 # CLEVR is smaller, but 100 epochs ensures good convergence
PRECISION = "16-mixed"
NUM_SLOTS = 16 # CLEVR scenes have at most 10 objects, 16 slots is sufficient
TRANSFORMER_LAYERS = 6
ENCODER_NAME = "vit_base_patch16_dinov3"
CLEVR_PATH = "./datasets/clevr/CLEVR_v1.0"  # Default path, update if needed

# --- Main Training Function ---
def train_clevr():
    """Main function to run CLEVR training."""
    
    # 1. Setup Fabric for multi-GPU training
    # Use a single GPU, no DDP strategy needed.
    fabric = L.Fabric(accelerator="auto", devices=1, precision=PRECISION)
    fabric.launch()

    fabric.print(f"=== Starting SlotFormer Training on CLEVR ===")
    fabric.print(f"Epochs: {NUM_EPOCHS}, Batch Size: {BATCH_SIZE}, Slots: {NUM_SLOTS}")

    # 2. Create Model and Optimizer
    # We use fewer slots as CLEVR has a limited number of objects per scene
    model = SlotFormer(NUM_SLOTS, TRANSFORMER_LAYERS, ENCODER_NAME)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    model, optimizer = fabric.setup(model, optimizer)

    # 3. Create DataLoaders
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)), # Resize to fit ViT
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] # Normalize for CLEVR
            ),
        ]
    )

    # Check if CLEVR path is valid
    clevr_image_path = os.path.join(CLEVR_PATH, 'images', 'train')
    if not os.path.isdir(clevr_image_path):
        fabric.print(f"\n⚠️ CLEVR path not found at '{clevr_image_path}'!")
        fabric.print("Please run 'scripts/download_clevr.sh' first or update the CLEVR_PATH.")
        return

    fabric.print(f"Loading CLEVR from: {CLEVR_PATH}")
    # The CLEVR dataset images are in a single folder, so we can use a simple custom dataset
    # or just ImageFolder if we create a dummy label structure.
    train_data = torchvision.datasets.ImageFolder(root=os.path.join(CLEVR_PATH, 'images'), transform=transform)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    # 4. Learning Rate Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 5. Training Loop
    best_loss = float('inf')
    checkpoint_dir = Path("./checkpoints_clevr")
    fabric.barrier()
    if fabric.global_rank == 0:
        checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for i, (images, _) in enumerate(train_dataloader):
            num_slots_to_use = torch.randint(1, NUM_SLOTS + 1, (1,)).item()

            # Forward pass returns the loss directly
            loss = model(images, num_slots_to_use)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            fabric.backward(loss)
            optimizer.step()

            total_loss += loss.item()

            if i % 50 == 0 and fabric.global_rank == 0:
                fabric.print(f"Epoch {epoch}/{NUM_EPOCHS} | Batch {i}/{len(train_dataloader)} | Loss: {loss.item():.4f}")
        
        avg_train_loss = total_loss / len(train_dataloader)
        scheduler.step()

        fabric.print(f"Epoch {epoch} | Avg Train Loss: {avg_train_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # --- Checkpointing ---
        if avg_train_loss < best_loss and fabric.global_rank == 0:
            best_loss = avg_train_loss
            model_path = checkpoint_dir / "best_model_clevr.ckpt"
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
            fabric.save(model_path, state)
            fabric.print(f"Saved new best model to {model_path} with train loss {avg_train_loss:.4f}")

if __name__ == "__main__":
    train_clevr()
