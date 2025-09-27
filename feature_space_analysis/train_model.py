#!/usr/bin/env python3
"""
Pre-training script for SlotFormer, designed to be run as part of the
feature space analysis pipeline.

This script trains the model on ImageNette for a fixed 50 epochs and saves
the final checkpoint to the project's root directory.
"""

import torch
import torch.nn.functional as F
import torchvision
import lightning as L
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

# Add parent directory to path to import train module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import SlotFormer

# --- Configuration ---
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
NUM_EPOCHS = 50
PRECISION = "16-mixed"
NUM_SLOTS = 128
TRANSFORMER_LAYERS = 3
ENCODER_NAME = "vit_base_patch16_dinov3"
CHECKPOINT_NAME = "model-49.ckpt" # Final epoch is 49

def train_model():
    """Main function to pre-train the SlotFormer model."""
    
    # 1. Setup Fabric for single-GPU training
    fabric = L.Fabric(accelerator="auto", devices=1, precision=PRECISION)
    fabric.launch()

    fabric.print(f"=== Starting SlotFormer Pre-training for Analysis ===")
    fabric.print(f"Training for {NUM_EPOCHS} epochs on ImageNette.")

    # 2. Create Model and Optimizer
    model = SlotFormer(NUM_SLOTS, TRANSFORMER_LAYERS, ENCODER_NAME)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model, optimizer = fabric.setup(model, optimizer)

    # 3. Create DataLoaders
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        train_data = torchvision.datasets.Imagenette("./datasets", split="train", transform=transform, download=True)
    except Exception as e:
        fabric.print(f"❌ Could not download or find ImageNette dataset: {e}")
        fabric.print("Please ensure you have an internet connection or the dataset is in './datasets'.")
        return

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    # 4. Learning Rate Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 5. Training Loop
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for i, (images, _) in enumerate(train_dataloader):
            num_slots_to_use = torch.randint(1, NUM_SLOTS + 1, (1,)).item()
            
            loss = model(images, num_slots_to_use)
            
            optimizer.zero_grad()
            fabric.backward(loss)
            optimizer.step()

            total_loss += loss.item()

            if i % 50 == 0:
                fabric.print(f"Epoch {epoch}/{NUM_EPOCHS-1} | Batch {i}/{len(train_dataloader)} | Loss: {loss.item():.4f}")
        
        avg_train_loss = total_loss / len(train_dataloader)
        scheduler.step()
        fabric.print(f"Epoch {epoch} | Avg Train Loss: {avg_train_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # --- Save Final Checkpoint ---
    if fabric.global_rank == 0:
        # Save to project root directory
        save_path = Path("./") / CHECKPOINT_NAME
        state = {"model": model.state_dict()}
        fabric.save(save_path, state)
        fabric.print(f"\n✅ Training complete. Final model saved to: {save_path.resolve()}")

if __name__ == "__main__":
    train_model()
