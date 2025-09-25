#!/usr/bin/env python3
"""
A100-Optimized SlotFormer Training
Optimized for dual A100 40GB GPUs with maximum performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
import lightning as L
from tqdm.auto import tqdm
import os

# A100-Optimized Hyperparameters
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# Model Configuration
NUM_SLOTS = 256  # Increased for A100 memory
TRANSFORMER_LAYERS = 6  # Deeper model for A100
MODELS = {
    "dinov2": "vit_base_patch14_dinov2",
    "dinov2_large": "vit_large_patch14_dinov2",
    "dino": "vit_base_patch16_224.dino",
}
ENCODER_NAME = MODELS["dinov2"]  # Use available model

# A100-Optimized Training Parameters
BATCH_SIZE = 512  # Large batch for A100 40GB
LEARNING_RATE = 1e-3  # Higher LR for large batch
EPOCHS = 100
PRECISION = "bf16-mixed"  # A100 native bfloat16
COMPILE = True  # Enable torch.compile for A100
GRADIENT_CLIP = 1.0

# Data and I/O
NUM_WORKERS = 32  # Match CPU count
PIN_MEMORY = True
PERSISTENT_WORKERS = True

class SlotFormerA100(nn.Module):
    """A100-optimized SlotFormer with enhanced architecture."""
    
    def __init__(self, num_slots: int, num_layers: int, encoder_name: str):
        super().__init__()
        self.num_slots = num_slots
        
        # Frozen ViT encoder
        self.encoder = timm.create_model(encoder_name, pretrained=True).eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.embed_dim = self.encoder.embed_dim
        self.num_patches = self.encoder.patch_embed.num_patches

        # Enhanced slot queries with better initialization
        self.slot_queries = nn.Parameter(torch.randn(1, self.num_slots, self.embed_dim))
        nn.init.xavier_uniform_(self.slot_queries)

        # Positional embeddings
        if self.encoder.pos_embed is not None:
            m = self.encoder.num_prefix_tokens
            pos_embed_init = self.encoder.pos_embed[:, m:].clone()
        else:
            pos_embed_init = torch.empty(1, self.num_patches, self.embed_dim)
            nn.init.normal_(pos_embed_init, std=0.02)

        self.decoder_pos_embed = nn.Parameter(pos_embed_init)

        # Enhanced transformer layers with better configuration
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=16,  # More attention heads for A100
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,  # Pre-norm for better training
        )
        
        self.generator = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.reconstructor = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Causal mask
        self.register_buffer(
            "tgt_mask",
            nn.Transformer.generate_square_subsequent_mask(self.num_slots),
            persistent=False,
        )

        # Enhanced NULL tokens
        self.null_slots = nn.Parameter(torch.zeros(1, self.num_slots, self.embed_dim))
        nn.init.normal_(self.null_slots, std=0.02)

    def forward(self, x):
        B, K = x.shape[0], self.num_slots
        
        # Extract patch tokens
        with torch.no_grad():
            patch_tokens = self.encoder.forward_features(x)
            patch_tokens = patch_tokens[:, self.encoder.num_prefix_tokens:]

        # Generate slots with causal masking
        slots = self.generator(
            tgt=self.slot_queries.repeat(B, 1, 1),
            memory=patch_tokens,
            tgt_mask=self.tgt_mask.to(patch_tokens.device),
            tgt_is_causal=True,
        )

        # Random masking for training
        arange = torch.arange(K, device=slots.device)
        m = torch.randint(1, K + 1, (B,), device=slots.device)
        keep = arange.unsqueeze(0) < m.unsqueeze(1)

        # Apply masking
        keep_3d = keep.unsqueeze(-1)
        null = self.null_slots.expand(B, -1, -1).type_as(slots)
        masked_slots = torch.where(keep_3d, slots, null)

        # Reconstruct patches
        reconstructed_patches = self.reconstructor(
            tgt=self.decoder_pos_embed.repeat(B, 1, 1),
            memory=masked_slots,
        )

        # Compute loss
        loss = F.mse_loss(reconstructed_patches, patch_tokens)
        return loss, slots, reconstructed_patches, patch_tokens

def train():
    """A100-optimized training loop."""
    
    # Lightning Fabric setup for A100
    fabric = L.Fabric(
        accelerator="cuda",
        devices=1,  # Use single GPU for now
        precision=PRECISION,
        strategy="auto"
    )
    fabric.launch()

    # Model setup
    model = SlotFormerA100(NUM_SLOTS, TRANSFORMER_LAYERS, ENCODER_NAME)
    
    # A100-optimized optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    
    # Compile model for A100 optimization
    if COMPILE:
        model = torch.compile(model, mode="max-autotune")
    
    model, optimizer = fabric.setup(model, optimizer)

    # A100-optimized data loading
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    dataset = torchvision.datasets.Imagenette(
        "./datasets", split="train", transform=transform, download=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        drop_last=True,
    )
    dataloader = fabric.setup_dataloaders(dataloader)

    # Training loop
    fabric.print(f"Starting training on {fabric.device}")
    fabric.print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    fabric.print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}")
    
    model.train()
    global_step = 0
    
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        progress = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch+1:03d}/{EPOCHS}",
            leave=True,
        )
        
        for batch_idx, (images, _) in progress:
            optimizer.zero_grad()
            
            # Forward pass
            loss, slots, recon, targets = model(images)
            
            # Backward pass with gradient clipping
            fabric.backward(loss)
            if GRADIENT_CLIP > 0:
                fabric.clip_gradients(model, optimizer, max_norm=GRADIENT_CLIP)
            
            optimizer.step()
            
            # Logging
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            progress.set_postfix({
                'loss': f"{loss.item():.6f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                'step': global_step
            })
            
            # Log every 50 steps
            if global_step % 50 == 0:
                fabric.print(f"Step {global_step}: Loss = {loss.item():.6f}")
        
        # End of epoch
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        fabric.print(f"Epoch {epoch+1} completed: Avg Loss = {avg_loss:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"checkpoints/slotformer_a100_epoch_{epoch+1:03d}.ckpt"
            fabric.save(checkpoint_path, {"model": model, "optimizer": optimizer, "epoch": epoch})
            fabric.print(f"Checkpoint saved: {checkpoint_path}")

    # Final save
    final_path = f"checkpoints/slotformer_a100_final.ckpt"
    fabric.save(final_path, {"model": model, "optimizer": optimizer, "epoch": EPOCHS})
    fabric.print(f"Training completed! Final model saved: {final_path}")

if __name__ == "__main__":
    train()
