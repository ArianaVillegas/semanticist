#!/usr/bin/env python3
"""
SlotFormer training with DINOv3 model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import lightning as L
from tqdm.auto import tqdm
from dinov3_model import create_dinov3_model

# Training Configuration
torch.set_float32_matmul_precision("high")
NUM_SLOTS = 128
TRANSFORMER_LAYERS = 3
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
EPOCHS = 50
PRECISION = "bf16-mixed"

class SlotFormerDINOv3(nn.Module):
    """SlotFormer using DINOv3 model."""
    
    def __init__(self, num_slots: int, num_layers: int, model_size: str = "large"):
        super().__init__()
        self.num_slots = num_slots
        
        # Use custom DINOv3 model
        self.encoder = create_dinov3_model(model_size, pretrained=True).eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.embed_dim = self.encoder.embed_dim
        self.num_patches = self.encoder.patch_embed.num_patches

        # Slot queries
        self.slot_queries = nn.Parameter(torch.randn(1, self.num_slots, self.embed_dim))
        nn.init.normal_(self.slot_queries, std=0.02)

        # Positional embeddings
        if self.encoder.pos_embed is not None:
            m = self.encoder.num_prefix_tokens
            pos_embed_init = self.encoder.pos_embed[:, m:].clone()
        else:
            pos_embed_init = torch.empty(1, self.num_patches, self.embed_dim)
            nn.init.normal_(pos_embed_init, std=0.02)

        self.decoder_pos_embed = nn.Parameter(pos_embed_init)

        # Transformer layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=self.embed_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.generator = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.reconstructor = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Causal mask
        self.register_buffer(
            "tgt_mask",
            nn.Transformer.generate_square_subsequent_mask(self.num_slots),
            persistent=False,
        )

        # NULL tokens
        self.null_slots = nn.Parameter(torch.zeros(1, self.num_slots, self.embed_dim))
        nn.init.normal_(self.null_slots, std=0.02)

    def forward(self, x):
        B, K = x.shape[0], self.num_slots
        
        # Extract features
        with torch.no_grad():
            patch_tokens = self.encoder.forward_features(x)
            patch_tokens = patch_tokens[:, self.encoder.num_prefix_tokens:]

        # Generate slots
        slots = self.generator(
            tgt=self.slot_queries.repeat(B, 1, 1),
            memory=patch_tokens,
            tgt_mask=self.tgt_mask.to(patch_tokens.device),
            tgt_is_causal=True,
        )

        # Random masking
        arange = torch.arange(K, device=slots.device)
        m = torch.randint(1, K + 1, (B,), device=slots.device)
        keep = arange.unsqueeze(0) < m.unsqueeze(1)

        keep_3d = keep.unsqueeze(-1)
        null = self.null_slots.expand(B, -1, -1).type_as(slots)
        masked_slots = torch.where(keep_3d, slots, null)

        # Reconstruct
        reconstructed_patches = self.reconstructor(
            tgt=self.decoder_pos_embed.repeat(B, 1, 1),
            memory=masked_slots,
        )

        # Loss
        loss = F.mse_loss(reconstructed_patches, patch_tokens)
        return loss

def train():
    """Train SlotFormer with DINOv3."""
    
    fabric = L.Fabric(accelerator="auto", precision=PRECISION)
    fabric.launch()

    # Test DINOv3 model first
    print("Testing DINOv3 model...")
    try:
        test_model = create_dinov3_model("large", pretrained=True)
        print(f"✓ DINOv3 model loaded successfully")
        print(f"  Embed dim: {test_model.embed_dim}")
    except Exception as e:
        print(f"✗ DINOv3 model failed: {e}")
        print("Falling back to timm DINOv2...")
        return

    model = SlotFormerDINOv3(NUM_SLOTS, TRANSFORMER_LAYERS, "large")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model, optimizer = fabric.setup(model, optimizer)

    # Data loading
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])
    
    data = torchvision.datasets.Imagenette(
        "./datasets", split="train", transform=transform, download=True
    )

    dataloader = torch.utils.data.DataLoader(
        data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True
    )
    dataloader = fabric.setup_dataloaders(dataloader)

    # Training loop
    fabric.print("Starting SlotFormer training with DINOv3...")
    model.train()
    
    for epoch in range(EPOCHS):
        progress = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch+1:02d}",
            leave=True,
        )
        for i, (images, _) in progress:
            optimizer.zero_grad()
            loss = model(images)
            fabric.backward(loss)
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")

    fabric.print("Training finished")
    fabric.save(f"model-dinov3-{epoch:02d}.ckpt", {"model": model})

if __name__ == "__main__":
    train()
