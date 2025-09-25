#!/usr/bin/env python3
"""
Baseline models for SlotFormer comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

class RandomBaseline(nn.Module):
    """Random reconstruction baseline."""
    
    def __init__(self, num_slots=128, embed_dim=768, num_patches=196):
        super().__init__()
        self.num_slots = num_slots
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        
    def forward(self, x, num_slots_used=None):
        B = x.shape[0]
        if num_slots_used is None:
            num_slots_used = self.num_slots
            
        # Return random reconstructions
        return torch.randn(B, self.num_patches, self.embed_dim, device=x.device)

class FixedOrderBaseline(nn.Module):
    """Baseline that uses slots in fixed order (no causal learning)."""
    
    def __init__(self, num_slots=128, encoder_name="vit_base_patch16_dinov3"):
        super().__init__()
        self.num_slots = num_slots
        
        # Same encoder as SlotFormer
        self.encoder = timm.create_model(encoder_name, pretrained=True).eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.embed_dim = self.encoder.embed_dim
        self.num_patches = self.encoder.patch_embed.num_patches
        
        # Fixed slot embeddings (no causal ordering)
        self.fixed_slots = nn.Parameter(torch.randn(1, self.num_slots, self.embed_dim))
        nn.init.normal_(self.fixed_slots, std=0.02)
        
        # Simple decoder
        self.decoder = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(self, x, num_slots_used=None):
        B = x.shape[0]
        if num_slots_used is None:
            num_slots_used = self.num_slots
            
        # Use first N slots (no causal ordering)
        slots = self.fixed_slots[:, :num_slots_used].repeat(B, 1, 1)
        
        # Simple reconstruction: average slot features
        avg_slot = slots.mean(dim=1, keepdim=True)  # [B, 1, embed_dim]
        reconstruction = self.decoder(avg_slot).repeat(1, self.num_patches, 1)
        
        return reconstruction

class NoOrderingBaseline(nn.Module):
    """Baseline without causal ordering - uses all slots equally."""
    
    def __init__(self, num_slots=128, encoder_name="vit_base_patch16_dinov3"):
        super().__init__()
        self.num_slots = num_slots
        
        # Same encoder
        self.encoder = timm.create_model(encoder_name, pretrained=True).eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.embed_dim = self.encoder.embed_dim
        self.num_patches = self.encoder.patch_embed.num_patches
        
        # Slot queries (but no causal mask)
        self.slot_queries = nn.Parameter(torch.randn(1, self.num_slots, self.embed_dim))
        nn.init.normal_(self.slot_queries, std=0.02)
        
        # Transformer without causal masking
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=self.embed_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.generator = nn.TransformerDecoder(decoder_layer, num_layers=3)
        self.reconstructor = nn.TransformerDecoder(decoder_layer, num_layers=3)
        
        # Positional embeddings
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, self.embed_dim))
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        
    def forward(self, x, num_slots_used=None):
        B = x.shape[0]
        if num_slots_used is None:
            num_slots_used = self.num_slots
            
        # Extract features
        with torch.no_grad():
            patch_tokens = self.encoder.forward_features(x)
            patch_tokens = patch_tokens[:, self.encoder.num_prefix_tokens:]
        
        # Generate slots (NO CAUSAL MASK - key difference)
        slots = self.generator(
            tgt=self.slot_queries[:, :num_slots_used].repeat(B, 1, 1),
            memory=patch_tokens,
            # No tgt_mask = no causal ordering
        )
        
        # Reconstruct
        reconstruction = self.reconstructor(
            tgt=self.decoder_pos_embed.repeat(B, 1, 1),
            memory=slots,
        )
        
        return reconstruction

class SlotFormerAblation(nn.Module):
    """SlotFormer with different configurations for ablation studies."""
    
    def __init__(self, num_slots=128, num_layers=3, encoder_name="vit_base_patch16_dinov3", 
                 use_causal_mask=True, use_random_masking=True):
        super().__init__()
        self.num_slots = num_slots
        self.use_causal_mask = use_causal_mask
        self.use_random_masking = use_random_masking
        
        # Encoder
        self.encoder = timm.create_model(encoder_name, pretrained=True).eval()
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
        if self.use_causal_mask:
            self.register_buffer(
                "tgt_mask",
                nn.Transformer.generate_square_subsequent_mask(self.num_slots),
                persistent=False,
            )
        
        # NULL tokens for masking
        if self.use_random_masking:
            self.null_slots = nn.Parameter(torch.zeros(1, self.num_slots, self.embed_dim))
            nn.init.normal_(self.null_slots, std=0.02)
    
    def forward(self, x, num_slots_used=None):
        B = x.shape[0]
        if num_slots_used is None:
            num_slots_used = self.num_slots
            
        # Extract features
        with torch.no_grad():
            patch_tokens = self.encoder.forward_features(x)
            patch_tokens = patch_tokens[:, self.encoder.num_prefix_tokens:]
        
        # Generate slots
        tgt_mask = None
        if self.use_causal_mask:
            tgt_mask = self.tgt_mask[:num_slots_used, :num_slots_used].to(x.device)
            
        slots = self.generator(
            tgt=self.slot_queries[:, :num_slots_used].repeat(B, 1, 1),
            memory=patch_tokens,
            tgt_mask=tgt_mask,
            tgt_is_causal=self.use_causal_mask,
        )
        
        # Random masking (if enabled)
        if self.use_random_masking:
            arange = torch.arange(num_slots_used, device=x.device)
            m = torch.randint(1, num_slots_used + 1, (B,), device=x.device)
            keep = arange.unsqueeze(0) < m.unsqueeze(1)
            
            keep_3d = keep.unsqueeze(-1)
            null = self.null_slots[:, :num_slots_used].expand(B, -1, -1).type_as(slots)
            slots = torch.where(keep_3d, slots, null)
        
        # Reconstruct
        reconstruction = self.reconstructor(
            tgt=self.decoder_pos_embed.repeat(B, 1, 1),
            memory=slots,
        )
        
        return reconstruction

def create_baseline_models():
    """Create all baseline models for comparison."""
    
    baselines = {
        'random': RandomBaseline(),
        'fixed_order': FixedOrderBaseline(),
        'no_ordering': NoOrderingBaseline(),
        'no_causal_mask': SlotFormerAblation(use_causal_mask=False),
        'no_random_masking': SlotFormerAblation(use_random_masking=False),
        'no_causal_no_masking': SlotFormerAblation(use_causal_mask=False, use_random_masking=False),
        'fewer_layers': SlotFormerAblation(num_layers=1),
        'more_layers': SlotFormerAblation(num_layers=6),
    }
    
    return baselines

def test_baselines():
    """Test all baseline models."""
    
    print("=== Testing Baseline Models ===")
    
    baselines = create_baseline_models()
    
    # Test input
    x = torch.randn(2, 3, 224, 224)
    
    for name, model in baselines.items():
        try:
            model.eval()
            with torch.no_grad():
                output = model(x, num_slots_used=32)
                print(f"✓ {name}: {output.shape}")
        except Exception as e:
            print(f"✗ {name}: {e}")

if __name__ == "__main__":
    test_baselines()
