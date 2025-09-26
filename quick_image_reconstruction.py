#!/usr/bin/env python3
"""
Quick image reconstruction visualization (can run locally).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from train import SlotFormer

class SimpleFeatureDecoder(nn.Module):
    """Simple decoder for quick testing."""
    
    def __init__(self, feature_dim=768):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16 * 16 * 3),  # 16x16 RGB patch
            nn.Tanh()
        )
        
    def forward(self, features):
        """Convert features to small RGB patches."""
        B, num_patches, _ = features.shape
        rgb_patches = self.decoder(features)  # [B, 196, 768]
        rgb_patches = rgb_patches.view(B, num_patches, 3, 16, 16)
        
        # Arrange patches into 14x14 grid, then resize to 224x224
        patches_per_side = 14
        rgb_patches = rgb_patches.view(B, patches_per_side, patches_per_side, 3, 16, 16)
        
        # Combine patches
        images = rgb_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        images = images.view(B, 3, 224, 224)
        
        return images

def quick_image_reconstruction_demo(model_path="model-49.ckpt", num_images=3):
    """Quick demo of image reconstruction from features."""
    
    print("=== Quick Image Reconstruction Demo ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load SlotFormer
    model = SlotFormer(128, 3, "vit_base_patch16_dinov3")
    
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Loaded SlotFormer from {model_path}")
    else:
        print(f"⚠️ Model not found, using random weights")
    
    model = model.to(device).eval()
    
    # Create simple decoder
    decoder = SimpleFeatureDecoder().to(device)
    
    # Quick decoder training on a few samples
    print("Training simple feature decoder...")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])
    
    try:
        dataset = torchvision.datasets.Imagenette(
            "./datasets", split="val", transform=transform, download=False
        )
        print(f"✓ Loaded dataset with {len(dataset)} images")
    except:
        print("⚠️ Creating synthetic data")
        # Create synthetic data
        synthetic_images = []
        for i in range(20):
            img = torch.randn(3, 224, 224)
            img = transform(img)
            synthetic_images.append((img, i))
        dataset = synthetic_images
    
    # Train decoder quickly
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    decoder.train()
    
    for epoch in range(20):  # Quick training
        total_loss = 0
        count = 0
        
        for i in range(min(50, len(dataset))):  # Use 50 samples
            image, _ = dataset[i]
            image = image.unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = model.encoder.forward_features(image)
                patch_features = features[:, model.encoder.num_prefix_tokens:]
            
            # Train decoder
            recon = decoder(patch_features)
            loss = F.mse_loss(recon, image)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            count += 1
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss = {total_loss/count:.6f}")
    
    decoder.eval()
    print("✓ Quick decoder training complete!")
    
    # Create results directory
    results_dir = Path("./quick_image_reconstruction")
    results_dir.mkdir(exist_ok=True)
    
    # Test reconstruction with different slot counts
    slot_counts = [1, 4, 16, 64, 128]
    
    print(f"Creating reconstructions for {num_images} images...")
    
    with torch.no_grad():
        for img_idx in range(min(num_images, len(dataset))):
            print(f"Processing image {img_idx + 1}...")
            
            image, _ = dataset[img_idx]
            image = image.unsqueeze(0).to(device)
            
            # Get ground truth features
            gt_features = model.encoder.forward_features(image)
            gt_patches = gt_features[:, model.encoder.num_prefix_tokens:]
            
            # Generate slots
            slots = model.generator(
                tgt=model.slot_queries.repeat(1, 1, 1),
                memory=gt_patches,
                tgt_mask=model.tgt_mask.to(device),
                tgt_is_causal=True,
            )
            
            # Create visualization
            fig, axes = plt.subplots(3, len(slot_counts), figsize=(20, 12))
            
            # Denormalize original image
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            orig_img = torch.clamp(image * std + mean, 0, 1)[0].cpu().permute(1, 2, 0)
            
            # Ground truth reconstruction
            gt_recon = decoder(gt_patches)
            gt_recon_img = torch.clamp((gt_recon[0] + 1) / 2, 0, 1).cpu().permute(1, 2, 0)
            
            losses = {}
            
            for i, num_slots in enumerate(slot_counts):
                # Use first num_slots
                used_slots = slots[:, :num_slots]
                
                # Pad with NULL tokens
                if num_slots < model.num_slots:
                    null_padding = model.null_slots[:, :model.num_slots-num_slots]
                    null_padding = null_padding.expand(1, -1, -1).type_as(slots)
                    padded_slots = torch.cat([used_slots, null_padding], dim=1)
                else:
                    padded_slots = used_slots
                
                # Reconstruct features
                recon_features = model.reconstructor(
                    tgt=model.decoder_pos_embed.repeat(1, 1, 1),
                    memory=padded_slots,
                )
                
                # Convert to image
                recon_image = decoder(recon_features)
                recon_img = torch.clamp((recon_image[0] + 1) / 2, 0, 1).cpu().permute(1, 2, 0)
                
                # Compute loss
                loss = F.mse_loss(recon_features, gt_patches).item()
                losses[num_slots] = loss
                
                # Plot
                # Row 1: Original
                axes[0, i].imshow(orig_img)
                axes[0, i].set_title(f'Original\n({num_slots} slots)')
                axes[0, i].axis('off')
                
                # Row 2: GT reconstruction
                axes[1, i].imshow(gt_recon_img)
                axes[1, i].set_title(f'GT Features\n(Upper bound)')
                axes[1, i].axis('off')
                
                # Row 3: SlotFormer reconstruction
                axes[2, i].imshow(recon_img)
                axes[2, i].set_title(f'SlotFormer\nLoss: {loss:.4f}')
                axes[2, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(results_dir / f'quick_reconstruction_img{img_idx}.png', 
                       dpi=200, bbox_inches='tight')
            plt.close()
            
            # Loss curve
            plt.figure(figsize=(8, 6))
            loss_values = [losses[s] for s in slot_counts]
            plt.loglog(slot_counts, loss_values, 'b-o', linewidth=2, markersize=8)
            plt.xlabel('Number of Slots')
            plt.ylabel('Feature Loss')
            plt.title(f'Reconstruction Quality (Image {img_idx})')
            plt.grid(True, alpha=0.3)
            
            improvement = loss_values[0] / loss_values[-1]
            plt.text(0.02, 0.98, f'Improvement: {improvement:.1f}x', 
                    transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            plt.savefig(results_dir / f'quick_loss_curve_img{img_idx}.png', 
                       dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"  Improvement: {improvement:.1f}x")
    
    print(f"\n✅ Quick image reconstruction demo complete!")
    print(f"📁 Results saved to: {results_dir.absolute()}")
    print(f"📊 This shows:")
    print(f"   - Original images")
    print(f"   - Upper bound reconstruction (from perfect features)")
    print(f"   - SlotFormer reconstructions with different slot counts")
    print(f"   - Progressive improvement in visual quality")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick Image Reconstruction Demo')
    parser.add_argument('--model', type=str, default='model-49.ckpt', 
                       help='Path to trained model')
    parser.add_argument('--num_images', type=int, default=3,
                       help='Number of images to test')
    
    args = parser.parse_args()
    
    quick_image_reconstruction_demo(args.model, args.num_images)
