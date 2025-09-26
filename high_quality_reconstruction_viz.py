#!/usr/bin/env python3
"""
High-quality image reconstruction with storage size analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from train import SlotFormer
import json

class HighQualityFeatureDecoder(nn.Module):
    """High-quality decoder with upsampling for better reconstructions."""
    
    def __init__(self, feature_dim=768, patch_size=16, img_size=224):
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2  # 196
        
        # Multi-scale decoder for high quality
        self.decoder = nn.Sequential(
            # Feature processing
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Generate larger patches for better quality
            nn.Linear(256, patch_size * patch_size * 8),  # 8 channels for upsampling
        )
        
        # Convolutional upsampling layers
        self.upsample = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, features):
        """
        Args:
            features: [B, num_patches, feature_dim]
        Returns:
            images: [B, 3, img_size, img_size]
        """
        B, num_patches, _ = features.shape
        
        # Decode each patch
        patch_features = self.decoder(features)  # [B, num_patches, patch_size^2 * 8]
        
        # Reshape to spatial format
        patch_features = patch_features.view(B, num_patches, 8, self.patch_size, self.patch_size)
        
        # Reconstruct image from patches
        patches_per_side = int(np.sqrt(num_patches))  # 14
        
        # Rearrange patches to form image
        patch_features = patch_features.view(B, patches_per_side, patches_per_side, 8, 
                                           self.patch_size, self.patch_size)
        
        # Combine patches into full image
        images = patch_features.permute(0, 3, 1, 4, 2, 5).contiguous()
        images = images.view(B, 8, self.img_size, self.img_size)
        
        # Apply convolutional upsampling
        images = self.upsample(images)
        
        return images

class StorageAnalysisVisualizer:
    """Visualize reconstructions with storage size analysis."""
    
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model = self.load_model(model_path)
        
        # Create high-quality decoder
        self.feature_decoder = HighQualityFeatureDecoder().to(device)
        self.train_feature_decoder()
        
        # Results directory
        self.results_dir = Path("./storage_analysis_viz")
        self.results_dir.mkdir(exist_ok=True)
        
    def load_model(self, model_path):
        """Load trained SlotFormer model."""
        model = SlotFormer(128, 3, "vit_base_patch16_dinov3")
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            print(f"✓ Loaded SlotFormer from {model_path}")
        else:
            print(f"⚠️ Model not found: {model_path}")
            
        return model.to(self.device).eval()
    
    def train_feature_decoder(self, num_samples=300, epochs=100):
        """Train high-quality decoder."""
        print("Training high-quality feature-to-image decoder...")
        
        # Load training data
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
                "./datasets", split="train", transform=transform, download=False
            )
        except:
            print("⚠️ Using validation set for decoder training")
            dataset = torchvision.datasets.Imagenette(
                "./datasets", split="val", transform=transform, download=False
            )
        
        # Collect feature-image pairs
        features_list = []
        images_list = []
        
        print(f"Collecting {num_samples} feature-image pairs...")
        with torch.no_grad():
            for i in range(min(num_samples, len(dataset))):
                if i % 50 == 0:
                    print(f"  Collected {i}/{num_samples}")
                
                image, _ = dataset[i]
                image = image.unsqueeze(0).to(self.device)
                
                # Get DINOv3 features
                features = self.model.encoder.forward_features(image)
                patch_features = features[:, self.model.encoder.num_prefix_tokens:]
                
                features_list.append(patch_features[0].cpu())
                images_list.append(image[0].cpu())
        
        # Train decoder with better optimization
        optimizer = torch.optim.AdamW(self.feature_decoder.parameters(), 
                                     lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        print(f"Training high-quality decoder for {epochs} epochs...")
        self.feature_decoder.train()
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0
            count = 0
            
            # Shuffle data
            indices = torch.randperm(len(features_list))
            
            for i in range(0, len(features_list), 4):  # Batch size 4
                batch_indices = indices[i:i+4]
                batch_features = torch.stack([features_list[j] for j in batch_indices]).to(self.device)
                batch_images = torch.stack([images_list[j] for j in batch_indices]).to(self.device)
                
                # Forward pass
                reconstructed = self.feature_decoder(batch_features)
                
                # Multi-scale loss
                loss_mse = F.mse_loss(reconstructed, batch_images)
                
                # Perceptual loss (simple version using downsampled images)
                recon_small = F.interpolate(reconstructed, size=56, mode='bilinear')
                target_small = F.interpolate(batch_images, size=56, mode='bilinear')
                loss_perceptual = F.mse_loss(recon_small, target_small)
                
                total_loss_batch = loss_mse + 0.1 * loss_perceptual
                
                # Backward pass
                optimizer.zero_grad()
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.feature_decoder.parameters(), 1.0)
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                count += 1
            
            scheduler.step()
            avg_loss = total_loss / count
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                # Save best model
                torch.save(self.feature_decoder.state_dict(), 
                          self.results_dir / 'best_decoder.pth')
            
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: Loss = {avg_loss:.6f}, LR = {scheduler.get_last_lr()[0]:.6f}")
        
        # Load best model
        self.feature_decoder.load_state_dict(
            torch.load(self.results_dir / 'best_decoder.pth')
        )
        self.feature_decoder.eval()
        print("✓ High-quality feature decoder training complete!")
    
    def calculate_storage_sizes(self, slot_counts):
        """Calculate storage requirements for different representations."""
        
        # Image storage (RGB, 224x224, uint8)
        image_size_bytes = 224 * 224 * 3  # 150,528 bytes
        
        # DINOv3 features (196 patches, 768 dims, float32)
        dinov3_size_bytes = 196 * 768 * 4  # 602,112 bytes
        
        # SlotFormer representations
        slot_sizes = {}
        for num_slots in slot_counts:
            # Each slot: 768 dims, float32
            slot_size_bytes = num_slots * 768 * 4
            slot_sizes[num_slots] = slot_size_bytes
        
        return {
            'image_rgb': image_size_bytes,
            'dinov3_features': dinov3_size_bytes,
            'slots': slot_sizes
        }
    
    def create_storage_comparison_visualization(self, image_idx=0, 
                                              slot_counts=[1, 2, 4, 8, 16, 32, 64, 128]):
        """Create visualization with storage size analysis."""
        
        # Load test data
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
        except:
            print("⚠️ Creating synthetic test image")
            synthetic_img = torch.randn(3, 224, 224)
            dataset = [(synthetic_img, 0)]
        
        image, label = dataset[image_idx]
        image = image.unsqueeze(0).to(self.device)
        
        # Calculate storage sizes
        storage_sizes = self.calculate_storage_sizes(slot_counts)
        
        with torch.no_grad():
            # Get ground truth features
            gt_features = self.model.encoder.forward_features(image)
            gt_patches = gt_features[:, self.model.encoder.num_prefix_tokens:]
            
            # Generate all slots
            slots = self.model.generator(
                tgt=self.model.slot_queries.repeat(1, 1, 1),
                memory=gt_patches,
                tgt_mask=self.model.tgt_mask.to(self.device),
                tgt_is_causal=True,
            )
            
            # Get reconstructions
            reconstructed_images = {}
            losses = {}
            
            for num_slots in slot_counts:
                print(f"Processing {num_slots} slots...")
                
                # Use only first num_slots
                used_slots = slots[:, :num_slots]
                
                # Pad with NULL tokens
                if num_slots < self.model.num_slots:
                    null_padding = self.model.null_slots[:, :self.model.num_slots-num_slots]
                    null_padding = null_padding.expand(1, -1, -1).type_as(slots)
                    padded_slots = torch.cat([used_slots, null_padding], dim=1)
                else:
                    padded_slots = used_slots
                
                # Reconstruct features
                recon_features = self.model.reconstructor(
                    tgt=self.model.decoder_pos_embed.repeat(1, 1, 1),
                    memory=padded_slots,
                )
                
                # Convert features to high-quality image
                recon_image = self.feature_decoder(recon_features)
                
                reconstructed_images[num_slots] = recon_image
                losses[num_slots] = F.mse_loss(recon_features, gt_patches).item()
            
            # Ground truth reconstruction
            gt_image_recon = self.feature_decoder(gt_patches)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(40, 20))
        
        # Denormalize function
        def denormalize(tensor):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
            return torch.clamp(tensor * std + mean, 0, 1)
        
        # Original image
        orig_img = denormalize(image)[0].cpu().permute(1, 2, 0)
        gt_recon_img = torch.clamp((gt_image_recon[0] + 1) / 2, 0, 1).cpu().permute(1, 2, 0)
        
        # Create grid layout
        gs = fig.add_gridspec(5, len(slot_counts), height_ratios=[1, 1, 1, 0.3, 1])
        
        for i, num_slots in enumerate(slot_counts):
            # Row 1: Original image
            ax1 = fig.add_subplot(gs[0, i])
            ax1.imshow(orig_img)
            storage_kb = storage_sizes['image_rgb'] / 1024
            ax1.set_title(f'Original Image\nStorage: {storage_kb:.1f} KB', fontsize=14)
            ax1.axis('off')
            
            # Row 2: Ground truth reconstruction
            ax2 = fig.add_subplot(gs[1, i])
            ax2.imshow(gt_recon_img)
            dinov3_kb = storage_sizes['dinov3_features'] / 1024
            ax2.set_title(f'GT Features (DINOv3)\nStorage: {dinov3_kb:.1f} KB', fontsize=14)
            ax2.axis('off')
            
            # Row 3: SlotFormer reconstruction
            ax3 = fig.add_subplot(gs[2, i])
            recon_img = torch.clamp((reconstructed_images[num_slots][0] + 1) / 2, 0, 1)
            recon_img = recon_img.cpu().permute(1, 2, 0)
            ax3.imshow(recon_img)
            
            slot_kb = storage_sizes['slots'][num_slots] / 1024
            compression_ratio = storage_sizes['image_rgb'] / storage_sizes['slots'][num_slots]
            
            ax3.set_title(f'SlotFormer ({num_slots} slots)\n'
                         f'Storage: {slot_kb:.1f} KB\n'
                         f'Compression: {compression_ratio:.1f}x\n'
                         f'Loss: {losses[num_slots]:.4f}', fontsize=14)
            ax3.axis('off')
            
            # Row 4: Storage comparison bar
            ax4 = fig.add_subplot(gs[3, i])
            sizes = [storage_sizes['image_rgb'], storage_sizes['slots'][num_slots]]
            labels = ['RGB', f'{num_slots} Slots']
            colors = ['lightcoral', 'lightblue']
            
            bars = ax4.bar(labels, [s/1024 for s in sizes], color=colors)
            ax4.set_ylabel('Storage (KB)', fontsize=12)
            ax4.set_title('Storage Comparison', fontsize=12)
            
            # Add value labels on bars
            for bar, size in zip(bars, sizes):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{size/1024:.0f}KB', ha='center', va='bottom', fontsize=10)
        
        # Row 5: Overall analysis
        ax5 = fig.add_subplot(gs[4, :])
        
        # Storage vs Quality plot
        slot_storage = [storage_sizes['slots'][s]/1024 for s in slot_counts]
        slot_losses = [losses[s] for s in slot_counts]
        
        ax5_twin = ax5.twinx()
        
        # Storage line
        line1 = ax5.plot(slot_counts, slot_storage, 'b-o', linewidth=2, markersize=8, label='Storage (KB)')
        ax5.set_xlabel('Number of Slots', fontsize=14)
        ax5.set_ylabel('Storage (KB)', color='b', fontsize=14)
        ax5.set_xscale('log')
        ax5.tick_params(axis='y', labelcolor='b')
        
        # Quality line
        line2 = ax5_twin.plot(slot_counts, slot_losses, 'r-s', linewidth=2, markersize=8, label='Reconstruction Loss')
        ax5_twin.set_ylabel('Reconstruction Loss', color='r', fontsize=14)
        ax5_twin.set_yscale('log')
        ax5_twin.tick_params(axis='y', labelcolor='r')
        
        # Add compression ratios as text
        for i, (slots, storage, loss) in enumerate(zip(slot_counts, slot_storage, slot_losses)):
            compression = storage_sizes['image_rgb'] / storage_sizes['slots'][slots]
            if i % 2 == 0:  # Show every other one to avoid crowding
                ax5.text(slots, storage + 10, f'{compression:.1f}x', 
                        ha='center', va='bottom', fontsize=10, color='green')
        
        ax5.set_title('Storage vs Quality Trade-off', fontsize=16)
        ax5.grid(True, alpha=0.3)
        
        # Legend
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_twin.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'storage_analysis_img{image_idx}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed analysis
        analysis = {
            'image_idx': image_idx,
            'storage_sizes': storage_sizes,
            'reconstruction_losses': losses,
            'compression_ratios': {
                s: storage_sizes['image_rgb'] / storage_sizes['slots'][s] 
                for s in slot_counts
            },
            'quality_improvement': losses[slot_counts[0]] / losses[slot_counts[-1]]
        }
        
        with open(self.results_dir / f'storage_analysis_img{image_idx}.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"✓ Storage analysis saved for image {image_idx}")
        print(f"  Quality improvement: {analysis['quality_improvement']:.1f}x")
        print(f"  Best compression: {max(analysis['compression_ratios'].values()):.1f}x at 1 slot")
        print(f"  Best quality: {min(losses.values()):.4f} at {max(slot_counts)} slots")
        
        return analysis

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Storage Analysis Visualization')
    parser.add_argument('--model', type=str, default='model-49.ckpt', 
                       help='Path to trained model')
    parser.add_argument('--num_images', type=int, default=3,
                       help='Number of images to analyze')
    
    args = parser.parse_args()
    
    visualizer = StorageAnalysisVisualizer(args.model)
    
    print("=== Creating Storage Analysis Visualizations ===")
    
    for i in range(args.num_images):
        print(f"\nProcessing image {i+1}/{args.num_images}...")
        analysis = visualizer.create_storage_comparison_visualization(i)
    
    print(f"\n✅ Storage analysis complete!")
    print(f"📁 Results saved to: {visualizer.results_dir.absolute()}")

if __name__ == "__main__":
    main()
