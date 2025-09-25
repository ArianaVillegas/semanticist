#!/usr/bin/env python3
"""
Fixed SlotFormer visualization with proper feature space handling.
"""

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from train import SlotFormer
from sklearn.decomposition import PCA
import cv2

class FixedSlotVisualizer:
    """Fixed SlotFormer visualizer with proper feature space handling."""
    
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model = self.load_model(model_path)
        
        # Create results directory
        self.results_dir = Path("./fixed_slot_visualizations")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize PCA for feature visualization
        self.pca = None
        
    def load_model(self, model_path):
        """Load trained SlotFormer model."""
        model = SlotFormer(128, 3, "vit_base_patch16_dinov3")
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            print(f"✓ Loaded model from {model_path}")
        else:
            print(f"⚠️ Model not found: {model_path}")
            
        return model.to(self.device).eval()
    
    def fit_pca_for_visualization(self, num_samples=100):
        """Fit PCA on patch features for better visualization."""
        print("Fitting PCA for feature visualization...")
        
        # Load dataset
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])
        
        dataset = torchvision.datasets.Imagenette(
            "./datasets", split="val", transform=transform, download=False
        )
        
        # Collect features from multiple images
        all_features = []
        
        with torch.no_grad():
            for i in range(min(num_samples, len(dataset))):
                image, _ = dataset[i]
                image = image.unsqueeze(0).to(self.device)
                
                # Get patch features
                patch_tokens = self.model.encoder.forward_features(image)
                patch_tokens = patch_tokens[:, self.model.encoder.num_prefix_tokens:]
                
                all_features.append(patch_tokens[0].cpu().numpy())  # [196, 768]
        
        # Fit PCA on all collected features
        all_features = np.concatenate(all_features, axis=0)  # [num_samples*196, 768]
        
        self.pca = PCA(n_components=3)
        self.pca.fit(all_features)
        
        print(f"✓ PCA fitted on {all_features.shape[0]} patch features")
        print(f"  Explained variance ratio: {self.pca.explained_variance_ratio_}")
        
    def features_to_rgb(self, patches):
        """Convert 768D patch features to RGB using PCA."""
        B, num_patches, embed_dim = patches.shape
        
        if self.pca is None:
            self.fit_pca_for_visualization()
        
        # Flatten patches
        patches_flat = patches.view(-1, embed_dim).cpu().numpy()  # [B*196, 768]
        
        # Apply PCA to reduce to 3D
        rgb_patches = self.pca.transform(patches_flat)  # [B*196, 3]
        
        # Normalize to [0, 1] for visualization
        for i in range(3):
            channel = rgb_patches[:, i]
            channel_min, channel_max = channel.min(), channel.max()
            if channel_max > channel_min:
                rgb_patches[:, i] = (channel - channel_min) / (channel_max - channel_min)
            else:
                rgb_patches[:, i] = 0.5
        
        # Reshape to image format
        patch_size = int(np.sqrt(num_patches))  # 14
        rgb_patches = rgb_patches.reshape(B, patch_size, patch_size, 3)
        
        return rgb_patches
    
    def visualize_reconstruction_quality(self, image_idx=0, slot_counts=[1, 2, 4, 8, 16, 32, 64, 128]):
        """Visualize reconstruction quality with proper feature handling."""
        
        # Load test data
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])
        
        dataset = torchvision.datasets.Imagenette(
            "./datasets", split="val", transform=transform, download=False
        )
        
        image, label = dataset[image_idx]
        image = image.unsqueeze(0).to(self.device)
        
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
            
            # Get reconstructions for different slot counts
            reconstructions = {}
            losses = {}
            
            for num_slots in slot_counts:
                # Use only first num_slots
                used_slots = slots[:, :num_slots]
                
                # Pad with NULL tokens
                if num_slots < self.model.num_slots:
                    null_padding = self.model.null_slots[:, :self.model.num_slots-num_slots]
                    null_padding = null_padding.expand(1, -1, -1).type_as(slots)
                    padded_slots = torch.cat([used_slots, null_padding], dim=1)
                else:
                    padded_slots = used_slots
                
                # Reconstruct
                recon = self.model.reconstructor(
                    tgt=self.model.decoder_pos_embed.repeat(1, 1, 1),
                    memory=padded_slots,
                )
                
                reconstructions[num_slots] = recon
                losses[num_slots] = F.mse_loss(recon, gt_patches).item()
        
        # Create visualization
        fig, axes = plt.subplots(4, len(slot_counts), figsize=(24, 12))
        
        # Original image (denormalized)
        orig_img = image[0].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        orig_img = orig_img * std + mean
        orig_img = torch.clamp(orig_img, 0, 1)
        
        # Convert ground truth and reconstructions to RGB
        gt_rgb = self.features_to_rgb(gt_patches)
        
        for i, num_slots in enumerate(slot_counts):
            # Row 1: Original image
            axes[0, i].imshow(orig_img.permute(1, 2, 0))
            axes[0, i].set_title(f'Original Image', fontsize=10)
            axes[0, i].axis('off')
            
            # Row 2: Ground truth features (PCA visualization)
            axes[1, i].imshow(gt_rgb[0])
            axes[1, i].set_title(f'GT Features (PCA)', fontsize=10)
            axes[1, i].axis('off')
            
            # Row 3: Reconstruction features (PCA visualization)
            recon_rgb = self.features_to_rgb(reconstructions[num_slots])
            axes[2, i].imshow(recon_rgb[0])
            axes[2, i].set_title(f'{num_slots} Slots\nLoss: {losses[num_slots]:.4f}', fontsize=10)
            axes[2, i].axis('off')
            
            # Row 4: Feature difference (error map)
            diff = torch.abs(reconstructions[num_slots] - gt_patches)
            diff_rgb = self.features_to_rgb(diff)
            axes[3, i].imshow(diff_rgb[0])
            axes[3, i].set_title(f'Error Map', fontsize=10)
            axes[3, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'fixed_reconstruction_img{image_idx}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create loss curve
        plt.figure(figsize=(10, 6))
        loss_values = [losses[s] for s in slot_counts]
        plt.loglog(slot_counts, loss_values, 'b-o', linewidth=2, markersize=8)
        plt.xlabel('Number of Slots')
        plt.ylabel('Reconstruction Loss (MSE)')
        plt.title(f'Reconstruction Quality vs Slots (Image {image_idx})')
        plt.grid(True, alpha=0.3)
        
        # Add improvement annotation
        improvement = loss_values[0] / loss_values[-1]
        plt.text(0.02, 0.98, f'Improvement: {improvement:.1f}x\nMonotonic: {"✓" if all(loss_values[i] >= loss_values[i+1] for i in range(len(loss_values)-1)) else "✗"}', 
                transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.savefig(self.results_dir / f'loss_curve_img{image_idx}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Fixed visualization saved for image {image_idx}")
        print(f"  Improvement: {improvement:.1f}x")
        print(f"  Losses: {dict(zip(slot_counts, loss_values))}")
        
        return losses
    
    def analyze_feature_similarity(self, image_idx=0):
        """Analyze how similar reconstructed features are to ground truth."""
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])
        
        dataset = torchvision.datasets.Imagenette(
            "./datasets", split="val", transform=transform, download=False
        )
        
        image, label = dataset[image_idx]
        image = image.unsqueeze(0).to(self.device)
        
        slot_counts = [1, 4, 16, 64, 128]
        
        with torch.no_grad():
            # Get ground truth
            gt_features = self.model.encoder.forward_features(image)
            gt_patches = gt_features[:, self.model.encoder.num_prefix_tokens:]
            
            # Generate slots
            slots = self.model.generator(
                tgt=self.model.slot_queries.repeat(1, 1, 1),
                memory=gt_patches,
                tgt_mask=self.model.tgt_mask.to(self.device),
                tgt_is_causal=True,
            )
            
            similarities = {}
            
            for num_slots in slot_counts:
                # Reconstruct with num_slots
                used_slots = slots[:, :num_slots]
                if num_slots < self.model.num_slots:
                    null_padding = self.model.null_slots[:, :self.model.num_slots-num_slots]
                    null_padding = null_padding.expand(1, -1, -1).type_as(slots)
                    padded_slots = torch.cat([used_slots, null_padding], dim=1)
                else:
                    padded_slots = used_slots
                
                recon = self.model.reconstructor(
                    tgt=self.model.decoder_pos_embed.repeat(1, 1, 1),
                    memory=padded_slots,
                )
                
                # Compute cosine similarity per patch
                gt_norm = F.normalize(gt_patches[0], dim=-1)  # [196, 768]
                recon_norm = F.normalize(recon[0], dim=-1)   # [196, 768]
                
                patch_similarities = torch.sum(gt_norm * recon_norm, dim=-1)  # [196]
                similarities[num_slots] = patch_similarities.cpu().numpy()
        
        # Visualize similarity maps
        fig, axes = plt.subplots(1, len(slot_counts), figsize=(20, 4))
        
        for i, num_slots in enumerate(slot_counts):
            sim_map = similarities[num_slots].reshape(14, 14)
            
            im = axes[i].imshow(sim_map, cmap='RdYlGn', vmin=0, vmax=1)
            axes[i].set_title(f'{num_slots} Slots\nAvg Sim: {sim_map.mean():.3f}')
            axes[i].axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'feature_similarity_img{image_idx}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Feature similarity analysis saved")
        
        return similarities
    
    def create_comprehensive_analysis(self, num_images=3):
        """Create comprehensive analysis with fixed visualizations."""
        
        print("=== Creating Fixed SlotFormer Analysis ===")
        
        all_losses = []
        
        for i in range(num_images):
            print(f"Processing image {i+1}/{num_images}...")
            
            # Fixed reconstruction visualization
            losses = self.visualize_reconstruction_quality(i)
            all_losses.append(losses)
            
            # Feature similarity analysis
            self.analyze_feature_similarity(i)
        
        # Summary analysis
        slot_counts = [1, 2, 4, 8, 16, 32, 64, 128]
        
        plt.figure(figsize=(12, 8))
        
        for i, losses in enumerate(all_losses):
            loss_values = [losses[s] for s in slot_counts]
            plt.loglog(slot_counts, loss_values, 'o-', alpha=0.7, label=f'Image {i+1}')
        
        # Average
        avg_losses = []
        for s in slot_counts:
            avg_loss = np.mean([losses[s] for losses in all_losses])
            avg_losses.append(avg_loss)
        
        plt.loglog(slot_counts, avg_losses, 'k-', linewidth=3, 
                  marker='s', markersize=8, label='Average')
        
        plt.xlabel('Number of Slots', fontsize=12)
        plt.ylabel('Reconstruction Loss (MSE)', fontsize=12)
        plt.title('SlotFormer: Feature Space Reconstruction Quality', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add analysis text
        improvement = avg_losses[0] / avg_losses[-1]
        monotonic_count = sum(1 for i in range(len(avg_losses)-1) if avg_losses[i] >= avg_losses[i+1])
        monotonic_ratio = monotonic_count / (len(avg_losses) - 1)
        
        textstr = f'Average Improvement: {improvement:.1f}x\n'
        textstr += f'Monotonic Ratio: {monotonic_ratio:.3f}\n'
        textstr += f'Status: {"✓ EXCELLENT" if monotonic_ratio >= 0.8 else "⚠️ MODERATE" if monotonic_ratio >= 0.6 else "✗ POOR"}'
        
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'summary_fixed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Fixed analysis complete!")
        print(f"📁 Results saved to: {self.results_dir.absolute()}")
        print(f"📊 Key insights:")
        print(f"   - SlotFormer works in 768D DINOv3 feature space")
        print(f"   - PCA visualization shows feature reconstruction quality")
        print(f"   - Average improvement: {improvement:.1f}x")
        print(f"   - Monotonic ratio: {monotonic_ratio:.3f}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fixed SlotFormer visualization')
    parser.add_argument('--model', type=str, default='model-49.ckpt', 
                       help='Path to trained model')
    parser.add_argument('--num_images', type=int, default=3,
                       help='Number of images to analyze')
    
    args = parser.parse_args()
    
    visualizer = FixedSlotVisualizer(args.model)
    visualizer.create_comprehensive_analysis(args.num_images)

if __name__ == "__main__":
    main()
