#!/usr/bin/env python3
"""
Cluster-optimized fixed SlotFormer visualization.
"""

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from train import SlotFormer
import os
import time
import json

# Try to import sklearn, install if needed
try:
    from sklearn.decomposition import PCA
except ImportError:
    print("Installing scikit-learn...")
    os.system("pip install scikit-learn")
    from sklearn.decomposition import PCA

class ClusterFixedSlotVisualizer:
    """Cluster-optimized fixed SlotFormer visualizer."""
    
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model = self.load_model(model_path)
        
        # Create results directory
        self.results_dir = Path("./cluster_fixed_visualizations")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize PCA
        self.pca = None
        
        print(f"Cluster Fixed Visualizer initialized on {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
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
            print(f"⚠️ Model not found: {model_path}, using random weights")
            
        return model.to(self.device).eval()
    
    def setup_dataset(self):
        """Setup dataset with error handling."""
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
            print(f"✓ Loaded Imagenette dataset with {len(dataset)} images")
            return dataset
        except Exception as e:
            print(f"⚠️ Could not load Imagenette: {e}")
            print("Creating synthetic dataset for testing...")
            
            # Create synthetic dataset
            synthetic_images = []
            for i in range(50):
                # Create diverse synthetic images
                img = torch.randn(3, 224, 224)
                # Add some structure
                img[:, 50:150, 50:150] += 2.0  # Bright square
                img[:, 100:200, 100:200] -= 1.0  # Dark square
                synthetic_images.append((img, i % 10))
            
            return synthetic_images
    
    def fit_pca_efficiently(self, dataset, num_samples=50):
        """Efficiently fit PCA for feature visualization."""
        print(f"Fitting PCA on {num_samples} samples...")
        
        all_features = []
        
        with torch.no_grad():
            for i in range(min(num_samples, len(dataset))):
                if i % 10 == 0:
                    print(f"  Processing sample {i+1}/{num_samples}")
                
                if isinstance(dataset[i], tuple):
                    image, _ = dataset[i]
                else:
                    image = dataset[i]
                    
                image = image.unsqueeze(0).to(self.device)
                
                # Get patch features
                patch_tokens = self.model.encoder.forward_features(image)
                patch_tokens = patch_tokens[:, self.model.encoder.num_prefix_tokens:]
                
                # Sample patches to reduce memory usage
                sampled_patches = patch_tokens[0][::4].cpu().numpy()  # Every 4th patch
                all_features.append(sampled_patches)
        
        # Fit PCA
        all_features = np.concatenate(all_features, axis=0)
        print(f"  Fitting PCA on {all_features.shape[0]} patch features...")
        
        self.pca = PCA(n_components=3)
        self.pca.fit(all_features)
        
        explained_var = self.pca.explained_variance_ratio_
        print(f"✓ PCA fitted - Explained variance: {explained_var}")
        print(f"  Total explained: {explained_var.sum():.3f}")
        
    def features_to_rgb(self, patches):
        """Convert 768D patch features to RGB using PCA."""
        B, num_patches, embed_dim = patches.shape
        
        if self.pca is None:
            raise ValueError("PCA not fitted! Call fit_pca_efficiently first.")
        
        # Flatten and convert
        patches_flat = patches.view(-1, embed_dim).cpu().numpy()
        rgb_patches = self.pca.transform(patches_flat)
        
        # Normalize each channel independently
        for i in range(3):
            channel = rgb_patches[:, i]
            p5, p95 = np.percentile(channel, [5, 95])  # Robust normalization
            rgb_patches[:, i] = np.clip((channel - p5) / (p95 - p5 + 1e-8), 0, 1)
        
        # Reshape
        patch_size = int(np.sqrt(num_patches))
        rgb_patches = rgb_patches.reshape(B, patch_size, patch_size, 3)
        
        return rgb_patches
    
    def analyze_single_image(self, image, image_idx, slot_counts=[1, 2, 4, 8, 16, 32, 64, 128]):
        """Analyze a single image with different slot counts."""
        
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
            
            results = {
                'losses': {},
                'similarities': {},
                'reconstructions': {}
            }
            
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
                
                # Compute metrics
                loss = F.mse_loss(recon, gt_patches).item()
                
                # Cosine similarity
                gt_norm = F.normalize(gt_patches[0], dim=-1)
                recon_norm = F.normalize(recon[0], dim=-1)
                similarity = torch.sum(gt_norm * recon_norm, dim=-1).mean().item()
                
                results['losses'][num_slots] = loss
                results['similarities'][num_slots] = similarity
                results['reconstructions'][num_slots] = recon
            
            return results, gt_patches
    
    def create_comprehensive_visualization(self, image_idx, results, gt_patches, image):
        """Create comprehensive visualization for one image."""
        
        slot_counts = [1, 2, 4, 8, 16, 32, 64, 128]
        
        # Create main visualization
        fig, axes = plt.subplots(4, len(slot_counts), figsize=(32, 16))
        
        # Original image (denormalized)
        orig_img = image[0].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        orig_img = orig_img * std + mean
        orig_img = torch.clamp(orig_img, 0, 1)
        
        # Ground truth RGB
        gt_rgb = self.features_to_rgb(gt_patches)
        
        for i, num_slots in enumerate(slot_counts):
            # Row 1: Original image
            axes[0, i].imshow(orig_img.permute(1, 2, 0))
            axes[0, i].set_title(f'Original\n({num_slots} slots)', fontsize=12)
            axes[0, i].axis('off')
            
            # Row 2: Ground truth features
            axes[1, i].imshow(gt_rgb[0])
            axes[1, i].set_title(f'GT Features\n(PCA Projection)', fontsize=12)
            axes[1, i].axis('off')
            
            # Row 3: Reconstruction
            recon_rgb = self.features_to_rgb(results['reconstructions'][num_slots])
            axes[2, i].imshow(recon_rgb[0])
            loss = results['losses'][num_slots]
            sim = results['similarities'][num_slots]
            axes[2, i].set_title(f'Reconstruction\nLoss: {loss:.4f}\nSim: {sim:.3f}', fontsize=12)
            axes[2, i].axis('off')
            
            # Row 4: Error map
            diff = torch.abs(results['reconstructions'][num_slots] - gt_patches)
            diff_rgb = self.features_to_rgb(diff)
            axes[3, i].imshow(diff_rgb[0])
            axes[3, i].set_title(f'Error Map', fontsize=12)
            axes[3, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'comprehensive_analysis_img{image_idx}.png', 
                   dpi=200, bbox_inches='tight')  # Lower DPI for cluster
        plt.close()
        
        # Create metrics plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss curve
        losses = [results['losses'][s] for s in slot_counts]
        ax1.loglog(slot_counts, losses, 'b-o', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Slots')
        ax1.set_ylabel('Reconstruction Loss (MSE)')
        ax1.set_title(f'Reconstruction Quality (Image {image_idx})')
        ax1.grid(True, alpha=0.3)
        
        # Similarity curve
        similarities = [results['similarities'][s] for s in slot_counts]
        ax2.semilogx(slot_counts, similarities, 'g-s', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Slots')
        ax2.set_ylabel('Cosine Similarity')
        ax2.set_title(f'Feature Similarity (Image {image_idx})')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'metrics_img{image_idx}.png', 
                   dpi=200, bbox_inches='tight')
        plt.close()
        
        return losses, similarities
    
    def run_cluster_analysis(self, num_images=10):
        """Run comprehensive cluster analysis."""
        
        print("=== Cluster Fixed SlotFormer Analysis ===")
        
        # Setup
        dataset = self.setup_dataset()
        self.fit_pca_efficiently(dataset, num_samples=min(50, len(dataset)))
        
        # Analysis
        all_losses = []
        all_similarities = []
        slot_counts = [1, 2, 4, 8, 16, 32, 64, 128]
        
        print(f"\nAnalyzing {num_images} images...")
        
        for i in range(min(num_images, len(dataset))):
            print(f"Processing image {i+1}/{num_images}...")
            
            # Get image
            if isinstance(dataset[i], tuple):
                image, _ = dataset[i]
            else:
                image = dataset[i]
            image = image.unsqueeze(0).to(self.device)
            
            # Analyze
            results, gt_patches = self.analyze_single_image(image, i, slot_counts)
            
            # Visualize
            losses, similarities = self.create_comprehensive_visualization(i, results, gt_patches, image)
            
            all_losses.append(dict(zip(slot_counts, losses)))
            all_similarities.append(dict(zip(slot_counts, similarities)))
            
            print(f"  Loss improvement: {losses[0]/losses[-1]:.1f}x")
            print(f"  Similarity improvement: {similarities[-1]/similarities[0]:.2f}x")
        
        # Create summary
        self.create_summary_analysis(all_losses, all_similarities, slot_counts)
        
        # Save results
        self.save_analysis_results(all_losses, all_similarities, slot_counts)
        
        print(f"\n✅ Cluster analysis complete!")
        print(f"📁 Results saved to: {self.results_dir.absolute()}")
        
        return all_losses, all_similarities
    
    def create_summary_analysis(self, all_losses, all_similarities, slot_counts):
        """Create summary analysis across all images."""
        
        # Compute averages
        avg_losses = []
        avg_similarities = []
        
        for s in slot_counts:
            avg_loss = np.mean([losses[s] for losses in all_losses])
            avg_sim = np.mean([sims[s] for sims in all_similarities])
            avg_losses.append(avg_loss)
            avg_similarities.append(avg_sim)
        
        # Create summary plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Individual loss curves
        axes[0, 0].set_title('Individual Loss Curves')
        for i, losses in enumerate(all_losses):
            loss_values = [losses[s] for s in slot_counts]
            axes[0, 0].loglog(slot_counts, loss_values, 'o-', alpha=0.6, label=f'Img {i+1}')
        axes[0, 0].loglog(slot_counts, avg_losses, 'k-', linewidth=3, label='Average')
        axes[0, 0].set_xlabel('Number of Slots')
        axes[0, 0].set_ylabel('Reconstruction Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Individual similarity curves
        axes[0, 1].set_title('Individual Similarity Curves')
        for i, sims in enumerate(all_similarities):
            sim_values = [sims[s] for s in slot_counts]
            axes[0, 1].semilogx(slot_counts, sim_values, 's-', alpha=0.6, label=f'Img {i+1}')
        axes[0, 1].semilogx(slot_counts, avg_similarities, 'k-', linewidth=3, label='Average')
        axes[0, 1].set_xlabel('Number of Slots')
        axes[0, 1].set_ylabel('Cosine Similarity')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].legend()
        
        # Improvement analysis
        axes[1, 0].set_title('Loss Improvement per Image')
        improvements = [losses[slot_counts[0]]/losses[slot_counts[-1]] for losses in all_losses]
        axes[1, 0].bar(range(len(improvements)), improvements)
        axes[1, 0].set_xlabel('Image Index')
        axes[1, 0].set_ylabel('Improvement Factor')
        axes[1, 0].axhline(y=np.mean(improvements), color='red', linestyle='--', 
                          label=f'Avg: {np.mean(improvements):.1f}x')
        axes[1, 0].legend()
        
        # Monotonic analysis
        axes[1, 1].set_title('Monotonic Improvement Analysis')
        monotonic_ratios = []
        for losses in all_losses:
            loss_values = [losses[s] for s in slot_counts]
            decreases = sum(1 for i in range(1, len(loss_values)) if loss_values[i] < loss_values[i-1])
            ratio = decreases / (len(loss_values) - 1)
            monotonic_ratios.append(ratio)
        
        axes[1, 1].bar(range(len(monotonic_ratios)), monotonic_ratios)
        axes[1, 1].set_xlabel('Image Index')
        axes[1, 1].set_ylabel('Monotonic Ratio')
        axes[1, 1].axhline(y=0.8, color='red', linestyle='--', label='Target (0.8)')
        axes[1, 1].axhline(y=np.mean(monotonic_ratios), color='green', linestyle='--', 
                          label=f'Avg: {np.mean(monotonic_ratios):.3f}')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'summary_analysis.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        # Print summary
        avg_improvement = np.mean(improvements)
        avg_monotonic = np.mean(monotonic_ratios)
        
        print(f"\n📊 SUMMARY RESULTS:")
        print(f"   Average improvement: {avg_improvement:.1f}x")
        print(f"   Average monotonic ratio: {avg_monotonic:.3f}")
        print(f"   Status: {'✓ EXCELLENT' if avg_monotonic >= 0.8 else '⚠️ MODERATE' if avg_monotonic >= 0.6 else '✗ POOR'}")
        
    def save_analysis_results(self, all_losses, all_similarities, slot_counts):
        """Save analysis results to JSON."""
        
        results = {
            'timestamp': time.time(),
            'date': time.ctime(),
            'device': str(self.device),
            'slot_counts': slot_counts,
            'losses': all_losses,
            'similarities': all_similarities,
            'pca_explained_variance': self.pca.explained_variance_ratio_.tolist() if self.pca else None,
            'summary': {
                'avg_improvement': np.mean([losses[slot_counts[0]]/losses[slot_counts[-1]] for losses in all_losses]),
                'avg_monotonic_ratio': np.mean([
                    sum(1 for i in range(1, len(slot_counts)) 
                        if [losses[s] for s in slot_counts][i] < [losses[s] for s in slot_counts][i-1]) / (len(slot_counts) - 1)
                    for losses in all_losses
                ])
            }
        }
        
        with open(self.results_dir / 'cluster_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved to cluster_analysis_results.json")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cluster Fixed SlotFormer Visualization')
    parser.add_argument('--model', type=str, default='model-49.ckpt', 
                       help='Path to trained model')
    parser.add_argument('--num_images', type=int, default=10,
                       help='Number of images to analyze')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    visualizer = ClusterFixedSlotVisualizer(args.model, args.device)
    visualizer.run_cluster_analysis(args.num_images)

if __name__ == "__main__":
    main()
