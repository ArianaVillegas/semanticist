#!/usr/bin/env python3
"""
Library for performing feature space analysis on SlotFormer models.

This module provides tools to:
- Load trained models.
- Create diverse datasets (real and synthetic).
- Analyze model reconstructions with varying slot counts.
- Visualize feature spaces using PCA.
- Generate quantitative and qualitative results.
"""

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import os
from sklearn.decomposition import PCA

# Add parent directory to path to import train module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import SlotFormer


def create_synthetic_dataset(num_images=20, img_size=224):
    """Create a dataset of diverse, synthetic images for analysis."""
    print("🎨 Creating diverse synthetic dataset...")
    synthetic_images = []
    
    for i in range(num_images):
        img = torch.zeros(3, img_size, img_size)
        label = "synthetic"
        
        # Type of synthetic image
        img_type = i % 5
        
        if img_type == 0: # --- Red Circle ---
            center_x, center_y = img_size // 2, img_size // 2
            radius = img_size // 4
            Y, X = np.ogrid[:img_size, :img_size]
            dist_from_center = np.sqrt((X - center_x)**2 + (Y-center_y)**2)
            mask = dist_from_center <= radius
            img[0, mask] = 1.0
            label = "red_circle"

        elif img_type == 1: # --- Green Square ---
            start = img_size // 4
            end = 3 * img_size // 4
            img[1, start:end, start:end] = 1.0
            label = "green_square"

        elif img_type == 2: # --- Blue Gradient ---
            Y, X = np.ogrid[:img_size, :img_size]
            gradient = X / img_size
            img[2, :, :] = torch.from_numpy(gradient)
            label = "blue_gradient"

        elif img_type == 3: # --- Checkerboard ---
            Y, X = np.ogrid[:img_size, :img_size]
            checkerboard = (X // 16 % 2) ^ (Y // 16 % 2)
            img[:, :, :] = torch.from_numpy(checkerboard).float()
            label = "checkerboard"

        elif img_type == 4: # --- Horizontal Stripes ---
            Y, X = np.ogrid[:img_size, :img_size]
            stripes = (Y // 20 % 2)
            img[0, :, :] = torch.from_numpy(stripes).float()
            img[1, :, :] = torch.from_numpy(1 - stripes).float()
            label = "stripes"

        synthetic_images.append((img, label))
        
    return synthetic_images


class FeatureSpaceAnalyzer:
    """Analyzes the feature space of a trained SlotFormer model."""
    
    def __init__(self, model_path, results_dir, device="cuda"):
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.model = self._load_model(model_path)
        self.pca = None
        print(f"FeatureSpaceAnalyzer initialized on {device}.")

    def _load_model(self, model_path):
        """Load a trained SlotFormer model."""
        # Note: Model parameters should match the saved model.
        model = SlotFormer(num_slots=128, num_layers=3, encoder_name="vit_base_patch16_dinov3")
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get('model', checkpoint)
            model.load_state_dict(state_dict)
            print(f"✓ Loaded model from {model_path}")
        else:
            print(f"⚠️ Model not found at '{model_path}'. Using random weights.")
            
        return model.to(self.device).eval()

    def _get_dataset(self, use_synthetic=True):
        """Load Imagenette and optionally combine with synthetic data."""
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        try:
            real_dataset = torchvision.datasets.Imagenette("./datasets", split="val", transform=transform, download=False)
            print(f"✓ Loaded Imagenette dataset with {len(real_dataset)} images.")
        except Exception as e:
            print(f"⚠️ Could not load Imagenette: {e}. Using synthetic data only.")
            real_dataset = []

        if use_synthetic:
            synthetic_dataset = create_synthetic_dataset()
            # We need to apply the same normalization to synthetic data
            normalizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            normalized_synthetic = [(normalizer(img), label) for img, label in synthetic_dataset]
            return list(real_dataset) + normalized_synthetic
        
        return list(real_dataset)

    def fit_pca(self, dataset, num_samples=100):
        """Fit PCA on a subset of the dataset's patch features."""
        print(f"Fitting PCA on {num_samples} samples...")
        all_features = []
        
        with torch.no_grad():
            for i in range(min(num_samples, len(dataset))):
                image, _ = dataset[i]
                image = image.unsqueeze(0).to(self.device)
                
                patch_tokens = self.model.encoder.forward_features(image)
                patch_tokens = patch_tokens[:, self.model.encoder.num_prefix_tokens:]
                
                # Sample patches to keep memory usage low
                sampled_patches = patch_tokens[0, ::4].cpu().numpy() # Every 4th patch
                all_features.append(sampled_patches)
        
        all_features = np.concatenate(all_features, axis=0)
        print(f"  Fitting PCA on {all_features.shape[0]} patch features...")
        
        self.pca = PCA(n_components=3)
        self.pca.fit(all_features)
        
        explained_var = self.pca.explained_variance_ratio_
        print(f"✓ PCA fitted. Explained variance: {explained_var.sum():.3f}")

    def features_to_rgb(self, patches):
        """Convert 768D patch features to a visualizable RGB image using PCA."""
        if self.pca is None:
            raise RuntimeError("PCA must be fitted before visualizing features. Call `fit_pca()`.")
        
        B, num_patches, embed_dim = patches.shape
        patches_flat = patches.view(-1, embed_dim).cpu().numpy()
        rgb_patches = self.pca.transform(patches_flat)
        
        # Robust normalization for better contrast
        for i in range(3):
            channel = rgb_patches[:, i]
            p5, p95 = np.percentile(channel, [5, 95])
            rgb_patches[:, i] = np.clip((channel - p5) / (p95 - p5 + 1e-6), 0, 1)
        
        patch_side_len = int(np.sqrt(num_patches))
        return rgb_patches.reshape(B, patch_side_len, patch_side_len, 3)

    def analyze_image(self, image, slot_counts):
        """Run the core analysis for a single image across different slot counts."""
        with torch.no_grad():
            gt_features = self.model.encoder.forward_features(image)
            gt_patches = gt_features[:, self.model.encoder.num_prefix_tokens:]
            
            # Generate all slots once
            all_slots = self.model.generator(
                tgt=self.model.slot_queries.repeat(1, 1, 1),
                memory=gt_patches,
                tgt_mask=self.model.tgt_mask.to(self.device)
            )
            
            results = {'losses': {}, 'similarities': {}, 'reconstructions': {}}
            
            for num_slots in slot_counts:
                used_slots = all_slots[:, :num_slots]
                
                # Reconstruct using the subset of slots
                recon_patches = self.model.reconstructor(
                    tgt=self.model.decoder_pos_embed.repeat(1, 1, 1),
                    memory=used_slots
                )
                
                results['losses'][num_slots] = F.mse_loss(recon_patches, gt_patches).item()
                
                gt_norm = F.normalize(gt_patches[0], dim=-1)
                recon_norm = F.normalize(recon_patches[0], dim=-1)
                results['similarities'][num_slots] = torch.sum(gt_norm * recon_norm, dim=-1).mean().item()
                
                results['reconstructions'][num_slots] = recon_patches
            
            return results, gt_patches

    def create_visualizations(self, image_idx, label, results, gt_patches, image, slot_counts):
        """Generate and save all plots for a single image analysis."""
        # Create main grid visualization
        fig, axes = plt.subplots(4, len(slot_counts), figsize=(len(slot_counts) * 4, 16))
        
        # Denormalize original image for viewing
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        orig_img_view = torch.clamp(image[0].cpu() * std + mean, 0, 1)
        gt_rgb = self.features_to_rgb(gt_patches)
        
        for i, num_slots in enumerate(slot_counts):
            axes[0, i].imshow(orig_img_view.permute(1, 2, 0))
            axes[0, i].set_title(f'Original Image\n(Label: {label})', fontsize=12)
            axes[0, i].axis('off')
            
            axes[1, i].imshow(gt_rgb[0])
            axes[1, i].set_title(f'GT Features (PCA)', fontsize=12)
            axes[1, i].axis('off')
            
            recon_rgb = self.features_to_rgb(results['reconstructions'][num_slots])
            axes[2, i].imshow(recon_rgb[0])
            loss = results['losses'][num_slots]
            sim = results['similarities'][num_slots]
            axes[2, i].set_title(f'Reconstruction ({num_slots} slots)\nLoss: {loss:.4f} | Sim: {sim:.3f}', fontsize=12)
            axes[2, i].axis('off')
            
            diff_rgb = self.features_to_rgb(torch.abs(results['reconstructions'][num_slots] - gt_patches))
            axes[3, i].imshow(diff_rgb[0])
            axes[3, i].set_title(f'Error Map', fontsize=12)
            axes[3, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'img_{image_idx}_{label}_grid.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

    def run_full_analysis(self, num_images=20, use_synthetic=True):
        """Run the complete analysis pipeline."""
        print("=== Starting Feature Space Analysis ===")
        dataset = self._get_dataset(use_synthetic=use_synthetic)
        if not dataset:
            print("❌ No data available. Aborting analysis.")
            return

        self.fit_pca(dataset, num_samples=min(100, len(dataset)))
        
        all_results = []
        slot_counts = [1, 2, 4, 8, 16, 32, 64, 128]
        
        print(f"\nAnalyzing {num_images} images...")
        for i in range(min(num_images, len(dataset))):
            image, label = dataset[i]
            print(f"Processing image {i+1}/{num_images} (Label: {label})...")
            image = image.unsqueeze(0).to(self.device)
            
            results, gt_patches = self.analyze_image(image, slot_counts)
            self.create_visualizations(i, label, results, gt_patches, image, slot_counts)
            
            all_results.append({'label': label, 'metrics': results})

        self._create_summary_plots(all_results, slot_counts)
        self._save_results_summary(all_results, slot_counts)
        
        print(f"\n✅ Analysis complete! Results saved to: {self.results_dir.resolve()}")

    def _create_summary_plots(self, all_results, slot_counts):
        """Generate and save summary plots across all analyzed images."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Space Analysis Summary', fontsize=16)

        # Average Loss Curve
        avg_losses = [np.mean([r['metrics']['losses'][s] for r in all_results]) for s in slot_counts]
        axes[0, 0].loglog(slot_counts, avg_losses, 'b-o')
        axes[0, 0].set_title('Average Reconstruction Loss vs. Slots')
        axes[0, 0].set_xlabel('Number of Slots')
        axes[0, 0].set_ylabel('MSE Loss (log scale)')
        axes[0, 0].grid(True, which="both", ls="--", alpha=0.5)

        # Average Similarity Curve
        avg_sims = [np.mean([r['metrics']['similarities'][s] for r in all_results]) for s in slot_counts]
        axes[0, 1].semilogx(slot_counts, avg_sims, 'g-s')
        axes[0, 1].set_title('Average Feature Similarity vs. Slots')
        axes[0, 1].set_xlabel('Number of Slots')
        axes[0, 1].set_ylabel('Cosine Similarity')
        axes[0, 1].grid(True, which="both", ls="--", alpha=0.5)
        axes[0, 1].set_ylim(0, 1)

        # Monotonic Ratio per Image
        monotonic_ratios = []
        for result in all_results:
            losses = [result['metrics']['losses'][s] for s in slot_counts]
            decreases = sum(1 for i in range(1, len(losses)) if losses[i] < losses[i-1])
            monotonic_ratios.append(decreases / (len(losses) - 1))
        
        axes[1, 0].bar(range(len(monotonic_ratios)), monotonic_ratios, color='purple')
        axes[1, 0].set_title('Monotonic Improvement Ratio per Image')
        axes[1, 0].set_xlabel('Image Index')
        axes[1, 0].set_ylabel('Monotonic Ratio')
        axes[1, 0].axhline(np.mean(monotonic_ratios), color='r', ls='--', label=f'Avg: {np.mean(monotonic_ratios):.3f}')
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0, 1.05)

        # Final vs. Initial Loss Improvement
        improvements = []
        for result in all_results:
            losses = result['metrics']['losses']
            improvements.append(losses[slot_counts[0]] / losses[slot_counts[-1]])

        axes[1, 1].bar(range(len(improvements)), improvements, color='orange')
        axes[1, 1].set_title('Loss Improvement Factor (1 vs 128 Slots)')
        axes[1, 1].set_xlabel('Image Index')
        axes[1, 1].set_ylabel('Improvement Factor (X times better)')
        axes[1, 1].axhline(np.mean(improvements), color='r', ls='--', label=f'Avg: {np.mean(improvements):.1f}x')
        axes[1, 1].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.results_dir / 'summary_plots.png', dpi=200)
        plt.close(fig)

    def _save_results_summary(self, all_results, slot_counts):
        """Save a JSON summary of the entire analysis."""
        monotonic_ratios = []
        for result in all_results:
            losses = [result['metrics']['losses'][s] for s in slot_counts]
            decreases = sum(1 for i in range(1, len(losses)) if losses[i] < losses[i-1])
            monotonic_ratios.append(decreases / (len(losses) - 1))

        summary_data = {
            'timestamp': time.time(),
            'model_path': self.model.name if hasattr(self.model, 'name') else 'N/A',
            'num_images_analyzed': len(all_results),
            'slot_counts': slot_counts,
            'pca_explained_variance': self.pca.explained_variance_ratio_.tolist(),
            'average_monotonic_ratio': np.mean(monotonic_ratios),
            'average_loss_improvement_factor': np.mean([r['metrics']['losses'][slot_counts[0]] / r['metrics']['losses'][slot_counts[-1]] for r in all_results]),
            'detailed_results': all_results
        }

        with open(self.results_dir / 'analysis_summary.json', 'w') as f:
            json.dump(summary_data, f, indent=2)
