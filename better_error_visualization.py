#!/usr/bin/env python3
"""
Better error visualization for SlotFormer that shows meaningful patterns.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_error_magnitude_maps(gt_features, reconstructions, slot_counts, save_dir):
    """Visualize error as magnitude heatmaps instead of PCA colors."""
    
    fig, axes = plt.subplots(2, len(slot_counts), figsize=(24, 8))
    
    for i, num_slots in enumerate(slot_counts):
        recon = reconstructions[num_slots]
        
        # Compute L2 error magnitude per patch
        error_magnitude = torch.norm(recon - gt_features, dim=-1)[0]  # [196]
        error_map = error_magnitude.cpu().numpy().reshape(14, 14)
        
        # Compute cosine similarity per patch  
        gt_norm = F.normalize(gt_features[0], dim=-1)
        recon_norm = F.normalize(recon[0], dim=-1)
        similarity = torch.sum(gt_norm * recon_norm, dim=-1).cpu().numpy().reshape(14, 14)
        
        # Row 1: Error magnitude (red = high error)
        im1 = axes[0, i].imshow(error_map, cmap='Reds', vmin=0, vmax=error_map.max())
        axes[0, i].set_title(f'{num_slots} Slots\nL2 Error: {error_magnitude.mean():.4f}')
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
        
        # Row 2: Cosine similarity (green = high similarity)
        im2 = axes[1, i].imshow(similarity, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1, i].set_title(f'Cosine Similarity\nAvg: {similarity.mean():.3f}')
        axes[1, i].axis('off')
        plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'meaningful_error_maps.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_error_statistics(gt_features, reconstructions, slot_counts, save_dir):
    """Show error statistics instead of trying to visualize high-D errors."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Collect statistics
    total_errors = []
    patch_errors = []
    dim_errors = []
    similarities = []
    
    for num_slots in slot_counts:
        recon = reconstructions[num_slots]
        
        # Total reconstruction error
        total_error = F.mse_loss(recon, gt_features).item()
        total_errors.append(total_error)
        
        # Per-patch error variance
        patch_error_var = torch.var(torch.norm(recon - gt_features, dim=-1)).item()
        patch_errors.append(patch_error_var)
        
        # Per-dimension error variance  
        dim_error_var = torch.var(torch.mean((recon - gt_features)**2, dim=1)).item()
        dim_errors.append(dim_error_var)
        
        # Average cosine similarity
        gt_norm = F.normalize(gt_features[0], dim=-1)
        recon_norm = F.normalize(recon[0], dim=-1)
        avg_sim = torch.mean(torch.sum(gt_norm * recon_norm, dim=-1)).item()
        similarities.append(avg_sim)
    
    # Plot 1: Total error
    axes[0, 0].loglog(slot_counts, total_errors, 'r-o', linewidth=2)
    axes[0, 0].set_xlabel('Number of Slots')
    axes[0, 0].set_ylabel('Total MSE Loss')
    axes[0, 0].set_title('Overall Reconstruction Error')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Spatial error variance
    axes[0, 1].semilogx(slot_counts, patch_errors, 'b-s', linewidth=2)
    axes[0, 1].set_xlabel('Number of Slots')
    axes[0, 1].set_ylabel('Spatial Error Variance')
    axes[0, 1].set_title('Error Distribution Across Patches')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Feature dimension errors
    axes[1, 0].semilogx(slot_counts, dim_errors, 'g-^', linewidth=2)
    axes[1, 0].set_xlabel('Number of Slots')
    axes[1, 0].set_ylabel('Feature Dimension Error Variance')
    axes[1, 0].set_title('Error Distribution Across Feature Dimensions')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Cosine similarity
    axes[1, 1].semilogx(slot_counts, similarities, 'm-d', linewidth=2)
    axes[1, 1].set_xlabel('Number of Slots')
    axes[1, 1].set_ylabel('Average Cosine Similarity')
    axes[1, 1].set_title('Feature Direction Alignment')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'error_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'total_errors': total_errors,
        'similarities': similarities,
        'improvement': total_errors[0] / total_errors[-1],
        'final_similarity': similarities[-1]
    }

def analyze_error_patterns(gt_features, reconstructions, slot_counts, save_dir):
    """Analyze which types of errors decrease fastest."""
    
    # Analyze error in different frequency bands (rough approximation)
    low_freq_errors = []  # First 256 dimensions
    mid_freq_errors = []  # Middle 256 dimensions  
    high_freq_errors = [] # Last 256 dimensions
    
    for num_slots in slot_counts:
        recon = reconstructions[num_slots]
        error = (recon - gt_features) ** 2
        
        low_freq_error = torch.mean(error[:, :, :256]).item()
        mid_freq_error = torch.mean(error[:, :, 256:512]).item()
        high_freq_error = torch.mean(error[:, :, 512:]).item()
        
        low_freq_errors.append(low_freq_error)
        mid_freq_errors.append(mid_freq_error)
        high_freq_errors.append(high_freq_error)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.loglog(slot_counts, low_freq_errors, 'b-o', label='Low Freq (dims 0-255)', linewidth=2)
    plt.loglog(slot_counts, mid_freq_errors, 'g-s', label='Mid Freq (dims 256-511)', linewidth=2)
    plt.loglog(slot_counts, high_freq_errors, 'r-^', label='High Freq (dims 512-767)', linewidth=2)
    plt.xlabel('Number of Slots')
    plt.ylabel('Mean Squared Error')
    plt.title('Error by Feature Frequency Band')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Normalize to show relative improvement
    low_norm = np.array(low_freq_errors) / low_freq_errors[0]
    mid_norm = np.array(mid_freq_errors) / mid_freq_errors[0]
    high_norm = np.array(high_freq_errors) / high_freq_errors[0]
    
    plt.semilogx(slot_counts, low_norm, 'b-o', label='Low Freq', linewidth=2)
    plt.semilogx(slot_counts, mid_norm, 'g-s', label='Mid Freq', linewidth=2)
    plt.semilogx(slot_counts, high_norm, 'r-^', label='High Freq', linewidth=2)
    plt.xlabel('Number of Slots')
    plt.ylabel('Normalized Error (relative to 1 slot)')
    plt.title('Relative Error Improvement by Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'error_frequency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_meaningful_error_analysis(model, dataset, save_dir="./meaningful_error_analysis"):
    """Create meaningful error analysis instead of random-looking PCA projections."""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print("Creating meaningful error analysis...")
    
    # Analyze first few images
    for img_idx in range(3):
        print(f"Analyzing image {img_idx}...")
        
        image, _ = dataset[img_idx]
        image = image.unsqueeze(0).to(model.device)
        
        with torch.no_grad():
            # Get features and reconstructions
            gt_features = model.encoder.forward_features(image)
            gt_patches = gt_features[:, model.encoder.num_prefix_tokens:]
            
            slots = model.generator(
                tgt=model.slot_queries.repeat(1, 1, 1),
                memory=gt_patches,
                tgt_mask=model.tgt_mask.to(model.device),
                tgt_is_causal=True,
            )
            
            slot_counts = [1, 2, 4, 8, 16, 32, 64, 128]
            reconstructions = {}
            
            for num_slots in slot_counts:
                used_slots = slots[:, :num_slots]
                if num_slots < model.num_slots:
                    null_padding = model.null_slots[:, :model.num_slots-num_slots]
                    null_padding = null_padding.expand(1, -1, -1).type_as(slots)
                    padded_slots = torch.cat([used_slots, null_padding], dim=1)
                else:
                    padded_slots = used_slots
                
                recon = model.reconstructor(
                    tgt=model.decoder_pos_embed.repeat(1, 1, 1),
                    memory=padded_slots,
                )
                reconstructions[num_slots] = recon
            
            # Create meaningful visualizations
            visualize_error_magnitude_maps(gt_patches, reconstructions, slot_counts, 
                                         save_dir / f"img_{img_idx}")
            
            stats = visualize_error_statistics(gt_patches, reconstructions, slot_counts,
                                             save_dir / f"img_{img_idx}")
            
            analyze_error_patterns(gt_patches, reconstructions, slot_counts,
                                 save_dir / f"img_{img_idx}")
            
            print(f"  Improvement: {stats['improvement']:.2f}x")
            print(f"  Final similarity: {stats['final_similarity']:.3f}")
    
    print(f"✓ Meaningful error analysis saved to {save_dir}")

if __name__ == "__main__":
    print("This script provides better error visualization approaches.")
    print("The key insight: PCA visualization of 768D errors is inherently noisy.")
    print("Better approaches:")
    print("  1. Error magnitude heatmaps")
    print("  2. Statistical error analysis") 
    print("  3. Frequency band error analysis")
    print("  4. Focus on quantitative metrics (which are excellent!)")
