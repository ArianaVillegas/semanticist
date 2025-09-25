#!/usr/bin/env python3
"""
Quick local visualization of SlotFormer (no cluster needed).
"""

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from train import SlotFormer

def quick_slot_visualization(model_path="model-49.ckpt", num_images=3):
    """Quick visualization that can run locally."""
    
    print("=== Quick SlotFormer Visualization ===")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = SlotFormer(128, 3, "vit_base_patch16_dinov3")
    
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Loaded model from {model_path}")
    else:
        print(f"⚠️ Model not found: {model_path}, using random weights")
    
    model = model.to(device).eval()
    
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
        print(f"✓ Loaded dataset with {len(dataset)} images")
    except:
        print("⚠️ Dataset not found, creating dummy data")
        # Create dummy data for testing
        dummy_images = torch.randn(10, 3, 224, 224)
        dummy_labels = torch.zeros(10)
        dataset = list(zip(dummy_images, dummy_labels))
    
    # Create results directory
    results_dir = Path("./quick_visualizations")
    results_dir.mkdir(exist_ok=True)
    
    # Test different slot counts
    slot_counts = [1, 4, 16, 64, 128]
    
    print(f"Analyzing {num_images} images with slot counts: {slot_counts}")
    
    all_results = []
    
    with torch.no_grad():
        for img_idx in range(min(num_images, len(dataset))):
            print(f"Processing image {img_idx + 1}/{num_images}...")
            
            if isinstance(dataset[img_idx], tuple):
                image, label = dataset[img_idx]
            else:
                image, label = dataset[img_idx], 0
                
            image = image.unsqueeze(0).to(device)
            
            # Get patch tokens
            patch_tokens = model.encoder.forward_features(image)
            patch_tokens = patch_tokens[:, model.encoder.num_prefix_tokens:]
            
            # Generate all slots
            slots = model.generator(
                tgt=model.slot_queries.repeat(1, 1, 1),
                memory=patch_tokens,
                tgt_mask=model.tgt_mask.to(device),
                tgt_is_causal=True,
            )
            
            # Test different slot counts
            losses = {}
            
            for num_slots in slot_counts:
                # Use only first num_slots
                used_slots = slots[:, :num_slots]
                
                # Pad with NULL tokens
                if num_slots < model.num_slots:
                    null_padding = model.null_slots[:, :model.num_slots-num_slots]
                    null_padding = null_padding.expand(1, -1, -1).type_as(slots)
                    padded_slots = torch.cat([used_slots, null_padding], dim=1)
                else:
                    padded_slots = used_slots
                
                # Reconstruct
                recon = model.reconstructor(
                    tgt=model.decoder_pos_embed.repeat(1, 1, 1),
                    memory=padded_slots,
                )
                
                # Compute loss
                loss = F.mse_loss(recon, patch_tokens).item()
                losses[num_slots] = loss
            
            all_results.append(losses)
            print(f"  Losses: {losses}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot individual images
    for i, losses in enumerate(all_results):
        loss_values = [losses[s] for s in slot_counts]
        plt.loglog(slot_counts, loss_values, 'o-', alpha=0.7, label=f'Image {i+1}')
    
    # Plot average
    if all_results:
        avg_losses = []
        for s in slot_counts:
            avg_loss = np.mean([result[s] for result in all_results])
            avg_losses.append(avg_loss)
        
        plt.loglog(slot_counts, avg_losses, 'k-', linewidth=3, 
                  marker='s', markersize=8, label='Average')
    
    plt.xlabel('Number of Slots', fontsize=12)
    plt.ylabel('Reconstruction Loss (MSE)', fontsize=12)
    plt.title('SlotFormer: Reconstruction Quality vs Number of Slots', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text box with key insights
    if all_results:
        improvement_ratio = avg_losses[0] / avg_losses[-1]  # First vs last
        textstr = f'Improvement: {improvement_ratio:.1f}x\n'
        textstr += f'Best loss: {min(avg_losses):.4f}\n'
        textstr += f'Worst loss: {max(avg_losses):.4f}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'quick_slot_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create slot activation heatmap
    if all_results:
        plt.figure(figsize=(10, 6))
        
        # Show how loss decreases with more slots
        slot_data = np.array([[result[s] for s in slot_counts] for result in all_results])
        
        plt.subplot(1, 2, 1)
        plt.imshow(slot_data, aspect='auto', cmap='viridis')
        plt.colorbar(label='Reconstruction Loss')
        plt.xlabel('Slot Count Index')
        plt.ylabel('Image Index')
        plt.title('Loss Heatmap')
        plt.xticks(range(len(slot_counts)), slot_counts)
        
        plt.subplot(1, 2, 2)
        # Show monotonic improvement
        improvements = []
        for result in all_results:
            loss_values = [result[s] for s in slot_counts]
            # Count how many times loss decreases
            decreases = sum(1 for i in range(1, len(loss_values)) 
                          if loss_values[i] < loss_values[i-1])
            monotonic_ratio = decreases / (len(loss_values) - 1)
            improvements.append(monotonic_ratio)
        
        plt.bar(range(len(improvements)), improvements)
        plt.xlabel('Image Index')
        plt.ylabel('Monotonic Improvement Ratio')
        plt.title('Causal Ordering Quality')
        plt.axhline(y=0.8, color='red', linestyle='--', label='Target (0.8)')
        plt.legend()
        plt.ylim(0, 1)
        
        avg_monotonic = np.mean(improvements)
        plt.text(0.5, 0.95, f'Average: {avg_monotonic:.3f}', 
                transform=plt.gca().transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(results_dir / 'quick_causal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\n✅ Quick visualization complete!")
    print(f"📁 Results saved to: {results_dir.absolute()}")
    print(f"📊 Key findings:")
    
    if all_results:
        print(f"   - Average improvement: {improvement_ratio:.1f}x")
        print(f"   - Average monotonic ratio: {avg_monotonic:.3f}")
        print(f"   - Status: {'✓ PASS' if avg_monotonic > 0.8 else '⚠️ MODERATE' if avg_monotonic > 0.6 else '✗ FAIL'}")
    
    return all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick SlotFormer visualization')
    parser.add_argument('--model', type=str, default='model-49.ckpt', 
                       help='Path to trained model')
    parser.add_argument('--num_images', type=int, default=5,
                       help='Number of images to analyze')
    
    args = parser.parse_args()
    
    quick_slot_visualization(args.model, args.num_images)
