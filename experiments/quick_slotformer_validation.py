#!/usr/bin/env python3
"""
Quick SlotFormer Validation Script
Minimal code to validate core SlotFormer concepts and compare with baselines.
"""

import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from train import SlotFormer

def quick_reconstruction_test():
    """Test reconstruction quality with different slot counts."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model
    model = SlotFormer(num_slots=128, num_layers=3, encoder_name="vit_base_patch16_dinov3.lvd_1689m")
    model = model.to(device).eval()
    
    # Load test image
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Use example images if available
    test_images = []
    for img_path in ['examples/city.jpg', 'examples/food.jpg', 'examples/highland.webp']:
        if Path(img_path).exists():
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
            test_images.append(transform(img).unsqueeze(0))
    
    if not test_images:
        # Generate random test image
        test_images = [torch.randn(1, 3, 224, 224)]
    
    results = {}
    slot_counts = [1, 2, 4, 8, 16, 32, 64, 128]
    
    with torch.no_grad():
        for i, test_img in enumerate(test_images):
            test_img = test_img.to(device)
            
            # Get patch tokens
            patch_tokens = model.encoder.forward_features(test_img)
            patch_tokens = patch_tokens[:, model.encoder.num_prefix_tokens:]
            
            # Generate all slots
            slots = model.generator(
                tgt=model.slot_queries.repeat(1, 1, 1),
                memory=patch_tokens,
                tgt_mask=model.tgt_mask.to(patch_tokens.device),
                tgt_is_causal=True,
            )
            
            img_results = []
            for num_slots in slot_counts:
                # Use first num_slots
                used_slots = slots[:, :num_slots]
                null_slots = model.null_slots[:, num_slots:].expand(1, -1, -1)
                masked_slots = torch.cat([used_slots, null_slots], dim=1)
                
                # Reconstruct
                reconstructed = model.reconstructor(
                    tgt=model.decoder_pos_embed.repeat(1, 1, 1),
                    memory=masked_slots,
                )
                
                # Calculate MSE
                mse = F.mse_loss(reconstructed, patch_tokens).item()
                img_results.append(mse)
                print(f"Image {i+1}, {num_slots} slots: MSE = {mse:.6f}")
            
            results[f'image_{i+1}'] = dict(zip(slot_counts, img_results))
    
    return results

def test_causal_ordering():
    """Test if causal ordering produces meaningful progression."""
    results = quick_reconstruction_test()
    
    print("\nCausal Ordering Analysis:")
    for img_name, img_results in results.items():
        slot_counts = list(img_results.keys())
        mse_values = list(img_results.values())
        
        # Check if MSE decreases (improvement)
        improvements = sum(1 for i in range(1, len(mse_values)) if mse_values[i] < mse_values[i-1])
        total_comparisons = len(mse_values) - 1
        monotonic_ratio = improvements / total_comparisons
        
        print(f"{img_name}: {improvements}/{total_comparisons} improvements ({monotonic_ratio:.2%})")
        
        # Calculate total improvement
        total_improvement = mse_values[0] / mse_values[-1] if mse_values[-1] > 0 else float('inf')
        print(f"{img_name}: Total improvement ratio: {total_improvement:.2f}x")
    
    return results

def compare_with_random_baseline():
    """Compare SlotFormer ordering with random slot ordering."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SlotFormer(num_slots=128, num_layers=3, encoder_name="vit_base_patch16_dinov3.lvd_1689m")
    model = model.to(device).eval()
    
    # Generate test data
    test_img = torch.randn(1, 3, 224, 224).to(device)
    
    with torch.no_grad():
        # Get slots
        patch_tokens = model.encoder.forward_features(test_img)
        patch_tokens = patch_tokens[:, model.encoder.num_prefix_tokens:]
        
        slots = model.generator(
            tgt=model.slot_queries.repeat(1, 1, 1),
            memory=patch_tokens,
            tgt_mask=model.tgt_mask.to(patch_tokens.device),
            tgt_is_causal=True,
        )
        
        # Test different orderings
        slot_counts = [4, 8, 16, 32]
        
        print("\nOrdering Comparison:")
        print("Slots | Causal MSE | Random MSE | Improvement")
        print("-" * 50)
        
        for num_slots in slot_counts:
            # Causal ordering (first N slots)
            causal_slots = slots[:, :num_slots]
            causal_null = model.null_slots[:, num_slots:].expand(1, -1, -1)
            causal_masked = torch.cat([causal_slots, causal_null], dim=1)
            
            causal_recon = model.reconstructor(
                tgt=model.decoder_pos_embed.repeat(1, 1, 1),
                memory=causal_masked,
            )
            causal_mse = F.mse_loss(causal_recon, patch_tokens).item()
            
            # Random ordering
            random_indices = torch.randperm(128)[:num_slots]
            random_slots = slots[:, random_indices]
            random_null = model.null_slots[:, num_slots:].expand(1, -1, -1)
            random_masked = torch.cat([random_slots, random_null], dim=1)
            
            random_recon = model.reconstructor(
                tgt=model.decoder_pos_embed.repeat(1, 1, 1),
                memory=random_masked,
            )
            random_mse = F.mse_loss(random_recon, patch_tokens).item()
            
            improvement = random_mse / causal_mse if causal_mse > 0 else 1.0
            print(f"{num_slots:5d} | {causal_mse:10.6f} | {random_mse:10.6f} | {improvement:8.2f}x")

def analyze_slot_diversity():
    """Analyze diversity and specialization of learned slots."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SlotFormer(num_slots=128, num_layers=3, encoder_name="vit_base_patch16_dinov3.lvd_1689m")
    model = model.to(device).eval()
    
    # Collect slot activations from multiple images
    all_slots = []
    
    with torch.no_grad():
        for _ in range(10):  # Test on 10 random images
            test_img = torch.randn(1, 3, 224, 224).to(device)
            
            patch_tokens = model.encoder.forward_features(test_img)
            patch_tokens = patch_tokens[:, model.encoder.num_prefix_tokens:]
            
            slots = model.generator(
                tgt=model.slot_queries.repeat(1, 1, 1),
                memory=patch_tokens,
                tgt_mask=model.tgt_mask.to(patch_tokens.device),
                tgt_is_causal=True,
            )
            
            all_slots.append(slots.cpu().numpy())
    
    all_slots = np.concatenate(all_slots, axis=0)  # [10, 128, embed_dim]
    
    # Calculate slot diversity metrics
    slot_means = np.mean(all_slots, axis=0)  # [128, embed_dim]
    
    # Pairwise cosine similarities
    similarities = []
    for i in range(slot_means.shape[0]):
        for j in range(i+1, slot_means.shape[0]):
            sim = np.dot(slot_means[i], slot_means[j]) / (
                np.linalg.norm(slot_means[i]) * np.linalg.norm(slot_means[j])
            )
            similarities.append(sim)
    
    print(f"\nSlot Diversity Analysis:")
    print(f"Mean pairwise similarity: {np.mean(similarities):.4f}")
    print(f"Std pairwise similarity: {np.std(similarities):.4f}")
    print(f"Min similarity: {np.min(similarities):.4f}")
    print(f"Max similarity: {np.max(similarities):.4f}")
    
    # Slot specialization (variance across samples)
    slot_variances = np.var(all_slots, axis=0)  # [128, embed_dim]
    specialization_scores = np.mean(slot_variances, axis=1)  # [128]
    
    print(f"Slot specialization scores:")
    print(f"Mean: {np.mean(specialization_scores):.6f}")
    print(f"Std: {np.std(specialization_scores):.6f}")
    print(f"Top 5 most specialized slots: {np.argsort(specialization_scores)[-5:]}")

def main():
    """Run quick validation experiments."""
    print("SlotFormer Quick Validation")
    print("=" * 50)
    
    # Test 1: Causal ordering
    print("\n1. Testing Causal Ordering...")
    ordering_results = test_causal_ordering()
    
    # Test 2: Compare with random baseline
    print("\n2. Comparing with Random Baseline...")
    compare_with_random_baseline()
    
    # Test 3: Slot diversity
    print("\n3. Analyzing Slot Diversity...")
    analyze_slot_diversity()
    
    # Save results
    with open('quick_validation_results.json', 'w') as f:
        json.dump(ordering_results, f, indent=2)
    
    print(f"\nResults saved to quick_validation_results.json")
    print("\nValidation complete! Check results to assess SlotFormer performance.")

if __name__ == "__main__":
    main()
