#!/usr/bin/env python3
"""
Comprehensive causal validation experiments.
Compares SlotFormer against multiple baselines to prove causal learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import sys
import os

# Add parent directory to path to import train module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import SlotFormer

class CausalValidationExperiments:
    """Comprehensive causal validation against baselines."""
    
    def __init__(self, device="cuda"):
        self.device = device
        self.results_dir = Path("./causal_validation_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def create_baseline_models(self):
        """Create various baseline models for comparison."""
        
        baselines = {}
        
        # 1. Random Baseline - slots in random order
        class RandomSlotFormer(SlotFormer):
            def forward_slots(self, patch_tokens, num_slots):
                device = patch_tokens.device
                # Generate slots normally
                slots = self.generator(
                    tgt=self.slot_queries[:, :num_slots].repeat(patch_tokens.size(0), 1, 1),
                    memory=patch_tokens,
                    tgt_mask=self.tgt_mask[:num_slots, :num_slots].to(device),
                    tgt_is_causal=True,
                )
                
                # RANDOMLY SHUFFLE SLOTS - breaks causal ordering
                B, N, D = slots.shape
                for b in range(B):
                    perm = torch.randperm(N)
                    slots[b] = slots[b, perm]
                
                return slots
        
        baselines['random'] = RandomSlotFormer(128, 3, "vit_base_patch16_dinov3")
        
        # 2. No Causal Mask - bidirectional attention
        class NoCausalSlotFormer(SlotFormer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Remove causal mask
                self.tgt_mask = None
            
            def forward_slots(self, patch_tokens, num_slots):
                slots = self.generator(
                    tgt=self.slot_queries[:, :num_slots].repeat(patch_tokens.size(0), 1, 1),
                    memory=patch_tokens,
                    tgt_mask=None,  # No causal masking
                    tgt_is_causal=False,
                )
                return slots
        
        baselines['no_causal'] = NoCausalSlotFormer(128, 3, "vit_base_patch16_dinov3")
        
        # 3. Fixed Order - slots always used in same order (no learning)
        class FixedOrderSlotFormer(SlotFormer):
            def forward_slots(self, patch_tokens, num_slots):
                # Always use first num_slots, regardless of content
                slots = self.slot_queries[:, :num_slots].repeat(patch_tokens.size(0), 1, 1)
                # No generator - just return fixed slots
                return slots
        
        baselines['fixed_order'] = FixedOrderSlotFormer(128, 3, "vit_base_patch16_dinov3")
        
        # 4. Reverse Causal - causal mask in reverse order
        class ReverseCausalSlotFormer(SlotFormer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Create reverse causal mask
                self.tgt_mask = torch.triu(torch.ones(self.num_slots, self.num_slots), diagonal=1).bool()
                # Reverse it
                self.tgt_mask = self.tgt_mask.flip(dims=[0, 1])
            
            def forward_slots(self, patch_tokens, num_slots):
                device = patch_tokens.device
                slots = self.generator(
                    tgt=self.slot_queries[:, :num_slots].repeat(patch_tokens.size(0), 1, 1),
                    memory=patch_tokens,
                    tgt_mask=self.tgt_mask[:num_slots, :num_slots].to(device),
                    tgt_is_causal=True,
                )
                return slots
        
        baselines['reverse_causal'] = ReverseCausalSlotFormer(128, 3, "vit_base_patch16_dinov3")
        
        # 5. SlotFormer (our method)
        baselines['slotformer'] = SlotFormer(128, 3, "vit_base_patch16_dinov3")
        
        return baselines
    
    def train_baseline_models(self, baselines, epochs=20, dataset_size=1000):
        """Train all baseline models on the same data."""
        print("=== Training Baseline Models ===")
        
        # Prepare dataset
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])
        
        try:
            full_dataset = torchvision.datasets.Imagenette(
                "./datasets", split="train", transform=transform, download=False
            )
            indices = torch.randperm(len(full_dataset))[:dataset_size]
            dataset = torch.utils.data.Subset(full_dataset, indices)
            print(f"✓ Using Imagenette training set: {dataset_size} samples")
        except Exception as e:
            print(f"⚠️ Imagenette not available ({e}), using synthetic dataset")
            dataset = [(torch.randn(3, 224, 224), i % 10) for i in range(dataset_size)]
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        
        training_results = {}
        
        for name, model in baselines.items():
            print(f"\nTraining {name}...")
            model = model.to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            model.train()
            losses = []
            
            for epoch in range(epochs):
                epoch_loss = 0
                count = 0
                
                for batch_idx, (images, _) in enumerate(dataloader):
                    if batch_idx >= 50:  # Limit for speed
                        break
                    
                    images = images.to(self.device)
                    
                    # Forward pass
                    patch_tokens = model.encoder.forward_features(images)
                    patch_tokens = patch_tokens[:, model.encoder.num_prefix_tokens:]
                    
                    # Random masking training
                    B, N, D = patch_tokens.shape
                    num_slots = torch.randint(1, model.num_slots + 1, (1,)).item()
                    
                    # Get slots (different for each baseline)
                    if hasattr(model, 'forward_slots'):
                        slots = model.forward_slots(patch_tokens, num_slots)
                    else:
                        # Standard SlotFormer
                        device = images.device
                        slots = model.generator(
                            tgt=model.slot_queries[:, :num_slots].repeat(B, 1, 1),
                            memory=patch_tokens,
                            tgt_mask=model.tgt_mask[:num_slots, :num_slots].to(device) if model.tgt_mask is not None else None,
                            tgt_is_causal=True,
                        )
                    
                    # Pad and reconstruct
                    if num_slots < model.num_slots:
                        null_padding = model.null_slots[:, :model.num_slots-num_slots]
                        null_padding = null_padding.expand(B, -1, -1).type_as(slots)
                        padded_slots = torch.cat([slots, null_padding], dim=1)
                    else:
                        padded_slots = slots
                    
                    reconstructed = model.reconstructor(
                        tgt=model.decoder_pos_embed.repeat(B, 1, 1),
                        memory=padded_slots,
                    )
                    
                    loss = F.mse_loss(reconstructed, patch_tokens)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    count += 1
                
                avg_loss = epoch_loss / count if count > 0 else 0
                losses.append(avg_loss)
                
                if epoch % 5 == 0:
                    print(f"  Epoch {epoch}: Loss = {avg_loss:.6f}")
            
            training_results[name] = {
                'losses': losses,
                'final_loss': losses[-1] if losses else float('inf')
            }
            
            # Save trained model
            torch.save(model.state_dict(), self.results_dir / f'{name}_model.pth')
        
        return training_results
    
    def evaluate_causal_properties(self, baselines, num_test_images=100):
        """Evaluate causal properties of all models."""
        print("=== Evaluating Causal Properties ===")
        
        # Load test dataset
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
            dataset = [(torch.randn(3, 224, 224), 0) for _ in range(num_test_images)]
        
        evaluation_results = {}
        slot_counts = [1, 2, 4, 8, 16, 32, 64, 128]
        
        for name, model in baselines.items():
            print(f"\nEvaluating {name}...")
            
            # Load trained weights
            try:
                model.load_state_dict(torch.load(self.results_dir / f'{name}_model.pth'))
            except:
                print(f"  ⚠️ Could not load weights for {name}")
            
            model = model.to(self.device).eval()
            
            monotonic_scores = []
            reconstruction_losses = []
            consistency_scores = []
            
            with torch.no_grad():
                for test_idx in range(min(num_test_images, len(dataset))):
                    if test_idx % 20 == 0:
                        print(f"  Processing image {test_idx}/{num_test_images}")
                    
                    image, _ = dataset[test_idx]
                    image = image.unsqueeze(0).to(self.device)
                    
                    patch_tokens = model.encoder.forward_features(image)
                    patch_tokens = patch_tokens[:, model.encoder.num_prefix_tokens:]
                    
                    # Test reconstruction with different slot counts
                    image_losses = []
                    
                    for num_slots in slot_counts:
                        # Get slots
                        if hasattr(model, 'forward_slots'):
                            slots = model.forward_slots(patch_tokens, num_slots)
                        else:
                            device = image.device
                            slots = model.generator(
                                tgt=model.slot_queries[:, :num_slots].repeat(1, 1, 1),
                                memory=patch_tokens,
                                tgt_mask=model.tgt_mask[:num_slots, :num_slots].to(device) if model.tgt_mask is not None else None,
                                tgt_is_causal=True,
                            )
                        
                        # Pad and reconstruct
                        if num_slots < model.num_slots:
                            null_padding = model.null_slots[:, :model.num_slots-num_slots]
                            null_padding = null_padding.expand(1, -1, -1).type_as(slots)
                            padded_slots = torch.cat([slots, null_padding], dim=1)
                        else:
                            padded_slots = slots
                        
                        reconstructed = model.reconstructor(
                            tgt=model.decoder_pos_embed.repeat(1, 1, 1),
                            memory=padded_slots,
                        )
                        
                        loss = F.mse_loss(reconstructed, patch_tokens).item()
                        image_losses.append(loss)
                    
                    # Calculate monotonic score
                    decreases = sum(1 for i in range(1, len(image_losses)) 
                                  if image_losses[i] < image_losses[i-1])
                    monotonic_score = decreases / (len(image_losses) - 1)
                    monotonic_scores.append(monotonic_score)
                    
                    # Store reconstruction losses
                    reconstruction_losses.append(image_losses)
                    
                    # Test consistency (same input should give same output)
                    if hasattr(model, 'forward_slots'):
                        slots1 = model.forward_slots(patch_tokens, 64)
                        slots2 = model.forward_slots(patch_tokens, 64)
                        consistency = F.cosine_similarity(slots1.flatten(), slots2.flatten(), dim=0).item()
                    else:
                        device = image.device
                        slots1 = model.generator(
                            tgt=model.slot_queries[:, :64].repeat(1, 1, 1),
                            memory=patch_tokens,
                            tgt_mask=model.tgt_mask[:64, :64].to(device) if model.tgt_mask is not None else None,
                            tgt_is_causal=True,
                        )
                        slots2 = model.generator(
                            tgt=model.slot_queries[:, :64].repeat(1, 1, 1),
                            memory=patch_tokens,
                            tgt_mask=model.tgt_mask[:64, :64].to(device) if model.tgt_mask is not None else None,
                            tgt_is_causal=True,
                        )
                        consistency = F.cosine_similarity(slots1.flatten(), slots2.flatten(), dim=0).item()
                    
                    consistency_scores.append(consistency)
            
            # Calculate aggregate metrics
            avg_monotonic = np.mean(monotonic_scores)
            std_monotonic = np.std(monotonic_scores)
            avg_consistency = np.mean(consistency_scores)
            
            # Calculate improvement ratio
            avg_losses = np.mean(reconstruction_losses, axis=0)
            improvement_ratio = avg_losses[0] / avg_losses[-1] if avg_losses[-1] > 0 else 1.0
            
            evaluation_results[name] = {
                'monotonic_score_mean': avg_monotonic,
                'monotonic_score_std': std_monotonic,
                'consistency_score': avg_consistency,
                'improvement_ratio': improvement_ratio,
                'avg_reconstruction_losses': avg_losses.tolist(),
                'individual_monotonic_scores': monotonic_scores
            }
            
            print(f"  Monotonic score: {avg_monotonic:.3f} ± {std_monotonic:.3f}")
            print(f"  Improvement ratio: {improvement_ratio:.2f}x")
            print(f"  Consistency score: {avg_consistency:.3f}")
        
        # Save results
        with open(self.results_dir / 'causal_evaluation.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Create comparison plots
        self.plot_causal_comparison(evaluation_results, slot_counts)
        
        return evaluation_results
    
    def plot_causal_comparison(self, results, slot_counts):
        """Create comprehensive comparison plots."""
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        models = list(results.keys())
        colors = ['red', 'orange', 'gray', 'purple', 'blue']  # SlotFormer in blue
        
        # 1. Monotonic scores comparison
        monotonic_means = [results[m]['monotonic_score_mean'] for m in models]
        monotonic_stds = [results[m]['monotonic_score_std'] for m in models]
        
        bars = axes[0, 0].bar(models, monotonic_means, yerr=monotonic_stds, 
                             color=colors, alpha=0.7, capsize=5)
        axes[0, 0].set_ylabel('Monotonic Score')
        axes[0, 0].set_title('Causal Learning Comparison')
        axes[0, 0].axhline(y=0.8, color='green', linestyle='--', label='Target (0.8)')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mean in zip(bars, monotonic_means):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Improvement ratios
        improvement_ratios = [results[m]['improvement_ratio'] for m in models]
        
        bars = axes[0, 1].bar(models, improvement_ratios, color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('Improvement Ratio')
        axes[0, 1].set_title('Reconstruction Improvement')
        axes[0, 1].set_yscale('log')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        for bar, ratio in zip(bars, improvement_ratios):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height * 1.1,
                           f'{ratio:.1f}x', ha='center', va='bottom', fontweight='bold')
        
        # 3. Consistency scores
        consistency_scores = [results[m]['consistency_score'] for m in models]
        
        bars = axes[0, 2].bar(models, consistency_scores, color=colors, alpha=0.7)
        axes[0, 2].set_ylabel('Consistency Score')
        axes[0, 2].set_title('Output Consistency')
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Reconstruction curves
        axes[1, 0].set_title('Reconstruction Quality Curves')
        for i, model in enumerate(models):
            losses = results[model]['avg_reconstruction_losses']
            axes[1, 0].loglog(slot_counts, losses, 'o-', color=colors[i], 
                             linewidth=2, label=model, markersize=6)
        axes[1, 0].set_xlabel('Number of Slots')
        axes[1, 0].set_ylabel('Reconstruction Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Monotonic score distributions
        axes[1, 1].set_title('Monotonic Score Distributions')
        for i, model in enumerate(models):
            scores = results[model]['individual_monotonic_scores']
            axes[1, 1].hist(scores, bins=20, alpha=0.6, label=model, color=colors[i])
        axes[1, 1].set_xlabel('Monotonic Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(x=0.8, color='green', linestyle='--', label='Target')
        axes[1, 1].legend()
        
        # 6. Summary ranking
        axes[1, 2].set_title('Overall Performance Ranking')
        
        # Calculate composite score
        composite_scores = []
        for model in models:
            score = (results[model]['monotonic_score_mean'] * 0.4 +
                    min(results[model]['improvement_ratio'] / 10, 1.0) * 0.3 +
                    results[model]['consistency_score'] * 0.3)
            composite_scores.append(score)
        
        # Sort by composite score
        sorted_indices = np.argsort(composite_scores)[::-1]
        sorted_models = [models[i] for i in sorted_indices]
        sorted_scores = [composite_scores[i] for i in sorted_indices]
        sorted_colors = [colors[i] for i in sorted_indices]
        
        bars = axes[1, 2].barh(range(len(sorted_models)), sorted_scores, 
                              color=sorted_colors, alpha=0.7)
        axes[1, 2].set_yticks(range(len(sorted_models)))
        axes[1, 2].set_yticklabels(sorted_models)
        axes[1, 2].set_xlabel('Composite Score')
        axes[1, 2].set_xlim(0, 1)
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            width = bar.get_width()
            axes[1, 2].text(width + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{score:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'causal_validation_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed statistical analysis
        self.create_statistical_analysis(results)
    
    def create_statistical_analysis(self, results):
        """Create detailed statistical analysis report."""
        
        report = []
        report.append("# SlotFormer Causal Validation Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary table
        report.append("## Summary Results")
        report.append("")
        report.append("| Model | Monotonic Score | Improvement | Consistency | Status |")
        report.append("|-------|----------------|-------------|-------------|---------|")
        
        for model in results.keys():
            mono = results[model]['monotonic_score_mean']
            improv = results[model]['improvement_ratio']
            consist = results[model]['consistency_score']
            
            if mono >= 0.8 and improv >= 2.0:
                status = "✅ EXCELLENT"
            elif mono >= 0.6 and improv >= 1.5:
                status = "⚠️ MODERATE"
            else:
                status = "❌ POOR"
            
            report.append(f"| {model} | {mono:.3f} ± {results[model]['monotonic_score_std']:.3f} | {improv:.2f}x | {consist:.3f} | {status} |")
        
        report.append("")
        report.append("## Key Findings")
        report.append("")
        
        # Find best model
        best_model = max(results.keys(), key=lambda m: results[m]['monotonic_score_mean'])
        best_score = results[best_model]['monotonic_score_mean']
        
        report.append(f"- **Best performing model**: {best_model} (monotonic score: {best_score:.3f})")
        
        # Compare SlotFormer to baselines
        if 'slotformer' in results:
            sf_score = results['slotformer']['monotonic_score_mean']
            baseline_scores = [results[m]['monotonic_score_mean'] for m in results.keys() if m != 'slotformer']
            avg_baseline = np.mean(baseline_scores)
            
            report.append(f"- **SlotFormer vs baselines**: {sf_score:.3f} vs {avg_baseline:.3f} (average)")
            report.append(f"- **Improvement over baselines**: {(sf_score - avg_baseline) / avg_baseline * 100:.1f}%")
        
        # Statistical significance
        report.append("")
        report.append("## Statistical Analysis")
        report.append("")
        
        for model in results.keys():
            scores = results[model]['individual_monotonic_scores']
            above_threshold = sum(1 for s in scores if s >= 0.8)
            percentage = above_threshold / len(scores) * 100
            
            report.append(f"- **{model}**: {above_threshold}/{len(scores)} images ({percentage:.1f}%) achieve monotonic score ≥ 0.8")
        
        # Save report
        with open(self.results_dir / 'causal_validation_report.md', 'w') as f:
            f.write('\n'.join(report))

def main():
    """Run causal validation experiments."""
    validator = CausalValidationExperiments()
    
    print("=== SlotFormer Causal Validation Experiments ===")
    
    # Create baseline models
    baselines = validator.create_baseline_models()
    print(f"Created {len(baselines)} baseline models: {list(baselines.keys())}")
    
    # Train all models
    training_results = validator.train_baseline_models(baselines)
    
    # Evaluate causal properties
    evaluation_results = validator.evaluate_causal_properties(baselines)
    
    print(f"\n✅ Causal validation complete!")
    print(f"📁 Results saved to: {validator.results_dir.absolute()}")

if __name__ == "__main__":
    main()
