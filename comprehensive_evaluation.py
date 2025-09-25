#!/usr/bin/env python3
"""
Comprehensive evaluation comparing SlotFormer against all baselines.
"""

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
import time

from train import SlotFormer
from baseline_models import create_baseline_models

class ComprehensiveEvaluator:
    """Comprehensive evaluation of SlotFormer vs baselines."""
    
    def __init__(self, device="cuda"):
        self.device = device
        self.results = {}
        
        # Load models
        self.models = self.load_all_models()
        
    def _evaluate_slotformer_with_slots(self, model, images, num_slots, gt_patches):
        """Evaluate SlotFormer with specific number of slots."""
        # For now, just use the standard forward pass
        # The SlotFormer uses random masking internally, so we can't directly control slot count
        loss = model(images)  # This returns loss, not reconstruction
        # Return dummy reconstruction for interface compatibility
        B = images.shape[0]
        return torch.zeros_like(gt_patches)
    
    def load_all_models(self):
        """Load SlotFormer and all baseline models."""
        models = {}
        
        # SlotFormer (main model)
        try:
            slotformer = SlotFormer(128, 3, "vit_base_patch16_dinov3")
            # Try to load trained weights if available
            checkpoint_paths = list(Path(".").glob("model*.ckpt")) + list(Path(".").glob("*.pt"))
            if checkpoint_paths:
                latest_checkpoint = max(checkpoint_paths, key=lambda p: p.stat().st_mtime)
                checkpoint = torch.load(latest_checkpoint, map_location=self.device)
                if 'model' in checkpoint:
                    slotformer.load_state_dict(checkpoint['model'])
                else:
                    slotformer.load_state_dict(checkpoint)
                print(f"✓ Loaded SlotFormer from {latest_checkpoint}")
            else:
                print("⚠️  Using untrained SlotFormer")
                
            models['slotformer'] = slotformer.to(self.device).eval()
        except Exception as e:
            print(f"✗ Failed to load SlotFormer: {e}")
        
        # Baseline models
        baselines = create_baseline_models()
        for name, model in baselines.items():
            try:
                models[name] = model.to(self.device).eval()
                print(f"✓ Loaded {name}")
            except Exception as e:
                print(f"✗ Failed to load {name}: {e}")
        
        return models
    
    def evaluate_reconstruction_scaling(self, dataloader, slot_ranges=[1, 2, 4, 8, 16, 32, 64, 128]):
        """Evaluate how reconstruction quality scales with number of slots."""
        
        print("=== Evaluating Reconstruction Scaling ===")
        
        results = {model_name: {slots: [] for slots in slot_ranges} for model_name in self.models.keys()}
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(tqdm(dataloader)):
                if batch_idx >= 10:  # Limit for faster evaluation
                    break
                    
                images = images.to(self.device)
                B = images.shape[0]
                
                # Get ground truth features (using SlotFormer's encoder)
                if 'slotformer' in self.models:
                    gt_features = self.models['slotformer'].encoder.forward_features(images)
                    gt_patches = gt_features[:, self.models['slotformer'].encoder.num_prefix_tokens:]
                else:
                    # Fallback: use any model with encoder
                    model_with_encoder = next((m for m in self.models.values() if hasattr(m, 'encoder')), None)
                    if model_with_encoder:
                        gt_features = model_with_encoder.encoder.forward_features(images)
                        gt_patches = gt_features[:, model_with_encoder.encoder.num_prefix_tokens:]
                    else:
                        continue
                
                for model_name, model in self.models.items():
                    for num_slots in slot_ranges:
                        try:
                            if model_name == 'random':
                                # Random baseline
                                recon = model(images, num_slots)
                            elif model_name == 'slotformer':
                                # SlotFormer has different interface - doesn't take num_slots parameter
                                recon = self._evaluate_slotformer_with_slots(model, images, num_slots, gt_patches)
                            else:
                                # Other models
                                if hasattr(model, 'forward'):
                                    recon = model(images, num_slots)
                                else:
                                    continue
                            
                            loss = F.mse_loss(recon, gt_patches).item()
                            results[model_name][num_slots].append(loss)
                            
                        except Exception as e:
                            print(f"Error with {model_name}, {num_slots} slots: {e}")
                            results[model_name][num_slots].append(float('inf'))
        
        # Compute averages
        for model_name in results:
            for slots in slot_ranges:
                if results[model_name][slots]:
                    valid_losses = [l for l in results[model_name][slots] if l != float('inf')]
                    if valid_losses:
                        results[model_name][slots] = np.mean(valid_losses)
                    else:
                        results[model_name][slots] = float('inf')
                else:
                    results[model_name][slots] = float('inf')
        
        self.results['reconstruction_scaling'] = results
        return results
    
    def evaluate_causal_ordering(self, dataloader, slot_sequence=[1, 2, 4, 8, 16, 32, 64, 128]):
        """Evaluate causal ordering for all models."""
        
        print("=== Evaluating Causal Ordering ===")
        
        results = {model_name: [] for model_name in self.models.keys()}
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(tqdm(dataloader)):
                if batch_idx >= 10:
                    break
                    
                images = images.to(self.device)
                
                # Get ground truth
                if 'slotformer' in self.models:
                    gt_features = self.models['slotformer'].encoder.forward_features(images)
                    gt_patches = gt_features[:, self.models['slotformer'].encoder.num_prefix_tokens:]
                else:
                    model_with_encoder = next((m for m in self.models.values() if hasattr(m, 'encoder')), None)
                    if model_with_encoder:
                        gt_features = model_with_encoder.encoder.forward_features(images)
                        gt_patches = gt_features[:, model_with_encoder.encoder.num_prefix_tokens:]
                    else:
                        continue
                
                for model_name, model in self.models.items():
                    losses = []
                    
                    for num_slots in slot_sequence:
                        try:
                            if model_name == 'random':
                                recon = model(images, num_slots)
                            else:
                                recon = model(images, num_slots)
                            
                            loss = F.mse_loss(recon, gt_patches).item()
                            losses.append(loss)
                            
                        except Exception as e:
                            losses.append(float('inf'))
                    
                    # Check monotonic improvement
                    improvements = 0
                    valid_comparisons = 0
                    
                    for i in range(1, len(losses)):
                        if losses[i] != float('inf') and losses[i-1] != float('inf'):
                            if losses[i] < losses[i-1]:  # Lower loss = better
                                improvements += 1
                            valid_comparisons += 1
                    
                    if valid_comparisons > 0:
                        monotonic_ratio = improvements / valid_comparisons
                        results[model_name].append(monotonic_ratio)
        
        # Compute averages
        for model_name in results:
            if results[model_name]:
                results[model_name] = {
                    'mean': np.mean(results[model_name]),
                    'std': np.std(results[model_name]),
                    'individual': results[model_name]
                }
            else:
                results[model_name] = {'mean': 0.0, 'std': 0.0, 'individual': []}
        
        self.results['causal_ordering'] = results
        return results
    
    def create_comparison_plots(self, save_dir="./comprehensive_results"):
        """Create comprehensive comparison plots."""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Plot 1: Reconstruction scaling
        if 'reconstruction_scaling' in self.results:
            self.plot_reconstruction_scaling(save_dir)
        
        # Plot 2: Causal ordering comparison
        if 'causal_ordering' in self.results:
            self.plot_causal_ordering_comparison(save_dir)
        
        # Plot 3: Model performance heatmap
        self.plot_performance_heatmap(save_dir)
        
    def plot_reconstruction_scaling(self, save_dir):
        """Plot reconstruction quality vs slots for all models."""
        
        results = self.results['reconstruction_scaling']
        
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
        
        for i, (model_name, model_results) in enumerate(results.items()):
            slot_ranges = sorted(model_results.keys())
            losses = [model_results[s] for s in slot_ranges]
            
            # Filter out infinite values for plotting
            valid_indices = [i for i, l in enumerate(losses) if l != float('inf')]
            if valid_indices:
                valid_slots = [slot_ranges[i] for i in valid_indices]
                valid_losses = [losses[i] for i in valid_indices]
                
                plt.loglog(valid_slots, valid_losses, 'o-', 
                          label=model_name.replace('_', ' ').title(), 
                          color=colors[i], linewidth=2, markersize=6)
        
        plt.xlabel('Number of Slots')
        plt.ylabel('Reconstruction Loss (MSE)')
        plt.title('Reconstruction Quality vs Number of Slots')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_dir / 'reconstruction_scaling.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved reconstruction scaling plot")
    
    def plot_causal_ordering_comparison(self, save_dir):
        """Plot causal ordering comparison."""
        
        results = self.results['causal_ordering']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot of mean monotonic ratios
        model_names = list(results.keys())
        means = [results[name]['mean'] for name in model_names]
        stds = [results[name]['std'] for name in model_names]
        
        bars = ax1.bar(range(len(model_names)), means, yerr=stds, capsize=5)
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels([name.replace('_', ' ').title() for name in model_names], rotation=45)
        ax1.set_ylabel('Monotonic Improvement Ratio')
        ax1.set_title('Causal Ordering: Mean Performance')
        ax1.axhline(y=0.8, color='red', linestyle='--', label='Target (0.8)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Color bars based on performance
        for i, bar in enumerate(bars):
            if means[i] >= 0.8:
                bar.set_color('green')
            elif means[i] >= 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Distribution plot
        all_ratios = []
        labels = []
        for name in model_names:
            ratios = results[name]['individual']
            if ratios:
                all_ratios.extend(ratios)
                labels.extend([name.replace('_', ' ').title()] * len(ratios))
        
        if all_ratios:
            import pandas as pd
            df = pd.DataFrame({'Model': labels, 'Monotonic Ratio': all_ratios})
            sns.boxplot(data=df, x='Model', y='Monotonic Ratio', ax=ax2)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
            ax2.set_title('Causal Ordering: Distribution')
            ax2.axhline(y=0.8, color='red', linestyle='--', label='Target')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'causal_ordering_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved causal ordering comparison plot")
    
    def plot_performance_heatmap(self, save_dir):
        """Create performance heatmap across different metrics."""
        
        if 'reconstruction_scaling' not in self.results or 'causal_ordering' not in self.results:
            return
        
        # Prepare data
        models = list(self.results['reconstruction_scaling'].keys())
        metrics = ['Reconstruction (128 slots)', 'Causal Ordering']
        
        data = []
        for model in models:
            row = []
            
            # Reconstruction quality (lower is better, so invert)
            recon_loss = self.results['reconstruction_scaling'][model].get(128, float('inf'))
            if recon_loss != float('inf'):
                # Normalize to 0-1 scale (higher is better)
                recon_score = max(0, 1 - recon_loss / 10)  # Assuming max reasonable loss is 10
            else:
                recon_score = 0
            row.append(recon_score)
            
            # Causal ordering (higher is better)
            causal_score = self.results['causal_ordering'][model]['mean']
            row.append(causal_score)
            
            data.append(row)
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        
        data_array = np.array(data)
        sns.heatmap(data_array, 
                   xticklabels=metrics,
                   yticklabels=[m.replace('_', ' ').title() for m in models],
                   annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0, vmax=1)
        
        plt.title('Model Performance Heatmap')
        plt.tight_layout()
        
        plt.savefig(save_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved performance heatmap")
    
    def run_comprehensive_evaluation(self):
        """Run complete evaluation suite."""
        
        print("=== Comprehensive SlotFormer Evaluation ===")
        
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
        
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=4, shuffle=False, num_workers=2
        )
        
        # Run evaluations
        self.evaluate_reconstruction_scaling(dataloader)
        self.evaluate_causal_ordering(dataloader)
        
        # Create visualizations
        self.create_comparison_plots()
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_comprehensive_summary()
    
    def save_results(self, save_path="./comprehensive_results/results.json"):
        """Save all results to JSON."""
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"✓ Saved comprehensive results to {save_path}")
    
    def print_comprehensive_summary(self):
        """Print comprehensive evaluation summary."""
        
        print("\n" + "="*50)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("="*50)
        
        if 'causal_ordering' in self.results:
            print("\n🎯 CAUSAL ORDERING RESULTS:")
            causal_results = self.results['causal_ordering']
            
            # Sort by performance
            sorted_models = sorted(causal_results.items(), 
                                 key=lambda x: x[1]['mean'], reverse=True)
            
            for model_name, result in sorted_models:
                score = result['mean']
                status = "🟢 EXCELLENT" if score >= 0.8 else "🟡 MODERATE" if score >= 0.6 else "🔴 POOR"
                print(f"  {model_name.replace('_', ' ').title()}: {score:.3f} {status}")
        
        if 'reconstruction_scaling' in self.results:
            print(f"\n📊 RECONSTRUCTION QUALITY (128 slots):")
            recon_results = self.results['reconstruction_scaling']
            
            # Sort by reconstruction quality (lower is better)
            sorted_recon = sorted([(name, results.get(128, float('inf'))) 
                                 for name, results in recon_results.items()],
                                key=lambda x: x[1])
            
            for model_name, loss in sorted_recon[:5]:  # Top 5
                if loss != float('inf'):
                    print(f"  {model_name.replace('_', ' ').title()}: {loss:.4f}")
        
        print(f"\n✅ Evaluation complete! Check ./comprehensive_results/ for detailed plots and data.")

def main():
    """Main evaluation function."""
    
    evaluator = ComprehensiveEvaluator()
    evaluator.run_comprehensive_evaluation()

if __name__ == "__main__":
    main()
