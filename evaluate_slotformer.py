#!/usr/bin/env python3
"""
SlotFormer Evaluation Framework - Test the core hypothesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
from train import SlotFormer

class SlotFormerEvaluator:
    """Evaluate SlotFormer against baselines and ablations."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.model = self.load_model(model_path)
        self.results = {}
        
    def load_model(self, model_path: str):
        """Load trained SlotFormer model."""
        # Load model architecture
        model = SlotFormer(128, 3, "vit_base_patch16_dinov3")
        
        # Load weights if available
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            print(f"✓ Loaded model from {model_path}")
        else:
            print(f"⚠️  Model path {model_path} not found, using random weights")
            
        return model.to(self.device).eval()
    
    def create_random_baseline(self):
        """Create random baseline model for comparison."""
        class RandomBaseline(nn.Module):
            def __init__(self, num_slots, embed_dim, num_patches):
                super().__init__()
                self.num_slots = num_slots
                self.embed_dim = embed_dim
                self.num_patches = num_patches
                
            def forward(self, x, num_slots_used):
                B = x.shape[0]
                # Return random reconstructions
                return torch.randn(B, self.num_patches, self.embed_dim, device=x.device)
        
        return RandomBaseline(128, 768, 196).to(self.device)
    
    def evaluate_reconstruction_quality(self, dataloader, max_batches=10):
        """Evaluate reconstruction quality vs number of slots."""
        print("=== Evaluating Reconstruction Quality ===")
        
        slot_ranges = [1, 2, 4, 8, 16, 32, 64, 128]
        results = {
            'slotformer': {slots: [] for slots in slot_ranges},
            'random': {slots: [] for slots in slot_ranges}
        }
        
        random_model = self.create_random_baseline()
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(tqdm(dataloader)):
                if batch_idx >= max_batches:
                    break
                    
                images = images.to(self.device)
                B = images.shape[0]
                
                # Get ground truth features
                gt_features = self.model.encoder.forward_features(images)
                gt_patches = gt_features[:, self.model.encoder.num_prefix_tokens:]
                
                for num_slots in slot_ranges:
                    # SlotFormer reconstruction
                    try:
                        # Manually set number of slots for evaluation
                        slots = self.model.generator(
                            tgt=self.model.slot_queries[:, :num_slots].repeat(B, 1, 1),
                            memory=gt_patches,
                            tgt_mask=self.model.tgt_mask[:num_slots, :num_slots].to(images.device),
                            tgt_is_causal=True,
                        )
                        
                        recon = self.model.reconstructor(
                            tgt=self.model.decoder_pos_embed.repeat(B, 1, 1),
                            memory=slots,
                        )
                        
                        sf_loss = F.mse_loss(recon, gt_patches).item()
                        results['slotformer'][num_slots].append(sf_loss)
                        
                    except Exception as e:
                        print(f"SlotFormer error with {num_slots} slots: {e}")
                        results['slotformer'][num_slots].append(float('inf'))
                    
                    # Random baseline
                    random_recon = random_model(images, num_slots)
                    random_loss = F.mse_loss(random_recon, gt_patches).item()
                    results['random'][num_slots].append(random_loss)
        
        # Compute averages
        for model_name in results:
            for slots in slot_ranges:
                if results[model_name][slots]:
                    results[model_name][slots] = np.mean(results[model_name][slots])
                else:
                    results[model_name][slots] = float('inf')
        
        self.results['reconstruction_quality'] = results
        return results
    
    def evaluate_causal_ordering(self, dataloader, max_batches=10):
        """Test if more slots consistently improve reconstruction."""
        print("=== Evaluating Causal Ordering ===")
        
        slot_sequence = [1, 2, 4, 8, 16, 32, 64, 128]
        monotonic_improvements = []
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(tqdm(dataloader)):
                if batch_idx >= max_batches:
                    break
                    
                images = images.to(self.device)
                B = images.shape[0]
                
                # Get ground truth
                gt_features = self.model.encoder.forward_features(images)
                gt_patches = gt_features[:, self.model.encoder.num_prefix_tokens:]
                
                losses = []
                for num_slots in slot_sequence:
                    try:
                        # Generate slots
                        slots = self.model.generator(
                            tgt=self.model.slot_queries[:, :num_slots].repeat(B, 1, 1),
                            memory=gt_patches,
                            tgt_mask=self.model.tgt_mask[:num_slots, :num_slots].to(images.device),
                            tgt_is_causal=True,
                        )
                        
                        # Reconstruct
                        recon = self.model.reconstructor(
                            tgt=self.model.decoder_pos_embed.repeat(B, 1, 1),
                            memory=slots,
                        )
                        
                        loss = F.mse_loss(recon, gt_patches).item()
                        losses.append(loss)
                        
                    except Exception as e:
                        print(f"Error with {num_slots} slots: {e}")
                        losses.append(float('inf'))
                
                # Check monotonic improvement (loss should decrease)
                improvements = 0
                for i in range(1, len(losses)):
                    if losses[i] < losses[i-1]:  # Lower loss = better
                        improvements += 1
                
                monotonic_ratio = improvements / (len(losses) - 1)
                monotonic_improvements.append(monotonic_ratio)
        
        avg_monotonic = np.mean(monotonic_improvements)
        self.results['causal_ordering'] = {
            'monotonic_ratio': avg_monotonic,
            'individual_ratios': monotonic_improvements
        }
        
        print(f"Average monotonic improvement ratio: {avg_monotonic:.3f}")
        return avg_monotonic
    
    def create_visualizations(self, save_dir="./results"):
        """Create visualizations of results."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        if 'reconstruction_quality' in self.results:
            self.plot_reconstruction_quality(save_dir)
        
        if 'causal_ordering' in self.results:
            self.plot_causal_ordering(save_dir)
    
    def plot_reconstruction_quality(self, save_dir):
        """Plot reconstruction quality vs number of slots."""
        results = self.results['reconstruction_quality']
        
        plt.figure(figsize=(10, 6))
        
        slot_ranges = sorted(results['slotformer'].keys())
        sf_losses = [results['slotformer'][s] for s in slot_ranges]
        random_losses = [results['random'][s] for s in slot_ranges]
        
        plt.loglog(slot_ranges, sf_losses, 'b-o', label='SlotFormer', linewidth=2)
        plt.loglog(slot_ranges, random_losses, 'r--s', label='Random Baseline', linewidth=2)
        
        plt.xlabel('Number of Slots')
        plt.ylabel('Reconstruction Loss (MSE)')
        plt.title('SlotFormer: Reconstruction Quality vs Number of Slots')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_dir / 'reconstruction_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved reconstruction quality plot to {save_dir}/reconstruction_quality.png")
    
    def plot_causal_ordering(self, save_dir):
        """Plot causal ordering results."""
        results = self.results['causal_ordering']
        
        plt.figure(figsize=(10, 6))
        
        ratios = results['individual_ratios']
        plt.hist(ratios, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(results['monotonic_ratio'], color='red', linestyle='--', 
                   label=f'Average: {results["monotonic_ratio"]:.3f}')
        
        plt.xlabel('Monotonic Improvement Ratio')
        plt.ylabel('Frequency')
        plt.title('SlotFormer: Causal Ordering Validation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_dir / 'causal_ordering.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved causal ordering plot to {save_dir}/causal_ordering.png")
    
    def save_results(self, save_path="./results/evaluation_results.json"):
        """Save evaluation results to JSON."""
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✓ Saved results to {save_path}")
    
    def run_full_evaluation(self, model_path=None):
        """Run complete evaluation suite."""
        print("=== SlotFormer Evaluation Suite ===")
        
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
            "./datasets", split="val", transform=transform, download=True
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=False, num_workers=4
        )
        
        # Run evaluations
        self.evaluate_reconstruction_quality(dataloader)
        self.evaluate_causal_ordering(dataloader)
        
        # Create visualizations and save results
        self.create_visualizations()
        self.save_results()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print evaluation summary."""
        print("\n=== EVALUATION SUMMARY ===")
        
        if 'reconstruction_quality' in self.results:
            sf_results = self.results['reconstruction_quality']['slotformer']
            random_results = self.results['reconstruction_quality']['random']
            
            print("Reconstruction Quality:")
            print(f"  SlotFormer (128 slots): {sf_results.get(128, 'N/A'):.4f}")
            print(f"  Random Baseline (128 slots): {random_results.get(128, 'N/A'):.4f}")
            
            if sf_results.get(128) and random_results.get(128):
                improvement = random_results[128] / sf_results[128]
                print(f"  Improvement over random: {improvement:.2f}x")
        
        if 'causal_ordering' in self.results:
            monotonic = self.results['causal_ordering']['monotonic_ratio']
            print(f"Causal Ordering:")
            print(f"  Monotonic improvement ratio: {monotonic:.3f}")
            print(f"  Expected for good model: >0.8")
            print(f"  Status: {'✓ PASS' if monotonic > 0.8 else '✗ FAIL'}")

def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate SlotFormer')
    parser.add_argument('--model', type=str, default='model-00.ckpt', 
                       help='Path to trained model')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use')
    
    args = parser.parse_args()
    
    evaluator = SlotFormerEvaluator(args.model, args.device)
    evaluator.run_full_evaluation()

if __name__ == "__main__":
    main()
