#!/usr/bin/env python3
"""
SlotFormer Experiment Pipeline
Complete experimental pipeline for validating SlotFormer approach.
"""

import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import SlotFormer

class SlotFormerExperiments:
    """Run comprehensive SlotFormer validation experiments."""
    
    def __init__(self, data_dir="./datasets", device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.data_dir = Path(data_dir)
        self.results = {}
        
        print(f"Using device: {self.device}")
        
    def setup_model(self, model_path=None):
        """Setup SlotFormer model."""
        self.model = SlotFormer(
            num_slots=128, 
            num_layers=3, 
            encoder_name="vit_base_patch16_dinov3"
        ).to(self.device).eval()
        
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            print(f"Loaded model from {model_path}")
        else:
            print("Using untrained model (for architecture validation)")
    
    def get_datasets(self):
        """Get validation datasets."""
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        datasets = {}
        
        # ImageNette
        try:
            datasets['imagenette'] = torchvision.datasets.Imagenette(
                str(self.data_dir), split='val', transform=transform
            )
        except:
            print("ImageNette not found - run scripts/download_validation_datasets.py")
        
        # CIFAR-10
        try:
            datasets['cifar10'] = torchvision.datasets.CIFAR10(
                str(self.data_dir), train=False, transform=transform
            )
        except:
            print("CIFAR-10 not found - run scripts/download_validation_datasets.py")
        
        # STL-10
        try:
            datasets['stl10'] = torchvision.datasets.STL10(
                str(self.data_dir), split='test', transform=transform
            )
        except:
            print("STL-10 not found - run scripts/download_validation_datasets.py")
        
        return {k: v for k, v in datasets.items() if v is not None}
    
    def experiment_1_reconstruction_scaling(self, datasets, slot_counts=[1, 2, 4, 8, 16, 32, 64, 128]):
        """Test reconstruction quality vs slot count."""
        print("\n=== Experiment 1: Reconstruction Scaling ===")
        
        results = {}
        
        for dataset_name, dataset in datasets.items():
            print(f"\nTesting {dataset_name} ({len(dataset)} samples)")
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=8, shuffle=False, num_workers=2
            )
            
            dataset_results = {}
            
            for num_slots in slot_counts:
                print(f"  Testing {num_slots} slots...")
                mse_scores = []
                
                with torch.no_grad():
                    for i, (images, _) in enumerate(dataloader):
                        if i >= 5:  # Test on 5 batches
                            break
                        
                        images = images.to(self.device)
                        
                        # Get patch tokens
                        patch_tokens = self.model.encoder.forward_features(images)
                        patch_tokens = patch_tokens[:, self.model.encoder.num_prefix_tokens:]
                        
                        # Generate slots
                        slots = self.model.generator(
                            tgt=self.model.slot_queries.repeat(images.shape[0], 1, 1),
                            memory=patch_tokens,
                            tgt_mask=self.model.tgt_mask.to(patch_tokens.device),
                            tgt_is_causal=True,
                        )
                        
                        # Use first num_slots
                        used_slots = slots[:, :num_slots]
                        if num_slots < 128:
                            null_slots = self.model.null_slots[:, num_slots:].expand(images.shape[0], -1, -1)
                            masked_slots = torch.cat([used_slots, null_slots], dim=1)
                        else:
                            masked_slots = used_slots
                        
                        # Reconstruct
                        reconstructed = self.model.reconstructor(
                            tgt=self.model.decoder_pos_embed.repeat(images.shape[0], 1, 1),
                            memory=masked_slots,
                        )
                        
                        # Calculate MSE
                        mse = F.mse_loss(reconstructed, patch_tokens, reduction='none').mean(dim=[1,2])
                        mse_scores.extend(mse.cpu().numpy())
                
                dataset_results[num_slots] = {
                    'mse_mean': float(np.mean(mse_scores)),
                    'mse_std': float(np.std(mse_scores))
                }
                print(f"    MSE: {np.mean(mse_scores):.6f} ± {np.std(mse_scores):.6f}")
            
            results[dataset_name] = dataset_results
        
        self.results['reconstruction_scaling'] = results
        return results
    
    def experiment_2_causal_ordering_validation(self, datasets):
        """Validate causal ordering produces meaningful progression."""
        print("\n=== Experiment 2: Causal Ordering Validation ===")
        
        results = {}
        
        for dataset_name, dataset in datasets.items():
            print(f"\nTesting {dataset_name}")
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=4, shuffle=False, num_workers=2
            )
            
            slot_counts = [1, 2, 4, 8, 16, 32, 64, 128]
            progression_scores = []
            
            with torch.no_grad():
                for i, (images, _) in enumerate(dataloader):
                    if i >= 3:  # Test on 3 batches
                        break
                    
                    images = images.to(self.device)
                    
                    # Get reconstruction scores for different slot counts
                    batch_scores = []
                    for num_slots in slot_counts:
                        patch_tokens = self.model.encoder.forward_features(images)
                        patch_tokens = patch_tokens[:, self.model.encoder.num_prefix_tokens:]
                        
                        slots = self.model.generator(
                            tgt=self.model.slot_queries.repeat(images.shape[0], 1, 1),
                            memory=patch_tokens,
                            tgt_mask=self.model.tgt_mask.to(patch_tokens.device),
                            tgt_is_causal=True,
                        )
                        
                        used_slots = slots[:, :num_slots]
                        if num_slots < 128:
                            null_slots = self.model.null_slots[:, num_slots:].expand(images.shape[0], -1, -1)
                            masked_slots = torch.cat([used_slots, null_slots], dim=1)
                        else:
                            masked_slots = used_slots
                        
                        reconstructed = self.model.reconstructor(
                            tgt=self.model.decoder_pos_embed.repeat(images.shape[0], 1, 1),
                            memory=masked_slots,
                        )
                        
                        mse = F.mse_loss(reconstructed, patch_tokens).item()
                        batch_scores.append(mse)
                    
                    progression_scores.append(batch_scores)
            
            # Analyze progression
            progression_array = np.array(progression_scores)
            mean_progression = np.mean(progression_array, axis=0)
            
            # Check monotonicity (lower MSE = better)
            improvements = sum(1 for i in range(1, len(mean_progression)) 
                             if mean_progression[i] < mean_progression[i-1])
            monotonic_ratio = improvements / (len(mean_progression) - 1)
            
            results[dataset_name] = {
                'progression': mean_progression.tolist(),
                'slot_counts': slot_counts,
                'monotonic_ratio': monotonic_ratio,
                'total_improvement': float(mean_progression[0] / mean_progression[-1])
            }
            
            print(f"  Monotonic improvements: {improvements}/{len(mean_progression)-1} ({monotonic_ratio:.2%})")
            print(f"  Total improvement: {mean_progression[0]/mean_progression[-1]:.2f}x")
        
        self.results['causal_ordering'] = results
        return results
    
    def experiment_3_random_baseline_comparison(self, datasets):
        """Compare causal ordering vs random slot selection."""
        print("\n=== Experiment 3: Random Baseline Comparison ===")
        
        results = {}
        slot_counts = [4, 8, 16, 32]
        
        for dataset_name, dataset in datasets.items():
            print(f"\nTesting {dataset_name}")
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=4, shuffle=False, num_workers=2
            )
            
            dataset_results = {}
            
            with torch.no_grad():
                for num_slots in slot_counts:
                    causal_scores = []
                    random_scores = []
                    
                    for i, (images, _) in enumerate(dataloader):
                        if i >= 3:  # Test on 3 batches
                            break
                        
                        images = images.to(self.device)
                        
                        # Get slots
                        patch_tokens = self.model.encoder.forward_features(images)
                        patch_tokens = patch_tokens[:, self.model.encoder.num_prefix_tokens:]
                        
                        slots = self.model.generator(
                            tgt=self.model.slot_queries.repeat(images.shape[0], 1, 1),
                            memory=patch_tokens,
                            tgt_mask=self.model.tgt_mask.to(patch_tokens.device),
                            tgt_is_causal=True,
                        )
                        
                        # Causal ordering (first N slots)
                        causal_slots = slots[:, :num_slots]
                        causal_null = self.model.null_slots[:, num_slots:].expand(images.shape[0], -1, -1)
                        causal_masked = torch.cat([causal_slots, causal_null], dim=1)
                        
                        causal_recon = self.model.reconstructor(
                            tgt=self.model.decoder_pos_embed.repeat(images.shape[0], 1, 1),
                            memory=causal_masked,
                        )
                        causal_mse = F.mse_loss(causal_recon, patch_tokens).item()
                        causal_scores.append(causal_mse)
                        
                        # Random ordering
                        random_indices = torch.randperm(128)[:num_slots]
                        random_slots = slots[:, random_indices]
                        random_null = self.model.null_slots[:, num_slots:].expand(images.shape[0], -1, -1)
                        random_masked = torch.cat([random_slots, random_null], dim=1)
                        
                        random_recon = self.model.reconstructor(
                            tgt=self.model.decoder_pos_embed.repeat(images.shape[0], 1, 1),
                            memory=random_masked,
                        )
                        random_mse = F.mse_loss(random_recon, patch_tokens).item()
                        random_scores.append(random_mse)
                    
                    causal_mean = np.mean(causal_scores)
                    random_mean = np.mean(random_scores)
                    improvement = random_mean / causal_mean if causal_mean > 0 else 1.0
                    
                    dataset_results[num_slots] = {
                        'causal_mse': causal_mean,
                        'random_mse': random_mean,
                        'improvement': improvement
                    }
                    
                    print(f"  {num_slots} slots: Causal={causal_mean:.6f}, Random={random_mean:.6f}, Improvement={improvement:.2f}x")
            
            results[dataset_name] = dataset_results
        
        self.results['random_baseline'] = results
        return results
    
    def plot_results(self):
        """Generate plots for all experiments."""
        print("\n=== Generating Plots ===")
        
        # Plot 1: Reconstruction scaling
        if 'reconstruction_scaling' in self.results:
            plt.figure(figsize=(12, 8))
            
            for dataset_name, dataset_results in self.results['reconstruction_scaling'].items():
                slot_counts = list(dataset_results.keys())
                mse_values = [dataset_results[k]['mse_mean'] for k in slot_counts]
                mse_stds = [dataset_results[k]['mse_std'] for k in slot_counts]
                
                plt.errorbar(slot_counts, mse_values, yerr=mse_stds, 
                           label=dataset_name, marker='o', capsize=5)
            
            plt.xlabel('Number of Slots')
            plt.ylabel('Reconstruction MSE')
            plt.title('SlotFormer: Reconstruction Quality vs Number of Slots')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            plt.savefig('experiments/slotformer_reconstruction_scaling.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Plot 2: Random baseline comparison
        if 'random_baseline' in self.results:
            fig, axes = plt.subplots(1, len(self.results['random_baseline']), figsize=(15, 5))
            if len(self.results['random_baseline']) == 1:
                axes = [axes]
            
            for i, (dataset_name, dataset_results) in enumerate(self.results['random_baseline'].items()):
                slot_counts = list(dataset_results.keys())
                improvements = [dataset_results[k]['improvement'] for k in slot_counts]
                
                axes[i].bar(range(len(slot_counts)), improvements)
                axes[i].set_xticks(range(len(slot_counts)))
                axes[i].set_xticklabels(slot_counts)
                axes[i].set_xlabel('Number of Slots')
                axes[i].set_ylabel('Improvement over Random')
                axes[i].set_title(f'{dataset_name}')
                axes[i].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
                axes[i].grid(True, alpha=0.3)
            
            plt.suptitle('SlotFormer: Causal vs Random Ordering')
            plt.tight_layout()
            plt.savefig('experiments/slotformer_random_baseline.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def save_results(self, filename='experiments/slotformer_validation_results.json'):
        """Save all experimental results."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filename}")
    
    def run_all_experiments(self, model_path=None):
        """Run complete experimental pipeline."""
        print("SlotFormer Validation Experiments")
        print("=" * 50)
        
        # Setup
        self.setup_model(model_path)
        datasets = self.get_datasets()
        
        if not datasets:
            print("No datasets found! Run: python scripts/download_validation_datasets.py")
            return
        
        print(f"Available datasets: {list(datasets.keys())}")
        
        # Run experiments
        self.experiment_1_reconstruction_scaling(datasets)
        self.experiment_2_causal_ordering_validation(datasets)
        self.experiment_3_random_baseline_comparison(datasets)
        
        # Generate outputs
        self.plot_results()
        self.save_results()
        
        print("\n" + "=" * 50)
        print("Experiments completed!")
        return self.results

def main():
    """Run SlotFormer validation experiments."""
    experiments = SlotFormerExperiments()
    results = experiments.run_all_experiments()
    
    # Print summary
    print("\nExperiment Summary:")
    if 'reconstruction_scaling' in results:
        print("✓ Reconstruction scaling analysis completed")
    if 'causal_ordering' in results:
        print("✓ Causal ordering validation completed")
    if 'random_baseline' in results:
        print("✓ Random baseline comparison completed")

if __name__ == "__main__":
    main()
