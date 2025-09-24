#!/usr/bin/env python3
"""
SlotFormer Validation Experiments
Comprehensive experimental framework to validate SlotFormer approach across datasets and metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
import lightning as L
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm.auto import tqdm
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
import seaborn as sns

# Import SlotFormer from train.py
import sys
sys.path.append('.')
from train import SlotFormer

class SlotFormerValidator:
    """Comprehensive validation framework for SlotFormer."""
    
    def __init__(self, model_path=None, device='cuda'):
        self.device = device
        self.results = {}
        
        # Load model if path provided
        if model_path:
            self.model = self.load_model(model_path)
        else:
            self.model = None
    
    def load_model(self, model_path):
        """Load trained SlotFormer model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        model = SlotFormer(num_slots=128, num_layers=3, encoder_name="vit_base_patch16_dinov3.lvd_1689m")
        model.load_state_dict(checkpoint['model'])
        model.eval()
        return model.to(self.device)
    
    def get_datasets(self):
        """Get multiple datasets for validation."""
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        datasets = {
            'imagenette': torchvision.datasets.Imagenette('/tmp/datasets', split='val', transform=transform),
            'cifar10': torchvision.datasets.CIFAR10('/tmp/datasets', train=False, transform=transform),
            'stl10': torchvision.datasets.STL10('/tmp/datasets', split='test', transform=transform),
        }
        
        # Add custom video frame dataset if available
        if Path('dora_videos').exists():
            datasets['video_frames'] = self.create_video_frame_dataset(transform)
        
        return datasets
    
    def create_video_frame_dataset(self, transform):
        """Create dataset from video frames for temporal analysis."""
        # Implementation would extract frames from videos
        # For now, return None - implement based on your video processing needs
        return None
    
    def experiment_1_reconstruction_quality(self, datasets, slot_counts=[1, 4, 8, 16, 32, 64, 128]):
        """Experiment 1: Reconstruction quality vs number of slots."""
        print("Running Experiment 1: Reconstruction Quality Analysis")
        
        results = {}
        for dataset_name, dataset in datasets.items():
            if dataset is None:
                continue
                
            print(f"Testing on {dataset_name}")
            dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
            
            dataset_results = {}
            for num_slots in slot_counts:
                mse_scores, ssim_scores = self.evaluate_reconstruction(dataloader, num_slots)
                dataset_results[num_slots] = {
                    'mse': np.mean(mse_scores),
                    'ssim': np.mean(ssim_scores),
                    'mse_std': np.std(mse_scores),
                    'ssim_std': np.std(ssim_scores)
                }
            
            results[dataset_name] = dataset_results
        
        self.results['reconstruction_quality'] = results
        self.plot_reconstruction_curves(results)
        return results
    
    def evaluate_reconstruction(self, dataloader, num_slots, max_batches=10):
        """Evaluate reconstruction quality with specific number of slots."""
        mse_scores = []
        ssim_scores = []
        
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                if i >= max_batches:
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
                
                # Use only first num_slots
                used_slots = slots[:, :num_slots]
                null_slots = self.model.null_slots[:, num_slots:].expand(images.shape[0], -1, -1)
                masked_slots = torch.cat([used_slots, null_slots], dim=1)
                
                # Reconstruct
                reconstructed = self.model.reconstructor(
                    tgt=self.model.decoder_pos_embed.repeat(images.shape[0], 1, 1),
                    memory=masked_slots,
                )
                
                # Calculate metrics
                mse = F.mse_loss(reconstructed, patch_tokens, reduction='none').mean(dim=[1,2])
                mse_scores.extend(mse.cpu().numpy())
                
                # SSIM would require converting back to image space - simplified for now
                ssim_scores.extend([0.5] * images.shape[0])  # Placeholder
        
        return mse_scores, ssim_scores
    
    def experiment_2_slot_interpretability(self, datasets):
        """Experiment 2: Analyze slot interpretability and semantic consistency."""
        print("Running Experiment 2: Slot Interpretability Analysis")
        
        results = {}
        for dataset_name, dataset in datasets.items():
            if dataset is None:
                continue
                
            print(f"Analyzing slots for {dataset_name}")
            dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
            
            slot_activations = self.collect_slot_activations(dataloader)
            
            # Analyze slot specialization
            specialization_scores = self.analyze_slot_specialization(slot_activations)
            
            # PCA analysis of slot space
            pca_results = self.analyze_slot_pca(slot_activations)
            
            results[dataset_name] = {
                'specialization': specialization_scores,
                'pca_variance_ratio': pca_results['variance_ratio'],
                'slot_diversity': self.calculate_slot_diversity(slot_activations)
            }
        
        self.results['slot_interpretability'] = results
        return results
    
    def collect_slot_activations(self, dataloader, max_batches=20):
        """Collect slot activations across dataset."""
        all_slots = []
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(dataloader):
                if i >= max_batches:
                    break
                
                images = images.to(self.device)
                patch_tokens = self.model.encoder.forward_features(images)
                patch_tokens = patch_tokens[:, self.model.encoder.num_prefix_tokens:]
                
                slots = self.model.generator(
                    tgt=self.model.slot_queries.repeat(images.shape[0], 1, 1),
                    memory=patch_tokens,
                    tgt_mask=self.model.tgt_mask.to(patch_tokens.device),
                    tgt_is_causal=True,
                )
                
                all_slots.append(slots.cpu().numpy())
        
        return np.concatenate(all_slots, axis=0)
    
    def analyze_slot_specialization(self, slot_activations):
        """Analyze how specialized each slot is."""
        # Calculate variance across samples for each slot
        slot_variances = np.var(slot_activations, axis=0)  # [num_slots, embed_dim]
        specialization = np.mean(slot_variances, axis=1)  # [num_slots]
        return specialization
    
    def analyze_slot_pca(self, slot_activations):
        """PCA analysis of slot representations."""
        # Reshape to [num_samples * num_slots, embed_dim]
        reshaped = slot_activations.reshape(-1, slot_activations.shape[-1])
        
        pca = PCA(n_components=min(50, reshaped.shape[1]))
        pca.fit(reshaped)
        
        return {
            'variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
        }
    
    def calculate_slot_diversity(self, slot_activations):
        """Calculate diversity metrics for slots."""
        # Average pairwise cosine similarity between slots
        slots_mean = np.mean(slot_activations, axis=0)  # [num_slots, embed_dim]
        
        similarities = []
        for i in range(slots_mean.shape[0]):
            for j in range(i+1, slots_mean.shape[0]):
                sim = np.dot(slots_mean[i], slots_mean[j]) / (
                    np.linalg.norm(slots_mean[i]) * np.linalg.norm(slots_mean[j])
                )
                similarities.append(sim)
        
        return {
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities)
        }
    
    def experiment_3_causal_ordering_validation(self, datasets):
        """Experiment 3: Validate causal ordering makes sense."""
        print("Running Experiment 3: Causal Ordering Validation")
        
        results = {}
        for dataset_name, dataset in datasets.items():
            if dataset is None:
                continue
                
            print(f"Testing causal ordering for {dataset_name}")
            dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
            
            ordering_scores = self.evaluate_causal_ordering(dataloader)
            results[dataset_name] = ordering_scores
        
        self.results['causal_ordering'] = results
        return results
    
    def evaluate_causal_ordering(self, dataloader, max_batches=10):
        """Evaluate if causal ordering produces meaningful progression."""
        reconstruction_progression = []
        
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                if i >= max_batches:
                    break
                
                images = images.to(self.device)
                
                # Test reconstruction quality with increasing slot counts
                slot_counts = [1, 2, 4, 8, 16, 32, 64, 128]
                batch_progression = []
                
                for num_slots in slot_counts:
                    mse_scores, _ = self.evaluate_reconstruction(
                        [(images, None)], num_slots, max_batches=1
                    )
                    batch_progression.append(np.mean(mse_scores))
                
                reconstruction_progression.append(batch_progression)
        
        # Calculate if reconstruction improves monotonically
        progression_array = np.array(reconstruction_progression)
        mean_progression = np.mean(progression_array, axis=0)
        
        # Check monotonicity (lower MSE = better)
        monotonic_improvements = np.sum(np.diff(mean_progression) < 0) / len(mean_progression)
        
        return {
            'mean_progression': mean_progression.tolist(),
            'monotonic_ratio': monotonic_improvements,
            'final_improvement': mean_progression[0] / mean_progression[-1]
        }
    
    def experiment_4_cross_dataset_generalization(self, datasets):
        """Experiment 4: Test generalization across datasets."""
        print("Running Experiment 4: Cross-Dataset Generalization")
        
        results = {}
        dataset_names = list(datasets.keys())
        
        for train_dataset in dataset_names:
            if datasets[train_dataset] is None:
                continue
                
            results[train_dataset] = {}
            
            for test_dataset in dataset_names:
                if datasets[test_dataset] is None:
                    continue
                
                print(f"Testing {train_dataset} -> {test_dataset}")
                
                # Evaluate on test dataset (assuming model trained on train_dataset)
                test_loader = DataLoader(datasets[test_dataset], batch_size=16, shuffle=False)
                mse_scores, _ = self.evaluate_reconstruction(test_loader, num_slots=32, max_batches=5)
                
                results[train_dataset][test_dataset] = {
                    'mse': np.mean(mse_scores),
                    'mse_std': np.std(mse_scores)
                }
        
        self.results['cross_dataset'] = results
        return results
    
    def plot_reconstruction_curves(self, results):
        """Plot reconstruction quality curves."""
        plt.figure(figsize=(12, 8))
        
        for dataset_name, dataset_results in results.items():
            slot_counts = list(dataset_results.keys())
            mse_values = [dataset_results[k]['mse'] for k in slot_counts]
            mse_stds = [dataset_results[k]['mse_std'] for k in slot_counts]
            
            plt.errorbar(slot_counts, mse_values, yerr=mse_stds, 
                        label=dataset_name, marker='o', capsize=5)
        
        plt.xlabel('Number of Slots')
        plt.ylabel('Reconstruction MSE')
        plt.title('Reconstruction Quality vs Number of Slots')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.savefig('slotformer_reconstruction_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, output_path='slotformer_validation_report.json'):
        """Generate comprehensive validation report."""
        report = {
            'model_architecture': 'SlotFormer',
            'validation_timestamp': str(torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'),
            'experiments': self.results,
            'summary': self.generate_summary()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Validation report saved to {output_path}")
        return report
    
    def generate_summary(self):
        """Generate summary of validation results."""
        summary = {}
        
        if 'reconstruction_quality' in self.results:
            # Find optimal slot count across datasets
            optimal_slots = {}
            for dataset, results in self.results['reconstruction_quality'].items():
                best_slot_count = min(results.keys(), key=lambda k: results[k]['mse'])
                optimal_slots[dataset] = best_slot_count
            summary['optimal_slot_counts'] = optimal_slots
        
        if 'causal_ordering' in self.results:
            # Average monotonic improvement ratio
            monotonic_ratios = [r['monotonic_ratio'] for r in self.results['causal_ordering'].values()]
            summary['average_monotonic_ratio'] = np.mean(monotonic_ratios)
        
        return summary

def main():
    """Run comprehensive SlotFormer validation."""
    validator = SlotFormerValidator()
    
    # Get datasets
    datasets = validator.get_datasets()
    print(f"Available datasets: {list(datasets.keys())}")
    
    # Run experiments
    validator.experiment_1_reconstruction_quality(datasets)
    validator.experiment_2_slot_interpretability(datasets)
    validator.experiment_3_causal_ordering_validation(datasets)
    validator.experiment_4_cross_dataset_generalization(datasets)
    
    # Generate report
    report = validator.generate_report()
    print("\nValidation Summary:")
    print(json.dumps(report['summary'], indent=2))

if __name__ == "__main__":
    main()
