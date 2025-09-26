#!/usr/bin/env python3
"""
Large-scale scalability study for SlotFormer.
Tests scaling across: datasets, slot counts, model sizes, sequence lengths.
"""

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from train import SlotFormer

class ScalabilityStudy:
    """Comprehensive scalability analysis."""
    
    def __init__(self, device="cuda"):
        self.device = device
        self.results_dir = Path("./scalability_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def test_dataset_scaling(self):
        """Test performance across different dataset sizes."""
        print("=== Dataset Scaling Experiment ===")
        
        # Test on progressively larger subsets
        dataset_sizes = [100, 500, 1000, 2000, 5000, 10000]
        results = {}
        
        for size in dataset_sizes:
            print(f"Testing dataset size: {size}")
            
            # Create model
            model = SlotFormer(128, 3, "vit_base_patch16_dinov3").to(self.device)
            
            # Load subset of data
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
                # Subsample
                indices = torch.randperm(len(full_dataset))[:size]
                dataset = torch.utils.data.Subset(full_dataset, indices)
                print(f"✓ Using Imagenette dataset subset: {size} samples")
            except Exception as e:
                print(f"⚠️ Imagenette not available ({e}), creating synthetic dataset")
                # Create synthetic dataset
                dataset = [(torch.randn(3, 224, 224), i % 10) for i in range(size)]
            
            # Quick training (10 epochs)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
            
            start_time = time.time()
            losses = []
            
            model.train()
            for epoch in range(10):
                epoch_loss = 0
                count = 0
                
                for batch_idx, (images, _) in enumerate(dataloader):
                    if batch_idx >= 50:  # Limit batches for speed
                        break
                        
                    images = images.to(self.device)
                    
                    # Forward pass
                    patch_tokens = model.encoder.forward_features(images)
                    patch_tokens = patch_tokens[:, model.encoder.num_prefix_tokens:]
                    
                    # Random masking for training
                    B, N, D = patch_tokens.shape
                    num_slots = torch.randint(1, model.num_slots + 1, (1,)).item()
                    
                    slots = model.generator(
                        tgt=model.slot_queries[:, :num_slots].repeat(B, 1, 1),
                        memory=patch_tokens,
                        tgt_mask=model.tgt_mask[:num_slots, :num_slots].to(self.device),
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
            
            training_time = time.time() - start_time
            
            # Test monotonic property
            model.eval()
            monotonic_scores = []
            
            with torch.no_grad():
                for test_idx in range(min(20, len(dataset))):
                    image, _ = dataset[test_idx]
                    image = image.unsqueeze(0).to(self.device)
                    
                    patch_tokens = model.encoder.forward_features(image)
                    patch_tokens = patch_tokens[:, model.encoder.num_prefix_tokens:]
                    
                    slots = model.generator(
                        tgt=model.slot_queries.repeat(1, 1, 1),
                        memory=patch_tokens,
                        tgt_mask=model.tgt_mask.to(self.device),
                        tgt_is_causal=True,
                    )
                    
                    # Test different slot counts
                    test_losses = []
                    for num_slots in [1, 4, 16, 64, 128]:
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
                        
                        test_loss = F.mse_loss(recon, patch_tokens).item()
                        test_losses.append(test_loss)
                    
                    # Check monotonic decrease
                    decreases = sum(1 for i in range(1, len(test_losses)) 
                                  if test_losses[i] < test_losses[i-1])
                    monotonic_score = decreases / (len(test_losses) - 1)
                    monotonic_scores.append(monotonic_score)
            
            avg_monotonic = np.mean(monotonic_scores)
            
            results[size] = {
                'training_losses': losses,
                'final_loss': losses[-1] if losses else float('inf'),
                'training_time': training_time,
                'monotonic_score': avg_monotonic,
                'convergence_rate': (losses[0] - losses[-1]) / losses[0] if losses and losses[0] > 0 else 0
            }
            
            print(f"  Final loss: {losses[-1]:.6f}")
            print(f"  Monotonic score: {avg_monotonic:.3f}")
            print(f"  Training time: {training_time:.1f}s")
        
        # Save results
        with open(self.results_dir / 'dataset_scaling.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot results
        self.plot_dataset_scaling(results)
        
        return results
    
    def test_slot_scaling(self, model_path="model-49.ckpt"):
        """Test performance with different maximum slot counts."""
        print("=== Slot Count Scaling Experiment ===")
        
        slot_configs = [16, 32, 64, 128, 256, 512]
        results = {}
        
        for max_slots in slot_configs:
            print(f"Testing max slots: {max_slots}")
            
            # Create model with different slot count
            model = SlotFormer(max_slots, 3, "vit_base_patch16_dinov3").to(self.device)
            
            # Load pretrained weights if available (for lower slot counts)
            if max_slots <= 128 and Path(model_path).exists():
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    if 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                    
                    # Load compatible weights
                    model_dict = model.state_dict()
                    compatible_dict = {k: v for k, v in state_dict.items() 
                                     if k in model_dict and v.shape == model_dict[k].shape}
                    model_dict.update(compatible_dict)
                    model.load_state_dict(model_dict)
                    print(f"  Loaded {len(compatible_dict)} compatible weights")
                except Exception as e:
                    print(f"  Could not load weights: {e}")
            
            # Test computational efficiency
            model.eval()
            
            # Memory usage test
            torch.cuda.empty_cache()
            start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Forward pass timing
            test_image = torch.randn(1, 3, 224, 224).to(self.device)
            
            times = []
            for _ in range(10):  # Average over 10 runs
                start_time = time.time()
                
                with torch.no_grad():
                    patch_tokens = model.encoder.forward_features(test_image)
                    patch_tokens = patch_tokens[:, model.encoder.num_prefix_tokens:]
                    
                    slots = model.generator(
                        tgt=model.slot_queries.repeat(1, 1, 1),
                        memory=patch_tokens,
                        tgt_mask=model.tgt_mask.to(self.device),
                        tgt_is_causal=True,
                    )
                    
                    reconstructed = model.reconstructor(
                        tgt=model.decoder_pos_embed.repeat(1, 1, 1),
                        memory=slots,
                    )
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            memory_usage = peak_memory - start_memory
            
            # Test quality scaling
            quality_scores = []
            
            try:
                dataset = torchvision.datasets.Imagenette(
                    "./datasets", split="val", transform=torchvision.transforms.Compose([
                        torchvision.transforms.Resize(224),
                        torchvision.transforms.CenterCrop(224),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]), download=False
                )
                
                with torch.no_grad():
                    for test_idx in range(min(10, len(dataset))):
                        image, _ = dataset[test_idx]
                        image = image.unsqueeze(0).to(self.device)
                        
                        patch_tokens = model.encoder.forward_features(image)
                        patch_tokens = patch_tokens[:, model.encoder.num_prefix_tokens:]
                        
                        slots = model.generator(
                            tgt=model.slot_queries.repeat(1, 1, 1),
                            memory=patch_tokens,
                            tgt_mask=model.tgt_mask.to(self.device),
                            tgt_is_causal=True,
                        )
                        
                        reconstructed = model.reconstructor(
                            tgt=model.decoder_pos_embed.repeat(1, 1, 1),
                            memory=slots,
                        )
                        
                        loss = F.mse_loss(reconstructed, patch_tokens).item()
                        quality_scores.append(loss)
                
                avg_quality = np.mean(quality_scores)
            except:
                avg_quality = float('inf')
            
            results[max_slots] = {
                'avg_inference_time': avg_time,
                'memory_usage_mb': memory_usage / (1024 * 1024) if memory_usage > 0 else 0,
                'avg_reconstruction_loss': avg_quality,
                'parameters': sum(p.numel() for p in model.parameters()),
                'slots_per_second': max_slots / avg_time if avg_time > 0 else 0
            }
            
            print(f"  Inference time: {avg_time:.4f}s")
            print(f"  Memory usage: {memory_usage / (1024 * 1024):.1f} MB")
            print(f"  Reconstruction loss: {avg_quality:.6f}")
        
        # Save results
        with open(self.results_dir / 'slot_scaling.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot results
        self.plot_slot_scaling(results)
        
        return results
    
    def plot_dataset_scaling(self, results):
        """Plot dataset scaling results."""
        sizes = list(results.keys())
        final_losses = [results[s]['final_loss'] for s in sizes]
        monotonic_scores = [results[s]['monotonic_score'] for s in sizes]
        training_times = [results[s]['training_time'] for s in sizes]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Final loss vs dataset size
        axes[0, 0].semilogx(sizes, final_losses, 'b-o', linewidth=2)
        axes[0, 0].set_xlabel('Dataset Size')
        axes[0, 0].set_ylabel('Final Reconstruction Loss')
        axes[0, 0].set_title('Learning Quality vs Dataset Size')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Monotonic score vs dataset size
        axes[0, 1].semilogx(sizes, monotonic_scores, 'g-s', linewidth=2)
        axes[0, 1].set_xlabel('Dataset Size')
        axes[0, 1].set_ylabel('Monotonic Score')
        axes[0, 1].set_title('Causal Learning vs Dataset Size')
        axes[0, 1].axhline(y=0.8, color='red', linestyle='--', label='Target')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Training time vs dataset size
        axes[1, 0].loglog(sizes, training_times, 'r-^', linewidth=2)
        axes[1, 0].set_xlabel('Dataset Size')
        axes[1, 0].set_ylabel('Training Time (seconds)')
        axes[1, 0].set_title('Training Efficiency vs Dataset Size')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning curves for different sizes
        axes[1, 1].set_title('Learning Curves by Dataset Size')
        for size in sizes:
            if 'training_losses' in results[size]:
                losses = results[size]['training_losses']
                axes[1, 1].plot(losses, label=f'{size} samples', alpha=0.7)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'dataset_scaling_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_slot_scaling(self, results):
        """Plot slot scaling results."""
        slot_counts = list(results.keys())
        inference_times = [results[s]['avg_inference_time'] for s in slot_counts]
        memory_usage = [results[s]['memory_usage_mb'] for s in slot_counts]
        reconstruction_losses = [results[s]['avg_reconstruction_loss'] for s in slot_counts]
        parameters = [results[s]['parameters'] / 1e6 for s in slot_counts]  # In millions
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Inference time vs slot count
        axes[0, 0].loglog(slot_counts, inference_times, 'b-o', linewidth=2)
        axes[0, 0].set_xlabel('Max Slot Count')
        axes[0, 0].set_ylabel('Inference Time (seconds)')
        axes[0, 0].set_title('Computational Efficiency vs Slot Count')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Memory usage vs slot count
        axes[0, 1].loglog(slot_counts, memory_usage, 'g-s', linewidth=2)
        axes[0, 1].set_xlabel('Max Slot Count')
        axes[0, 1].set_ylabel('Memory Usage (MB)')
        axes[0, 1].set_title('Memory Efficiency vs Slot Count')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Reconstruction quality vs slot count
        axes[1, 0].loglog(slot_counts, reconstruction_losses, 'r-^', linewidth=2)
        axes[1, 0].set_xlabel('Max Slot Count')
        axes[1, 0].set_ylabel('Reconstruction Loss')
        axes[1, 0].set_title('Quality vs Slot Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Parameters vs slot count
        axes[1, 1].loglog(slot_counts, parameters, 'm-d', linewidth=2)
        axes[1, 1].set_xlabel('Max Slot Count')
        axes[1, 1].set_ylabel('Parameters (Millions)')
        axes[1, 1].set_title('Model Size vs Slot Count')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'slot_scaling_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Run scalability study."""
    study = ScalabilityStudy()
    
    print("=== SlotFormer Large-Scale Scalability Study ===")
    
    # Run experiments
    dataset_results = study.test_dataset_scaling()
    slot_results = study.test_slot_scaling()
    
    print(f"\n✅ Scalability study complete!")
    print(f"📁 Results saved to: {study.results_dir.absolute()}")

if __name__ == "__main__":
    main()
