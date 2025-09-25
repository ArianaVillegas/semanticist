#!/usr/bin/env python3
"""
Visualize what SlotFormer slots are learning.
"""

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from train import SlotFormer
import cv2

class SlotVisualizer:
    """Visualize SlotFormer slots and reconstructions."""
    
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model = self.load_model(model_path)
        
        # Create results directory
        self.results_dir = Path("./slot_visualizations")
        self.results_dir.mkdir(exist_ok=True)
        
    def load_model(self, model_path):
        """Load trained SlotFormer model."""
        model = SlotFormer(128, 3, "vit_base_patch16_dinov3")
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            print(f"✓ Loaded model from {model_path}")
        else:
            print(f"⚠️ Model not found: {model_path}")
            
        return model.to(self.device).eval()
    
    def get_progressive_reconstructions(self, image, slot_counts=[1, 2, 4, 8, 16, 32, 64, 128]):
        """Get reconstructions using different numbers of slots."""
        
        with torch.no_grad():
            # Get patch tokens
            patch_tokens = self.model.encoder.forward_features(image)
            patch_tokens = patch_tokens[:, self.model.encoder.num_prefix_tokens:]
            
            # Generate all slots
            slots = self.model.generator(
                tgt=self.model.slot_queries.repeat(image.shape[0], 1, 1),
                memory=patch_tokens,
                tgt_mask=self.model.tgt_mask.to(image.device),
                tgt_is_causal=True,
            )
            
            reconstructions = {}
            losses = {}
            
            for num_slots in slot_counts:
                # Use only first num_slots
                used_slots = slots[:, :num_slots]
                
                # Pad with NULL tokens if needed
                if num_slots < self.model.num_slots:
                    null_padding = self.model.null_slots[:, :self.model.num_slots-num_slots]
                    null_padding = null_padding.expand(image.shape[0], -1, -1).type_as(slots)
                    padded_slots = torch.cat([used_slots, null_padding], dim=1)
                else:
                    padded_slots = used_slots
                
                # Reconstruct
                recon = self.model.reconstructor(
                    tgt=self.model.decoder_pos_embed.repeat(image.shape[0], 1, 1),
                    memory=padded_slots,
                )
                
                reconstructions[num_slots] = recon
                losses[num_slots] = F.mse_loss(recon, patch_tokens).item()
            
            return reconstructions, losses, patch_tokens, slots
    
    def patches_to_image(self, patches, img_size=224):
        """Convert patch tokens back to image format."""
        # patches: [B, 196, 768] -> need to convert to [B, 3, 224, 224]
        B, num_patches, embed_dim = patches.shape
        patch_size = int(np.sqrt(num_patches))  # 14
        
        # For visualization, we'll use PCA to reduce 768 -> 3 (RGB)
        patches_flat = patches.view(-1, embed_dim)  # [B*196, 768]
        
        # Simple projection to RGB (learned linear layer would be better)
        # For now, just take first 3 dimensions and normalize
        rgb_patches = patches_flat[:, :3]  # [B*196, 3]
        
        # Normalize to [0, 1]
        rgb_patches = (rgb_patches - rgb_patches.min()) / (rgb_patches.max() - rgb_patches.min() + 1e-8)
        
        # Reshape to image
        rgb_patches = rgb_patches.view(B, patch_size, patch_size, 3)
        
        # Resize to target image size
        images = []
        for b in range(B):
            img = rgb_patches[b].cpu().numpy()
            img_resized = cv2.resize(img, (img_size, img_size))
            images.append(img_resized)
        
        return np.array(images)
    
    def visualize_progressive_reconstruction(self, image_idx=0, slot_counts=[1, 2, 4, 8, 16, 32, 64, 128]):
        """Visualize how reconstruction improves with more slots."""
        
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
        
        # Get specific image
        image, label = dataset[image_idx]
        image = image.unsqueeze(0).to(self.device)
        
        # Get reconstructions
        reconstructions, losses, patch_tokens, slots = self.get_progressive_reconstructions(image, slot_counts)
        
        # Create visualization
        fig, axes = plt.subplots(3, len(slot_counts), figsize=(20, 8))
        
        # Original image (denormalized)
        orig_img = image[0].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        orig_img = orig_img * std + mean
        orig_img = torch.clamp(orig_img, 0, 1)
        
        for i, num_slots in enumerate(slot_counts):
            # Row 1: Original image (repeated)
            axes[0, i].imshow(orig_img.permute(1, 2, 0))
            axes[0, i].set_title(f'Original\n(Target)', fontsize=10)
            axes[0, i].axis('off')
            
            # Row 2: Reconstruction
            recon_img = self.patches_to_image(reconstructions[num_slots])
            axes[1, i].imshow(recon_img[0])
            axes[1, i].set_title(f'{num_slots} Slots\nLoss: {losses[num_slots]:.4f}', fontsize=10)
            axes[1, i].axis('off')
            
            # Row 3: Loss plot (will be filled at the end)
            if i == len(slot_counts) - 1:
                # Plot loss curve on the last subplot
                axes[2, i].plot(slot_counts, [losses[s] for s in slot_counts], 'b-o')
                axes[2, i].set_xlabel('Number of Slots')
                axes[2, i].set_ylabel('Reconstruction Loss')
                axes[2, i].set_title('Loss vs Slots')
                axes[2, i].grid(True, alpha=0.3)
                axes[2, i].set_xscale('log')
                axes[2, i].set_yscale('log')
            else:
                axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'progressive_reconstruction_img{image_idx}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved progressive reconstruction for image {image_idx}")
        return losses
    
    def visualize_slot_attention_maps(self, image_idx=0, top_slots=8):
        """Visualize what each slot is attending to."""
        
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
        
        image, label = dataset[image_idx]
        image = image.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get patch tokens and slots
            patch_tokens = self.model.encoder.forward_features(image)
            patch_tokens = patch_tokens[:, self.model.encoder.num_prefix_tokens:]
            
            slots = self.model.generator(
                tgt=self.model.slot_queries.repeat(1, 1, 1),
                memory=patch_tokens,
                tgt_mask=self.model.tgt_mask.to(image.device),
                tgt_is_causal=True,
            )
            
            # Compute attention between slots and patches
            # Simple dot product attention
            attention = torch.matmul(slots[0], patch_tokens[0].T)  # [128, 196]
            attention = F.softmax(attention, dim=-1)
        
        # Visualize attention maps
        fig, axes = plt.subplots(2, top_slots//2, figsize=(16, 8))
        axes = axes.flatten()
        
        # Original image
        orig_img = image[0].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        orig_img = orig_img * std + mean
        orig_img = torch.clamp(orig_img, 0, 1)
        
        for i in range(top_slots):
            # Get attention map for slot i
            attn_map = attention[i].cpu().numpy()  # [196]
            attn_map = attn_map.reshape(14, 14)  # 14x14 patches
            
            # Resize to image size
            attn_map_resized = cv2.resize(attn_map, (224, 224))
            
            # Overlay on original image
            axes[i].imshow(orig_img.permute(1, 2, 0))
            axes[i].imshow(attn_map_resized, alpha=0.6, cmap='hot')
            axes[i].set_title(f'Slot {i+1} Attention')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'slot_attention_maps_img{image_idx}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved slot attention maps for image {image_idx}")
    
    def analyze_slot_specialization(self, num_images=10):
        """Analyze what different slots specialize in."""
        
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
        
        slot_activations = []
        
        with torch.no_grad():
            for i in range(num_images):
                image, _ = dataset[i]
                image = image.unsqueeze(0).to(self.device)
                
                # Get slots
                patch_tokens = self.model.encoder.forward_features(image)
                patch_tokens = patch_tokens[:, self.model.encoder.num_prefix_tokens:]
                
                slots = self.model.generator(
                    tgt=self.model.slot_queries.repeat(1, 1, 1),
                    memory=patch_tokens,
                    tgt_mask=self.model.tgt_mask.to(image.device),
                    tgt_is_causal=True,
                )
                
                # Compute slot activation strength (L2 norm)
                slot_norms = torch.norm(slots[0], dim=-1).cpu().numpy()  # [128]
                slot_activations.append(slot_norms)
        
        slot_activations = np.array(slot_activations)  # [num_images, 128]
        
        # Plot slot activation patterns
        plt.figure(figsize=(15, 8))
        
        # Heatmap of slot activations
        plt.subplot(2, 2, 1)
        sns.heatmap(slot_activations.T, cmap='viridis', cbar=True)
        plt.xlabel('Image Index')
        plt.ylabel('Slot Index')
        plt.title('Slot Activation Patterns')
        
        # Average activation per slot
        plt.subplot(2, 2, 2)
        avg_activations = np.mean(slot_activations, axis=0)
        plt.plot(avg_activations, 'b-')
        plt.xlabel('Slot Index')
        plt.ylabel('Average Activation')
        plt.title('Average Slot Activation')
        plt.grid(True, alpha=0.3)
        
        # Slot usage distribution
        plt.subplot(2, 2, 3)
        plt.hist(avg_activations, bins=20, alpha=0.7)
        plt.xlabel('Activation Level')
        plt.ylabel('Number of Slots')
        plt.title('Distribution of Slot Activations')
        
        # Top vs bottom slots
        plt.subplot(2, 2, 4)
        top_10 = np.mean(slot_activations[:, :10], axis=1)
        bottom_10 = np.mean(slot_activations[:, -10:], axis=1)
        
        plt.plot(top_10, 'g-', label='Top 10 Slots', linewidth=2)
        plt.plot(bottom_10, 'r-', label='Bottom 10 Slots', linewidth=2)
        plt.xlabel('Image Index')
        plt.ylabel('Average Activation')
        plt.title('Top vs Bottom Slot Usage')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'slot_specialization_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved slot specialization analysis")
        
        return slot_activations
    
    def create_comprehensive_visualization(self, num_images=5):
        """Create comprehensive visualization of SlotFormer behavior."""
        
        print("=== Creating Comprehensive SlotFormer Visualizations ===")
        
        # 1. Progressive reconstructions for multiple images
        all_losses = []
        for i in range(num_images):
            print(f"Processing image {i+1}/{num_images}...")
            losses = self.visualize_progressive_reconstruction(i)
            all_losses.append(losses)
        
        # 2. Slot attention maps
        for i in range(min(3, num_images)):
            self.visualize_slot_attention_maps(i)
        
        # 3. Slot specialization analysis
        slot_activations = self.analyze_slot_specialization(num_images)
        
        # 4. Summary plot of all images
        plt.figure(figsize=(12, 8))
        
        slot_counts = [1, 2, 4, 8, 16, 32, 64, 128]
        
        for i, losses in enumerate(all_losses):
            loss_values = [losses[s] for s in slot_counts]
            plt.loglog(slot_counts, loss_values, 'o-', alpha=0.7, label=f'Image {i+1}')
        
        # Average across all images
        avg_losses = []
        for s in slot_counts:
            avg_loss = np.mean([losses[s] for losses in all_losses])
            avg_losses.append(avg_loss)
        
        plt.loglog(slot_counts, avg_losses, 'k-', linewidth=3, label='Average')
        
        plt.xlabel('Number of Slots')
        plt.ylabel('Reconstruction Loss')
        plt.title('SlotFormer: Reconstruction Quality vs Number of Slots')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.results_dir / 'summary_reconstruction_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Comprehensive visualization complete!")
        print(f"📁 Results saved to: {self.results_dir.absolute()}")
        print(f"📊 Files created:")
        for file in sorted(self.results_dir.glob("*.png")):
            print(f"   - {file.name}")

def main():
    """Main visualization function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize SlotFormer slots')
    parser.add_argument('--model', type=str, default='model-49.ckpt', 
                       help='Path to trained model')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use')
    parser.add_argument('--num_images', type=int, default=5,
                       help='Number of images to visualize')
    
    args = parser.parse_args()
    
    visualizer = SlotVisualizer(args.model, args.device)
    visualizer.create_comprehensive_visualization(args.num_images)

if __name__ == "__main__":
    main()
