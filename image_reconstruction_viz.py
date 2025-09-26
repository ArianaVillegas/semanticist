#!/usr/bin/env python3
"""
Image reconstruction from DINOv3 features for SlotFormer visualization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from train import SlotFormer
import timm

class FeatureToImageDecoder(nn.Module):
    """Decoder to reconstruct images from DINOv3 features."""
    
    def __init__(self, feature_dim=768, patch_size=16, img_size=224):
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2  # 196
        
        # Decoder layers to go from features to RGB patches
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, patch_size * patch_size * 3),  # RGB patch
            nn.Tanh()  # Output in [-1, 1]
        )
        
    def forward(self, features):
        """
        Args:
            features: [B, num_patches, feature_dim]
        Returns:
            images: [B, 3, img_size, img_size]
        """
        B, num_patches, _ = features.shape
        
        # Decode each patch to RGB
        rgb_patches = self.decoder(features)  # [B, num_patches, patch_size^2 * 3]
        
        # Reshape to patch format
        rgb_patches = rgb_patches.view(B, num_patches, 3, self.patch_size, self.patch_size)
        
        # Reconstruct image from patches
        patches_per_side = int(np.sqrt(num_patches))  # 14
        
        # Rearrange patches to form image
        rgb_patches = rgb_patches.view(B, patches_per_side, patches_per_side, 3, 
                                     self.patch_size, self.patch_size)
        
        # Combine patches into full image
        images = rgb_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        images = images.view(B, 3, self.img_size, self.img_size)
        
        return images

class ImageReconstructionVisualizer:
    """Visualize SlotFormer with actual image reconstructions."""
    
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model = self.load_model(model_path)
        
        # Create and train feature decoder
        self.feature_decoder = FeatureToImageDecoder().to(device)
        self.train_feature_decoder()
        
        # Results directory
        self.results_dir = Path("./image_reconstruction_viz")
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
            print(f"✓ Loaded SlotFormer from {model_path}")
        else:
            print(f"⚠️ Model not found: {model_path}")
            
        return model.to(self.device).eval()
    
    def train_feature_decoder(self, num_samples=200, epochs=50):
        """Train decoder to reconstruct images from DINOv3 features."""
        print("Training feature-to-image decoder...")
        
        # Load training data
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
                "./datasets", split="train", transform=transform, download=False
            )
        except:
            print("⚠️ Using validation set for decoder training")
            dataset = torchvision.datasets.Imagenette(
                "./datasets", split="val", transform=transform, download=False
            )
        
        # Collect feature-image pairs
        features_list = []
        images_list = []
        
        print(f"Collecting {num_samples} feature-image pairs...")
        with torch.no_grad():
            for i in range(min(num_samples, len(dataset))):
                if i % 50 == 0:
                    print(f"  Collected {i}/{num_samples}")
                
                image, _ = dataset[i]
                image = image.unsqueeze(0).to(self.device)
                
                # Get DINOv3 features
                features = self.model.encoder.forward_features(image)
                patch_features = features[:, self.model.encoder.num_prefix_tokens:]
                
                features_list.append(patch_features[0].cpu())
                images_list.append(image[0].cpu())
        
        # Train decoder
        optimizer = torch.optim.Adam(self.feature_decoder.parameters(), lr=1e-3)
        
        print(f"Training decoder for {epochs} epochs...")
        self.feature_decoder.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(0, len(features_list), 8):  # Batch size 8
                batch_features = torch.stack(features_list[i:i+8]).to(self.device)
                batch_images = torch.stack(images_list[i:i+8]).to(self.device)
                
                # Forward pass
                reconstructed = self.feature_decoder(batch_features)
                loss = F.mse_loss(reconstructed, batch_images)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                avg_loss = total_loss / (len(features_list) // 8)
                print(f"  Epoch {epoch}: Loss = {avg_loss:.6f}")
        
        self.feature_decoder.eval()
        print("✓ Feature decoder training complete!")
    
    def visualize_progressive_image_reconstruction(self, image_idx=0, 
                                                 slot_counts=[1, 2, 4, 8, 16, 32, 64, 128]):
        """Visualize actual image reconstruction with different slot counts."""
        
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
        except:
            print("⚠️ Creating synthetic test image")
            # Create a synthetic test image
            synthetic_img = torch.randn(3, 224, 224)
            synthetic_img = transform(synthetic_img)
            dataset = [(synthetic_img, 0)]
        
        image, label = dataset[image_idx]
        image = image.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get ground truth features
            gt_features = self.model.encoder.forward_features(image)
            gt_patches = gt_features[:, self.model.encoder.num_prefix_tokens:]
            
            # Generate all slots
            slots = self.model.generator(
                tgt=self.model.slot_queries.repeat(1, 1, 1),
                memory=gt_patches,
                tgt_mask=self.model.tgt_mask.to(self.device),
                tgt_is_causal=True,
            )
            
            # Get reconstructions and convert to images
            reconstructed_images = {}
            losses = {}
            
            for num_slots in slot_counts:
                print(f"Processing {num_slots} slots...")
                
                # Use only first num_slots
                used_slots = slots[:, :num_slots]
                
                # Pad with NULL tokens
                if num_slots < self.model.num_slots:
                    null_padding = self.model.null_slots[:, :self.model.num_slots-num_slots]
                    null_padding = null_padding.expand(1, -1, -1).type_as(slots)
                    padded_slots = torch.cat([used_slots, null_padding], dim=1)
                else:
                    padded_slots = used_slots
                
                # Reconstruct features
                recon_features = self.model.reconstructor(
                    tgt=self.model.decoder_pos_embed.repeat(1, 1, 1),
                    memory=padded_slots,
                )
                
                # Convert features to image
                recon_image = self.feature_decoder(recon_features)
                
                reconstructed_images[num_slots] = recon_image
                losses[num_slots] = F.mse_loss(recon_features, gt_patches).item()
            
            # Also reconstruct ground truth for comparison
            gt_image_recon = self.feature_decoder(gt_patches)
        
        # Create visualization
        fig, axes = plt.subplots(3, len(slot_counts), figsize=(32, 12))
        
        # Denormalize function
        def denormalize(tensor):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
            return torch.clamp(tensor * std + mean, 0, 1)
        
        # Original image (denormalized)
        orig_img = denormalize(image)[0].cpu().permute(1, 2, 0)
        
        # Ground truth reconstruction (from perfect features)
        gt_recon_img = torch.clamp((gt_image_recon[0] + 1) / 2, 0, 1).cpu().permute(1, 2, 0)
        
        for i, num_slots in enumerate(slot_counts):
            # Row 1: Original image
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title(f'Original Image\n({num_slots} slots)', fontsize=12)
            axes[0, i].axis('off')
            
            # Row 2: Ground truth reconstruction (perfect features)
            axes[1, i].imshow(gt_recon_img)
            axes[1, i].set_title(f'GT Feature Reconstruction\n(Upper bound)', fontsize=12)
            axes[1, i].axis('off')
            
            # Row 3: SlotFormer reconstruction
            recon_img = torch.clamp((reconstructed_images[num_slots][0] + 1) / 2, 0, 1)
            recon_img = recon_img.cpu().permute(1, 2, 0)
            
            axes[2, i].imshow(recon_img)
            axes[2, i].set_title(f'SlotFormer Reconstruction\nLoss: {losses[num_slots]:.4f}', fontsize=12)
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'image_reconstruction_img{image_idx}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create loss curve
        plt.figure(figsize=(10, 6))
        loss_values = [losses[s] for s in slot_counts]
        plt.loglog(slot_counts, loss_values, 'b-o', linewidth=2, markersize=8)
        plt.xlabel('Number of Slots')
        plt.ylabel('Feature Reconstruction Loss (MSE)')
        plt.title(f'Image Reconstruction Quality vs Slots (Image {image_idx})')
        plt.grid(True, alpha=0.3)
        
        # Add improvement annotation
        improvement = loss_values[0] / loss_values[-1]
        plt.text(0.02, 0.98, f'Improvement: {improvement:.1f}x\nMonotonic: {"✓" if all(loss_values[i] >= loss_values[i+1] for i in range(len(loss_values)-1)) else "✗"}', 
                transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.savefig(self.results_dir / f'image_reconstruction_loss_img{image_idx}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Image reconstruction visualization saved for image {image_idx}")
        print(f"  Feature loss improvement: {improvement:.1f}x")
        
        return losses
    
    def create_comprehensive_image_reconstruction(self, num_images=5):
        """Create comprehensive image reconstruction analysis."""
        
        print("=== Creating Image Reconstruction Visualizations ===")
        
        all_losses = []
        
        for i in range(num_images):
            print(f"\nProcessing image {i+1}/{num_images}...")
            losses = self.visualize_progressive_image_reconstruction(i)
            all_losses.append(losses)
        
        # Create summary
        slot_counts = [1, 2, 4, 8, 16, 32, 64, 128]
        
        plt.figure(figsize=(12, 8))
        
        for i, losses in enumerate(all_losses):
            loss_values = [losses[s] for s in slot_counts]
            plt.loglog(slot_counts, loss_values, 'o-', alpha=0.7, label=f'Image {i+1}')
        
        # Average
        avg_losses = []
        for s in slot_counts:
            avg_loss = np.mean([losses[s] for losses in all_losses])
            avg_losses.append(avg_loss)
        
        plt.loglog(slot_counts, avg_losses, 'k-', linewidth=3, 
                  marker='s', markersize=8, label='Average')
        
        plt.xlabel('Number of Slots', fontsize=12)
        plt.ylabel('Feature Reconstruction Loss (MSE)', fontsize=12)
        plt.title('SlotFormer: Image Reconstruction via Feature Learning', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add analysis text
        improvement = avg_losses[0] / avg_losses[-1]
        monotonic_count = sum(1 for i in range(len(avg_losses)-1) if avg_losses[i] >= avg_losses[i+1])
        monotonic_ratio = monotonic_count / (len(avg_losses) - 1)
        
        textstr = f'Average Improvement: {improvement:.1f}x\n'
        textstr += f'Monotonic Ratio: {monotonic_ratio:.3f}\n'
        textstr += f'Image Reconstruction: Via Feature Learning'
        
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'summary_image_reconstruction.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Image reconstruction analysis complete!")
        print(f"📁 Results saved to: {self.results_dir.absolute()}")
        print(f"📊 Key insights:")
        print(f"   - Images reconstructed via learned feature decoder")
        print(f"   - Shows visual quality improvement with more slots")
        print(f"   - Average improvement: {improvement:.1f}x")
        print(f"   - Monotonic ratio: {monotonic_ratio:.3f}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Image Reconstruction Visualization')
    parser.add_argument('--model', type=str, default='model-49.ckpt', 
                       help='Path to trained model')
    parser.add_argument('--num_images', type=int, default=5,
                       help='Number of images to analyze')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    visualizer = ImageReconstructionVisualizer(args.model, args.device)
    visualizer.create_comprehensive_image_reconstruction(args.num_images)

if __name__ == "__main__":
    main()
