import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import os
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import SlotFormer


class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(TensorEncoder, self).default(obj)


class FeatureSpaceAnalyzer:
    def __init__(self, model_path, results_dir, device="cuda"):
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.model_path = model_path
        self.model = self._load_model(model_path)
        self.pca = None
        print(f"FeatureSpaceAnalyzer initialized on {device}.")

    def _load_model(self, model_path):
        model = SlotFormer(num_slots=128, num_layers=3, encoder_name="vit_base_patch16_dinov3")

        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = (
                checkpoint.get('model')
                if isinstance(checkpoint, dict) and 'model' in checkpoint
                else checkpoint.get('state_dict')
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint
                else checkpoint
            )

            if isinstance(state_dict, dict) and any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

            load_res = model.load_state_dict(state_dict, strict=False)
            missing = sorted(list(load_res.missing_keys)) if hasattr(load_res, 'missing_keys') else []
            unexpected = sorted(list(load_res.unexpected_keys)) if hasattr(load_res, 'unexpected_keys') else []
            if missing or unexpected:
                print(f"⚠️ Partial checkpoint load from {model_path}")
                if missing:
                    print(f"   Missing keys ({len(missing)}): {missing}")
                if unexpected:
                    print(f"   Unexpected keys ({len(unexpected)}): {unexpected}")
            else:
                print(f"✓ Loaded model from {model_path}")
        else:
            print(f"⚠️ Model not found at '{model_path}'. Using random weights.")

        return model.to(self.device).eval()

    def _get_dataset(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset_path = Path('./datasets/imagenette2/val')
        if not dataset_path.exists():
            print(f"⚠️ Imagenette dataset not found at {dataset_path}.")
            print("Please download it from https://github.com/fastai/imagenette")
            return None

        try:
            real_dataset = torchvision.datasets.ImageFolder(
                root=str(dataset_path), transform=transform
            )
            print(f"✓ Loaded Imagenette dataset with {len(real_dataset)} images from {len(real_dataset.classes)} classes.")
        except Exception as e:
            print(f"⚠️ Could not load Imagenette: {e}.")
            real_dataset = None

        return real_dataset

    def fit_pca(self, dataset, num_samples=100):
        all_features = []

        with torch.no_grad():
            for i in range(min(num_samples, len(dataset))):
                image, _ = dataset[i]
                image = image.unsqueeze(0).to(self.device)

                patch_tokens = self.model.encoder.forward_features(image)
                patch_tokens = patch_tokens[:, self.model.encoder.num_prefix_tokens:]

                sampled_patches = patch_tokens[0, ::4].cpu().numpy()  # Every 4th patch
                all_features.append(sampled_patches)

        all_features = np.concatenate(all_features, axis=0)

        self.pca = PCA(n_components=3)
        self.pca.fit(all_features)

        explained_var = self.pca.explained_variance_ratio_
        print(f"✓ PCA fitted. Explained variance: {explained_var.sum():.3f}")

    def features_to_rgb(self, patches):
        if self.pca is None:
            raise RuntimeError("PCA must be fitted before visualizing features. Call `fit_pca()`.")

        B, num_patches, embed_dim = patches.shape
        patches_flat = patches.view(-1, embed_dim).cpu().numpy()
        rgb_patches = self.pca.transform(patches_flat)

        for i in range(3):
            channel = rgb_patches[:, i]
            p5, p95 = np.percentile(channel, [5, 95])
            rgb_patches[:, i] = np.clip((channel - p5) / (p95 - p5 + 1e-6), 0, 1)

        patch_side_len = int(np.sqrt(num_patches))
        return rgb_patches.reshape(B, patch_side_len, patch_side_len, 3)

    def analyze_image(self, image, slot_counts, store_reconstructions=True):
        with torch.no_grad():
            gt_features = self.model.encoder.forward_features(image)
            gt_patches = gt_features[:, self.model.encoder.num_prefix_tokens:]

            all_slots = self.model.generator(
                tgt=self.model.slot_queries.repeat(1, 1, 1),
                memory=gt_patches,
                tgt_mask=self.model.tgt_mask.to(self.device)
            )

            results = {'losses': {}, 'similarities': {}}
            if store_reconstructions:
                results['reconstructions'] = {}

            for num_slots in slot_counts:
                used_slots = all_slots[:, :num_slots]

                recon_patches = self.model.reconstructor(
                    tgt=self.model.decoder_pos_embed.repeat(1, 1, 1),
                    memory=used_slots
                )

                results['losses'][num_slots] = F.mse_loss(recon_patches, gt_patches).item()

                gt_norm = F.normalize(gt_patches[0], dim=-1)
                recon_norm = F.normalize(recon_patches[0], dim=-1)
                results['similarities'][num_slots] = torch.sum(gt_norm * recon_norm, dim=-1).mean().item()

                if store_reconstructions:
                    results['reconstructions'][num_slots] = recon_patches

            return results, gt_patches

    def create_visualizations(self, image_idx, label, results, gt_patches, image, slot_counts):
        fig, axes = plt.subplots(3, len(slot_counts), figsize=(len(slot_counts) * 4, 16))

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        orig_img_view = torch.clamp(image[0].cpu() * std + mean, 0, 1)
        gt_rgb = self.features_to_rgb(gt_patches)

        for i, num_slots in enumerate(slot_counts):
            axes[0, i].imshow(orig_img_view.permute(1, 2, 0))
            axes[0, i].set_title(f'Original Image\n(Label: {label})', fontsize=12)
            axes[0, i].axis('off')

            axes[1, i].imshow(gt_rgb[0])
            axes[1, i].set_title(f'GT Features (PCA)', fontsize=12)
            axes[1, i].axis('off')

            recon_rgb = self.features_to_rgb(results['reconstructions'][num_slots])
            loss = results['losses'][num_slots]
            sim = results['similarities'][num_slots]
            axes[2, i].imshow(recon_rgb[0])
            axes[2, i].set_title(f'Reconstruction ({num_slots} slots)\nLoss: {loss:.4f} | Sim: {sim:.3f}', fontsize=12)
            axes[2, i].axis('off')

        plt.tight_layout()
        plt.savefig(self.results_dir / f'img_{image_idx}_{label}_grid.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

    def _select_showcase_samples(self, dataset, num_images):
        samples = []
        if num_images <= 0:
            return samples

        if hasattr(dataset, 'targets'):
            targets = np.array(dataset.targets)
            unique_classes = np.unique(targets)
            selected_classes = np.random.choice(unique_classes, size=min(num_images, len(unique_classes)), replace=False)

            for class_idx in selected_classes:
                candidate_indices = np.where(targets == class_idx)[0]
                image_idx = int(np.random.choice(candidate_indices))
                image, label = dataset[image_idx]
                class_name = dataset.classes[label] if hasattr(dataset, 'classes') else int(label)
                samples.append((image, class_name, image_idx))
        else:
            indices = np.random.choice(len(dataset), size=min(num_images, len(dataset)), replace=False)
            for image_idx in indices:
                image, label = dataset[int(image_idx)]
                samples.append((image, label, int(image_idx)))

        return samples

    def run_full_analysis(self, num_images=20):
        dataset = self._get_dataset()
        if not dataset:
            print("❌ No data available. Aborting analysis.")
            return

        self.fit_pca(dataset, num_samples=min(100, len(dataset)))

        slot_counts = [1, 2, 4, 8, 16, 32, 64, 128]

        agg_storage = {s: {'losses': [], 'similarities': []} for s in slot_counts}
        monotonic_ratios = []
        improvement_factors = []

        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        print(f"\nAggregating reconstruction metrics across {len(dataset)} samples...")
        with torch.no_grad():
            for idx, (image, _) in enumerate(loader):
                image = image.to(self.device)

                results, _ = self.analyze_image(image, slot_counts, store_reconstructions=False)

                losses = [results['losses'][s] for s in slot_counts]
                sims = [results['similarities'][s] for s in slot_counts]

                for s, loss_val, sim_val in zip(slot_counts, losses, sims):
                    agg_storage[s]['losses'].append(loss_val)
                    agg_storage[s]['similarities'].append(sim_val)

                decreases = sum(1 for i in range(1, len(losses)) if losses[i] < losses[i - 1])
                monotonic_ratios.append(decreases / (len(losses) - 1))
                improvement = losses[0] / max(losses[-1], 1e-9)
                improvement_factors.append(improvement)

                if (idx + 1) % 200 == 0 or (idx + 1) == len(loader):
                    print(f"  Processed {idx + 1}/{len(loader)} samples")

        agg_metrics = {
            s: {
                'loss_mean': float(np.mean(values['losses'])),
                'loss_std': float(np.std(values['losses'])),
                'similarity_mean': float(np.mean(values['similarities'])),
                'similarity_std': float(np.std(values['similarities'])),
            }
            for s, values in agg_storage.items()
        }

        showcase_results = []
        showcase_samples = self._select_showcase_samples(dataset, num_images)
        if showcase_samples:
            print(f"\nGenerating visualizations for {len(showcase_samples)} showcase samples...")
        for image, label, image_idx in showcase_samples:
            batched_image = image.unsqueeze(0).to(self.device)
            results, gt_patches = self.analyze_image(batched_image, slot_counts, store_reconstructions=True)
            self.create_visualizations(image_idx, label, results, gt_patches, batched_image, slot_counts)

            # Drop heavy tensors before serializing results
            results.pop('reconstructions', None)
            showcase_results.append({
                'label': label,
                'image_idx': int(image_idx),
                'metrics': {
                    'losses': {s: float(results['losses'][s]) for s in slot_counts},
                    'similarities': {s: float(results['similarities'][s]) for s in slot_counts},
                }
            })

        self._create_summary_plots(slot_counts, agg_metrics, monotonic_ratios, improvement_factors)
        self._save_results_summary(slot_counts, agg_metrics, monotonic_ratios, improvement_factors, len(dataset), showcase_results)

        print(f"\n✅ Analysis complete! Results saved to: {self.results_dir.resolve()}")

    def _create_summary_plots(self, slot_counts, agg_metrics, monotonic_ratios, improvement_factors):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Space Analysis Summary', fontsize=16)

        avg_losses = [agg_metrics[s]['loss_mean'] for s in slot_counts]
        axes[0, 0].loglog(slot_counts, avg_losses, 'b-o')
        axes[0, 0].set_title('Average Reconstruction Loss vs. Slots')
        axes[0, 0].set_xlabel('Number of Slots')
        axes[0, 0].set_ylabel('MSE Loss (log scale)')
        axes[0, 0].grid(True, which="both", ls="--", alpha=0.5)

        avg_sims = [agg_metrics[s]['similarity_mean'] for s in slot_counts]
        axes[0, 1].semilogx(slot_counts, avg_sims, 'g-s')
        axes[0, 1].set_title('Average Feature Similarity vs. Slots')
        axes[0, 1].set_xlabel('Number of Slots')
        axes[0, 1].set_ylabel('Cosine Similarity')
        axes[0, 1].grid(True, which="both", ls="--", alpha=0.5)
        axes[0, 1].set_ylim(0, 1)

        axes[1, 0].hist(monotonic_ratios, bins=20, color='purple', alpha=0.75)
        axes[1, 0].set_title('Monotonic Improvement Ratio Distribution')
        axes[1, 0].set_xlabel('Ratio of Improving Steps')
        axes[1, 0].set_ylabel('Image Count')
        axes[1, 0].axvline(np.mean(monotonic_ratios), color='r', ls='--', label=f'Mean: {np.mean(monotonic_ratios):.3f}')
        axes[1, 0].legend()
        axes[1, 0].set_xlim(0, 1)

        axes[1, 1].hist(improvement_factors, bins=20, color='orange', alpha=0.75)
        axes[1, 1].set_title('Loss Improvement (1 Slot vs Max Slots)')
        axes[1, 1].set_xlabel('Improvement Factor (x better)')
        axes[1, 1].set_ylabel('Image Count')
        axes[1, 1].axvline(np.mean(improvement_factors), color='r', ls='--', label=f'Mean: {np.mean(improvement_factors):.2f}x')
        axes[1, 1].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.results_dir / 'summary_plots.png', dpi=200)
        plt.close(fig)

    def _save_results_summary(self, slot_counts, agg_metrics, monotonic_ratios, improvement_factors, dataset_size, showcase_results):
        monotonic_ratios = np.array(monotonic_ratios)
        improvement_factors = np.array(improvement_factors)

        summary_data = {
            'timestamp': time.time(),
            'model_path': self.model_path,
            'dataset_size': dataset_size,
            'slot_counts': slot_counts,
            'pca_explained_variance': self.pca.explained_variance_ratio_.tolist() if self.pca is not None else None,
            'full_dataset_metrics': {
                int(s): {
                    'loss_mean': agg_metrics[s]['loss_mean'],
                    'loss_std': agg_metrics[s]['loss_std'],
                    'similarity_mean': agg_metrics[s]['similarity_mean'],
                    'similarity_std': agg_metrics[s]['similarity_std'],
                }
                for s in slot_counts
            },
            'monotonic_ratio': {
                'mean': float(monotonic_ratios.mean()),
                'std': float(monotonic_ratios.std()),
                'min': float(monotonic_ratios.min()),
                'max': float(monotonic_ratios.max()),
            },
            'loss_improvement_factor': {
                'mean': float(improvement_factors.mean()),
                'std': float(improvement_factors.std()),
                'min': float(improvement_factors.min()),
                'max': float(improvement_factors.max()),
            },
            'showcase_samples': showcase_results,
        }

        with open(self.results_dir / 'analysis_summary.json', 'w') as f:
            json.dump(summary_data, f, indent=2, cls=TensorEncoder)
