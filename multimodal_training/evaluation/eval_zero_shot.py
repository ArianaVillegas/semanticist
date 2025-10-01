"""
Zero-shot classification evaluation
Compares Slot-CoCa against CLIP and CoCa
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import argparse
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.slot_coca import SlotCoCa
from imagenette_captions_dataset import ImagenetteWithCaptions

# Try to import CLIP for comparison
try:
    import clip
    CLIP_AVAILABLE = True
except:
    CLIP_AVAILABLE = False
    print("⚠️  CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")


class ZeroShotEvaluator:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        
        # Load Slot-CoCa
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Load CLIP for comparison if available
        if CLIP_AVAILABLE:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            self.clip_model.eval()
    
    def _load_model(self, model_path):
        """Load trained Slot-CoCa model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get args from checkpoint
        args = checkpoint.get('args', {})
        model = SlotCoCa(
            num_slots=args.get('num_slots', 128),
            num_layers=args.get('num_layers', 3),
            encoder_name=args.get('encoder_name', 'vit_base_patch16_dinov3'),
            projection_dim=args.get('projection_dim', 256)
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded Slot-CoCa from {model_path}")
        
        return model
    
    @torch.no_grad()
    def evaluate_imagenet(self, data_dir, batch_size=256, num_samples=None, use_imagenette=False):
        """
        Evaluate zero-shot classification on ImageNet or Imagenette
        
        Args:
            data_dir: Path to ImageNet/Imagenette validation set
            batch_size: Batch size for evaluation
            num_samples: Limit number of samples (for quick testing)
            use_imagenette: If True, use Imagenette classes
        """
        dataset_name = "Imagenette" if use_imagenette else "ImageNet"
        print("\n" + "="*60)
        print(f"ZERO-SHOT {dataset_name.upper()} CLASSIFICATION")
        print("="*60)
        
        # Load dataset
        if use_imagenette:
            # Use Imagenette dataset
            imagenette_dataset = ImagenetteWithCaptions(
                imagenette_root=data_dir,
                split='val',
                captions_per_image=1  # Only need images, not captions
            )
            # Just use the underlying ImageFolder
            dataset = imagenette_dataset.image_dataset
            if num_samples:
                dataset = torch.utils.data.Subset(dataset, range(min(num_samples, len(dataset))))
        else:
            # Load full ImageNet validation set
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
            if num_samples:
                dataset = torch.utils.data.Subset(dataset, range(min(num_samples, len(dataset))))
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Create text prompts for all classes
        if use_imagenette:
            # Imagenette class names
            class_mapping = {
                'n01440764': 'tench',
                'n02102040': 'English springer spaniel',
                'n02979186': 'cassette player',
                'n03000684': 'chain saw',
                'n03028079': 'church',
                'n03394916': 'French horn',
                'n03417042': 'garbage truck',
                'n03425413': 'gas pump',
                'n03445777': 'golf ball',
                'n03888257': 'parachute'
            }
            class_names = [class_mapping.get(c, c.replace('_', ' ')) for c in dataset.classes]
        else:
            class_names = [name.replace('_', ' ') for name in dataset.classes]
        text_prompts = [f"a photo of a {name}" for name in class_names]
        
        # Encode text prompts (Slot-CoCa)
        text_tokens = self.model.tokenizer(
            text_prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors='pt'
        ).to(self.device)
        
        text_features = self.model.encode_text(text_tokens)
        text_embeds = self.model.text_projection(text_features[:, 0])
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        # Encode text with CLIP if available
        if CLIP_AVAILABLE:
            clip_text = clip.tokenize(text_prompts).to(self.device)
            clip_text_features = self.clip_model.encode_text(clip_text)
            clip_text_features = F.normalize(clip_text_features, dim=-1)
        
        # Evaluate
        correct_slotcoca = 0
        correct_clip = 0
        total = 0
        
        top5_correct_slotcoca = 0
        top5_correct_clip = 0
        
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size_actual = images.shape[0]
            
            # Slot-CoCa predictions
            slots, _ = self.model.encode_image(images)
            vision_features = slots.mean(dim=1)
            vision_embeds = self.model.vision_projection(vision_features)
            vision_embeds = F.normalize(vision_embeds, dim=-1)
            
            # Compute similarity
            logits = vision_embeds @ text_embeds.t()
            preds = logits.argmax(dim=-1)
            correct_slotcoca += (preds == labels).sum().item()
            
            # Top-5 accuracy
            _, top5_preds = logits.topk(5, dim=-1)
            top5_correct_slotcoca += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
            
            # CLIP predictions
            if CLIP_AVAILABLE:
                clip_images = F.interpolate(images, size=(224, 224), mode='bilinear')
                clip_image_features = self.clip_model.encode_image(clip_images)
                clip_image_features = F.normalize(clip_image_features, dim=-1)
                
                clip_logits = clip_image_features @ clip_text_features.t()
                clip_preds = clip_logits.argmax(dim=-1)
                correct_clip += (clip_preds == labels).sum().item()
                
                _, top5_clip_preds = clip_logits.topk(5, dim=-1)
                top5_correct_clip += (top5_clip_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
            
            total += batch_size_actual
        
        # Results
        slotcoca_acc = 100 * correct_slotcoca / total
        slotcoca_top5 = 100 * top5_correct_slotcoca / total
        
        print(f"\n📊 Results ({total} images):")
        print(f"\nSlot-CoCa:")
        print(f"  Top-1 Accuracy: {slotcoca_acc:.2f}%")
        print(f"  Top-5 Accuracy: {slotcoca_top5:.2f}%")
        
        if CLIP_AVAILABLE:
            clip_acc = 100 * correct_clip / total
            clip_top5 = 100 * top5_correct_clip / total
            print(f"\nCLIP (ViT-B/32) Baseline:")
            print(f"  Top-1 Accuracy: {clip_acc:.2f}%")
            print(f"  Top-5 Accuracy: {clip_top5:.2f}%")
            print(f"\nGap: {clip_acc - slotcoca_acc:+.2f}% (Top-1)")
        
        return {
            'slotcoca_top1': slotcoca_acc,
            'slotcoca_top5': slotcoca_top5,
            'clip_top1': clip_acc if CLIP_AVAILABLE else None,
            'clip_top5': clip_top5 if CLIP_AVAILABLE else None
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained Slot-CoCa checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to ImageNet/Imagenette val directory')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_samples', type=int, default=None, help='Limit samples for quick test')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_imagenette', action='store_true', help='Use Imagenette dataset (10 classes)')
    
    args = parser.parse_args()
    
    evaluator = ZeroShotEvaluator(args.model_path, args.device)
    results = evaluator.evaluate_imagenet(
        args.data_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        use_imagenette=args.use_imagenette
    )
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
