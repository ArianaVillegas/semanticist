"""
Image-Text Retrieval Evaluation
Measures Recall@K for image→text and text→image retrieval
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.slot_coca import SlotCoCa
from imagenette_captions_dataset import ImagenetteWithCaptions, collate_fn

try:
    import clip
    CLIP_AVAILABLE = True
except:
    CLIP_AVAILABLE = False


class RetrievalEvaluator:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()
        
        if CLIP_AVAILABLE:
            self.clip_model, _ = clip.load("ViT-B/32", device=device)
            self.clip_model.eval()
    
    def _load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        args = checkpoint.get('args', {})
        
        model = SlotCoCa(
            num_slots=args.get('num_slots', 128),
            num_layers=args.get('num_layers', 3),
            encoder_name=args.get('encoder_name', 'vit_base_patch16_dinov3'),
            projection_dim=args.get('projection_dim', 256)
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    @torch.no_grad()
    def compute_embeddings(self, loader):
        """Compute embeddings for all images and captions"""
        image_embeds = []
        text_embeds = []
        
        for batch in tqdm(loader, desc="Computing embeddings"):
            images = batch['images'].to(self.device)
            captions = batch['captions']
            
            # Encode images
            slots, _ = self.model.encode_image(images)
            vision_features = slots.mean(dim=1)
            vision_embeds = self.model.vision_projection(vision_features)
            vision_embeds = F.normalize(vision_embeds, dim=-1)
            image_embeds.append(vision_embeds.cpu())
            
            # Encode text
            text_tokens = self.model.tokenizer(
                captions,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors='pt'
            ).to(self.device)
            
            text_features = self.model.encode_text(text_tokens)
            text_proj = self.model.text_projection(text_features[:, 0])
            text_proj = F.normalize(text_proj, dim=-1)
            text_embeds.append(text_proj.cpu())
        
        image_embeds = torch.cat(image_embeds, dim=0)
        text_embeds = torch.cat(text_embeds, dim=0)
        
        return image_embeds, text_embeds
    
    def compute_recall(self, image_embeds, text_embeds, ks=[1, 5, 10]):
        """
        Compute Recall@K for both image→text and text→image retrieval
        
        Args:
            image_embeds: [N, D] image embeddings
            text_embeds: [N, D] text embeddings
            ks: List of K values for Recall@K
        """
        # Compute similarity matrix
        similarity = image_embeds @ text_embeds.t()  # [N, N]
        
        results = {}
        
        # Image→Text Retrieval
        for k in ks:
            # For each image, get top-k most similar texts
            _, top_k_indices = similarity.topk(k, dim=1)
            
            # Check if correct text (same index) is in top-k
            correct = 0
            for i in range(len(similarity)):
                if i in top_k_indices[i]:
                    correct += 1
            
            recall = 100 * correct / len(similarity)
            results[f'i2t_R@{k}'] = recall
        
        # Text→Image Retrieval  
        for k in ks:
            _, top_k_indices = similarity.t().topk(k, dim=1)
            
            correct = 0
            for i in range(len(similarity)):
                if i in top_k_indices[i]:
                    correct += 1
            
            recall = 100 * correct / len(similarity)
            results[f't2i_R@{k}'] = recall
        
        return results
    
    def evaluate(self, data_dir, split='val', batch_size=128, num_samples=5000, use_imagenette=False):
        """Run full retrieval evaluation"""
        print("\n" + "="*60)
        print("IMAGE-TEXT RETRIEVAL EVALUATION")
        print("="*60)
        
        # Load dataset
        if use_imagenette:
            print("Using Imagenette dataset...")
            # Use real captions if available
            captions_file = os.path.join(os.path.dirname(data_dir), 'multimodal_training', 'data', 'imagenette_captions', 'imagenette_captions.json')
            if not os.path.exists(captions_file):
                captions_file = None
            
            dataset = ImagenetteWithCaptions(
                imagenette_root=data_dir,
                split=split,
                captions_per_image=1,
                captions_file=captions_file
            )
            if num_samples and num_samples < len(dataset):
                import torch
                dataset = torch.utils.data.Subset(dataset, range(num_samples))
        else:
            from imagenet_captions_dataset import ImageNetCaptionsDataset
            dataset = ImageNetCaptionsDataset(
                root_dir=data_dir,
                split=split,
                subset_size=num_samples
            )
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        # Compute embeddings
        print("\nSlot-CoCa:")
        image_embeds, text_embeds = self.compute_embeddings(loader)
        
        # Compute metrics
        results = self.compute_recall(image_embeds, text_embeds)
        
        print(f"\n📊 Results ({len(dataset)} samples):")
        print("\nImage → Text Retrieval:")
        for k in [1, 5, 10]:
            print(f"  Recall@{k}: {results[f'i2t_R@{k}']:.2f}%")
        
        print("\nText → Image Retrieval:")
        for k in [1, 5, 10]:
            print(f"  Recall@{k}: {results[f't2i_R@{k}']:.2f}%")
        
        # Compare with CLIP if available
        if CLIP_AVAILABLE:
            print("\n" + "-"*60)
            print("CLIP Baseline (ViT-B/32):")
            clip_results = self.evaluate_clip(loader)
            
            print("\nImage → Text Retrieval:")
            for k in [1, 5, 10]:
                print(f"  Recall@{k}: {clip_results[f'i2t_R@{k}']:.2f}%")
            
            print("\nText → Image Retrieval:")
            for k in [1, 5, 10]:
                print(f"  Recall@{k}: {clip_results[f't2i_R@{k}']:.2f}%")
            
            # Compute gaps
            print("\n" + "-"*60)
            print("Gap (CLIP - Slot-CoCa):")
            print("\nImage → Text:")
            for k in [1, 5, 10]:
                gap = clip_results[f'i2t_R@{k}'] - results[f'i2t_R@{k}']
                print(f"  Recall@{k}: {gap:+.2f}%")
            
            print("\nText → Image:")
            for k in [1, 5, 10]:
                gap = clip_results[f't2i_R@{k}'] - results[f't2i_R@{k}']
                print(f"  Recall@{k}: {gap:+.2f}%")
        
        return results
    
    @torch.no_grad()
    def evaluate_clip(self, loader):
        """Evaluate CLIP baseline"""
        image_embeds = []
        text_embeds = []
        
        for batch in tqdm(loader, desc="CLIP embeddings"):
            images = batch['images'].to(self.device)
            captions = batch['captions']
            
            # CLIP image encoding
            images_clip = F.interpolate(images, size=(224, 224), mode='bilinear')
            img_feats = self.clip_model.encode_image(images_clip)
            img_feats = F.normalize(img_feats, dim=-1)
            image_embeds.append(img_feats.cpu())
            
            # CLIP text encoding
            text_tokens = clip.tokenize(captions, truncate=True).to(self.device)
            txt_feats = self.clip_model.encode_text(text_tokens)
            txt_feats = F.normalize(txt_feats, dim=-1)
            text_embeds.append(txt_feats.cpu())
        
        image_embeds = torch.cat(image_embeds, dim=0)
        text_embeds = torch.cat(text_embeds, dim=0)
        
        return self.compute_recall(image_embeds, text_embeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_imagenette', action='store_true', help='Use Imagenette dataset')
    
    args = parser.parse_args()
    
    evaluator = RetrievalEvaluator(args.model_path, args.device)
    evaluator.evaluate(
        args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        use_imagenette=args.use_imagenette
    )
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
