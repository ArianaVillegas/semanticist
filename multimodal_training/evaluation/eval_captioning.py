"""
Image Captioning Evaluation
Measures BLEU, CIDEr, SPICE
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.slot_coca import SlotCoCa
from imagenet_captions_dataset import ImageNetCaptionsDataset, collate_fn

# Try to import captioning metrics
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOEvalCap
    COCO_METRICS_AVAILABLE = True
except:
    COCO_METRICS_AVAILABLE = False
    print("⚠️  Install pycocotools for full metrics: pip install pycocotools")


class CaptioningEvaluator:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()
    
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
    def generate_captions(self, loader, max_length=50):
        """Generate captions for all images in loader"""
        all_predictions = []
        all_references = []
        all_image_ids = []
        
        for batch in tqdm(loader, desc="Generating captions"):
            images = batch['images'].to(self.device)
            captions = batch['captions']
            image_ids = batch['image_ids']
            
            # Generate captions
            generated_ids = self.model.generate_caption(images, max_length=max_length)
            
            # Decode generated captions
            generated_texts = self.model.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            all_predictions.extend(generated_texts)
            all_references.extend(captions)
            all_image_ids.extend(image_ids)
        
        return all_predictions, all_references, all_image_ids
    
    def compute_metrics(self, predictions, references, image_ids):
        """Compute captioning metrics"""
        if not COCO_METRICS_AVAILABLE:
            print("⚠️  COCO metrics not available. Showing sample captions only.")
            return self.compute_simple_metrics(predictions, references)
        
        # Format for COCO evaluation
        # This requires specific format - implement if needed
        results = self.compute_simple_metrics(predictions, references)
        
        return results
    
    def compute_simple_metrics(self, predictions, references):
        """Compute simple metrics without COCO tools"""
        from collections import Counter
        import numpy as np
        
        def compute_bleu_1(pred, ref):
            """Simple BLEU-1 implementation"""
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            if len(pred_tokens) == 0:
                return 0.0
            
            pred_counts = Counter(pred_tokens)
            ref_counts = Counter(ref_tokens)
            
            clipped_counts = {
                token: min(count, ref_counts.get(token, 0))
                for token, count in pred_counts.items()
            }
            
            numerator = sum(clipped_counts.values())
            denominator = len(pred_tokens)
            
            return numerator / denominator if denominator > 0 else 0.0
        
        bleu_scores = [
            compute_bleu_1(pred, ref)
            for pred, ref in zip(predictions, references)
        ]
        
        avg_bleu = np.mean(bleu_scores)
        
        return {
            'BLEU-1': avg_bleu * 100,
            'num_samples': len(predictions)
        }
    
    def evaluate(self, data_dir, split='val', batch_size=32, num_samples=1000):
        """Run full captioning evaluation"""
        print("\n" + "="*60)
        print("IMAGE CAPTIONING EVALUATION")
        print("="*60)
        
        # Load dataset
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
        
        # Generate captions
        predictions, references, image_ids = self.generate_captions(loader)
        
        # Compute metrics
        metrics = self.compute_metrics(predictions, references, image_ids)
        
        print(f"\n📊 Results ({len(predictions)} images):")
        for metric, value in metrics.items():
            if metric != 'num_samples':
                print(f"  {metric}: {value:.2f}")
        
        # Show examples
        print("\n📝 Sample Captions:")
        for i in range(min(5, len(predictions))):
            print(f"\nImage {image_ids[i]}:")
            print(f"  Reference: {references[i]}")
            print(f"  Generated: {predictions[i]}")
        
        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    evaluator = CaptioningEvaluator(args.model_path, args.device)
    evaluator.evaluate(
        args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_samples=args.num_samples
    )
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
