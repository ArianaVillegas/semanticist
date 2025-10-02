"""
Test caption generation on REAL COCO images
(debug_model.py uses random noise, which doesn't work)
"""
import torch
from models.slot_coca import SlotCoCa
from coco_dataset import COCOCaptionsDataset
from pathlib import Path
import random

print("="*60)
print("COCO CAPTION GENERATION TEST")
print("="*60)

# Load checkpoint
checkpoint_path = "checkpoints/coco_small/best_model.pt"
print(f"\n1. Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Get training args
args = checkpoint['args']
print(f"\n2. Training config:")
print(f"   num_slots: {args['num_slots']}")
print(f"   lambda_caption: {args['lambda_caption']}")
print(f"   epochs trained: {checkpoint['epoch']}")

# Create model
print(f"\n3. Creating model...")
model = SlotCoCa(
    num_slots=args['num_slots'],
    num_layers=args.get('num_layers', 3),
    encoder_name=args.get('encoder_name', 'vit_base_patch16_dinov3'),
    projection_dim=args.get('projection_dim', 256),
    temperature=args.get('temperature', 0.07)
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"   ✅ Model loaded")

# Load COCO validation dataset
print(f"\n4. Loading COCO validation set...")
dataset = COCOCaptionsDataset(
    root_dir=Path(args['data_dir']) / 'val2017',
    ann_file=Path(args['data_dir']) / 'annotations' / 'captions_val2017.json',
    captions_per_image=5
)
print(f"   ✅ Loaded {len(dataset)} samples")

# Test on random real images
print(f"\n5. Generating captions on REAL COCO images...")
print("="*60)

num_samples = 10
indices = random.sample(range(len(dataset)), num_samples)

with torch.no_grad():
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample['image'].unsqueeze(0)  # Add batch dim
        reference = sample['caption']
        
        # Generate caption
        generated_ids = model.generate_caption(image, max_length=20)
        generated_text = model.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        print(f"\nSample {i+1}:")
        print(f"  Reference: {reference}")
        print(f"  Generated: '{generated_text}'")
        print(f"  Token IDs: {generated_ids[0].tolist()[:10]}...")  # First 10 tokens
        
        # Check if generation worked
        if len(generated_text.strip()) == 0:
            print(f"  ⚠️  Empty caption!")
        elif len(generated_ids[0]) <= 2:
            print(f"  ⚠️  Only start/end tokens!")
        else:
            print(f"  ✅ Generated {len(generated_text.split())} words")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)

# Summary
print("\n📊 Summary:")
print(f"   Checkpoint: {checkpoint_path}")
print(f"   Epochs trained: {checkpoint['epoch']}")
print(f"   Dataset: COCO validation")
print(f"   Samples tested: {num_samples}")
print("\nIf all captions are empty:")
print("  → Caption decoder didn't learn (despite low loss)")
print("  → Possible issue: Training used wrong mode or decoder wasn't used")
print("\nIf captions have words:")
print("  → Caption decoder IS working!")
print("  → debug_model.py failed because it used random noise, not real images")
