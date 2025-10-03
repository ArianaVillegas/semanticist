"""
Test 1: Does the model produce different captions for different images?
If all images get the same caption → model ignores visual input
"""
import torch
from models.slot_coca import SlotCoCa
from coco_dataset import COCOCaptionsDataset
from pathlib import Path

print("="*60)
print("TEST 1: Visual Grounding")
print("="*60)

# Load checkpoint
checkpoint = torch.load('checkpoints/coco_test_retraining/checkpoint_epoch_4.pt', map_location='cpu')
model = SlotCoCa(num_slots=128)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load dataset
dataset = COCOCaptionsDataset(
    root_dir=Path('/home/avillegas/semanticist/datasets/coco/val2017'),
    ann_file=Path('/home/avillegas/semanticist/datasets/coco/annotations/captions_val2017.json')
)

print("\nTesting 5 different images...")
print("-" * 60)

captions = []
for i in range(5):
    sample = dataset[i * 100]  # Sample every 100 to get diverse images
    image = sample['image'].unsqueeze(0)
    
    # Generate caption
    with torch.no_grad():
        generated_ids = model.generate_caption(image, max_length=10)
        generated_caption = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    captions.append(generated_caption)
    print(f"\nImage {i+1}:")
    print(f"  Ground truth: {sample['caption']}")
    print(f"  Generated:    {generated_caption}")

# Check if all captions are identical
print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

unique_captions = set(captions)
if len(unique_captions) == 1:
    print("❌ FAIL: All images produce IDENTICAL captions!")
    print(f"   → Caption: '{captions[0]}'")
    print("   → Model is IGNORING visual input")
else:
    print(f"✅ PASS: Generated {len(unique_captions)} unique captions")
    print("   → Model uses visual input (at least partially)")
