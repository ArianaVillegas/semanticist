"""Test if model produces same caption for different images"""
import torch
import sys
sys.path.insert(0, '/media/ariana/exp/Wonderland/semanticist/multimodal_training')

from models.slot_coca import SlotCoCa
from coco_dataset import COCOCaptionsDataset
from pathlib import Path

checkpoint = torch.load('checkpoints/coco_test_retraining/checkpoint_epoch_4.pt', map_location='cpu')
model = SlotCoCa(num_slots=128)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

dataset = COCOCaptionsDataset(
    root_dir=Path('/home/avillegas/semanticist/datasets/coco/val2017'),
    ann_file=Path('/home/avillegas/semanticist/datasets/coco/annotations/captions_val2017.json')
)

print("Testing 5 different images...\n")
captions = []

for i in range(5):
    sample = dataset[i * 100]
    image = sample['image'].unsqueeze(0)
    
    with torch.no_grad():
        generated_ids = model.generate_caption(image, max_length=10)
        generated = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    captions.append(generated)
    print(f"Image {i+1}: '{generated}' (GT: {sample['caption'][:40]}...)")

print(f"\nUnique captions: {len(set(captions))}/5")
if len(set(captions)) == 1:
    print("❌ Model ignores visual input!")
else:
    print("✅ Model uses visual input")
