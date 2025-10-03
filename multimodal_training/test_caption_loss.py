"""
Test 3: Is caption loss computed correctly?
Test: Does loss decrease when given correct captions vs random?
"""
import torch
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

print("="*60)
print("TEST 3: Caption Loss Computation")
print("="*60)

# Get a batch of samples
samples = [dataset[i] for i in range(4)]
images = torch.stack([s['image'] for s in samples])
correct_captions = [s['caption'] for s in samples]
wrong_captions = ["random text here" for _ in samples]

# Test with CORRECT captions
with torch.no_grad():
    correct_tokens = model.tokenizer(
        correct_captions,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors='pt'
    )
    outputs_correct = model(images, caption_tokens=correct_tokens, mode='caption')
    loss_correct = outputs_correct['caption_loss'].item()

# Test with WRONG/RANDOM captions
with torch.no_grad():
    wrong_tokens = model.tokenizer(
        wrong_captions,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors='pt'
    )
    outputs_wrong = model(images, caption_tokens=wrong_tokens, mode='caption')
    loss_wrong = outputs_wrong['caption_loss'].item()

print(f"\nLoss with CORRECT captions: {loss_correct:.4f}")
print(f"Loss with WRONG captions:   {loss_wrong:.4f}")
print(f"Difference: {loss_wrong - loss_correct:.4f}")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

if abs(loss_correct - loss_wrong) < 0.1:
    print("❌ FAIL: Losses are nearly identical!")
    print("   → Model hasn't learned to distinguish good/bad captions")
    print("   → Caption decoder didn't train properly")
elif loss_correct < loss_wrong:
    print("✅ PASS: Correct captions have lower loss")
    print(f"   → Model learned caption task ({loss_wrong/loss_correct:.2f}x difference)")
else:
    print("⚠️  WEIRD: Wrong captions have LOWER loss!")
    print("   → Something is very wrong with training")
