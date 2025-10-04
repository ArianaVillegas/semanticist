"""
Test 2: Does decoder cross-attention actually use visual slots?
Compare: normal generation vs. replacing slots with zeros
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

sample = dataset[0]
image = sample['image'].unsqueeze(0)

print("="*60)
print("TEST 2: Cross-Attention Usage")
print("="*60)
print(f"\nGround truth: {sample['caption']}\n")

# Test 1: Normal generation with real visual slots
with torch.no_grad():
    generated_ids = model.generate_caption(image, max_length=10)
    normal_caption = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(f"With REAL slots:  '{normal_caption}'")

# Test 2: Replace slots with zeros (no visual information)
with torch.no_grad():
    # Encode image to get slot shape
    real_slots, _ = model.encode_image(image)
    
    # Create zero slots (no visual information)
    zero_slots = torch.zeros_like(real_slots)
    
    # Manually run generation with zero slots
    input_ids = torch.tensor([[model.tokenizer.cls_token_id]])
    
    for step in range(10):
        outputs = model.text_encoder(input_ids=input_ids)
        caption_embeds = outputs.last_hidden_state
        
        seq_len = caption_embeds.size(1)
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len)
        
        # Use ZERO slots instead of real visual features
        decoded = model.caption_decoder(
            tgt=caption_embeds,
            memory=zero_slots,
            tgt_mask=causal_mask
        )
        
        logits = model.caption_head(decoded[:, -1, :])
        
        # Apply same masking logic
        if step < 3:
            logits[:, model.tokenizer.sep_token_id] = -float('inf')
        
        if input_ids.size(1) > 1:
            for prev_token in input_ids[0, 1:]:
                logits[:, prev_token] /= 1.5
        
        next_token = logits.argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        if (next_token == model.tokenizer.sep_token_id).all():
            break
    
    zero_caption = model.tokenizer.decode(input_ids[0], skip_special_tokens=True)

print(f"With ZERO slots:  '{zero_caption}'")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)
if normal_caption == zero_caption:
    print("❌ FAIL: Captions are IDENTICAL!")
    print("   → Cross-attention is NOT using visual slots")
    print("   → Decoder only uses text context, ignores images")
else:
    print("✅ PASS: Captions are DIFFERENT")
    print("   → Cross-attention uses visual information")
