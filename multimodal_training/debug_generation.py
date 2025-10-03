"""
Deep debug of caption generation
Shows what the model is actually predicting
"""
import torch
from models.slot_coca import SlotCoCa
from coco_dataset import COCOCaptionsDataset
from pathlib import Path

print("="*60)
print("GENERATION DEBUG")
print("="*60)

# Load checkpoint (use latest retraining)
checkpoint = torch.load('checkpoints/coco_test_retraining/checkpoint_epoch_4.pt', map_location='cpu')
model = SlotCoCa(num_slots=128)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load one image
dataset = COCOCaptionsDataset(
    root_dir=Path('/home/avillegas/semanticist/datasets/coco/val2017'),
    ann_file=Path('/home/avillegas/semanticist/datasets/coco/annotations/captions_val2017.json')
)
sample = dataset[0]
image = sample['image'].unsqueeze(0)

print(f"\nReference caption: {sample['caption']}")
print("\n" + "="*60)
print("STEP-BY-STEP GENERATION")
print("="*60)

with torch.no_grad():
    # Encode image
    slots, _ = model.encode_image(image)
    print(f"\n✅ Image encoded to slots: {slots.shape}")
    
    # Start with [CLS]
    input_ids = torch.full((1, 1), model.tokenizer.cls_token_id, dtype=torch.long)
    print(f"✅ Starting with [CLS] token (ID: {model.tokenizer.cls_token_id})")
    
    for step in range(5):  # Only 5 steps for debugging
        print(f"\n{'='*60}")
        print(f"Step {step + 1}")
        print(f"{'='*60}")
        print(f"Current sequence: {input_ids[0].tolist()}")
        print(f"Decoded: {model.tokenizer.decode(input_ids[0])}")
        
        # Encode current sequence
        outputs = model.text_encoder(input_ids=input_ids)
        caption_embeds = outputs.last_hidden_state
        print(f"✅ Text encoded: {caption_embeds.shape}")
        
        # Create causal mask
        seq_len = caption_embeds.size(1)
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len)
        print(f"✅ Causal mask created: {causal_mask.shape}")
        
        # Decode with cross-attention
        decoded = model.caption_decoder(
            tgt=caption_embeds,
            memory=slots,
            tgt_mask=causal_mask
        )
        print(f"✅ Decoded: {decoded.shape}")
        
        # Get logits for last position
        logits = model.caption_head(decoded[:, -1, :])
        print(f"✅ Logits: {logits.shape}")
        
        # Get top-5 predictions BEFORE masking
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs[0], k=5)
        
        print(f"\nTop 5 predictions (BEFORE masking):")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            token = model.tokenizer.decode([idx.item()])
            print(f"  {i+1}. '{token}' (ID: {idx.item()}, prob: {prob.item():.4f})")
        
        # Apply [SEP] masking for first 3 tokens
        min_length = 3
        if step < min_length:
            print(f"\n⚠️  Step {step} < {min_length}: Masking [SEP] token!")
            logits[:, model.tokenizer.sep_token_id] = -float('inf')
        
        # Repetition penalty: reduce probability of recently generated tokens
        repetition_penalty = 1.5
        if input_ids.size(1) > 1:
            print(f"\n🔄 Applying repetition penalty ({repetition_penalty}x) to previous tokens")
            for prev_token in input_ids[0, 1:]:  # Skip [CLS]
                logits[:, prev_token] /= repetition_penalty
        
        # Recompute top-5 AFTER masking and penalty
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs[0], k=5)
        
        print(f"\nTop 5 predictions (AFTER masking + penalty):")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            token = model.tokenizer.decode([idx.item()])
            print(f"  {i+1}. '{token}' (ID: {idx.item()}, prob: {prob.item():.4f})")
        
        # Predict next token
        next_token = logits.argmax(dim=-1, keepdim=True)
        next_token_str = model.tokenizer.decode([next_token[0].item()])
        print(f"\n➡️  Predicted: '{next_token_str}' (ID: {next_token[0].item()})")
        
        # Append to sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Check if it's [SEP]
        if next_token.item() == model.tokenizer.sep_token_id:
            print(f"\n⚠️  Generated [SEP]! Stopping...")
            break

print("\n" + "="*60)
print("FINAL RESULT")
print("="*60)
print(f"Generated sequence: {input_ids[0].tolist()}")
print(f"Decoded: '{model.tokenizer.decode(input_ids[0], skip_special_tokens=True)}'")
print("\n" + "="*60)
print("ANALYSIS")
print("="*60)
print("If the model predicts [SEP] at step 1:")
print("  → caption_head is biased toward [SEP]")
print("  → Likely cause: Decoder not learning properly during training")
print("\nIf the model predicts reasonable words:")
print("  → Model is working! Something else is wrong.")
