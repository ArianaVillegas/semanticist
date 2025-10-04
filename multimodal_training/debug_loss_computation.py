"""Debug why caption loss is zero"""
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

# Get one sample
sample = dataset[0]
image = sample['image'].unsqueeze(0)
caption = sample['caption']

print("="*60)
print("DEBUGGING CAPTION LOSS COMPUTATION")
print("="*60)
print(f"\nGround truth caption: {caption}\n")

# Tokenize
tokens = model.tokenizer(
    [caption],
    padding=True,
    truncation=True,
    max_length=77,
    return_tensors='pt'
)

print(f"Input IDs shape: {tokens['input_ids'].shape}")
print(f"Input IDs: {tokens['input_ids'][0][:20].tolist()}")
print(f"Decoded: {model.tokenizer.decode(tokens['input_ids'][0][:20])}")

# Forward pass
with torch.no_grad():
    outputs = model(image, caption_tokens=tokens, mode='caption')

print(f"\nOutputs keys: {outputs.keys()}")

if 'caption_loss' in outputs:
    loss = outputs['caption_loss']
    print(f"\nCaption loss: {loss.item()}")
    
    if 'caption_logits' in outputs:
        logits = outputs['caption_logits']
        print(f"Logits shape: {logits.shape}")
        
        # Check what model predicts
        predictions = logits.argmax(dim=-1)
        print(f"\nPredictions shape: {predictions.shape}")
        print(f"First 10 predictions: {predictions[0][:10].tolist()}")
        print(f"Decoded predictions: {model.tokenizer.decode(predictions[0][:20])}")
        
        # Check if predictions match targets
        targets = tokens['input_ids']
        matches = (predictions[:, :-1] == targets[:, 1:]).float().mean()
        print(f"\nAccuracy (predictions match targets): {matches.item()*100:.1f}%")
        
        if matches > 0.99:
            print("❌ FOUND THE BUG: Model has 100% accuracy!")
            print("   → This means it perfectly memorized the training data")
            print("   → Or loss computation is using wrong targets")
else:
    print("❌ No caption_loss in outputs!")
