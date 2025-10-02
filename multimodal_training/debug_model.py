"""
Debug script to check model components
"""
import torch
from models.slot_coca import SlotCoCa

print("="*60)
print("MODEL DEBUG CHECK")
print("="*60)

# Load checkpoint
checkpoint_path = "/home/avillegas/semanticist/multimodal_training/checkpoints/slot_coca_real_captions/best_model.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\n1. Checkpoint Contents:")
print(f"   Keys: {list(checkpoint.keys())}")

if 'args' in checkpoint:
    args = checkpoint['args']
    print("\n2. Training Arguments:")
    print(f"   lambda_recon: {args.get('lambda_recon', 'NOT FOUND')}")
    print(f"   lambda_contrast: {args.get('lambda_contrast', 'NOT FOUND')}")
    print(f"   lambda_caption: {args.get('lambda_caption', 'NOT FOUND')}")
    print(f"   num_slots: {args.get('num_slots', 'NOT FOUND')}")
    print(f"   num_epochs: {args.get('epochs', 'NOT FOUND')}")

# Load model
print("\n3. Loading Model...")
model = SlotCoCa(
    num_slots=checkpoint.get('args', {}).get('num_slots', 128),
    num_layers=checkpoint.get('args', {}).get('num_layers', 3),
    encoder_name=checkpoint.get('args', {}).get('encoder_name', 'vit_base_patch16_dinov3'),
    projection_dim=checkpoint.get('args', {}).get('projection_dim', 256)
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("   ✅ Model loaded successfully")

# Test caption generation
print("\n4. Testing Caption Generation...")
print("   Creating dummy image batch...")
dummy_image = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    # Generate caption
    try:
        generated_ids = model.generate_caption(
            dummy_image,
            max_length=20,
            temperature=1.0
        )
        
        # Decode
        generated_text = model.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        print(f"   Generated caption: '{generated_text}'")
        print(f"   Generated IDs: {generated_ids[0].tolist()}")
        
        if not generated_text or len(generated_text.strip()) == 0:
            print("\n   ❌ PROBLEM: Caption is empty!")
            print("   This means the caption decoder is not generating text.")
            print("\n   Possible causes:")
            print("   1. lambda_caption was 0 or too low during training")
            print("   2. Caption loss didn't decrease")
            print("   3. Decoder weights weren't updated")
        else:
            print("\n   ✅ Caption generation works!")
            
    except Exception as e:
        print(f"   ❌ Error during generation: {e}")

# Check tokenizer
print("\n5. Tokenizer Check:")
test_text = "a photo of a dog"
tokens = model.tokenizer(test_text, return_tensors='pt')
decoded = model.tokenizer.decode(tokens['input_ids'][0])
print(f"   Test encode/decode: '{test_text}' → '{decoded}'")

print("\n" + "="*60)
print("DEBUG COMPLETE")
print("="*60)
