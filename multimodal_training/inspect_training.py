"""Check training configuration and why caption loss is zero"""
import torch

ckpt = torch.load('checkpoints/coco_test_retraining/checkpoint_epoch_4.pt', map_location='cpu')

print("="*60)
print("TRAINING CONFIGURATION")
print("="*60)

if 'args' in ckpt:
    args = ckpt['args']
    print("\nLoss weights:")
    print(f"  lambda_recon:    {getattr(args, 'lambda_recon', 'N/A')}")
    print(f"  lambda_contrast: {getattr(args, 'lambda_contrast', 'N/A')}")
    print(f"  lambda_caption:  {getattr(args, 'lambda_caption', 'N/A')}")
    
    print(f"\nTraining:")
    print(f"  Epochs: {ckpt['epoch']}")
    print(f"  Learning rate: {getattr(args, 'lr', 'N/A')}")
    print(f"  Batch size: {getattr(args, 'batch_size', 'N/A')}")
else:
    print("No args found in checkpoint")

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

if 'args' in ckpt:
    lambda_caption = getattr(args, 'lambda_caption', None)
    
    if lambda_caption is None:
        print("⚠️  lambda_caption not found in checkpoint!")
    elif lambda_caption == 0:
        print("❌ FOUND THE BUG: lambda_caption = 0")
        print("   → Caption loss had ZERO weight during training")
        print("   → Decoder never learned because it didn't contribute to loss")
    elif lambda_caption < 0.1:
        print(f"⚠️  lambda_caption is very small: {lambda_caption}")
        print("   → Caption loss had minimal impact on training")
    else:
        print(f"✅ lambda_caption looks OK: {lambda_caption}")
        print("   → Issue is elsewhere (loss computation or architecture)")
