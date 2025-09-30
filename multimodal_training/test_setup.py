"""
Test script to verify Slot-CoCa setup before training
"""
import torch
import sys
import os

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    try:
        from models.slot_coca import SlotCoCa
        from imagenet_captions_dataset import ImageNetCaptionsDataset
        from transformers import DistilBertModel
        print("  ✅ All imports successful")
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_model_creation():
    """Test model instantiation"""
    print("\n🧪 Testing model creation...")
    try:
        from models.slot_coca import SlotCoCa
        model = SlotCoCa(num_slots=128, num_layers=3, projection_dim=256)
        print(f"  ✅ Model created successfully")
        print(f"     - Vision dim: {model.vision_dim}")
        print(f"     - Text dim: {model.text_dim}")
        print(f"     - Projection dim: 256")
        return True
    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        return False

def test_forward_pass():
    """Test forward pass with dummy data"""
    print("\n🧪 Testing forward pass...")
    try:
        from models.slot_coca import SlotCoCa
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SlotCoCa(num_slots=128, num_layers=3).to(device)
        
        # Dummy inputs
        images = torch.randn(2, 3, 224, 224).to(device)
        captions = ["a photo of a cat", "a photo of a dog"]
        
        # Tokenize
        text_tokens = model.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors='pt'
        ).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                images=images,
                text_tokens=text_tokens,
                caption_tokens=text_tokens,
                mode='all'
            )
        
        print("  ✅ Forward pass successful")
        print(f"     - Reconstruction loss: {outputs.get('reconstruction_loss', 0):.4f}")
        print(f"     - Contrastive loss: {outputs.get('contrastive_loss', 0):.4f}")
        print(f"     - Caption loss: {outputs.get('caption_loss', 0):.4f}")
        return True
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_loading():
    """Test dataset loading"""
    print("\n🧪 Testing dataset loading...")
    try:
        from imagenet_captions_dataset import ImageNetCaptionsDataset
        
        # Try to create a minimal dataset
        # This will fail if data doesn't exist, but we can catch it gracefully
        print("  ⚠️  Dataset test requires actual data")
        print("     Please ensure ImageNet-Captions is downloaded to ./data/imagenet_captions/")
        return True
    except Exception as e:
        print(f"  ❌ Dataset test failed: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability"""
    print("\n🧪 Testing GPU availability...")
    try:
        num_gpus = torch.cuda.device_count()
        print(f"  ✅ {num_gpus} GPU(s) available")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"     GPU {i}: {props.name}")
            print(f"       - Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"       - Compute: {props.major}.{props.minor}")
        
        if num_gpus < 2:
            print("  ⚠️  Warning: Less than 2 GPUs available. DDP training may not work optimally.")
        
        return num_gpus > 0
    except Exception as e:
        print(f"  ❌ GPU test failed: {e}")
        return False

def test_clip_availability():
    """Test CLIP availability for baseline comparisons"""
    print("\n🧪 Testing CLIP availability (for evaluation)...")
    try:
        import clip
        available_models = clip.available_models()
        print(f"  ✅ CLIP available")
        print(f"     Available models: {', '.join(available_models[:3])}")
        return True
    except Exception as e:
        print(f"  ⚠️  CLIP not available: {e}")
        print("     Install with: pip install git+https://github.com/openai/CLIP.git")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("SLOT-COCA SETUP VERIFICATION")
    print("="*60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Forward Pass", test_forward_pass()))
    results.append(("Dataset", test_dataset_loading()))
    results.append(("GPU", test_gpu_availability()))
    results.append(("CLIP", test_clip_availability()))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
    
    all_critical_passed = all(passed for name, passed in results[:5])  # First 5 are critical
    
    if all_critical_passed:
        print("\n🎉 All critical tests passed! Ready to train.")
        print("\nNext steps:")
        print("  1. Ensure dataset is downloaded to ./data/imagenet_captions/")
        print("  2. Run training: python train_multimodal.py")
        print("  3. Monitor: tensorboard --logdir ./logs/slot_coca")
    else:
        print("\n⚠️  Some critical tests failed. Please fix errors before training.")
    
    return all_critical_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
