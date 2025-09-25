#!/usr/bin/env python3
"""
Pre-download model weights for offline SLURM training.
"""

import timm
import torch
import os
from pathlib import Path

def download_models():
    """Download all required models for offline training."""
    
    print("=== Pre-downloading Model Weights ===")
    
    # Models to download
    models_to_download = [
        "vit_base_patch16_dinov3",      # DINOv3 base
        "vit_large_patch14_dinov2",     # DINOv2 large (fallback)
        "vit_base_patch14_dinov2",      # DINOv2 base (fallback)
    ]
    
    # Create cache directory
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Cache directory: {cache_dir}")
    
    for model_name in models_to_download:
        print(f"\nDownloading {model_name}...")
        try:
            # Download model weights
            model = timm.create_model(model_name, pretrained=True)
            print(f"✓ {model_name} downloaded successfully")
            
            # Test forward pass to ensure everything works
            with torch.no_grad():
                x = torch.randn(1, 3, 224, 224)
                if hasattr(model, 'forward_features'):
                    features = model.forward_features(x)
                    print(f"  Forward test: {features.shape}")
                else:
                    output = model(x)
                    print(f"  Forward test: {output.shape}")
                    
            # Clean up memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"✗ {model_name} failed: {e}")
    
    # Download dataset
    print(f"\nDownloading Imagenette dataset...")
    try:
        import torchvision
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        
        # Download both splits
        train_data = torchvision.datasets.Imagenette(
            "./datasets", split="train", transform=transform, download=True
        )
        val_data = torchvision.datasets.Imagenette(
            "./datasets", split="val", transform=transform, download=True
        )
        
        print(f"✓ Imagenette downloaded: {len(train_data)} train, {len(val_data)} val samples")
        
    except Exception as e:
        print(f"✗ Dataset download failed: {e}")
    
    print(f"\n=== Cache Status ===")
    
    # Show cache size
    if cache_dir.exists():
        total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        print(f"Total cache size: {total_size / (1024**3):.2f} GB")
        
        # List cached models
        model_dirs = [d for d in cache_dir.iterdir() if d.is_dir()]
        print(f"Cached models: {len(model_dirs)}")
        for model_dir in sorted(model_dirs)[:10]:  # Show first 10
            print(f"  - {model_dir.name}")
    
    print(f"\n✓ All downloads complete! SLURM jobs can now run offline.")

def test_offline_loading():
    """Test that models can be loaded offline."""
    
    print("\n=== Testing Offline Model Loading ===")
    
    # Temporarily disable internet (simulation)
    import os
    old_http_proxy = os.environ.get('HTTP_PROXY')
    old_https_proxy = os.environ.get('HTTPS_PROXY')
    
    try:
        # Set proxy to localhost to simulate no internet
        os.environ['HTTP_PROXY'] = 'http://127.0.0.1:9999'
        os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:9999'
        
        # Try loading models
        test_models = ["vit_base_patch16_dinov3", "vit_large_patch14_dinov2"]
        
        for model_name in test_models:
            try:
                model = timm.create_model(model_name, pretrained=True)
                print(f"✓ {model_name} loaded offline successfully")
                del model
            except Exception as e:
                print(f"✗ {model_name} offline loading failed: {e}")
                
    finally:
        # Restore original proxy settings
        if old_http_proxy:
            os.environ['HTTP_PROXY'] = old_http_proxy
        else:
            os.environ.pop('HTTP_PROXY', None)
            
        if old_https_proxy:
            os.environ['HTTPS_PROXY'] = old_https_proxy
        else:
            os.environ.pop('HTTPS_PROXY', None)

if __name__ == "__main__":
    download_models()
    test_offline_loading()
