#!/usr/bin/env python3
"""
Check available vision transformer models in timm.
"""

import timm

def check_available_models():
    """Check which ViT models are available."""
    print("Checking available vision transformer models...")
    print(f"timm version: {timm.__version__}")
    
    # Get all models
    all_models = timm.list_models()
    
    # Filter for vision transformers
    vit_models = [m for m in all_models if 'vit' in m.lower()]
    
    # Filter for specific types
    dino_models = [m for m in vit_models if 'dino' in m.lower()]
    dinov2_models = [m for m in vit_models if 'dinov2' in m.lower()]
    dinov3_models = [m for m in vit_models if 'dinov3' in m.lower()]
    
    print(f"\nTotal ViT models: {len(vit_models)}")
    print(f"DINO models: {len(dino_models)}")
    print(f"DINOv2 models: {len(dinov2_models)}")
    print(f"DINOv3 models: {len(dinov3_models)}")
    
    # Check for specific DINOv3 variants
    print("\n=== Searching for DINOv3 variants ===")
    dinov3_variants = [
        "vit_small_patch16_dinov3",
        "vit_base_patch16_dinov3", 
        "vit_large_patch16_dinov3",
        "vit_huge_patch16_dinov3",
        "vit_giant_patch16_dinov3",
        "dinov3_vit_small_patch16",
        "dinov3_vit_base_patch16",
        "dinov3_vit_large_patch16"
    ]
    
    for variant in dinov3_variants:
        if variant in all_models:
            print(f"  ✓ Found: {variant}")
        else:
            print(f"  ✗ Missing: {variant}")
    
    print("\n=== DINO Models ===")
    for model in sorted(dino_models):
        print(f"  {model}")
    
    print("\n=== DINOv2 Models ===")
    for model in sorted(dinov2_models):
        print(f"  {model}")
    
    print("\n=== DINOv3 Models ===")
    for model in sorted(dinov3_models):
        print(f"  {model}")
    
    # Test model creation
    print("\n=== Testing Model Creation ===")
    test_models = [
        "vit_base_patch14_dinov2",
        "vit_large_patch14_dinov2", 
        "vit_base_patch16_224.dino"
    ]
    
    for model_name in test_models:
        try:
            model = timm.create_model(model_name, pretrained=False)
            print(f"✓ {model_name} - OK")
        except Exception as e:
            print(f"✗ {model_name} - ERROR: {e}")

if __name__ == "__main__":
    check_available_models()
