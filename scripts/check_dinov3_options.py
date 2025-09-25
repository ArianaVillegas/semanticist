#!/usr/bin/env python3
"""
Check DINOv3 availability and alternative installation methods.
"""

import timm
import torch

def check_dinov3_availability():
    """Check different ways to access DINOv3 models."""
    
    print("=== DINOv3 Availability Check ===")
    print(f"timm version: {timm.__version__}")
    print(f"torch version: {torch.__version__}")
    
    # Method 1: Check timm models
    print("\n1. Checking timm models...")
    all_models = timm.list_models()
    dinov3_models = [m for m in all_models if 'dinov3' in m.lower()]
    
    if dinov3_models:
        print("✓ DINOv3 models found in timm:")
        for model in sorted(dinov3_models):
            print(f"  - {model}")
    else:
        print("✗ No DINOv3 models found in timm")
    
    # Method 2: Try direct torch.hub access
    print("\n2. Checking torch.hub for DINOv3...")
    try:
        # DINOv3 is available through torch.hub from facebookresearch
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=False)
        print("✓ DINOv2 available via torch.hub")
        
        # Try DINOv3 (might be in a different repo)
        try:
            # This might not work, but let's try
            model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vits14', pretrained=False)
            print("✓ DINOv3 available via torch.hub")
        except:
            print("✗ DINOv3 not available via torch.hub")
            
    except Exception as e:
        print(f"✗ torch.hub access failed: {e}")
    
    # Method 3: Check if we can use DINOv2 as substitute
    print("\n3. Available DINOv2 models (DINOv3 alternative):")
    dinov2_models = [m for m in all_models if 'dinov2' in m.lower()]
    for model in sorted(dinov2_models):
        print(f"  - {model}")
    
    # Method 4: Test model creation
    print("\n4. Testing model creation...")
    test_models = [
        "vit_base_patch16_dinov3",  # Original attempt
        "vit_base_patch14_dinov2",  # Current fallback
        "vit_large_patch14_dinov2", # Larger alternative
    ]
    
    for model_name in test_models:
        try:
            model = timm.create_model(model_name, pretrained=False)
            print(f"✓ {model_name} - SUCCESS")
        except Exception as e:
            print(f"✗ {model_name} - FAILED: {e}")
    
    # Method 5: Check timm version requirements
    print(f"\n5. timm version analysis:")
    print(f"Current version: {timm.__version__}")
    print("DINOv3 might require timm >= 0.9.0")
    
    # Suggest upgrade if needed
    try:
        from packaging import version
        current_version = version.parse(timm.__version__)
        required_version = version.parse("0.9.0")
        
        if current_version < required_version:
            print("⚠️  Consider upgrading timm for DINOv3 support:")
            print("   pip install timm>=0.9.0")
        else:
            print("✓ timm version should support DINOv3")
    except:
        print("Cannot determine version compatibility")

def suggest_alternatives():
    """Suggest alternatives to DINOv3."""
    print("\n=== DINOv3 Alternatives ===")
    
    alternatives = [
        ("vit_large_patch14_dinov2", "Larger DINOv2 model - similar performance to DINOv3"),
        ("vit_base_patch14_dinov2", "Standard DINOv2 - proven performance"),
        ("vit_base_patch16_224.dino", "Original DINO - still very good"),
        ("vit_huge_patch14_dinov2", "Largest DINOv2 - best performance"),
    ]
    
    print("Recommended alternatives (in order of preference):")
    for i, (model, desc) in enumerate(alternatives, 1):
        print(f"{i}. {model}")
        print(f"   {desc}")
        print()

if __name__ == "__main__":
    check_dinov3_availability()
    suggest_alternatives()
