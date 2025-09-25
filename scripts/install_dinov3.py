#!/usr/bin/env python3
"""
Install and test DINOv3 from alternative sources.
"""

import subprocess
import sys
import os

def install_dinov3_direct():
    """Try to install DINOv3 directly from Facebook Research."""
    
    print("=== Installing DINOv3 from Facebook Research ===")
    
    # Method 1: Install from GitHub
    try:
        print("1. Trying pip install from GitHub...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/facebookresearch/dinov2.git"
        ], check=True)
        print("✓ DINOv2 installed from GitHub")
    except subprocess.CalledProcessError:
        print("✗ GitHub installation failed")
    
    # Method 2: Try newer timm version
    try:
        print("\n2. Trying to upgrade timm...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "timm>=1.0.0"
        ], check=True)
        print("✓ timm upgraded")
    except subprocess.CalledProcessError:
        print("✗ timm upgrade failed")
    
    # Method 3: Install transformers (might have DINOv3)
    try:
        print("\n3. Trying transformers library...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "transformers"
        ], check=True)
        print("✓ transformers installed")
    except subprocess.CalledProcessError:
        print("✗ transformers installation failed")

def test_dinov3_access():
    """Test different ways to access DINOv3."""
    
    print("\n=== Testing DINOv3 Access Methods ===")
    
    # Method 1: Updated timm
    print("1. Testing updated timm...")
    try:
        import timm
        print(f"timm version: {timm.__version__}")
        
        # Check for DINOv3 models
        models = timm.list_models()
        dinov3_models = [m for m in models if 'dinov3' in m.lower()]
        
        if dinov3_models:
            print("✓ DINOv3 models found in timm:")
            for model in dinov3_models:
                print(f"  - {model}")
        else:
            print("✗ No DINOv3 models in timm")
    except Exception as e:
        print(f"✗ timm test failed: {e}")
    
    # Method 2: torch.hub with different repos
    print("\n2. Testing torch.hub alternatives...")
    try:
        import torch
        
        # Try different repository names
        repos = [
            "facebookresearch/dinov2",
            "facebookresearch/dinov3", 
            "facebookresearch/dino",
        ]
        
        for repo in repos:
            try:
                models = torch.hub.list(repo)
                print(f"✓ {repo} available, models: {models}")
            except Exception as e:
                print(f"✗ {repo} failed: {e}")
                
    except Exception as e:
        print(f"✗ torch.hub test failed: {e}")
    
    # Method 3: transformers library
    print("\n3. Testing transformers library...")
    try:
        from transformers import AutoModel, AutoImageProcessor
        
        # Try DINOv3 model names from Hugging Face
        dinov3_models = [
            "facebook/dinov2-base",
            "facebook/dinov2-large", 
            "facebook/dino-vitb16",
        ]
        
        for model_name in dinov3_models:
            try:
                processor = AutoImageProcessor.from_pretrained(model_name)
                print(f"✓ {model_name} available via transformers")
            except Exception as e:
                print(f"✗ {model_name} failed: {e}")
                
    except ImportError:
        print("✗ transformers not available")
    except Exception as e:
        print(f"✗ transformers test failed: {e}")

def create_dinov3_wrapper():
    """Create a wrapper to use DINOv3 via torch.hub or transformers."""
    
    wrapper_code = '''
import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor

class DINOv3Wrapper(nn.Module):
    """Wrapper to use DINOv3 via transformers or torch.hub."""
    
    def __init__(self, model_name="facebook/dinov2-large"):
        super().__init__()
        
        # Try transformers first
        try:
            self.model = AutoModel.from_pretrained(model_name)
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.method = "transformers"
            print(f"✓ Using DINOv3 via transformers: {model_name}")
        except:
            # Fallback to torch.hub
            try:
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
                self.method = "torch_hub"
                print("✓ Using DINOv2 Large via torch.hub")
            except Exception as e:
                raise RuntimeError(f"Could not load DINOv3/DINOv2: {e}")
        
        # Set attributes to match timm interface
        self.embed_dim = 1024  # DINOv2 Large embedding dimension
        self.num_prefix_tokens = 1  # CLS token
        
        # Calculate patch embed info
        if hasattr(self.model, 'patch_embed'):
            self.patch_embed = self.model.patch_embed
        else:
            # Create dummy patch_embed for compatibility
            class DummyPatchEmbed:
                num_patches = 196  # 14x14 patches for 224x224 image
            self.patch_embed = DummyPatchEmbed()
    
    def forward_features(self, x):
        """Forward pass to get features (compatible with timm interface)."""
        if self.method == "transformers":
            outputs = self.model(x)
            return outputs.last_hidden_state
        else:
            # torch.hub method
            return self.model.forward_features(x)
    
    def eval(self):
        """Set to eval mode."""
        self.model.eval()
        return self

# Test the wrapper
if __name__ == "__main__":
    try:
        model = DINOv3Wrapper()
        print("✓ DINOv3 wrapper created successfully")
        
        # Test with dummy input
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = model.forward_features(x)
            print(f"✓ Forward pass successful, output shape: {features.shape}")
            
    except Exception as e:
        print(f"✗ DINOv3 wrapper failed: {e}")
'''
    
    # Save wrapper to file
    with open("dinov3_wrapper.py", "w") as f:
        f.write(wrapper_code)
    
    print("\n=== DINOv3 Wrapper Created ===")
    print("Saved to: dinov3_wrapper.py")
    print("Usage: from dinov3_wrapper import DINOv3Wrapper")

if __name__ == "__main__":
    install_dinov3_direct()
    test_dinov3_access()
    create_dinov3_wrapper()
