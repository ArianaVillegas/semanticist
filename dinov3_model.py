#!/usr/bin/env python3
"""
DINOv3 model implementation using alternative methods.
"""

import torch
import torch.nn as nn
import warnings

class DINOv3Model(nn.Module):
    """DINOv3 model using torch.hub or transformers."""
    
    def __init__(self, model_size="large", pretrained=True):
        super().__init__()
        self.model_size = model_size
        
        # Try different methods to load DINOv3/DINOv2
        self.model = self._load_model(model_size, pretrained)
        self._setup_attributes()
    
    def _load_model(self, model_size, pretrained):
        """Try different methods to load the model."""
        
        # Method 1: Try torch.hub DINOv2 (closest to DINOv3)
        try:
            model_map = {
                "small": "dinov2_vits14",
                "base": "dinov2_vitb14", 
                "large": "dinov2_vitl14",
                "giant": "dinov2_vitg14"
            }
            
            model_name = model_map.get(model_size, "dinov2_vitl14")
            model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=pretrained)
            print(f"✓ Loaded {model_name} via torch.hub")
            return model
            
        except Exception as e:
            print(f"torch.hub failed: {e}")
        
        # Method 2: Try transformers
        try:
            from transformers import AutoModel
            
            model_map = {
                "small": "facebook/dinov2-small",
                "base": "facebook/dinov2-base",
                "large": "facebook/dinov2-large",
                "giant": "facebook/dinov2-giant"
            }
            
            model_name = model_map.get(model_size, "facebook/dinov2-large")
            model = AutoModel.from_pretrained(model_name)
            print(f"✓ Loaded {model_name} via transformers")
            return model
            
        except Exception as e:
            print(f"transformers failed: {e}")
        
        # Method 3: Fallback to timm DINOv2
        try:
            import timm
            
            model_map = {
                "small": "vit_small_patch14_dinov2",
                "base": "vit_base_patch14_dinov2",
                "large": "vit_large_patch14_dinov2",
                "giant": "vit_giant_patch14_dinov2"
            }
            
            model_name = model_map.get(model_size, "vit_large_patch14_dinov2")
            model = timm.create_model(model_name, pretrained=pretrained)
            print(f"✓ Loaded {model_name} via timm (fallback)")
            return model
            
        except Exception as e:
            print(f"timm fallback failed: {e}")
            raise RuntimeError("Could not load any DINOv3/DINOv2 model")
    
    def _setup_attributes(self):
        """Setup attributes to match timm interface."""
        
        # Get embedding dimension
        if hasattr(self.model, 'embed_dim'):
            self.embed_dim = self.model.embed_dim
        elif hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
            self.embed_dim = self.model.config.hidden_size
        else:
            # Default based on model size
            embed_dims = {"small": 384, "base": 768, "large": 1024, "giant": 1536}
            self.embed_dim = embed_dims.get(self.model_size, 1024)
        
        # Number of prefix tokens (usually 1 for CLS)
        self.num_prefix_tokens = getattr(self.model, 'num_prefix_tokens', 1)
        
        # Patch embed info
        if hasattr(self.model, 'patch_embed'):
            self.patch_embed = self.model.patch_embed
        else:
            # Create dummy patch embed
            class DummyPatchEmbed:
                num_patches = 196  # 14x14 for 224x224
            self.patch_embed = DummyPatchEmbed()
        
        # Position embeddings
        if hasattr(self.model, 'pos_embed'):
            self.pos_embed = self.model.pos_embed
        else:
            self.pos_embed = None
    
    def forward_features(self, x):
        """Forward pass to get features."""
        
        # Handle different model interfaces
        if hasattr(self.model, 'forward_features'):
            # timm or torch.hub interface
            return self.model.forward_features(x)
        elif hasattr(self.model, 'forward'):
            # transformers interface
            outputs = self.model(x)
            if hasattr(outputs, 'last_hidden_state'):
                return outputs.last_hidden_state
            else:
                return outputs
        else:
            raise RuntimeError("Unknown model interface")
    
    def eval(self):
        """Set model to eval mode."""
        self.model.eval()
        return self

def create_dinov3_model(model_size="large", pretrained=True):
    """Factory function to create DINOv3 model."""
    return DINOv3Model(model_size, pretrained)

# Test function
def test_dinov3_model():
    """Test the DINOv3 model implementation."""
    
    print("=== Testing DINOv3 Model ===")
    
    try:
        # Create model
        model = create_dinov3_model("large", pretrained=True)
        model.eval()
        
        print(f"✓ Model created successfully")
        print(f"  Embed dim: {model.embed_dim}")
        print(f"  Num prefix tokens: {model.num_prefix_tokens}")
        print(f"  Num patches: {model.patch_embed.num_patches}")
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            features = model.forward_features(x)
            print(f"✓ Forward pass successful")
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {features.shape}")
        
        return model
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return None

if __name__ == "__main__":
    test_dinov3_model()
