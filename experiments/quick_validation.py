#!/usr/bin/env python3
"""Quick validation of tokenizer reconstruction quality"""

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from semanticist.stage1.diffuse_slot import DiffuseSlot
from semanticist.utils.datasets import vae_transforms
from skimage.metrics import structural_similarity as ssim
import os

def convert_to_numpy(img_tensor):
    """Convert tensor to numpy array for visualization"""
    return img_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

def load_model():
    """Load tokenizer model with minimal RAM usage"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use XL model (only one available)
    config_path = 'configs/tokenizer_xl.yaml'
    ckpt_path = 'cache/models--tennant--semanticist/snapshots/dbc5ff601442eb23ba82d33c1b104cd894743633/semanticist_tok_XL.pkl'
    cfg = OmegaConf.load(config_path)
    
    # Load with CPU to reduce GPU memory
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()}
    
    # Create model
    model_params = cfg['trainer']['params']['model']['params']
    tokenizer = DiffuseSlot(**model_params)
    tokenizer.load_state_dict(ckpt, strict=False)
    
    # Clear checkpoint from memory immediately
    del ckpt
    torch.cuda.empty_cache() if device == 'cuda' else None
    
    tokenizer = tokenizer.to(device).eval()
    tokenizer.enable_nest = True
    
    return tokenizer, device

def test_reconstruction(image_path, token_counts=[1]):
    """Test reconstruction with minimal memory usage"""
    tokenizer, device = load_model()
    transform = vae_transforms('test')
    
    # Load image at model's expected size (256x256)
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    print(f"Testing reconstruction on {image_path}")
    print(f"Image tensor shape: {img_tensor.shape}")
    
    results = {}
    # Get original at same size as model output (256x256)
    original_256 = image.resize((256, 256))
    original_np = np.array(original_256)
    
    with torch.no_grad():
        for num_tokens in token_counts:
            print(f"\nTesting with {num_tokens} tokens...")
            
            # Reconstruct
            recon = tokenizer(
                img_tensor, 
                sample=True, 
                cfg=1.0,  # Lower CFG to reduce memory
                inference_with_n_slots=num_tokens,
            )
            
            # Convert to numpy
            recon_np = convert_to_numpy(recon[0])
            
            # Calculate SSIM
            if original_np.shape == recon_np.shape:
                ssim_score = ssim(original_np, recon_np, multichannel=True, channel_axis=2)
            else:
                print(f"Shape mismatch: orig {original_np.shape} vs recon {recon_np.shape}")
                ssim_score = 0.0
            
            results[num_tokens] = {
                'ssim': ssim_score,
                'image': recon_np
            }
            
            print(f"SSIM with {num_tokens} tokens: {ssim_score:.4f}")
            
            # Clear memory immediately
            del recon
            torch.cuda.empty_cache() if device == 'cuda' else None
    
    return results

if __name__ == "__main__":
    # Test on city image with just 1 token
    results = test_reconstruction('examples/city.jpg', [1])
    
    print("\n=== SUMMARY ===")
    for tokens, result in results.items():
        print(f"{tokens} tokens: SSIM = {result['ssim']:.4f}")
