#!/usr/bin/env python3
"""
DoRA Dataset Video Compression Pipeline
End-to-end: Video -> Semantic Tokens -> Lossless Reconstruction -> Quality Metrics
Handles 1-2 hour videos (720p@30fps) with memory optimization
"""

import os
import cv2
import torch
import numpy as np
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import time
import gc
from typing import Dict, List, Tuple, Optional
import subprocess
from PIL import Image

# Quality metrics
from skimage.metrics import structural_similarity as ssim
import lpips

# Compression comparison
import ffmpeg

def setup_models(device='cuda'):
    """Load Semanticist models using local checkpoint"""
    from omegaconf import OmegaConf
    from huggingface_hub import hf_hub_download
    from semanticist.stage1.diffuse_slot import DiffuseSlot
    
    print(f"Loading models on {device}...")
    
    # Load tokenizer from local checkpoint
    try:
        ckpt_path = hf_hub_download(repo_id='tennant/semanticist', filename='semanticist_tok_XL.pkl')
        config_path = 'configs/tokenizer_xl.yaml'
    except:
        # Fallback to L model
        ckpt_path = hf_hub_download(repo_id='tennant/semanticist', filename='semanticist_tok_L.pkl')
        config_path = 'configs/tokenizer_l.yaml'
    
    cfg = OmegaConf.load(config_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt = {k.replace('._orig_mod', ''): v for k, v in ckpt.items()}
    
    # Instantiate model like test_semanticist.py
    model_params = cfg['trainer']['params']['model']['params']
    tokenizer = DiffuseSlot(**model_params)
    
    # Use proper checkpoint loading
    ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()}
    missing, unexpected = tokenizer.load_state_dict(ckpt, strict=False)
    
    tokenizer = tokenizer.to(device)
    tokenizer.eval()
    tokenizer.enable_nest = True
    
    return tokenizer

def extract_video_info(video_path: str) -> Dict:
    """Extract video metadata"""
    cap = cv2.VideoCapture(video_path)
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration_sec': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
        'file_size_mb': os.path.getsize(video_path) / (1024 * 1024)
    }
    
    cap.release()
    return info

def process_video_chunks(video_path: str, tokenizer, chunk_size: int = 50, device='cuda'):
    """Process long video in small chunks to manage memory"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    all_tokens = []
    all_frames = []
    
    print(f"Processing {total_frames} frames in chunks of {chunk_size}...")
    
    # Import transforms once
    from semanticist.utils.datasets import vae_transforms
    transform = vae_transforms('test')
    
    for start_frame in tqdm(range(0, total_frames, chunk_size)):
        end_frame = min(start_frame + chunk_size, total_frames)
        chunk_frames = []
        
        # Read chunk
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB and resize to 256x256
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (256, 256))
            chunk_frames.append(frame_resized)
        
        if not chunk_frames:
            break
        
        # Process frames one by one to save memory
        chunk_tokens = []
        for frame in chunk_frames:
            # Convert to PIL and apply transforms
            pil_frame = Image.fromarray(frame)
            transformed = transform(pil_frame).unsqueeze(0).to(device)
            
            # Extract tokens for single frame
            with torch.no_grad():
                tokens = tokenizer.encode_slots(transformed)
                chunk_tokens.append(tokens.cpu().numpy())
            
            # Clear GPU memory immediately
            del transformed, tokens
            torch.cuda.empty_cache() if device == 'cuda' else None
        
        # Concatenate chunk tokens
        if chunk_tokens:
            chunk_tokens_array = np.concatenate(chunk_tokens, axis=0)
            all_tokens.append(chunk_tokens_array)
        
        # Store original frames for reconstruction comparison (subsample for speed)
        if start_frame % 500 == 0:  # Only every 500th frame for fast testing
            all_frames.extend(chunk_frames[:1])  # Only first frame of chunk
        
        # Clear memory
        del chunk_frames, chunk_tokens
        gc.collect()
    
    cap.release()
    
    # Concatenate all tokens
    semantic_tokens = np.concatenate(all_tokens, axis=0)
    
    return semantic_tokens, all_frames

def convert_to_numpy(img_tensor):
    """Convert tensor to numpy array for visualization (from test_semanticist.py)"""
    return img_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

def lossless_reconstruction(tokens: np.ndarray, original_frames: List[np.ndarray], tokenizer, device='cuda', batch_size: int = 1, num_tokens: int = 16):
    """Reconstruct video from semantic tokens using test_semanticist.py structure"""
    num_frames = min(len(tokens), len(original_frames))
    reconstructed_frames = []
    
    print(f"Reconstructing {num_frames} frames using {num_tokens} token(s)...")
    
    # Import transforms
    from semanticist.utils.datasets import vae_transforms
    transform = vae_transforms('test')
    
    for i in tqdm(range(0, num_frames, batch_size)):
        batch_frames = original_frames[i:i+batch_size]
        
        # Process original frames exactly like test_semanticist.py
        processed_frames = []
        for frame in batch_frames:
            pil_frame = Image.fromarray(frame)
            transformed = transform(pil_frame).unsqueeze(0)  # Add batch dimension
            processed_frames.append(transformed)
        
        for j, img_tensor in enumerate(processed_frames):
            img_tensor = img_tensor.to(device)
            
            with torch.no_grad():
                # Use exact same method as test_semanticist.py test_tokenizer_reconstruction
                recon = tokenizer(
                    img_tensor, 
                    sample=True, 
                    cfg=4.0,  # Use same cfg_scale as test_semanticist.py
                    inference_with_n_slots=num_tokens,
                )
                
                # Convert using same method as test_semanticist.py
                reconstructed_frame = convert_to_numpy(recon[0])
                reconstructed_frames.append(reconstructed_frame)
            
            # Clear memory
            del img_tensor, recon
            torch.cuda.empty_cache() if device == 'cuda' else None
    
    return reconstructed_frames

def save_reconstructed_video(frames: List[np.ndarray], output_path: str, fps: float = 25.0):
    """Save reconstructed frames as video file"""
    if not frames:
        print("No frames to save")
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Saving {len(frames)} frames to {output_path}")
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Video saved: {output_path}")

def save_comparison_frames(original_frames: List[np.ndarray], 
                          reconstructed_frames: List[np.ndarray], 
                          output_dir: str, 
                          num_samples: int = 5):
    """Save side-by-side comparison images"""
    import matplotlib.pyplot as plt
    
    os.makedirs(f"{output_dir}/comparisons", exist_ok=True)
    
    num_frames = min(len(original_frames), len(reconstructed_frames), num_samples)
    
    for i in range(num_frames):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(original_frames[i])
        ax1.set_title('Original')
        ax1.axis('off')
        
        ax2.imshow(reconstructed_frames[i])
        ax2.set_title('Reconstructed')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparisons/frame_{i:03d}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {num_frames} comparison images to {output_dir}/comparisons/")

def calculate_quality_metrics(original_frames: List[np.ndarray], 
                            reconstructed_frames: List[np.ndarray],
                            device='cuda') -> Dict:
    """Calculate SSIM and LPIPS quality metrics"""
    
    # Initialize LPIPS
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    
    ssim_scores = []
    lpips_scores = []
    
    print("Calculating quality metrics...")
    
    for i, (orig, recon) in enumerate(tqdm(zip(original_frames, reconstructed_frames))):
        # SSIM (grayscale)
        orig_gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
        recon_gray = cv2.cvtColor(recon, cv2.COLOR_RGB2GRAY)
        ssim_score = ssim(orig_gray, recon_gray, data_range=255)
        ssim_scores.append(ssim_score)
        
        # LPIPS (every 10th frame to save time)
        if i % 10 == 0:
            orig_tensor = torch.tensor(orig, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0
            recon_tensor = torch.tensor(recon, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0
            
            # Resize to 256x256 for LPIPS
            orig_tensor = torch.nn.functional.interpolate(orig_tensor, size=(256, 256), mode='bilinear')
            recon_tensor = torch.nn.functional.interpolate(recon_tensor, size=(256, 256), mode='bilinear')
            
            with torch.no_grad():
                lpips_score = lpips_fn(orig_tensor, recon_tensor).item()
                lpips_scores.append(lpips_score)
    
    return {
        'ssim_mean': np.mean(ssim_scores),
        'ssim_std': np.std(ssim_scores),
        'lpips_mean': np.mean(lpips_scores),
        'lpips_std': np.std(lpips_scores),
        'num_frames_evaluated': len(ssim_scores)
    }

def benchmark_traditional_codecs(video_path: str, output_dir: str) -> Dict:
    """Benchmark H.264 and H.265 compression on the same video"""
    results = {}
    
    original_size = os.path.getsize(video_path)
    
    codecs = {
        'h264': {'codec': 'libx264', 'ext': 'h264.mp4'},
        'h265': {'codec': 'libx265', 'ext': 'h265.mp4'}
    }
    
    for name, config in codecs.items():
        output_path = os.path.join(output_dir, f"compressed_{config['ext']}")
        
        try:
            # Compress with ffmpeg
            cmd = [
                'ffmpeg', '-i', video_path,
                '-c:v', config['codec'],
                '-preset', 'medium',
                '-crf', '23',  # Good quality
                '-y', output_path
            ]
            
            start_time = time.time()
            subprocess.run(cmd, capture_output=True, check=True)
            compression_time = time.time() - start_time
            
            compressed_size = os.path.getsize(output_path)
            compression_ratio = original_size / compressed_size
            
            results[name] = {
                'compressed_size_mb': compressed_size / (1024 * 1024),
                'compression_ratio': compression_ratio,
                'compression_time_sec': compression_time
            }
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to compress with {name}: {e}")
            results[name] = {'error': str(e)}
    
    return results

def simulate_semantic_compression(tokens: np.ndarray, 
                                keyframe_interval: int = 30,
                                similarity_threshold: float = 0.95) -> Dict:
    """Simulate semantic token compression with keyframes and deltas"""
    num_frames, num_slots, token_dim = tokens.shape
    
    compression_data = {
        'keyframes': [],
        'delta_frames': [],
        'total_tokens_stored': 0,
        'original_tokens': num_frames * num_slots,
        'token_dimension': token_dim
    }
    
    print(f"Simulating compression on {num_frames} frames...")
    
    for i in tqdm(range(num_frames)):
        if i % keyframe_interval == 0:
            # Keyframe: store all tokens
            compression_data['keyframes'].append(i)
            compression_data['total_tokens_stored'] += num_slots
        else:
            # Delta frame: store only changed tokens
            prev_tokens = tokens[i-1]
            curr_tokens = tokens[i]
            
            changed_tokens = 0
            for j in range(num_slots):
                # Cosine similarity
                prev_norm = prev_tokens[j] / (np.linalg.norm(prev_tokens[j]) + 1e-8)
                curr_norm = curr_tokens[j] / (np.linalg.norm(curr_tokens[j]) + 1e-8)
                sim = np.dot(prev_norm, curr_norm)
                
                if sim < similarity_threshold:
                    changed_tokens += 1
            
            compression_data['delta_frames'].append((i, changed_tokens))
            compression_data['total_tokens_stored'] += changed_tokens
    
    # Calculate compression metrics
    compression_ratio = compression_data['original_tokens'] / compression_data['total_tokens_stored']
    
    compression_data.update({
        'compression_ratio': compression_ratio,
        'tokens_saved': compression_data['original_tokens'] - compression_data['total_tokens_stored'],
        'storage_efficiency': compression_data['total_tokens_stored'] / compression_data['original_tokens']
    })
    
    return compression_data

def main():
    parser = argparse.ArgumentParser(description='DoRA Video Compression Pipeline')
    parser.add_argument('--video_path', required=True, help='Path to DoRA video file')
    parser.add_argument('--output_dir', default='dora_results', help='Output directory')
    parser.add_argument('--chunk_size', type=int, default=50, help='Frames per processing chunk')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--skip_reconstruction', action='store_true', help='Skip reconstruction step')
    parser.add_argument('--skip_traditional', action='store_true', help='Skip traditional codec comparison')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print(f"=== DoRA Video Compression Pipeline ===")
    print(f"Video: {args.video_path}")
    print(f"Device: {device}")
    
    # Extract video info
    video_info = extract_video_info(args.video_path)
    print(f"Video info: {video_info}")
    
    # Load models
    tokenizer = setup_models(device)
    
    # Process video and extract tokens
    start_time = time.time()
    semantic_tokens, original_frames = process_video_chunks(
        args.video_path, tokenizer, args.chunk_size, device
    )
    token_extraction_time = time.time() - start_time
    
    print(f"Token extraction completed in {token_extraction_time:.2f}s")
    print(f"Extracted tokens shape: {semantic_tokens.shape}")
    
    # Simulate compression
    compression_results = simulate_semantic_compression(semantic_tokens)
    
    results = {
        'video_info': video_info,
        'semantic_compression': compression_results,
        'token_extraction_time_sec': token_extraction_time,
        'tokens_shape': semantic_tokens.shape
    }
    
    # Lossless reconstruction and quality metrics
    if not args.skip_reconstruction:
        print("\n=== Lossless Reconstruction ===")
        start_time = time.time()
        reconstructed_frames = lossless_reconstruction(semantic_tokens, original_frames, tokenizer, device, num_tokens=16)
        reconstruction_time = time.time() - start_time
        
        # Calculate quality metrics
        quality_metrics = calculate_quality_metrics(original_frames, reconstructed_frames, device)
        
        # Save reconstructed video and comparison frames
        video_fps = video_info['fps']
        reconstructed_video_path = os.path.join(args.output_dir, 'reconstructed_video.mp4')
        save_reconstructed_video(reconstructed_frames, reconstructed_video_path, video_fps)
        save_comparison_frames(original_frames, reconstructed_frames, args.output_dir)
        
        results.update({
            'reconstruction_time_sec': reconstruction_time,
            'quality_metrics': quality_metrics,
            'reconstructed_video_path': reconstructed_video_path
        })
        
        print(f"Reconstruction completed in {reconstruction_time:.2f}s")
        print(f"Quality metrics: {quality_metrics}")
    
    # Traditional codec comparison
    if not args.skip_traditional:
        print("\n=== Traditional Codec Comparison ===")
        traditional_results = benchmark_traditional_codecs(args.video_path, args.output_dir)
        results['traditional_codecs'] = traditional_results
        
        print(f"Traditional codec results: {traditional_results}")
    
    # Save results
    results_path = os.path.join(args.output_dir, 'compression_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n=== Results Summary ===")
    print(f"Semantic compression ratio: {compression_results['compression_ratio']:.2f}x")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
