#!/usr/bin/env python3
"""
Video Token Analysis for Semanticist
Extract and analyze semantic tokens from video frames to study temporal consistency.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import seaborn as sns

from semanticist.utils.datasets import vae_transforms
from test_semanticist import load_tokenizer_model, setup_device

def extract_video_frames(video_path, max_frames=100, skip_frames=1):
    """Extract frames from video."""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return []
    
    frames = []
    frame_count = 0
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % skip_frames == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {len(frames)} frames from {video_path}")
    return frames

def extract_tokens_from_frames(model, frames, device):
    """Extract semantic tokens from video frames."""
    transform = vae_transforms('test')
    all_tokens = []
    
    print("Extracting tokens from frames...")
    with torch.no_grad():
        for i, frame in enumerate(frames):
            if i % 10 == 0:
                print(f"Processing frame {i+1}/{len(frames)}")
            
            # Preprocess frame
            img_tensor = transform(frame).unsqueeze(0).to(device)
            
            # Extract semantic tokens using encode_slots method
            tokens = model.encode_slots(img_tensor)  # Shape: [1, num_slots, slot_dim]
            all_tokens.append(tokens.cpu().numpy()[0])  # Remove batch dim
    
    return np.array(all_tokens)  # Shape: [num_frames, num_slots, slot_dim]

def compute_token_similarities(tokens):
    """Compute frame-to-frame token similarities."""
    num_frames = tokens.shape[0]
    similarities = []
    
    for i in range(1, num_frames):
        # Flatten tokens for each frame
        prev_tokens = tokens[i-1].flatten()
        curr_tokens = tokens[i].flatten()
        
        # Compute cosine similarity
        sim = cosine_similarity([prev_tokens], [curr_tokens])[0][0]
        similarities.append(sim)
    
    return np.array(similarities)

def analyze_token_evolution(tokens, top_k=16):
    """Analyze how the most important tokens evolve over time."""
    # Use only the first k tokens (most important due to PCA ordering)
    important_tokens = tokens[:, :top_k, :]  # [frames, top_k, dim]
    
    # Flatten for analysis
    flattened = important_tokens.reshape(tokens.shape[0], -1)
    
    # Compute PCA to visualize token evolution
    pca = PCA(n_components=2)
    token_pca = pca.fit_transform(flattened)
    
    return token_pca, important_tokens

def plot_token_analysis(tokens, similarities, token_pca, save_dir="results"):
    """Create comprehensive visualization of token analysis."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Frame-to-frame similarity over time
    axes[0, 0].plot(similarities, linewidth=2)
    axes[0, 0].set_title('Frame-to-Frame Token Similarity')
    axes[0, 0].set_xlabel('Frame Number')
    axes[0, 0].set_ylabel('Cosine Similarity')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Token evolution in PCA space
    scatter = axes[0, 1].scatter(token_pca[:, 0], token_pca[:, 1], 
                                c=range(len(token_pca)), cmap='viridis', s=50)
    axes[0, 1].set_title('Token Evolution in Semantic Space')
    axes[0, 1].set_xlabel('PC1')
    axes[0, 1].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[0, 1], label='Frame Number')
    
    # 3. Heatmap of token similarities
    num_frames = min(50, tokens.shape[0])  # Limit for readability
    frame_similarities = np.zeros((num_frames, num_frames))
    
    for i in range(num_frames):
        for j in range(num_frames):
            tokens_i = tokens[i].flatten()
            tokens_j = tokens[j].flatten()
            frame_similarities[i, j] = cosine_similarity([tokens_i], [tokens_j])[0][0]
    
    sns.heatmap(frame_similarities, ax=axes[1, 0], cmap='viridis', 
                xticklabels=5, yticklabels=5)
    axes[1, 0].set_title('Frame Similarity Matrix')
    axes[1, 0].set_xlabel('Frame Number')
    axes[1, 0].set_ylabel('Frame Number')
    
    # 4. Token variance over time (which tokens change most)
    token_variance = np.var(tokens.reshape(tokens.shape[0], -1), axis=0)
    top_varying_indices = np.argsort(token_variance)[-20:]  # Top 20 most varying
    
    axes[1, 1].bar(range(len(top_varying_indices)), token_variance[top_varying_indices])
    axes[1, 1].set_title('Most Variable Token Dimensions')
    axes[1, 1].set_xlabel('Token Dimension (sorted by variance)')
    axes[1, 1].set_ylabel('Variance')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/video_token_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()

def detect_scene_changes(similarities, threshold=0.8):
    """Detect scene changes based on token similarity drops."""
    scene_changes = []
    
    for i, sim in enumerate(similarities):
        if sim < threshold:
            scene_changes.append(i + 1)  # +1 because similarities start from frame 1
    
    return scene_changes

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze video tokens with Semanticist')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--max_frames', type=int, default=100, help='Max frames to process')
    parser.add_argument('--skip_frames', type=int, default=1, help='Process every N frames')
    parser.add_argument('--top_k', type=int, default=16, help='Number of top tokens to analyze')
    parser.add_argument('--scene_threshold', type=float, default=0.8, help='Threshold for scene change detection')
    parser.add_argument('--cache_dir', default='./cache', help='Model cache directory')
    parser.add_argument('--output_dir', default='./results', help='Output directory')
    
    args = parser.parse_args()
    
    # Setup
    device = setup_device()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading tokenizer model...")
    model = load_tokenizer_model(device, args.cache_dir)
    
    # Extract video frames
    frames = extract_video_frames(args.video_path, args.max_frames, args.skip_frames)
    
    if len(frames) == 0:
        print("Error: No frames extracted from video. Please check the video file path.")
        print(f"Available video files in current directory:")
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files = list(Path('.').glob(ext))
            for f in video_files:
                print(f"  {f}")
        return
    
    # Extract tokens
    tokens = extract_tokens_from_frames(model, frames, device)
    print(f"Extracted tokens shape: {tokens.shape}")
    
    if tokens.shape[0] < 2:
        print("Error: Need at least 2 frames to compute similarities")
        return
    
    # Analyze similarities
    similarities = compute_token_similarities(tokens)
    print(f"Average frame-to-frame similarity: {similarities.mean():.3f}")
    
    # Analyze token evolution
    token_pca, important_tokens = analyze_token_evolution(tokens, args.top_k)
    
    # Detect scene changes
    scene_changes = detect_scene_changes(similarities, args.scene_threshold)
    print(f"Detected scene changes at frames: {scene_changes}")
    
    # Create visualizations
    plot_token_analysis(tokens, similarities, token_pca, args.output_dir)
    
    # Save results
    results = {
        'tokens': tokens,
        'similarities': similarities,
        'token_pca': token_pca,
        'scene_changes': scene_changes,
        'avg_similarity': similarities.mean(),
        'similarity_std': similarities.std()
    }
    
    np.savez(f"{args.output_dir}/video_analysis_results.npz", **results)
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Average similarity: {similarities.mean():.3f} ± {similarities.std():.3f}")
    print(f"Scene changes detected: {len(scene_changes)}")

if __name__ == '__main__':
    main()
