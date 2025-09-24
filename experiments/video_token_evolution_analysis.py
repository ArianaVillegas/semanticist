#!/usr/bin/env python3
"""
Video Token Evolution Analysis
Extracts frames from videos and analyzes token evolution using Semanticist tokenization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import os
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

from semanticist.engine.trainer_utils import instantiate_from_config
from semanticist.stage1.diffuse_slot import DiffuseSlot
from semanticist.utils.datasets import vae_transforms

def setup_device():
    """Setup device and print info."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    return device

def load_checkpoint(ckpt_path, model):
    """Load model checkpoint."""
    if ckpt_path.endswith(".pkl"):
        state_dict = torch.load(ckpt_path, map_location="cpu")
    else:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")
    
    # Remove '_orig_mod' prefix if present
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint from {ckpt_path}")

def load_tokenizer_model(device, cache_dir="./cache"):
    """Load the tokenizer model."""
    print("Loading tokenizer model...")
    
    # Download model
    ckpt_path = hf_hub_download(
        repo_id='tennant/semanticist', 
        filename='semanticist_tok_XL.pkl', 
        cache_dir=cache_dir
    )
    
    # Load config
    config_path = 'configs/tokenizer_xl.yaml'
    if not os.path.exists(config_path):
        config_path = 'configs/tokenizer_l.yaml'  # fallback
    
    cfg = OmegaConf.load(config_path)
    
    # Create model
    model = DiffuseSlot(**cfg['trainer']['params']['model']['params'])
    load_checkpoint(ckpt_path, model)
    model = model.to(device).eval()
    model.enable_nest = True
    
    return model

def extract_video_frames(video_path, output_dir, fps=1, max_frames=None, start_time=0):
    """Extract frames from video at specified fps."""
    print(f"Extracting frames from {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps
    
    print(f"Video info: {original_fps:.2f} FPS, {total_frames} frames, {duration:.2f}s duration")
    
    # Calculate frame extraction parameters
    frame_interval = int(original_fps / fps)  # Extract every N frames
    start_frame = int(start_time * original_fps)
    
    os.makedirs(output_dir, exist_ok=True)
    extracted_frames = []
    frame_count = 0
    current_frame = 0
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save frame
            frame_filename = f"frame_{frame_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            Image.fromarray(frame_rgb).save(frame_path, quality=95)
            
            extracted_frames.append(frame_path)
            frame_count += 1
            
            if max_frames and frame_count >= max_frames:
                break
        
        current_frame += 1
    
    cap.release()
    print(f"Extracted {len(extracted_frames)} frames to {output_dir}")
    return extracted_frames

def extract_tokens_from_image(model, image_path, device, num_tokens=32):
    """Extract tokens from a single image using Semanticist."""
    transform = vae_transforms('test')
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get the semantic slots using the model's encode_slots method
        slots = model.encode_slots(img_tensor)  # Shape: (batch_size, num_slots, slot_dim)
        
        # If we need to limit tokens, we can slice
        if num_tokens < slots.shape[1]:
            slots = slots[:, :num_tokens]
    
    return slots.cpu().numpy()

def extract_all_tokens(model, frame_paths, device, num_tokens=32):
    """Extract tokens from all frames."""
    print(f"Extracting tokens from {len(frame_paths)} frames using {num_tokens} tokens...")
    
    all_tokens = []
    valid_frames = []
    
    for frame_path in tqdm(frame_paths, desc="Processing frames"):
        try:
            tokens = extract_tokens_from_image(model, frame_path, device, num_tokens)
            all_tokens.append(tokens[0])  # Remove batch dimension
            valid_frames.append(frame_path)
        except Exception as e:
            print(f"Error processing {frame_path}: {e}")
            continue
    
    return np.array(all_tokens), valid_frames

def calculate_token_distances(tokens):
    """Calculate distances between frames for each token."""
    n_frames, n_tokens, token_dim = tokens.shape
    
    # Calculate cosine similarity and euclidean distance for each token
    token_cosine_similarities = []
    token_euclidean_distances = []
    
    for token_idx in range(n_tokens):
        token_vectors = tokens[:, token_idx, :]  # (n_frames, token_dim)
        
        # Cosine similarity
        cos_sim = cosine_similarity(token_vectors)
        token_cosine_similarities.append(cos_sim)
        
        # Euclidean distance
        euc_dist = euclidean_distances(token_vectors)
        token_euclidean_distances.append(euc_dist)
    
    return np.array(token_cosine_similarities), np.array(token_euclidean_distances)

def plot_token_evolution_heatmap(similarities, distances, output_dir, video_name):
    """Plot heatmaps showing token evolution."""
    n_tokens = similarities.shape[0]
    
    # Create subplots for similarities and distances
    fig, axes = plt.subplots(2, min(4, n_tokens), figsize=(16, 8))
    if n_tokens == 1:
        axes = axes.reshape(2, 1)
    
    # Plot first few tokens
    tokens_to_plot = min(4, n_tokens)
    
    for i in range(tokens_to_plot):
        # Similarity heatmap
        sns.heatmap(similarities[i], ax=axes[0, i], cmap='viridis', 
                   cbar=i==0, square=True)
        axes[0, i].set_title(f'Token {i+1} Cosine Similarity')
        axes[0, i].set_xlabel('Frame Index')
        axes[0, i].set_ylabel('Frame Index')
        
        # Distance heatmap
        sns.heatmap(distances[i], ax=axes[1, i], cmap='plasma', 
                   cbar=i==0, square=True)
        axes[1, i].set_title(f'Token {i+1} Euclidean Distance')
        axes[1, i].set_xlabel('Frame Index')
        axes[1, i].set_ylabel('Frame Index')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{video_name}_token_evolution_heatmaps.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()

def plot_token_similarity_trends(similarities, output_dir, video_name):
    """Plot how token similarities change over time."""
    n_tokens, n_frames, _ = similarities.shape
    
    # Calculate average similarity to adjacent frames
    adjacent_similarities = []
    for token_idx in range(n_tokens):
        sim_matrix = similarities[token_idx]
        # Get similarity to next frame (diagonal +1)
        adj_sim = [sim_matrix[i, i+1] for i in range(n_frames-1)]
        adjacent_similarities.append(adj_sim)
    
    # Plot trends
    plt.figure(figsize=(12, 8))
    
    # Plot first 8 tokens
    tokens_to_plot = min(8, n_tokens)
    for i in range(tokens_to_plot):
        plt.plot(adjacent_similarities[i], label=f'Token {i+1}', alpha=0.7)
    
    plt.xlabel('Frame Transition')
    plt.ylabel('Cosine Similarity to Next Frame')
    plt.title(f'Token Similarity Evolution - {video_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{video_name}_token_similarity_trends.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()

def plot_token_pca_analysis(tokens, output_dir, video_name):
    """Perform PCA analysis on tokens and visualize."""
    n_frames, n_tokens, token_dim = tokens.shape
    
    # Analyze first 16 tokens with PCA
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()
    
    tokens_to_analyze = min(16, n_tokens)
    
    for i in range(tokens_to_analyze):
        token_vectors = tokens[:, i, :]  # (n_frames, token_dim)
        
        # PCA
        pca = PCA(n_components=2)
        token_pca = pca.fit_transform(token_vectors)
        
        # Plot PCA
        scatter = axes[i].scatter(token_pca[:, 0], token_pca[:, 1], 
                                c=range(n_frames), cmap='viridis', alpha=0.7)
        axes[i].set_title(f'Token {i+1} PCA\n(Var: {pca.explained_variance_ratio_.sum():.3f})')
        axes[i].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
        axes[i].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
        
        # Add colorbar for frame progression
        plt.colorbar(scatter, ax=axes[i], label='Frame Index')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{video_name}_token_pca_analysis.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()

def plot_token_umap_analysis(tokens, output_dir, video_name):
    """Perform UMAP analysis on tokens and visualize."""
    n_frames, n_tokens, token_dim = tokens.shape
    
    # Analyze first 16 tokens with UMAP
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()
    
    tokens_to_analyze = min(16, n_tokens)
    
    for i in range(tokens_to_analyze):
        token_vectors = tokens[:, i, :]  # (n_frames, token_dim)
        
        # UMAP
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, n_frames-1))
        token_umap = reducer.fit_transform(token_vectors)
        
        # Plot UMAP
        scatter = axes[i].scatter(token_umap[:, 0], token_umap[:, 1], 
                                c=range(n_frames), cmap='viridis', alpha=0.7)
        axes[i].set_title(f'Token {i+1} UMAP')
        axes[i].set_xlabel('UMAP1')
        axes[i].set_ylabel('UMAP2')
        
        # Add colorbar for frame progression
        plt.colorbar(scatter, ax=axes[i], label='Frame Index')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{video_name}_token_umap_analysis.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()

def plot_token_statistics(tokens, similarities, distances, output_dir, video_name):
    """Plot various token statistics."""
    n_frames, n_tokens, token_dim = tokens.shape
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Token variance across frames
    token_variances = np.var(tokens, axis=0).mean(axis=1)  # Average variance per token
    axes[0, 0].bar(range(min(16, n_tokens)), token_variances[:min(16, n_tokens)])
    axes[0, 0].set_title('Token Variance Across Frames')
    axes[0, 0].set_xlabel('Token Index')
    axes[0, 0].set_ylabel('Average Variance')
    
    # 2. Average similarity per token
    avg_similarities = [np.mean(similarities[i][np.triu_indices_from(similarities[i], k=1)]) 
                       for i in range(min(16, n_tokens))]
    axes[0, 1].bar(range(len(avg_similarities)), avg_similarities)
    axes[0, 1].set_title('Average Inter-frame Similarity per Token')
    axes[0, 1].set_xlabel('Token Index')
    axes[0, 1].set_ylabel('Average Cosine Similarity')
    
    # 3. Token stability (similarity to adjacent frames)
    token_stability = []
    for i in range(min(16, n_tokens)):
        adj_sims = [similarities[i][j, j+1] for j in range(n_frames-1)]
        token_stability.append(np.mean(adj_sims))
    
    axes[1, 0].bar(range(len(token_stability)), token_stability)
    axes[1, 0].set_title('Token Stability (Adjacent Frame Similarity)')
    axes[1, 0].set_xlabel('Token Index')
    axes[1, 0].set_ylabel('Average Adjacent Similarity')
    
    # 4. Token distinctiveness (how different tokens are from each other)
    if n_tokens > 1:
        token_means = np.mean(tokens, axis=0)  # (n_tokens, token_dim)
        token_distinctiveness = euclidean_distances(token_means)
        im = axes[1, 1].imshow(token_distinctiveness[:min(16, n_tokens), :min(16, n_tokens)], 
                              cmap='viridis')
        axes[1, 1].set_title('Token Distinctiveness Matrix')
        axes[1, 1].set_xlabel('Token Index')
        axes[1, 1].set_ylabel('Token Index')
        plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{video_name}_token_statistics.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Video Token Evolution Analysis')
    parser.add_argument('--video', type=str, help='Specific video file to analyze')
    parser.add_argument('--video_dir', default='dora_videos', help='Directory containing videos')
    parser.add_argument('--output_dir', default='video_token_analysis', help='Output directory')
    parser.add_argument('--fps', type=float, default=1.0, help='Frame extraction rate (fps)')
    parser.add_argument('--max_frames', type=int, default=120, help='Maximum frames to extract (for 2 minutes at 1fps)')
    parser.add_argument('--num_tokens', type=int, default=32, help='Number of tokens to analyze')
    parser.add_argument('--start_time', type=float, default=0, help='Start time in seconds')
    parser.add_argument('--cache_dir', default='./cache', help='Cache directory for models')
    
    args = parser.parse_args()
    
    # Setup
    device = setup_device()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_tokenizer_model(device, args.cache_dir)
    
    # Get video files
    if args.video:
        video_files = [Path(args.video)]
    else:
        video_dir = Path(args.video_dir)
        video_files = list(video_dir.glob('*.mp4'))
    
    for video_path in video_files:
        print(f"\n{'='*60}")
        print(f"Processing: {video_path.name}")
        print(f"{'='*60}")
        
        video_name = video_path.stem.replace(' ', '_')
        video_output_dir = os.path.join(args.output_dir, video_name)
        frames_dir = os.path.join(video_output_dir, 'frames')
        
        # Extract frames
        frame_paths = extract_video_frames(
            video_path, frames_dir, 
            fps=args.fps, 
            max_frames=args.max_frames,
            start_time=args.start_time
        )
        
        if len(frame_paths) < 2:
            print(f"Not enough frames extracted from {video_path.name}, skipping...")
            continue
        
        # Extract tokens
        tokens, valid_frames = extract_all_tokens(model, frame_paths, device, args.num_tokens)
        
        if len(tokens) < 2:
            print(f"Not enough valid tokens extracted from {video_path.name}, skipping...")
            continue
        
        print(f"Extracted tokens shape: {tokens.shape}")
        
        # Calculate distances
        similarities, distances = calculate_token_distances(tokens)
        
        # Generate visualizations
        plot_token_evolution_heatmap(similarities, distances, video_output_dir, video_name)
        plot_token_similarity_trends(similarities, video_output_dir, video_name)
        plot_token_pca_analysis(tokens, video_output_dir, video_name)
        plot_token_umap_analysis(tokens, video_output_dir, video_name)
        plot_token_statistics(tokens, similarities, distances, video_output_dir, video_name)
        
        # Save data
        np.save(os.path.join(video_output_dir, f'{video_name}_tokens.npy'), tokens)
        np.save(os.path.join(video_output_dir, f'{video_name}_similarities.npy'), similarities)
        np.save(os.path.join(video_output_dir, f'{video_name}_distances.npy'), distances)
        
        print(f"Analysis complete for {video_name}!")
        print(f"Results saved to: {video_output_dir}")

if __name__ == '__main__':
    main()
