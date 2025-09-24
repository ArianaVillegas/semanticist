#!/usr/bin/env python3
"""
Video Compression Experiments using Semantic Tokens
Test the feasibility of video compression using Semanticist tokens.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from pathlib import Path
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import seaborn as sns
from collections import defaultdict

from semanticist.utils.datasets import vae_transforms
from test_semanticist import load_tokenizer_model, setup_device
from video_token_analysis import extract_video_frames, extract_tokens_from_frames

def analyze_compression_potential(tokens, similarity_threshold=0.95):
    """Analyze compression potential by finding redundant tokens."""
    num_frames, num_slots, slot_dim = tokens.shape
    
    # Compute frame-to-frame token differences
    token_changes = []
    for i in range(1, num_frames):
        # Compare each token between consecutive frames
        prev_tokens = tokens[i-1]  # [num_slots, slot_dim]
        curr_tokens = tokens[i]
        
        # Compute cosine similarity for each token
        token_similarities = []
        for j in range(num_slots):
            sim = cosine_similarity([prev_tokens[j]], [curr_tokens[j]])[0][0]
            token_similarities.append(sim)
        
        # Count how many tokens changed significantly
        changed_tokens = sum(1 for sim in token_similarities if sim < similarity_threshold)
        token_changes.append(changed_tokens)
    
    return np.array(token_changes), token_similarities

def simulate_compression(tokens, keyframe_interval=30, similarity_threshold=0.95):
    """Simulate semantic token-based video compression."""
    num_frames, num_slots, slot_dim = tokens.shape
    
    compression_data = {
        'keyframes': [],
        'delta_frames': [],
        'total_tokens_stored': 0,
        'compression_ratio': 0
    }
    
    for i in range(num_frames):
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
                sim = cosine_similarity([prev_tokens[j]], [curr_tokens[j]])[0][0]
                if sim < similarity_threshold:
                    changed_tokens += 1
            
            compression_data['delta_frames'].append((i, changed_tokens))
            compression_data['total_tokens_stored'] += changed_tokens
    
    # Calculate compression ratio
    original_tokens = num_frames * num_slots
    compression_data['compression_ratio'] = original_tokens / compression_data['total_tokens_stored']
    
    return compression_data

def analyze_scene_complexity(tokens, window_size=10):
    """Analyze scene complexity based on token variance."""
    num_frames, num_slots, slot_dim = tokens.shape
    complexities = []
    
    for i in range(window_size, num_frames - window_size):
        window_tokens = tokens[i-window_size:i+window_size]
        # Compute variance across time for each token dimension
        token_variance = np.var(window_tokens.reshape(-1, num_slots * slot_dim), axis=0)
        complexity = np.mean(token_variance)
        complexities.append(complexity)
    
    return np.array(complexities)

def adaptive_compression(tokens, base_tokens=8, max_tokens=64):
    """Simulate adaptive compression based on scene complexity."""
    complexities = analyze_scene_complexity(tokens)
    
    # Normalize complexities to [0, 1]
    if len(complexities) > 0:
        norm_complexities = (complexities - complexities.min()) / (complexities.max() - complexities.min())
    else:
        norm_complexities = np.array([0.5])
    
    # Adaptive token allocation
    adaptive_tokens = []
    for complexity in norm_complexities:
        tokens_needed = int(base_tokens + complexity * (max_tokens - base_tokens))
        adaptive_tokens.append(tokens_needed)
    
    return adaptive_tokens, norm_complexities

def test_video_dataset_compression(dataset_path, model, device, max_videos=10):
    """Test compression on a video dataset."""
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        video_files.extend(Path(dataset_path).glob(f"**/{ext}"))
    
    video_files = video_files[:max_videos]
    results = []
    
    for video_path in video_files:
        print(f"Processing: {video_path.name}")
        
        # Extract frames and tokens
        frames = extract_video_frames(str(video_path), max_frames=100, skip_frames=2)
        if len(frames) < 10:
            continue
            
        tokens = extract_tokens_from_frames(model, frames, device)
        
        # Analyze compression potential
        token_changes, _ = analyze_compression_potential(tokens)
        compression_data = simulate_compression(tokens)
        adaptive_tokens, complexities = adaptive_compression(tokens)
        
        result = {
            'video': video_path.name,
            'num_frames': len(frames),
            'avg_token_changes': token_changes.mean(),
            'compression_ratio': compression_data['compression_ratio'],
            'avg_complexity': complexities.mean() if len(complexities) > 0 else 0,
            'adaptive_compression_ratio': len(frames) * 256 / sum(adaptive_tokens) if adaptive_tokens else 1
        }
        results.append(result)
    
    return results

def plot_compression_analysis(compression_results, output_dir="results"):
    """Plot comprehensive compression analysis."""
    if not compression_results:
        print("No compression results to plot")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Compression ratios distribution
    ratios = [r['compression_ratio'] for r in compression_results]
    axes[0, 0].hist(ratios, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Compression Ratios Distribution')
    axes[0, 0].set_xlabel('Compression Ratio')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(ratios), color='red', linestyle='--', label=f'Mean: {np.mean(ratios):.1f}x')
    axes[0, 0].legend()
    
    # 2. Token changes vs compression ratio
    changes = [r['avg_token_changes'] for r in compression_results]
    axes[0, 1].scatter(changes, ratios, alpha=0.7)
    axes[0, 1].set_title('Token Changes vs Compression Ratio')
    axes[0, 1].set_xlabel('Avg Token Changes per Frame')
    axes[0, 1].set_ylabel('Compression Ratio')
    
    # 3. Scene complexity vs compression
    complexities = [r['avg_complexity'] for r in compression_results]
    adaptive_ratios = [r['adaptive_compression_ratio'] for r in compression_results]
    axes[1, 0].scatter(complexities, adaptive_ratios, alpha=0.7, color='green')
    axes[1, 0].set_title('Scene Complexity vs Adaptive Compression')
    axes[1, 0].set_xlabel('Average Scene Complexity')
    axes[1, 0].set_ylabel('Adaptive Compression Ratio')
    
    # 4. Comparison of compression methods
    methods = ['Fixed Tokens', 'Adaptive Tokens']
    fixed_ratios = [r['compression_ratio'] for r in compression_results]
    adaptive_ratios = [r['adaptive_compression_ratio'] for r in compression_results]
    
    axes[1, 1].boxplot([fixed_ratios, adaptive_ratios], tick_labels=methods)
    axes[1, 1].set_title('Compression Method Comparison')
    axes[1, 1].set_ylabel('Compression Ratio')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/compression_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()

def estimate_bitrates(compression_results, fps=30):
    """Estimate bitrates for semantic token compression."""
    for result in compression_results:
        # Original video bitrate (assuming 1080p)
        original_bitrate = 1920 * 1080 * 3 * 8 * fps / 1e6  # Mbps
        
        # Semantic token bitrate
        avg_tokens_per_frame = 256 / result['compression_ratio']
        token_bitrate = avg_tokens_per_frame * 16 * 32 * fps / 1e6  # Mbps (16 dim, 32-bit float)
        
        result['original_bitrate_mbps'] = original_bitrate
        result['semantic_bitrate_mbps'] = token_bitrate
        result['bitrate_reduction'] = original_bitrate / token_bitrate

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Video compression experiments with semantic tokens')
    parser.add_argument('--dataset_path', default='.', help='Path to video dataset')
    parser.add_argument('--max_videos', type=int, default=5, help='Max videos to process')
    parser.add_argument('--similarity_threshold', type=float, default=0.95, help='Token similarity threshold')
    parser.add_argument('--keyframe_interval', type=int, default=30, help='Keyframe interval')
    parser.add_argument('--cache_dir', default='./cache', help='Model cache directory')
    parser.add_argument('--output_dir', default='./results', help='Output directory')
    
    args = parser.parse_args()
    
    # Setup
    device = setup_device()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading tokenizer model...")
    model = load_tokenizer_model(device, args.cache_dir)
    
    # Test compression on dataset
    print(f"Testing compression on videos in: {args.dataset_path}")
    compression_results = test_video_dataset_compression(
        args.dataset_path, model, device, args.max_videos
    )
    
    if not compression_results:
        print("No videos processed successfully")
        return
    
    # Estimate bitrates
    estimate_bitrates(compression_results)
    
    # Print results
    print(f"\n=== Compression Analysis Results ===")
    print(f"Videos processed: {len(compression_results)}")
    
    avg_compression = np.mean([r['compression_ratio'] for r in compression_results])
    avg_bitrate_reduction = np.mean([r['bitrate_reduction'] for r in compression_results])
    
    print(f"Average compression ratio: {avg_compression:.1f}x")
    print(f"Average bitrate reduction: {avg_bitrate_reduction:.1f}x")
    
    # Detailed results
    for result in compression_results:
        print(f"\n{result['video']}:")
        print(f"  Frames: {result['num_frames']}")
        print(f"  Compression ratio: {result['compression_ratio']:.1f}x")
        print(f"  Avg token changes: {result['avg_token_changes']:.1f}")
        print(f"  Bitrate: {result['original_bitrate_mbps']:.1f} → {result['semantic_bitrate_mbps']:.3f} Mbps")
    
    # Create visualizations
    plot_compression_analysis(compression_results, args.output_dir)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Clean results for JSON serialization
    json_results = []
    for result in compression_results:
        clean_result = {}
        for key, value in result.items():
            clean_result[key] = convert_numpy_types(value)
        json_results.append(clean_result)
    
    # Save results
    with open(f"{args.output_dir}/compression_results.json", 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
