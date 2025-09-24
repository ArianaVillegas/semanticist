#!/usr/bin/env python3
"""
Full Dora Videos Experimental Suite
Execute complete temporal dynamics analysis on all dora_videos for preliminary results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pandas as pd
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
import argparse
from tqdm import tqdm

from semanticist.stage1.diffuse_slot import DiffuseSlot
from semanticist.utils.datasets import vae_transforms
from temporal_dynamics_framework import TemporalDynamicsAnalyzer, ExperimentalFramework
from video_token_evolution_analysis import load_tokenizer_model, extract_video_frames, extract_all_tokens
import gc
import psutil
import os

def run_full_dora_experiment(batch_size=1, max_videos=None, save_intermediate=True):
    """Execute complete experimental suite on all dora videos.
    
    Args:
        batch_size: Number of videos to process in each batch (default: 1 for memory safety)
        max_videos: Maximum number of videos to process (default: None for all)
        save_intermediate: Save results after each batch to prevent data loss (default: True)
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("Loading Semanticist model...")
    model = load_tokenizer_model(device)
    
    # Initialize experimental framework
    framework = ExperimentalFramework(model, output_dir='full_dora_experiments')
    
    # Get all dora videos
    dora_dir = Path('dora_videos')
    video_files = list(dora_dir.glob('*.mp4'))
    
    if max_videos:
        video_files = video_files[:max_videos]
    
    print(f"Found {len(video_files)} dora videos:")
    for video in video_files:
        print(f"  - {video.name}")
    
    # Experimental parameters (reduced for memory efficiency)
    params = {
        'fps': 1.0,                    # 1 frame per second
        'max_frames': 60,              # 1 minute of content (reduced from 120)
        'num_tokens': 16,              # Reduced from 32 for memory
        'start_time': 30,              # Skip first 30 seconds (intro/stabilization)
    }
    
    # Memory monitoring
    def print_memory_usage():
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"Memory: RAM {memory_mb:.1f}MB, GPU {gpu_memory_mb:.1f}MB")
        else:
            print(f"Memory: RAM {memory_mb:.1f}MB")
    
    print(f"\nExperimental parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Process videos in batches
    all_video_data = {}
    failed_videos = []
    
    # Process in batches to manage memory
    for batch_start in range(0, len(video_files), batch_size):
        batch_end = min(batch_start + batch_size, len(video_files))
        batch_videos = video_files[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1}/{(len(video_files)-1)//batch_size + 1}")
        
        for video_path in tqdm(batch_videos, desc=f"Batch {batch_start//batch_size + 1}"):
            video_name = video_path.stem.replace(' ', '_')
            
            try:
                print(f"\n{'='*60}")
                print(f"Processing: {video_path.name}")
                print(f"{'='*60}")
                
                # Create output directory
                video_output_dir = Path('full_dora_experiments') / video_name
                frames_dir = video_output_dir / 'frames'
                
                # Extract frames
                print("Extracting frames...")
                frame_paths = extract_video_frames(
                    video_path, frames_dir,
                    fps=params['fps'],
                    max_frames=params['max_frames'],
                    start_time=params['start_time']
                )
                
                if len(frame_paths) < 10:
                    print(f"  WARNING: Only {len(frame_paths)} frames extracted, skipping...")
                    failed_videos.append(video_name)
                    continue
                
                # Extract tokens
                print("Extracting semantic tokens...")
                tokens, valid_frames = extract_all_tokens(model, frame_paths, device, params['num_tokens'])
                
                if len(tokens) < 10:
                    print(f"  WARNING: Only {len(tokens)} valid tokens, skipping...")
                    failed_videos.append(video_name)
                    continue
                
                print(f"  Tokens shape: {tokens.shape}")
                
                # Store data (without loading frames into memory)
                all_video_data[video_name] = {
                    'tokens': tokens,
                    'frame_paths': valid_frames[:30],  # Store paths, not frames
                    'video_path': str(video_path),
                    'category': 'urban_walking'  # All dora videos are urban walking tours
                }
                
                # Aggressive memory cleanup after each video
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Force garbage collection
                del tokens  # Explicitly delete large tensor
                gc.collect()
                print_memory_usage()
                
                print(f"✅ Successfully processed {video_name}")
                
            except Exception as e:
                print(f"❌ Failed to process {video_path.name}: {e}")
                failed_videos.append(video_name)
                # Clear memory on failure
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                continue
        
        # Save intermediate results after each batch
        if save_intermediate and all_video_data:
            batch_results_file = Path('full_dora_experiments') / f'batch_{batch_start//batch_size + 1}_results.json'
            with open(batch_results_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_data = {}
                for video_name, data in all_video_data.items():
                    serializable_data[video_name] = {
                        'tokens_shape': data['tokens'].shape,
                        'video_path': data['video_path'],
                        'category': data['category'],
                        'num_frames': len(data['frame_paths'])
                    }
                json.dump(serializable_data, f, indent=2)
            print(f"Intermediate results saved to {batch_results_file}")
        
        # Aggressive memory cleanup after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        print_memory_usage()
        print(f"Batch {batch_start//batch_size + 1} completed. Memory cleared.")
    
    print(f"\n{'='*60}")
    print(f"DATA PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(all_video_data)} videos")
    print(f"Failed videos: {len(failed_videos)}")
    if failed_videos:
        print(f"Failed: {', '.join(failed_videos)}")
    
    if len(all_video_data) < 3:
        print("❌ Insufficient data for analysis. Need at least 3 videos.")
        return None
    
    # Run experimental suite
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENTAL SUITE")
    print(f"{'='*60}")
    
    results = framework.run_full_experimental_suite(all_video_data)
    
    # Generate summary statistics
    generate_summary_statistics(all_video_data, results)
    
    print(f"\n✅ Full experimental suite complete!")
    print(f"Results saved to: full_dora_experiments/")
    
    return results

def generate_summary_statistics(video_data, results):
    """Generate summary statistics for preliminary analysis."""
    
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    # Video statistics
    total_frames = sum(len(data['tokens']) for data in video_data.values())
    avg_frames = total_frames / len(video_data)
    
    print(f"Dataset Statistics:")
    print(f"  Total videos: {len(video_data)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Average frames per video: {avg_frames:.1f}")
    
    # Token statistics (calculate without loading all into memory)
    total_tokens = sum(data['tokens'].shape[0] for data in video_data.values())
    token_dims = list(video_data.values())[0]['tokens'].shape[1:]
    print(f"  Combined token tensor shape: ({total_tokens}, {token_dims[0]}, {token_dims[1]})")
    
    # Motion statistics (if available)
    if 'motion_awareness' in results:
        motion_results = results['motion_awareness']
        motion_magnitudes = []
        motion_coherences = []
        
        for video_name, video_results in motion_results.items():
            if 'motion_analysis' in video_results:
                motion_magnitudes.append(video_results['motion_analysis'].motion_magnitude)
                motion_coherences.append(video_results['motion_analysis'].motion_coherence)
        
        if motion_magnitudes:
            print(f"Motion Analysis:")
            print(f"  Average motion magnitude: {np.mean(motion_magnitudes):.3f} ± {np.std(motion_magnitudes):.3f}")
            print(f"  Average motion coherence: {np.mean(motion_coherences):.3f} ± {np.std(motion_coherences):.3f}")
    
    # Temporal consistency statistics (if available)
    if 'temporal_consistency' in results:
        consistency_results = results['temporal_consistency']
        short_term_stabilities = []
        token_persistences = []
        
        for video_name, video_results in consistency_results.items():
            if 'short_term_stability' in video_results:
                short_term_stabilities.append(np.mean(video_results['short_term_stability']))
            if 'token_persistence' in video_results:
                token_persistences.append(np.mean(video_results['token_persistence']))
        
        if short_term_stabilities:
            print(f"Temporal Consistency:")
            print(f"  Average short-term stability: {np.mean(short_term_stabilities):.3f} ± {np.std(short_term_stabilities):.3f}")
            print(f"  Average token persistence: {np.mean(token_persistences):.1f} ± {np.std(token_persistences):.1f} frames")

def create_preliminary_figures(results):
    """Create preliminary figures for analysis."""
    
    # This will be implemented based on the actual results structure
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full Dora Videos Experimental Suite')
    parser.add_argument('--dry_run', action='store_true', help='Just list videos without processing')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of videos per batch (default: 2)')
    parser.add_argument('--max_videos', type=int, help='Maximum number of videos to process')
    
    args = parser.parse_args()
    
    if args.dry_run:
        dora_dir = Path('dora_videos')
        video_files = list(dora_dir.glob('*.mp4'))
        print(f"Found {len(video_files)} dora videos:")
        for video in video_files:
            print(f"  - {video.name} ({video.stat().st_size / 1e9:.1f} GB)")
    else:
        results = run_full_dora_experiment(batch_size=args.batch_size, max_videos=args.max_videos)
        
        if results:
            print(f"\n🎉 Experiment completed successfully!")
            print(f"Ready for results analysis and preliminary writeup.")
        else:
            print(f"\n❌ Experiment failed. Check logs for details.")
