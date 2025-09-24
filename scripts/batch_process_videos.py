#!/usr/bin/env python3
"""
Ultra Memory-Efficient Batch Video Processing
Process videos one at a time with immediate result saving to handle large datasets.
"""

import torch
import numpy as np
import json
from pathlib import Path
import gc
import psutil
import os
from tqdm import tqdm
import argparse

from video_token_evolution_analysis import load_tokenizer_model, extract_video_frames, extract_all_tokens
from temporal_dynamics_framework import TemporalDynamicsAnalyzer, ExperimentalFramework

def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_max_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"Memory: RAM {memory_mb:.1f}MB, GPU {gpu_memory_mb:.1f}MB (max: {gpu_max_mb:.1f}MB)")
    else:
        print(f"Memory: RAM {memory_mb:.1f}MB")

def aggressive_cleanup():
    """Aggressive memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

def process_single_video(video_path, model, device, params, output_dir):
    """Process a single video and return results immediately"""
    video_name = video_path.stem.replace(' ', '_')
    
    print(f"\n{'='*60}")
    print(f"Processing: {video_path.name}")
    print(f"{'='*60}")
    
    try:
        # Create output directory
        video_output_dir = output_dir / video_name
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
            print(f"WARNING: Only {len(frame_paths)} frames extracted, skipping...")
            return None
        
        # Extract tokens
        print("Extracting semantic tokens...")
        tokens, valid_frames = extract_all_tokens(model, frame_paths, device, params['num_tokens'])
        
        if len(tokens) < 10:
            print(f"WARNING: Only {len(tokens)} valid tokens, skipping...")
            return None
        
        print(f"Tokens shape: {tokens.shape}")
        print_memory_usage()
        
        # Create analyzer for this video only
        analyzer = TemporalDynamicsAnalyzer(model, device)
        
        # Compute temporal consistency (most memory intensive)
        print("Computing temporal consistency...")
        consistency_metrics = analyzer.compute_hierarchical_temporal_consistency(tokens)
        
        # Analyze motion patterns from paths (memory efficient)
        print("Analyzing motion patterns...")
        motion_analysis = analyzer.analyze_motion_patterns_from_paths(valid_frames[:30])
        
        # Motion-aware token analysis
        print("Motion-aware token analysis...")
        motion_results = analyzer.motion_aware_token_analysis(tokens, motion_analysis)
        
        # Statistical significance testing
        significance = analyzer.statistical_significance_testing(motion_results)
        
        # Prepare results
        video_results = {
            'video_name': video_name,
            'video_path': str(video_path),
            'tokens_shape': tokens.shape,
            'num_frames': len(valid_frames),
            'temporal_consistency': {
                'short_term_stability': consistency_metrics.short_term_stability.tolist(),
                'medium_term_stability': consistency_metrics.medium_term_stability.tolist(),
                'long_term_stability': consistency_metrics.long_term_stability.tolist(),
                'token_persistence': consistency_metrics.token_persistence.tolist(),
                'semantic_drift': consistency_metrics.semantic_drift.tolist()
            },
            'motion_awareness': {
                'motion_magnitude': motion_analysis.motion_magnitude,
                'motion_coherence': motion_analysis.motion_coherence,
                'motion_token_results': motion_results,
                'statistical_significance': significance
            }
        }
        
        # Save individual video results immediately
        video_results_file = video_output_dir / f'{video_name}_results.json'
        video_output_dir.mkdir(parents=True, exist_ok=True)
        with open(video_results_file, 'w') as f:
            json.dump(video_results, f, indent=2)
        
        print(f"Results saved to: {video_results_file}")
        
        # Cleanup large objects
        del tokens, consistency_metrics, motion_analysis, motion_results, analyzer
        aggressive_cleanup()
        print_memory_usage()
        
        print(f"✅ Successfully processed {video_name}")
        return video_results
        
    except Exception as e:
        print(f"❌ Failed to process {video_path.name}: {e}")
        aggressive_cleanup()
        return None

def combine_results(output_dir):
    """Combine all individual video results into final experimental results"""
    print("\nCombining individual results...")
    
    combined_results = {
        'temporal_consistency': {},
        'motion_awareness': {},
        'hierarchical_modeling': {},
        'cross_domain_analysis': {}
    }
    
    # Find all individual result files
    result_files = list(output_dir.glob('*/*_results.json'))
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                video_results = json.load(f)
            
            video_name = video_results['video_name']
            
            # Add to combined results
            combined_results['temporal_consistency'][video_name] = video_results['temporal_consistency']
            combined_results['motion_awareness'][video_name] = video_results['motion_awareness']
            
            print(f"Added {video_name} to combined results")
            
        except Exception as e:
            print(f"Failed to load {result_file}: {e}")
    
    # Save combined results
    combined_file = output_dir / 'experimental_results.json'
    with open(combined_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"Combined results saved to: {combined_file}")
    return combined_results

def main():
    parser = argparse.ArgumentParser(description='Ultra Memory-Efficient Video Processing')
    parser.add_argument('--max_videos', type=int, help='Maximum number of videos to process')
    parser.add_argument('--start_from', type=int, default=0, help='Start from video index (for resuming)')
    parser.add_argument('--dry_run', action='store_true', help='Just list videos without processing')
    parser.add_argument('--combine_only', action='store_true', help='Only combine existing results')
    
    args = parser.parse_args()
    
    output_dir = Path('full_dora_experiments')
    output_dir.mkdir(exist_ok=True)
    
    # Get all dora videos
    dora_dir = Path('dora_videos')
    video_files = list(dora_dir.glob('*.mp4'))
    
    if args.max_videos:
        video_files = video_files[:args.max_videos]
    
    if args.dry_run:
        print(f"Found {len(video_files)} dora videos:")
        for i, video in enumerate(video_files):
            print(f"  {i}: {video.name} ({video.stat().st_size / 1e9:.1f} GB)")
        return
    
    if args.combine_only:
        combine_results(output_dir)
        return
    
    # Memory-efficient parameters
    params = {
        'fps': 1.0,
        'max_frames': 60,      # Reduced for memory
        'num_tokens': 16,      # Reduced for memory  
        'start_time': 30,
    }
    
    print(f"Processing {len(video_files)} videos with parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    print_memory_usage()
    
    # Load model once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    print("Loading Semanticist model...")
    model = load_tokenizer_model(device)
    print_memory_usage()
    
    # Process videos one at a time
    successful_videos = []
    failed_videos = []
    
    for i, video_path in enumerate(tqdm(video_files[args.start_from:], desc="Processing videos")):
        actual_index = i + args.start_from
        print(f"\n[{actual_index + 1}/{len(video_files)}] Processing video...")
        
        result = process_single_video(video_path, model, device, params, output_dir)
        
        if result:
            successful_videos.append(result['video_name'])
        else:
            failed_videos.append(video_path.stem.replace(' ', '_'))
        
        # Memory check after each video
        print_memory_usage()
        
        # Optional: Save checkpoint every 3 videos
        if (actual_index + 1) % 3 == 0:
            checkpoint_file = output_dir / f'checkpoint_{actual_index + 1}.json'
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'processed_videos': actual_index + 1,
                    'successful': successful_videos,
                    'failed': failed_videos
                }, f, indent=2)
            print(f"Checkpoint saved: {checkpoint_file}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(successful_videos)} videos")
    print(f"Failed videos: {len(failed_videos)}")
    
    if failed_videos:
        print(f"Failed: {', '.join(failed_videos)}")
    
    # Combine all results
    if successful_videos:
        combine_results(output_dir)
        print(f"\n🎉 All videos processed and results combined!")
    else:
        print(f"\n❌ No videos were successfully processed.")

if __name__ == '__main__':
    main()
