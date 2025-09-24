#!/usr/bin/env python3
"""
Enhanced Long-Form Video Experiment
Comprehensive temporal dynamics analysis for hour-long videos to demonstrate 
the full potential of semantic token stability across extended timeframes.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import gc
import psutil
import os
from tqdm import tqdm
import argparse
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from video_token_evolution_analysis import load_tokenizer_model, extract_video_frames, extract_all_tokens
from temporal_dynamics_framework import TemporalDynamicsAnalyzer

class LongFormTemporalAnalyzer:
    """Enhanced analyzer for long-form video temporal dynamics"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.analyzer = TemporalDynamicsAnalyzer(model, device)
    
    def multi_scale_sampling(self, video_path, output_dir, test_minutes=None):
        """Extract frames at multiple temporal scales for comprehensive analysis"""
        
        # Adaptive sampling strategy based on test duration
        if test_minutes:
            # Testing mode - scale down proportionally
            test_seconds = test_minutes * 60
            sampling_strategies = {
                'dense': {'fps': 1.0, 'max_frames': min(test_seconds, 300), 'start_time': 0},
                'medium': {'fps': 0.5, 'max_frames': min(test_seconds//2, 300), 'start_time': min(test_seconds//4, 300)},
                'sparse': {'fps': 0.25, 'max_frames': min(test_seconds//4, 300), 'start_time': 0},
                'ultra_sparse': {'fps': 0.1, 'max_frames': min(test_seconds//10, 180), 'start_time': 0}
            }
            print(f"Testing mode: Using {test_minutes} minutes of video")
        else:
            # Full analysis mode for 1-hour videos
            sampling_strategies = {
                'dense': {'fps': 1.0, 'max_frames': 300, 'start_time': 0},      # 5 minutes dense
                'medium': {'fps': 0.5, 'max_frames': 600, 'start_time': 300},   # 20 minutes medium  
                'sparse': {'fps': 0.25, 'max_frames': 900, 'start_time': 1500}, # 60 minutes sparse
                'ultra_sparse': {'fps': 0.1, 'max_frames': 360, 'start_time': 0} # Full hour ultra-sparse
            }
        
        all_frame_sets = {}
        
        for scale_name, params in sampling_strategies.items():
            print(f"\nExtracting {scale_name} frames...")
            scale_dir = output_dir / f'frames_{scale_name}'
            
            frame_paths = extract_video_frames(
                video_path, scale_dir,
                fps=params['fps'],
                max_frames=params['max_frames'],
                start_time=params['start_time']
            )
            
            all_frame_sets[scale_name] = frame_paths
            print(f"  {scale_name}: {len(frame_paths)} frames")
        
        return all_frame_sets
    
    def extract_tokens_multi_scale(self, frame_sets, num_tokens=32):
        """Extract tokens for all temporal scales"""
        
        token_sets = {}
        
        for scale_name, frame_paths in frame_sets.items():
            if len(frame_paths) < 10:
                print(f"Skipping {scale_name}: insufficient frames")
                continue
                
            print(f"\nExtracting tokens for {scale_name} scale...")
            tokens, valid_frames = extract_all_tokens(
                self.model, frame_paths, self.device, num_tokens
            )
            
            if len(tokens) >= 10:
                token_sets[scale_name] = {
                    'tokens': tokens,
                    'frame_paths': valid_frames,
                    'scale_params': {'fps': 1.0, 'frames_extracted': len(valid_frames)}
                }
                print(f"  {scale_name}: {tokens.shape}")
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        return token_sets
    
    def analyze_long_term_stability(self, tokens):
        """Analyze stability patterns across extended timeframes"""
        
        n_frames, n_tokens, token_dim = tokens.shape
        
        # Multi-scale temporal windows
        windows = {
            'immediate': 1,      # Adjacent frames
            'short': 5,          # 5-frame windows  
            'medium': 30,        # 30-frame windows
            'long': 120,         # 2-minute windows
            'ultra_long': min(300, n_frames//2)  # 5-minute windows
        }
        
        stability_results = {}
        
        for window_name, window_size in windows.items():
            if n_frames <= window_size:
                continue
                
            stabilities = []
            
            for i in range(n_frames - window_size):
                frame_similarities = []
                
                for t in range(n_tokens):
                    start_token = tokens[i, t, :]
                    end_token = tokens[i + window_size, t, :]
                    
                    # Cosine similarity
                    similarity = np.dot(start_token, end_token) / (
                        np.linalg.norm(start_token) * np.linalg.norm(end_token) + 1e-8
                    )
                    frame_similarities.append(similarity)
                
                stabilities.append(np.mean(frame_similarities))
            
            stability_results[window_name] = {
                'mean_stability': np.mean(stabilities),
                'std_stability': np.std(stabilities),
                'stability_series': stabilities,
                'window_size': window_size
            }
            
            print(f"  {window_name} ({window_size} frames): {np.mean(stabilities):.3f} ± {np.std(stabilities):.3f}")
        
        return stability_results
    
    def analyze_semantic_evolution(self, tokens, time_intervals=10):
        """Analyze how semantic content evolves over long timeframes"""
        
        n_frames, n_tokens, token_dim = tokens.shape
        
        # Divide video into time intervals
        interval_size = n_frames // time_intervals
        if interval_size < 5:
            interval_size = 5
            time_intervals = n_frames // interval_size
        
        evolution_results = {
            'token_centroids': [],
            'interval_similarities': [],
            'semantic_drift_rate': [],
            'token_variance_evolution': []
        }
        
        for interval in range(time_intervals):
            start_idx = interval * interval_size
            end_idx = min((interval + 1) * interval_size, n_frames)
            
            if end_idx - start_idx < 3:
                continue
            
            # Extract interval tokens
            interval_tokens = tokens[start_idx:end_idx]
            
            # Compute centroid for each token
            token_centroids = np.mean(interval_tokens, axis=0)  # Shape: (n_tokens, token_dim)
            evolution_results['token_centroids'].append(token_centroids)
            
            # Compute token variance within interval
            token_variances = np.var(interval_tokens, axis=0).mean(axis=1)  # Per token
            evolution_results['token_variance_evolution'].append(token_variances)
        
        # Compute similarities between consecutive intervals
        for i in range(len(evolution_results['token_centroids']) - 1):
            current_centroids = evolution_results['token_centroids'][i]
            next_centroids = evolution_results['token_centroids'][i + 1]
            
            interval_similarities = []
            for t in range(n_tokens):
                similarity = np.dot(current_centroids[t], next_centroids[t]) / (
                    np.linalg.norm(current_centroids[t]) * np.linalg.norm(next_centroids[t]) + 1e-8
                )
                interval_similarities.append(similarity)
            
            evolution_results['interval_similarities'].append(np.mean(interval_similarities))
        
        # Compute semantic drift rate
        if len(evolution_results['interval_similarities']) > 0:
            drift_rate = 1.0 - np.mean(evolution_results['interval_similarities'])
            evolution_results['semantic_drift_rate'] = drift_rate
        
        return evolution_results
    
    def analyze_token_hierarchy_stability(self, tokens):
        """Analyze which tokens maintain hierarchical importance over time"""
        
        n_frames, n_tokens, token_dim = tokens.shape
        
        # Compute token importance over time using variance
        token_importance_series = []
        window_size = 30
        
        for i in range(0, n_frames - window_size, 10):  # Every 10 frames
            window_tokens = tokens[i:i+window_size]
            
            # Compute importance as inverse of stability (more changing = more important)
            token_importance = []
            for t in range(n_tokens):
                token_sequence = window_tokens[:, t, :]
                # Use variance as importance measure
                importance = np.var(token_sequence, axis=0).mean()
                token_importance.append(importance)
            
            token_importance_series.append(token_importance)
        
        token_importance_series = np.array(token_importance_series)
        
        # Analyze hierarchy stability
        hierarchy_results = {
            'importance_rankings': [],
            'rank_stability': [],
            'top_tokens_consistency': []
        }
        
        for frame_importance in token_importance_series:
            # Rank tokens by importance
            rankings = np.argsort(frame_importance)[::-1]  # Descending order
            hierarchy_results['importance_rankings'].append(rankings)
        
        # Compute rank stability (how consistent are the rankings)
        if len(hierarchy_results['importance_rankings']) > 1:
            rank_correlations = []
            for i in range(len(hierarchy_results['importance_rankings']) - 1):
                current_ranks = hierarchy_results['importance_rankings'][i]
                next_ranks = hierarchy_results['importance_rankings'][i + 1]
                
                # Spearman correlation of rankings
                correlation, _ = stats.spearmanr(current_ranks, next_ranks)
                rank_correlations.append(correlation)
            
            hierarchy_results['rank_stability'] = np.mean(rank_correlations)
            
            # Top-k consistency
            top_k = 8
            top_token_consistency = []
            for i in range(len(hierarchy_results['importance_rankings']) - 1):
                current_top = set(hierarchy_results['importance_rankings'][i][:top_k])
                next_top = set(hierarchy_results['importance_rankings'][i + 1][:top_k])
                consistency = len(current_top.intersection(next_top)) / top_k
                top_token_consistency.append(consistency)
            
            hierarchy_results['top_tokens_consistency'] = np.mean(top_token_consistency)
        
        return hierarchy_results

def process_long_form_video(video_path, output_dir, num_tokens=32, test_minutes=None, model=None):
    """Process a single long-form video with comprehensive analysis"""
    
    video_name = video_path.stem.replace(' ', '_')
    video_output_dir = output_dir / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"PROCESSING LONG-FORM VIDEO: {video_path.name}")
    if test_minutes:
        print(f"TEST MODE: {test_minutes} minutes")
    print(f"{'='*80}")
    
    # Use provided model or load new one
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model = load_tokenizer_model(device)
        model_loaded_here = True
    else:
        model_loaded_here = False
        
    analyzer = LongFormTemporalAnalyzer(model, device)
    
    # Multi-scale frame extraction
    print("Phase 1: Multi-scale frame extraction...")
    frame_sets = analyzer.multi_scale_sampling(video_path, video_output_dir, test_minutes)
    
    # Multi-scale token extraction
    print("Phase 2: Multi-scale token extraction...")
    token_sets = analyzer.extract_tokens_multi_scale(frame_sets, num_tokens)
    
    if not token_sets:
        print(f"❌ No valid token sets extracted for {video_name}")
        return None
    
    # Comprehensive analysis for each scale
    video_results = {
        'video_name': video_name,
        'video_path': str(video_path),
        'scales_analyzed': list(token_sets.keys()),
        'analysis_results': {}
    }
    
    for scale_name, token_data in token_sets.items():
        print(f"\nPhase 3: Analyzing {scale_name} scale...")
        tokens = token_data['tokens']
        
        scale_results = {}
        
        # Long-term stability analysis
        print(f"  Long-term stability analysis...")
        stability_results = analyzer.analyze_long_term_stability(tokens)
        scale_results['long_term_stability'] = stability_results
        
        # Semantic evolution analysis
        print(f"  Semantic evolution analysis...")
        evolution_results = analyzer.analyze_semantic_evolution(tokens)
        scale_results['semantic_evolution'] = evolution_results
        
        # Token hierarchy stability
        print(f"  Token hierarchy analysis...")
        hierarchy_results = analyzer.analyze_token_hierarchy_stability(tokens)
        scale_results['hierarchy_stability'] = hierarchy_results
        
        # Standard temporal consistency
        print(f"  Standard temporal consistency...")
        consistency_metrics = analyzer.analyzer.compute_hierarchical_temporal_consistency(tokens)
        scale_results['temporal_consistency'] = {
            'short_term_stability': consistency_metrics.short_term_stability.mean(),
            'medium_term_stability': consistency_metrics.medium_term_stability.mean(),
            'long_term_stability': consistency_metrics.long_term_stability.mean(),
            'token_persistence': consistency_metrics.token_persistence.mean()
        }
        
        video_results['analysis_results'][scale_name] = scale_results
        
        # Memory cleanup
        del tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Save comprehensive results
    results_file = video_output_dir / f'{video_name}_comprehensive_results.json'
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_numpy(video_results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"✅ Comprehensive results saved to: {results_file}")
    
    # Cleanup (only if we loaded the model here)
    if model_loaded_here:
        del model
    del analyzer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return video_results

def main():
    parser = argparse.ArgumentParser(description='Enhanced Long-Form Video Experiment')
    parser.add_argument('--num_tokens', type=int, default=32, help='Number of tokens to extract')
    parser.add_argument('--test_minutes', type=int, help='Test with only N minutes of video (for quick testing)')
    parser.add_argument('--dry_run', action='store_true', help='Just list videos without processing')
    
    args = parser.parse_args()
    
    output_dir = Path('long_form_experiments')
    output_dir.mkdir(exist_ok=True)
    
    # Get dora videos
    dora_dir = Path('dora_videos')
    video_files = list(dora_dir.glob('*.mp4'))
    
    if args.dry_run:
        print(f"Found {len(video_files)} long-form videos:")
        for i, video in enumerate(video_files):
            duration_hours = video.stat().st_size / 1e9 / 0.5  # Rough estimate
            print(f"  {i+1}: {video.name} (~{duration_hours:.1f}h estimated)")
        return
    
    print(f"Processing {len(video_files)} long-form videos...")
    print(f"Tokens per frame: {args.num_tokens}")
    if args.test_minutes:
        print(f"Test mode: {args.test_minutes} minutes per video")
    
    # Load model once for all videos (memory efficient)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model once for all videos...")
    model = load_tokenizer_model(device)
    
    all_results = []
    
    for video_path in video_files:
        result = process_long_form_video(
            video_path, output_dir, args.num_tokens, 
            test_minutes=args.test_minutes, model=model
        )
        if result:
            all_results.append(result)
    
    # Cleanup shared model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Save combined results with numpy conversion
    combined_file = output_dir / 'combined_long_form_results.json'
    
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_numpy(all_results)
    
    with open(combined_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n🎉 Enhanced long-form experiment complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Videos processed: {len(all_results)}")

if __name__ == '__main__':
    main()
