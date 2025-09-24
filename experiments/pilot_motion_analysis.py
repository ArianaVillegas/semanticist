#!/usr/bin/env python3
"""
Pilot Motion Analysis Test
Test the motion analysis pipeline on existing videos to validate framework.
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import json
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

from semanticist.stage1.diffuse_slot import DiffuseSlot
from semanticist.utils.datasets import vae_transforms
from temporal_dynamics_framework import TemporalDynamicsAnalyzer, ExperimentalFramework

def load_semanticist_model(device, cache_dir="./cache"):
    """Load the tokenizer model."""
    ckpt_path = hf_hub_download(
        repo_id='tennant/semanticist', 
        filename='semanticist_tok_XL.pkl', 
        cache_dir=cache_dir
    )
    
    config_path = 'configs/tokenizer_xl.yaml'
    cfg = OmegaConf.load(config_path)
    model = DiffuseSlot(**cfg['trainer']['params']['model']['params'])
    
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    model.enable_nest = True
    
    return model

def load_video_frames(frame_dir, max_frames=30):
    """Load frames from directory."""
    frame_paths = sorted(list(Path(frame_dir).glob("*.jpg")))[:max_frames]
    frames = []
    
    for frame_path in frame_paths:
        frame = np.array(Image.open(frame_path).convert('RGB'))
        frames.append(frame)
    
    return frames

def extract_tokens_from_frames(model, frames, device, num_tokens=16):
    """Extract tokens from frames."""
    transform = vae_transforms('test')
    all_tokens = []
    
    for frame in frames:
        image = Image.fromarray(frame)
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            slots = model.encode_slots(img_tensor)
            all_tokens.append(slots[0, :num_tokens].cpu().numpy())
    
    return np.array(all_tokens)

def run_pilot_experiment():
    """Run pilot experiment on existing video data."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("Loading Semanticist model...")
    model = load_semanticist_model(device)
    
    # Initialize analyzer
    analyzer = TemporalDynamicsAnalyzer(model, device)
    
    # Test on existing videos
    video_dirs = [
        "video_token_analysis/Walking_Tour_Wildlife/frames",
        "video_token_analysis/Walking_Tour_Venice/frames"
    ]
    
    results = {}
    
    for video_dir in video_dirs:
        if not Path(video_dir).exists():
            print(f"Directory not found: {video_dir}")
            continue
            
        video_name = Path(video_dir).parent.name
        print(f"\nAnalyzing {video_name}...")
        
        # Load frames and tokens
        frames = load_video_frames(video_dir, max_frames=30)
        tokens = extract_tokens_from_frames(model, frames, device, num_tokens=16)
        
        print(f"Loaded {len(frames)} frames, tokens shape: {tokens.shape}")
        
        # Motion analysis
        print("Performing motion analysis...")
        motion_analysis = analyzer.analyze_motion_patterns(frames)
        
        print(f"Motion magnitude: {motion_analysis.motion_magnitude:.3f}")
        print(f"Motion coherence: {motion_analysis.motion_coherence:.3f}")
        
        # Temporal consistency analysis
        print("Computing temporal consistency...")
        consistency_metrics = analyzer.compute_hierarchical_temporal_consistency(tokens)
        
        print(f"Average short-term stability: {consistency_metrics.short_term_stability.mean():.3f}")
        print(f"Average token persistence: {consistency_metrics.token_persistence.mean():.3f}")
        
        # Motion-aware analysis
        print("Performing motion-aware token analysis...")
        motion_token_results = analyzer.motion_aware_token_analysis(tokens, motion_analysis)
        
        for motion_type, data in motion_token_results.items():
            print(f"{motion_type}: {data['frame_count']} frames, avg stability: {data['token_stabilities'].mean():.3f}")
        
        # Statistical significance
        significance = analyzer.statistical_significance_testing(motion_token_results)
        print(f"Statistical tests: {len(significance)} comparisons")
        
        results[video_name] = {
            'motion_magnitude': motion_analysis.motion_magnitude,
            'motion_coherence': motion_analysis.motion_coherence,
            'avg_short_term_stability': consistency_metrics.short_term_stability.mean(),
            'avg_token_persistence': consistency_metrics.token_persistence.mean(),
            'motion_token_results': {k: {'avg_stability': v['token_stabilities'].mean(), 
                                       'frame_count': v['frame_count']} 
                                   for k, v in motion_token_results.items()},
            'significance_tests': len(significance)
        }
    
    # Create comparison visualization
    if len(results) >= 2:
        create_pilot_comparison_plot(results)
    
    # Save results
    with open('pilot_experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nPilot experiment complete! Results saved to pilot_experiment_results.json")
    return results

def create_pilot_comparison_plot(results):
    """Create comparison plot for pilot results."""
    video_names = list(results.keys())
    metrics = ['motion_magnitude', 'motion_coherence', 'avg_short_term_stability', 'avg_token_persistence']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [results[video][metric] for video in video_names]
        axes[i].bar(video_names, values)
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('pilot_experiment_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Comparison plot saved as pilot_experiment_comparison.png")

if __name__ == '__main__':
    results = run_pilot_experiment()
    
    print("\n" + "="*60)
    print("PILOT EXPERIMENT SUMMARY")
    print("="*60)
    
    for video_name, data in results.items():
        print(f"\n{video_name}:")
        print(f"  Motion Magnitude: {data['motion_magnitude']:.3f}")
        print(f"  Motion Coherence: {data['motion_coherence']:.3f}")
        print(f"  Avg Short-term Stability: {data['avg_short_term_stability']:.3f}")
        print(f"  Avg Token Persistence: {data['avg_token_persistence']:.3f}")
        print(f"  Statistical Tests: {data['significance_tests']}")
