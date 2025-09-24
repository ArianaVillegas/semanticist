#!/usr/bin/env python3
"""
Temporal Dynamics Framework for Semantic Token Analysis
Comprehensive experimental framework for CVPR 2025 submission.
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import silhouette_score
from scipy import stats
import pandas as pd

@dataclass
class VideoMetadata:
    """Metadata for video analysis"""
    path: str
    category: str  # urban, nature, sports, indoor, etc.
    motion_type: str  # static, camera_motion, object_motion, mixed
    duration: float
    fps: float
    resolution: Tuple[int, int]

@dataclass
class MotionAnalysis:
    """Motion analysis results"""
    optical_flow: np.ndarray
    camera_motion: np.ndarray  # estimated camera motion parameters
    object_motion_mask: np.ndarray  # binary mask for object motion
    motion_magnitude: float
    motion_coherence: float  # how coherent the motion is

@dataclass
class TemporalConsistencyMetrics:
    """Temporal consistency metrics for tokens"""
    short_term_stability: np.ndarray  # adjacent frame similarity
    medium_term_stability: np.ndarray  # 5-frame window similarity
    long_term_stability: np.ndarray  # 30-frame window similarity
    token_persistence: np.ndarray  # how long tokens remain stable
    semantic_drift: np.ndarray  # gradual semantic change over time

class TemporalDynamicsAnalyzer:
    """Main analyzer for temporal dynamics of semantic tokens"""
    
    def __init__(self, semanticist_model, device='cuda'):
        self.model = semanticist_model
        self.device = device
        self.video_categories = [
            'urban_walking', 'nature_scenes', 'indoor_spaces', 
            'vehicle_motion', 'sports_action', 'static_scenes',
            'crowd_scenes', 'architectural'
        ]
        
    def extract_optical_flow_from_paths(self, frame_paths: List[str]) -> List[np.ndarray]:
        """Extract optical flow between consecutive frames using Farneback method (memory efficient)"""
        flows = []
        
        if len(frame_paths) < 2:
            return flows
            
        # Load first frame
        from PIL import Image
        prev_frame = np.array(Image.open(frame_paths[0]).convert('RGB'))
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        
        for i in range(1, min(len(frame_paths), 30)):  # Limit to 30 frames for memory
            curr_frame = np.array(Image.open(frame_paths[i]).convert('RGB'))
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
            
            # Use Farneback optical flow (dense optical flow)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flows.append(flow)
            prev_gray = curr_gray
            
            # Clear frame from memory
            del curr_frame
            
        del prev_frame
        return flows
    
    def estimate_camera_motion(self, optical_flows: List[np.ndarray]) -> List[np.ndarray]:
        """Estimate camera motion parameters from optical flow"""
        camera_motions = []
        
        for flow in optical_flows:
            # Simplified camera motion estimation from dense flow
            # Calculate mean flow vector as camera motion estimate
            if flow is not None and flow.size > 0:
                # For dense flow, compute mean motion
                mean_flow_x = np.mean(flow[:, :, 0])
                mean_flow_y = np.mean(flow[:, :, 1])
                camera_motion = np.array([mean_flow_x, mean_flow_y])
            else:
                camera_motion = np.array([0.0, 0.0])
                
            camera_motions.append(camera_motion)
            
        return camera_motions
    
    def analyze_motion_patterns_from_paths(self, frame_paths: List[str]) -> MotionAnalysis:
        """Comprehensive motion analysis (memory efficient)"""
        optical_flows = self.extract_optical_flow_from_paths(frame_paths)
        camera_motions = self.estimate_camera_motion(optical_flows)
        
        # Calculate motion statistics
        if optical_flows:
            motion_magnitude = np.mean([np.linalg.norm(flow) for flow in optical_flows])
            motion_coherence = np.std([np.linalg.norm(cm) for cm in camera_motions]) if camera_motions else 0.0
        else:
            motion_magnitude = 0.0
            motion_coherence = 0.0
        
        return MotionAnalysis(
            optical_flow=np.array(optical_flows) if optical_flows else np.array([]),
            camera_motion=np.array(camera_motions) if camera_motions else np.array([]),
            object_motion_mask=np.array([]),  # Empty to save memory
            motion_magnitude=motion_magnitude,
            motion_coherence=motion_coherence
        )
    
    def compute_hierarchical_temporal_consistency(self, tokens: np.ndarray) -> TemporalConsistencyMetrics:
        """Compute multi-scale temporal consistency metrics"""
        n_frames, n_tokens, token_dim = tokens.shape
        
        # Short-term stability (adjacent frames)
        short_term = []
        for i in range(n_frames - 1):
            similarities = []
            for t in range(n_tokens):
                sim = np.dot(tokens[i, t], tokens[i+1, t]) / (
                    np.linalg.norm(tokens[i, t]) * np.linalg.norm(tokens[i+1, t]) + 1e-8
                )
                similarities.append(sim)
            short_term.append(similarities)
        short_term_stability = np.array(short_term)
        
        # Medium-term stability (5-frame windows)
        window_size = 5
        medium_term = []
        for i in range(n_frames - window_size):
            similarities = []
            for t in range(n_tokens):
                window_tokens = tokens[i:i+window_size, t, :]
                # Compute pairwise similarities within window
                window_sims = []
                for j in range(window_size-1):
                    for k in range(j+1, window_size):
                        sim = np.dot(window_tokens[j], window_tokens[k]) / (
                            np.linalg.norm(window_tokens[j]) * np.linalg.norm(window_tokens[k]) + 1e-8
                        )
                        window_sims.append(sim)
                similarities.append(np.mean(window_sims))
            medium_term.append(similarities)
        medium_term_stability = np.array(medium_term)
        
        # Long-term stability (30-frame windows)
        long_window_size = min(30, n_frames // 2)
        long_term = []
        for i in range(n_frames - long_window_size):
            similarities = []
            for t in range(n_tokens):
                start_token = tokens[i, t, :]
                end_token = tokens[i + long_window_size, t, :]
                sim = np.dot(start_token, end_token) / (
                    np.linalg.norm(start_token) * np.linalg.norm(end_token) + 1e-8
                )
                similarities.append(sim)
            long_term.append(similarities)
        long_term_stability = np.array(long_term)
        
        # Token persistence (how long tokens remain above similarity threshold)
        threshold = 0.8
        persistence = []
        for t in range(n_tokens):
            token_sequence = tokens[:, t, :]
            persist_lengths = []
            current_length = 1
            
            for i in range(1, len(token_sequence)):
                sim = np.dot(token_sequence[i-1], token_sequence[i]) / (
                    np.linalg.norm(token_sequence[i-1]) * np.linalg.norm(token_sequence[i]) + 1e-8
                )
                if sim > threshold:
                    current_length += 1
                else:
                    persist_lengths.append(current_length)
                    current_length = 1
            persist_lengths.append(current_length)
            persistence.append(np.mean(persist_lengths))
        
        # Semantic drift (gradual change over time)
        semantic_drift = []
        for t in range(n_tokens):
            token_sequence = tokens[:, t, :]
            drifts = []
            for i in range(0, len(token_sequence) - 10, 10):  # Every 10 frames
                if i + 10 < len(token_sequence):
                    start = token_sequence[i]
                    end = token_sequence[i + 10]
                    drift = 1.0 - np.dot(start, end) / (
                        np.linalg.norm(start) * np.linalg.norm(end) + 1e-8
                    )
                    drifts.append(drift)
            semantic_drift.append(np.mean(drifts))
        
        return TemporalConsistencyMetrics(
            short_term_stability=short_term_stability,
            medium_term_stability=medium_term_stability,
            long_term_stability=long_term_stability,
            token_persistence=np.array(persistence),
            semantic_drift=np.array(semantic_drift)
        )
    
    def motion_aware_token_analysis(self, tokens: np.ndarray, motion_analysis: MotionAnalysis) -> Dict:
        """Analyze how tokens respond to different types of motion"""
        n_frames, n_tokens, token_dim = tokens.shape
        
        # Categorize frames by motion type
        motion_magnitudes = [np.linalg.norm(cm) for cm in motion_analysis.camera_motion]
        low_motion_threshold = np.percentile(motion_magnitudes, 33)
        high_motion_threshold = np.percentile(motion_magnitudes, 67)
        
        low_motion_frames = [i for i, mag in enumerate(motion_magnitudes) if mag < low_motion_threshold]
        medium_motion_frames = [i for i, mag in enumerate(motion_magnitudes) 
                              if low_motion_threshold <= mag < high_motion_threshold]
        high_motion_frames = [i for i, mag in enumerate(motion_magnitudes) if mag >= high_motion_threshold]
        
        results = {}
        
        # Analyze token stability under different motion conditions
        for motion_type, frame_indices in [
            ('low_motion', low_motion_frames),
            ('medium_motion', medium_motion_frames), 
            ('high_motion', high_motion_frames)
        ]:
            if len(frame_indices) < 2:
                continue
                
            motion_tokens = tokens[frame_indices]
            token_stabilities = []
            
            for t in range(n_tokens):
                token_seq = motion_tokens[:, t, :]
                similarities = []
                for i in range(len(token_seq) - 1):
                    sim = np.dot(token_seq[i], token_seq[i+1]) / (
                        np.linalg.norm(token_seq[i]) * np.linalg.norm(token_seq[i+1]) + 1e-8
                    )
                    similarities.append(sim)
                token_stabilities.append(np.mean(similarities))
            
            results[motion_type] = {
                'token_stabilities': np.array(token_stabilities),
                'frame_count': len(frame_indices),
                'avg_motion_magnitude': np.mean([motion_magnitudes[i] for i in frame_indices])
            }
        
        return results
    
    def statistical_significance_testing(self, results_dict: Dict) -> Dict:
        """Perform statistical significance testing on results"""
        significance_results = {}
        
        # Test if token stabilities are significantly different across motion types
        if len(results_dict) >= 2:
            motion_types = list(results_dict.keys())
            for i in range(len(motion_types)):
                for j in range(i+1, len(motion_types)):
                    type1, type2 = motion_types[i], motion_types[j]
                    stab1 = results_dict[type1]['token_stabilities']
                    stab2 = results_dict[type2]['token_stabilities']
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(stab1, stab2)
                    
                    significance_results[f'{type1}_vs_{type2}'] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'effect_size': (np.mean(stab1) - np.mean(stab2)) / np.sqrt(
                            (np.var(stab1) + np.var(stab2)) / 2
                        )
                    }
        
        return significance_results

class ExperimentalFramework:
    """Main experimental framework for the paper"""
    
    def __init__(self, semanticist_model, output_dir='temporal_dynamics_experiments'):
        self.analyzer = TemporalDynamicsAnalyzer(semanticist_model)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define experimental conditions
        self.experiments = {
            'temporal_consistency': self.run_temporal_consistency_experiment,
            'motion_awareness': self.run_motion_awareness_experiment,
            'hierarchical_modeling': self.run_hierarchical_modeling_experiment,
            'cross_domain_analysis': self.run_cross_domain_experiment
        }
    
    def run_temporal_consistency_experiment(self, video_data: Dict) -> Dict:
        """Experiment 1: Temporal Consistency Analysis"""
        results = {}
        
        for video_name, data in video_data.items():
            tokens = data['tokens']
            
            # Compute temporal consistency metrics
            consistency_metrics = self.analyzer.compute_hierarchical_temporal_consistency(tokens)
            
            # Clear tokens from memory after processing
            import gc
            del tokens
            gc.collect()
            
            results[video_name] = {
                'short_term_stability': consistency_metrics.short_term_stability.mean(axis=0),
                'medium_term_stability': consistency_metrics.medium_term_stability.mean(axis=0),
                'long_term_stability': consistency_metrics.long_term_stability.mean(axis=0),
                'token_persistence': consistency_metrics.token_persistence,
                'semantic_drift': consistency_metrics.semantic_drift
            }
        
        return results
    
    def run_motion_awareness_experiment(self, video_data: Dict) -> Dict:
        """Experiment 2: Motion-Aware Token Analysis"""
        results = {}
        
        for video_name, data in video_data.items():
            tokens = data['tokens']
            frame_paths = data['frame_paths']
            
            # Analyze motion patterns (memory efficient)
            motion_analysis = self.analyzer.analyze_motion_patterns_from_paths(frame_paths)
            
            # Motion-aware token analysis
            motion_results = self.analyzer.motion_aware_token_analysis(tokens, motion_analysis)
            
            # Clear tokens from memory after processing
            import gc
            del tokens
            gc.collect()
            
            # Statistical significance testing
            significance = self.analyzer.statistical_significance_testing(motion_results)
            
            results[video_name] = {
                'motion_analysis': motion_analysis,
                'motion_token_results': motion_results,
                'statistical_significance': significance
            }
        
        return results
    
    def run_hierarchical_modeling_experiment(self, video_data: Dict) -> Dict:
        """Experiment 3: Hierarchical Temporal Modeling"""
        # Implementation for hierarchical analysis
        pass
    
    def run_cross_domain_experiment(self, video_data: Dict) -> Dict:
        """Experiment 4: Cross-Domain Generalization"""
        # Implementation for cross-domain analysis
        pass
    
    def generate_paper_figures(self, results: Dict):
        """Generate publication-quality figures"""
        # Implementation for figure generation
        pass
    
    def run_full_experimental_suite(self, video_data: Dict) -> Dict:
        """Run all experiments and generate results"""
        all_results = {}
        
        for exp_name, exp_func in self.experiments.items():
            print(f"Running {exp_name} experiment...")
            all_results[exp_name] = exp_func(video_data)
        
        # Generate figures and save results
        self.generate_paper_figures(all_results)
        
        # Save results
        with open(self.output_dir / 'experimental_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        return all_results

if __name__ == '__main__':
    # Example usage
    print("Temporal Dynamics Framework initialized")
    print("Ready for comprehensive video token analysis")
