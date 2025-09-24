#!/usr/bin/env python3
"""
Semantic Analysis and Visualization for Video Token Dynamics
Adds PCA, t-SNE, UMAP analysis and publication-quality plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import json
from pathlib import Path
import torch
import argparse
import gc

# Import functions from existing modules
import sys
sys.path.append('.')
from test_semanticist import load_tokenizer_model, convert_to_numpy
from video_token_evolution_analysis import extract_all_tokens, extract_video_frames

class SemanticAnalysisVisualizer:
    """Enhanced visualization for semantic token analysis"""
    
    def __init__(self, output_dir='semantic_analysis_plots'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_long_form_results(self, results_dir):
        """Load results from long-form experiments"""
        results_dir = Path(results_dir)
        all_data = {}
        
        for video_dir in results_dir.iterdir():
            if video_dir.is_dir():
                results_file = video_dir / f'{video_dir.name}_comprehensive_results.json'
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                        all_data[video_dir.name] = data
                        
        return all_data
    
    def extract_token_matrices(self, results_data):
        """Extract token matrices from results for dimensionality reduction"""
        token_data = {}
        
        for video_name, video_data in results_data.items():
            token_data[video_name] = {}
            
            # Extract tokens from each scale
            for scale_name in video_data['analysis_results'].keys():
                # We need to re-extract tokens since they're not stored in JSON
                # This is a limitation - we should store flattened tokens
                print(f"Note: Token matrices not stored in JSON for {video_name}:{scale_name}")
                
        return token_data
    
    def perform_dimensionality_reduction(self, tokens, methods=['pca', 'tsne', 'umap']):
        """Apply PCA, t-SNE, and UMAP to token data"""
        n_frames, n_tokens, token_dim = tokens.shape
        
        # Flatten tokens: (frames * tokens, token_dim)
        flattened_tokens = tokens.reshape(-1, token_dim)
        
        results = {}
        
        if 'pca' in methods:
            print("Computing PCA...")
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(flattened_tokens)
            results['pca'] = {
                'embedding': pca_result.reshape(n_frames, n_tokens, 2),
                'explained_variance': pca.explained_variance_ratio_,
                'components': pca.components_
            }
            
        if 'tsne' in methods:
            print("Computing t-SNE...")
            # Sample for t-SNE if too many points
            if flattened_tokens.shape[0] > 5000:
                indices = np.random.choice(flattened_tokens.shape[0], 5000, replace=False)
                sample_tokens = flattened_tokens[indices]
            else:
                sample_tokens = flattened_tokens
                indices = np.arange(flattened_tokens.shape[0])
                
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            tsne_result = tsne.fit_transform(sample_tokens)
            
            # Map back to full shape
            full_tsne = np.zeros((flattened_tokens.shape[0], 2))
            full_tsne[indices] = tsne_result
            
            results['tsne'] = {
                'embedding': full_tsne.reshape(n_frames, n_tokens, 2),
                'sampled_indices': indices
            }
            
        if 'umap' in methods:
            print("Computing UMAP...")
            umap_reducer = umap.UMAP(n_components=2, random_state=42)
            umap_result = umap_reducer.fit_transform(flattened_tokens)
            results['umap'] = {
                'embedding': umap_result.reshape(n_frames, n_tokens, 2),
                'reducer': umap_reducer
            }
            
        return results
    
    def plot_semantic_trajectories(self, embeddings, video_name, method='pca'):
        """Plot token trajectories in reduced semantic space"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{video_name}: Token Trajectories in {method.upper()} Space', fontsize=16)
        
        n_frames, n_tokens, _ = embeddings.shape
        
        # Plot 1: All token trajectories
        ax = axes[0, 0]
        for token_idx in range(min(n_tokens, 8)):  # Show first 8 tokens
            trajectory = embeddings[:, token_idx, :]
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   alpha=0.7, linewidth=1, label=f'Token {token_idx}')
            ax.scatter(trajectory[0, 0], trajectory[0, 1], 
                      marker='o', s=50, alpha=0.8)  # Start point
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                      marker='s', s=50, alpha=0.8)  # End point
        ax.set_title('Token Trajectories Over Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Temporal evolution heatmap
        ax = axes[0, 1]
        distances = np.zeros((n_frames-1, n_tokens))
        for t in range(n_frames-1):
            for token_idx in range(n_tokens):
                dist = np.linalg.norm(embeddings[t+1, token_idx] - embeddings[t, token_idx])
                distances[t, token_idx] = dist
        
        im = ax.imshow(distances.T, aspect='auto', cmap='viridis')
        ax.set_title('Token Movement Magnitude')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Token Index')
        plt.colorbar(im, ax=ax)
        
        # Plot 3: Token clustering at different time points
        ax = axes[1, 0]
        time_points = [0, n_frames//3, 2*n_frames//3, n_frames-1]
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, (t, color) in enumerate(zip(time_points, colors)):
            tokens_at_t = embeddings[t, :, :]
            ax.scatter(tokens_at_t[:, 0], tokens_at_t[:, 1], 
                      c=color, alpha=0.6, s=30, label=f't={t}')
        ax.set_title('Token Positions at Different Times')
        ax.legend()
        
        # Plot 4: Semantic stability visualization
        ax = axes[1, 1]
        stability_scores = np.zeros(n_tokens)
        for token_idx in range(n_tokens):
            trajectory = embeddings[:, token_idx, :]
            # Compute path length as inverse stability measure
            path_length = np.sum([np.linalg.norm(trajectory[i+1] - trajectory[i]) 
                                 for i in range(n_frames-1)])
            stability_scores[token_idx] = 1.0 / (1.0 + path_length)
        
        bars = ax.bar(range(n_tokens), stability_scores)
        ax.set_title('Token Semantic Stability')
        ax.set_xlabel('Token Index')
        ax.set_ylabel('Stability Score')
        
        # Color bars by stability
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.RdYlGn(stability_scores[i]))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'{video_name}_{method}_trajectories.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved trajectory plot: {plot_path}")
        
        return fig
    
    def plot_cross_video_comparison(self, all_embeddings, method='pca'):
        """Compare semantic spaces across videos"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Cross-Video Semantic Space Comparison ({method.upper()})', fontsize=16)
        
        video_names = list(all_embeddings.keys())
        
        # Plot individual video spaces
        for i, (video_name, embeddings) in enumerate(all_embeddings.items()):
            if i >= 6:  # Limit to 6 videos
                break
                
            ax = axes[i//3, i%3]
            
            # Plot token clouds for this video
            n_frames, n_tokens, _ = embeddings.shape
            
            # Sample frames for visualization
            sample_frames = np.linspace(0, n_frames-1, min(10, n_frames), dtype=int)
            
            for frame_idx in sample_frames:
                tokens_at_frame = embeddings[frame_idx, :, :]
                alpha = 0.3 + 0.5 * (frame_idx / (n_frames-1))  # Fade over time
                ax.scatter(tokens_at_frame[:, 0], tokens_at_frame[:, 1], 
                          alpha=alpha, s=20, c=plt.cm.viridis(frame_idx / (n_frames-1)))
            
            ax.set_title(f'{video_name}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'cross_video_{method}_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved cross-video comparison: {plot_path}")
        
        return fig
    
    def plot_stability_analysis(self, results_data):
        """Create comprehensive stability analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Temporal Stability Analysis', fontsize=16)
        
        # Collect stability data across all videos
        all_stabilities = {}
        video_names = []
        
        for video_name, video_data in results_data.items():
            video_names.append(video_name)
            
            # Extract stability metrics from each scale
            for scale_name, scale_data in video_data['analysis_results'].items():
                if 'long_term_stability' in scale_data:
                    stability_data = scale_data['long_term_stability']
                    
                    if scale_name not in all_stabilities:
                        all_stabilities[scale_name] = {}
                    
                    for window_name, window_data in stability_data.items():
                        if window_name not in all_stabilities[scale_name]:
                            all_stabilities[scale_name][window_name] = []
                        all_stabilities[scale_name][window_name].append(
                            window_data['mean_stability']
                        )
        
        # Plot 1: Stability vs temporal window size
        ax = axes[0, 0]
        window_names = ['immediate', 'short', 'medium', 'long', 'ultra_long']
        
        for scale_name in ['dense', 'medium', 'sparse', 'ultra_sparse']:
            if scale_name in all_stabilities:
                stabilities = []
                for window_name in window_names:
                    if window_name in all_stabilities[scale_name]:
                        mean_stability = np.mean(all_stabilities[scale_name][window_name])
                        stabilities.append(mean_stability)
                    else:
                        stabilities.append(np.nan)
                
                ax.plot(range(len(window_names)), stabilities, 
                       marker='o', label=scale_name, linewidth=2)
        
        ax.set_xticks(range(len(window_names)))
        ax.set_xticklabels(window_names, rotation=45)
        ax.set_ylabel('Stability Score')
        ax.set_title('Stability vs Temporal Window')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Cross-video stability comparison
        ax = axes[0, 1]
        video_stabilities = []
        for video_name in video_names:
            video_data = results_data[video_name]
            # Get average stability across all scales and windows
            total_stability = 0
            count = 0
            
            for scale_data in video_data['analysis_results'].values():
                if 'temporal_consistency' in scale_data:
                    tc = scale_data['temporal_consistency']
                    for key in ['short_term_stability', 'medium_term_stability', 'long_term_stability']:
                        if key in tc:
                            total_stability += tc[key]
                            count += 1
            
            if count > 0:
                video_stabilities.append(total_stability / count)
            else:
                video_stabilities.append(0)
        
        bars = ax.bar(range(len(video_names)), video_stabilities)
        ax.set_xticks(range(len(video_names)))
        ax.set_xticklabels([name.replace('Walking_Tour_', '') for name in video_names], 
                          rotation=45)
        ax.set_ylabel('Average Stability')
        ax.set_title('Cross-Video Stability Comparison')
        
        # Color bars by stability
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.RdYlGn(video_stabilities[i] / max(video_stabilities)))
        
        # Plot 3: Token hierarchy heatmap
        ax = axes[1, 0]
        # This would need token importance data - placeholder for now
        ax.text(0.5, 0.5, 'Token Hierarchy\n(Requires token importance data)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Token Importance Hierarchy')
        
        # Plot 4: Temporal consistency evolution
        ax = axes[1, 1]
        # This would show how consistency changes over video duration
        ax.text(0.5, 0.5, 'Temporal Consistency Evolution\n(Requires time-series data)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Consistency Over Time')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'stability_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved stability analysis: {plot_path}")
        
        return fig

def extract_tokens_for_analysis(video_path, model, device, num_tokens=16, max_frames=300):
    """Extract tokens from video for semantic analysis"""
    
    # Extract frames
    frames_dir = Path('temp_frames_for_analysis')
    frames_dir.mkdir(exist_ok=True)
    
    frame_paths = extract_video_frames(
        video_path, frames_dir,
        fps=1.0, max_frames=max_frames
    )
    
    # Extract tokens
    tokens, valid_frames = extract_all_tokens(model, frame_paths, device, num_tokens)
    
    # Cleanup
    import shutil
    shutil.rmtree(frames_dir)
    
    return tokens

def main():
    parser = argparse.ArgumentParser(description='Semantic Analysis and Visualization')
    parser.add_argument('--results_dir', default='long_form_experiments', 
                       help='Directory with long-form experiment results')
    parser.add_argument('--video_dir', default='dora_videos',
                       help='Directory with video files')
    parser.add_argument('--num_tokens', type=int, default=16,
                       help='Number of tokens to extract')
    parser.add_argument('--max_frames', type=int, default=300,
                       help='Maximum frames to analyze per video')
    parser.add_argument('--methods', nargs='+', default=['pca', 'umap'],
                       choices=['pca', 'tsne', 'umap'],
                       help='Dimensionality reduction methods to use')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = SemanticAnalysisVisualizer()
    
    # Load existing results for stability analysis
    print("Loading long-form experiment results...")
    results_data = visualizer.load_long_form_results(args.results_dir)
    
    if results_data:
        print("Creating stability analysis plots...")
        visualizer.plot_stability_analysis(results_data)
    
    # Extract tokens for semantic analysis
    print("Extracting tokens for semantic analysis...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_tokenizer_model(device)
    
    video_dir = Path(args.video_dir)
    video_files = list(video_dir.glob('*.mp4'))[:3]  # Limit to 3 videos for demo
    
    all_embeddings = {}
    
    for video_path in video_files:
        print(f"\nProcessing {video_path.name}...")
        
        # Extract tokens
        tokens = extract_tokens_for_analysis(
            video_path, model, device, args.num_tokens, args.max_frames
        )
        
        if len(tokens) < 10:
            print(f"Skipping {video_path.name} - insufficient frames")
            continue
        
        # Perform dimensionality reduction
        embeddings = visualizer.perform_dimensionality_reduction(tokens, args.methods)
        
        video_name = video_path.stem.replace(' ', '_')
        
        # Create trajectory plots for each method
        for method in args.methods:
            if method in embeddings:
                print(f"Creating {method} trajectory plots...")
                visualizer.plot_semantic_trajectories(
                    embeddings[method]['embedding'], video_name, method
                )
                
                all_embeddings[video_name] = embeddings[method]['embedding']
        
        # Memory cleanup
        del tokens, embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Cross-video comparison
    if len(all_embeddings) > 1:
        print("Creating cross-video comparison...")
        visualizer.plot_cross_video_comparison(all_embeddings, args.methods[0])
    
    print(f"\n✅ Semantic analysis complete! Plots saved to: {visualizer.output_dir}")

if __name__ == '__main__':
    main()
