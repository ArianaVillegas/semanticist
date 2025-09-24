#!/usr/bin/env python3
"""
Cross-Video Token Analysis
Compares token evolution patterns across different videos and creates comparative visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import argparse

def load_video_analysis_data(analysis_dir):
    """Load analysis data from a video analysis directory."""
    import json
    
    # First try to load from consolidated JSON file (new format)
    json_file = Path(analysis_dir) / "experimental_results.json"
    if json_file.exists():
        print(f"Loading from consolidated results: {json_file}")
        with open(json_file, 'r') as f:
            results = json.load(f)
        
        video_data = {}
        for video_name, data in results.get('video_results', {}).items():
            if 'tokens' in data and 'token_similarities' in data:
                video_data[video_name] = {
                    'tokens': np.array(data['tokens']),
                    'similarities': np.array(data['token_similarities']),
                    'distances': np.array(data.get('token_distances', []))
                }
                print(f"Loaded data for {video_name}: {video_data[video_name]['tokens'].shape}")
        
        if video_data:
            return video_data
    
    # Fallback to individual .npy files (old format)
    video_dirs = [d for d in Path(analysis_dir).iterdir() if d.is_dir()]
    
    video_data = {}
    for video_dir in video_dirs:
        video_name = video_dir.name
        
        # Load data files
        tokens_file = video_dir / f"{video_name}_tokens.npy"
        similarities_file = video_dir / f"{video_name}_similarities.npy"
        distances_file = video_dir / f"{video_name}_distances.npy"
        
        if all(f.exists() for f in [tokens_file, similarities_file, distances_file]):
            video_data[video_name] = {
                'tokens': np.load(tokens_file),
                'similarities': np.load(similarities_file),
                'distances': np.load(distances_file)
            }
            print(f"Loaded data for {video_name}: {video_data[video_name]['tokens'].shape}")
    
    return video_data

def analyze_token_consistency_across_videos(video_data, output_dir):
    """Analyze how consistent each token is across different videos."""
    
    # Calculate token stability metrics for each video
    stability_data = []
    
    for video_name, data in video_data.items():
        tokens = data['tokens']
        similarities = data['similarities']
        n_frames, n_tokens, token_dim = tokens.shape
        
        for token_idx in range(n_tokens):
            # Adjacent frame similarity (stability)
            adj_similarities = [similarities[token_idx][i, i+1] for i in range(n_frames-1)]
            avg_stability = np.mean(adj_similarities)
            
            # Token variance (how much it changes)
            token_variance = np.var(tokens[:, token_idx, :]).mean()
            
            # Overall similarity (how similar frames are for this token)
            upper_tri = similarities[token_idx][np.triu_indices_from(similarities[token_idx], k=1)]
            avg_similarity = np.mean(upper_tri)
            
            stability_data.append({
                'video': video_name,
                'token': token_idx + 1,
                'stability': avg_stability,
                'variance': token_variance,
                'avg_similarity': avg_similarity
            })
    
    df = pd.DataFrame(stability_data)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Token stability across videos
    pivot_stability = df.pivot(index='token', columns='video', values='stability')
    sns.heatmap(pivot_stability, ax=axes[0,0], cmap='viridis', annot=True, fmt='.3f')
    axes[0,0].set_title('Token Stability Across Videos')
    axes[0,0].set_ylabel('Token Index')
    
    # 2. Token variance across videos
    pivot_variance = df.pivot(index='token', columns='video', values='variance')
    sns.heatmap(pivot_variance, ax=axes[0,1], cmap='plasma', annot=True, fmt='.3f')
    axes[0,1].set_title('Token Variance Across Videos')
    axes[0,1].set_ylabel('Token Index')
    
    # 3. Average similarity across videos
    pivot_similarity = df.pivot(index='token', columns='video', values='avg_similarity')
    sns.heatmap(pivot_similarity, ax=axes[1,0], cmap='coolwarm', annot=True, fmt='.3f')
    axes[1,0].set_title('Average Token Similarity Across Videos')
    axes[1,0].set_ylabel('Token Index')
    
    # 4. Token ranking by stability
    avg_stability_by_token = df.groupby('token')['stability'].mean().sort_values(ascending=False)
    axes[1,1].bar(range(len(avg_stability_by_token)), avg_stability_by_token.values)
    axes[1,1].set_title('Tokens Ranked by Average Stability')
    axes[1,1].set_xlabel('Token Rank')
    axes[1,1].set_ylabel('Average Stability')
    axes[1,1].set_xticks(range(len(avg_stability_by_token)))
    axes[1,1].set_xticklabels([f'T{i}' for i in avg_stability_by_token.index])
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'cross_video_token_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return df

def create_token_evolution_comparison(video_data, output_dir, max_tokens=8):
    """Create side-by-side comparison of token evolution across videos."""
    
    n_videos = len(video_data)
    video_names = list(video_data.keys())
    
    fig, axes = plt.subplots(max_tokens, n_videos, figsize=(4*n_videos, 3*max_tokens))
    if n_videos == 1:
        axes = axes.reshape(-1, 1)
    
    for video_idx, (video_name, data) in enumerate(video_data.items()):
        similarities = data['similarities']
        n_tokens = min(max_tokens, similarities.shape[0])
        
        for token_idx in range(n_tokens):
            ax = axes[token_idx, video_idx]
            
            # Plot similarity matrix
            im = ax.imshow(similarities[token_idx], cmap='viridis', aspect='auto')
            ax.set_title(f'{video_name}\nToken {token_idx+1}')
            
            if video_idx == 0:
                ax.set_ylabel(f'Token {token_idx+1}')
            if token_idx == n_tokens - 1:
                ax.set_xlabel('Frame Index')
            
            # Add colorbar for the first column
            if video_idx == 0:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'token_evolution_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_token_clustering(video_data, output_dir):
    """Analyze how tokens cluster across different videos."""
    
    # Collect all token vectors from all videos
    all_tokens = []
    token_labels = []
    video_labels = []
    
    for video_name, data in video_data.items():
        tokens = data['tokens']
        n_frames, n_tokens, token_dim = tokens.shape
        
        # Take mean token representation for each token across all frames
        mean_tokens = np.mean(tokens, axis=0)  # Shape: (n_tokens, token_dim)
        
        for token_idx in range(min(16, n_tokens)):  # Limit to first 16 tokens
            all_tokens.append(mean_tokens[token_idx])
            token_labels.append(f'Token_{token_idx+1}')
            video_labels.append(video_name)
    
    all_tokens = np.array(all_tokens)
    
    # Perform PCA and t-SNE
    pca = PCA(n_components=2)
    tokens_pca = pca.fit_transform(all_tokens)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_tokens)-1))
    tokens_tsne = tsne.fit_transform(all_tokens)
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # PCA plot
    unique_videos = list(set(video_labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_videos)))
    
    for i, video in enumerate(unique_videos):
        mask = [v == video for v in video_labels]
        axes[0].scatter(tokens_pca[mask, 0], tokens_pca[mask, 1], 
                       c=[colors[i]], label=video, alpha=0.7, s=60)
    
    axes[0].set_title(f'Token Clustering (PCA)\nExplained Variance: {pca.explained_variance_ratio_.sum():.3f}')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE plot
    for i, video in enumerate(unique_videos):
        mask = [v == video for v in video_labels]
        axes[1].scatter(tokens_tsne[mask, 0], tokens_tsne[mask, 1], 
                       c=[colors[i]], label=video, alpha=0.7, s=60)
    
    axes[1].set_title('Token Clustering (t-SNE)')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'token_clustering_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_summary_report(video_data, stability_df, output_dir):
    """Create a comprehensive summary report."""
    
    report = []
    report.append("# Video Token Evolution Analysis Report\n")
    
    # Video summary
    report.append("## Video Analysis Summary\n")
    for video_name, data in video_data.items():
        tokens = data['tokens']
        n_frames, n_tokens, token_dim = tokens.shape
        report.append(f"- **{video_name}**: {n_frames} frames, {n_tokens} tokens, {token_dim}D embeddings")
    
    report.append("\n## Key Findings\n")
    
    # Most stable tokens
    most_stable_tokens = stability_df.groupby('token')['stability'].mean().sort_values(ascending=False).head(5)
    report.append("### Most Stable Tokens (across all videos):")
    for token_idx, stability in most_stable_tokens.items():
        report.append(f"- Token {token_idx}: {stability:.3f} average stability")
    
    # Most variable tokens
    most_variable_tokens = stability_df.groupby('token')['variance'].mean().sort_values(ascending=False).head(5)
    report.append("\n### Most Variable Tokens (across all videos):")
    for token_idx, variance in most_variable_tokens.items():
        report.append(f"- Token {token_idx}: {variance:.3f} average variance")
    
    # Video-specific insights
    report.append("\n### Video-Specific Insights:")
    for video_name in video_data.keys():
        video_stability = stability_df[stability_df['video'] == video_name]['stability'].mean()
        video_variance = stability_df[stability_df['video'] == video_name]['variance'].mean()
        report.append(f"- **{video_name}**: Avg stability {video_stability:.3f}, Avg variance {video_variance:.3f}")
    
    # Save report
    with open(Path(output_dir) / 'analysis_report.md', 'w') as f:
        f.write('\n'.join(report))
    
    print("Analysis Report:")
    print('\n'.join(report))

def main():
    parser = argparse.ArgumentParser(description='Cross-Video Token Analysis')
    parser.add_argument('--analysis_dir', default='video_token_analysis', help='Directory containing video analysis results')
    parser.add_argument('--output_dir', default='cross_video_analysis', help='Output directory for comparative analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Load all video analysis data
    print("Loading video analysis data...")
    video_data = load_video_analysis_data(args.analysis_dir)
    
    if len(video_data) < 1:
        print("No video analysis data found. Please run video_token_evolution_analysis.py first.")
        return
    
    print(f"Found analysis data for {len(video_data)} videos")
    
    # Perform cross-video analysis
    print("\nAnalyzing token consistency across videos...")
    stability_df = analyze_token_consistency_across_videos(video_data, args.output_dir)
    
    print("\nCreating token evolution comparison...")
    create_token_evolution_comparison(video_data, args.output_dir)
    
    print("\nAnalyzing token clustering...")
    analyze_token_clustering(video_data, args.output_dir)
    
    print("\nGenerating summary report...")
    create_summary_report(video_data, stability_df, args.output_dir)
    
    print(f"\nCross-video analysis complete! Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
