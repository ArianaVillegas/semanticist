#!/usr/bin/env python3
"""
Benchmark Semantic Token Compression Against SOTA Methods
Test on standard video compression datasets and compare with H.264, H.265, AV1, etc.
"""

import os
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from urllib.request import urlretrieve
import zipfile
import tarfile

from video_compression_experiments import test_video_dataset_compression
from test_semanticist import load_tokenizer_model, setup_device

# Standard video compression test datasets
STANDARD_DATASETS = {
    'sample_videos': {
        'videos': [
            {
                'url': 'https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4',
                'name': 'sample_720p.mp4'
            },
            {
                'url': 'https://sample-videos.com/zip/10/mp4/SampleVideo_640x360_1mb.mp4', 
                'name': 'sample_360p.mp4'
            },
            {
                'url': 'https://sample-videos.com/zip/10/mp4/SampleVideo_1920x1080_1mb.mp4',
                'name': 'sample_1080p.mp4'
            }
        ],
        'description': 'Sample test videos for compression benchmarking'
    },
    'big_buck_bunny': {
        'videos': [
            {
                'url': 'https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4',
                'name': 'bigbuckbunny_320x180.mp4'
            },
            {
                'url': 'https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_640x360.mp4',
                'name': 'bigbuckbunny_640x360.mp4'
            }
        ],
        'description': 'Big Buck Bunny - Creative Commons test video'
    },
    'hevc_test_sequences': {
        'url': 'ftp://ftp.kw.bbc.co.uk/hevc/hm-16.20+SCM-8.8/',
        'videos': [
            'BasketballDrill_832x480_50.yuv',
            'BQMall_832x480_60.yuv',
            'PartyScene_832x480_50.yuv',
            'RaceHorses_832x480_30.yuv'
        ],
        'description': 'HEVC test sequences used in standardization'
    },
    'jvet_test_sequences': {
        'url': 'https://www.itu.int/wftp3/av-arch/jvet-site/bitstream_exchange/VVC_draft_bitstreams/',
        'description': 'JVET VVC test sequences (latest standard)'
    }
}

# SOTA compression methods to benchmark against
COMPRESSION_METHODS = {
    'x264': {
        'command': 'ffmpeg -i {input} -c:v libx264 -preset medium -crf {quality} {output}',
        'qualities': [18, 23, 28, 33],  # CRF values
        'description': 'H.264/AVC (x264)'
    },
    'x265': {
        'command': 'ffmpeg -i {input} -c:v libx265 -preset medium -crf {quality} {output}',
        'qualities': [18, 23, 28, 33],
        'description': 'H.265/HEVC (x265)'
    },
    'av1': {
        'command': 'ffmpeg -i {input} -c:v libaom-av1 -crf {quality} -b:v 0 {output}',
        'qualities': [20, 30, 40, 50],
        'description': 'AV1 (libaom)'
    },
    'vp9': {
        'command': 'ffmpeg -i {input} -c:v libvpx-vp9 -crf {quality} -b:v 0 {output}',
        'qualities': [20, 30, 40, 50],
        'description': 'VP9 (libvpx)'
    }
}

def download_test_videos(output_dir="benchmark_videos"):
    """Download standard test videos."""
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded = []
    
    # Download from all available datasets
    for dataset_name, dataset_info in STANDARD_DATASETS.items():
        if 'videos' in dataset_info and isinstance(dataset_info['videos'], list):
            for video_info in dataset_info['videos']:
                if isinstance(video_info, dict):
                    url = video_info['url']
                    filename = video_info['name']
                else:
                    continue
                    
                video_path = Path(output_dir) / filename
                if video_path.exists():
                    print(f"Already exists: {filename}")
                    downloaded.append(str(video_path))
                    continue
                    
                try:
                    print(f"Downloading: {filename}")
                    urlretrieve(url, video_path)
                    downloaded.append(str(video_path))
                    print(f"Downloaded: {filename}")
                except Exception as e:
                    print(f"Failed to download {filename}: {e}")
    
    return downloaded

def convert_yuv_to_mp4(yuv_files, output_dir="benchmark_videos"):
    """Convert YUV files to MP4 for processing."""
    converted = []
    
    for yuv_file in yuv_files:
        if not yuv_file.endswith('.yuv') and not yuv_file.endswith('.y4m'):
            converted.append(yuv_file)
            continue
            
        mp4_file = Path(output_dir) / (Path(yuv_file).stem + '.mp4')
        if mp4_file.exists():
            converted.append(str(mp4_file))
            continue
            
        try:
            if yuv_file.endswith('.y4m'):
                # Y4M files have header with format info
                cmd = f"ffmpeg -i {yuv_file} -c:v libx264 -preset fast -crf 18 {mp4_file}"
            else:
                # YUV files need format specification (assuming common format)
                cmd = f"ffmpeg -f rawvideo -pix_fmt yuv420p -s 832x480 -r 30 -i {yuv_file} -c:v libx264 -preset fast -crf 18 {mp4_file}"
            
            print(f"Converting: {Path(yuv_file).name}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                converted.append(str(mp4_file))
                print(f"Converted: {mp4_file.name}")
            else:
                print(f"Failed to convert {yuv_file}: {result.stderr}")
        except Exception as e:
            print(f"Error converting {yuv_file}: {e}")
    
    return converted

def benchmark_traditional_compression(video_files, output_dir="benchmark_results"):
    """Benchmark traditional compression methods."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for video_file in video_files:
        video_name = Path(video_file).stem
        print(f"Benchmarking traditional compression for: {video_name}")
        
        # Get original file size
        original_size = os.path.getsize(video_file)
        
        for method_name, method_config in COMPRESSION_METHODS.items():
            for quality in method_config['qualities']:
                output_file = Path(output_dir) / f"{video_name}_{method_name}_q{quality}.mp4"
                
                if output_file.exists():
                    compressed_size = os.path.getsize(output_file)
                else:
                    try:
                        cmd = method_config['command'].format(
                            input=video_file,
                            quality=quality,
                            output=output_file
                        )
                        
                        print(f"  Running: {method_name} quality {quality}")
                        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                        
                        if result.returncode == 0 and output_file.exists():
                            compressed_size = os.path.getsize(output_file)
                        else:
                            print(f"    Failed: {result.stderr}")
                            continue
                    except Exception as e:
                        print(f"    Error: {e}")
                        continue
                
                compression_ratio = original_size / compressed_size
                
                results.append({
                    'video': video_name,
                    'method': method_name,
                    'quality': quality,
                    'original_size_mb': original_size / (1024*1024),
                    'compressed_size_mb': compressed_size / (1024*1024),
                    'compression_ratio': compression_ratio,
                    'description': method_config['description']
                })
    
    return results

def compare_with_semantic_compression(video_files, model, device, traditional_results):
    """Compare semantic token compression with traditional methods."""
    print("Running semantic token compression...")
    
    # Run semantic compression
    semantic_results = test_video_dataset_compression(
        Path(video_files[0]).parent, model, device, len(video_files)
    )
    
    # Create comparison dataframe
    comparison_data = []
    
    # Add traditional results
    for result in traditional_results:
        comparison_data.append({
            'video': result['video'],
            'method': result['description'],
            'compression_ratio': result['compression_ratio'],
            'type': 'traditional'
        })
    
    # Add semantic results
    for result in semantic_results:
        comparison_data.append({
            'video': result['video'].replace('.mp4', ''),
            'method': 'Semantic Tokens',
            'compression_ratio': result['compression_ratio'],
            'type': 'semantic'
        })
    
    return pd.DataFrame(comparison_data)

def plot_compression_comparison(comparison_df, output_dir="benchmark_results"):
    """Plot comprehensive comparison of compression methods."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Compression ratios by method
    method_ratios = comparison_df.groupby('method')['compression_ratio'].mean().sort_values(ascending=False)
    
    colors = ['red' if 'Semantic' in method else 'blue' for method in method_ratios.index]
    bars = axes[0, 0].bar(range(len(method_ratios)), method_ratios.values, color=colors, alpha=0.7)
    axes[0, 0].set_title('Average Compression Ratios by Method')
    axes[0, 0].set_ylabel('Compression Ratio')
    axes[0, 0].set_xticks(range(len(method_ratios)))
    axes[0, 0].set_xticklabels(method_ratios.index, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, method_ratios.values):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{value:.1f}x', ha='center', va='bottom')
    
    # 2. Box plot comparison
    traditional_data = comparison_df[comparison_df['type'] == 'traditional']['compression_ratio']
    semantic_data = comparison_df[comparison_df['type'] == 'semantic']['compression_ratio']
    
    axes[0, 1].boxplot([traditional_data, semantic_data], 
                      tick_labels=['Traditional', 'Semantic Tokens'])
    axes[0, 1].set_title('Compression Ratio Distribution')
    axes[0, 1].set_ylabel('Compression Ratio')
    
    # 3. Per-video comparison
    videos = comparison_df['video'].unique()
    x = np.arange(len(videos))
    
    semantic_ratios = []
    traditional_best = []
    
    for video in videos:
        video_data = comparison_df[comparison_df['video'] == video]
        semantic_ratio = video_data[video_data['type'] == 'semantic']['compression_ratio'].iloc[0] if len(video_data[video_data['type'] == 'semantic']) > 0 else 0
        trad_ratio = video_data[video_data['type'] == 'traditional']['compression_ratio'].max() if len(video_data[video_data['type'] == 'traditional']) > 0 else 0
        
        semantic_ratios.append(semantic_ratio)
        traditional_best.append(trad_ratio)
    
    width = 0.35
    axes[1, 0].bar(x - width/2, traditional_best, width, label='Best Traditional', alpha=0.7)
    axes[1, 0].bar(x + width/2, semantic_ratios, width, label='Semantic Tokens', alpha=0.7, color='red')
    axes[1, 0].set_title('Per-Video Compression Comparison')
    axes[1, 0].set_ylabel('Compression Ratio')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(videos, rotation=45, ha='right')
    axes[1, 0].legend()
    
    # 4. Efficiency analysis
    semantic_avg = comparison_df[comparison_df['type'] == 'semantic']['compression_ratio'].mean()
    traditional_avg = comparison_df[comparison_df['type'] == 'traditional']['compression_ratio'].mean()
    
    categories = ['Average Compression Ratio']
    semantic_vals = [semantic_avg]
    traditional_vals = [traditional_avg]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, traditional_vals, width, label='Traditional Average', alpha=0.7)
    axes[1, 1].bar(x + width/2, semantic_vals, width, label='Semantic Tokens', alpha=0.7, color='red')
    axes[1, 1].set_title('Overall Performance Comparison')
    axes[1, 1].set_ylabel('Compression Ratio')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(categories)
    axes[1, 1].legend()
    
    # Add performance improvement text
    improvement = (semantic_avg / traditional_avg - 1) * 100
    axes[1, 1].text(0, max(semantic_avg, traditional_avg) * 1.1,
                   f'Semantic tokens: {improvement:+.1f}% vs traditional',
                   ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/compression_benchmark.png", dpi=150, bbox_inches='tight')
    plt.show()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark semantic compression against SOTA methods')
    parser.add_argument('--download_datasets', action='store_true', help='Download standard test datasets')
    parser.add_argument('--dataset_dir', default='benchmark_videos', help='Directory for test videos')
    parser.add_argument('--results_dir', default='benchmark_results', help='Directory for results')
    parser.add_argument('--max_videos', type=int, default=5, help='Max videos to process')
    parser.add_argument('--skip_traditional', action='store_true', help='Skip traditional compression benchmarks')
    parser.add_argument('--cache_dir', default='./cache', help='Model cache directory')
    
    args = parser.parse_args()
    
    # Setup
    device = setup_device()
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Download datasets if requested
    if args.download_datasets:
        print("Downloading standard test datasets...")
        video_files = download_test_videos(args.dataset_dir)
        video_files = convert_yuv_to_mp4(video_files, args.dataset_dir)
    else:
        # Use existing videos
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files.extend(list(Path(args.dataset_dir).glob(ext)))
        video_files = [str(f) for f in video_files[:args.max_videos]]
    
    if not video_files:
        print("No video files found. Use --download_datasets to download test videos.")
        return
    
    print(f"Found {len(video_files)} video files for benchmarking")
    
    # Load semantic model
    print("Loading semantic tokenizer model...")
    model = load_tokenizer_model(device, args.cache_dir)
    
    # Benchmark traditional compression methods
    traditional_results = []
    if not args.skip_traditional:
        print("Benchmarking traditional compression methods...")
        traditional_results = benchmark_traditional_compression(video_files, args.results_dir)
    
    # Compare with semantic compression
    comparison_df = compare_with_semantic_compression(video_files, model, device, traditional_results)
    
    # Create visualizations
    plot_compression_comparison(comparison_df, args.results_dir)
    
    # Save results
    comparison_df.to_csv(f"{args.results_dir}/compression_benchmark.csv", index=False)
    
    # Print summary
    print(f"\n=== Benchmark Results Summary ===")
    semantic_avg = comparison_df[comparison_df['type'] == 'semantic']['compression_ratio'].mean()
    if len(comparison_df[comparison_df['type'] == 'traditional']) > 0:
        traditional_avg = comparison_df[comparison_df['type'] == 'traditional']['compression_ratio'].mean()
        improvement = (semantic_avg / traditional_avg - 1) * 100
        print(f"Semantic tokens average: {semantic_avg:.1f}x compression")
        print(f"Traditional average: {traditional_avg:.1f}x compression")
        print(f"Improvement: {improvement:+.1f}%")
    else:
        print(f"Semantic tokens average: {semantic_avg:.1f}x compression")
    
    print(f"\nDetailed results saved to: {args.results_dir}")

if __name__ == '__main__':
    main()
