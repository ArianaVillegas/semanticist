#!/usr/bin/env python3
"""
Download sample videos from common datasets for compression testing.
"""

import os
import urllib.request
from pathlib import Path
import subprocess

def download_sample_videos(output_dir="video_samples"):
    """Download sample videos for testing."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample videos from different categories
    sample_urls = [
        # Short test videos (Creative Commons)
        ("https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4", "sample_720p.mp4"),
        ("https://sample-videos.com/zip/10/mp4/SampleVideo_640x360_1mb.mp4", "sample_360p.mp4"),
        
        # Alternative: Use youtube-dl for public domain videos
        # These are fallback options if the above don't work
    ]
    
    downloaded = []
    
    for url, filename in sample_urls:
        filepath = Path(output_dir) / filename
        if filepath.exists():
            print(f"Already exists: {filename}")
            downloaded.append(str(filepath))
            continue
            
        try:
            print(f"Downloading: {filename}")
            urllib.request.urlretrieve(url, filepath)
            downloaded.append(str(filepath))
            print(f"Downloaded: {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
    
    return downloaded

def create_synthetic_videos(output_dir="video_samples"):
    """Create synthetic test videos with known properties."""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        import cv2
        import numpy as np
        
        # Create different types of test videos
        test_videos = [
            ("static_scene.mp4", "static"),
            ("moving_object.mp4", "moving"),
            ("scene_change.mp4", "scene_change"),
            ("complex_motion.mp4", "complex")
        ]
        
        for filename, video_type in test_videos:
            filepath = Path(output_dir) / filename
            if filepath.exists():
                continue
                
            print(f"Creating synthetic video: {filename}")
            
            # Video properties
            width, height = 640, 480
            fps = 30
            duration = 3  # seconds
            total_frames = fps * duration
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
            
            for frame_idx in range(total_frames):
                if video_type == "static":
                    # Static scene - same image
                    frame = np.full((height, width, 3), [100, 150, 200], dtype=np.uint8)
                    cv2.circle(frame, (width//2, height//2), 50, (255, 255, 255), -1)
                    
                elif video_type == "moving":
                    # Moving object - circle moves across screen
                    frame = np.full((height, width, 3), [50, 50, 50], dtype=np.uint8)
                    x = int((frame_idx / total_frames) * width)
                    cv2.circle(frame, (x, height//2), 30, (0, 255, 0), -1)
                    
                elif video_type == "scene_change":
                    # Scene changes every second
                    scene = (frame_idx // fps) % 3
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
                    frame = np.full((height, width, 3), colors[scene], dtype=np.uint8)
                    
                elif video_type == "complex":
                    # Complex motion - multiple moving objects
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    for i in range(5):
                        x = int((frame_idx * (i+1) / total_frames) * width) % width
                        y = int((frame_idx * (i+2) / total_frames) * height) % height
                        cv2.circle(frame, (x, y), 20, (255-i*50, i*50, 128), -1)
                
                out.write(frame)
            
            out.release()
            print(f"Created: {filename}")
            
    except ImportError:
        print("OpenCV not available for synthetic video creation")
        return []
    
    return [str(Path(output_dir) / f[0]) for f in test_videos]

def main():
    print("Setting up video datasets for compression testing...")
    
    # Download sample videos
    downloaded = download_sample_videos()
    
    # Create synthetic videos
    synthetic = create_synthetic_videos()
    
    all_videos = downloaded + synthetic
    
    if all_videos:
        print(f"\nAvailable test videos:")
        for video in all_videos:
            if os.path.exists(video):
                size_mb = os.path.getsize(video) / (1024*1024)
                print(f"  {video} ({size_mb:.1f} MB)")
        
        print(f"\nRun compression experiments with:")
        print(f"python video_compression_experiments.py --dataset_path video_samples --max_videos 5")
    else:
        print("No test videos available. Using existing video files.")

if __name__ == '__main__':
    main()
