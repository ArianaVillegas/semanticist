#!/bin/bash
# Simple script to download standard video compression test datasets

echo "Creating benchmark_videos directory..."
mkdir -p benchmark_videos
cd benchmark_videos

echo "=== Downloading Standard Test Videos ==="

# 1. Sample Videos (small, reliable)
echo "1. Downloading sample test videos..."
wget -O sample_720p.mp4 "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4"
wget -O sample_360p.mp4 "https://sample-videos.com/zip/10/mp4/SampleVideo_640x360_1mb.mp4"

# 2. Big Buck Bunny (Creative Commons)
echo "2. Downloading Big Buck Bunny test videos..."
wget -O bigbuckbunny_320x180.mp4 "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4"
wget -O bigbuckbunny_640x360.mp4 "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_640x360.mp4"

# 3. Xiph.org test media (alternative URLs)
echo "3. Downloading Xiph.org test videos..."
wget -O akiyo.y4m "https://media.xiph.org/video/derf/y4m/akiyo_cif.y4m" 2>/dev/null || echo "Skipping akiyo (not available)"
wget -O foreman.y4m "https://media.xiph.org/video/derf/y4m/foreman_cif.y4m" 2>/dev/null || echo "Skipping foreman (not available)"

# 4. Convert Y4M to MP4 if ffmpeg is available
echo "4. Converting Y4M files to MP4..."
if command -v ffmpeg &> /dev/null; then
    for file in *.y4m; do
        if [ -f "$file" ]; then
            mp4_file="${file%.y4m}.mp4"
            echo "Converting $file to $mp4_file"
            ffmpeg -i "$file" -c:v libx264 -preset fast -crf 18 "$mp4_file" -y 2>/dev/null
        fi
    done
else
    echo "ffmpeg not found - keeping Y4M files"
fi

echo "=== Download Summary ==="
echo "Available video files:"
ls -lh *.mp4 *.y4m 2>/dev/null | head -10

echo ""
echo "Total files downloaded: $(ls *.mp4 *.y4m 2>/dev/null | wc -l)"
echo "Total size: $(du -sh . | cut -f1)"

cd ..
echo "Videos saved in: $(pwd)/benchmark_videos/"
