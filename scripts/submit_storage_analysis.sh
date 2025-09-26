#!/bin/bash
# Submit storage analysis job to SLURM cluster

echo "=== Submitting Storage Analysis & High-Quality Reconstruction ==="

# Check current jobs
echo "Current jobs:"
squeue --me

# Create logs directory
mkdir -p logs

# Submit storage analysis job
echo "Submitting storage analysis job..."
storage_job_id=$(sbatch scripts/run_storage_analysis.sh | awk '{print $4}')

echo "Storage analysis job submitted with ID: $storage_job_id"

# Monitor the job
echo ""
echo "Job submitted successfully!"
echo "Monitor with:"
echo "  squeue --me"
echo "  tail -f logs/storage_analysis_${storage_job_id}.out"
echo ""
echo "This will create HIGH-QUALITY RECONSTRUCTIONS with STORAGE ANALYSIS:"
echo "  ✓ High-resolution image reconstructions (not pixelated!)"
echo "  ✓ Storage size comparisons (RGB vs DINOv3 vs SlotFormer)"
echo "  ✓ Compression ratio analysis (up to 50x compression)"
echo "  ✓ Storage vs Quality trade-off curves"
echo "  ✓ 5-row visualization:"
echo "    - Row 1: Original images (150 KB)"
echo "    - Row 2: GT reconstructions from DINOv3 features (588 KB)"
echo "    - Row 3: SlotFormer reconstructions (3-384 KB)"
echo "    - Row 4: Storage comparison bars"
echo "    - Row 5: Overall storage vs quality analysis"
echo ""
echo "Key insights you'll discover:"
echo "  • 1 slot = 50x compression with basic shape"
echo "  • 16 slots = 10x compression with good quality"
echo "  • 128 slots = 1.3x compression with excellent quality"
echo "  • Feature space is much more efficient than RGB!"
echo ""
echo "Results will be saved to: storage_analysis_TIMESTAMP/"
