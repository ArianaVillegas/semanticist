#!/bin/bash
# Submit image reconstruction visualization job to SLURM cluster

echo "=== Submitting Image Reconstruction Visualization ==="

# Check current jobs
echo "Current jobs:"
squeue --me

# Create logs directory
mkdir -p logs

# Submit image reconstruction job
echo "Submitting image reconstruction job..."
recon_job_id=$(sbatch scripts/run_image_reconstruction.sh | awk '{print $4}')

echo "Image reconstruction job submitted with ID: $recon_job_id"

# Monitor the job
echo ""
echo "Job submitted successfully!"
echo "Monitor with:"
echo "  squeue --me"
echo "  tail -f logs/image_reconstruction_${recon_job_id}.out"
echo ""
echo "This will create IMAGE RECONSTRUCTIONS showing:"
echo "  ✓ Original images from Imagenette dataset"
echo "  ✓ Ground truth reconstructions (upper bound from perfect DINOv3 features)"
echo "  ✓ SlotFormer reconstructions with 1, 2, 4, 8, 16, 32, 64, 128 slots"
echo "  ✓ Progressive visual improvement as slots increase"
echo "  ✓ Quantitative loss curves showing monotonic improvement"
echo "  ✓ 3-row comparison: Original → GT → SlotFormer"
echo ""
echo "Results will be saved to: image_reconstruction_TIMESTAMP/"
echo ""
echo "This visualization will show ACTUAL IMAGES reconstructed from features,"
echo "making it much more intuitive than PCA projections!"
