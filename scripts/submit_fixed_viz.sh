#!/bin/bash
# Submit fixed visualization job to SLURM cluster

echo "=== Submitting Fixed SlotFormer Visualization ==="

# Check current jobs
echo "Current jobs:"
squeue --me

# Create logs directory
mkdir -p logs

# Submit visualization job
echo "Submitting fixed visualization job..."
viz_job_id=$(sbatch scripts/run_fixed_visualization.sh | awk '{print $4}')

echo "Fixed visualization job submitted with ID: $viz_job_id"

# Monitor the job
echo ""
echo "Job submitted successfully!"
echo "Monitor with:"
echo "  squeue --me"
echo "  tail -f logs/fixed_visualization_${viz_job_id}.out"
echo ""
echo "This will create PROPER visualizations showing:"
echo "  ✓ Feature space reconstructions (768D -> PCA -> RGB)"
echo "  ✓ Cosine similarity maps between GT and reconstructed features"
echo "  ✓ Error maps showing reconstruction quality"
echo "  ✓ Progressive improvement with more slots"
echo "  ✓ Comprehensive analysis across multiple images"
echo ""
echo "Results will be saved to: fixed_visualizations_TIMESTAMP/"
