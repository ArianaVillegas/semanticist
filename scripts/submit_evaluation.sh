#!/bin/bash
# Submit evaluation job to SLURM cluster

echo "=== Submitting SlotFormer Evaluation to Cluster ==="

# Check if training is still running
echo "Checking current jobs..."
squeue --me

# Create logs directory
mkdir -p logs

# Submit evaluation job
echo "Submitting evaluation job..."
eval_job_id=$(sbatch scripts/run_experiments_cluster.sh | awk '{print $4}')

echo "Evaluation job submitted with ID: $eval_job_id"

# Monitor the job
echo "Monitoring evaluation job..."
echo "You can check status with: squeue --me"
echo "View logs with: tail -f logs/evaluation_${eval_job_id}.out"
echo ""
echo "The evaluation will:"
echo "  1. Test all baseline models"
echo "  2. Run comprehensive evaluation"
echo "  3. Evaluate any trained SlotFormer models found"
echo "  4. Generate plots and save results"
echo ""
echo "Results will be saved to: evaluation_results_TIMESTAMP/"
