#!/bin/bash
# Convenient job submission script for ag001 cluster

# Make scripts executable
chmod +x scripts/train_slotformer_slurm.sh
chmod +x scripts/train_slotformer_a100_optimized.sh

echo "Available SLURM scripts:"
echo "1. Basic training (1 A100, 12h): scripts/train_slotformer_slurm.sh"
echo "2. Optimized training (2 A100, 24h): scripts/train_slotformer_a100_optimized.sh"
echo ""

# Check which script to submit
if [ "$1" == "basic" ]; then
    echo "Submitting basic SlotFormer training..."
    sbatch scripts/train_slotformer_slurm.sh
elif [ "$1" == "optimized" ]; then
    echo "Submitting A100-optimized SlotFormer training..."
    sbatch scripts/train_slotformer_a100_optimized.sh
else
    echo "Usage: $0 [basic|optimized]"
    echo ""
    echo "Examples:"
    echo "  $0 basic     # Submit basic training job"
    echo "  $0 optimized # Submit optimized training job"
    echo ""
    echo "Monitor jobs with:"
    echo "  squeue -u \$USER"
    echo "  tail -f logs/slotformer_*_\$JOBID.out"
fi
