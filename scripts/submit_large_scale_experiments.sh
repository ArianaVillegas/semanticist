#!/bin/bash
# Submit comprehensive large-scale validation experiments

echo "=== Submitting Large-Scale SlotFormer Validation ==="

# FIRST: Prepare dependencies (on login node with internet)
echo "🔧 Preparing dependencies (requires internet access)..."
bash scripts/prepare_dependencies.sh

if [ $? -ne 0 ]; then
    echo "❌ Dependency preparation failed!"
    echo "Make sure you're on the login node with internet access."
    exit 1
fi

echo "✅ Dependencies prepared!"

# Check current jobs
echo "Current jobs:"
squeue --me

# Create logs directory
mkdir -p logs
mkdir -p experiments

# Submit large-scale experiments job
echo "Submitting large-scale experiments job..."
experiments_job_id=$(sbatch scripts/run_large_scale_experiments.sh | awk '{print $4}')

echo "Large-scale experiments job submitted with ID: $experiments_job_id"

# Monitor the job
echo ""
echo "🚀 COMPREHENSIVE VALIDATION LAUNCHED!"
echo ""
echo "Monitor with:"
echo "  squeue --me"
echo "  tail -f logs/large_scale_experiments_${experiments_job_id}.out"
echo ""
echo "This will run 3 MAJOR EXPERIMENTS:"
echo ""
echo "📊 1. SCALABILITY STUDY"
echo "  ✓ Dataset scaling: 100 → 10,000 samples"
echo "  ✓ Slot scaling: 16 → 512 slots"
echo "  ✓ Computational efficiency analysis"
echo "  ✓ Memory usage profiling"
echo "  ✓ Quality vs scale trade-offs"
echo ""
echo "🧪 2. CAUSAL VALIDATION"
echo "  ✓ SlotFormer vs 4 baseline methods:"
echo "    - Random slot ordering"
echo "    - No causal masking"
echo "    - Fixed slot ordering"
echo "    - Reverse causal ordering"
echo "  ✓ Statistical significance testing"
echo "  ✓ Monotonic improvement analysis"
echo "  ✓ Consistency evaluation"
echo ""
echo "🌐 3. MULTI-MODAL EXTENSION"
echo "  ✓ Text causal learning"
echo "  ✓ Vision-text cross-modal alignment"
echo "  ✓ Modality-agnostic slot learning"
echo "  ✓ Cross-modal consistency"
echo ""
echo "📈 EXPECTED OUTCOMES:"
echo "  • Proof of scalability to real datasets"
echo "  • Definitive evidence of causal learning superiority"
echo "  • Demonstration of multi-modal generalization"
echo "  • Publication-ready statistical validation"
echo ""
echo "⏱️  Estimated runtime: 8-12 hours"
echo "💾 Expected results size: ~2GB"
echo "🎯 Success criteria: Monotonic scores > 0.8 across all tests"
echo ""
echo "Results will be saved to: large_scale_results_TIMESTAMP/"
echo ""
echo "This is the DEFINITIVE VALIDATION of your SlotFormer breakthrough! 🎉"
