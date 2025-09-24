# SlotFormer Training on ag001 Cluster

## Quick Start

1. **Submit Job**:
   ```bash
   # Basic training (1 A100, 12h)
   bash scripts/submit_job.sh basic
   
   # Optimized training (2 A100, 24h) 
   bash scripts/submit_job.sh optimized
   ```

2. **Monitor Job**:
   ```bash
   squeue -u $USER
   tail -f logs/slotformer_*_$JOBID.out
   ```

## Hardware Specifications (ag001)

- **CPU**: AMD EPYC 7742 (128 cores)
- **GPU**: 2x NVIDIA A100 40GB
- **RAM**: 1TB DDR4
- **Network**: Infiniband Mellanox

## Optimizations for A100

### Memory & Compute
- **Batch Size**: 512 (leveraging 40GB VRAM)
- **Model Size**: 256 slots, 6 transformer layers
- **Precision**: bfloat16 (A100 native)
- **Compilation**: torch.compile with max-autotune

### Data Loading
- **Workers**: 32 (matching CPU cores)
- **Persistent Workers**: Enabled
- **Pin Memory**: Enabled

### Training Configuration
- **Learning Rate**: 1e-3 (scaled for large batch)
- **Gradient Clipping**: 1.0
- **Scheduler**: Cosine annealing
- **Checkpointing**: Every 10 epochs

## File Structure

```
scripts/
├── train_slotformer_slurm.sh          # Basic SLURM script
├── train_slotformer_a100_optimized.sh # A100-optimized SLURM
└── submit_job.sh                      # Job submission helper

train_a100_optimized.py               # A100-optimized training script
logs/                                  # Job output logs
checkpoints/                           # Model checkpoints
```

## Expected Performance

- **Training Time**: ~12-24 hours for 100 epochs
- **Memory Usage**: ~30GB VRAM per A100
- **Throughput**: ~2-3 steps/second with batch size 512

## Troubleshooting

### Common Issues
1. **OOM Error**: Reduce batch size in `train_a100_optimized.py`
2. **Slow Data Loading**: Check `NUM_WORKERS` setting
3. **Job Queued**: Check cluster status with `sinfo`

### Monitoring Commands
```bash
# Check job status
squeue -u $USER

# View job details
scontrol show job $JOBID

# Cancel job
scancel $JOBID

# Check GPU usage
nvidia-smi
```
