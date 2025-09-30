# Getting Started - Slot-CoCa with Imagenette

## ✅ You're Ready to Train!

Good news: **You already have everything you need!** No need to download full ImageNet.

## 🎯 What You Have

- ✅ **Imagenette dataset**: `datasets/imagenette2/` (13,394 train + 3,925 val images)
- ✅ **Slot-CoCa model**: Fully implemented with 3 objectives
- ✅ **Training pipeline**: DDP-ready (works with 1 or 2 GPUs)
- ✅ **Evaluation scripts**: Compare against CLIP baselines
- ✅ **CLIP installed**: For baseline comparisons

## 🚀 Quick Start (3 Commands)

### 1. Test Dataset Loading
```bash
cd multimodal_training
python imagenette_captions_dataset.py
```

**Expected output:**
```
✅ Loaded 66,970 image-caption pairs for train
✅ Loaded 19,625 image-caption pairs for val
```

### 2. Verify Setup
```bash
python test_setup.py
```

**Expected: All ✅ checks pass**

### 3. Start Training
```bash
python train_multimodal.py \
    --data_dir ../datasets/imagenette2 \
    --batch_size 32 \
    --world_size 1 \
    --epochs 30 \
    --log_dir ../logs/slot_coca_imagenette \
    --checkpoint_dir ../checkpoints/slot_coca_imagenette
```

## 📊 Training Configuration

### Your Hardware
- **GPU**: NVIDIA GeForce RTX 2060 (5.8 GB VRAM)
- **Recommendation**: Batch size 32 (fits comfortably)
- **Training time**: ~10 hours for 30 epochs

### Dataset
- **Images**: 13,394 training, 3,925 validation
- **Captions**: 5 per image (synthetic from class names)
- **Total pairs**: 66,970 training samples
- **Classes**: 10 (tench, springer, cassette_player, chain_saw, church, french_horn, garbage_truck, gas_pump, golf_ball, parachute)

### Model
- **Vision**: DINOv3 ViT-Base → SlotFormer (128 slots)
- **Text**: DistilBERT-base
- **Projection**: 256-dim shared space
- **Parameters**: ~180M total

## 📈 Monitor Training

### TensorBoard
```bash
# In a separate terminal
tensorboard --logdir ../logs/slot_coca_imagenette
```

Open: http://localhost:6006

**Metrics to watch:**
- `train/reconstruction_loss` - Should decrease steadily
- `train/contrastive_loss` - Should decrease and stabilize
- `train/caption_loss` - Should decrease
- `epoch/val_loss` - Should decrease (best model saved when this is lowest)

### Expected Loss Values

| Epoch | Recon Loss | Contrast Loss | Caption Loss |
|-------|------------|---------------|--------------|
| 1 | ~1.2 | ~0.7 | ~9.5 |
| 10 | ~0.3 | ~0.3 | ~5.0 |
| 30 | ~0.1 | ~0.2 | ~3.0 |

## 🎯 After Training (Evaluation)

### 1. Zero-Shot Classification on Imagenette
```bash
python evaluation/eval_zero_shot.py \
    --model_path ../checkpoints/slot_coca_imagenette/best_model.pt \
    --data_dir ../datasets/imagenette2/val \
    --batch_size 128
```

**Expected Results:**
- Imagenette Top-1: 40-60% (vs CLIP ViT-B/32: ~85%)
- Imagenette Top-5: 80-95%

### 2. Image-Text Retrieval
```bash
python evaluation/eval_retrieval.py \
    --model_path ../checkpoints/slot_coca_imagenette/best_model.pt \
    --data_dir ../datasets/imagenette2 \
    --num_samples 3925
```

**Expected Results:**
- Image→Text R@1: 35-50%
- Text→Image R@1: 30-45%
- Image→Text R@5: 70-85%

### 3. Image Captioning
```bash
python evaluation/eval_captioning.py \
    --model_path ../checkpoints/slot_coca_imagenette/best_model.pt \
    --data_dir ../datasets/imagenette2 \
    --num_samples 1000
```

**Expected Results:**
- BLEU-1: 30-45%
- Shows generated captions vs ground truth

## 💡 Optimization Tips

### If Training is Too Slow
```bash
# Reduce batch size
python train_multimodal.py --batch_size 16 ...

# Reduce slots
python train_multimodal.py --num_slots 64 ...

# Fewer captions per image (in imagenette_captions_dataset.py)
# Change captions_per_image=5 to captions_per_image=3
```

### If Out of Memory
```bash
# Smaller batch size
--batch_size 16

# Gradient accumulation (simulate larger batch)
--gradient_accumulation_steps 2

# Mixed precision (add to train_multimodal.py if needed)
# Use torch.cuda.amp for FP16 training
```

### If Validation is Slow
```bash
# Validate less frequently
--val_interval 5  # Validate every 5 epochs instead of every epoch
```

## 🎓 What to Expect

### Training Dynamics

**Epoch 1-5**: High losses, model learning basic alignment
- Reconstruction loss drops quickly
- Contrastive loss oscillates
- Caption loss decreases slowly

**Epoch 10-20**: Steady improvement
- All losses decreasing
- Validation loss tracking training loss
- Captions start making sense

**Epoch 20-30**: Refinement
- Small improvements
- Risk of overfitting (watch val loss)
- Best model likely in epoch 20-28

### Signs of Success ✅

- **Reconstruction loss < 0.2** by epoch 20
- **Contrastive loss < 0.3** by epoch 20
- **Caption loss < 4.0** by epoch 20
- **Val loss tracking train loss** (gap < 20%)
- **Generated captions mention correct objects**

### Red Flags 🚩

- **Loss exploding**: Reduce learning rate (--lr 1e-5)
- **Val loss >> train loss**: Overfitting, add dropout or reduce epochs
- **NaN losses**: Gradient explosion, clip gradients harder
- **No improvement after 10 epochs**: Check data loading

## 📝 Checkpoints

Saved in `../checkpoints/slot_coca_imagenette/`:

- `checkpoint_epoch_5.pt` - Every 5 epochs
- `checkpoint_epoch_10.pt`
- `checkpoint_epoch_15.pt`
- ...
- `best_model.pt` - Best validation loss (use this for evaluation!)

Each checkpoint contains:
- Model weights
- Optimizer state
- Scheduler state
- Training epoch
- All hyperparameters

## 🔄 Resume Training

If training is interrupted:

```bash
python train_multimodal.py \
    --resume_from ../checkpoints/slot_coca_imagenette/checkpoint_epoch_15.pt \
    --data_dir ../datasets/imagenette2 \
    ...
```

## 📊 Compare to Memory Baseline

Your previous experiments showed:
- **Cross-modal alignment: 0.722**

After training, check if Slot-CoCa matches or exceeds this!

## 🎉 Success Criteria

Your proof of concept is successful if:

1. ✅ Training completes without errors
2. ✅ Losses decrease over epochs
3. ✅ Zero-shot accuracy > 40% on Imagenette
4. ✅ Image-text retrieval R@1 > 30%
5. ✅ Generated captions are coherent
6. ✅ Slot visualizations show object decomposition

If all pass → Scale to larger datasets!

## 🚀 Next Steps After PoC

1. **Analyze results**: Which objective helps most?
2. **Ablation studies**: Turn off reconstruction/contrast/caption
3. **Visualize slots**: Do they correspond to objects?
4. **Scale up**: Try Conceptual Captions (3M images)
5. **Architecture improvements**: Larger models, better encoders

## 📚 Files Reference

- `imagenette_captions_dataset.py` - Dataset loader
- `train_multimodal.py` - Training script
- `models/slot_coca.py` - Model architecture
- `evaluation/eval_*.py` - Evaluation scripts
- `PROOF_OF_CONCEPT_SETUP.md` - Detailed guide

## ❓ Troubleshooting

### "No module named 'imagenette_captions_dataset'"
```bash
# Make sure you're in the right directory
cd multimodal_training
python train_multimodal.py ...
```

### "CUDA out of memory"
```bash
# Reduce batch size
--batch_size 16
# Or even smaller
--batch_size 8
```

### "Dataset not found"
```bash
# Check path
ls -la ../datasets/imagenette2/
# Should see: train/ and val/ directories
```

### Training very slow
```bash
# Check GPU usage
nvidia-smi
# Should see python process using GPU

# Reduce workers if I/O bottleneck
--num_workers 2
```

---

## 🎯 Ready? Let's Go!

```bash
cd multimodal_training
python train_multimodal.py --data_dir ../datasets/imagenette2 --batch_size 32 --world_size 1 --epochs 30
```

**Training time**: ~10 hours  
**Results**: Tomorrow morning! ☕

Good luck! 🚀
