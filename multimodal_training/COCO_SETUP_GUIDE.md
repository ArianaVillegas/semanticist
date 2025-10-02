# Complete Guide: Training Slot-CoCa on COCO Captions

## 📋 Overview

This guide will help you:
1. Download COCO dataset (123K images, 615K captions)
2. Set up training with high-quality captions
3. Train Slot-CoCa from scratch
4. Evaluate results
5. Compare with Imagenette baseline

**Total time**: ~15-20 hours (2 hours setup + 15 hours training)

---

## Step 1: Download COCO Dataset (~30 minutes)

### 1.1 Create Directory
```bash
cd /home/avillegas/semanticist/datasets
mkdir -p coco
cd coco
```

### 1.2 Download Images (13GB)
```bash
# Training images
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
```

### 1.3 Download Annotations (241MB)
```bash
# Annotations (includes captions)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

### 1.4 Download Validation Set (Optional, for evaluation)
```bash
# Validation images (1GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
```

### 1.5 Verify Structure
```bash
tree -L 2 coco/
# Should see:
# coco/
# ├── train2017/         (118,287 images)
# ├── val2017/           (5,000 images)
# └── annotations/
#     ├── captions_train2017.json
#     └── captions_val2017.json
```

---

## Step 2: Install Dependencies

```bash
cd /home/avillegas/semanticist
conda activate semanticist

# Install pycocotools (required for COCO API)
pip install pycocotools

# Verify installation
python -c "from pycocotools.coco import COCO; print('✅ COCO API ready')"
```

---

## Step 3: Test COCO Dataset

```bash
cd multimodal_training

# Test dataset loading
python coco_dataset.py

# Expected output:
# ✅ Loaded 591,753 image-caption pairs
#    - Unique images: 118,287
#    - Captions per image: 5
#
# Sample captions show real descriptions like:
#   "A person riding a skateboard down a street"
#   "A large commercial airplane flying through the sky"
```

---

## Step 4: Update Training Script

The training script needs to support COCO. Updates made:
- `--use_coco` flag to switch datasets
- COCO dataset integration
- Adjusted batch size for larger dataset
- Learning rate scheduling for longer training

**All changes in**: `train_multimodal.py`

---

## Step 5: Train Slot-CoCa on COCO

### 5.1 Quick Test (5 minutes)
```bash
cd /home/avillegas/semanticist/multimodal_training

# Test with tiny dataset (100 samples, 2 epochs)
python train_multimodal.py \
    --use_coco \
    --data_dir /home/avillegas/semanticist/datasets/coco \
    --num_samples 100 \
    --batch_size 16 \
    --epochs 2 \
    --device cpu \
    --checkpoint_dir checkpoints/coco_test

# Should complete in ~5 minutes
# Verify losses decrease
```

### 5.2 Full Training (~15 hours on CPU, ~4 hours on GPU)
```bash
# Full COCO training
python train_multimodal.py \
    --use_coco \
    --data_dir /home/avillegas/semanticist/datasets/coco \
    --batch_size 128 \
    --epochs 30 \
    --num_slots 128 \
    --lambda_recon 1.0 \
    --lambda_contrast 1.0 \
    --lambda_caption 1.0 \
    --learning_rate 1e-4 \
    --device cuda \
    --checkpoint_dir checkpoints/slot_coca_coco \
    --save_every 5

# Training will save checkpoints every 5 epochs
# Monitor with: tail -f logs/slot_coca_coco/training.log
```

### 5.3 Resume Training (if interrupted)
```bash
python train_multimodal.py \
    --use_coco \
    --data_dir /home/avillegas/semanticist/datasets/coco \
    --resume_from checkpoints/slot_coca_coco/checkpoint_epoch_15.pt \
    --device cuda
```

---

## Step 6: Monitor Training

### 6.1 Watch Logs
```bash
# In another terminal
tail -f logs/slot_coca_coco/training.log

# Look for:
# - recon_loss: Should decrease from ~0.8 → ~0.3
# - contrast_loss: Should decrease from ~6.0 → ~2.0
# - caption_loss: Should decrease from ~9.0 → ~2.5
```

### 6.2 Check GPU Usage (if using GPU)
```bash
watch -n 1 nvidia-smi
```

### 6.3 Expected Training Time
- **CPU**: ~15-18 hours (591K samples, 30 epochs)
- **GPU (single)**: ~4-6 hours
- **GPU (multi)**: ~2-3 hours

---

## Step 7: Evaluate Results

### 7.1 Zero-Shot Classification (on COCO)
```bash
cd evaluation

python eval_zero_shot_coco.py \
    --model_path ../checkpoints/slot_coca_coco/best_model.pt \
    --data_dir /home/avillegas/semanticist/datasets/coco \
    --device cpu

# Expected results:
# - Much better than Imagenette (more diverse training)
# - Top-1: 60-75% (COCO has 80 classes, harder than Imagenette's 10)
```

### 7.2 Image-Text Retrieval
```bash
python eval_retrieval.py \
    --model_path ../checkpoints/slot_coca_coco/best_model.pt \
    --data_dir /home/avillegas/semanticist/datasets/coco \
    --use_coco \
    --num_samples 5000 \
    --device cpu

# Expected results:
# - I→T Recall@1: 25-40% (MUCH better than 0.20% on Imagenette)
# - T→I Recall@1: 20-35%
# - I→T Recall@5: 50-70%
```

### 7.3 Image Captioning
```bash
python eval_captioning.py \
    --model_path ../checkpoints/slot_coca_coco/best_model.pt \
    --data_dir /home/avillegas/semanticist/datasets/coco \
    --use_coco \
    --num_samples 1000 \
    --device cpu

# Expected results:
# - BLEU-1: 30-45% (vs 0% on Imagenette)
# - Real captions: "A person riding a bike"
# - No more empty strings!
```

### 7.4 Test Caption Generation
```bash
# Quick caption test
cd ..
python debug_model.py --checkpoint checkpoints/slot_coca_coco/best_model.pt

# Should now generate real captions instead of empty strings!
```

---

## Step 8: Compare Results

### Imagenette (Defective Captions) vs COCO (High Quality)

| Metric | Imagenette | COCO (Expected) | Improvement |
|--------|-----------|-----------------|-------------|
| **Zero-Shot Top-1** | 99.67% | 60-75% | Different tasks* |
| **I→T Retrieval R@1** | 0.20% | 25-40% | **125-200x better!** |
| **Captioning BLEU-1** | 0% | 30-45% | **∞ better!** |
| **Caption Quality** | Empty | Real text | ✅ Works! |

*Imagenette has 10 classes, COCO has 80 classes

---

## Step 9: Advanced Experiments

### 9.1 Ablation Study: Loss Weights
```bash
# Try different loss weights to see what matters most

# Reconstruction-focused
python train_multimodal.py --use_coco --lambda_recon 2.0 --lambda_contrast 0.5 --lambda_caption 0.5

# Contrastive-focused
python train_multimodal.py --use_coco --lambda_recon 0.5 --lambda_contrast 2.0 --lambda_caption 0.5

# Caption-focused
python train_multimodal.py --use_coco --lambda_recon 0.5 --lambda_contrast 0.5 --lambda_caption 2.0
```

### 9.2 Slot Count Experiment
```bash
# Test different slot counts (memory says 128 is optimal)
for slots in 64 128 256; do
    python train_multimodal.py \
        --use_coco \
        --num_slots $slots \
        --checkpoint_dir checkpoints/coco_slots_${slots}
done
```

### 9.3 Compare to CLIP Baseline
```bash
# CLIP should get ~40-50% on COCO retrieval
# Your Slot-CoCa should be competitive or better
```

---

## Step 10: Visualize Results

### 10.1 Generate Sample Captions
```bash
python generate_samples.py \
    --model_path checkpoints/slot_coca_coco/best_model.pt \
    --data_dir /home/avillegas/semanticist/datasets/coco \
    --num_samples 20 \
    --output_dir results/coco_samples

# Creates visualizations with:
# - Original image
# - Ground truth caption
# - Generated caption
```

### 10.2 Attention Visualization
```bash
python visualize_attention.py \
    --model_path checkpoints/slot_coca_coco/best_model.pt \
    --image_path /path/to/image.jpg

# Shows which slots attend to which objects
```

---

## 📊 Expected Timeline

| Step | Time | Can Parallelize? |
|------|------|------------------|
| Download COCO | 30 min | No |
| Setup & Test | 10 min | No |
| Quick Test (100 samples) | 5 min | No |
| **Full Training** | **15 hours (CPU)** | Yes (use GPU) |
| Evaluation (all metrics) | 20 min | No |
| Generate samples | 10 min | No |
| **Total** | **~16-17 hours** | |

**With GPU**: ~5-6 hours total

---

## 🎯 Success Criteria

Your COCO training is successful if:

1. ✅ **Caption loss decreases**: 9.0 → 2.5
2. ✅ **Generated captions are real text**: Not empty strings
3. ✅ **Retrieval R@1 > 20%**: Much better than 0.20%
4. ✅ **BLEU-1 > 30%**: Meaningful caption quality
5. ✅ **Captions make sense**: "A person..." not "20080815_276"

---

## 🐛 Troubleshooting

### Problem: "CUDA out of memory"
**Solution**: Reduce batch size
```bash
--batch_size 64  # or 32
```

### Problem: Training too slow on CPU
**Solution**: Use fewer samples for proof-of-concept
```bash
--num_samples 50000  # ~1/12 of full dataset
--epochs 30
# Still 5x more data than Imagenette!
```

### Problem: "pycocotools not found"
**Solution**: 
```bash
pip install pycocotools
```

### Problem: Download interrupted
**Solution**: Resume with `wget -c`
```bash
wget -c http://images.cocodataset.org/zips/train2017.zip
```

---

## 📈 After Training

### Share Results
1. Zero-shot accuracy on COCO
2. Retrieval metrics (R@1, R@5, R@10)
3. Sample generated captions
4. Comparison to CLIP baseline

### Next Steps
1. Scale to Conceptual Captions 3M
2. Add object detection integration
3. Multi-modal reasoning tasks
4. Publish findings

---

**Ready to start? Run the quick start script!**

```bash
cd /home/avillegas/semanticist/multimodal_training
./coco_quick_start.sh
```
