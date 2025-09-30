# Proof of Concept Setup - Using Imagenette (No Full ImageNet Download!)

## 🎯 Smart Strategy: Imagenette + Captions

Instead of downloading full ImageNet (150GB), we use:
- ✅ **Imagenette** (already have): 13,394 images from 10 classes (~1.5GB)
- ✅ **Synthetic captions**: Generated from class names
- ✅ **Optional**: Download ImageNet-Captions annotations (~100MB) for real captions

This gives you **~67K image-caption pairs** for training - perfect for proof of concept!

## 📊 Dataset Stats

| Split | Images | Captions/Image | Total Pairs |
|-------|--------|----------------|-------------|
| Train | 13,394 | 5 | **66,970** |
| Val | 3,925 | 5 | **19,625** |

## 🚀 Quick Start (3 Options)

### Option 1: Use Synthetic Captions (Fastest - No Download!)

This works immediately with your existing Imagenette:

```bash
cd multimodal_training

# Test the dataset
python imagenette_captions_dataset.py

# Train immediately
python train_multimodal.py \
    --data_dir ../datasets/imagenette2 \
    --batch_size 128 \
    --world_size 1 \
    --epochs 30
```

**Captions generated**: "a photo of a tench", "an image of a golf ball", etc.

### Option 2: Download ImageNet-Captions Annotations (Recommended)

Get real human-written captions for Imagenette classes:

```bash
# 1. Download annotations (small files ~100MB)
git clone https://github.com/mlfoundations/imagenet-captions.git /tmp/imagenet-captions

# 2. Extract captions for Imagenette classes only
python extract_imagenette_captions.py

# 3. Train with real captions
python train_multimodal.py \
    --data_dir ../datasets/imagenette2 \
    --captions_file ./data/imagenette_captions.json \
    --batch_size 128 \
    --world_size 1 \
    --epochs 30
```

### Option 3: Small ImageNet Subset (If You Want More Data)

Download only the 10 Imagenette classes from full ImageNet:

```bash
# This downloads ~3GB (10 classes only, not full 150GB)
# Instructions: https://github.com/fastai/imagenette#imagewang
```

## 🔧 Modified Training Command

Since you have 1 GPU (RTX 2060, 5.8GB), adjust the training:

```bash
python train_multimodal.py \
    --data_dir ../datasets/imagenette2 \
    --batch_size 32 \
    --world_size 1 \
    --epochs 30 \
    --lr 5e-5 \
    --num_slots 128 \
    --lambda_recon 1.0 \
    --lambda_contrast 1.0 \
    --lambda_caption 1.0 \
    --log_dir ../logs/slot_coca_imagenette \
    --checkpoint_dir ../checkpoints/slot_coca_imagenette
```

**Why batch_size 32?**
- Your RTX 2060 has 5.8GB VRAM
- SlotFormer (128 slots) + text encoder needs ~4-5GB
- Batch size 32 should fit comfortably

## 📈 Expected Results (Imagenette PoC)

Since Imagenette is only 10 classes (vs 1000 for full ImageNet):

### Zero-Shot Classification

| Metric | Expected | Notes |
|--------|----------|-------|
| **Imagenette Top-1** | 40-60% | Much higher than full ImageNet (easier task) |
| **Imagenette Top-5** | 80-95% | Should be very high (only 10 classes) |

### Image-Text Retrieval (Imagenette val set)

| Metric | Expected | Notes |
|--------|----------|-------|
| **Image→Text R@1** | 35-50% | |
| **Text→Image R@1** | 30-45% | |
| **Image→Text R@5** | 70-85% | |

### Captioning

| Metric | Expected | Notes |
|--------|----------|-------|
| **BLEU-1** | 30-45% | Simpler vocabulary (10 classes) |
| **CIDEr** | 50-80 | Good baseline |

## 💡 Advantages of This Approach

✅ **No large downloads**: Use existing Imagenette  
✅ **Fast iteration**: Smaller dataset = faster training  
✅ **Valid proof of concept**: Tests architecture fully  
✅ **Easy baseline**: CLIP also works on Imagenette  
✅ **Memory efficient**: Fits on single RTX 2060  

## ⚡ Training Time Estimates

With RTX 2060 (5.8GB VRAM):

| Configuration | Time/Epoch | Total Time (30 epochs) |
|---------------|------------|------------------------|
| Batch 32, all objectives | ~20 min | **~10 hours** |
| Batch 16, all objectives | ~35 min | **~17 hours** |

**Overnight training**: Start before bed, results by morning! ☕

## 🔄 Workflow

### Day 1: Setup & Validation (1-2 hours)
```bash
# 1. Test dataset loading
python imagenette_captions_dataset.py

# 2. Run setup verification
python test_setup.py

# 3. Verify training starts
python train_multimodal.py --epochs 1 --batch_size 16
```

### Day 2: Full Training (~10-17 hours)
```bash
# Start training
python train_multimodal.py \
    --batch_size 32 \
    --epochs 30

# Monitor
tensorboard --logdir ../logs/slot_coca_imagenette
```

### Day 3: Evaluation (1-2 hours)
```bash
# Zero-shot on Imagenette
python evaluation/eval_zero_shot.py \
    --model_path ../checkpoints/slot_coca_imagenette/best_model.pt \
    --data_dir ../datasets/imagenette2/val

# Retrieval
python evaluation/eval_retrieval.py \
    --model_path ../checkpoints/slot_coca_imagenette/best_model.pt \
    --data_dir ../datasets/imagenette2

# Captioning
python evaluation/eval_captioning.py \
    --model_path ../checkpoints/slot_coca_imagenette/best_model.pt \
    --data_dir ../datasets/imagenette2
```

## 🎓 What You'll Learn

1. **Does multimodal training improve slot quality?**
   - Compare reconstruction before/after
   
2. **How much vision-language alignment can we get?**
   - Measure against previous 0.722 benchmark from memory
   
3. **Do slots help or hurt captioning?**
   - Compare to dense baseline
   
4. **Is the architecture working?**
   - Validate all three objectives train properly

## 📊 Comparison to Full ImageNet

| Aspect | Imagenette PoC | Full ImageNet |
|--------|----------------|---------------|
| Images | 13K train | 1.28M train |
| Classes | 10 | 1000 |
| Download | ✅ You have it | ❌ 150GB |
| Train time | ~10 hours | ~7 days |
| VRAM needed | 5.8GB (fits RTX 2060) | 40GB+ (needs A100) |
| Valid PoC | ✅ Yes | - |

## 🚀 After Proof of Concept

If Imagenette results look good:

### Scale Up Options
1. **Use full Imagenette**: Already doing this ✅
2. **Add COCO Captions**: 120K images with detailed captions
3. **Use Conceptual Captions 3M**: Free, 3M images
4. **Scale to full ImageNet**: Only if results justify it

### Architecture Improvements
1. Larger projection dim (256 → 512)
2. More slots (128 → 256)
3. Better text encoder (DistilBERT → BERT/T5)
4. Mixed precision (FP16) for faster training

## ✅ Ready to Start!

You have everything needed:
- ✅ Imagenette dataset at `datasets/imagenette2/`
- ✅ Slot-CoCa model implemented
- ✅ Training script with DDP support
- ✅ Evaluation against CLIP baselines
- ✅ CLIP installed for comparisons

**Just run**:
```bash
cd multimodal_training
python imagenette_captions_dataset.py  # Test dataset
python train_multimodal.py --batch_size 32 --world_size 1  # Start training!
```

No need to download anything else! 🎉
