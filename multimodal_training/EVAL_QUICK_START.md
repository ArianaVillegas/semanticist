# Quick Evaluation Guide

## ⚡ Run All Evaluations at Once

```bash
cd multimodal_training
./RUN_EVAL.sh
```

This will run all three evaluations sequentially (~15-20 minutes total).

---

## 🎯 Individual Evaluations

### 1. Zero-Shot Classification (Top-1 & Top-5 Accuracy)

```bash
cd multimodal_training/evaluation

python eval_zero_shot.py \
    --model_path /home/avillegas/semanticist/multimodal_training/checkpoints/slot_coca_real_captions/best_model.pt \
    --data_dir /home/avillegas/semanticist/datasets/imagenette2 \
    --use_imagenette \
    --device cpu
```

**Expected**: Top-1: 40-65%, Top-5: 80-95%

---

### 2. Image-Text Retrieval (Recall@K)

```bash
python eval_retrieval.py \
    --model_path /home/avillegas/semanticist/multimodal_training/checkpoints/slot_coca_real_captions/best_model.pt \
    --data_dir /home/avillegas/semanticist/datasets/imagenette2 \
    --use_imagenette \
    --device cpu \
    --num_samples 3925
```

**Expected**: I→T R@1: 30-50%, T→I R@1: 25-45%

---

### 3. Image Captioning (BLEU Score)

```bash
python eval_captioning.py \
    --model_path /home/avillegas/semanticist/multimodal_training/checkpoints/slot_coca_real_captions/best_model.pt \
    --data_dir /home/avillegas/semanticist/datasets/imagenette2 \
    --use_imagenette \
    --device cpu \
    --num_samples 1000
```

**Expected**: BLEU-1: 30-45%

---

## 📝 Key Flags

- `--use_imagenette` - **REQUIRED** for Imagenette dataset
- `--device cpu` - Use CPU (change to `cuda` for GPU)
- `--num_samples N` - Limit evaluation to N samples (faster testing)
- `--batch_size B` - Adjust batch size for memory constraints

---

## ⚠️ Common Issues

**Problem**: "No dataset found"  
**Solution**: Make sure you're using `--use_imagenette` flag

**Problem**: "CUDA out of memory"  
**Solution**: Use `--device cpu` or reduce `--batch_size`

**Problem**: Import errors  
**Solution**: Make sure you're in the `semanticist` conda environment

---

## 📊 Interpreting Results

### Good Results ✅
- Zero-shot Top-1 > 40%
- I→T Retrieval R@1 > 30%
- Generated captions are coherent

### Compare to Baselines
- CLIP ViT-B/32 on Imagenette: ~85% Top-1
- Your gap is expected (47K training pairs vs CLIP's 400M)
- Previous cross-modal score from memory: 0.722

---

## 🚀 Quick Test (Fast)

Test all scripts quickly with small samples:

```bash
cd evaluation

# Quick zero-shot (100 samples)
python eval_zero_shot.py --model_path ../checkpoints/slot_coca_real_captions/best_model.pt \
    --data_dir ../../datasets/imagenette2 --use_imagenette --num_samples 100 --device cpu

# Quick retrieval (100 samples)
python eval_retrieval.py --model_path ../checkpoints/slot_coca_real_captions/best_model.pt \
    --data_dir ../../datasets/imagenette2 --use_imagenette --num_samples 100 --device cpu

# Quick captioning (10 samples)
python eval_captioning.py --model_path ../checkpoints/slot_coca_real_captions/best_model.pt \
    --data_dir ../../datasets/imagenette2 --use_imagenette --num_samples 10 --device cpu
```

Total time: ~2-3 minutes

---

**Ready to evaluate!** 🎉
