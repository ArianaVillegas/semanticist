# Evaluation Guide for Slot-CoCa on Imagenette

Congratulations on completing training! 🎉 Here's how to evaluate your trained model.

## 📋 Prerequisites

- ✅ Trained model checkpoint (e.g., `../checkpoints/slot_coca_real_captions/best_model.pt`)
- ✅ Imagenette dataset at `../datasets/imagenette2/`
- ✅ CLIP installed for baseline comparisons

## 🎯 Evaluation Scripts

All evaluation scripts now support the `--use_imagenette` flag to work with your Imagenette dataset.

### 1. Zero-Shot Classification

Evaluates how well the model can classify images into the 10 Imagenette classes without seeing any training examples for classification.

```bash
cd evaluation

python eval_zero_shot.py \
    --model_path ../checkpoints/slot_coca_real_captions/best_model.pt \
    --data_dir ../../datasets/imagenette2 \
    --use_imagenette \
    --batch_size 128 \
    --device cuda
```

**Expected Output:**
```
ZERO-SHOT IMAGENETTE CLASSIFICATION
====================================
Slot-CoCa:
  Top-1 Accuracy: 45-65%
  Top-5 Accuracy: 85-95%

CLIP (ViT-B/32) Baseline:
  Top-1 Accuracy: 85-90%
  Top-5 Accuracy: 98-99%

Gap: -25% to -40% (Top-1)
```

**What it means:**
- **Top-1**: Model's first prediction is correct
- **Top-5**: Correct answer in top 5 predictions
- Gap to CLIP is expected (they trained on 400M pairs vs your 47K)

---

### 2. Image-Text Retrieval

Evaluates how well the model can match images to captions and vice versa.

```bash
python eval_retrieval.py \
    --model_path ../checkpoints/slot_coca_real_captions/best_model.pt \
    --data_dir ../../datasets/imagenette2 \
    --use_imagenette \
    --num_samples 3925 \
    --batch_size 128 \
    --device cuda
```

**Expected Output:**
```
IMAGE-TEXT RETRIEVAL EVALUATION
================================
Slot-CoCa:

Image → Text Retrieval:
  Recall@1: 30-50%
  Recall@5: 60-80%
  Recall@10: 75-90%

Text → Image Retrieval:
  Recall@1: 25-45%
  Recall@5: 55-75%
  Recall@10: 70-85%

CLIP Baseline:
  I→T Recall@1: 65-75%
  T→I Recall@1: 50-60%
```

**What it means:**
- **Recall@1**: Top retrieved item is correct
- **Recall@5**: Correct item in top 5 retrievals
- **Recall@10**: Correct item in top 10 retrievals

---

### 3. Image Captioning

Evaluates the quality of generated captions.

```bash
python eval_captioning.py \
    --model_path ../checkpoints/slot_coca_real_captions/best_model.pt \
    --data_dir ../../datasets/imagenette2 \
    --use_imagenette \
    --num_samples 1000 \
    --batch_size 32 \
    --device cuda
```

**Expected Output:**
```
IMAGE CAPTIONING EVALUATION
============================
📊 Results (1000 images):
  BLEU-1: 30-45%

📝 Sample Captions:
Image 123:
  Reference: Country Church at Dusk
  Generated: a church in the evening

Image 456:
  Reference: marlborough pool mrs tench
  Generated: a fish in water

Image 789:
  Reference: Gas pump
  Generated: a gas station pump
```

**What it means:**
- **BLEU-1**: Measures unigram overlap between generated and reference captions
- Higher is better (100% = perfect match)
- 30-45% is reasonable for this task

---

## 📊 Quick Evaluation (All Metrics)

Run all evaluations in sequence:

```bash
cd evaluation

# 1. Zero-shot classification
echo "Running zero-shot classification..."
python eval_zero_shot.py \
    --model_path ../checkpoints/slot_coca_real_captions/best_model.pt \
    --data_dir ../../datasets/imagenette2 \
    --use_imagenette \
    --batch_size 128

# 2. Image-text retrieval  
echo "Running image-text retrieval..."
python eval_retrieval.py \
    --model_path ../checkpoints/slot_coca_real_captions/best_model.pt \
    --data_dir ../../datasets/imagenette2 \
    --use_imagenette \
    --num_samples 3925

# 3. Image captioning
echo "Running image captioning..."
python eval_captioning.py \
    --model_path ../checkpoints/slot_coca_real_captions/best_model.pt \
    --data_dir ../../datasets/imagenette2 \
    --use_imagenette \
    --num_samples 1000

echo "✅ All evaluations complete!"
```

**Total time**: ~15-20 minutes

---

## 🎯 Interpreting Results

### Good Results ✅

- **Zero-shot Top-1**: >40%
- **Zero-shot Top-5**: >80%
- **I→T Retrieval R@1**: >30%
- **T→I Retrieval R@1**: >25%
- **BLEU-1**: >30%

If you hit these targets, your multimodal training was successful!

### What Affects Performance

**Architecture**:
- Slot count (128 is optimal from memory)
- Projection dimension (256)
- Number of training epochs

**Training**:
- Loss weights (λ_recon, λ_contrast, λ_caption)
- Learning rate
- Batch size

**Data**:
- Caption quality (real vs synthetic)
- Number of training pairs
- Caption diversity

---

## 📈 Comparison to Baselines

### vs CLIP (ViT-B/32)

| Metric | Slot-CoCa (Expected) | CLIP | Notes |
|--------|---------------------|------|-------|
| Zero-shot Top-1 | 45-65% | 85-90% | CLIP trained on 400M pairs |
| I→T R@1 | 30-50% | 65-75% | Gap due to scale |
| T→I R@1 | 25-45% | 50-60% | Still demonstrates alignment |

**Why the gap?**
- Scale: 47K training pairs vs 400M
- Architecture: Object-centric vs dense
- Training: From scratch vs pre-trained components

### vs Previous Cross-Modal Score

From memory, you had:
- **Cross-modal alignment: 0.722**

Compare your I→T and T→I Recall@1 scores to see if Slot-CoCa matches or exceeds this!

---

## 🔍 Debugging Poor Results

### If Zero-Shot Accuracy < 30%

**Check:**
1. Model loaded correctly (no errors)
2. Captions were used during training (not just reconstruction)
3. Contrastive loss decreased during training
4. Text encoder is working (test with random text)

**Try:**
- Re-run with `--num_samples 100` to test quickly
- Check if random chance (10% for 10 classes) is beaten

### If Retrieval R@1 < 15%

**Check:**
1. Similarity matrix computation is correct
2. Vision and text embeddings are normalized
3. Temperature parameter is reasonable (0.07)

**Try:**
- Visualize embedding space (t-SNE)
- Check if vision/text embeddings are in same range

### If Generated Captions are Nonsense

**Check:**
1. Caption loss decreased during training
2. Greedy decoding is working
3. Tokenizer vocabulary is correct

**Try:**
- Sample multiple captions per image
- Use beam search (if implemented)
- Check caption decoder initialization

---

## 📊 Save Results

Create a results file:

```bash
cd evaluation

# Run all and save to file
{
  echo "SLOT-COCA EVALUATION RESULTS"
  echo "============================"
  echo ""
  echo "Model: best_model.pt"
  echo "Date: $(date)"
  echo ""
  
  echo "1. ZERO-SHOT CLASSIFICATION"
  python eval_zero_shot.py \
    --model_path ../checkpoints/slot_coca_real_captions/best_model.pt \
    --data_dir ../../datasets/imagenette2 \
    --use_imagenette
  
  echo ""
  echo "2. IMAGE-TEXT RETRIEVAL"
  python eval_retrieval.py \
    --model_path ../checkpoints/slot_coca_real_captions/best_model.pt \
    --data_dir ../../datasets/imagenette2 \
    --use_imagenette
  
  echo ""
  echo "3. IMAGE CAPTIONING"
  python eval_captioning.py \
    --model_path ../checkpoints/slot_coca_real_captions/best_model.pt \
    --data_dir ../../datasets/imagenette2 \
    --use_imagenette
    
} | tee evaluation_results.txt

echo "Results saved to evaluation_results.txt"
```

---

## 🎓 Next Steps After Evaluation

### If Results Are Good

1. **Analyze what worked**:
   - Which loss component helped most?
   - Did real captions matter?
   - How do slots compare to dense features?

2. **Scale up**:
   - Train on larger datasets (CC3M, CC12M)
   - Use more training epochs
   - Try larger models

3. **Publish findings**:
   - Document architecture choices
   - Compare to baselines thoroughly
   - Share results with community

### If Results Need Improvement

1. **Ablation studies**:
   - Turn off reconstruction loss
   - Turn off contrastive loss
   - Turn off caption loss
   - See which helps most

2. **Hyperparameter tuning**:
   - Adjust loss weights
   - Try different learning rates
   - Experiment with slot counts

3. **Architecture improvements**:
   - Larger text encoder (BERT-base → T5)
   - More projection dimensions
   - Better slot initialization

---

## 💡 Key Insights to Look For

1. **Do slots help multimodal learning?**
   - Compare to non-slot baseline
   - Analyze per-slot attention

2. **Multi-task synergy**:
   - Does reconstruction improve retrieval?
   - Does contrastive learning help captioning?

3. **Object-centric advantage**:
   - Do multi-object scenes perform better?
   - Can slots bind attributes correctly?

4. **Scaling behavior**:
   - How does performance improve with more data?
   - What's the minimum viable dataset size?

---

## ✅ Success Criteria

Your Slot-CoCa proof-of-concept is successful if:

1. ✅ Zero-shot accuracy beats random (>10%)
2. ✅ Retrieval recall beats chance significantly
3. ✅ Generated captions are coherent
4. ✅ All three objectives trained successfully
5. ✅ Results demonstrate multimodal alignment

If all pass → You've validated the architecture! 🎉

---

**Questions or issues?** Check the evaluation scripts for detailed error messages and debugging output.
