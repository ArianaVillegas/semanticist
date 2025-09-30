# Slot-CoCa Project Overview

## 🎯 Objective

Train a multimodal SlotFormer model that combines:
1. Object-centric visual representations (slots)
2. Vision-language alignment (CLIP-style)
3. Image captioning capability (CoCa-style)

Then benchmark against SOTA models (CLIP, CoCa) on standard metrics.

## 📁 Project Structure

```
multimodal_training/
├── models/
│   └── slot_coca.py              # Main model architecture
├── evaluation/
│   ├── eval_zero_shot.py         # ImageNet zero-shot classification
│   ├── eval_retrieval.py         # Image-text retrieval (Recall@K)
│   └── eval_captioning.py        # Image captioning (BLEU, CIDEr)
├── imagenet_captions_dataset.py  # Dataset loader
├── train_multimodal.py           # DDP training script (2x A100)
├── quick_start.sh                # One-click setup and training
├── README.md                     # Full documentation
└── PROJECT_OVERVIEW.md           # This file
```

## 🏗️ Architecture Details

### Model: Slot-CoCa

```python
class SlotCoCa(nn.Module):
    """
    Components:
    1. Vision Encoder: DINOv3 ViT → SlotFormer (128 slots)
    2. Text Encoder: DistilBERT
    3. Contrastive Heads: Project to shared 256-dim space
    4. Caption Decoder: Transformer decoder with cross-attention
    """
```

### Training Objectives

**Total Loss = λ₁ * L_recon + λ₂ * L_contrast + λ₃ * L_caption**

1. **L_recon (Reconstruction Loss)**
   - MSE between original and reconstructed patches
   - Preserves SlotFormer's object-centric learning
   - Weight: λ₁ = 1.0

2. **L_contrast (Contrastive Loss)**
   - InfoNCE loss (CLIP-style)
   - Aligns image and text embeddings
   - Temperature: τ = 0.07
   - Weight: λ₂ = 1.0

3. **L_caption (Captioning Loss)**
   - Cross-entropy for next-token prediction
   - Cross-attention from slots to caption tokens
   - Weight: λ₃ = 1.0

## 📊 Training Plan

### Phase 1: Proof-of-Concept ✅ (Current)

**Goal**: Validate architecture and training pipeline

**Configuration**:
- Dataset: 100K image-caption pairs (subset)
- Hardware: 2x A100 (80GB each)
- Batch size: 256 total (128 per GPU)
- Epochs: 30
- Time: ~1-2 days
- Memory: ~40GB per GPU

**Expected Results**:
- Zero-shot ImageNet: 15-25% top-1
- Image→Text R@1: 20-30%
- Text→Image R@1: 15-25%
- Captioning CIDEr: 40-60

### Phase 2: Full Scale (Next)

**Configuration**:
- Dataset: 1M image-caption pairs (full ImageNet-Captions)
- Epochs: 50
- Time: ~1 week
- Expected improvement: +5-10% across all metrics

## 🎯 Evaluation Metrics

### 1. Zero-Shot Classification
- **Dataset**: ImageNet validation (50K images)
- **Metrics**: Top-1 and Top-5 accuracy
- **Baseline**: CLIP ViT-B/32 (63.4% top-1)

### 2. Image-Text Retrieval
- **Dataset**: ImageNet-Captions validation (5K pairs)
- **Metrics**: 
  - Image→Text: Recall@1, @5, @10
  - Text→Image: Recall@1, @5, @10
- **Baseline**: CLIP ViT-B/32 (58.4% I2T R@1)

### 3. Image Captioning
- **Dataset**: ImageNet-Captions validation
- **Metrics**: BLEU-1, BLEU-4, CIDEr, SPICE
- **Baseline**: CoCa (CIDEr ~120)

## 🔄 Workflow

### Step 1: Setup

```bash
cd multimodal_training
./quick_start.sh
```

This will:
- ✅ Install dependencies (transformers, sentence-transformers, CLIP)
- ✅ Check dataset availability
- ✅ Create necessary directories
- ✅ Verify GPU access

### Step 2: Training

```bash
python train_multimodal.py \
    --data_dir ./data/imagenet_captions \
    --subset_size 100000 \
    --batch_size 128 \
    --world_size 2 \
    --epochs 30
```

**Monitoring**:
```bash
tensorboard --logdir ./logs/slot_coca
```

**Checkpoints** saved to: `./checkpoints/slot_coca/`
- `checkpoint_epoch_{N}.pt` - Every 5 epochs
- `best_model.pt` - Best validation loss

### Step 3: Evaluation

**Zero-Shot Classification**:
```bash
python evaluation/eval_zero_shot.py \
    --model_path ./checkpoints/slot_coca/best_model.pt \
    --data_dir ./data/imagenet/val
```

**Image-Text Retrieval**:
```bash
python evaluation/eval_retrieval.py \
    --model_path ./checkpoints/slot_coca/best_model.pt \
    --data_dir ./data/imagenet_captions
```

**Image Captioning**:
```bash
python evaluation/eval_captioning.py \
    --model_path ./checkpoints/slot_coca/best_model.pt \
    --data_dir ./data/imagenet_captions
```

## 📈 Expected Timeline

| Phase | Duration | Milestone |
|-------|----------|-----------|
| **Week 1** | 2-3 days | Setup + Proof-of-concept training |
| **Week 1** | 1 day | Initial evaluation & analysis |
| **Week 2** | 5-7 days | Full-scale training (1M samples) |
| **Week 2** | 2 days | Comprehensive evaluation |
| **Week 3** | Variable | Hyperparameter tuning & improvements |

## 🎓 Key Research Questions

1. **Do slot-based representations benefit multimodal learning?**
   - Compare against non-slot baseline
   - Analyze slot-level semantics

2. **How does multi-task learning affect slot quality?**
   - Reconstruction vs reconstruction+contrastive+caption
   - Ablation studies on loss weights

3. **Can slots enable better compositional understanding?**
   - Test on multi-object scenes
   - Evaluate attribute binding

4. **What's the gap to SOTA models?**
   - CLIP trained on 400M pairs
   - CoCa trained on 800M pairs
   - Our model: 100K-1M pairs
   - Isolate impact of scale vs architecture

## 🔬 Ablation Studies (Planned)

1. **Loss Weights**: Vary λ₁, λ₂, λ₃
2. **Slot Count**: 64 vs 128 vs 256 slots
3. **Text Encoder**: DistilBERT vs BERT vs T5
4. **Training Objectives**:
   - Recon only
   - Recon + Contrast
   - Recon + Caption
   - All three (full Slot-CoCa)

## 📊 Comparison Matrix

| Model | Data | Zero-Shot IN | I2T R@1 | T2I R@1 | CIDEr |
|-------|------|--------------|---------|---------|-------|
| **Slot-CoCa (PoC)** | 100K | 15-25% | 20-30% | 15-25% | 40-60 |
| **Slot-CoCa (Full)** | 1M | 25-35% | 30-40% | 25-35% | 70-90 |
| **CLIP ViT-B/32** | 400M | 63.4% | 58.4% | 37.8% | - |
| **CoCa** | 800M | 65.5% | 62.1% | 45.2% | 120+ |

**Gap Analysis**:
- Scale difference: 4-800x fewer training pairs
- Architecture: Object-centric vs dense features
- Training: From scratch vs pre-trained components

## 🚀 Future Improvements

### Short-term
- [x] Implement basic Slot-CoCa
- [x] DDP training pipeline
- [x] Evaluation scripts
- [ ] Run proof-of-concept training
- [ ] Analyze initial results

### Medium-term
- [ ] Mixed precision training (FP16)
- [ ] Gradient checkpointing for larger batches
- [ ] Better text encoder (T5-base)
- [ ] Slot-level captioning
- [ ] Visual grounding evaluation

### Long-term
- [ ] Scale to full ImageNet-Captions
- [ ] Pre-training on larger datasets (CC3M, CC12M)
- [ ] Video understanding extension
- [ ] Interactive object manipulation

## 📝 Notes

### Why Slot-CoCa?
- **Object-centric**: Natural fit for compositional understanding
- **Multi-task**: Leverages synergies between objectives
- **Flexible**: Supports various downstream tasks

### Challenges
- **Scale gap**: CLIP/CoCa trained on 100-1000x more data
- **Compute**: Limited to 2x A100 (vs multi-node clusters)
- **Dataset**: ImageNet-Captions smaller than LAION/CC

### Opportunities
- **Inductive bias**: Slots may excel on multi-object scenes
- **Interpretability**: Slot-level analysis possible
- **Efficiency**: Fewer parameters than dense models

## 📚 References

1. **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
2. **CoCa**: Yu et al., "CoCa: Contrastive Captioners are Image-Text Foundation Models", ECCV 2022
3. **SlotFormer**: Wu et al., "SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models", CVPR 2023
4. **ImageNet-Captions**: Fang et al., "From Recognition to Cognition: Visual Commonsense Reasoning", CVPR 2015

## 🤝 Team & Resources

**Hardware**: 2x NVIDIA A100 (80GB)
**Framework**: PyTorch 2.0 + DDP
**Libraries**: transformers, timm, sentence-transformers
**Monitoring**: TensorBoard
**Evaluation**: CLIP (OpenAI), pycocotools

---

*Last Updated: 2025-09-30*
