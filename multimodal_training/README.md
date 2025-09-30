# Slot-CoCa: Multimodal SlotFormer Training

Vision-language training for SlotFormer with contrastive learning and captioning.

## Architecture

**Slot-CoCa** combines three objectives:
1. **Visual Reconstruction** (SlotFormer): Learn object-centric slot representations
2. **Contrastive Learning** (CLIP-style): Align image and text embeddings
3. **Autoregressive Captioning** (CoCa-style): Generate image captions

```
Image → DINOv3 → Slots → {Reconstructor, Contrastive Head, Caption Decoder}
Text → DistilBERT → {Contrastive Head, Caption Supervision}
```

## Setup

### 1. Install Dependencies

```bash
pip install transformers sentence-transformers
pip install git+https://github.com/openai/CLIP.git  # For CLIP baseline
```

### 2. Download ImageNet-Captions Dataset

**Option A: Use existing ImageNet + generate captions**
```bash
# If you have ImageNet, we'll generate synthetic captions
python multimodal_training/imagenet_captions_dataset.py
```

**Option B: Download ImageNet-Captions**
```bash
# Download from: https://github.com/mlfoundations/imagenet-captions
# Place in: ./data/imagenet_captions/
```

## Training

### Proof-of-Concept (100K samples, 2x A100)

```bash
cd multimodal_training
python train_multimodal.py \
    --data_dir ./data/imagenet_captions \
    --subset_size 100000 \
    --batch_size 128 \
    --epochs 30 \
    --world_size 2 \
    --lr 1e-4 \
    --lambda_recon 1.0 \
    --lambda_contrast 1.0 \
    --lambda_caption 1.0
```

**Training time**: ~1-2 days on 2x A100
**Total batch size**: 256 (128 per GPU)
**Memory**: ~40GB per GPU

### Full Scale (1M samples)

```bash
python train_multimodal.py \
    --subset_size 1000000 \
    --epochs 50 \
    --batch_size 128 \
    --world_size 2
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_slots` | 128 | Number of object slots |
| `--num_layers` | 3 | Transformer layers in SlotFormer |
| `--projection_dim` | 256 | Dimension for contrastive projections |
| `--temperature` | 0.07 | Temperature for contrastive loss |
| `--lr` | 1e-4 | Learning rate |
| `--lambda_recon` | 1.0 | Weight for reconstruction loss |
| `--lambda_contrast` | 1.0 | Weight for contrastive loss |
| `--lambda_caption` | 1.0 | Weight for caption loss |

## Evaluation

### 1. Zero-Shot Classification

Evaluate on ImageNet validation set:

```bash
python evaluation/eval_zero_shot.py \
    --model_path ./checkpoints/slot_coca/best_model.pt \
    --data_dir ./data/imagenet/val \
    --batch_size 256
```

**Metrics**:
- Top-1 Accuracy
- Top-5 Accuracy
- Comparison with CLIP ViT-B/32

### 2. Image-Text Retrieval

```bash
python evaluation/eval_retrieval.py \
    --model_path ./checkpoints/slot_coca/best_model.pt \
    --data_dir ./data/imagenet_captions \
    --num_samples 5000
```

**Metrics**:
- Image→Text Recall@1, @5, @10
- Text→Image Recall@1, @5, @10
- Comparison with CLIP

### 3. Image Captioning

```bash
python evaluation/eval_captioning.py \
    --model_path ./checkpoints/slot_coca/best_model.pt \
    --data_dir ./data/imagenet_captions
```

**Metrics**:
- BLEU-1, BLEU-4
- CIDEr
- SPICE

## Expected Results

### Proof-of-Concept (100K samples, 30 epochs)

| Metric | Slot-CoCa (Expected) | CLIP ViT-B/32 | CoCa |
|--------|---------------------|---------------|------|
| **Zero-Shot ImageNet** |
| Top-1 Accuracy | 15-25% | 63.4% | 65.5% |
| Top-5 Accuracy | 35-50% | 87.8% | 88.2% |
| **Image-Text Retrieval** |
| Image→Text R@1 | 20-30% | 58.4% | 62.1% |
| Text→Image R@1 | 15-25% | 37.8% | 45.2% |
| **Captioning** |
| CIDEr | 40-60 | - | 120+ |

**Note**: Lower performance is expected since:
1. Slot-CoCa is object-centric (different inductive bias)
2. CLIP trained on 400M pairs, CoCa on 800M pairs
3. We're training from scratch on smaller data

### Full Scale (1M samples, 50 epochs)

Expected improvement of 5-10% across all metrics.

## Baselines

We compare against:

1. **CLIP (ViT-B/32)** - OpenAI's contrastive model
2. **CoCa** - Google's contrastive captioner (if available)
3. **Untrained SlotFormer** - To measure improvement from multimodal training

## Key Findings (Expected)

✅ **Slot-based representations learn cross-modal alignment**
- Proves slots can capture semantic meaning

✅ **Multi-task learning improves all objectives**
- Reconstruction + contrastive + caption works better than any single task

⚠️ **Gap to SOTA models**
- Expected due to scale and data differences
- Room for improvement with more data/compute

✅ **Object-centric bias helps certain tasks**
- May excel at multi-object scenes
- Better compositional understanding

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir ./logs/slot_coca
```

**Tracked metrics**:
- Training/validation loss (total, reconstruction, contrastive, caption)
- Learning rate schedule
- Gradient norms

### Checkpoints

Saved in `./checkpoints/slot_coca/`:
- `checkpoint_epoch_{N}.pt` - Every 5 epochs
- `best_model.pt` - Best validation loss

## Troubleshooting

### OOM Errors
- Reduce `--batch_size`
- Reduce `--num_slots`
- Use gradient checkpointing (add `--gradient_checkpointing`)

### Slow Training
- Increase `--num_workers`
- Check data loading bottleneck
- Use mixed precision (add `--fp16`)

### Poor Results
- Check loss weights (`--lambda_*`)
- Verify data loading correctly
- Try warming up learning rate
- Increase training epochs

## Citation

```bibtex
@misc{slot_coca_2025,
  title={Slot-CoCa: Object-Centric Vision-Language Learning},
  author={Your Name},
  year={2025},
  howpublished={https://github.com/dgcnz/semanticist}
}
```

## References

- [CLIP](https://arxiv.org/abs/2103.00020) - Radford et al., 2021
- [CoCa](https://arxiv.org/abs/2205.01917) - Yu et al., 2022
- [SlotFormer](https://arxiv.org/abs/2210.05861) - Wu et al., 2022
- [ImageNet-Captions](https://github.com/mlfoundations/imagenet-captions)
