# How to Fix the Caption Generation Bug

## 🐛 The Problem
Model generates garbage ("a the the the") because it used **DistilBERT (bidirectional)** for captions. During training, BERT sees all future tokens, so the model just copies instead of learning to generate from images.

## ✅ The Solution
Use **GPT-2 (causal)** for caption generation. Model is forced to learn from visual input.

---

## 🚀 Step-by-Step Fix

### Step 1: Training Script Already Updated ✅
The training script (`train_multimodal.py`) has been automatically updated to use the fixed model.

No manual changes needed!

---

### Step 2: Retrain from Scratch

**Quick test (5 epochs, 1000 samples):**
```bash
python train_multimodal.py \
    --use_coco \
    --data_dir /home/avillegas/semanticist/datasets/coco \
    --subset_size 1000 \
    --val_subset_size 200 \
    --epochs 5 \
    --batch_size 16 \
    --lr 1e-4 \
    --num_slots 128 \
    --checkpoint_dir checkpoints/fixed_model_test \
    --log_dir logs/fixed_model_test
```

**What to watch for:**
- ✅ Caption loss should START high (~8-10) and decrease
- ✅ Model should NOT achieve 100% accuracy immediately
- ✅ Generated captions should vary between images (even if bad at first)

---

### Step 3: Test the Fixed Model

After training, test if it works:

```bash
python -c "
import torch
from models.slot_coca_fixed import SlotCoCaFixed as SlotCoCa
from coco_dataset import COCOCaptionsDataset
from pathlib import Path

# Load model
ckpt = torch.load('checkpoints/fixed_model_test/best_model.pt', map_location='cpu')
model = SlotCoCa(num_slots=128)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Load dataset
dataset = COCOCaptionsDataset(
    root_dir=Path('/home/avillegas/semanticist/datasets/coco/val2017'),
    ann_file=Path('/home/avillegas/semanticist/datasets/coco/annotations/captions_val2017.json')
)

# Test 3 images
print('Testing 3 different images:')
for i in range(3):
    sample = dataset[i * 100]
    image = sample['image'].unsqueeze(0)
    
    with torch.no_grad():
        generated_ids = model.generate_caption(image, max_length=20)
        caption = model.caption_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print(f'{i+1}. {caption}')
    print(f'   GT: {sample[\"caption\"][:50]}...')
    print()

print('✅ If captions are DIFFERENT for each image, the fix worked!')
"
```

---

### Step 4: Full Training (Once Test Works)

If the 5-epoch test looks good, run full training:

```bash
python train_multimodal.py \
    --use_coco \
    --data_dir /home/avillegas/semanticist/datasets/coco \
    --subset_size 10000 \
    --val_subset_size 1000 \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --num_slots 128 \
    --checkpoint_dir checkpoints/fixed_model_full \
    --log_dir logs/fixed_model_full
```

---

## 📊 Expected Results After Fix

| Metric | Broken Model | Fixed Model |
|--------|-------------|-------------|
| **Training accuracy** | 100% (cheating) | 20-40% (learning) |
| **Generated captions** | All identical | Different per image |
| **Visual grounding** | Ignored | Used |
| **Caption quality** | Garbage | Improving over epochs |

---

## 🔍 Key Differences: Old vs New

### Old Architecture (Broken)
```
Image → Slots → [ignored]
Caption text → BERT → sees all tokens → 100% accuracy
```

### New Architecture (Fixed)
```
Image → Slots → Prepended to GPT-2 → must use for generation
Caption text → GPT-2 → sees only previous tokens → learns properly
```

---

## ❓ FAQ

**Q: Why not just fix the causal mask in the old model?**  
A: The mask applies to the Transformer decoder, but BERT embeddings already contain bidirectional context. Can't fix it with masking.

**Q: Will this break contrastive learning?**  
A: No, contrastive learning still uses BERT (which is fine for that task).

**Q: Do I need to change the dataset or training loop?**  
A: No, only the model architecture changed. Same training code works.

**Q: What if GPT-2 is slower?**  
A: It's actually faster for generation since no cross-attention decoder needed.
