# Critical Bug: Model Cheating During Training

## 🐛 The Problem

The model achieved **100% accuracy** during training but generates garbage ("a the the the") at test time.

### Test Results
1. ✅ **Test 1**: All images → identical caption (ignores visual input)
2. ✅ **Test 2**: Real slots = Zero slots (cross-attention not working)
3. ✅ **Test 3**: Caption loss ≈ 0, accuracy = 100% (model is cheating)

---

## 🔍 Root Cause

**The model uses DistilBERT (bidirectional encoder) for caption generation.**

### What Happens During Training:
```python
# Step 1: Encode FULL caption with BERT (sees all tokens)
caption_embeds = self.text_encoder(caption_tokens)  # ← Bidirectional!

# Step 2: Apply causal mask to decoder
decoded = self.caption_decoder(
    tgt=caption_embeds,  # ← Already contains info about future tokens!
    memory=slots,
    tgt_mask=causal_mask  # ← Too late, damage already done
)
```

**Problem**: `caption_embeds` already contains information about future tokens because BERT is bidirectional. The causal mask can't fix this.

**Result**: Model perfectly predicts captions (100% accuracy) because it sees the answer, not because it learned to generate from images.

###

 What Happens During Generation:
- No ground truth available → can't cheat
- Model never learned to use visual input → generates garbage

---

## ✅ The Solution

**Use GPT-2 (causal language model) for caption generation instead of BERT.**

### Fixed Architecture:
- **BERT** for contrastive learning (bidirectional is fine here)
- **GPT-2** for caption generation (properly causal)

### Key Changes in `slot_coca_fixed.py`:
1. Separate models for contrastive (BERT) vs generation (GPT-2)
2. Visual slots prefixed to GPT-2 input (no cross-attention needed)
3. GPT-2 naturally causal - can't see future tokens

---

## 📊 Why This Fix Works

| Component | Old (Broken) | New (Fixed) |
|-----------|-------------|-------------|
| **Text encoder** | DistilBERT | DistilBERT (for contrastive only) |
| **Caption model** | DistilBERT + Decoder | GPT-2 (causal LM) |
| **Training** | Sees future tokens | Truly autoregressive |
| **Visual grounding** | Ignored | Required (no cheating) |

---

## 🚀 Next Steps

1. Install GPT-2: `pip install transformers` (already have it)
2. Update training script to use `SlotCoCaFixed`
3. Retrain from scratch with fixed architecture
4. Verify generation uses visual input

---

## 📝 Lessons Learned

1. **Bidirectional encoders can't do causal generation** - embeddings already contain future context
2. **100% training accuracy is suspicious** - usually means cheating or data leakage  
3. **Test with ablations** - zero slots test revealed the cross-attention wasn't being used
4. **Trust the diagnostics** - systematic testing pinpointed the exact issue
