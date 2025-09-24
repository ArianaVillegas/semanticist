# Semanticist Model Testing Guide

## Overview
The Semanticist model implements a novel PCA-guided image tokenization approach with two main components:
1. **Stage 1**: DiffuseSlot tokenizer - converts images to semantic tokens
2. **Stage 2**: GPT autoregressive model - generates images from class labels

## Quick Setup

### 1. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate semanticist
```

### 2. Test Scripts Available

#### Simple Test (Verify Setup)
```bash
python simple_test.py
```

#### Full Testing Suite
```bash
# Test tokenizer reconstruction on example images
python test_semanticist.py --mode tokenizer --images examples/*

# Test class-conditional generation
python test_semanticist.py --mode generation --classes 1 281 285

# Test both modes
python test_semanticist.py --mode both
```

#### Original Demo Scripts
```bash
# Tokenizer demo (Gradio interface)
python tok_demo.py

# Generation demo (Gradio interface)  
python gen_demo.py
```

## Model Architecture

### DiffuseSlot Tokenizer
- **Input**: 256×256 RGB images
- **Output**: 256 semantic tokens (16-dim each)
- **Key Feature**: Can reconstruct with fewer tokens (1, 4, 16, 32, etc.)
- **Encoder**: Vision Transformer (ViT-Base)
- **Decoder**: Diffusion Transformer (DiT-L)

### GPT Autoregressive Model
- **Input**: ImageNet class labels (0-999)
- **Output**: 32 semantic tokens
- **Architecture**: GPT-L with 32 slots, 16-dim embeddings
- **Training**: Uses pre-trained tokenizer

## Testing Capabilities

### 1. Image Reconstruction Testing
- Load your own images
- Test reconstruction quality with different token counts
- Visualize how semantic information is preserved with fewer tokens
- Compare original vs reconstructed images

### 2. Class-Conditional Generation
- Generate images from ImageNet class labels
- Test with different numbers of tokens (1, 4, 16, 32)
- Visualize progressive image quality improvement
- Compare different CFG scales

### 3. Visualization Features
- Side-by-side comparisons
- Token count analysis
- Quality progression plots
- Automatic saving of results

## Key Parameters

### Tokenizer Parameters
- `--cfg_scale`: Classifier-free guidance scale (default: 4.0)
- `--tokens`: Token counts to test (default: [1,4,16,32,64,128,256])

### Generation Parameters
- `--gen_cfg`: Generation CFG scale (default: 6.0)
- `--ae_cfg`: Autoencoder CFG scale (default: 1.0)
- `--classes`: ImageNet class IDs to test

## Expected Results

### Tokenizer Reconstruction
- **1 token**: Basic shape/color
- **4 tokens**: Rough structure
- **16 tokens**: Clear object recognition
- **32+ tokens**: High-quality reconstruction

### Class Generation
- **1 token**: Abstract representation
- **4 tokens**: Basic object shape
- **16 tokens**: Recognizable object
- **32 tokens**: Detailed, high-quality image

## File Structure
```
semanticist/
├── test_semanticist.py      # Main testing script
├── simple_test.py           # Basic setup verification
├── tok_demo.py             # Tokenizer Gradio demo
├── gen_demo.py             # Generation Gradio demo
├── environment.yml         # Conda environment
├── setup_env.sh           # Setup script
├── examples/              # Sample images
├── results/               # Output directory
└── semanticist/           # Model code
    ├── stage1/           # Tokenizer
    ├── stage2/           # Generation
    └── utils/            # Utilities
```

## Next Steps After Environment Setup

1. **Verify Setup**: `python simple_test.py`
2. **Test Tokenizer**: `python test_semanticist.py --mode tokenizer`
3. **Test Generation**: `python test_semanticist.py --mode generation`
4. **Explore Results**: Check `results/` directory for plots
5. **Try Custom Images**: Add your images and test reconstruction

## Troubleshooting

- **CUDA Issues**: Model will fallback to CPU if CUDA unavailable
- **Memory Issues**: Reduce batch size or use fewer tokens
- **Missing Models**: Models download automatically from HuggingFace
- **Import Errors**: Ensure conda environment is activated

## Model Downloads
Models are automatically downloaded from HuggingFace:
- `tennant/semanticist/semanticist_tok_XL.pkl` (Tokenizer)
- `tennant/semanticist/semanticist_ar_gen_L.pkl` (Generator)
