# Repository Structure

This document outlines the organized structure of the Semanticist repository.

## Directory Structure

```
semanticist/
├── semanticist/           # Core model implementation
│   ├── stage1/           # Tokenizer models (DiffuseSlot)
│   ├── stage2/           # Autoregressive models (GPT)
│   ├── engine/           # Training utilities
│   └── utils/            # Utility functions
├── experiments/          # Experimental scripts and analysis
│   ├── temporal_dynamics_framework.py
│   ├── benchmark_compression.py
│   ├── dora_compression_pipeline.py
│   ├── pilot_experiment_comparison.png
│   └── pilot_experiment_results.json
├── validation/           # Model testing and validation
│   ├── simple_test.py
│   ├── simple_token_test.py
│   ├── test_semanticist.py
│   └── test_net.py
├── docs/                 # Documentation
│   ├── experimental_design_plan.md
│   ├── execution_steps.md
│   ├── temporal_dynamics_technical_report.md
│   └── slotformer_experimental_design.md
├── scripts/              # Utility scripts
│   ├── setup_env.sh
│   ├── download_datasets_simple.sh
│   ├── download_video_datasets.py
│   └── batch_process_videos.py
├── configs/              # Configuration files
├── data_analysis/        # Data analysis results
├── video_token_analysis/ # Video analysis results
├── dora_videos/          # Video datasets
├── examples/             # Example images
└── pages/                # Web interface
```

## Core Files (Root Level)

- `train.py` - SlotFormer training script
- `train_net.py` - Network training utilities
- `tok_demo.py` - Tokenizer demo
- `gen_demo.py` - Generation demo
- `ar_gen.ipynb` - Autoregressive generation notebook
- `requirements.txt` - Python dependencies
- `environment.yml` - Conda environment

## Key Directories

### `semanticist/` - Core Implementation
Contains the main model implementations:
- **Stage 1**: DiffuseSlot tokenizer with diffusion-based reconstruction
- **Stage 2**: GPT autoregressive model for token generation
- **Engine**: Training utilities and trainer classes
- **Utils**: Dataset loading, device utilities, logging

### `experiments/` - Research Experiments
Active experimental scripts and results:
- Temporal dynamics analysis
- Video compression benchmarks
- Cross-video token analysis
- Pilot experiment results

### `validation/` - Model Testing
Scripts for validating model performance:
- Simple tokenizer tests
- Full model validation
- Network testing utilities

### `docs/` - Documentation
Research plans, technical reports, and experimental designs:
- Experimental design plans
- Technical reports
- SlotFormer validation framework

### `scripts/` - Utility Scripts
Setup and data processing scripts:
- Environment setup
- Dataset downloading
- Video processing utilities

## Usage Patterns

### Training a Model
```bash
# SlotFormer training
python train.py

# Full Semanticist training
python train_net.py
```

### Running Experiments
```bash
# Temporal dynamics analysis
python experiments/temporal_dynamics_framework.py

# Compression benchmarks
python experiments/benchmark_compression.py
```

### Model Validation
```bash
# Quick tokenizer test
python validation/simple_test.py

# Full model validation
python validation/test_semanticist.py
```

### Data Processing
```bash
# Setup environment
bash scripts/setup_env.sh

# Download datasets
bash scripts/download_datasets_simple.sh
```

## File Organization Principles

1. **Core models** in `semanticist/` package
2. **Experimental code** in `experiments/`
3. **Testing/validation** in `validation/`
4. **Documentation** in `docs/`
5. **Utilities** in `scripts/`
6. **Results** in appropriate analysis folders

This structure makes it easier to:
- Navigate between different types of work
- Find relevant scripts and documentation
- Maintain clean separation of concerns
- Scale the project as it grows
