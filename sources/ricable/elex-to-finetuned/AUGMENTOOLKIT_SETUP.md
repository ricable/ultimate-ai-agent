# Augmentoolkit Integration Setup

This document explains how to set up and use the integrated Augmentoolkit functionality for advanced dataset generation.

## Overview

The Flow4 pipeline now includes integrated Augmentoolkit for generating high-quality instruction-response datasets optimized for MLX fine-tuning on Apple Silicon.

### Features Integrated:

- **Advanced QA Generation**: Multi-stage validation with question quality, answer relevancy, and accuracy checks
- **MLX Optimization**: Native Apple Silicon support with MLX-compatible dataset formats
- **Factual Dataset Generation**: Specialized for technical telecommunications documentation
- **Quality Control**: Configurable validation pipeline for dataset quality assurance
- **JSONL Output**: Ready-to-use format for MLX fine-tuning

## Installation Requirements

### Core Dependencies
```bash
# Already installed via uv
source .venv/bin/activate
uv pip install pyyaml
```

### For Full Augmentoolkit Support
```bash
# Install Augmentoolkit dependencies
uv pip install openai transformers torch

# For MLX support (Apple Silicon only)
uv pip install mlx mlx-lm
```

### Alternative: Use Backup Code
The system will automatically use the Augmentoolkit implementation from `backup-code/flow4/augmentoolkit/` if available.

## Usage

### 1. Complete Pipeline with Augmentoolkit
```bash
# Process documents and generate advanced datasets
python src/run_flow4.py --verbose pipeline \
  --input data/ \
  --output-dir output \
  --use-augmentoolkit \
  --augmentoolkit-config src/flow4_factual_full.yaml
```

### 2. Generate Datasets Only
```bash
# Generate datasets from existing chunks
python src/run_flow4.py --verbose generate \
  --input output/chunks \
  --output-dir augmentoolkit_output \
  --config src/flow4_factual_full.yaml \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

### 3. Fine-tune with Generated Datasets
```bash
# Fine-tune using Augmentoolkit-generated datasets
python src/run_flow4.py finetune \
  --dataset augmentoolkit_output/mlx_dataset.jsonl \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --num-iters 300 \
  --batch-size 4
```

## Configuration

The `src/flow4_factual_full.yaml` configuration is optimized for:

- **Full Processing**: No subset limitations
- **MLX Models**: Apple Silicon optimized
- **Quality Assurance**: Multiple validation stages
- **Telecommunications Focus**: Domain-specific prompts
- **High Concurrency**: Faster generation (concurrency_limit: 8)

### Key Configuration Options:
```yaml
factual_sft_settings:
  factual_small_model: mlx-community/Llama-3.2-3B-Instruct-4bit
  factual_large_model: mlx-community/Llama-3.2-3B-Instruct-4bit
  factual_small_mode: mlx
  factual_large_mode: mlx

system:
  concurrency_limit: 8
  use_subset: False  # Process all chunks
  
final_datasaving_settings:
  template: "chatml"  # ChatML format for instruction tuning
```

## Output Structure

Augmentoolkit generates the following outputs:

```
augmentoolkit_output/
├── mlx_dataset.jsonl              # MLX-ready training data
├── augmentoolkit_dataset.json     # Original Augmentoolkit format  
└── generation_summary.json        # Generation statistics
```

### MLX Dataset Format
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an AI assistant specialized in telecommunications..."
    },
    {
      "role": "user", 
      "content": "What is the purpose of PTP synchronization in 5G networks?"
    },
    {
      "role": "assistant",
      "content": "PTP synchronization in 5G networks ensures..."
    }
  ]
}
```

## Quality Validation

Augmentoolkit includes multi-stage validation:

1. **Question Quality Check**: Ensures questions are well-formed and answerable
2. **Answer Relevancy Check**: Validates answers are relevant to questions
3. **Answer Accuracy Check**: Verifies factual accuracy against source material

### Disable for Faster Generation:
```bash
python src/run_flow4.py generate \
  --input output/chunks \
  --output-dir augmentoolkit_output \
  --no-quality-checks
```

## Troubleshooting

### Common Issues:

1. **"Augmentoolkit not available"**
   - Install missing dependencies: `uv pip install openai transformers torch`
   - Ensure backup-code/flow4/augmentoolkit/ exists

2. **"MLX not available"** 
   - Apple Silicon only: `uv pip install mlx mlx-lm`
   - Use API mode for other platforms

3. **Memory Issues**
   - Reduce concurrency: `--concurrency 4`
   - Use smaller model or subset processing

### Validation:
```bash
# Test Augmentoolkit availability
python -c "from src.core.augmentoolkit_generator import AugmentoolkitGenerator; print('Available:', AugmentoolkitGenerator.is_available())"
```

## Performance Tips

1. **Concurrency**: Adjust based on system capabilities (default: 8)
2. **Model Size**: Use 3B models for faster generation, 7B+ for quality
3. **Chunk Processing**: Process 100-500 chunks per batch for optimal performance
4. **Apple Silicon**: MLX provides significant speedup over CPU inference

## Next Steps

After generating datasets:

1. **Validate Quality**: Review generation_summary.json for statistics
2. **Fine-tune Model**: Use mlx_dataset.jsonl for training
3. **Test Performance**: Evaluate fine-tuned model on domain tasks
4. **Iterate**: Adjust configuration based on results