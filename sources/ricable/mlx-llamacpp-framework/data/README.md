# Flow2 Data Directory

This directory contains all data files for the Flow2 project, organized into logical subdirectories.

## Directory Structure

```
data/
├── README.md           # This file
├── datasets/           # Training and evaluation datasets
├── results/            # Benchmark results and performance data
└── outputs/            # Model outputs, adapters, and fine-tuned models
```

## 📁 Datasets (`data/datasets/`)

Contains training and evaluation datasets in various formats:

- **`quotes_train.jsonl`** - Inspirational quotes training dataset
- **`train.jsonl`** - General training dataset (80 samples)
- **`valid.jsonl`** - Validation dataset (15 samples)  
- **`test.jsonl`** - Test dataset
- **`sample_training_data.jsonl`** - Sample training data for testing
- **`sample_training_data.txt`** - Text format training data

### Dataset Formats

Most datasets use JSONL format with one of these structures:

```json
{"text": "### User\nQuestion\n\n### Assistant\nAnswer"}
{"input": "Question", "output": "Answer"}  
{"prompt": "Question", "response": "Answer"}
```

## 📊 Results (`data/results/`)

Contains all benchmark results, performance data, and analysis outputs:

### Framework Benchmarks
- **`mlx_comprehensive/`** - MLX framework comprehensive benchmarks
- **`mlx_flash_attention/`** - Flash attention performance benchmarks
- **`mlx_workflow/`** - MLX workflow benchmarks
- **`multi_model/`** - Multi-model comparison benchmarks

### Model-Specific Results
- **`mlx_Meta-Llama-3.1-8B-Instruct-4bit_quantized/`** - 8B model 4-bit quantization results
- **`mlx_llama-3.1-8b-bf16_quantized/`** - 8B model bf16 quantization results
- **`quantization_test/`** - Quantization comparison results

### Performance Data
- **`*_benchmark.json`** - Individual benchmark result files
- **`*_benchmark.csv`** - Benchmark data in CSV format
- **`comprehensive_flash_attention_results.json`** - Flash attention analysis
- **`llamacpp_comprehensive_test_results.json`** - LlamaCpp framework results

### Logs and Analysis
- **`model_manager.log`** - Model management operation logs
- **`quantization.log`** - Quantization process logs
- **`*_research.md`** - Research notes and analysis
- **`performance_benchmarks.md`** - Performance analysis documentation

## 🎯 Outputs (`data/outputs/`)

Contains model outputs, fine-tuned adapters, and training results:

### LoRA Adapters
- **`quotes_lora_adapter/`** - LoRA adapter trained on quotes dataset
- **`qwen_enhanced/`** - Enhanced Qwen model with LoRA adapters
- **`tinyllama_enhanced/`** - Enhanced TinyLlama model with LoRA adapters

### Training Outputs
- **`hf_8b_test_output/`** - HuggingFace 8B model fine-tuning output
- **`test_lora_output/`** - Test LoRA training output
- **`adapters.safetensors`** - Individual adapter files
- **`best_adapters.safetensors`** - Best performing adapters

### Adapter Structure

Each adapter directory typically contains:
```
adapter_name/
├── adapter_config.json      # LoRA configuration
├── adapters.safetensors     # Trained adapter weights
├── training_args.bin        # Training arguments
└── README.md               # Adapter documentation
```

## 💡 Usage Guidelines

### Adding New Datasets
Place new training datasets in `data/datasets/` using standardized JSONL format:

```bash
# Example: Add new dataset
cp my_new_dataset.jsonl data/datasets/
```

### Benchmark Results
All benchmark scripts should save results to `data/results/`:

```python
# Example: Save benchmark results
output_dir = "data/results/my_benchmark/"
save_benchmark_results(results, output_dir)
```

### Model Outputs
Fine-tuning outputs should go to `data/outputs/`:

```python
# Example: Save fine-tuned model
output_dir = "data/outputs/my_finetuned_model/"
save_adapter(adapter, output_dir)
```

## 🧹 Maintenance

### Cleanup Old Results
Periodically clean up old benchmark results:

```bash
# Remove results older than 30 days
find data/results/ -name "*.json" -mtime +30 -delete
```

### Archive Large Outputs
Archive large model outputs to save space:

```bash
# Archive old outputs
tar -czf archived_outputs_$(date +%Y%m%d).tar.gz data/outputs/old_*/
```

## 📊 Data Size Guidelines

- **Datasets**: Keep individual files under 100MB
- **Results**: JSON files typically 1-50MB
- **Outputs**: Adapter files typically 10-500MB

## 🔍 Finding Data

### Locate Specific Results
```bash
# Find all MLX benchmarks
find data/results/ -name "*mlx*" -type f

# Find recent benchmark results  
find data/results/ -name "*.json" -mtime -7

# Find specific adapter
find data/outputs/ -name "*quotes*" -type d
```

### Data Inventory
Use these commands to get an overview:

```bash
# Count datasets
ls data/datasets/ | wc -l

# Size of results directory
du -sh data/results/

# List all adapters
ls data/outputs/*/adapters.safetensors 2>/dev/null
```

---

This organization ensures that data is easy to find, manage, and maintain while supporting the Flow2 framework's multi-framework approach to ML model development.