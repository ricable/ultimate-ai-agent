# CLAUDE.md
Always use uv for virtual env and python package management.
Always activate the venv prior to running "flow4" cli command
**LOCAL-ONLY PIPELINE**: All processing uses MLX models on Apple Silicon with no external API dependencies.
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Flow4 is a comprehensive document processing pipeline that converts HTML and PDF documents to Markdown, cleans them, and chunks them for RAG applications. The project features integrated Augmentoolkit for advanced dataset generation and MLX fine-tuning capabilities optimized for Apple Silicon.

### Key Features
- **Document Processing**: HTML/PDF → Markdown conversion with intelligent cleaning
- **Semantic Chunking**: Context-aware chunking with quality scoring
- **Advanced Dataset Generation**: Integrated Augmentoolkit with multi-stage validation
- **MLX Fine-tuning**: Apple Silicon optimized training with LoRA adaptation
- **Production Ready**: Enterprise-grade CLI with comprehensive configuration

## Core Architecture

### Module Structure
- **`src/`**: Main application code with streamlined Flow4 implementation
- **`src/cli/`**: Command-line interface with subcommands (pipeline, convert, chunk, generate, finetune)
- **`src/core/`**: Core processing components (converter, chunker, cleaner, pipeline orchestration, MLX finetuner, Augmentoolkit generator)
- **`src/utils/`**: Configuration management, logging, and utility functions
- **`src/flow4_factual_full.yaml`**: Augmentoolkit configuration for advanced dataset generation
- **`configs/`**: Configuration files (centralized config management)
- **`data/`**: Input documents (HTML files for processing)
- **`backup-code/flow4/`**: Reference implementation (for development reference only)

### Key Components
- **DocumentPipeline**: Main orchestrator that coordinates conversion, cleaning, chunking, and RAG preparation
- **DocumentConverter**: Uses IBM Docling for HTML/PDF to Markdown conversion with table/figure extraction
- **DocumentChunker**: Intelligent chunking with semantic understanding and configurable strategies
- **HTMLCleaner/MarkdownCleaner**: Content cleaning to remove headers, footers, disclaimers
- **AugmentoolkitGenerator**: Advanced dataset generation with multi-stage validation and MLX optimization
- **MLXFineTuner**: Apple Silicon optimized fine-tuning with LoRA adaptation and interactive chat

## CLI Commands and Development Workflow

### Main Commands
```bash
# Activate virtual environment first
source .venv/bin/activate

# Complete pipeline processing
python src/run_flow4.py --verbose pipeline --input data/ --output-dir output

# Advanced pipeline with LOCAL MLX dataset generation
python src/run_flow4.py --verbose pipeline --input data/ --output-dir output --use-augmentoolkit --augmentoolkit-config configs/flow4_local_mlx.yaml

# LOCAL dataset generation only (using existing chunks)
python src/run_flow4.py --verbose generate --input output/chunks --output-dir augmentoolkit_output --config configs/flow4_local_mlx.yaml

# Document conversion only
python src/run_flow4.py convert --input data/ --output-dir output/markdown

# Chunking existing markdown
python src/run_flow4.py chunk --input output/combined/combined_document.md --output-dir output/chunks

# MLX fine-tuning with Augmentoolkit datasets (Apple Silicon)
python src/run_flow4.py finetune --dataset augmentoolkit_output/mlx_dataset.jsonl --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

### Development Commands
```bash
# Set up development environment with uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# For LOCAL dataset generation (Apple Silicon REQUIRED)
uv pip install mlx mlx-lm

# Optional: Enhanced document processing
uv pip install docling

# Test pipeline with sample data
python src/run_flow4.py --verbose pipeline --input data/ --output-dir output

# Test individual components
python src/run_flow4.py convert --input data/ --output-dir output/test
python src/run_flow4.py chunk --input output/combined/combined_document.md --output-dir output/chunks
```

## Configuration System

Flow4 uses a comprehensive dataclass-based configuration system with specialized support for Augmentoolkit and MLX:

### Configuration Sources (in order of precedence)
1. Command line arguments
2. Environment variables (`FLOW4_*`)
3. Configuration files (pyproject.toml, .env, YAML)
4. Default values

### Key Configuration Classes
- **PipelineConfig**: Overall pipeline settings
- **DoclingConfig**: Document processing options (tables, figures, multimodal)
- **ChunkingConfig**: Chunking strategies and semantic processing
- **AugmentoolkitConfig**: Advanced dataset generation settings (MLX-optimized)
- **MLXConfig**: Apple Silicon fine-tuning configuration

### Augmentoolkit Configuration
The `configs/flow4_factual_full.yaml` (also available in `src/`) provides comprehensive settings for:
- **MLX Models**: Apple Silicon optimized (`mlx-community/Llama-3.2-3B-Instruct-4bit`)
- **Quality Control**: Multi-stage validation (question quality, answer relevancy, accuracy)
- **Domain Focus**: Telecommunications-specific prompts and context
- **Performance**: High concurrency (8 parallel) with no subset limitations
- **Output Format**: ChatML format ready for MLX fine-tuning

## Processing Pipeline Flow

1. **Extraction**: Extract files from ZIP archives with intelligent filtering
2. **Analysis**: Detect document types and content structure
3. **Cleaning**: Remove unwanted elements (headers, footers, legal disclaimers)
4. **Conversion**: Convert to Markdown using Docling with advanced feature extraction
5. **Consolidation**: Combine multiple documents with metadata preservation
6. **Chunking**: Split into semantic chunks using hybrid strategies
7. **Dataset Generation**: 
   - **Basic**: Standard RAG datasets (JSON/JSONL/CSV)
   - **Advanced**: Augmentoolkit with multi-stage validation and MLX optimization
8. **Fine-tuning**: MLX-based training with LoRA adaptation (Apple Silicon)

## Code Patterns and Conventions

### Error Handling
- Uses comprehensive logging throughout with configurable levels
- Graceful fallbacks for optional dependencies (MLX, Augmentoolkit, multimodal features)
- Validation at each pipeline stage with detailed error reporting

### Modular Design
- Each component is independently testable and configurable
- Optional feature detection with import guards
- Plugin-style architecture for extensibility

### Output Structure
```
output/
├── markdown/              # Individual converted files
├── combined/              # Concatenated documents  
├── chunks/                # Document chunks with metadata
├── rag/                   # Basic RAG datasets (JSON/JSONL/CSV)
├── extracted/             # Tables, figures, images
├── augmentoolkit/         # Advanced Augmentoolkit datasets
│   ├── mlx_dataset.jsonl  # MLX-ready training data
│   ├── augmentoolkit_dataset.json  # Original format
│   └── generation_summary.json     # Generation statistics
└── pipeline_summary.json  # Execution summary
```

## Testing and Quality

### Test Structure
- Comprehensive test suite in `tests/` directory
- Component-specific tests for each core module
- Integration tests for complete pipeline workflows
- Performance and quality metric validation

### Development Best Practices
- Type hints throughout the codebase
- Dataclass-based configuration for type safety
- Extensive logging for debugging and monitoring
- Modular architecture enables isolated component testing

## Advanced Features

### Augmentoolkit Integration
- **Quality-First Generation**: Multi-stage validation (question quality, answer relevancy, accuracy)
- **MLX Optimization**: Native Apple Silicon support with optimized models
- **Domain Expertise**: Telecommunications-specific prompts and context
- **Scalable Processing**: High concurrency (8 parallel) with no subset limitations
- **Enterprise Ready**: Production-grade dataset generation with comprehensive logging

### MLX Fine-tuning (Apple Silicon)
- **Native Optimization**: Hardware-accelerated training on M1/M2/M3 processors
- **Model Support**: Optimized for `mlx-community/Llama-3.2-3B-Instruct-4bit`
- **LoRA Adaptation**: Efficient parameter adaptation with model fusing
- **Interactive Chat**: Built-in chat interface for testing fine-tuned models
- **Performance Tuned**: Optimized batch sizes and learning rates for Apple Silicon

### Dataset Generation Options
- **Basic Mode**: Standard RAG datasets (JSON/JSONL/CSV) for vector databases
- **Advanced Mode**: Augmentoolkit with multi-turn conversations and validation
- **MLX Format**: ChatML format ready for immediate fine-tuning
- **Quality Assurance**: Configurable validation pipeline with detailed metrics

### Dependencies and Installation
- **Core**: `beautifulsoup4`, `lxml`, `html2text`, `tiktoken`, `pyyaml`
- **Local Processing**: `mlx`, `mlx-lm` (Apple Silicon REQUIRED for dataset generation)
- **Enhanced Processing**: `docling` (optional, for advanced PDF/HTML processing)
- **No External APIs**: Complete local-only operation with no internet dependencies

## Quick Reference

### Essential Commands
```bash
# Setup
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt

# Basic Pipeline
python src/run_flow4.py --verbose pipeline --input data/ --output-dir output

# Advanced Pipeline (with LOCAL MLX)
python src/run_flow4.py --verbose pipeline --input data/ --output-dir output --use-augmentoolkit --augmentoolkit-config configs/flow4_local_mlx.yaml

# Fine-tune Model (Apple Silicon)
python src/run_flow4.py finetune --dataset output/augmentoolkit/mlx_dataset.jsonl --chat
```

### Available Commands
- **`pipeline`**: Complete document processing with optional Augmentoolkit integration
- **`convert`**: Document conversion only (HTML/PDF → Markdown)
- **`chunk`**: Intelligent document chunking with semantic awareness
- **`generate`**: Advanced dataset generation using Augmentoolkit
- **`finetune`**: MLX fine-tuning with LoRA adaptation (Apple Silicon)

### Key Files
- **`configs/flow4_local_mlx.yaml`**: LOCAL-ONLY MLX configuration (no external APIs)
- **`src/run_flow4.py`**: Main CLI entry point
- **`MINIMAL_SETUP.md`**: Local-only setup instructions
- **`.gitignore`**: Configured for ML workflows (models, outputs, cache)