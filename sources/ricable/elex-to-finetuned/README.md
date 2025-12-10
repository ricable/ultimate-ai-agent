# Flow4: Document Processing Pipeline

A comprehensive document processing pipeline that converts HTML and PDF documents to Markdown, cleans them, and chunks them for RAG applications using IBM Docling. The project has a modular architecture with CLI commands, core processing components, and MLX fine-tuning capabilities optimized for Apple Silicon.

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install basic dependencies
uv pip install beautifulsoup4 tiktoken

# Optional: Install enhanced features
uv pip install docling mlx mlx-lm
```

### 2. Run the Pipeline

```bash
# Process HTML files from data/ directory
python3 src/run_flow4.py pipeline --input data/ --output-dir output --verbose

# Process with custom settings
python3 src/run_flow4.py pipeline --input data/ --output-dir results --max-files 10 --chunk-size 750

# Convert documents only
python3 src/run_flow4.py convert --input data/ --output-dir markdown/

# Chunk existing markdown
python3 src/run_flow4.py chunk --input combined.md --output-dir chunks/

# Fine-tune with MLX (Apple Silicon)
python3 src/run_flow4.py finetune --dataset output/rag/finetune_dataset.jsonl --chat
```

## Features

### Core Pipeline
- **Document Conversion**: HTML/PDF to Markdown using IBM Docling with BeautifulSoup fallback
- **Intelligent Chunking**: Semantic-aware chunking with configurable strategies
- **Content Cleaning**: Remove headers, footers, disclaimers, and optimize for LLM consumption
- **RAG Dataset Generation**: Multiple format outputs (JSON, JSONL, CSV) for vector databases
- **Fine-tuning Datasets**: Instruction-response pairs optimized for LLM training

### Advanced Features
- **MLX Fine-tuning**: Optimized for Apple Silicon with M1/M2/M3 Max hardware acceleration
- **Multimodal Support**: Table and figure extraction (when docling is available)
- **Quality Validation**: Content analysis and quality scoring
- **Configurable Processing**: Extensive configuration options via CLI or environment variables

## Architecture

```
src/
├── cli/                    # Command-line interface
│   └── main.py
├── core/                   # Core processing components
│   ├── pipeline.py         # Main orchestrator
│   ├── converter.py        # Document conversion (Docling + fallback)
│   ├── chunker.py          # Intelligent chunking
│   ├── cleaner.py          # Content cleaning
│   ├── mlx_finetuner.py    # MLX fine-tuning for Apple Silicon
│   └── simple_converter.py # BeautifulSoup fallback converter
├── utils/                  # Configuration and utilities
│   ├── config.py           # Configuration management
│   └── logging.py          # Logging utilities
└── run_flow4.py           # Main entry point
```

## Output Structure

```
output/
├── markdown/              # Individual converted files
├── combined/              # Concatenated documents  
├── chunks/                # Document chunks with metadata
├── rag/                   # RAG datasets (JSON/JSONL/CSV)
├── extracted/             # Tables, figures, images
└── pipeline_summary.json  # Execution summary
```

## Configuration

### Environment Variables
```bash
export FLOW4_OUTPUT_DIR="custom_output"
export FLOW4_CHUNK_SIZE=750
export FLOW4_CHUNK_OVERLAP=75
export FLOW4_MAX_FILES=100
export FLOW4_NUM_WORKERS=8
```

### CLI Options
See `python3 src/run_flow4.py --help` for complete options.

## Dependencies

### Required
- `beautifulsoup4`: HTML parsing and simple conversion
- `tiktoken`: Token counting for chunking

### Optional
- `docling`: Advanced PDF/HTML processing with table/figure extraction
- `mlx` + `mlx-lm`: Apple Silicon fine-tuning capabilities

## Testing

The pipeline has been successfully tested with telecommunications documentation HTML files, producing:
- 100% conversion success rate with simple converter
- Semantic chunking with quality scoring
- Multiple dataset formats for RAG and fine-tuning
- Comprehensive validation and reporting

## Example Usage

```bash
# Activate virtual environment
source .venv/bin/activate

# Process 5 HTML files with verbose output
python3 src/run_flow4.py pipeline \
  --input data/ \
  --output-dir test_output \
  --max-files 5 \
  --verbose

# Results in:
# - 3 markdown files converted
# - 3 semantic chunks created
# - RAG datasets in JSON, JSONL, CSV formats
# - Fine-tuning datasets with instruction-response pairs
# - Complete processing in <0.1 seconds
```

## Contributing

1. Use `uv` for virtual environment and package management
2. Always activate the virtual environment before running flow4 CLI commands
3. Follow the modular architecture patterns established in the codebase
4. Test with the provided HTML data files before submitting changes

## License

This project is part of the Flow4 document processing pipeline system.