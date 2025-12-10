"""Configuration management for Flow4."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import os
from pathlib import Path


@dataclass
class DoclingConfig:
    """Configuration for Docling document processing."""
    
    # Processing options
    with_accelerator: bool = True
    extract_tables: bool = True
    extract_figures: bool = True
    custom_convert: bool = True
    multimodal: bool = True
    do_ocr: bool = True
    num_threads: int = 8
    
    # Enhanced chunking options
    chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    split_on_headings: bool = True
    keep_headings: bool = True
    respect_document_structure: bool = True
    tokenizer: str = "cl100k_base"
    
    # Advanced tokenization configuration
    tokenizer_type: str = "auto"  # auto, huggingface, openai, simple
    tokenizer_model: Optional[str] = None
    embedding_model: Optional[str] = None
    max_tokens: int = 500
    overlap_tokens: int = 50
    cache_tokenizers: bool = True
    enable_token_aware_splitting: bool = True
    enable_merge_peers: bool = True
    min_tokens: int = 100
    merge_threshold: float = 0.7
    
    # Semantic chunking options
    enable_semantic_chunking: bool = True
    preserve_code_blocks: bool = True
    preserve_class_definitions: bool = True
    preserve_enum_definitions: bool = True
    technical_content_boost: bool = True
    quality_threshold: float = 0.5
    
    # Text cleaning options
    clean_malformed_tables: bool = True
    fix_broken_formatting: bool = True
    normalize_whitespace: bool = True
    remove_redundant_metadata: bool = False


@dataclass
class TimeoutConfig:
    """Configuration for timeout handling across Flow4 operations."""
    
    # Base timeout settings (in seconds)
    default_operation_timeout: int = 1800  # 30 minutes
    
    # File processing timeouts
    file_conversion_timeout: int = 600     # 10 minutes per file
    file_extraction_timeout: int = 300     # 5 minutes per file
    
    # Dataset generation timeouts
    dataset_generation_timeout: Optional[int] = None  # No timeout
    
    # MLX fine-tuning timeouts  
    mlx_training_timeout: Optional[int] = None        # No timeout for training
    mlx_model_fusing_timeout: int = 3600             # 1 hour for model fusing
    mlx_generation_timeout: int = 600                # 10 minutes for generation
    mlx_chat_response_timeout: int = 300             # 5 minutes per chat response
    
    # Package installation timeouts
    package_install_timeout: int = 1800              # 30 minutes for installation


@dataclass
class PipelineConfig:
    """Configuration for the document processing pipeline."""
    
    # Input/Output paths
    input_path: Optional[str] = None
    output_dir: str = "output"
    extract_dir: Optional[str] = None
    
    # Processing options
    pattern_exclude: Optional[str] = None
    max_files: Optional[int] = None
    num_workers: int = 4
    disable_filtering: bool = False
    
    # File patterns
    html_pattern: str = "*.html"
    pdf_pattern: str = "*.pdf"
    
    # Docling configuration
    docling: DoclingConfig = field(default_factory=DoclingConfig)
    
    # Timeout configuration
    timeouts: TimeoutConfig = field(default_factory=TimeoutConfig)
    
    # Output structure
    markdown_subdir: str = "markdown"
    combined_subdir: str = "combined"
    chunks_subdir: str = "chunks"
    rag_subdir: str = "rag"
    extracted_subdir: str = "extracted"


@dataclass
class MLXConfig:
    """Configuration for MLX fine-tuning."""
    
    # Model configuration
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_size: str = "medium"  # small, medium, large, max
    
    # Training parameters
    num_iters: int = 500
    batch_size: int = 16
    learning_rate: float = 2e-5
    steps_per_eval: int = 200
    num_layers: int = 16
    
    # LoRA parameters
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    
    # Dataset configuration
    dataset_path: str = "output/rag/finetune_dataset.jsonl"
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Output configuration
    adapter_dir: str = "fine_tuned_adapters"
    fused_model_dir: str = "fused_model"
    
    # Workflow options
    fuse_model: bool = True
    enable_chat: bool = False
    auto_optimize_m3: bool = True
    
    # Timeout configuration
    timeouts: TimeoutConfig = field(default_factory=TimeoutConfig)
    
    # Hardware optimization (M3 Max specific)
    m3_max_settings: Dict[str, Any] = field(default_factory=lambda: {
        "small": {"batch_size": 1, "learning_rate": 5e-5, "num_iters": 200},
        "medium": {"batch_size": 1, "learning_rate": 2e-5, "num_iters": 500},  # Reduced batch size for memory
        "large": {"batch_size": 2, "learning_rate": 1e-5, "num_iters": 300},   # Reduced batch size for memory
        "max": {"batch_size": 4, "learning_rate": 1e-5, "num_iters": 1000}     # Reduced batch size for memory
    })


@dataclass
class CLIConfig:
    """Configuration for CLI options."""
    
    verbose: bool = False
    debug: bool = False
    log_file: Optional[str] = None
    quiet: bool = False


def get_default_config() -> PipelineConfig:
    """Get default pipeline configuration."""
    return PipelineConfig()


def load_config_from_env() -> PipelineConfig:
    """Load configuration from environment variables."""
    config = get_default_config()
    
    # Pipeline settings
    if os.getenv("FLOW4_OUTPUT_DIR"):
        config.output_dir = os.getenv("FLOW4_OUTPUT_DIR")
    
    if os.getenv("FLOW4_NUM_WORKERS"):
        config.num_workers = int(os.getenv("FLOW4_NUM_WORKERS"))
    
    if os.getenv("FLOW4_MAX_FILES"):
        config.max_files = int(os.getenv("FLOW4_MAX_FILES"))
    
    # Docling settings
    if os.getenv("FLOW4_CHUNK_SIZE"):
        config.docling.chunk_size = int(os.getenv("FLOW4_CHUNK_SIZE"))
    
    if os.getenv("FLOW4_CHUNK_OVERLAP"):
        config.docling.chunk_overlap = int(os.getenv("FLOW4_CHUNK_OVERLAP"))
    
    if os.getenv("FLOW4_NO_ACCELERATOR"):
        config.docling.with_accelerator = False
    
    if os.getenv("FLOW4_NO_TABLES"):
        config.docling.extract_tables = False
    
    if os.getenv("FLOW4_NO_FIGURES"):
        config.docling.extract_figures = False
    
    return config


def validate_config(config: PipelineConfig) -> List[str]:
    """Validate configuration and return list of errors."""
    errors = []
    
    # Validate paths
    if config.input_path and not os.path.exists(config.input_path):
        errors.append(f"Input path does not exist: {config.input_path}")
    
    # Validate numeric values
    if config.num_workers < 1:
        errors.append("Number of workers must be at least 1")
    
    if config.docling.chunk_size < 1:
        errors.append("Chunk size must be at least 1")
    
    if config.docling.chunk_overlap < 0:
        errors.append("Chunk overlap cannot be negative")
    
    if config.docling.chunk_overlap >= config.docling.chunk_size:
        errors.append("Chunk overlap must be less than chunk size")
    
    return errors


def create_output_structure(config: PipelineConfig) -> Dict[str, Path]:
    """Create output directory structure and return paths."""
    base_path = Path(config.output_dir)
    
    paths = {
        "base": base_path,
        "markdown": base_path / config.markdown_subdir,
        "combined": base_path / config.combined_subdir,
        "chunks": base_path / config.chunks_subdir,
        "rag": base_path / config.rag_subdir,
        "extracted": base_path / config.extracted_subdir,
    }
    
    # Create directories
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return paths