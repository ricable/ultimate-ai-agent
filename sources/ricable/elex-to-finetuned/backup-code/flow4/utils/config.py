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
    
    # Advanced table processing
    table_mode: str = "accurate"  # "fast" or "accurate"
    table_cell_matching: bool = True
    table_export_formats: List[str] = field(default_factory=lambda: ["csv", "json"])
    extract_table_metadata: bool = True
    match_table_captions: bool = True
    
    # Advanced image processing
    enable_vlm: bool = True
    image_classification: bool = True
    generate_image_descriptions: bool = True
    extract_image_metadata: bool = True
    match_figure_captions: bool = True
    image_scale_factor: float = 2.0
    image_export_formats: List[str] = field(default_factory=lambda: ["png", "json"])
    
    # Multimodal training data
    generate_multimodal_data: bool = False
    multimodal_export_format: str = "jsonl"  # "jsonl" or "json"
    include_vision_embeddings: bool = False
    
    # Remote services and caching
    enable_remote_services: bool = False
    artifacts_path: Optional[str] = None
    cache_models: bool = True
    
    # Enhanced multimodal options
    table_format: str = "csv"  # csv, json, html, markdown
    image_format: str = "png"  # png, jpg, jpeg
    generate_image_descriptions: bool = False  # Generate AI descriptions for images
    image_scale: float = 2.0  # Image scaling factor for high-resolution extraction
    enable_picture_classification: bool = True  # Classify image types
    table_cell_matching: bool = True  # Enable precise table cell matching
    extract_table_structure: bool = True  # Extract table structure information
    preserve_image_metadata: bool = True  # Keep image metadata during extraction
    
    # Enhanced chunking options
    chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_size: int = 600  # Improved minimum for better semantic coherence
    max_chunk_size: int = 2000  # Increased maximum for complete sections
    split_on_headings: bool = True
    keep_headings: bool = True
    respect_document_structure: bool = True
    tokenizer: str = "cl100k_base"
    
    # Advanced tokenization configuration
    tokenizer_type: str = "auto"  # auto, huggingface, openai, simple
    tokenizer_model: Optional[str] = None  # Specific tokenizer model
    embedding_model: Optional[str] = None  # Embedding model to align with
    max_tokens: int = 500  # Maximum tokens per chunk
    overlap_tokens: int = 50  # Token overlap between chunks
    cache_tokenizers: bool = True  # Cache tokenizer instances
    enable_token_aware_splitting: bool = True  # Use token-aware splitting
    enable_merge_peers: bool = True  # Merge undersized chunks
    min_tokens: int = 100  # Minimum tokens per chunk
    merge_threshold: float = 0.7  # Merge if chunk is < threshold * max_tokens
    
    # Semantic chunking options
    enable_semantic_chunking: bool = True
    preserve_code_blocks: bool = True
    preserve_class_definitions: bool = True
    preserve_enum_definitions: bool = True
    technical_content_boost: bool = True
    quality_threshold: float = 0.5  # Minimum quality score for chunks
    
    # Content type priorities (higher = more likely to be kept together)
    content_type_priorities: Dict[str, int] = field(default_factory=lambda: {
        'class_definition': 10,
        'enum_definition': 10,
        'kpi_calculation': 9,
        'procedure': 8,
        'configuration': 7,
        'feature_description': 6,
        'kpi_definition': 6,
        'general_content': 5
    })
    
    # Advanced multimodal chunking options
    include_images_in_chunks: bool = False  # Include image references in text chunks
    include_tables_in_chunks: bool = True  # Include table data in text chunks
    multimodal_chunk_strategy: str = "separate"  # separate, inline, or mixed
    
    # Text cleaning options
    clean_malformed_tables: bool = True
    fix_broken_formatting: bool = True
    normalize_whitespace: bool = True
    remove_redundant_metadata: bool = False  # Keep metadata for technical docs


@dataclass
class TimeoutConfig:
    """Configuration for timeout handling across Flow4 operations."""
    
    # Base timeout settings (in seconds)
    default_operation_timeout: int = 1800  # 30 minutes
    
    # File processing timeouts
    file_conversion_timeout: int = 600     # 10 minutes per file
    file_extraction_timeout: int = 300     # 5 minutes per file
    
    # Dataset generation timeouts
    dataset_generation_timeout: Optional[int] = None  # No timeout (None = unlimited)
    augmentoolkit_timeout: Optional[int] = None       # No timeout
    
    # MLX fine-tuning timeouts  
    mlx_training_timeout: Optional[int] = None        # No timeout for training
    mlx_model_fusing_timeout: int = 3600             # 1 hour for model fusing
    mlx_generation_timeout: int = 600                # 10 minutes for generation
    mlx_chat_response_timeout: int = 300             # 5 minutes per chat response
    
    # Package installation timeouts
    package_install_timeout: int = 1800              # 30 minutes for installation
    
    # Enable dynamic timeout calculation
    enable_dynamic_timeouts: bool = True
    
    # Timeout scaling factors for dynamic calculation
    dataset_size_scaling_factor: float = 1.0         # Scale with dataset size
    model_size_scaling_factor: float = 1.0           # Scale with model complexity
    hardware_scaling_factor: float = 1.0             # Scale with hardware capabilities


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
    """Configuration for MLX fine-tuning with timeout support."""
    
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
    auto_optimize_m3: bool = True  # Auto-optimize for M3 Max hardware
    
    # Timeout configuration
    timeouts: TimeoutConfig = field(default_factory=TimeoutConfig)
    
    # Hardware optimization (M3 Max specific)
    m3_max_settings: Dict[str, Any] = field(default_factory=lambda: {
        "small": {"batch_size": 1, "learning_rate": 5e-5, "num_iters": 200},
        "medium": {"batch_size": 2, "learning_rate": 2e-5, "num_iters": 500},
        "large": {"batch_size": 8, "learning_rate": 1e-5, "num_iters": 300},
        "max": {"batch_size": 24, "learning_rate": 1e-5, "num_iters": 1000}
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
    
    # Enhanced multimodal environment variables
    if os.getenv("FLOW4_TABLE_FORMAT"):
        config.docling.table_format = os.getenv("FLOW4_TABLE_FORMAT")
    
    if os.getenv("FLOW4_IMAGE_FORMAT"):
        config.docling.image_format = os.getenv("FLOW4_IMAGE_FORMAT")
    
    if os.getenv("FLOW4_IMAGE_DESCRIPTIONS"):
        config.docling.generate_image_descriptions = os.getenv("FLOW4_IMAGE_DESCRIPTIONS").lower() == "true"
    
    if os.getenv("FLOW4_IMAGE_SCALE"):
        config.docling.image_scale = float(os.getenv("FLOW4_IMAGE_SCALE"))
    
    if os.getenv("FLOW4_MULTIMODAL_STRATEGY"):
        config.docling.multimodal_chunk_strategy = os.getenv("FLOW4_MULTIMODAL_STRATEGY")
    
    # Advanced tokenization environment variables
    if os.getenv("FLOW4_TOKENIZER_TYPE"):
        config.docling.tokenizer_type = os.getenv("FLOW4_TOKENIZER_TYPE")
    
    if os.getenv("FLOW4_TOKENIZER_MODEL"):
        config.docling.tokenizer_model = os.getenv("FLOW4_TOKENIZER_MODEL")
    
    if os.getenv("FLOW4_EMBEDDING_MODEL"):
        config.docling.embedding_model = os.getenv("FLOW4_EMBEDDING_MODEL")
    
    if os.getenv("FLOW4_MAX_TOKENS"):
        config.docling.max_tokens = int(os.getenv("FLOW4_MAX_TOKENS"))
    
    if os.getenv("FLOW4_MIN_TOKENS"):
        config.docling.min_tokens = int(os.getenv("FLOW4_MIN_TOKENS"))
    
    if os.getenv("FLOW4_OVERLAP_TOKENS"):
        config.docling.overlap_tokens = int(os.getenv("FLOW4_OVERLAP_TOKENS"))
    
    if os.getenv("FLOW4_DISABLE_TOKEN_SPLITTING"):
        config.docling.enable_token_aware_splitting = False
    
    if os.getenv("FLOW4_DISABLE_MERGE_PEERS"):
        config.docling.enable_merge_peers = False
    
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
    
    # Validate multimodal options
    valid_table_formats = ["csv", "json", "html", "markdown"]
    if config.docling.table_format not in valid_table_formats:
        errors.append(f"Table format must be one of: {', '.join(valid_table_formats)}")
    
    valid_image_formats = ["png", "jpg", "jpeg"]
    if config.docling.image_format not in valid_image_formats:
        errors.append(f"Image format must be one of: {', '.join(valid_image_formats)}")
    
    if config.docling.image_scale <= 0:
        errors.append("Image scale must be greater than 0")
    
    valid_chunk_strategies = ["separate", "inline", "mixed"]
    if config.docling.multimodal_chunk_strategy not in valid_chunk_strategies:
        errors.append(f"Multimodal chunk strategy must be one of: {', '.join(valid_chunk_strategies)}")
    
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