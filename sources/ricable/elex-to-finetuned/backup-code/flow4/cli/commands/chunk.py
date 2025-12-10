"""Advanced chunk command for chunking documents using optimized strategies."""

import argparse
import glob
import os
import json
from typing import List

from ...core.chunker import OptimizedDocumentChunker
# Try to import serialization with graceful fallback
try:
    from ...core.serialization import SerializationConfig
    HAS_SERIALIZATION = True
except ImportError:
    SerializationConfig = None
    HAS_SERIALIZATION = False
from ...utils.config import DoclingConfig
from ...utils.logging import get_logger

logger = get_logger(__name__)


class ChunkCommand:
    """Advanced command to chunk documents using optimized Docling strategies with serialization."""
    
    def add_parser(self, subparsers, name: str) -> argparse.ArgumentParser:
        """Add parser for chunk command.
        
        Args:
            subparsers: Subparser object to add to
            name: Name of the command
            
        Returns:
            Created parser
        """
        parser = subparsers.add_parser(
            name,
            help='Chunk Markdown documents using advanced optimized strategies',
            description="""
Chunk Markdown documents into smaller pieces using advanced optimization features.

This command uses state-of-the-art chunking strategies:
- Optimized Docling hybrid chunking with custom tokenizers
- Advanced serialization for tables, images, and multimodal content
- Semantic-aware chunking with content type classification
- High-performance caching and parallel processing
- Enhanced quality scoring and relationship tracking

The output includes optimized chunks with rich metadata, enhanced RAG datasets,
and comprehensive serialization support for complex documents.
""",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Input/Output options
        io_group = parser.add_argument_group('Input/Output')
        io_group.add_argument(
            '--input', '-i',
            required=True,
            help='Input Markdown file or directory containing Markdown files'
        )
        io_group.add_argument(
            '--output-dir', '-o',
            required=True,
            help='Output directory for chunks'
        )
        io_group.add_argument(
            '--pattern',
            default='*.md',
            help='File pattern to match (default: *.md)'
        )
        
        # Chunking options
        chunk_group = parser.add_argument_group('Chunking Configuration')
        chunk_group.add_argument(
            '--size', '--chunk-size',
            type=int,
            default=900,
            help='Target chunk size in tokens (default: 900, optimized based on validation findings)'
        )
        chunk_group.add_argument(
            '--overlap', '--chunk-overlap',
            type=int,
            default=100,
            help='Overlap between chunks in tokens (default: 100, optimal for semantic boundaries)'
        )
        chunk_group.add_argument(
            '--min-size',
            type=int,
            default=600,
            help='Minimum chunk size in tokens (default: 600, validated optimal range)'
        )
        chunk_group.add_argument(
            '--max-size',
            type=int,
            default=1500,
            help='Maximum chunk size in tokens (default: 1500, preserves complete concepts)'
        )
        chunk_group.add_argument(
            '--tokenizer',
            default='sentence-transformers/all-MiniLM-L6-v2',
            help='Tokenizer model to use (default: sentence-transformers/all-MiniLM-L6-v2)'
        )
        chunk_group.add_argument(
            '--hf-tokenizer',
            help='HuggingFace tokenizer model (overrides --tokenizer for HF models)'
        )
        chunk_group.add_argument(
            '--openai-tokenizer',
            help='OpenAI tokenizer encoding (e.g., cl100k_base, p50k_base)'
        )
        chunk_group.add_argument(
            '--quality-threshold',
            type=float,
            default=0.5,
            help='Minimum quality score for chunks (default: 0.5)'
        )
        chunk_group.add_argument(
            '--merge-threshold',
            type=float,
            default=0.7,
            help='Merge threshold for undersized chunks (default: 0.7)'
        )
        chunk_group.add_argument(
            '--no-semantic-chunking',
            action='store_true',
            help='Disable semantic-aware chunking'
        )
        chunk_group.add_argument(
            '--no-preserve-code',
            action='store_true',
            help='Disable code block preservation'
        )
        
        # Processing options
        proc_group = parser.add_argument_group('Processing Options')
        proc_group.add_argument(
            '--no-docling',
            action='store_true',
            help='Skip Docling and use fallback chunking methods (recommended if encountering API issues)'
        )
        proc_group.add_argument(
            '--no-headings',
            action='store_true',
            help='Disable splitting on headings'
        )
        proc_group.add_argument(
            '--no-structure',
            action='store_true',
            help='Disable respecting document structure'
        )
        proc_group.add_argument(
            '--workers',
            type=int,
            default=2,
            help='Number of parallel workers (default: 2, conservative for stability)'
        )
        proc_group.add_argument(
            '--batch-size',
            type=int,
            default=5,
            help='Batch size for processing multiple files (default: 5, optimized for memory)'
        )
        proc_group.add_argument(
            '--memory-limit',
            type=int,
            default=1024,
            help='Memory limit in MB for resource management (default: 1024, stable processing)'
        )
        proc_group.add_argument(
            '--disable-caching',
            action='store_true',
            help='Disable intelligent caching for converters and tokenizers'
        )
        proc_group.add_argument(
            '--disable-advanced-tokenizers',
            action='store_true',
            help='Disable advanced tokenizer support (HuggingFace/OpenAI)'
        )
        
        # Advanced Processing Options
        advanced_group = parser.add_argument_group('Advanced Processing Options')
        advanced_group.add_argument(
            '--table-mode',
            choices=['fast', 'accurate'],
            default='accurate',
            help='Table processing mode (default: accurate)'
        )
        advanced_group.add_argument(
            '--no-multimodal',
            action='store_true',
            help='Disable multimodal processing'
        )
        advanced_group.add_argument(
            '--no-ocr',
            action='store_true',
            help='Disable OCR processing'
        )
        advanced_group.add_argument(
            '--threads',
            type=int,
            default=8,
            help='Number of processing threads (default: 8)'
        )
        
        # Serialization options
        serial_group = parser.add_argument_group('Serialization Options')
        serial_group.add_argument(
            '--table-format',
            choices=['markdown', 'csv', 'json', 'html'],
            default='csv',
            help='Table serialization format (default: csv)'
        )
        serial_group.add_argument(
            '--image-format',
            choices=['png', 'jpg', 'jpeg'],
            default='png',
            help='Image export format (default: png)'
        )
        serial_group.add_argument(
            '--enhance-serialization',
            action='store_true',
            help='Enable enhanced serialization with multimodal support'
        )
        serial_group.add_argument(
            '--include-table-captions',
            action='store_true',
            default=True,
            help='Include table captions in serialization (default: True)'
        )
        serial_group.add_argument(
            '--include-picture-descriptions',
            action='store_true',
            default=True,
            help='Include picture descriptions in serialization (default: True)'
        )
        
        # Output options
        output_group = parser.add_argument_group('Output Options')
        output_group.add_argument(
            '--create-rag-dataset',
            action='store_true',
            default=True,
            help='Create RAG dataset file (default: True)'
        )
        output_group.add_argument(
            '--enhanced-rag-datasets',
            action='store_true',
            help='Create enhanced RAG datasets with deduplication and multimodal support'
        )
        output_group.add_argument(
            '--no-individual-files',
            action='store_true',
            help='Skip creating individual chunk files'
        )
        output_group.add_argument(
            '--enable-deduplication',
            action='store_true',
            help='Enable dataset deduplication for enhanced datasets'
        )
        output_group.add_argument(
            '--chunk-quality-analysis',
            action='store_true',
            help='Generate comprehensive chunk quality analysis report'
        )
        
        return parser
    
    def _find_markdown_files(self, input_path: str, pattern: str) -> List[str]:
        """Find markdown files to process.
        
        Args:
            input_path: Input file or directory path
            pattern: File pattern to match
            
        Returns:
            List of markdown file paths
        """
        if os.path.isfile(input_path):
            return [input_path]
        
        if not os.path.isdir(input_path):
            raise ValueError(f"Input path does not exist: {input_path}")
        
        markdown_files = glob.glob(os.path.join(input_path, "**", pattern), recursive=True)
        return markdown_files
    
    def run(self, args: argparse.Namespace) -> int:
        """Run the chunk command.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            # Create chunking configuration with advanced options
            config = DoclingConfig(
                # Core chunking parameters
                chunk_size=args.size,
                chunk_overlap=args.overlap,
                min_chunk_size=args.min_size,
                max_chunk_size=args.max_size,
                
                # Structural processing
                split_on_headings=not args.no_headings,
                keep_headings=True,
                respect_document_structure=not args.no_structure,
                
                # Tokenization
                tokenizer=args.openai_tokenizer or args.tokenizer,
                
                # Content extraction
                extract_tables=True,
                extract_figures=True,
                
                # Performance
                with_accelerator=True
            )
            
            # Add advanced options if supported by the config class
            if hasattr(config, 'multimodal'):
                config.multimodal = not args.no_multimodal
            if hasattr(config, 'do_ocr'):
                config.do_ocr = not args.no_ocr
            if hasattr(config, 'num_threads'):
                config.num_threads = args.threads
            if hasattr(config, 'table_mode'):
                config.table_mode = args.table_mode
            if hasattr(config, 'enable_semantic_chunking'):
                config.enable_semantic_chunking = not args.no_semantic_chunking
            if hasattr(config, 'preserve_code_blocks'):
                config.preserve_code_blocks = not args.no_preserve_code
            if hasattr(config, 'quality_threshold'):
                config.quality_threshold = args.quality_threshold
            if hasattr(config, 'merge_threshold'):
                config.merge_threshold = args.merge_threshold
            if hasattr(config, 'table_format'):
                config.table_format = args.table_format
            if hasattr(config, 'image_format'):
                config.image_format = args.image_format
            
            # Add HuggingFace tokenizer if specified
            if args.hf_tokenizer:
                config.hf_tokenizer_model = args.hf_tokenizer
            
            # Create serialization configuration if available
            serialization_config = None
            if HAS_SERIALIZATION and SerializationConfig:
                serialization_config = SerializationConfig(
                    table_format=args.table_format,
                    picture_format='annotation',  # Default to annotation format
                    include_table_captions=True,  # Always include table captions
                    include_picture_descriptions=True,  # Always include picture descriptions
                    preserve_markup=True,
                    include_source_metadata=True
                )
            elif args.enhance_serialization:
                logger.warning("Serialization framework not available, enhanced serialization disabled")
            
            # Find input files
            try:
                input_files = self._find_markdown_files(args.input, args.pattern)
            except ValueError as e:
                logger.error(str(e))
                return 1
            
            if not input_files:
                logger.error("No Markdown files found")
                return 1
            
            logger.info(f"Found {len(input_files)} Markdown files to chunk")
            for i, file in enumerate(input_files[:5]):  # Log first 5 files
                logger.info(f"  File {i+1}: {file}")
            if len(input_files) > 5:
                logger.info(f"  ... and {len(input_files) - 5} more files")
            
            # Log configuration
            logger.info("\\n=== Enhanced Chunking Configuration ===")
            logger.info(f"  Chunk size: {config.chunk_size} tokens")
            logger.info(f"  Overlap: {config.chunk_overlap} tokens")
            logger.info(f"  Size range: {config.min_chunk_size}-{config.max_chunk_size} tokens")
            logger.info(f"  Split on headings: {config.split_on_headings}")
            logger.info(f"  Respect structure: {config.respect_document_structure}")
            logger.info(f"  Tokenizer: {config.tokenizer}")
            logger.info(f"  Use Docling: {not args.no_docling}")
            # Log advanced features if available
            if hasattr(config, 'enable_semantic_chunking'):
                logger.info(f"  Semantic chunking: {config.enable_semantic_chunking}")
            if hasattr(config, 'preserve_code_blocks'):
                logger.info(f"  Preserve code blocks: {config.preserve_code_blocks}")
            if hasattr(config, 'quality_threshold'):
                logger.info(f"  Quality threshold: {config.quality_threshold}")
            if hasattr(config, 'table_mode'):
                logger.info(f"  Table mode: {config.table_mode}")
            if hasattr(config, 'multimodal'):
                logger.info(f"  Multimodal: {config.multimodal}")
            if hasattr(config, 'do_ocr'):
                logger.info(f"  OCR enabled: {config.do_ocr}")
            if hasattr(config, 'num_threads'):
                logger.info(f"  Processing threads: {config.num_threads}")
            if hasattr(config, 'table_format'):
                logger.info(f"  Table format: {config.table_format}")
            if hasattr(config, 'image_format'):
                logger.info(f"  Image format: {config.image_format}")
            
            # Create optimized chunker with advanced features
            logger.info("Creating OptimizedDocumentChunker...")
            chunker = OptimizedDocumentChunker(
                config=config,
                max_workers=args.workers,
                enable_caching=not args.disable_caching,
                memory_limit_mb=args.memory_limit,
                enable_advanced_tokenizers=not args.disable_advanced_tokenizers
            )
            logger.info("Chunker created successfully")
            
            # Note: Serialization framework integration could be added here in the future
            if args.enhance_serialization and not HAS_SERIALIZATION:
                logger.warning("Enhanced serialization requested but serialization framework not available")
            
            # Process files
            logger.info("Starting file processing...")
            all_chunks = []
            
            if len(input_files) == 1:
                # Single file - process directly
                logger.info(f"Processing single file: {input_files[0]}")
                chunks = chunker.chunk_file(input_files[0], use_docling=not args.no_docling)
                logger.info(f"Created {len(chunks)} chunks from single file")
                all_chunks.extend(chunks)
                
                # Save chunks if requested
                if not args.no_individual_files:
                    logger.info("Saving individual chunk files...")
                    chunker.save_chunks(chunks, args.output_dir)
                    logger.info("Individual chunk files saved")
            else:
                # Multiple files - use optimized batch processing
                logger.info(f"Processing {len(input_files)} files in batch mode...")
                chunk_output_dir = args.output_dir if not args.no_individual_files else None
                all_chunks = chunker.batch_chunk(
                    input_files,
                    chunk_output_dir,
                    num_workers=args.workers,
                    use_docling=not args.no_docling,
                    batch_size=args.batch_size
                )
                logger.info(f"Batch processing completed. Created {len(all_chunks)} total chunks")
            
            # Create RAG datasets
            os.makedirs(args.output_dir, exist_ok=True)
            
            if args.enhanced_rag_datasets:
                # Create enhanced RAG datasets with deduplication and multimodal support
                multimodal_metadata = {
                    "serialization_config": serialization_config.__dict__ if (serialization_config and args.enhance_serialization) else None,
                    "processing_options": {
                        "workers": args.workers,
                        "batch_size": args.batch_size,
                        "memory_limit_mb": args.memory_limit,
                        "caching_enabled": not args.disable_caching,
                        "advanced_tokenizers": not args.disable_advanced_tokenizers
                    }
                }
                
                if hasattr(chunker, 'create_enhanced_rag_dataset'):
                    dataset_files = chunker.create_enhanced_rag_dataset(
                        all_chunks,
                        args.output_dir,
                        enable_deduplication=args.enable_deduplication,
                        multimodal_metadata=multimodal_metadata
                    )
                    logger.info(f"Enhanced RAG datasets created: {list(dataset_files.keys())}")
                    for format_name, file_path in dataset_files.items():
                        logger.info(f"  {format_name}: {file_path}")
                else:
                    logger.warning("Enhanced RAG dataset creation not available, using standard method")
                    rag_dataset = chunker.create_rag_dataset(all_chunks)
                    rag_dataset_path = os.path.join(args.output_dir, "rag_dataset.json")
                    with open(rag_dataset_path, 'w', encoding='utf-8') as f:
                        json.dump(rag_dataset, f, indent=2)
                    logger.info(f"Standard RAG dataset saved to {rag_dataset_path}")
            
            elif args.create_rag_dataset:
                # Create standard RAG dataset
                rag_dataset = chunker.create_rag_dataset(all_chunks)
                rag_dataset_path = os.path.join(args.output_dir, "rag_dataset.json")
                with open(rag_dataset_path, 'w', encoding='utf-8') as f:
                    json.dump(rag_dataset, f, indent=2)
                logger.info(f"Standard RAG dataset saved to {rag_dataset_path}")
            
            # Generate comprehensive reports
            logger.info("\n=== Advanced Chunking Summary ===")
            logger.info(f"  Input files: {len(input_files)}")
            logger.info(f"  Total chunks created: {len(all_chunks)}")
            
            if all_chunks:
                # Calculate enhanced statistics
                chunk_sizes = [chunk.metadata.get('tokens', len(chunk.text.split())) for chunk in all_chunks]
                quality_scores = [chunk.metadata.get('quality_score', 0.0) for chunk in all_chunks]
                
                avg_size = sum(chunk_sizes) / len(chunk_sizes)
                min_size = min(chunk_sizes)
                max_size = max(chunk_sizes)
                avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
                
                logger.info(f"  Average chunk size: {avg_size:.1f} tokens")
                logger.info(f"  Size range: {min_size}-{max_size} tokens")
                logger.info(f"  Average quality score: {avg_quality:.3f}")
                
                # Count chunking methods used
                methods = {}
                content_types = {}
                for chunk in all_chunks:
                    method = chunk.metadata.get('chunking_method', 'unknown')
                    methods[method] = methods.get(method, 0) + 1
                    
                    content_type = chunk.metadata.get('content_type', 'general')
                    content_types[content_type] = content_types.get(content_type, 0) + 1
                
                logger.info("  Chunking methods used:")
                for method, count in methods.items():
                    logger.info(f"    {method}: {count} chunks")
                
                logger.info("  Content types detected:")
                for content_type, count in content_types.items():
                    logger.info(f"    {content_type}: {count} chunks")
                
                # Performance metrics
                if hasattr(chunker, 'get_performance_metrics'):
                    metrics = chunker.get_performance_metrics()
                    logger.info(f"  Processing performance: {metrics.chunks_per_second:.2f} chunks/second")
                    if hasattr(metrics, 'cache_hit_ratio'):
                        logger.info(f"  Cache hit ratio: {metrics.cache_hit_ratio:.2f}")
                    if metrics.errors > 0:
                        logger.warning(f"  Processing errors: {metrics.errors}")
            
            # Generate chunk quality analysis if requested
            if args.chunk_quality_analysis and hasattr(chunker, 'analyze_chunk_quality'):
                logger.info("\n=== Chunk Quality Analysis ===")
                quality_analysis = chunker.analyze_chunk_quality(all_chunks)
                
                if 'quality_metrics' in quality_analysis:
                    qm = quality_analysis['quality_metrics']
                    logger.info(f"  Quality score range: {qm['min_quality_score']:.3f} - {qm['max_quality_score']:.3f}")
                    logger.info(f"  High quality chunks: {qm['quality_distribution']['high_quality']}")
                    logger.info(f"  Medium quality chunks: {qm['quality_distribution']['medium_quality']}")
                    logger.info(f"  Low quality chunks: {qm['quality_distribution']['low_quality']}")
                
                if 'recommendations' in quality_analysis:
                    logger.info("  Recommendations:")
                    for rec in quality_analysis['recommendations']:
                        logger.info(f"    - {rec}")
                
                # Save quality analysis report
                quality_report_path = os.path.join(args.output_dir, "chunk_quality_analysis.json")
                with open(quality_report_path, 'w', encoding='utf-8') as f:
                    json.dump(quality_analysis, f, indent=2)
                logger.info(f"  Quality analysis saved to {quality_report_path}")
            
            # Cleanup resources
            if hasattr(chunker, 'cleanup'):
                chunker.cleanup()
            
            if len(all_chunks) == 0:
                logger.error("No chunks were created")
                return 1
            else:
                logger.info("\n✅ Advanced chunking completed successfully")
                return 0
        
        except Exception as e:
            logger.error(f"Chunking failed: {str(e)}")
            return 1