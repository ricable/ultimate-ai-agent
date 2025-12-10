"""Pipeline command for running the complete document processing pipeline."""

import argparse
import asyncio
from typing import Dict, Any, List

from ...core.pipeline import DocumentPipeline
# Legacy synthetic dataset removed - using Augmentoolkit only
from ...utils.config import PipelineConfig, DoclingConfig, load_config_from_env, validate_config
from ...utils.logging import get_logger

# Augmentoolkit will be run via subprocess to avoid import issues

logger = get_logger(__name__)


class PipelineCommand:
    """Command to run the complete document processing pipeline."""
    
    def add_parser(self, subparsers, name: str) -> argparse.ArgumentParser:
        """Add parser for pipeline command.
        
        Args:
            subparsers: Subparser object to add to
            name: Name of the command
            
        Returns:
            Created parser
        """
        parser = subparsers.add_parser(
            name,
            help='Run the complete document processing pipeline',
            description="""
Run the complete Flow4 document processing pipeline:
1. Extract files from ZIP (if input is ZIP)
2. Convert HTML/PDF files to Markdown using Docling with advanced multimodal features
3. Extract tables and images in configurable formats with AI descriptions
4. Concatenate Markdown files with multimodal content handling
5. Chunk documents using hybrid chunking with multimodal strategies
6. Create RAG datasets optimized for multimodal retrieval applications

This is the recommended way to process documents end-to-end with full
multimodal support including tables, images, and structured content.
""",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Input/Output options
        io_group = parser.add_argument_group('Input/Output')
        io_group.add_argument(
            '--input', '-i',
            required=True,
            help='Input ZIP file or directory containing documents'
        )
        io_group.add_argument(
            '--output-dir', '-o',
            default='output',
            help='Output directory for processed files (default: output)'
        )
        io_group.add_argument(
            '--extract-dir',
            help='Directory to extract ZIP contents (default: output/extracted)'
        )
        
        # Processing options
        proc_group = parser.add_argument_group('Processing')
        proc_group.add_argument(
            '--workers', '-w',
            type=int,
            default=4,
            help='Number of parallel workers (default: 4)'
        )
        proc_group.add_argument(
            '--max-files',
            type=int,
            help='Maximum number of files to process (process all if not specified)'
        )
        proc_group.add_argument(
            '--exclude',
            help='Pattern to exclude from processing (e.g., "test" to skip files containing "test")'
        )
        
        # Docling options
        docling_group = parser.add_argument_group('Docling Processing')
        docling_group.add_argument(
            '--no-accelerator',
            action='store_true',
            help='Disable GPU acceleration'
        )
        docling_group.add_argument(
            '--no-tables',
            action='store_true',
            help='Disable table extraction'
        )
        docling_group.add_argument(
            '--no-figures',
            action='store_true',
            help='Disable figure extraction'
        )
        docling_group.add_argument(
            '--no-multimodal',
            action='store_true',
            help='Disable multimodal output'
        )
        docling_group.add_argument(
            '--no-custom',
            action='store_true',
            help='Disable custom conversion rules'
        )
        docling_group.add_argument(
            '--threads',
            type=int,
            default=8,
            help='Number of threads for processing (default: 8)'
        )
        
        # Enhanced multimodal options
        multimodal_group = parser.add_argument_group('Advanced Multimodal Processing')
        multimodal_group.add_argument(
            '--table-format',
            choices=['csv', 'json', 'html', 'markdown'],
            default='csv',
            help='Format for extracted tables (default: csv)'
        )
        multimodal_group.add_argument(
            '--image-format',
            choices=['png', 'jpg', 'jpeg'],
            default='png',
            help='Format for extracted images (default: png)'
        )
        multimodal_group.add_argument(
            '--image-descriptions',
            action='store_true',
            help='Generate AI descriptions for extracted images'
        )
        multimodal_group.add_argument(
            '--image-scale',
            type=float,
            default=2.0,
            help='Image scaling factor for high-resolution extraction (default: 2.0)'
        )
        multimodal_group.add_argument(
            '--no-picture-classification',
            action='store_true',
            help='Disable automatic image type classification'
        )
        multimodal_group.add_argument(
            '--no-table-cell-matching',
            action='store_true',
            help='Disable precise table cell matching'
        )
        multimodal_group.add_argument(
            '--no-image-metadata',
            action='store_true',
            help='Skip preserving image metadata during extraction'
        )
        multimodal_group.add_argument(
            '--multimodal-chunk-strategy',
            choices=['separate', 'inline', 'mixed'],
            default='separate',
            help='How to handle multimodal content in chunks (default: separate)'
        )
        multimodal_group.add_argument(
            '--include-images-in-chunks',
            action='store_true',
            help='Include image references in text chunks for better context'
        )
        multimodal_group.add_argument(
            '--no-tables-in-chunks',
            action='store_true',
            help='Exclude table data from text chunks (keep separate)'
        )
        
        # Chunking options
        chunk_group = parser.add_argument_group('Chunking')
        chunk_group.add_argument(
            '--chunk-size',
            type=int,
            default=500,
            help='Target chunk size in tokens (default: 500)'
        )
        chunk_group.add_argument(
            '--chunk-overlap',
            type=int,
            default=50,
            help='Overlap between chunks in tokens (default: 50)'
        )
        chunk_group.add_argument(
            '--no-headings',
            action='store_true',
            help='Disable splitting on headings'
        )
        chunk_group.add_argument(
            '--tokenizer',
            default='cl100k_base',
            help='Tokenizer to use for chunking (default: cl100k_base)'
        )
        
        # Dataset generation options
        dataset_group = parser.add_argument_group('Dataset Generation')
        dataset_group.add_argument(
            '--generate-datasets',
            action='store_true',
            help='Generate datasets from chunks (choose method with --dataset-method)'
        )
        dataset_group.add_argument(
            '--dataset-method',
            choices=['augmentoolkit'],
            default='augmentoolkit',
            help='Dataset generation method (default: augmentoolkit)'
        )
        dataset_group.add_argument(
            '--enable-deduplication',
            action='store_true',
            default=True,
            help='Enable dataset deduplication (default: enabled)'
        )
        dataset_group.add_argument(
            '--disable-deduplication',
            action='store_true',
            help='Disable dataset deduplication'
        )
        dataset_group.add_argument(
            '--dedup-strategy',
            choices=['exact', 'normalized', 'hash', 'semantic', 'progressive'],
            default='progressive',
            help='Deduplication strategy (default: progressive)'
        )
        
        # Removed legacy synthetic dataset options - use Augmentoolkit instead
        
        # Augmentoolkit dataset options
        augment_group = parser.add_argument_group('Augmentoolkit Dataset Generation')
        augment_group.add_argument(
            '--augment-type',
            choices=['factual', 'rag', 'multi_source', 'repvar', 'complete'],
            default='factual',
            help='Type of Augmentoolkit dataset to generate (default: factual)'
        )
        augment_group.add_argument(
            '--augment-model',
            choices=['small', 'medium', 'large'],
            default='medium',
            help='MLX model size for Augmentoolkit (default: medium)'
        )
        augment_group.add_argument(
            '--augment-model-name',
            help='Specific MLX model name for Augmentoolkit (overrides --augment-model)'
        )
        augment_group.add_argument(
            '--augment-context',
            default='technical documentation',
            help='Dataset context for Augmentoolkit (default: "technical documentation")'
        )
        augment_group.add_argument(
            '--augment-concurrency',
            type=int,
            default=5,
            help='Concurrency limit for Augmentoolkit (default: 5)'
        )
        augment_group.add_argument(
            '--augment-subset-size',
            type=int,
            default=100,
            help='Subset size for Augmentoolkit testing (0 to disable, default: 100)'
        )
        
        return parser
    
    def _validate_multimodal_args(self, args: argparse.Namespace) -> List[str]:
        """Validate multimodal command line arguments.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate image scale
        if hasattr(args, 'image_scale') and args.image_scale <= 0:
            errors.append("Image scale must be greater than 0")
        
        # Check feature dependencies
        if (hasattr(args, 'table_format') and args.table_format != 'csv' and 
            args.no_tables):
            errors.append("Table format options cannot be used with --no-tables")
        
        if (hasattr(args, 'image_format') and args.image_format != 'png' and 
            args.no_figures):
            errors.append("Image format options cannot be used with --no-figures")
        
        if (hasattr(args, 'image_descriptions') and args.image_descriptions and 
            args.no_multimodal):
            errors.append("Image descriptions cannot be used with --no-multimodal")
        
        return errors
    
    def _create_config_from_args(self, args: argparse.Namespace) -> PipelineConfig:
        """Create pipeline configuration from command line arguments.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Pipeline configuration
        """
        # Start with environment-based config
        config = load_config_from_env()
        
        # Override with command line arguments
        config.input_path = args.input
        config.output_dir = args.output_dir
        config.extract_dir = args.extract_dir
        config.num_workers = args.workers
        config.max_files = args.max_files
        config.pattern_exclude = args.exclude
        
        # Configure Docling options
        config.docling.with_accelerator = not args.no_accelerator
        config.docling.extract_tables = not args.no_tables
        config.docling.extract_figures = not args.no_figures
        config.docling.multimodal = not args.no_multimodal
        config.docling.custom_convert = not args.no_custom
        config.docling.num_threads = args.threads
        
        # Configure enhanced multimodal options
        if hasattr(args, 'table_format'):
            config.docling.table_format = args.table_format
        if hasattr(args, 'image_format'):
            config.docling.image_format = args.image_format
        if hasattr(args, 'image_descriptions'):
            config.docling.generate_image_descriptions = args.image_descriptions
        if hasattr(args, 'image_scale'):
            config.docling.image_scale = args.image_scale
        if hasattr(args, 'no_picture_classification'):
            config.docling.enable_picture_classification = not args.no_picture_classification
        if hasattr(args, 'no_table_cell_matching'):
            config.docling.table_cell_matching = not args.no_table_cell_matching
        if hasattr(args, 'no_image_metadata'):
            config.docling.preserve_image_metadata = not args.no_image_metadata
        if hasattr(args, 'multimodal_chunk_strategy'):
            config.docling.multimodal_chunk_strategy = args.multimodal_chunk_strategy
        if hasattr(args, 'include_images_in_chunks'):
            config.docling.include_images_in_chunks = args.include_images_in_chunks
        if hasattr(args, 'no_tables_in_chunks'):
            config.docling.include_tables_in_chunks = not args.no_tables_in_chunks
        
        # Configure chunking options
        config.docling.chunk_size = args.chunk_size
        config.docling.chunk_overlap = args.chunk_overlap
        config.docling.split_on_headings = not args.no_headings
        config.docling.tokenizer = args.tokenizer
        
        return config
    
    def run(self, args: argparse.Namespace) -> int:
        """Run the pipeline command.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            # Validate multimodal arguments
            validation_errors = self._validate_multimodal_args(args)
            if validation_errors:
                logger.error("Validation errors found:")
                for error in validation_errors:
                    logger.error(f"  - {error}")
                return 1
            
            # Create configuration from arguments
            config = self._create_config_from_args(args)
            
            # Validate configuration
            validation_errors = validate_config(config)
            if validation_errors:
                logger.error("Configuration validation failed:")
                for error in validation_errors:
                    logger.error(f"  - {error}")
                return 1
            
            # Log configuration summary
            logger.info("Pipeline Configuration:")
            logger.info(f"  Input: {config.input_path}")
            logger.info(f"  Output: {config.output_dir}")
            logger.info(f"  Workers: {config.num_workers}")
            logger.info(f"  Docling features: Tables={config.docling.extract_tables} ({config.docling.table_format}), "
                       f"Figures={config.docling.extract_figures} ({config.docling.image_format}), "
                       f"Accelerator={config.docling.with_accelerator}")
            logger.info(f"  Multimodal options: Strategy={config.docling.multimodal_chunk_strategy}, "
                       f"Descriptions={config.docling.generate_image_descriptions}, "
                       f"Scale={config.docling.image_scale}")
            logger.info(f"  Chunking: Size={config.docling.chunk_size}, "
                       f"Overlap={config.docling.chunk_overlap}")
            
            # Create and run pipeline
            pipeline = DocumentPipeline(config)
            summary = pipeline.run()
            
            # Log summary
            stats = summary.get("statistics", {})
            logger.info("Pipeline Execution Summary:")
            logger.info(f"  Files found: {stats.get('total_files_found', 0)}")
            logger.info(f"  Files processed: {stats.get('files_processed', 0)}")
            logger.info(f"  Files skipped: {stats.get('files_skipped', 0)}")
            logger.info(f"  Markdown files created: {stats.get('markdown_files_created', 0)}")
            logger.info(f"  Chunks created: {stats.get('chunks_created', 0)}")
            logger.info(f"  Success rate: {stats.get('success_rate', 0):.1f}%")
            logger.info(f"  Processing time: {summary.get('processing_time_seconds', 0):.2f} seconds")
            
            # Generate datasets if requested (Augmentoolkit only)
            should_generate_datasets = args.generate_datasets
            
            if should_generate_datasets and stats.get('chunks_created', 0) > 0:
                chunks_dir = f"{config.output_dir}/chunks"
                
                # Use Augmentoolkit for advanced dataset generation
                if args.dataset_method == 'augmentoolkit':
                    asyncio.run(self._generate_augmentoolkit_datasets(args, chunks_dir, config.output_dir))
                else:
                    logger.warning("Only Augmentoolkit dataset generation is supported. Use --dataset-method augmentoolkit")
            
            # Report errors if any
            errors = summary.get("errors", [])
            if errors:
                logger.warning(f"Pipeline completed with {len(errors)} errors:")
                for error in errors[:5]:  # Show first 5 errors
                    logger.warning(f"  - {error}")
                if len(errors) > 5:
                    logger.warning(f"  ... and {len(errors) - 5} more errors")
            
            # Determine exit code based on success rate
            success_rate = stats.get('success_rate', 0)
            if success_rate == 100:
                logger.info("Pipeline completed successfully!")
                return 0
            elif success_rate >= 50:
                logger.warning("Pipeline completed with some errors")
                return 0  # Still consider this a success
            else:
                logger.error("Pipeline failed - low success rate")
                return 1
        
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            return 1
    
    async def _generate_augmentoolkit_datasets(self, args: argparse.Namespace, chunks_dir: str, output_dir: str):
        """Generate datasets using the integrated Augmentoolkit functionality."""
        
        logger.info(f"Generating {args.augment_type} dataset using Augmentoolkit...")
        logger.info("⚠️ Dataset generation has NO TIMEOUT and will run until completion")
        logger.info("💡 This process may take 30 minutes to several hours depending on:")
        logger.info("   • Number of chunks to process")
        logger.info("   • Model complexity and generation parameters")
        logger.info("   • System performance")
        logger.info("📊 Progress will be shown during generation...")
        
        try:
            # Create Augmentoolkit output directory
            augment_output_dir = f"{output_dir}/augmentoolkit_datasets"
            import os
            os.makedirs(augment_output_dir, exist_ok=True)
            
            # Import the enhanced generator function
            from ...core.augmentoolkit_generator import generate_advanced_dataset
            
            # Determine model name
            if args.augment_model_name:
                model_name = args.augment_model_name
            else:
                # Map model size to specific model names
                model_mapping = {
                    'small': 'mlx-community/Llama-3.2-1B-Instruct-4bit',
                    'medium': 'mlx-community/Llama-3.2-3B-Instruct-4bit', 
                    'large': 'mlx-community/Llama-3.1-8B-Instruct-4bit'
                }
                model_name = model_mapping.get(args.augment_model, model_mapping['medium'])
            
            logger.info(f"Using model: {model_name}")
            logger.info(f"Dataset type: {args.augment_type}")
            logger.info(f"Context: {args.augment_context}")
            logger.info(f"Concurrency: {args.augment_concurrency}")
            
            # Configure generation parameters
            config_kwargs = {
                'dataset_context': args.augment_context,
                'concurrency_limit': args.augment_concurrency,
                'use_subset': hasattr(args, 'augment_subset_size') and args.augment_subset_size > 0,
                'subset_size': getattr(args, 'augment_subset_size', 100),
                'temperature': getattr(args, 'augment_temperature', 0.7)
            }
            
            # Generate the dataset with timeout handling
            try:
                results = await generate_advanced_dataset(
                    input_path=chunks_dir,
                    output_path=augment_output_dir,
                    dataset_type=args.augment_type,
                    model_name=model_name,
                    mode='mlx',
                    max_chunks=getattr(args, 'max_files', None),
                    **config_kwargs
                )
            except asyncio.TimeoutError:
                logger.error("❌ Dataset generation timed out")
                logger.info("💡 Try reducing the number of chunks or subset size")
                return
            except KeyboardInterrupt:
                logger.warning("⚠️ Dataset generation interrupted by user")
                return
            
            # Check for errors
            if "error" in results:
                logger.error(f"❌ Augmentoolkit generation failed: {results['error']}")
                if "validation_issues" in results:
                    logger.error("📋 Validation issues:")
                    for issue in results["validation_issues"]:
                        logger.error(f"  • {issue}")
                    logger.info("💡 To fix issues, try:")
                    logger.info("  • Install missing dependencies: pip install -e '.[augmentoolkit]'")
                    logger.info("  • For MLX models, ensure you're on Apple Silicon with MLX installed")
                    logger.info("  • Check disk space and write permissions")
                
                # Additional debugging information
                logger.error("🔍 Debugging information:")
                logger.error(f"  • Chunks directory: {chunks_dir}")
                logger.error(f"  • Output directory: {augment_output_dir}")
                logger.error(f"  • Model: {model_name}")
                logger.error(f"  • Dataset type: {args.augment_type}")
                
                # Check if chunks directory exists and has content
                if os.path.exists(chunks_dir):
                    chunk_files = [f for f in os.listdir(chunks_dir) if f.endswith('.json')]
                    logger.error(f"  • Chunk files found: {len(chunk_files)}")
                    if chunk_files:
                        logger.error(f"  • Example chunk files: {chunk_files[:3]}")
                else:
                    logger.error(f"  • ❌ Chunks directory does not exist!")
                
                return
            
            # Report successful results
            logger.info("✅ Augmentoolkit dataset generation completed!")
            
            success_count = 0
            total_count = 0
            
            for dataset_name, result in results.items():
                total_count += 1
                if "error" in result:
                    logger.error(f"❌ {dataset_name} generation failed: {result['error']}")
                else:
                    success_count += 1
                    logger.info(f"📁 {dataset_name} dataset: {result.get('output_dir', 'Unknown location')}")
                    if "input_chunks" in result:
                        logger.info(f"   Processed {result['input_chunks']} chunks")
            
            if success_count == total_count:
                logger.info(f"🎉 All {total_count} datasets saved to: {augment_output_dir}")
            elif success_count > 0:
                logger.warning(f"⚠️ {success_count}/{total_count} datasets completed successfully")
            else:
                logger.error(f"❌ All {total_count} dataset generations failed")
                
        except Exception as e:
            logger.error(f"Error generating Augmentoolkit dataset: {e}")
            import traceback
            traceback.print_exc()
            # Don't fail the pipeline for dataset generation errors