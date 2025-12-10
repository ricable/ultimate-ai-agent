"""Dataset Generation Command for Flow4

Advanced dataset generation using Augmentoolkit pipelines with MLX model support.
Includes efficient large-scale MLX dataset generation for maximum chunk utilization.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Optional, List

from ...utils.logging import get_logger
from ...utils.augmentoolkit_config import (
    AugmentoolkitConfigManager,
    create_default_config,
    get_recommended_mlx_models
)
from ...core.augmentoolkit_generator import generate_advanced_dataset
from ...core.dataset_generator import LargeMLXDatasetGenerator

logger = get_logger(__name__)


class GenerateCommand:
    """Command to generate advanced datasets using Augmentoolkit pipelines."""
    
    def add_parser(self, subparsers, name: str) -> argparse.ArgumentParser:
        """Add parser for generate command.
        
        Args:
            subparsers: Subparser object to add to
            name: Name of the command
            
        Returns:
            Created parser
        """
        parser = subparsers.add_parser(
            name,
            help='Generate advanced datasets using Augmentoolkit',
            description="""
Generate advanced datasets using Augmentoolkit pipelines with MLX model support.

This command uses Augmentoolkit's sophisticated dataset generation capabilities
to create high-quality QA datasets, RAG training data, and instruction-following
conversations from Flow4 document chunks, with full multimodal support.

Features:
- Multi-stage validation (question quality, answer relevancy, answer accuracy)
- RAG training data generation with context simulation
- Conversation generation for instruction tuning
- MLX model support optimized for Apple Silicon
- Configurable generation parameters and quality controls
- Multimodal dataset generation for tables, images, and structured content
- Cross-modal question generation combining text, tables, and visual elements
- Flexible output formats for different training scenarios

Examples:
  # Generate factual QA dataset with medium model
  flow4 generate --input ./output/chunks --output ./datasets --type factual

  # Generate RAG training data with large model
  flow4 generate --input ./chunks --type rag --model large --context "medical documentation"

  # Generate multimodal dataset with table and image QA
  flow4 generate --input ./chunks --type multimodal --include-table-qa --include-image-qa

  # Generate complete dataset suite with custom model
  flow4 generate --input ./chunks --type complete --model-name "mlx-community/Llama-3.1-8B-Instruct-4bit"

  # Generate complex cross-modal questions
  flow4 generate --input ./chunks --type multimodal --cross-modal-questions --table-qa-complexity complex

  # Test with small subset
  flow4 generate --input ./chunks --subset-size 10 --dry-run
""",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Input/Output options
        io_group = parser.add_argument_group('Input/Output')
        io_group.add_argument(
            '--input', '-i',
            required=True,
            help='Input directory containing chunk files from Flow4 pipeline'
        )
        io_group.add_argument(
            '--output', '-o',
            default='./output/datasets',
            help='Output directory for generated datasets (default: ./output/datasets)'
        )
        
        # Dataset generation options
        dataset_group = parser.add_argument_group('Dataset Generation')
        dataset_group.add_argument(
            '--type',
            choices=['factual', 'rag', 'complete', 'multimodal'],
            default='factual',
            help='Type of dataset to generate (default: factual)'
        )
        dataset_group.add_argument(
            '--context',
            default='technical documentation',
            help='Dataset context description (default: "technical documentation")'
        )
        dataset_group.add_argument(
            '--max-chunks',
            type=int,
            help='Maximum number of chunks to process'
        )
        dataset_group.add_argument(
            '--subset-size',
            type=int,
            default=100,
            help='Size of subset for testing (0 to disable, default: 100)'
        )
        dataset_group.add_argument(
            '--large-mlx',
            action='store_true',
            help='Use efficient large MLX dataset generation for maximum chunk utilization'
        )
        dataset_group.add_argument(
            '--target-examples',
            type=int,
            default=800,
            help='Target number of training examples for large MLX generation (default: 800)'
        )
        
        # Multimodal dataset options
        multimodal_dataset_group = parser.add_argument_group('Multimodal Dataset Generation')
        multimodal_dataset_group.add_argument(
            '--include-table-qa',
            action='store_true',
            help='Generate QA pairs specifically for tables and structured data'
        )
        multimodal_dataset_group.add_argument(
            '--include-image-qa',
            action='store_true',
            help='Generate QA pairs that reference images and visual content'
        )
        multimodal_dataset_group.add_argument(
            '--image-description-style',
            choices=['detailed', 'concise', 'technical', 'natural'],
            default='detailed',
            help='Style for image descriptions in generated content (default: detailed)'
        )
        multimodal_dataset_group.add_argument(
            '--table-qa-complexity',
            choices=['simple', 'intermediate', 'complex'],
            default='intermediate',
            help='Complexity level for table-based questions (default: intermediate)'
        )
        multimodal_dataset_group.add_argument(
            '--cross-modal-questions',
            action='store_true',
            help='Generate questions that combine text, tables, and images'
        )
        multimodal_dataset_group.add_argument(
            '--preserve-multimodal-context',
            action='store_true',
            default=True,
            help='Maintain references to tables and images in generated QA pairs'
        )
        multimodal_dataset_group.add_argument(
            '--multimodal-output-format',
            choices=['markdown', 'json', 'mixed'],
            default='mixed',
            help='Output format for multimodal dataset content (default: mixed)'
        )
        
        # Model configuration
        model_group = parser.add_argument_group('Model Configuration')
        model_group.add_argument(
            '--model',
            choices=['small', 'medium', 'large'],
            default='medium',
            help='MLX model size to use (default: medium)'
        )
        model_group.add_argument(
            '--model-name',
            help='Specific MLX model name (overrides --model)'
        )
        model_group.add_argument(
            '--temperature',
            type=float,
            default=0.7,
            help='Sampling temperature (default: 0.7)'
        )
        
        # Performance options
        perf_group = parser.add_argument_group('Performance')
        perf_group.add_argument(
            '--concurrency',
            type=int,
            default=5,
            help='Number of concurrent generations (default: 5)'
        )
        
        # Advanced options
        advanced_group = parser.add_argument_group('Advanced')
        advanced_group.add_argument(
            '--config',
            help='Custom Augmentoolkit configuration file'
        )
        advanced_group.add_argument(
            '--dry-run',
            action='store_true',
            help='Show configuration without running generation'
        )
        
        # Information commands
        info_group = parser.add_argument_group('Information')
        info_group.add_argument(
            '--list-models',
            action='store_true',
            help='List available MLX models and exit'
        )
        info_group.add_argument(
            '--list-configs',
            action='store_true',
            help='List available configuration templates and exit'
        )
        info_group.add_argument(
            '--show-config',
            help='Show details of a specific configuration and exit'
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
        
        # Validate multimodal-specific arguments
        if (getattr(args, 'include_table_qa', False) or getattr(args, 'include_image_qa', False)):
            if args.type not in ['multimodal', 'complete']:
                errors.append("Table and image QA options require --type multimodal or --type complete")
        
        if hasattr(args, 'cross_modal_questions') and args.cross_modal_questions:
            if args.type not in ['multimodal', 'complete']:
                errors.append("Cross-modal questions require --type multimodal or --type complete")
            
            # Check that both table and image QA are enabled for cross-modal
            if not (getattr(args, 'include_table_qa', False) and getattr(args, 'include_image_qa', False)):
                errors.append("Cross-modal questions require both --include-table-qa and --include-image-qa")
        
        return errors
    
    def run(self, args: argparse.Namespace) -> int:
        """Run the generate command.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            # Handle information commands first
            if args.list_models:
                self._list_models()
                return 0
            
            if args.list_configs:
                self._list_configs()
                return 0
            
            if args.show_config:
                self._show_config(args.show_config)
                return 0
            
            # Validate required arguments for generation
            if not args.input:
                logger.error("Input directory is required for generation")
                return 1
            
            # Validate multimodal arguments
            validation_errors = self._validate_multimodal_args(args)
            if validation_errors:
                logger.error("Validation errors found:")
                for error in validation_errors:
                    logger.error(f"  - {error}")
                return 1
            
            input_path = Path(args.input)
            if not input_path.exists():
                logger.error(f"Input directory does not exist: {input_path}")
                return 1
            
            # Run the generation
            return self._run_generation(args)
            
        except KeyboardInterrupt:
            logger.warning("⚠️ Generation interrupted by user")
            return 130
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return 1
    
    def _list_models(self):
        """List available MLX models."""
        print("Available MLX Models:")
        print("=" * 50)
        
        models = get_recommended_mlx_models()
        for size, info in models.items():
            print(f"\n{size.upper()} ({info['memory_usage']}):")
            print(f"  Model: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Best for: {', '.join(info['recommended_for'])}")
    
    def _list_configs(self):
        """List available configuration templates."""
        print("Available Configuration Templates:")
        print("=" * 50)
        
        config_manager = AugmentoolkitConfigManager()
        configs = config_manager.list_available_configs()
        
        for config_name in configs:
            print(f"  {config_name}")
        
        if not configs:
            print("  No configuration templates found")
    
    def _show_config(self, config_name: str):
        """Show details of a specific configuration."""
        print(f"Configuration Details: {config_name}")
        print("=" * 50)
        
        try:
            config_manager = AugmentoolkitConfigManager()
            config = config_manager.load_config_template(config_name)
            
            # Show key configuration details
            if "dataset_context" in config:
                print(f"Dataset Context: {config['dataset_context']}")
            
            if "path" in config and "input_dirs" in config["path"]:
                input_dirs = config["path"]["input_dirs"]
                if input_dirs:
                    print(f"Input Directory: {input_dirs[0].get('path', 'Not specified')}")
            
            if "system" in config:
                system = config["system"]
                print(f"Concurrency Limit: {system.get('concurrency_limit', 'Not specified')}")
                print(f"Chunk Size: {system.get('chunk_size', 'Not specified')}")
            
            # Show model configuration
            if "factual_sft_settings" in config:
                settings = config["factual_sft_settings"]
                model = settings.get("factual_small_model", "Not specified")
                print(f"Model: {model}")
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
    
    def _run_generation(self, args: argparse.Namespace) -> int:
        """Run the dataset generation process."""
        logger.info("🚀 Starting Augmentoolkit dataset generation")
        
        # Determine model name
        if args.model_name:
            selected_model = args.model_name
        else:
            models = get_recommended_mlx_models()
            selected_model = models[args.model]["name"]
        
        # Show configuration
        if args.dry_run:
            logger.info("Available MLX models:")
            models = get_recommended_mlx_models()
            for size, info in models.items():
                logger.info(f"  {size}: {info['name']}")
                logger.info(f"    Description: {info['description']}")
                logger.info(f"    Memory: {info['memory_usage']}")
                logger.info(f"    Best for: {', '.join(info['recommended_for'])}")
                logger.info("")
        
        logger.info(f"📊 Configuration:")
        logger.info(f"  Input: {args.input}")
        logger.info(f"  Output: {args.output}")
        logger.info(f"  Dataset type: {args.type}")
        logger.info(f"  Model: {selected_model}")
        logger.info(f"  Context: {args.context}")
        logger.info(f"  Max chunks: {args.max_chunks or 'all'}")
        logger.info(f"  Concurrency: {args.concurrency}")
        logger.info(f"  Subset size: {args.subset_size if args.subset_size > 0 else 'disabled'}")
        logger.info(f"  Large MLX mode: {args.large_mlx}")
        if args.large_mlx:
            logger.info(f"  Target examples: {args.target_examples}")
        
        # Log multimodal-specific configuration
        if args.type in ['multimodal', 'complete']:
            logger.info(f"  Multimodal Options:")
            logger.info(f"    Table QA: {getattr(args, 'include_table_qa', False)}")
            logger.info(f"    Image QA: {getattr(args, 'include_image_qa', False)}")
            logger.info(f"    Cross-modal: {getattr(args, 'cross_modal_questions', False)}")
            logger.info(f"    Image style: {getattr(args, 'image_description_style', 'detailed')}")
            logger.info(f"    Table complexity: {getattr(args, 'table_qa_complexity', 'intermediate')}")
            logger.info(f"    Output format: {getattr(args, 'multimodal_output_format', 'mixed')}")
        
        if args.dry_run:
            logger.info("🔍 Dry run mode - configuration shown above")
            return 0
        
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check for large MLX dataset generation
        if args.large_mlx:
            logger.info("🚀 Using large MLX dataset generation for maximum chunk utilization")
            return self._run_large_mlx_generation(args, output_path)
        
        # Run the generation
        logger.info("🎯 Starting dataset generation...")
        logger.info("⚠️ This process has NO TIMEOUT and will run until completion")
        logger.info("💡 Dataset generation may take 30 minutes to several hours depending on:")
        logger.info("   • Number of chunks to process")
        logger.info("   • Model size and generation complexity")
        logger.info("   • System performance")
        logger.info("📊 Progress will be shown during generation...")
        
        # Prepare multimodal options
        multimodal_options = {}
        if args.type in ['multimodal', 'complete']:
            multimodal_options = {
                'include_table_qa': getattr(args, 'include_table_qa', False),
                'include_image_qa': getattr(args, 'include_image_qa', False),
                'image_description_style': getattr(args, 'image_description_style', 'detailed'),
                'table_qa_complexity': getattr(args, 'table_qa_complexity', 'intermediate'),
                'cross_modal_questions': getattr(args, 'cross_modal_questions', False),
                'preserve_multimodal_context': getattr(args, 'preserve_multimodal_context', True),
                'multimodal_output_format': getattr(args, 'multimodal_output_format', 'mixed')
            }
        
        # Run generation with extended timeout handling
        try:
            results = asyncio.run(
                generate_advanced_dataset(
                    input_path=args.input,
                    output_path=args.output,
                    dataset_type=args.type,
                    model_name=selected_model,
                    mode="mlx",
                    max_chunks=args.max_chunks,
                    dataset_context=args.context,
                    concurrency_limit=args.concurrency,
                    temperature=args.temperature,
                    use_subset=(args.subset_size > 0),
                    subset_size=args.subset_size,
                    **multimodal_options
                )
            )
        except asyncio.TimeoutError:
            logger.error("❌ Dataset generation timed out")
            logger.info("💡 Try reducing the number of chunks or subset size")
            return 1
        except KeyboardInterrupt:
            logger.warning("⚠️ Dataset generation interrupted by user")
            return 130
        
        # Check for global errors first
        if "error" in results:
            logger.error(f"❌ Dataset generation failed: {results['error']}")
            if "validation_issues" in results:
                logger.error("📋 Validation issues:")
                for issue in results["validation_issues"]:
                    logger.error(f"  • {issue}")
                logger.info("💡 To fix issues, try:")
                logger.info("  • Install missing dependencies: pip install -e '.[augmentoolkit]'")
                logger.info("  • For MLX models, ensure you're on Apple Silicon with MLX installed")
                logger.info("  • Check disk space and write permissions")
            return 1
        
        # Report individual dataset results
        logger.info("✅ Dataset generation completed!")
        
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
            logger.info(f"🎉 All {total_count} datasets saved to: {output_path}")
            return 0
        elif success_count > 0:
            logger.warning(f"⚠️ {success_count}/{total_count} datasets completed successfully")
            return 0  # Partial success is still considered success
        else:
            logger.error(f"❌ All {total_count} dataset generations failed")
            return 1
    
    def _run_large_mlx_generation(self, args: argparse.Namespace, output_path: Path) -> int:
        """Run large MLX dataset generation for maximum chunk utilization."""
        logger.info("🔍 Searching for chunks for efficient processing...")
        
        input_path = Path(args.input)
        
        # Check for both cache files and individual chunk files
        cache_files = list(input_path.glob("chunk_cache_*.json"))
        chunk_files = list(input_path.glob("chunk_*.json"))
        
        if not cache_files and not chunk_files:
            logger.error("❌ No chunk files found for large MLX generation")
            logger.info("💡 Try running the pipeline first to generate chunks, or use standard generation mode")
            return 1
        
        try:
            # Initialize large MLX generator with chunks directory
            from flow4.core.dataset_generator import LargeMLXDatasetGeneratorConfig
            config = LargeMLXDatasetGeneratorConfig(chunks_dir=str(input_path))
            generator = LargeMLXDatasetGenerator(config)
            
            # Load chunks using the proper method that handles both cache and individual files
            chunks = generator.load_chunks()
            
            if not chunks:
                logger.error("❌ No chunks could be loaded")
                return 1
            
            logger.info(f"📋 Loaded {len(chunks)} chunks for processing")
            
            # Generate the dataset
            summary = generator.generate_large_mlx_dataset_from_chunks(
                chunks=chunks,
                output_dir=str(output_path),
                num_pairs=args.target_examples
            )
            
            # Report results
            logger.info("✅ Large MLX dataset generation completed!")
            logger.info(f"📊 Results:")
            logger.info(f"  Total examples: {summary['total_examples']}")
            logger.info(f"  Training examples: {summary['train_examples']}")
            logger.info(f"  Validation examples: {summary['valid_examples']}")
            logger.info(f"  Test examples: {summary['test_examples']}")
            logger.info(f"  Source chunks used: {summary['source_chunks']}")
            logger.info(f"  Domain: {summary['domain']}")
            logger.info(f"  Format: {summary['format']}")
            logger.info(f"📁 Dataset saved to: {output_path}")
            
            # Compare with typical small datasets
            if summary['total_examples'] > 500:
                logger.info(f"🎉 Successfully generated large dataset ({summary['total_examples']} examples)")
                logger.info("   This is significantly larger than typical subset-limited generation!")
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Large MLX generation failed: {e}")
            logger.info("💡 Falling back to standard Augmentoolkit generation might work")
            return 1