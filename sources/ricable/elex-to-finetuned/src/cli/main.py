"""Main CLI entry point for Flow4."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ..utils.config import PipelineConfig, DoclingConfig, MLXConfig, CLIConfig
from ..utils.logging import setup_logging, get_logger
from ..core.pipeline import DocumentPipeline

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="flow4",
        description="Flow4: Document Processing Pipeline for RAG and Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process HTML files from data/ directory
  flow4 pipeline --input data/ --output-dir output --verbose
  
  # Process with Augmentoolkit advanced dataset generation
  flow4 pipeline --input data/ --output-dir output --use-augmentoolkit --augmentoolkit-config src/flow4_factual_full.yaml
  
  # Generate datasets only using Augmentoolkit
  flow4 generate --input output/chunks --output-dir augmentoolkit_output --config src/flow4_factual_full.yaml
  
  # Convert documents only
  flow4 convert --input data/ --output-dir markdown/
  
  # Chunk existing markdown
  flow4 chunk --input combined.md --output-dir chunks/
  
  # Clean combined documents
  flow4 clean --input combined_document.md --output cleaned_document.md
  
  # Fine-tune with MLX using Augmentoolkit dataset
  flow4 finetune --dataset augmentoolkit_output/mlx_dataset.jsonl --model mlx-community/Llama-3.2-3B-Instruct-4bit
        """
    )
    
    # Global options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--debug",
        action="store_true", 
        help="Enable debug logging"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress console output"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log to file"
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run complete document processing pipeline"
    )
    add_pipeline_args(pipeline_parser)
    
    # Convert command  
    convert_parser = subparsers.add_parser(
        "convert", 
        help="Convert documents to Markdown"
    )
    add_convert_args(convert_parser)
    
    # Chunk command
    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Chunk markdown documents"
    )
    add_chunk_args(chunk_parser)
    
    # Generate command (Augmentoolkit)
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate advanced datasets using Augmentoolkit"
    )
    add_generate_args(generate_parser)
    
    # Clean command
    clean_parser = subparsers.add_parser(
        "clean",
        help="Clean combined documents to remove duplicates and improve formatting"
    )
    add_clean_args(clean_parser)
    
    # Fine-tune command
    finetune_parser = subparsers.add_parser(
        "finetune",
        help="Fine-tune model with MLX"
    )
    add_finetune_args(finetune_parser)
    
    return parser


def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """Add pipeline command arguments."""
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input ZIP file or directory"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to process"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--exclude-pattern",
        type=str,
        help="Pattern to exclude from processing"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size in tokens (default: 500)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap in tokens (default: 50)"
    )
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Disable table extraction"
    )
    parser.add_argument(
        "--no-figures", 
        action="store_true",
        help="Disable figure extraction"
    )
    parser.add_argument(
        "--no-accelerator",
        action="store_true",
        help="Disable hardware acceleration"
    )
    parser.add_argument(
        "--use-augmentoolkit",
        action="store_true",
        help="Use Augmentoolkit for advanced dataset generation"
    )
    parser.add_argument(
        "--augmentoolkit-config",
        type=str,
        help="Path to Augmentoolkit YAML configuration file"
    )
    parser.add_argument(
        "--disable-filtering",
        action="store_true",
        help="Disable HTML content filtering (process all files including code documentation)"
    )


def add_convert_args(parser: argparse.ArgumentParser) -> None:
    """Add convert command arguments."""
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory with HTML/PDF files"
    )
    parser.add_argument(
        "--output-dir", "-o", 
        type=str,
        default="markdown",
        help="Output directory for Markdown files (default: markdown)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--extract-tables",
        action="store_true",
        help="Extract tables from documents"
    )
    parser.add_argument(
        "--extract-figures",
        action="store_true", 
        help="Extract figures from documents"
    )


def add_chunk_args(parser: argparse.ArgumentParser) -> None:
    """Add chunk command arguments."""
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input markdown file or directory"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str, 
        default="chunks",
        help="Output directory for chunks (default: chunks)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size in tokens (default: 500)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap in tokens (default: 50)"
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Use semantic chunking"
    )


def add_generate_args(parser: argparse.ArgumentParser) -> None:
    """Add generate command arguments."""
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory with document chunks"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="augmentoolkit_output",
        help="Output directory for generated datasets (default: augmentoolkit_output)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to Augmentoolkit YAML configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="MLX model for generation (default: Llama-3.2-3B-Instruct-4bit)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Concurrency limit for generation (default: 8)"
    )
    parser.add_argument(
        "--no-quality-checks",
        action="store_true",
        help="Skip quality validation checks for faster generation"
    )


def add_clean_args(parser: argparse.ArgumentParser) -> None:
    """Add clean command arguments."""
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input markdown file to clean"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (optional, defaults to input file)"
    )


def add_finetune_args(parser: argparse.ArgumentParser) -> None:
    """Add finetune command arguments."""
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="Path to training dataset"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="fine_tuned_model",
        help="Output directory for fine-tuned model (default: fine_tuned_model)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model to fine-tune (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)"
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=500,
        help="Number of training iterations (default: 500)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (default: 2)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start interactive chat after training"
    )
    parser.add_argument(
        "--no-fuse",
        action="store_true",
        help="Don't fuse model (keep as LoRA adapter)"
    )


def handle_pipeline_command(args, config: PipelineConfig) -> int:
    """Handle the pipeline command."""
    # Update config with command line arguments
    config.input_path = args.input
    config.output_dir = args.output_dir
    config.max_files = args.max_files
    config.num_workers = args.num_workers
    config.pattern_exclude = args.exclude_pattern
    
    # Update docling config
    config.docling.chunk_size = args.chunk_size
    config.docling.max_tokens = args.chunk_size  # Fix: semantic chunking uses max_tokens
    config.docling.chunk_overlap = args.chunk_overlap
    config.docling.overlap_tokens = args.chunk_overlap  # Fix: semantic chunking uses overlap_tokens
    config.docling.extract_tables = not args.no_tables
    config.docling.extract_figures = not args.no_figures
    config.docling.with_accelerator = not args.no_accelerator
    config.disable_filtering = args.disable_filtering
    
    try:
        # Initialize and run pipeline
        pipeline = DocumentPipeline(config)
        summary = pipeline.run()
        
        if summary["statistics"]["success_rate"] > 0:
            print(f"\n✅ Pipeline completed successfully!")
            print(f"📊 Success rate: {summary['statistics']['success_rate']:.1f}%")
            print(f"📄 Files processed: {summary['statistics']['files_processed']}")
            print(f"🧩 Chunks created: {summary['statistics']['chunks_created']}")
            
            # Optional Augmentoolkit generation
            if args.use_augmentoolkit:
                print(f"\n" + "=" * 60)
                print(f"🚀 STARTING AUGMENTOOLKIT DATASET GENERATION")
                print(f"=" * 60)
                
                try:
                    from ..core.augmentoolkit_generator import AugmentoolkitGenerator, AugmentoolkitConfig
                    
                    if AugmentoolkitGenerator.is_available():
                        print(f"✅ Augmentoolkit is available")
                        print(f"📁 Input chunks: {args.output_dir}/chunks")
                        print(f"📁 Output directory: {args.output_dir}/augmentoolkit")
                        
                        if args.augmentoolkit_config:
                            print(f"📋 Using config: {args.augmentoolkit_config}")
                        else:
                            print(f"📋 Using default configuration")
                        
                        # Setup Augmentoolkit config
                        atk_config = AugmentoolkitConfig()
                        generator = AugmentoolkitGenerator(atk_config)
                        
                        # Use chunks from pipeline output
                        chunks_dir = f"{args.output_dir}/chunks"
                        augmentoolkit_output = f"{args.output_dir}/augmentoolkit"
                        
                        print(f"\n⏳ Running dataset generation... (this may take several minutes)")
                        print(f"📊 Check logs above for detailed progress information")
                        
                        result = generator.generate_dataset_sync(
                            chunks_dir=chunks_dir,
                            output_dir=augmentoolkit_output,
                            config_yaml=args.augmentoolkit_config
                        )
                        
                        print(f"\n" + "=" * 60)
                        if result.get("generation_complete", False):
                            print(f"🎉 AUGMENTOOLKIT GENERATION COMPLETED!")
                            print(f"📊 Statistics:")
                            print(f"   • Chunks processed: {result.get('total_chunks_processed', 0)}")
                            print(f"   • QA pairs generated: {result.get('qa_pairs_generated', 0)}")
                            print(f"   • Success rate: {result.get('success_rate', 0):.1f}%")
                            print(f"📁 Output files:")
                            output_files = result.get('output_files', {})
                            if 'mlx_dataset' in output_files:
                                print(f"   • MLX dataset: {output_files['mlx_dataset']}")
                            if 'original_dataset' in output_files:
                                print(f"   • Original dataset: {output_files['original_dataset']}")
                            print(f"🎯 Dataset is ready for MLX fine-tuning!")
                        else:
                            print(f"❌ AUGMENTOOLKIT GENERATION FAILED")
                            print(f"💥 Error: {result.get('error', 'Unknown error')}")
                        print(f"=" * 60)
                    else:
                        print(f"⚠️ Augmentoolkit not available - MLX dependencies missing")
                        print(f"💡 Install with: uv pip install mlx mlx-lm (Apple Silicon only)")
                        
                except Exception as e:
                    print(f"\n❌ Augmentoolkit generation failed with exception:")
                    print(f"💥 Error: {e}")
                    print(f"💡 Check logs above for detailed error information")
            
            return 0
        else:
            print(f"\n❌ Pipeline failed - no files were processed")
            return 1
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        return 1


def handle_convert_command(args, config: PipelineConfig) -> int:
    """Handle the convert command."""
    from ..core.converter import DocumentConverter
    
    # Update config
    config.docling.extract_tables = args.extract_tables
    config.docling.extract_figures = args.extract_figures
    
    try:
        converter = DocumentConverter(config.docling)
        
        if not converter.is_available():
            print("❌ Document converter not available. Install docling with: pip install docling")
            return 1
        
        # Find files to convert
        input_path = Path(args.input)
        if input_path.is_file():
            files = [str(input_path)]
        else:
            files = list(input_path.glob("**/*.html")) + list(input_path.glob("**/*.pdf"))
            files = [str(f) for f in files]
        
        if not files:
            print(f"❌ No HTML or PDF files found in {args.input}")
            return 1
        
        print(f"🔄 Converting {len(files)} files...")
        
        # Convert files
        successful, failed = converter.batch_convert(
            files, 
            args.output_dir, 
            args.num_workers
        )
        
        print(f"\n✅ Conversion complete!")
        print(f"📄 Successfully converted: {len(successful)} files")
        print(f"❌ Failed: {len(failed)} files")
        print(f"📁 Output directory: {args.output_dir}")
        
        return 0 if successful else 1
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        print(f"\n❌ Conversion failed: {e}")
        return 1


def handle_chunk_command(args, config: PipelineConfig) -> int:
    """Handle the chunk command."""
    from ..core.chunker import DocumentChunker
    
    # Update config
    config.docling.chunk_size = args.chunk_size
    config.docling.max_tokens = args.chunk_size  # Fix: semantic chunking uses max_tokens
    config.docling.chunk_overlap = args.chunk_overlap
    config.docling.overlap_tokens = args.chunk_overlap  # Fix: semantic chunking uses overlap_tokens
    config.docling.enable_semantic_chunking = args.semantic
    
    try:
        chunker = DocumentChunker(config.docling)
        
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"❌ Input path does not exist: {args.input}")
            return 1
        
        if input_path.is_file():
            # Chunk single file
            print(f"🧩 Chunking file: {input_path.name}")
            chunks = chunker.chunk_file(str(input_path))
        else:
            # Chunk all markdown files in directory
            md_files = list(input_path.glob("**/*.md"))
            if not md_files:
                print(f"❌ No markdown files found in {args.input}")
                return 1
            
            print(f"🧩 Chunking {len(md_files)} markdown files...")
            chunks = []
            for md_file in md_files:
                file_chunks = chunker.chunk_file(str(md_file))
                chunks.extend(file_chunks)
        
        if not chunks:
            print("❌ No chunks created")
            return 1
        
        # Save chunks
        chunker.save_chunks(chunks, args.output_dir)
        
        print(f"\n✅ Chunking complete!")
        print(f"🧩 Total chunks: {len(chunks)}")
        print(f"📁 Output directory: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        print(f"\n❌ Chunking failed: {e}")
        return 1


def handle_generate_command(args, config: PipelineConfig) -> int:
    """Handle the generate command."""
    from ..core.augmentoolkit_generator import AugmentoolkitGenerator, AugmentoolkitConfig
    
    try:
        # Check if Augmentoolkit is available
        if not AugmentoolkitGenerator.is_available():
            print("❌ Augmentoolkit not available. Please install augmentoolkit or check backup-code/ directory.")
            return 1
        
        if not Path(args.input).exists():
            print(f"❌ Input directory does not exist: {args.input}")
            return 1
        
        # Create Augmentoolkit config
        atk_config = AugmentoolkitConfig(
            model_name=args.model,
            concurrency_limit=args.concurrency,
            skip_question_check=args.no_quality_checks,
            skip_answer_relevancy_check=args.no_quality_checks,
            skip_answer_accuracy_check=args.no_quality_checks
        )
        
        print(f"🚀 Starting Augmentoolkit dataset generation...")
        print(f"📁 Input: {args.input}")
        print(f"📁 Output: {args.output_dir}")
        print(f"🤖 Model: {args.model}")
        print(f"⚙️  Concurrency: {args.concurrency}")
        if args.config:
            print(f"📋 Config: {args.config}")
        
        # Initialize generator
        generator = AugmentoolkitGenerator(atk_config)
        
        # Run generation
        result = generator.generate_dataset_sync(
            chunks_dir=args.input,
            output_dir=args.output_dir,
            config_yaml=args.config
        )
        
        if result.get("generation_complete", False):
            print(f"\n✅ Augmentoolkit generation complete!")
            print(f"📊 QA pairs generated: {result.get('qa_pairs_generated', 0)}")
            print(f"🧩 Chunks processed: {result.get('total_chunks_processed', 0)}")
            
            output_files = result.get("output_files", {})
            if "mlx_dataset" in output_files:
                print(f"🎯 MLX dataset ready: {output_files['mlx_dataset']}")
                print(f"🔧 Ready for MLX fine-tuning!")
            
            return 0
        else:
            print(f"\n❌ Generation failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        print(f"\n❌ Generation failed: {e}")
        return 1


def handle_clean_command(args, config: PipelineConfig) -> int:
    """Handle the clean command."""
    from ..core.document_cleaner import DocumentCleaner
    
    try:
        if not Path(args.input).exists():
            print(f"❌ Input file does not exist: {args.input}")
            return 1
        
        cleaner = DocumentCleaner(config)
        
        print(f"🧹 Cleaning document: {args.input}")
        
        # Clean the document
        output_path = cleaner.clean_document(
            input_path=args.input,
            output_path=args.output
        )
        
        print(f"\n✅ Document cleaning complete!")
        print(f"📄 Cleaned document: {output_path}")
        print(f"🎯 Ready for improved chunking and dataset generation!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Document cleaning failed: {e}")
        print(f"\n❌ Document cleaning failed: {e}")
        return 1


def handle_finetune_command(args, config: PipelineConfig) -> int:
    """Handle the finetune command."""
    from ..core.mlx_finetuner import MLXFineTuner
    from ..utils.config import MLXConfig
    
    # Create MLX config
    mlx_config = MLXConfig()
    mlx_config.model_name = args.model
    mlx_config.num_iters = args.num_iters
    mlx_config.batch_size = args.batch_size
    mlx_config.learning_rate = args.learning_rate
    mlx_config.fuse_model = not args.no_fuse
    
    # Disable auto-optimization when explicit CLI parameters are provided
    mlx_config.auto_optimize_m3 = False
    
    try:
        finetuner = MLXFineTuner(mlx_config)
        
        if not finetuner.is_available():
            print("❌ MLX not available. Install with: pip install mlx>=0.12.0 mlx-lm>=0.8.0")
            return 1
        
        if not Path(args.dataset).exists():
            print(f"❌ Dataset file does not exist: {args.dataset}")
            return 1
        
        print(f"=" * 80)
        print(f"🚀 STARTING MLX FINE-TUNING")
        print(f"=" * 80)
        print(f"📊 Fine-tuning Setup:")
        print(f"   📁 Dataset: {args.dataset}")
        print(f"   🤖 Model: {args.model}")
        print(f"   ⚙️  Iterations: {args.num_iters}")
        print(f"   🔄 Batch size: {args.batch_size}")
        print(f"   📈 Learning rate: {args.learning_rate}")
        print(f"   📁 Output: {args.output_dir}")
        print(f"\n⏳ Fine-tuning in progress... (this may take a while)")
        print(f"📊 Check logs above for detailed progress information")
        
        # Run fine-tuning
        model_path = finetuner.finetune(args.dataset, args.output_dir)
        
        if model_path:
            print(f"\n" + "=" * 80)
            print(f"🎉 FINE-TUNING COMPLETED SUCCESSFULLY!")
            print(f"📁 Model location: {model_path}")
            print(f"🎯 Model is ready for inference!")
            
            if args.chat:
                print(f"\n🗣️  Starting interactive chat session...")
                print(f"💬 You can now test your fine-tuned model")
                print(f"=" * 80)
                finetuner.interactive_chat(model_path)
            else:
                print(f"\n💡 To test your model, run:")
                print(f"   python src/run_flow4.py finetune --dataset {args.dataset} --chat")
                print(f"=" * 80)
            
            return 0
        else:
            print(f"\n" + "=" * 80)
            print(f"❌ FINE-TUNING FAILED")
            print(f"💥 Check logs above for detailed error information")
            print(f"=" * 80)
            return 1
            
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        print(f"\n❌ Fine-tuning failed: {e}")
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Show help if no command provided
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    cli_config = CLIConfig(
        verbose=args.verbose,
        debug=args.debug,
        quiet=args.quiet,
        log_file=args.log_file
    )
    
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(
        level=log_level,
        log_file=args.log_file,
        verbose=args.verbose,
        quiet=args.quiet
    )
    
    logger.info(f"Flow4 CLI started with command: {args.command}")
    
    # Create base configuration
    config = PipelineConfig()
    
    # Handle commands
    try:
        if args.command == "pipeline":
            return handle_pipeline_command(args, config)
        elif args.command == "convert":
            return handle_convert_command(args, config)
        elif args.command == "chunk":
            return handle_chunk_command(args, config)
        elif args.command == "generate":
            return handle_generate_command(args, config)
        elif args.command == "clean":
            return handle_clean_command(args, config)
        elif args.command == "finetune":
            return handle_finetune_command(args, config)
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())