"""
Flow4 MLX Dataset Formatter Command (argparse version)

CLI command for formatting datasets for MLX fine-tuning, addressing all the
formatting issues encountered during development.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from ...core.augmentoolkit_generator import AugmentoolkitGenerator, create_augmentoolkit_config
from ...utils.logging import get_logger
from ...utils.mlx_dataset_formatter import convert_existing_dataset_to_mlx_format

logger = get_logger(__name__)


class FormatMLXCommand:
    """Command for formatting datasets for MLX fine-tuning."""
    
    def add_parser(self, subparsers, name):
        """Add the format-mlx subcommand to the main parser."""
        parser = subparsers.add_parser(
            'format-mlx',
            help='Format datasets for MLX fine-tuning with proper JSONL validation',
            description='''
Format datasets for MLX fine-tuning, addressing all formatting issues:

1. Create MLX dataset from Flow4 chunks:
   flow4 format-mlx create --input chunks_dir --output mlx_dataset_dir

2. Convert existing dataset to MLX format:
   flow4 format-mlx convert --input dataset.jsonl --output mlx_dataset_dir

3. Validate MLX dataset:
   flow4 format-mlx validate --input mlx_dataset_dir

Features:
- Proper JSONL formatting with validation
- MLX batch size requirements (minimum 4 examples per split)
- Text cleaning and encoding fixes
- Automatic train/valid/test splitting
- Support for multiple input formats (chat, prompt-completion, etc.)
            ''',
            formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(prog, max_help_position=50)
        )
        
        # Add subcommands
        subcommands = parser.add_subparsers(dest='mlx_action', help='MLX formatting actions')
        
        # Create command
        create_parser = subcommands.add_parser(
            'create',
            help='Create MLX dataset from Flow4 chunks'
        )
        create_parser.add_argument(
            '--input', '-i', required=True,
            help='Directory containing Flow4 chunks (JSON files)'
        )
        create_parser.add_argument(
            '--output', '-o', required=True,
            help='Output directory for MLX-formatted dataset'
        )
        create_parser.add_argument(
            '--name', '-n', default='mlx_dataset',
            help='Name for the dataset'
        )
        create_parser.add_argument(
            '--model', '-m', default='mlx-community/Llama-3.2-3B-Instruct-4bit',
            help='Model name for configuration'
        )
        create_parser.add_argument(
            '--max-chunks', type=int,
            help='Maximum number of chunks to process'
        )
        
        # Convert command
        convert_parser = subcommands.add_parser(
            'convert',
            help='Convert existing dataset to MLX format'
        )
        convert_parser.add_argument(
            '--input', '-i', required=True,
            help='Existing dataset file to convert (JSONL format)'
        )
        convert_parser.add_argument(
            '--output', '-o', required=True,
            help='Output directory for MLX-formatted dataset'
        )
        convert_parser.add_argument(
            '--name', '-n', default='converted_dataset',
            help='Name for the converted dataset'
        )
        
        # Validate command
        validate_parser = subcommands.add_parser(
            'validate',
            help='Validate MLX dataset files'
        )
        validate_parser.add_argument(
            '--input', '-i', required=True,
            help='Directory containing MLX dataset files'
        )
        
        parser.set_defaults(func=self.run)
        return parser
    
    def run(self, args):
        """Run the format-mlx command."""
        try:
            if args.mlx_action == 'create':
                return self._create_dataset(args)
            elif args.mlx_action == 'convert':
                return self._convert_dataset(args)
            elif args.mlx_action == 'validate':
                return self._validate_dataset(args)
            else:
                logger.error("No MLX action specified. Use 'create', 'convert', or 'validate'.")
                return False
                
        except Exception as e:
            logger.error(f"❌ MLX formatting failed: {e}")
            return False
    
    def _create_dataset(self, args) -> bool:
        """Create MLX dataset from Flow4 chunks."""
        logger.info(f"🔧 Creating MLX dataset from chunks")
        logger.info(f"📁 Input: {args.input}")
        logger.info(f"📁 Output: {args.output}")
        
        # Validate input directory
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"❌ Input directory not found: {args.input}")
            return False
        
        # Create Augmentoolkit generator
        config = create_augmentoolkit_config(
            model_name=args.model,
            mode="mlx",
            dataset_context="technical documentation"
        )
        generator = AugmentoolkitGenerator(config)
        
        # Load chunks
        logger.info("📚 Loading chunks...")
        chunks = generator.process_chunks_from_directory(str(input_path), args.max_chunks)
        
        if not chunks:
            logger.error(f"❌ No chunks found in {args.input}")
            return False
        
        logger.info(f"✅ Loaded {len(chunks)} chunks")
        
        # Create MLX dataset
        logger.info("🔄 Creating MLX-formatted dataset...")
        results = generator.create_mlx_formatted_dataset(
            chunks, args.output, args.name
        )
        
        if not results["success"]:
            logger.error(f"❌ Failed to create dataset: {results.get('error', 'Unknown error')}")
            return False
        
        # Display results
        logger.info("✅ MLX dataset created successfully!")
        logger.info(f"📊 Examples: Train={results['counts']['train']}, Valid={results['counts']['valid']}, Test={results['counts']['test']}")
        logger.info(f"📁 Output directory: {results['output_dir']}")
        
        # Check validation
        all_valid = all(results['validation'].values())
        if all_valid:
            logger.info("✅ All files validated successfully")
        else:
            logger.warning("⚠️  Some files have validation issues")
            for split, is_valid in results['validation'].items():
                status = "✅" if is_valid else "❌"
                logger.info(f"  {split}: {status}")
        
        return True
    
    def _convert_dataset(self, args) -> bool:
        """Convert existing dataset to MLX format."""
        logger.info(f"🔄 Converting dataset to MLX format")
        logger.info(f"📁 Input: {args.input}")
        logger.info(f"📁 Output: {args.output}")
        
        # Validate input file
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"❌ Input file not found: {args.input}")
            return False
        
        # Convert dataset
        logger.info("🔄 Converting dataset...")
        success = convert_existing_dataset_to_mlx_format(input_path, Path(args.output))
        
        if not success:
            logger.error("❌ Conversion failed")
            return False
        
        # Get file info
        output_path = Path(args.output)
        train_file = output_path / "train.jsonl"
        valid_file = output_path / "valid.jsonl"
        test_file = output_path / "test.jsonl"
        
        try:
            train_count = sum(1 for line in open(train_file, 'r') if line.strip())
            valid_count = sum(1 for line in open(valid_file, 'r') if line.strip())
            test_count = sum(1 for line in open(test_file, 'r') if line.strip())
            
            logger.info("✅ Dataset converted successfully!")
            logger.info(f"📊 Examples: Train={train_count}, Valid={valid_count}, Test={test_count}")
            logger.info(f"📁 Output directory: {args.output}")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not count examples: {e}")
        
        return True
    
    def _validate_dataset(self, args) -> bool:
        """Validate MLX dataset files."""
        logger.info(f"🔍 Validating MLX dataset")
        logger.info(f"📁 Dataset: {args.input}")
        
        dataset_path = Path(args.input)
        if not dataset_path.exists():
            logger.error(f"❌ Dataset directory not found: {args.input}")
            return False
        
        from ...utils.mlx_dataset_formatter import MLXDatasetFormatter
        formatter = MLXDatasetFormatter()
        
        # Check for required files
        required_files = ["train.jsonl", "valid.jsonl", "test.jsonl"]
        missing_files = []
        
        for filename in required_files:
            filepath = dataset_path / filename
            if not filepath.exists():
                missing_files.append(filename)
        
        if missing_files:
            logger.error(f"❌ Missing files: {', '.join(missing_files)}")
            return False
        
        # Validate each file
        logger.info("🔍 Validating files...")
        total_examples = 0
        all_valid = True
        
        for filename in required_files:
            filepath = dataset_path / filename
            
            # Count lines
            try:
                example_count = sum(1 for line in open(filepath, 'r') if line.strip())
                total_examples += example_count
                
                # Validate
                is_valid = formatter.validate_jsonl_file(filepath)
                all_valid = all_valid and is_valid
                
                # Check minimum requirements
                meets_requirements = example_count >= 4
                
                status = "✅ Valid" if is_valid else "❌ Invalid"
                if not meets_requirements:
                    status += f" (only {example_count} examples, need ≥4)"
                    all_valid = False
                
                logger.info(f"  📄 {filename}: {example_count} examples - {status}")
                
            except Exception as e:
                logger.error(f"  📄 {filename}: ❌ Error reading file - {e}")
                all_valid = False
        
        logger.info(f"📊 Total examples: {total_examples}")
        
        if all_valid:
            logger.info("✅ All files are valid for MLX training!")
            return True
        else:
            logger.error("❌ Some files have validation issues")
            return False