"""Convert command for converting documents to Markdown."""

import argparse
import glob
import os
from typing import List

from ...core.converter import DocumentConverter
from ...utils.config import DoclingConfig
from ...utils.logging import get_logger

logger = get_logger(__name__)


class ConvertCommand:
    """Command to convert documents to Markdown using Docling."""
    
    def add_parser(self, subparsers, name: str) -> argparse.ArgumentParser:
        """Add parser for convert command.
        
        Args:
            subparsers: Subparser object to add to
            name: Name of the command
            
        Returns:
            Created parser
        """
        parser = subparsers.add_parser(
            name,
            help='Convert HTML/PDF documents to Markdown',
            description="""
Convert HTML and PDF documents to Markdown using Docling.

This command handles individual document conversion with advanced features:
- Clean HTML by removing headers, footers, and legal content
- Extract tables in multiple formats (CSV, JSON, HTML, Markdown)
- Extract images with configurable formats and scaling
- Generate AI descriptions for images (when enabled)
- Support for multimodal content with flexible chunking strategies
- GPU acceleration for faster processing
- Parallel processing for multiple files
- Advanced table structure recognition and cell matching
- Preserve image metadata and enable picture classification

The output includes Markdown files with YAML frontmatter metadata and
separate directories for extracted tables and images.
""",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Input/Output options
        io_group = parser.add_argument_group('Input/Output')
        io_group.add_argument(
            '--input', '-i',
            required=True,
            help='Input file or directory containing documents'
        )
        io_group.add_argument(
            '--output-dir', '-o',
            required=True,
            help='Output directory for Markdown files'
        )
        io_group.add_argument(
            '--pattern',
            default='*.{html,pdf}',
            help='File pattern to match (default: *.{html,pdf})'
        )
        io_group.add_argument(
            '--exclude',
            help='Pattern to exclude files (optional)'
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
            help='Maximum number of files to process'
        )
        
        # Docling feature options
        feature_group = parser.add_argument_group('Docling Features')
        feature_group.add_argument(
            '--with-accelerator',
            action='store_true',
            help='Use GPU acceleration if available'
        )
        feature_group.add_argument(
            '--extract-tables',
            action='store_true',
            help='Extract tables as separate files'
        )
        feature_group.add_argument(
            '--extract-figures',
            action='store_true',
            help='Extract figures as separate image files'
        )
        feature_group.add_argument(
            '--multimodal',
            action='store_true',
            help='Enable multimodal output (with images)'
        )
        feature_group.add_argument(
            '--custom-convert',
            action='store_true',
            help='Apply custom conversion rules'
        )
        feature_group.add_argument(
            '--threads',
            type=int,
            default=8,
            help='Number of threads for processing (default: 8)'
        )
        
        # Enhanced multimodal options
        multimodal_group = parser.add_argument_group('Advanced Multimodal Options')
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
        
        # Validate that multimodal features are enabled when using advanced options
        if (hasattr(args, 'image_descriptions') and args.image_descriptions and 
            not args.multimodal):
            errors.append("Image descriptions require --multimodal to be enabled")
        
        if (hasattr(args, 'table_format') and args.table_format != 'csv' and 
            not args.extract_tables):
            errors.append("Table format options require --extract-tables to be enabled")
        
        if (hasattr(args, 'image_format') and args.image_format != 'png' and 
            not args.extract_figures):
            errors.append("Image format options require --extract-figures to be enabled")
        
        # Warn about feature dependencies
        if args.image_descriptions and not args.multimodal:
            errors.append("Warning: Image descriptions work best with multimodal processing enabled")
        
        return errors
    
    def _find_input_files(self, input_path: str, pattern: str, exclude: str = None, max_files: int = None) -> List[str]:
        """Find input files matching the pattern.
        
        Args:
            input_path: Input file or directory path
            pattern: File pattern to match
            exclude: Pattern to exclude files
            max_files: Maximum number of files to return
            
        Returns:
            List of input file paths
        """
        if os.path.isfile(input_path):
            return [input_path]
        
        if not os.path.isdir(input_path):
            raise ValueError(f"Input path does not exist: {input_path}")
        
        # Handle multiple file patterns like *.{html,pdf}
        input_files = []
        if '{' in pattern and '}' in pattern:
            # Handle patterns like *.{html,pdf}
            base_pattern = pattern.split('{')[0]
            extensions = pattern.split('{')[1].split('}')[0].split(',')
            
            for ext in extensions:
                file_pattern = base_pattern + ext.strip()
                files = glob.glob(os.path.join(input_path, "**", file_pattern), recursive=True)
                input_files.extend(files)
        else:
            input_files = glob.glob(os.path.join(input_path, "**", pattern), recursive=True)
        
        # Apply exclude pattern if provided
        if exclude:
            input_files = [f for f in input_files if exclude not in f]
        
        # Limit number of files if specified
        if max_files and max_files > 0:
            original_count = len(input_files)
            input_files = input_files[:max_files]
            logger.info(f"Limited to {max_files} files (found {original_count} total)")
        
        return input_files
    
    def run(self, args: argparse.Namespace) -> int:
        """Run the convert command.
        
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
                    if error.startswith("Warning:"):
                        logger.warning(error)
                    else:
                        logger.error(f"  - {error}")
                # Only fail on non-warning errors
                if any(not error.startswith("Warning:") for error in validation_errors):
                    return 1
            
            # Create Docling configuration
            config = DoclingConfig(
                with_accelerator=args.with_accelerator,
                extract_tables=args.extract_tables,
                extract_figures=args.extract_figures,
                multimodal=args.multimodal,
                custom_convert=args.custom_convert,
                num_threads=args.threads,
                # Enhanced multimodal options
                table_format=args.table_format,
                image_format=args.image_format,
                generate_image_descriptions=args.image_descriptions,
                image_scale=args.image_scale,
                enable_picture_classification=not args.no_picture_classification,
                table_cell_matching=not args.no_table_cell_matching,
                preserve_image_metadata=not args.no_image_metadata,
                multimodal_chunk_strategy=args.multimodal_chunk_strategy
            )
            
            # Find input files
            try:
                input_files = self._find_input_files(
                    args.input,
                    args.pattern,
                    args.exclude,
                    args.max_files
                )
            except ValueError as e:
                logger.error(str(e))
                return 1
            
            if not input_files:
                logger.error("No input files found matching the pattern")
                return 1
            
            logger.info(f"Found {len(input_files)} files to convert")
            
            # Log configuration
            logger.info("Conversion Configuration:")
            logger.info(f"  Accelerator: {config.with_accelerator}")
            logger.info(f"  Extract tables: {config.extract_tables} (format: {config.table_format})")
            logger.info(f"  Extract figures: {config.extract_figures} (format: {config.image_format})")
            logger.info(f"  Multimodal: {config.multimodal} (strategy: {config.multimodal_chunk_strategy})")
            logger.info(f"  Custom conversion: {config.custom_convert}")
            logger.info(f"  Threads: {config.num_threads}")
            logger.info(f"  Enhanced Features:")
            logger.info(f"    Image descriptions: {config.generate_image_descriptions}")
            logger.info(f"    Image scale: {config.image_scale}")
            logger.info(f"    Picture classification: {config.enable_picture_classification}")
            logger.info(f"    Table cell matching: {config.table_cell_matching}")
            logger.info(f"    Preserve image metadata: {config.preserve_image_metadata}")
            
            # Create converter and process files
            converter = DocumentConverter(config)
            converted_files, skipped_files = converter.batch_convert(
                input_files,
                args.output_dir,
                args.workers
            )
            
            # Report results
            success_rate = len(converted_files) / len(input_files) * 100 if input_files else 0
            
            logger.info("Conversion Summary:")
            logger.info(f"  Total files: {len(input_files)}")
            logger.info(f"  Successfully converted: {len(converted_files)}")
            logger.info(f"  Skipped: {len(skipped_files)}")
            logger.info(f"  Success rate: {success_rate:.1f}%")
            
            if skipped_files:
                logger.warning("Skipped files:")
                for skipped in skipped_files[:10]:  # Show first 10 skipped files
                    logger.warning(f"  - {os.path.basename(skipped)}")
                if len(skipped_files) > 10:
                    logger.warning(f"  ... and {len(skipped_files) - 10} more")
            
            # Determine exit code
            if len(converted_files) == 0:
                logger.error("No files were successfully converted")
                return 1
            elif success_rate < 50:
                logger.warning("Low success rate - check for errors")
                return 1
            else:
                logger.info("Conversion completed successfully")
                return 0
        
        except Exception as e:
            logger.error(f"Conversion failed: {str(e)}")
            return 1