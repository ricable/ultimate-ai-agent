"""Main CLI entry point for Flow4 document processing pipeline."""

import argparse
import sys
from typing import Optional

from ..utils.config import PipelineConfig, CLIConfig, load_config_from_env, validate_config
from ..utils.logging import setup_logging, get_logger
from .commands.pipeline import PipelineCommand
from .commands.convert import ConvertCommand
from .commands.chunk import ChunkCommand
from .commands.finetune import FinetuneCommand
from .commands.generate import GenerateCommand
from .commands.deduplicate import DeduplicateCommand
from .commands.chat_command import ChatCommand
from .commands.optimize_command import OptimizeCommand
from .commands.format_mlx_command import FormatMLXCommand

logger = get_logger(__name__)


class Flow4CLI:
    """Main CLI application for Flow4."""
    
    def __init__(self):
        """Initialize the CLI application."""
        self.commands = {
            'pipeline': PipelineCommand(),
            'convert': ConvertCommand(),
            'chunk': ChunkCommand(),
            'finetune': FinetuneCommand(),
            'generate': GenerateCommand(),
            'deduplicate': DeduplicateCommand(),
            'chat': ChatCommand(),
            'optimize': OptimizeCommand(),
            'format-mlx': FormatMLXCommand(),
        }
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser with subcommands."""
        parser = argparse.ArgumentParser(
            prog='flow4',
            description='Flow4: Document Processing Pipeline - Convert HTML/PDF to Markdown and create RAG datasets',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run complete pipeline on ZIP file
  flow4 pipeline --input data/documents.zip --output-dir output

  # Convert specific files
  flow4 convert --input docs/ --output-dir markdown/ --workers 8

  # Chunk existing markdown files
  flow4 chunk --input combined.md --output-dir chunks/ --size 500

  # Generate advanced datasets using Augmentoolkit
  flow4 generate --input output/chunks --type factual --model medium

  # Fine-tune language models on generated datasets
  flow4 finetune --dataset-path output/rag/finetune_dataset.jsonl --chat

  # Chat with existing fine-tuned model
  flow4 finetune --chat-only --adapter-dir fine_tuned_adapters

  # Interactive chat with processed documents
  flow4 chat docs --chunks-dir output/chunks

  # Chat with fine-tuned model
  flow4 chat model --adapter-path fine_tuned_adapters

  # Optimize chunks for better training quality
  flow4 optimize chunks --input-file output/rag/rag_dataset.json

  # Generate optimized fine-tuning dataset
  flow4 optimize dataset --chunks-dir output/chunks

  # Compare dataset quality metrics
  flow4 optimize compare dataset1.json dataset2.json

For more information on each command, use:
  flow4 <command> --help
"""
        )
        
        # Global options
        parser.add_argument(
            '--version', 
            action='version', 
            version='Flow4 v0.2.0'
        )
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )
        parser.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug logging'
        )
        parser.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress all output except errors'
        )
        parser.add_argument(
            '--log-file',
            help='Log to file in addition to console'
        )
        
        # Create subcommands
        subparsers = parser.add_subparsers(
            dest='command',
            help='Available commands',
            metavar='COMMAND'
        )
        
        # Add command parsers
        for name, command in self.commands.items():
            command.add_parser(subparsers, name)
        
        return parser
    
    def _setup_logging_from_args(self, args: argparse.Namespace) -> None:
        """Set up logging based on command line arguments."""
        if args.quiet:
            level = "ERROR"
        elif args.debug:
            level = "DEBUG"
        elif args.verbose:
            level = "INFO"
        else:
            level = "WARNING"
        
        setup_logging(
            level=level,
            include_file=bool(args.log_file),
            log_file=args.log_file or "flow4.log"
        )
    
    def run(self, argv: Optional[list] = None) -> int:
        """Run the CLI application.
        
        Args:
            argv: Command line arguments (defaults to sys.argv)
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            args = self.parser.parse_args(argv)
            
            # Set up logging
            self._setup_logging_from_args(args)
            
            # Check if a command was provided
            if not args.command:
                self.parser.print_help()
                return 1
            
            # Get the command handler
            command = self.commands.get(args.command)
            if not command:
                logger.error(f"Unknown command: {args.command}")
                return 1
            
            # Run the command
            logger.debug(f"Running command: {args.command}")
            return command.run(args)
        
        except KeyboardInterrupt:
            logger.info("Operation cancelled by user")
            return 130  # Standard exit code for SIGINT
        
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            if args.debug if 'args' in locals() else False:
                import traceback
                logger.debug(traceback.format_exc())
            return 1


def main():
    """Main entry point for the flow4 CLI."""
    cli = Flow4CLI()
    exit_code = cli.run()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()