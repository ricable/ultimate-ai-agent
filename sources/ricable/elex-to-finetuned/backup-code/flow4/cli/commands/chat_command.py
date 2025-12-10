"""Chat command for interacting with documents and models."""

import argparse
import os
import sys

from ...core.chat_interface import DocumentChatInterface, ModelChatInterface
from ...utils.logging import get_logger

logger = get_logger(__name__)


class ChatCommand:
    """Command for interactive chat interfaces."""
    
    def add_parser(self, subparsers, name: str) -> argparse.ArgumentParser:
        """Add parser for chat command.
        
        Args:
            subparsers: Subparser object to add to
            name: Name of the command
            
        Returns:
            Created parser
        """
        parser = subparsers.add_parser(
            name,
            help='Interactive chat with documents or models',
            description="""
Interactive chat interfaces for Flow4.

This command provides two chat modes:
1. Document chat: Ask questions about processed documents using keyword search
2. Model chat: Converse with fine-tuned language models using MLX

The document chat loads chunks created by Flow4 and provides simple Q&A.
The model chat requires a fine-tuned model with LoRA adapters.
""",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Subcommand for chat type
        subparsers_chat = parser.add_subparsers(
            dest='chat_type',
            help='Type of chat interface',
            metavar='TYPE'
        )
        
        # Document chat subcommand
        docs_parser = subparsers_chat.add_parser(
            'docs',
            help='Chat with processed documents'
        )
        docs_parser.add_argument(
            '--chunks-dir',
            default='output/chunks',
            help='Directory containing chunk files',
            metavar='DIR'
        )
        docs_parser.add_argument(
            '--export-qa',
            help='Export Q&A pairs to JSON file instead of running chat',
            metavar='FILE'
        )
        
        # Model chat subcommand
        model_parser = subparsers_chat.add_parser(
            'model', 
            help='Chat with fine-tuned model'
        )
        model_parser.add_argument(
            '--model-name',
            default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            help='Base model name for fine-tuned model',
            metavar='NAME'
        )
        model_parser.add_argument(
            '--adapter-path',
            default='fine_tuned_adapters',
            help='Path to fine-tuned LoRA adapters',
            metavar='PATH'
        )
        model_parser.add_argument(
            '--max-tokens',
            type=int,
            default=150,
            help='Maximum tokens to generate',
            metavar='N'
        )
        model_parser.add_argument(
            '--temperature',
            type=float,
            default=0.7,
            help='Temperature for text generation',
            metavar='TEMP'
        )
        
        return parser
    
    def run(self, args: argparse.Namespace) -> int:
        """Run the chat command.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            if not args.chat_type:
                logger.error("Please specify chat type: 'docs' or 'model'")
                return 1
            
            if args.chat_type == 'docs':
                return self._run_document_chat(args)
            elif args.chat_type == 'model':
                return self._run_model_chat(args)
            else:
                logger.error(f"Unknown chat type: {args.chat_type}")
                return 1
                
        except Exception as e:
            logger.error(f"Error in chat command: {e}")
            return 1
    
    def _run_document_chat(self, args: argparse.Namespace) -> int:
        """Run document chat interface."""
        if not os.path.exists(args.chunks_dir):
            logger.error(f"Chunks directory not found: {args.chunks_dir}")
            logger.error("Run the Flow4 pipeline first to generate chunks.")
            return 1
        
        try:
            chat_interface = DocumentChatInterface(args.chunks_dir)
            
            if args.export_qa:
                chat_interface.export_qa_dataset(args.export_qa)
                logger.info(f"✅ Q&A dataset exported to: {args.export_qa}")
            else:
                chat_interface.run_chat_interface()
                
            return 0
            
        except Exception as e:
            logger.error(f"Error in document chat: {e}")
            return 1
    
    def _run_model_chat(self, args: argparse.Namespace) -> int:
        """Run model chat interface."""
        # Check if adapter path exists
        if not os.path.exists(args.adapter_path):
            logger.error(f"Adapter path not found: {args.adapter_path}")
            logger.error("Run 'flow4 finetune' first to create adapters.")
            return 1
        
        # Check if MLX is available
        try:
            import mlx
        except ImportError:
            logger.error("MLX is not available. This command requires Apple Silicon with MLX installed.")
            logger.error("Install with: pip install mlx-lm")
            return 1
        
        try:
            chat_interface = ModelChatInterface(args.model_name, args.adapter_path)
            chat_interface.run_chat_interface()
            return 0
            
        except Exception as e:
            logger.error(f"Error in model chat: {e}")
            return 1