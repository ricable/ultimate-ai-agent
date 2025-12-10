"""Chat interface commands for Flow4."""

import sys
import os
from pathlib import Path
from typing import Optional

import click

from ...core.chat_interface import DocumentChatInterface, ModelChatInterface
from ...utils.logging import get_logger

logger = get_logger(__name__)


@click.group()
def chat():
    """Chat interfaces for documents and models."""
    pass


@chat.command()
@click.option(
    "--chunks-dir",
    default="output/chunks",
    help="Directory containing chunk files",
    type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option(
    "--export-qa",
    help="Export Q&A pairs to specified JSON file instead of running chat",
    type=click.Path()
)
def docs(chunks_dir: str, export_qa: Optional[str]):
    """Chat with processed documents.
    
    Interactive chat interface for asking questions about processed documents.
    Uses simple keyword-based search to find relevant chunks and generate answers.
    """
    try:
        chat_interface = DocumentChatInterface(chunks_dir)
        
        if export_qa:
            chat_interface.export_qa_dataset(export_qa)
            click.echo(f"✅ Q&A dataset exported to: {export_qa}")
        else:
            chat_interface.run_chat_interface()
            
    except Exception as e:
        logger.error(f"Error in document chat: {e}")
        sys.exit(1)


@chat.command()
@click.option(
    "--model-name",
    default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    help="Base model name for fine-tuned model"
)
@click.option(
    "--adapter-path",
    default="fine_tuned_adapters",
    help="Path to fine-tuned LoRA adapters",
    type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option(
    "--max-tokens",
    default=150,
    help="Maximum tokens to generate",
    type=int
)
@click.option(
    "--temperature",
    default=0.7,
    help="Temperature for text generation",
    type=float
)
def model(model_name: str, adapter_path: str, max_tokens: int, temperature: float):
    """Chat with fine-tuned model.
    
    Interactive chat interface for conversing with a fine-tuned language model.
    Requires MLX and a fine-tuned model with LoRA adapters.
    """
    try:
        # Check if adapter path exists
        if not os.path.exists(adapter_path):
            click.echo(f"❌ Adapter path not found: {adapter_path}")
            click.echo("Run 'flow4 finetune' first to create adapters.")
            sys.exit(1)
        
        # Check if MLX is available
        try:
            import mlx
        except ImportError:
            click.echo("❌ MLX is not available. This command requires Apple Silicon with MLX installed.")
            click.echo("Install with: pip install mlx-lm")
            sys.exit(1)
        
        chat_interface = ModelChatInterface(model_name, adapter_path)
        chat_interface.run_chat_interface()
        
    except Exception as e:
        logger.error(f"Error in model chat: {e}")
        sys.exit(1)


if __name__ == "__main__":
    chat()