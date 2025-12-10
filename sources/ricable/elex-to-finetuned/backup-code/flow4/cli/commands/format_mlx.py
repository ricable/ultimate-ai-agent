"""
Flow4 MLX Dataset Formatter Command

CLI command for formatting datasets for MLX fine-tuning, addressing all the
formatting issues encountered during development.

Usage:
    flow4 format-mlx --input chunks_dir --output mlx_dataset_dir
    flow4 format-mlx --convert existing_dataset.jsonl --output mlx_dataset_dir
"""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ...core.augmentoolkit_generator import AugmentoolkitGenerator, create_augmentoolkit_config
from ...utils.logging import get_logger
from ...utils.mlx_dataset_formatter import convert_existing_dataset_to_mlx_format

logger = get_logger(__name__)
console = Console()

app = typer.Typer(help="Format datasets for MLX fine-tuning with proper JSONL validation")


@app.command("create")
def create_mlx_dataset(
    input_dir: str = typer.Option(
        ..., "--input", "-i",
        help="Directory containing Flow4 chunks (JSON files)"
    ),
    output_dir: str = typer.Option(
        ..., "--output", "-o",
        help="Output directory for MLX-formatted dataset"
    ),
    dataset_name: str = typer.Option(
        "mlx_dataset", "--name", "-n",
        help="Name for the dataset"
    ),
    model_name: str = typer.Option(
        "mlx-community/Llama-3.2-3B-Instruct-4bit", "--model", "-m",
        help="Model name for configuration"
    ),
    max_chunks: Optional[int] = typer.Option(
        None, "--max-chunks",
        help="Maximum number of chunks to process"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Enable verbose logging"
    )
):
    """
    Create MLX-formatted dataset from Flow4 chunks.
    
    This command addresses all the MLX dataset formatting issues:
    - Proper JSONL format with validation
    - MLX batch size requirements (minimum 4 examples per split)
    - Text cleaning and encoding
    - Automatic train/valid/test splitting
    """
    try:
        console.print(f"[bold blue]🔧 Creating MLX dataset from chunks[/bold blue]")
        console.print(f"📁 Input: {input_dir}")
        console.print(f"📁 Output: {output_dir}")
        
        # Validate input directory
        input_path = Path(input_dir)
        if not input_path.exists():
            console.print(f"[bold red]❌ Input directory not found: {input_dir}[/bold red]")
            raise typer.Exit(1)
        
        # Create Augmentoolkit generator
        config = create_augmentoolkit_config(
            model_name=model_name,
            mode="mlx",
            dataset_context="technical documentation"
        )
        generator = AugmentoolkitGenerator(config)
        
        # Load chunks
        console.print("📚 Loading chunks...")
        chunks = generator.load_chunks_from_directory(str(input_path), max_chunks)
        
        if not chunks:
            console.print(f"[bold red]❌ No chunks found in {input_dir}[/bold red]")
            raise typer.Exit(1)
        
        console.print(f"✅ Loaded {len(chunks)} chunks")
        
        # Create MLX dataset
        console.print("🔄 Creating MLX-formatted dataset...")
        results = generator.create_mlx_formatted_dataset(
            chunks, output_dir, dataset_name
        )
        
        if not results["success"]:
            console.print(f"[bold red]❌ Failed to create dataset: {results.get('error', 'Unknown error')}[/bold red]")
            raise typer.Exit(1)
        
        # Display results
        console.print("[bold green]✅ MLX dataset created successfully![/bold green]")
        
        # Create results table
        table = Table(title="MLX Dataset Summary")
        table.add_column("Split", style="cyan")
        table.add_column("File", style="yellow")
        table.add_column("Examples", style="green")
        table.add_column("Valid", style="blue")
        
        for split in ["train", "valid", "test"]:
            table.add_row(
                split.title(),
                results["files"][split],
                str(results["counts"][split]),
                "✅" if results["validation"][f"{split}_valid"] else "❌"
            )
        
        console.print(table)
        console.print(f"📊 Total examples: {results['counts']['total']}")
        console.print(f"📁 Output directory: {results['output_dir']}")
        
        if verbose:
            console.print("\n[bold]🔍 File Validation Details:[/bold]")
            for split, is_valid in results["validation"].items():
                status = "✅ Valid" if is_valid else "❌ Invalid"
                console.print(f"  {split}: {status}")
        
    except Exception as e:
        logger.error(f"Error creating MLX dataset: {e}")
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        raise typer.Exit(1)


@app.command("convert")
def convert_existing_dataset(
    input_file: str = typer.Option(
        ..., "--input", "-i",
        help="Existing dataset file to convert (JSONL format)"
    ),
    output_dir: str = typer.Option(
        ..., "--output", "-o",
        help="Output directory for MLX-formatted dataset"
    ),
    dataset_name: str = typer.Option(
        "converted_dataset", "--name", "-n",
        help="Name for the converted dataset"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Enable verbose logging"
    )
):
    """
    Convert existing dataset to MLX format.
    
    Handles conversion from various formats:
    - OpenAI chat format (messages array)
    - Prompt-completion format
    - Other JSONL formats
    
    Addresses common issues:
    - JSON encoding errors
    - MLX batch size requirements
    - Proper JSONL validation
    """
    try:
        console.print(f"[bold blue]🔄 Converting dataset to MLX format[/bold blue]")
        console.print(f"📁 Input: {input_file}")
        console.print(f"📁 Output: {output_dir}")
        
        # Validate input file
        input_path = Path(input_file)
        if not input_path.exists():
            console.print(f"[bold red]❌ Input file not found: {input_file}[/bold red]")
            raise typer.Exit(1)
        
        # Convert dataset
        console.print("🔄 Converting dataset...")
        success = convert_existing_dataset_to_mlx_format(input_path, Path(output_dir))
        
        if not success:
            console.print("[bold red]❌ Conversion failed[/bold red]")
            raise typer.Exit(1)
        
        # Get file info
        output_path = Path(output_dir)
        train_file = output_path / "train.jsonl"
        valid_file = output_path / "valid.jsonl"
        test_file = output_path / "test.jsonl"
        
        train_count = sum(1 for line in open(train_file, 'r') if line.strip())
        valid_count = sum(1 for line in open(valid_file, 'r') if line.strip())
        test_count = sum(1 for line in open(test_file, 'r') if line.strip())
        
        # Display results
        console.print("[bold green]✅ Dataset converted successfully![/bold green]")
        
        # Create results table
        table = Table(title="Converted Dataset Summary")
        table.add_column("Split", style="cyan")
        table.add_column("File", style="yellow")
        table.add_column("Examples", style="green")
        
        table.add_row("Train", str(train_file), str(train_count))
        table.add_row("Valid", str(valid_file), str(valid_count))
        table.add_row("Test", str(test_file), str(test_count))
        
        console.print(table)
        console.print(f"📊 Total examples: {train_count + valid_count + test_count}")
        console.print(f"📁 Output directory: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error converting dataset: {e}")
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        raise typer.Exit(1)


@app.command("validate")
def validate_mlx_dataset(
    dataset_dir: str = typer.Option(
        ..., "--input", "-i",
        help="Directory containing MLX dataset files"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Enable verbose logging"
    )
):
    """
    Validate MLX dataset files for proper formatting.
    
    Checks:
    - JSONL format validation
    - Required fields presence
    - MLX batch size requirements
    - Text encoding issues
    """
    try:
        console.print(f"[bold blue]🔍 Validating MLX dataset[/bold blue]")
        console.print(f"📁 Dataset: {dataset_dir}")
        
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            console.print(f"[bold red]❌ Dataset directory not found: {dataset_dir}[/bold red]")
            raise typer.Exit(1)
        
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
            console.print(f"[bold red]❌ Missing files: {', '.join(missing_files)}[/bold red]")
            raise typer.Exit(1)
        
        # Validate each file
        table = Table(title="Dataset Validation Results")
        table.add_column("File", style="cyan")
        table.add_column("Examples", style="yellow")
        table.add_column("Valid", style="green")
        table.add_column("Status", style="blue")
        
        total_examples = 0
        all_valid = True
        
        for filename in required_files:
            filepath = dataset_path / filename
            
            # Count lines
            example_count = sum(1 for line in open(filepath, 'r') if line.strip())
            total_examples += example_count
            
            # Validate
            is_valid = formatter.validate_jsonl_file(filepath)
            all_valid = all_valid and is_valid
            
            # Check minimum requirements
            meets_requirements = example_count >= 4
            status = "✅ Good" if is_valid and meets_requirements else "❌ Issues"
            
            if not meets_requirements:
                status += f" (< 4 examples)"
            
            table.add_row(
                filename,
                str(example_count),
                "✅" if is_valid else "❌",
                status
            )
        
        console.print(table)
        console.print(f"📊 Total examples: {total_examples}")
        
        if all_valid:
            console.print("[bold green]✅ All files are valid for MLX training![/bold green]")
        else:
            console.print("[bold red]❌ Some files have validation issues[/bold red]")
            raise typer.Exit(1)
        
    except Exception as e:
        logger.error(f"Error validating dataset: {e}")
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()