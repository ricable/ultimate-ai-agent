"""Dataset optimization commands for Flow4."""

import sys
import json
from pathlib import Path
from typing import Optional

import click

from ...core.dataset_generator import (
    ChunkOptimizer, 
    OptimizedFineTuneDatasetGenerator,
    FineTuningDatasetGenerator,
    DatasetConfig,
    DatasetComparator
)
from ...utils.logging import get_logger

logger = get_logger(__name__)


@click.group()
def optimize():
    """Dataset optimization and quality improvement commands."""
    pass


@optimize.command()
@click.option(
    "--input-file",
    default="output/rag/rag_dataset.json",
    help="Input RAG dataset file",
    type=click.Path(exists=True)
)
@click.option(
    "--output-dir",
    default="output/optimized",
    help="Output directory for optimized dataset",
    type=click.Path()
)
def chunks(input_file: str, output_dir: str):
    """Optimize chunks for better training dataset quality.
    
    Cleans and filters chunks to improve the quality of fine-tuning datasets.
    Removes malformed content, optimizes text structure, and filters low-quality chunks.
    """
    try:
        logger.info(f"Loading chunks from: {input_file}")
        with open(input_file, 'r') as f:
            rag_data = json.load(f)
            
        chunks = rag_data.get('chunks', [])
        logger.info(f"Original dataset: {len(chunks)} chunks")
        
        # Optimize chunks
        optimizer = ChunkOptimizer()
        optimized_chunks = optimizer.optimize_dataset(chunks)
        
        # Create optimized RAG dataset
        optimized_rag_data = {
            "chunks": optimized_chunks,
            "total_chunks": len(optimized_chunks),
            "optimization_applied": True,
            "original_chunk_count": len(chunks),
            "optimization_stats": {
                "kept_chunks": len(optimized_chunks),
                "removed_chunks": len(chunks) - len(optimized_chunks),
                "removal_rate": (len(chunks) - len(optimized_chunks)) / len(chunks) * 100,
                "avg_length_before": sum(len(c.get('text', '')) for c in chunks) / len(chunks) if chunks else 0,
                "avg_length_after": sum(len(c.get('text', '')) for c in optimized_chunks) / len(optimized_chunks) if optimized_chunks else 0
            }
        }
        
        # Save optimized chunks
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        optimized_rag_path = output_path / "rag_dataset_optimized.json"
        with open(optimized_rag_path, 'w', encoding='utf-8') as f:
            json.dump(optimized_rag_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Optimized RAG dataset saved to: {optimized_rag_path}")
        logger.info(f"Optimization stats: {optimized_rag_data['optimization_stats']}")
        
        click.echo(f"✅ Optimized {len(optimized_chunks)} chunks")
        click.echo(f"📉 Removed {len(chunks) - len(optimized_chunks)} low-quality chunks")
        click.echo(f"💾 Saved to: {optimized_rag_path}")
        
    except Exception as e:
        logger.error(f"Error optimizing chunks: {e}")
        sys.exit(1)


@optimize.command()
@click.option(
    "--chunks-dir",
    default="output/chunks",
    help="Directory containing chunks",
    type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option(
    "--input-file",
    help="Input RAG dataset file (alternative to chunks-dir)",
    type=click.Path(exists=True)
)
@click.option(
    "--output-dir",
    default="output/optimized",
    help="Output directory for optimized dataset",
    type=click.Path()
)
@click.option(
    "--dataset-context",
    default="telecommunications technical documentation and 5G network specifications",
    help="Context description for the dataset"
)
def dataset(chunks_dir: Optional[str], input_file: Optional[str], 
           output_dir: str, dataset_context: str):
    """Generate optimized fine-tuning dataset with quality filtering.
    
    Creates high-quality instruction-response pairs from chunks with:
    - Intelligent response generation
    - Domain-specific instruction templates
    - Technical entity extraction
    - Quality scoring and filtering
    - No data leakage
    """
    try:
        # Load chunks from either directory or file
        if input_file:
            logger.info(f"Loading chunks from RAG dataset: {input_file}")
            with open(input_file, 'r') as f:
                rag_data = json.load(f)
            chunks = rag_data.get('chunks', [])
        else:
            logger.info(f"Loading chunks from directory: {chunks_dir}")
            chunks_path = Path(chunks_dir)
            chunk_files = sorted(chunks_path.glob("chunk_*.json"))
            
            chunks = []
            for chunk_file in chunk_files:
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk_data = json.load(f)
                        chunks.append(chunk_data)
                except Exception as e:
                    logger.warning(f"Failed to load {chunk_file}: {e}")
        
        if not chunks:
            click.echo("❌ No chunks found!")
            sys.exit(1)
            
        logger.info(f"Processing {len(chunks)} chunks...")
        
        # Generate optimized dataset
        generator = OptimizedFineTuneDatasetGenerator()
        training_data = generator.create_quality_instruction_pairs(chunks)
        
        logger.info(f"Generated {len(training_data)} high-quality training examples")
        
        # Sort by quality score
        training_data.sort(key=lambda x: x['metadata']['quality_score'], reverse=True)
        
        # Create output
        output = {
            "training_data": training_data,
            "total_examples": len(training_data),
            "format": "optimized_instruction_response",
            "description": "High-quality instruction-response pairs for technical documentation",
            "version": "2.0.0",
            "dataset_context": dataset_context,
            "optimization_features": [
                "Intelligent response generation",
                "Domain-specific instruction templates", 
                "Technical entity extraction",
                "Quality filtering",
                "No data leakage",
                "Contextual response variation"
            ],
            "quality_stats": {
                "avg_quality_score": sum(item['metadata']['quality_score'] for item in training_data) / len(training_data) if training_data else 0,
                "high_quality_count": len([item for item in training_data if item['metadata']['quality_score'] >= 3.0]),
                "avg_response_length": sum(len(item['output']) for item in training_data) / len(training_data) if training_data else 0
            }
        }
        
        # Save optimized dataset
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        dataset_path = output_path / "finetune_dataset_optimized.json"
        
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Optimized dataset saved to: {dataset_path}")
        
        # Also create JSONL format for MLX compatibility
        try:
            # Create a temporary config and generator for JSONL conversion
            temp_config = DatasetConfig(
                output_dir=str(output_path),
                include_jsonl_format=True,
                include_mlx_format=True,
                dataset_context=dataset_context
            )
            temp_generator = FineTuningDatasetGenerator(temp_config)
            
            # Convert to JSONL
            jsonl_path = output_path / "finetune_dataset_optimized.jsonl"
            temp_generator.save_jsonl_format(training_data, jsonl_path)
            
            # Create MLX format
            mlx_dir = temp_generator.create_mlx_dataset_format(training_data, output_path)
            
            click.echo(f"📄 JSONL format: {jsonl_path}")
            click.echo(f"🚀 MLX format: {mlx_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to create JSONL/MLX formats: {e}")
        
        logger.info(f"Quality stats: {output['quality_stats']}")
        
        click.echo(f"✅ Generated {len(training_data)} optimized training examples")
        click.echo(f"📊 Average quality score: {output['quality_stats']['avg_quality_score']:.2f}")
        click.echo(f"🏆 High quality examples (≥3.0): {output['quality_stats']['high_quality_count']}")
        click.echo(f"💾 Saved to: {dataset_path}")
        
    except Exception as e:
        logger.error(f"Error generating optimized dataset: {e}")
        sys.exit(1)


@optimize.command()
@click.option(
    "--chunks-dir",
    default="output/chunks",
    help="Directory containing chunks",
    type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option(
    "--output-dir",
    default="output/fine_tuning_datasets",
    help="Output directory",
    type=click.Path()
)
@click.option(
    "--max-chunks",
    type=int,
    help="Maximum chunks to process"
)
@click.option(
    "--no-deduplication",
    is_flag=True,
    help="Disable deduplication"
)
@click.option(
    "--dedupe-strategy",
    default="progressive",
    type=click.Choice(["exact", "normalized", "hash", "semantic", "progressive"]),
    help="Deduplication strategy"
)
@click.option(
    "--qa-pairs-per-chunk",
    default=3,
    type=int,
    help="Q&A pairs per chunk"
)
@click.option(
    "--dataset-context",
    default="telecommunications technical documentation and 5G network specifications",
    help="Context description for the dataset"
)
def generate(chunks_dir: str, output_dir: str, max_chunks: Optional[int],
            no_deduplication: bool, dedupe_strategy: str, qa_pairs_per_chunk: int,
            dataset_context: str):
    """Generate comprehensive fine-tuning datasets with deduplication.
    
    Creates instruction-response pairs from chunks with multiple formats:
    - Standard instruction format
    - Chat format
    - RAG format
    - Train/validation splits
    """
    try:
        # Create configuration
        config = DatasetConfig(
            chunks_dir=chunks_dir,
            output_dir=output_dir,
            max_chunks=max_chunks,
            enable_deduplication=not no_deduplication,
            dedupe_strategy=dedupe_strategy,
            qa_pairs_per_chunk=qa_pairs_per_chunk,
            dataset_context=dataset_context
        )
        
        # Generate datasets
        generator = FineTuningDatasetGenerator(config)
        
        datasets = generator.generate_complete_dataset()
        generator.save_datasets(datasets)
        
        click.echo(f"✅ Fine-tuning dataset generation completed!")
        click.echo(f"📁 Output directory: {config.output_dir}")
        click.echo(f"📊 Total Q&A pairs: {len(datasets['instruction_dataset'])}")
        
        if config.enable_deduplication:
            dedup_stats = datasets['deduplication_report']
            if dedup_stats.get('status') != 'skipped':
                click.echo(f"🔍 Deduplication applied with {dedupe_strategy} strategy")
        
    except Exception as e:
        logger.error(f"Error generating dataset: {e}")
        sys.exit(1)


@optimize.command()
@click.argument("dataset_paths", nargs=-1, type=click.Path(exists=True))
def compare(dataset_paths):
    """Compare dataset quality across multiple datasets.
    
    Analyzes and compares quality metrics for multiple fine-tuning datasets.
    
    DATASET_PATHS: Paths to dataset JSON files to compare.
    """
    if not dataset_paths:
        # Default comparison paths
        dataset_paths = [
            "output/rag/finetune_instruction_response.json",
            "output/optimized/finetune_dataset_optimized.json", 
            "output/final/finetune_dataset_deduplicated.json"
        ]
        
        # Filter to existing files
        dataset_paths = [Path(p) for p in dataset_paths if Path(p).exists()]
        
        if not dataset_paths:
            click.echo("❌ No datasets found to compare.")
            click.echo("Specify dataset paths or ensure default datasets exist.")
            sys.exit(1)
    else:
        dataset_paths = [Path(p) for p in dataset_paths]
    
    try:
        comparator = DatasetComparator()
        comparator.compare_datasets(dataset_paths)
        
        click.echo("\n" + "=" * 60)
        click.echo("OPTIMIZATION IMPROVEMENTS:")
        click.echo("=" * 60)
        
        improvements = [
            "✅ Fixed input/output data leakage",
            "✅ Implemented intelligent response generation", 
            "✅ Added domain-specific instruction templates",
            "✅ Improved chunk content cleaning",
            "✅ Added quality scoring and filtering",
            "✅ Semantic deduplication capabilities",
            "✅ Technical entity extraction for better templates",
            "✅ Content structure optimization",
            "✅ Response variation and summarization",
            "✅ Quality validation pipeline"
        ]
        
        for improvement in improvements:
            click.echo(improvement)
        
    except Exception as e:
        logger.error(f"Error comparing datasets: {e}")
        sys.exit(1)


if __name__ == "__main__":
    optimize()