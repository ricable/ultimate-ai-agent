"""Optimize command for dataset optimization and quality improvement."""

import argparse
import json
import sys
from pathlib import Path

from ...core.dataset_generator import (
    ChunkOptimizer, 
    OptimizedFineTuneDatasetGenerator,
    FineTuningDatasetGenerator,
    DatasetConfig,
    DatasetComparator
)
from ...utils.logging import get_logger

logger = get_logger(__name__)


class OptimizeCommand:
    """Command for dataset optimization and quality improvement."""
    
    def add_parser(self, subparsers, name: str) -> argparse.ArgumentParser:
        """Add parser for optimize command.
        
        Args:
            subparsers: Subparser object to add to
            name: Name of the command
            
        Returns:
            Created parser
        """
        parser = subparsers.add_parser(
            name,
            help='Optimize datasets and improve quality',
            description="""
Dataset optimization and quality improvement for Flow4.

This command provides several optimization sub-commands:
1. chunks: Clean and filter chunks for better training quality
2. dataset: Generate optimized fine-tuning datasets with quality filtering
3. generate: Create comprehensive datasets with deduplication
4. compare: Compare quality metrics across multiple datasets

All optimization features include intelligent content filtering,
deduplication, and quality scoring.
""",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Subcommand for optimization type
        subparsers_opt = parser.add_subparsers(
            dest='optimize_type',
            help='Type of optimization',
            metavar='TYPE'
        )
        
        # Chunk optimization subcommand
        chunks_parser = subparsers_opt.add_parser(
            'chunks',
            help='Optimize chunks for better training quality'
        )
        chunks_parser.add_argument(
            '--input-file',
            default='output/rag/rag_dataset.json',
            help='Input RAG dataset file',
            metavar='FILE'
        )
        chunks_parser.add_argument(
            '--output-dir',
            default='output/optimized',
            help='Output directory for optimized dataset',
            metavar='DIR'
        )
        
        # Dataset optimization subcommand
        dataset_parser = subparsers_opt.add_parser(
            'dataset',
            help='Generate optimized fine-tuning dataset with quality filtering'
        )
        dataset_parser.add_argument(
            '--chunks-dir',
            default='output/chunks',
            help='Directory containing chunks',
            metavar='DIR'
        )
        dataset_parser.add_argument(
            '--input-file',
            help='Input RAG dataset file (alternative to chunks-dir)',
            metavar='FILE'
        )
        dataset_parser.add_argument(
            '--output-dir',
            default='output/optimized',
            help='Output directory for optimized dataset',
            metavar='DIR'
        )
        dataset_parser.add_argument(
            '--dataset-context',
            default='telecommunications technical documentation and 5G network specifications',
            help='Context description for the dataset'
        )
        
        # Generate comprehensive datasets
        generate_parser = subparsers_opt.add_parser(
            'generate',
            help='Generate comprehensive fine-tuning datasets with deduplication'
        )
        generate_parser.add_argument(
            '--chunks-dir',
            default='output/chunks',
            help='Directory containing chunks',
            metavar='DIR'
        )
        generate_parser.add_argument(
            '--output-dir',
            default='output/fine_tuning_datasets',
            help='Output directory',
            metavar='DIR'
        )
        generate_parser.add_argument(
            '--max-chunks',
            type=int,
            help='Maximum chunks to process',
            metavar='N'
        )
        generate_parser.add_argument(
            '--no-deduplication',
            action='store_true',
            help='Disable deduplication'
        )
        generate_parser.add_argument(
            '--dedupe-strategy',
            default='progressive',
            choices=['exact', 'normalized', 'hash', 'semantic', 'progressive'],
            help='Deduplication strategy'
        )
        generate_parser.add_argument(
            '--qa-pairs-per-chunk',
            type=int,
            default=3,
            help='Q&A pairs per chunk',
            metavar='N'
        )
        generate_parser.add_argument(
            '--dataset-context',
            default='telecommunications technical documentation and 5G network specifications',
            help='Context description for the dataset'
        )
        
        # Compare datasets
        compare_parser = subparsers_opt.add_parser(
            'compare',
            help='Compare dataset quality across multiple datasets'
        )
        compare_parser.add_argument(
            'dataset_paths',
            nargs='*',
            help='Paths to dataset JSON files to compare'
        )
        
        return parser
    
    def run(self, args: argparse.Namespace) -> int:
        """Run the optimize command.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            if not args.optimize_type:
                logger.error("Please specify optimization type: 'chunks', 'dataset', 'generate', or 'compare'")
                return 1
            
            if args.optimize_type == 'chunks':
                return self._optimize_chunks(args)
            elif args.optimize_type == 'dataset':
                return self._optimize_dataset(args)
            elif args.optimize_type == 'generate':
                return self._generate_dataset(args)
            elif args.optimize_type == 'compare':
                return self._compare_datasets(args)
            else:
                logger.error(f"Unknown optimization type: {args.optimize_type}")
                return 1
                
        except Exception as e:
            logger.error(f"Error in optimize command: {e}")
            return 1
    
    def _optimize_chunks(self, args: argparse.Namespace) -> int:
        """Optimize chunks for better training quality."""
        if not Path(args.input_file).exists():
            logger.error(f"Input file not found: {args.input_file}")
            return 1
        
        try:
            logger.info(f"Loading chunks from: {args.input_file}")
            with open(args.input_file, 'r') as f:
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
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            optimized_rag_path = output_path / "rag_dataset_optimized.json"
            with open(optimized_rag_path, 'w', encoding='utf-8') as f:
                json.dump(optimized_rag_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Optimized RAG dataset saved to: {optimized_rag_path}")
            logger.info(f"Optimization stats: {optimized_rag_data['optimization_stats']}")
            
            logger.info(f"✅ Optimized {len(optimized_chunks)} chunks")
            logger.info(f"📉 Removed {len(chunks) - len(optimized_chunks)} low-quality chunks")
            logger.info(f"💾 Saved to: {optimized_rag_path}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Error optimizing chunks: {e}")
            return 1
    
    def _optimize_dataset(self, args: argparse.Namespace) -> int:
        """Generate optimized fine-tuning dataset with quality filtering."""
        try:
            # Load chunks from either directory or file
            if args.input_file:
                if not Path(args.input_file).exists():
                    logger.error(f"Input file not found: {args.input_file}")
                    return 1
                
                logger.info(f"Loading chunks from RAG dataset: {args.input_file}")
                with open(args.input_file, 'r') as f:
                    rag_data = json.load(f)
                chunks = rag_data.get('chunks', [])
            else:
                if not Path(args.chunks_dir).exists():
                    logger.error(f"Chunks directory not found: {args.chunks_dir}")
                    return 1
                
                logger.info(f"Loading chunks from directory: {args.chunks_dir}")
                chunks_path = Path(args.chunks_dir)
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
                logger.error("No chunks found!")
                return 1
                
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
                "dataset_context": args.dataset_context,
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
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            dataset_path = output_path / "finetune_dataset_optimized.json"
            
            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Optimized dataset saved to: {dataset_path}")
            logger.info(f"Quality stats: {output['quality_stats']}")
            
            logger.info(f"✅ Generated {len(training_data)} optimized training examples")
            logger.info(f"📊 Average quality score: {output['quality_stats']['avg_quality_score']:.2f}")
            logger.info(f"🏆 High quality examples (≥3.0): {output['quality_stats']['high_quality_count']}")
            logger.info(f"💾 Saved to: {dataset_path}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Error generating optimized dataset: {e}")
            return 1
    
    def _generate_dataset(self, args: argparse.Namespace) -> int:
        """Generate comprehensive fine-tuning datasets with deduplication."""
        try:
            # Create configuration
            config = DatasetConfig(
                chunks_dir=args.chunks_dir,
                output_dir=args.output_dir,
                max_chunks=args.max_chunks,
                enable_deduplication=not args.no_deduplication,
                dedupe_strategy=args.dedupe_strategy,
                qa_pairs_per_chunk=args.qa_pairs_per_chunk,
                dataset_context=args.dataset_context
            )
            
            # Generate datasets
            generator = FineTuningDatasetGenerator(config)
            
            datasets = generator.generate_complete_dataset()
            generator.save_datasets(datasets)
            
            logger.info(f"✅ Fine-tuning dataset generation completed!")
            logger.info(f"📁 Output directory: {config.output_dir}")
            logger.info(f"📊 Total Q&A pairs: {len(datasets['instruction_dataset'])}")
            
            if config.enable_deduplication:
                dedup_stats = datasets['deduplication_report']
                if dedup_stats.get('status') != 'skipped':
                    logger.info(f"🔍 Deduplication applied with {args.dedupe_strategy} strategy")
            
            return 0
            
        except Exception as e:
            logger.error(f"Error generating dataset: {e}")
            return 1
    
    def _compare_datasets(self, args: argparse.Namespace) -> int:
        """Compare dataset quality across multiple datasets."""
        dataset_paths = args.dataset_paths
        
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
                logger.error("No datasets found to compare.")
                logger.error("Specify dataset paths or ensure default datasets exist.")
                return 1
        else:
            dataset_paths = [Path(p) for p in dataset_paths]
        
        try:
            comparator = DatasetComparator()
            comparator.compare_datasets(dataset_paths)
            
            logger.info("\n" + "=" * 60)
            logger.info("OPTIMIZATION IMPROVEMENTS:")
            logger.info("=" * 60)
            
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
                logger.info(improvement)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error comparing datasets: {e}")
            return 1