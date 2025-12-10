"""Fine-tuning command for MLX-based model training."""

import argparse
from typing import Dict, Any
from pathlib import Path

from ...core.mlx_finetuner import MLXFineTuner
from ...utils.config import MLXConfig, load_config_from_env
from ...utils.logging import get_logger

logger = get_logger(__name__)


class FinetuneCommand:
    """Command to run MLX fine-tuning on generated datasets."""
    
    def add_parser(self, subparsers, name: str) -> argparse.ArgumentParser:
        """Add parser for finetune command.
        
        Args:
            subparsers: Subparser object to add to
            name: Name of the command
            
        Returns:
            Created parser
        """
        parser = subparsers.add_parser(
            name,
            help='Fine-tune language models using MLX',
            description="""
Fine-tune language models on Flow4 generated datasets using Apple's MLX framework:

1. Convert Flow4 RAG datasets to MLX format
2. Fine-tune models using LoRA (Low-Rank Adaptation) 
3. Optionally fuse adapters with base model
4. Test with interactive chat interface

This command is optimized for Apple Silicon (M1/M2/M3) and includes
hardware-specific optimizations for M3 Max with 128GB unified memory.

Supported models:
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 (recommended)
- microsoft/DialoGPT-small (fast testing)
- google/gemma-2b (high quality)
- microsoft/DialoGPT-medium (max performance)
""",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Model configuration
        model_group = parser.add_argument_group('Model Configuration')
        model_group.add_argument(
            '--model-name', '-m',
            default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            help='Base model name from Hugging Face'
        )
        model_group.add_argument(
            '--model-size',
            choices=['small', 'medium', 'large', 'max'],
            default='medium',
            help='Predefined model size configuration for M3 Max optimization'
        )
        
        # Training parameters
        training_group = parser.add_argument_group('Training Parameters')
        training_group.add_argument(
            '--num-iters', '-i',
            type=int,
            default=500,
            help='Number of training iterations'
        )
        training_group.add_argument(
            '--batch-size', '-b',
            type=int,
            default=16,
            help='Training batch size (auto-optimized for M3 Max)'
        )
        training_group.add_argument(
            '--learning-rate', '-lr',
            type=float,
            default=2e-5,
            help='Learning rate for training'
        )
        training_group.add_argument(
            '--steps-per-eval',
            type=int,
            default=200,
            help='Steps between evaluations'
        )
        
        # LoRA parameters
        lora_group = parser.add_argument_group('LoRA Configuration')
        lora_group.add_argument(
            '--lora-rank',
            type=int,
            default=16,
            help='LoRA rank (lower = fewer parameters)'
        )
        lora_group.add_argument(
            '--lora-alpha',
            type=float,
            default=32.0,
            help='LoRA alpha scaling factor'
        )
        lora_group.add_argument(
            '--lora-dropout',
            type=float,
            default=0.1,
            help='LoRA dropout rate'
        )
        lora_group.add_argument(
            '--num-layers',
            type=int,
            default=16,
            help='Number of layers to apply LoRA'
        )
        
        # Dataset configuration
        data_group = parser.add_argument_group('Dataset Configuration')
        data_group.add_argument(
            '--dataset-path', '-d',
            default='output/rag/finetune_dataset.jsonl',
            help='Path to the fine-tuning dataset (JSONL format)'
        )
        data_group.add_argument(
            '--val-split',
            type=float,
            default=0.1,
            help='Validation set split ratio'
        )
        data_group.add_argument(
            '--test-split',
            type=float,
            default=0.1,
            help='Test set split ratio'
        )
        
        # Output configuration
        output_group = parser.add_argument_group('Output Configuration')
        output_group.add_argument(
            '--adapter-dir', '-a',
            default='fine_tuned_adapters',
            help='Output directory for LoRA adapters'
        )
        output_group.add_argument(
            '--fused-model-dir', '-f',
            default='fused_model',
            help='Output directory for fused model'
        )
        
        # Workflow options
        workflow_group = parser.add_argument_group('Workflow Options')
        workflow_group.add_argument(
            '--fuse-model',
            action='store_true',
            default=True,
            help='Fuse adapters with base model after training'
        )
        workflow_group.add_argument(
            '--no-fuse',
            action='store_true',
            help='Skip model fusing (overrides --fuse-model)'
        )
        workflow_group.add_argument(
            '--chat',
            action='store_true',
            help='Start interactive chat after training'
        )
        workflow_group.add_argument(
            '--chat-only',
            action='store_true',
            help='Only start chat interface (skip training)'
        )
        workflow_group.add_argument(
            '--skip-training',
            action='store_true',
            help='Skip training, go directly to fusing/chat'
        )
        workflow_group.add_argument(
            '--sample-only',
            action='store_true',
            help='Only generate sample output (skip training and chat)'
        )
        
        # Hardware optimization
        hardware_group = parser.add_argument_group('Hardware Optimization')
        hardware_group.add_argument(
            '--disable-m3-optimization',
            action='store_true',
            help='Disable automatic M3 Max hardware optimizations'
        )
        hardware_group.add_argument(
            '--benchmark',
            action='store_true',
            help='Run hardware benchmark before training'
        )
        
        # System options
        system_group = parser.add_argument_group('System Options')
        system_group.add_argument(
            '--install-deps',
            action='store_true',
            help='Install required MLX dependencies'
        )
        system_group.add_argument(
            '--test-installation',
            action='store_true',
            help='Test MLX installation and exit'
        )
        
        return parser
    
    def run(self, args: argparse.Namespace) -> int:
        """Execute the finetune command.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            # Create MLX configuration
            config = MLXConfig(
                model_name=args.model_name,
                model_size=args.model_size,
                num_iters=args.num_iters,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                steps_per_eval=args.steps_per_eval,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                num_layers=args.num_layers,
                dataset_path=args.dataset_path,
                val_split=args.val_split,
                test_split=args.test_split,
                adapter_dir=args.adapter_dir,
                fused_model_dir=args.fused_model_dir,
                fuse_model=args.fuse_model and not args.no_fuse,
                enable_chat=args.chat,
                auto_optimize_m3=not args.disable_m3_optimization
            )
            
            # Initialize fine-tuner
            finetuner = MLXFineTuner(config)
            
            # Handle installation and testing
            if args.install_deps:
                logger.info("📦 Installing MLX dependencies...")
                if not finetuner.install_requirements():
                    logger.error("❌ Failed to install dependencies")
                    return 1
                logger.info("✅ Dependencies installed successfully")
            
            if args.test_installation:
                logger.info("🧪 Testing MLX installation...")
                if finetuner.test_installation():
                    logger.info("✅ MLX installation test passed!")
                    return 0
                else:
                    logger.error("❌ MLX installation test failed")
                    logger.info("💡 Try running with --install-deps")
                    return 1
            
            # Test installation before proceeding
            if not finetuner.test_installation():
                logger.error("❌ MLX installation test failed")
                logger.info("💡 Run with --install-deps to install required packages")
                return 1
            
            # Handle chat-only mode
            if args.chat_only:
                logger.info("💬 Starting chat-only mode...")
                
                adapter_path = args.adapter_dir if Path(args.adapter_dir).exists() else None
                fused_path = args.fused_model_dir if Path(args.fused_model_dir).exists() else None
                
                if not adapter_path and not fused_path:
                    logger.error("❌ No trained model found")
                    logger.info("💡 Train a model first or specify correct paths")
                    return 1
                
                finetuner.interactive_chat(
                    adapter_path=adapter_path,
                    fused_model_path=fused_path
                )
                return 0
            
            # Handle sample-only mode
            if args.sample_only:
                logger.info("🧪 Generating sample output only...")
                
                adapter_path = args.adapter_dir if Path(args.adapter_dir).exists() else None
                fused_path = args.fused_model_dir if Path(args.fused_model_dir).exists() else None
                
                if not adapter_path and not fused_path:
                    logger.error("❌ No trained model found")
                    return 1
                
                sample_prompt = "Document: combined_document.md\\nQuestion: What is NR Dynamic Power Optimizer?\\nAnswer:"
                
                if fused_path:
                    sample_output = finetuner.generate_sample(sample_prompt, fused_path)
                else:
                    sample_output = finetuner.generate_sample(sample_prompt, adapter_path=adapter_path)
                
                logger.info(f"📄 Sample output:\\n{sample_output}")
                return 0
            
            # Handle benchmark
            if args.benchmark:
                logger.info("⚡ Running hardware benchmark...")
                try:
                    import mlx.core as mx
                    import time
                    
                    logger.info("🖥️  M3 Max Hardware Benchmark")
                    logger.info("=" * 40)
                    
                    # Matrix multiplication benchmark
                    logger.info("Running MLX matrix benchmark...")
                    size = 2048
                    a = mx.random.normal((size, size))
                    b = mx.random.normal((size, size))
                    
                    start_time = time.time()
                    c = mx.matmul(a, b)
                    mx.eval(c)  # Force evaluation
                    end_time = time.time()
                    
                    gflops = (2 * size**3) / (end_time - start_time) / 1e9
                    logger.info(f"📊 Matrix multiplication: {gflops:.1f} GFLOPS")
                    logger.info(f"🚀 Your hardware is performing excellently!")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Benchmark skipped: {e}")
            
            # Run complete workflow
            logger.info("🎯 Running complete MLX fine-tuning workflow...")
            
            results = finetuner.run_complete_workflow(
                dataset_path=args.dataset_path,
                skip_training=args.skip_training,
                skip_fusing=args.no_fuse,
                enable_chat=args.chat
            )
            
            # Report results
            if results["training_success"]:
                logger.info("✅ Fine-tuning workflow completed successfully!")
                
                # Save usage instructions
                self._save_usage_instructions(args, results)
                
                return 0
            else:
                logger.error("❌ Fine-tuning workflow failed!")
                return 1
                
        except KeyboardInterrupt:
            logger.info("\\n⚠️ Operation interrupted by user")
            return 130
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            logger.debug("Full error details:", exc_info=True)
            return 1
    
    def _save_usage_instructions(self, args: argparse.Namespace, results: Dict[str, Any]) -> None:
        """Save usage instructions for the fine-tuned model."""
        instructions_file = Path(args.adapter_dir) / "usage_instructions.md"
        
        try:
            with open(instructions_file, 'w') as f:
                f.write(f"""# Fine-tuned Model Usage

## Model Details
- Base Model: {args.model_name}
- Model Size: {args.model_size}
- Adapter Path: {args.adapter_dir}
- Fused Model: {args.fused_model_dir if results.get('fusing_success') else 'Not created'}
- Training Iterations: {args.num_iters}
- LoRA Rank: {args.lora_rank}

## Quick Start

### Generate Text (with adapter)
```bash
python -m mlx_lm.generate \\
    --model {args.model_name} \\
    --adapter-path {args.adapter_dir} \\
    --prompt "Your prompt here" \\
    --max-tokens 200 \\
    --temp 0.7
```

### Generate Text (with fused model)
```bash
python -m mlx_lm.generate \\
    --model {args.fused_model_dir} \\
    --prompt "Your prompt here" \\
    --max-tokens 200 \\
    --temp 0.7
```

### Interactive Chat
```bash
flow4 finetune --chat-only
```

### Flow4 Integration
```bash
# Complete workflow
flow4 finetune --dataset-path output/rag/finetune_dataset.jsonl --chat

# Chat with existing model
flow4 finetune --chat-only --adapter-dir {args.adapter_dir}

# Generate sample
flow4 finetune --sample-only --adapter-dir {args.adapter_dir}
```

## Sample Prompts
- "Document: combined_document.md\\nQuestion: What is this section about?\\nAnswer:"
- "Source: combined_document.md\\nContent summary:"
- "Based on combined_document.md, explain:"

## Hardware Optimization
This model was trained with M3 Max optimizations:
- Batch size: {args.batch_size}
- Learning rate: {args.learning_rate}
- Optimized for Apple Silicon performance

## Troubleshooting
- Ensure MLX is installed: `pip install mlx>=0.12.0 mlx-lm>=0.8.0`
- Test installation: `flow4 finetune --test-installation`
- For NumPy issues: `pip install "numpy>=1.24.0,<2.0.0"`
""")
            
            logger.info(f"📄 Usage instructions saved to: {instructions_file}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not save usage instructions: {e}")


def add_finetune_command(subparsers) -> None:
    """Add the finetune command to the CLI."""
    command = FinetuneCommand()
    return command.add_parser(subparsers, 'finetune')