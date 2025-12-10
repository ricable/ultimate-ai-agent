#!/usr/bin/env python3
"""
Enhanced MLX Fine-tuning Script combining high-level convenience with low-level control
Based on Apple's reference implementation with improvements for usability and monitoring
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import psutil
from datasets import load_dataset
from mlx.utils import tree_flatten, tree_unflatten

# Try to import MLX-LM utilities
try:
    from mlx_lm import load, generate
    from mlx_lm.tuner.lora import LoRALinear
    from mlx_lm.tuner import utils as lora_utils
    MLX_LM_AVAILABLE = True
except ImportError:
    print("⚠️  MLX-LM not fully available, using direct MLX implementation")
    MLX_LM_AVAILABLE = False

# Flash Attention Integration
try:
    from flash_attention_mlx import OptimizedMLXMultiHeadAttention, FlashAttentionBenchmark
    FLASH_ATTENTION_AVAILABLE = True
    print("✅ Flash Attention optimizations available")
except ImportError:
    print("⚠️  Flash Attention not available, using standard MLX attention")
    FLASH_ATTENTION_AVAILABLE = False

# Disable output buffering for real-time output
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

def build_parser():
    """Build comprehensive argument parser combining both approaches"""
    parser = argparse.ArgumentParser(description="Enhanced MLX LoRA Fine-tuning")
    
    # Model and data arguments
    parser.add_argument("--model", default="./models/mlx/tinyllama-1.1b-chat", 
                       help="Path to the local model directory or Hugging Face repo")
    parser.add_argument("--data", default="./data/datasets", 
                       help="Directory with {train, valid, test}.jsonl files")
    parser.add_argument("--output-dir", default="./data/outputs", 
                       help="Output directory for fine-tuned adapters")
    
    # Dataset preparation arguments
    parser.add_argument("--hf-dataset", default="Abirate/english_quotes", 
                       help="Hugging Face dataset to download")
    parser.add_argument("--dataset-size", type=int, default=100, 
                       help="Number of examples to use from dataset")
    parser.add_argument("--prepare-data", action="store_true", 
                       help="Download and prepare dataset from Hugging Face")
    
    # Training arguments
    parser.add_argument("--train", action="store_true", 
                       help="Enable training mode")
    parser.add_argument("--test", action="store_true", 
                       help="Evaluate on test set after training")
    parser.add_argument("--generate", action="store_true", 
                       help="Run generation tests after training")
    
    # LoRA configuration
    parser.add_argument("--lora-layers", type=int, default=16, 
                       help="Number of layers to fine-tune")
    parser.add_argument("--lora-rank", type=int, default=8, 
                       help="LoRA rank (r)")
    parser.add_argument("--lora-alpha", type=float, default=16.0, 
                       help="LoRA alpha scaling factor")
    parser.add_argument("--lora-dropout", type=float, default=0.0, 
                       help="LoRA dropout rate")
    
    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=1, 
                       help="Training batch size")
    parser.add_argument("--iters", type=int, default=100, 
                       help="Number of training iterations")
    parser.add_argument("--learning-rate", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, 
                       help="Weight decay")
    parser.add_argument("--max-seq-length", type=int, default=512, 
                       help="Maximum sequence length")
    
    # Evaluation and reporting
    parser.add_argument("--val-batches", type=int, default=25, 
                       help="Number of validation batches (-1 for all)")
    parser.add_argument("--steps-per-report", type=int, default=5, 
                       help="Steps between loss reporting")
    parser.add_argument("--steps-per-eval", type=int, default=20, 
                       help="Steps between validation")
    parser.add_argument("--save-every", type=int, default=50, 
                       help="Save adapter every N iterations")
    
    # Flash Attention arguments
    parser.add_argument("--use-flash-attention", action="store_true", default=True,
                       help="Enable Flash Attention optimization (default: True)")
    parser.add_argument("--disable-flash-attention", action="store_true",
                       help="Disable Flash Attention optimization")
    parser.add_argument("--flash-block-size", type=int, default=None,
                       help="Flash Attention block size (auto if None)")
    parser.add_argument("--benchmark-attention", action="store_true",
                       help="Run attention benchmark before training")
    
    # Generation arguments
    parser.add_argument("--max-tokens", type=int, default=100, 
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, 
                       help="Generation temperature")
    parser.add_argument("--prompt", type=str, 
                       help="Custom prompt for generation testing")
    
    # System and debugging
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--method", choices=["mlx-lm", "direct"], default="mlx-lm", 
                       help="Training method: mlx-lm (high-level) or direct (low-level)")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose output")
    parser.add_argument("--profile", action="store_true", 
                       help="Enable performance profiling")
    
    return parser

class SystemMonitor:
    """Monitor system resources during training"""
    
    def __init__(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / (1024**2)
        self.peak_memory = self.start_memory
        
    def update(self):
        current_memory = psutil.Process().memory_info().rss / (1024**2)
        self.peak_memory = max(self.peak_memory, current_memory)
        
    def get_stats(self) -> Dict[str, float]:
        current_time = time.time()
        current_memory = psutil.Process().memory_info().rss / (1024**2)
        
        return {
            "elapsed_time": current_time - self.start_time,
            "current_memory_mb": current_memory,
            "peak_memory_mb": self.peak_memory,
            "memory_delta_mb": current_memory - self.start_memory
        }

class EnhancedDataset:
    """Enhanced dataset class with better functionality"""
    
    def __init__(self, path: Path, key: str = "text"):
        self.path = path
        self.key = key
        self._data = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        if not self.path.exists():
            return []
        
        data = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"⚠️  Warning: Skipping invalid JSON on line {line_num}: {e}")
        return data
    
    def __getitem__(self, idx: int) -> str:
        return self._data[idx][self.key]
    
    def __len__(self) -> int:
        return len(self._data)
    
    def get_stats(self) -> Dict[str, Any]:
        if not self._data:
            return {"size": 0, "avg_length": 0, "max_length": 0}
            
        lengths = [len(item[self.key]) for item in self._data]
        return {
            "size": len(self._data),
            "avg_length": sum(lengths) / len(lengths),
            "max_length": max(lengths),
            "min_length": min(lengths)
        }

def prepare_hf_dataset(args) -> Tuple[int, int, int]:
    """Download and prepare dataset from Hugging Face"""
    print("🔄 Preparing dataset from Hugging Face...")
    
    # Create data directory
    data_dir = Path(args.data)
    data_dir.mkdir(exist_ok=True)
    
    # Download dataset
    print(f"📦 Downloading {args.hf_dataset} (first {args.dataset_size} examples)...")
    try:
        dataset = load_dataset(args.hf_dataset, split=f"train[:{args.dataset_size}]")
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return 0, 0, 0
    
    # Prepare splits
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_data, val_data, test_data = [], [], []
    
    for i, item in enumerate(dataset):
        # Create instruction-following format for quotes
        if 'quote' in item and 'author' in item:
            tag = item.get('tags', ['life'])[0] if item.get('tags') else 'life'
            text = f"<|im_start|>user\nWrite an inspirational quote about {tag}<|im_end|>\n<|im_start|>assistant\n\"{item['quote']}\" - {item['author']}<|im_end|>"
        else:
            # Generic text format
            text = item.get('text', str(item))
        
        entry = {"text": text}
        
        # Split data
        if i < train_size:
            train_data.append(entry)
        elif i < train_size + val_size:
            val_data.append(entry)
        else:
            test_data.append(entry)
    
    # Save datasets
    for split, data in [("train", train_data), ("valid", val_data), ("test", test_data)]:
        if data:  # Only save non-empty splits
            with open(data_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
                for entry in data:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"✅ Dataset prepared: {len(train_data)} train, {len(val_data)} valid, {len(test_data)} test")
    
    # Show sample
    if train_data:
        print("\n📝 Sample training entry:")
        print(train_data[0]['text'][:200] + "...")
    
    return len(train_data), len(val_data), len(test_data)

def load_datasets(args) -> Tuple[EnhancedDataset, EnhancedDataset, EnhancedDataset]:
    """Load train, validation, and test datasets"""
    
    def load_split(name: str) -> EnhancedDataset:
        path = Path(args.data) / f"{name}.jsonl"
        dataset = EnhancedDataset(path)
        if len(dataset) == 0 and name != "test":  # Test set is optional
            raise ValueError(f"{name.title()} set not found or empty at {path}")
        return dataset
    
    print("📚 Loading datasets...")
    train_set = load_split("train")
    valid_set = load_split("valid") 
    test_set = load_split("test")
    
    # Print dataset statistics
    for name, dataset in [("Train", train_set), ("Valid", valid_set), ("Test", test_set)]:
        if len(dataset) > 0:
            stats = dataset.get_stats()
            print(f"  {name}: {stats['size']} examples, avg length: {stats['avg_length']:.0f} chars")
    
    return train_set, valid_set, test_set

def create_lora_model(model, args):
    """Apply LoRA to model layers"""
    print(f"🔧 Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha}) to {args.lora_layers} layers...")
    
    # Freeze all parameters
    model.freeze()
    
    # Apply LoRA to the last N layers
    for layer in model.model.layers[-args.lora_layers:]:
        if hasattr(layer, 'self_attn'):
            layer.self_attn.q_proj = LoRALinear.from_base(
                layer.self_attn.q_proj, 
                r=args.lora_rank,
                dropout=args.lora_dropout,
                scale=args.lora_alpha
            )
            layer.self_attn.v_proj = LoRALinear.from_base(
                layer.self_attn.v_proj,
                r=args.lora_rank, 
                dropout=args.lora_dropout,
                scale=args.lora_alpha
            )
            
        # Handle MoE models if present
        if hasattr(layer, "block_sparse_moe"):
            layer.block_sparse_moe.gate = LoRALinear.from_base(
                layer.block_sparse_moe.gate,
                r=args.lora_rank,
                dropout=args.lora_dropout,
                scale=args.lora_alpha
            )
    
    # Calculate parameter counts
    total_params = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
    trainable_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    trainable_pct = (trainable_params / total_params) * 100
    
    print(f"📊 Model parameters: {total_params:.3f}M total, {trainable_params:.3f}M trainable ({trainable_pct:.3f}%)")
    
    return model

def apply_flash_attention_to_model(model, use_flash_attention=True, block_size=None):
    """Apply Flash Attention optimizations to model attention layers"""
    if not use_flash_attention or not FLASH_ATTENTION_AVAILABLE:
        print("ℹ️ Using standard MLX attention")
        return model, 0
    
    print("🚀 Applying Flash Attention optimizations...")
    attention_replacements = 0
    
    def replace_attention_recursive(module, name_prefix=""):
        nonlocal attention_replacements
        
        # Handle MLX models which may have different attribute access patterns
        try:
            for name in dir(module):
                if name.startswith('_') or name in ['training', 'parameters', 'modules']:
                    continue
                    
                try:
                    child = getattr(module, name)
                    if not isinstance(child, (nn.Module, type(None))):
                        continue
                        
                    full_name = f"{name_prefix}.{name}" if name_prefix else name
                    
                    # Check if this is an attention layer we should replace
                    if isinstance(child, nn.MultiHeadAttention):
                        print(f"🔄 Replacing {full_name} with Flash Attention")
                        
                        # Create optimized replacement
                        flash_attention = OptimizedMLXMultiHeadAttention(
                            child.dims,
                            child.num_heads,
                            bias=hasattr(child, 'bias'),
                            use_flash_attention=True,
                            block_size=block_size
                        )
                        
                        # Copy weights from original layer
                        if hasattr(child, 'q_proj') and hasattr(child.q_proj, 'weight'):
                            flash_attention.q_proj.weight = child.q_proj.weight
                            flash_attention.k_proj.weight = child.k_proj.weight  
                            flash_attention.v_proj.weight = child.v_proj.weight
                            flash_attention.out_proj.weight = child.out_proj.weight
                            
                            if hasattr(child.q_proj, 'bias') and child.q_proj.bias is not None:
                                flash_attention.q_proj.bias = child.q_proj.bias
                                flash_attention.k_proj.bias = child.k_proj.bias
                                flash_attention.v_proj.bias = child.v_proj.bias
                                flash_attention.out_proj.bias = child.out_proj.bias
                        
                        # Replace the layer
                        setattr(module, name, flash_attention)
                        attention_replacements += 1
                    else:
                        # Recursively process child modules
                        replace_attention_recursive(child, full_name)
                        
                except (AttributeError, TypeError):
                    continue
                    
        except (AttributeError, TypeError):
            pass
    
    try:
        replace_attention_recursive(model)
        
        if attention_replacements > 0:
            print(f"✅ Replaced {attention_replacements} attention layers with Flash Attention")
        else:
            print("ℹ️ No compatible attention layers found for replacement")
            
    except Exception as e:
        print(f"⚠️ Flash Attention integration failed: {e}")
        print("ℹ️ Continuing with standard MLX attention")
    
    return model, attention_replacements

def run_attention_benchmark(model, tokenizer, args):
    """Run Flash Attention benchmark before training"""
    if not FLASH_ATTENTION_AVAILABLE:
        print("⚠️ Flash Attention not available for benchmarking")
        return
    
    print("\n🔬 Running Flash Attention benchmark...")
    
    try:
        # Create benchmark instance
        benchmark = FlashAttentionBenchmark()
        
        # Run focused benchmark for training parameters
        results = benchmark.benchmark_attention_performance(
            batch_sizes=[args.batch_size],
            seq_lengths=[min(args.max_seq_length, 256)],  # Cap at 256 for benchmark
            head_dims=[64, 128],  # Common head dimensions
            num_heads=8,
            num_runs=3
        )
        
        # Print summary
        benchmark.print_summary()
        
        # Save results
        benchmark_file = Path(args.output_dir) / "flash_attention_benchmark.json"
        benchmark_file.parent.mkdir(parents=True, exist_ok=True)
        benchmark.save_results(str(benchmark_file))
        
    except Exception as e:
        print(f"⚠️ Benchmark failed: {e}")

def iterate_batches(dataset, tokenizer, batch_size, max_seq_length, train=False):
    """Generate batches from dataset"""
    while True:
        indices = np.arange(len(dataset))
        if train:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            # Encode batch
            batch = []
            for j in range(batch_size):
                try:
                    text = dataset[indices[i + j]]
                    tokens = tokenizer.encode(text)
                    if len(tokens) > max_seq_length:
                        tokens = tokens[:max_seq_length]
                    batch.append(tokens)
                except Exception as e:
                    print(f"⚠️  Warning: Skipping problematic example: {e}")
                    continue
            
            if not batch:
                continue
                
            lengths = [len(x) for x in batch]
            max_len = min(max(lengths), max_seq_length)
            
            # Pad batch
            batch_arr = np.zeros((len(batch), max_len), dtype=np.int32)
            for j, tokens in enumerate(batch):
                batch_arr[j, :len(tokens)] = tokens[:max_len]
            
            batch_tensor = mx.array(batch_arr)
            yield batch_tensor[:, :-1], batch_tensor[:, 1:], mx.array(lengths)
        
        if not train:
            break

def compute_loss(model, inputs, targets, lengths):
    """Compute cross-entropy loss with padding mask"""
    output = model(inputs)
    if isinstance(output, tuple):
        logits, _ = output
    else:
        logits = output
    logits = logits.astype(mx.float32)
    
    # Create padding mask
    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]
    
    # Compute cross-entropy loss
    ce_loss = nn.losses.cross_entropy(logits, targets) * length_mask
    total_tokens = length_mask.sum()
    
    return ce_loss.sum() / total_tokens, total_tokens

def evaluate_model(model, dataset, tokenizer, args, num_batches=-1):
    """Evaluate model on dataset"""
    model.eval()
    
    all_losses = []
    total_tokens = 0
    
    batch_iter = iterate_batches(dataset, tokenizer, args.batch_size, args.max_seq_length, train=False)
    
    if num_batches == -1:
        num_batches = len(dataset) // args.batch_size
    
    for batch_idx, batch in enumerate(batch_iter):
        if batch_idx >= num_batches:
            break
            
        loss_val, token_count = compute_loss(model, *batch)
        all_losses.append(loss_val.item() * token_count.item())
        total_tokens += token_count.item()
    
    model.train()
    return sum(all_losses) / total_tokens if total_tokens > 0 else float('inf')

def train_model(model, train_set, valid_set, tokenizer, args, monitor):
    """Enhanced training loop with detailed monitoring"""
    print("🚀 Starting training...")
    
    # Setup optimizer
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    
    # Create loss function with gradients
    loss_and_grad_fn = nn.value_and_grad(model, compute_loss)
    
    # Training state
    losses = []
    tokens_processed = 0
    best_val_loss = float('inf')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    report_start = start_time
    
    # Training loop
    batch_iter = iterate_batches(train_set, tokenizer, args.batch_size, args.max_seq_length, train=True)
    
    for iteration, batch in enumerate(batch_iter):
        if iteration >= args.iters:
            break
        
        # Forward and backward pass
        (loss_val, token_count), grads = loss_and_grad_fn(model, *batch)
        
        # Update parameters
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss_val)
        
        # Track metrics
        losses.append(loss_val.item())
        tokens_processed += token_count.item()
        monitor.update()
        
        # Progress reporting
        if (iteration + 1) % args.steps_per_report == 0:
            avg_loss = np.mean(losses)
            current_time = time.time()
            elapsed = current_time - report_start
            
            iters_per_sec = args.steps_per_report / elapsed
            tokens_per_sec = tokens_processed / elapsed
            
            sys_stats = monitor.get_stats()
            
            print(f"Iter {iteration + 1:4d}: "
                  f"Train loss {avg_loss:.3f}, "
                  f"LR {args.learning_rate:.2e}, "
                  f"It/sec {iters_per_sec:.2f}, "
                  f"Tok/sec {tokens_per_sec:.1f}, "
                  f"Mem {sys_stats['current_memory_mb']:.0f}MB")
            
            # Reset counters
            losses = []
            tokens_processed = 0
            report_start = current_time
        
        # Validation
        if iteration == 0 or (iteration + 1) % args.steps_per_eval == 0:
            val_start = time.time()
            val_loss = evaluate_model(model, valid_set, tokenizer, args, args.val_batches)
            val_time = time.time() - val_start
            
            print(f"Iter {iteration + 1:4d}: Val loss {val_loss:.3f}, Val took {val_time:.2f}s")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                adapter_path = output_dir / "best_adapters.safetensors"
                mx.save_safetensors(str(adapter_path), dict(tree_flatten(model.trainable_parameters())))
                print(f"💾 Saved best model (val_loss={val_loss:.3f}) to {adapter_path}")
        
        # Periodic saves
        if (iteration + 1) % args.save_every == 0:
            adapter_path = output_dir / f"adapters_iter_{iteration + 1}.safetensors"
            mx.save_safetensors(str(adapter_path), dict(tree_flatten(model.trainable_parameters())))
            print(f"💾 Saved checkpoint to {adapter_path}")
    
    # Final save
    final_path = output_dir / "adapters.safetensors"
    mx.save_safetensors(str(final_path), dict(tree_flatten(model.trainable_parameters())))
    print(f"💾 Saved final adapters to {final_path}")
    
    # Training summary
    total_time = time.time() - start_time
    final_stats = monitor.get_stats()
    
    print(f"\n📊 Training completed in {total_time:.1f}s")
    print(f"📊 Peak memory usage: {final_stats['peak_memory_mb']:.0f}MB")
    print(f"📊 Best validation loss: {best_val_loss:.3f}")

def generate_text(model, tokenizer, prompt, args):
    """Generate text with the fine-tuned model"""
    print(f"\n🎯 Generating text for prompt: {prompt}")
    print("-" * 50)
    
    # Format prompt
    if not prompt.startswith("<|im_start|>"):
        prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Encode prompt
    tokens = mx.array(tokenizer.encode(prompt))
    
    print(prompt, end="", flush=True)
    
    # Generate tokens
    generated_tokens = []
    for _ in range(args.max_tokens):
        # Get next token probabilities
        output = model(tokens[None])
        if isinstance(output, tuple):
            logits, _ = output
        else:
            logits = output
        logits = logits[0, -1, :] / args.temperature
        
        # Sample next token
        probs = mx.softmax(logits, axis=-1)
        next_token = mx.random.categorical(logits[None], axis=-1)[0].item()
        
        if next_token == tokenizer.eos_token_id:
            break
            
        generated_tokens.append(next_token)
        tokens = mx.concatenate([tokens, mx.array([next_token])])
        
        # Decode and print
        if len(generated_tokens) % 1 == 0:  # Print every token
            new_text = tokenizer.decode(generated_tokens[-1:])
            print(new_text, end="", flush=True)
    
    print("\n" + "-" * 50)
    
    if not generated_tokens:
        print("⚠️  No tokens generated")
    else:
        full_response = tokenizer.decode(generated_tokens)
        print(f"📝 Generated {len(generated_tokens)} tokens")

def run_generation_tests(model, tokenizer, args):
    """Run a series of generation tests"""
    print("\n🧪 Running generation tests...")
    
    test_prompts = [
        "Write an inspirational quote about success",
        "Write an inspirational quote about perseverance", 
        "Write an inspirational quote about dreams",
        "What is the meaning of life?",
        "How can I be more productive?"
    ]
    
    # Add custom prompt if provided
    if args.prompt:
        test_prompts.insert(0, args.prompt)
    
    for prompt in test_prompts:
        try:
            generate_text(model, tokenizer, prompt, args)
        except Exception as e:
            print(f"❌ Error generating for '{prompt}': {e}")
            continue

def main():
    """Enhanced main function with Flash Attention integration"""
    parser = build_parser()
    args = parser.parse_args()
    
    # Handle Flash Attention settings
    if args.disable_flash_attention:
        args.use_flash_attention = False
    
    # Set random seed
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    
    # Initialize system monitor
    monitor = SystemMonitor()
    
    # Print configuration
    print("🎉 Enhanced MLX Fine-tuning with Flash Attention")
    print("=" * 70)
    print(f"🔧 Method: {args.method}")
    print(f"📱 Model: {args.model}")
    print(f"📊 System: {psutil.cpu_count()} cores, {psutil.virtual_memory().total // (1024**3)}GB RAM")
    print(f"🎯 LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, layers={args.lora_layers}")
    print(f"📚 Training: {args.iters} iters, batch_size={args.batch_size}, lr={args.learning_rate}")
    print(f"⚡ Flash Attention: {'✅ Enabled' if args.use_flash_attention else '❌ Disabled'}")
    if args.flash_block_size:
        print(f"🔧 Flash Block Size: {args.flash_block_size}")
    
    # Data preparation
    if args.prepare_data:
        train_count, val_count, test_count = prepare_hf_dataset(args)
        if train_count == 0:
            print("❌ No training data prepared")
            return 1
    
    # Load model and tokenizer
    print(f"\n📦 Loading model from {args.model}...")
    try:
        if MLX_LM_AVAILABLE:
            model, tokenizer = load(args.model)
        else:
            raise ImportError("MLX-LM not available")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1
    
    # Apply Flash Attention optimizations
    if args.use_flash_attention and FLASH_ATTENTION_AVAILABLE:
        model, flash_replacements = apply_flash_attention_to_model(
            model, 
            use_flash_attention=args.use_flash_attention,
            block_size=args.flash_block_size
        )
        print(f"📊 Flash Attention: {flash_replacements} layers optimized")
    
    # Run attention benchmark if requested
    if args.benchmark_attention:
        run_attention_benchmark(model, tokenizer, args)
    
    # Apply LoRA
    model = create_lora_model(model, args)
    
    # Load datasets
    try:
        train_set, valid_set, test_set = load_datasets(args)
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return 1
    
    # Training
    if args.train:
        train_model(model, train_set, valid_set, tokenizer, args, monitor)
    
    # Load adapters for testing
    adapter_path = Path(args.output_dir) / "adapters.safetensors"
    if adapter_path.exists():
        print(f"\n📦 Loading adapters from {adapter_path}...")
        adapters = mx.load(str(adapter_path))
        model.update(tree_unflatten(list(adapters.items())))
    
    # Testing
    if args.test and len(test_set) > 0:
        print("\n🧪 Testing on test set...")
        test_loss = evaluate_model(model, test_set, tokenizer, args)
        test_ppl = math.exp(test_loss)
        print(f"📊 Test loss: {test_loss:.3f}, Test perplexity: {test_ppl:.2f}")
    
    # Generation
    if args.generate:
        run_generation_tests(model, tokenizer, args)
    
    # Final summary
    final_stats = monitor.get_stats()
    print(f"\n🎊 Process completed in {final_stats['elapsed_time']:.1f}s")
    print(f"📊 Peak memory usage: {final_stats['peak_memory_mb']:.0f}MB")
    print(f"📁 Output directory: {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())