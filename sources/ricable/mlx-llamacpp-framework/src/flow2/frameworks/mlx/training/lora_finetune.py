"""
LoRA fine-tuning implementation for MLX.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .utils import (
    set_seed, 
    load_jsonl_dataset, 
    prepare_dataset,
    create_optimizer,
    create_scheduler,
    compute_loss,
    create_data_loader
)

# Flash Attention Integration
try:
    from flash_attention_mlx import OptimizedMLXMultiHeadAttention, FlashAttentionBenchmark
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def apply_flash_attention_to_model(model, use_flash_attention=True, block_size=None):
    """
    Apply Flash Attention optimizations to model attention layers
    """
    if not use_flash_attention or not FLASH_ATTENTION_AVAILABLE:
        logger.info("Using standard MLX attention")
        return model, 0
    
    logger.info("Applying Flash Attention optimizations...")
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
                    if not hasattr(child, '__class__'):
                        continue
                        
                    full_name = f"{name_prefix}.{name}" if name_prefix else name
                    
                    # Check if this is an attention layer we should replace
                    if hasattr(child, '__class__') and 'MultiHeadAttention' in str(child.__class__):
                        logger.info(f"Replacing {full_name} with Flash Attention")
                        
                        # Create optimized replacement
                        try:
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
                        except Exception as e:
                            logger.warning(f"Failed to replace {full_name}: {e}")
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
            logger.info(f"Replaced {attention_replacements} attention layers with Flash Attention")
        else:
            logger.info("No compatible attention layers found for replacement")
            
    except Exception as e:
        logger.warning(f"Flash Attention integration failed: {e}")
        logger.info("Continuing with standard MLX attention")
    
    return model, attention_replacements

def finetune_lora(
    model_name: str,
    train_data_path: str,
    output_dir: str,
    val_data_path: Optional[str] = None,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    max_length: int = 512,
    warmup_steps: int = 100,
    log_interval: int = 10,
    save_interval: int = 100,
    optimizer_type: str = "adamw",
    scheduler_type: str = "linear",
    seed: int = 42,
    prompt_key: str = "prompt",
    response_key: str = "response",
    quantization: Optional[str] = None,
    use_flash_attention: bool = True,
    flash_block_size: Optional[int] = None
) -> Tuple[bool, str]:
    """
    Perform LoRA fine-tuning of a model using MLX.
    
    Args:
        model_name: Name or path of the base model
        train_data_path: Path to the training data (JSONL format)
        output_dir: Directory to save the fine-tuned model
        val_data_path: Path to the validation data (JSONL format)
        lora_rank: LoRA rank (r)
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: LoRA dropout rate
        target_modules: List of module names to apply LoRA to (None for default)
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        weight_decay: Weight decay value
        max_length: Maximum sequence length
        warmup_steps: Number of warmup steps
        log_interval: Interval for logging training progress
        save_interval: Interval for saving model checkpoints
        optimizer_type: Optimizer type (adamw, sgd)
        scheduler_type: Scheduler type (linear, cosine)
        seed: Random seed for reproducibility
        prompt_key: Key for prompt in the data
        response_key: Key for response in the data
        quantization: Quantization type (None, int4, int8)
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Set random seed
        set_seed(seed)
        
        # Set default device to GPU
        mx.set_default_device(mx.gpu)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model and tokenizer
        logger.info(f"Loading model: {model_name}")
        from mlx_lm import load, generate
        
        model, tokenizer = load(model_name, quantization=quantization)
        
        # Apply Flash Attention optimizations
        if use_flash_attention and FLASH_ATTENTION_AVAILABLE:
            model, flash_replacements = apply_flash_attention_to_model(
                model, 
                use_flash_attention=use_flash_attention,
                block_size=flash_block_size
            )
            if flash_replacements > 0:
                logger.info(f"Flash Attention: {flash_replacements} layers optimized")
        
        # Apply LoRA
        logger.info(f"Applying LoRA with rank {lora_rank}, alpha {lora_alpha}")
        from mlx_lm.lora import apply_lora
        
        # Default target modules if not specified
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        model = apply_lora(
            model,
            r=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=target_modules
        )
        
        # Save original model parameters
        original_params = {k: v.copy() for k, v in model.parameters().items() if "lora" not in k}
        
        # Load training data
        logger.info(f"Loading training data from: {train_data_path}")
        train_data = load_jsonl_dataset(train_data_path)
        logger.info(f"Loaded {len(train_data)} training examples")
        
        # Load validation data if provided
        val_data = None
        if val_data_path:
            logger.info(f"Loading validation data from: {val_data_path}")
            val_data = load_jsonl_dataset(val_data_path)
            logger.info(f"Loaded {len(val_data)} validation examples")
        
        # Prepare training data
        logger.info("Preparing training data")
        train_input_ids, train_labels = prepare_dataset(
            train_data,
            tokenizer,
            max_length=max_length,
            prompt_key=prompt_key,
            response_key=response_key
        )
        
        # Prepare validation data if provided
        val_input_ids, val_labels = None, None
        if val_data:
            logger.info("Preparing validation data")
            val_input_ids, val_labels = prepare_dataset(
                val_data,
                tokenizer,
                max_length=max_length,
                prompt_key=prompt_key,
                response_key=response_key
            )
        
        # Create optimizer
        optimizer = create_optimizer(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_type=optimizer_type
        )
        
        # Calculate total training steps
        total_steps = (len(train_data) // batch_size) * epochs
        
        # Create scheduler
        scheduler = create_scheduler(
            optimizer,
            scheduler_type=scheduler_type,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )
        
        # Define loss and grad functions
        def loss_fn(model, inputs, targets):
            logits = model(inputs)
            return compute_loss(logits, targets)
        
        # Get trainable (LoRA) parameters
        def trainable_params(model):
            params_dict = {}
            for k, v in model.parameters().items():
                if "lora" in k:
                    params_dict[k] = v
            return params_dict
        
        # Define training step with LoRA parameter updates only
        def train_step(model, inputs, targets):
            loss, grads = nn.value_and_grad(model, loss_fn)(model, inputs, targets)
            
            # Filter out non-LoRA parameters
            lora_grads = {k: v for k, v in grads.items() if "lora" in k}
            
            # Update only LoRA parameters
            optimizer.update(model, lora_grads)
            
            return loss
        
        # Training loop
        logger.info(f"Starting training for {epochs} epochs")
        step = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Create data loader
            train_loader = create_data_loader(
                train_input_ids,
                train_labels,
                batch_size=batch_size,
                shuffle=True
            )
            
            # Training loop
            train_loss = 0.0
            train_steps = 0
            
            for batch_inputs, batch_labels in train_loader:
                # Perform training step
                loss = train_step(model, batch_inputs, batch_labels)
                
                # Update learning rate
                if scheduler:
                    scheduler.step()
                
                train_loss += loss
                train_steps += 1
                step += 1
                
                # Log progress
                if step % log_interval == 0:
                    avg_loss = train_loss / train_steps
                    elapsed = time.time() - start_time
                    logger.info(f"Step {step}: loss = {avg_loss:.6f}, time = {elapsed:.2f}s")
                
                # Save checkpoint
                if step % save_interval == 0:
                    # Extract and save only LoRA parameters
                    lora_params = trainable_params(model)
                    checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    mx.save(os.path.join(checkpoint_path, "lora.npz"), lora_params)
                    logger.info(f"Saved LoRA checkpoint to {checkpoint_path}")
            
            # End of epoch
            avg_train_loss = train_loss / train_steps
            logger.info(f"Epoch {epoch+1}: Average training loss = {avg_train_loss:.6f}")
            
            # Validation
            if val_data:
                val_loader = create_data_loader(
                    val_input_ids,
                    val_labels,
                    batch_size=batch_size,
                    shuffle=False
                )
                
                val_loss = 0.0
                val_steps = 0
                
                for batch_inputs, batch_labels in val_loader:
                    logits = model(batch_inputs)
                    loss = compute_loss(logits, batch_labels)
                    
                    val_loss += loss
                    val_steps += 1
                
                avg_val_loss = val_loss / val_steps
                logger.info(f"Epoch {epoch+1}: Validation loss = {avg_val_loss:.6f}")
            
            # Save LoRA parameters at the end of each epoch
            lora_params = trainable_params(model)
            epoch_path = os.path.join(output_dir, f"epoch-{epoch+1}")
            os.makedirs(epoch_path, exist_ok=True)
            mx.save(os.path.join(epoch_path, "lora.npz"), lora_params)
            logger.info(f"Saved LoRA parameters to {epoch_path}")
        
        # Save final LoRA parameters
        lora_params = trainable_params(model)
        final_path = os.path.join(output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        mx.save(os.path.join(final_path, "lora.npz"), lora_params)
        
        # Also save model config
        config = {
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "target_modules": target_modules,
            "base_model": model_name,
            "quantization": quantization
        }
        
        import json
        with open(os.path.join(final_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved final LoRA parameters to {final_path}")
        
        return True, f"LoRA fine-tuning completed successfully. Parameters saved to {final_path}"
        
    except Exception as e:
        logger.error(f"LoRA fine-tuning failed: {str(e)}", exc_info=True)
        return False, f"LoRA fine-tuning failed: {str(e)}"

def apply_lora_adapter(
    model_name: str,
    lora_path: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: int = 42,
    quantization: Optional[str] = None,
    use_flash_attention: bool = True,
    flash_block_size: Optional[int] = None
) -> Tuple[bool, str]:
    """
    Apply a trained LoRA adapter to a model and run inference.
    
    Args:
        model_name: Name or path of the base model
        lora_path: Path to the LoRA adapter weights
        prompt: Text prompt for generation
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        top_p: Top-p sampling value
        seed: Random seed for reproducibility
        quantization: Quantization type (None, int4, int8)
        
    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        # Set random seed
        set_seed(seed)
        
        # Set default device to GPU
        mx.set_default_device(mx.gpu)
        
        # Check if lora_path exists
        if not os.path.exists(lora_path):
            return False, f"LoRA path does not exist: {lora_path}"
        
        # Check if lora_path is a directory or file
        if os.path.isdir(lora_path):
            lora_path = os.path.join(lora_path, "final", "lora.npz")
            config_path = os.path.join(os.path.dirname(os.path.dirname(lora_path)), "final", "config.json")
        else:
            config_path = os.path.join(os.path.dirname(lora_path), "config.json")
        
        # Load config if it exists
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        
        # Extract LoRA parameters from config
        lora_rank = config.get("lora_rank", 8)
        lora_alpha = config.get("lora_alpha", 16)
        target_modules = config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        base_model = config.get("base_model", model_name)
        
        # Use config quantization if not provided
        if quantization is None and "quantization" in config:
            quantization = config["quantization"]
        
        # Load model and tokenizer
        logger.info(f"Loading model: {base_model}")
        from mlx_lm import load, generate
        
        model, tokenizer = load(base_model, quantization=quantization)
        
        # Apply Flash Attention optimizations
        if use_flash_attention and FLASH_ATTENTION_AVAILABLE:
            model, flash_replacements = apply_flash_attention_to_model(
                model, 
                use_flash_attention=use_flash_attention,
                block_size=flash_block_size
            )
            if flash_replacements > 0:
                logger.info(f"Flash Attention: {flash_replacements} layers optimized")
        
        # Apply LoRA
        logger.info(f"Applying LoRA adapter from: {lora_path}")
        from mlx_lm.lora import apply_lora
        
        model = apply_lora(
            model,
            r=lora_rank,
            alpha=lora_alpha,
            target_modules=target_modules
        )
        
        # Load LoRA weights
        lora_weights = mx.load(lora_path)
        model.update(lora_weights)
        
        # Generate text
        logger.info("Generating text")
        tokens = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p
        )
        
        output = tokenizer.decode(tokens)
        return True, output
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        return False, f"Inference failed: {str(e)}"