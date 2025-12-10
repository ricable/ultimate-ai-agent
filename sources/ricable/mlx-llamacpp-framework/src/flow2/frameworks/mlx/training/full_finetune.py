"""
Full fine-tuning implementation for MLX.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

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

def finetune_full(
    model_name: str,
    train_data_path: str,
    output_dir: str,
    val_data_path: Optional[str] = None,
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-5,
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
    enable_gradient_checkpointing: bool = False,
    use_flash_attention: bool = True,
    flash_block_size: Optional[int] = None
) -> Tuple[bool, str]:
    """
    Perform full fine-tuning of a model using MLX.
    
    Args:
        model_name: Name or path of the base model
        train_data_path: Path to the training data (JSONL format)
        output_dir: Directory to save the fine-tuned model
        val_data_path: Path to the validation data (JSONL format)
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
        enable_gradient_checkpointing: Whether to enable gradient checkpointing
        
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
        
        # Enable gradient checkpointing if requested
        if enable_gradient_checkpointing:
            model.enable_checkpointing()
        
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
        
        def train_step(model, inputs, targets):
            loss, grads = nn.value_and_grad(model, loss_fn)(model, inputs, targets)
            optimizer.update(model, grads)
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
                    checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    mx.save(os.path.join(checkpoint_path, "model.npz"), model.parameters())
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            
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
            
            # Save model at the end of each epoch
            epoch_path = os.path.join(output_dir, f"epoch-{epoch+1}")
            os.makedirs(epoch_path, exist_ok=True)
            mx.save(os.path.join(epoch_path, "model.npz"), model.parameters())
            logger.info(f"Saved epoch checkpoint to {epoch_path}")
        
        # Save final model
        final_path = os.path.join(output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        mx.save(os.path.join(final_path, "model.npz"), model.parameters())
        logger.info(f"Saved final model to {final_path}")
        
        return True, f"Fine-tuning completed successfully. Model saved to {final_path}"
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {str(e)}", exc_info=True)
        return False, f"Fine-tuning failed: {str(e)}"