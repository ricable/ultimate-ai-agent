"""
LoRA and QLoRA Fine-tuning Implementation
========================================

Parameter-efficient fine-tuning with LoRA and QLoRA using PEFT and Accelerate.
"""

import os
import json
import torch
import warnings
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        BitsAndBytesConfig
    )
    from datasets import Dataset, load_dataset
    from accelerate import Accelerator
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel,
        prepare_model_for_kbit_training
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    warnings.warn("Required libraries not available for LoRA training")

from ..utils import setup_mps_device, get_optimal_dtype, cleanup_memory

@dataclass
class LoRAConfig:
    """LoRA configuration parameters."""
    r: int = 16  # Rank
    lora_alpha: int = 32  # LoRA scaling parameter
    target_modules: Optional[List[str]] = None  # Target modules for LoRA
    lora_dropout: float = 0.1  # LoRA dropout
    bias: str = "none"  # Bias type: "none", "all", "lora_only"
    task_type: str = "CAUSAL_LM"  # Task type
    
    def to_peft_config(self) -> LoraConfig:
        """Convert to PEFT LoraConfig."""
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=getattr(TaskType, self.task_type)
        )

@dataclass 
class QLoRAConfig:
    """QLoRA configuration with quantization."""
    # LoRA parameters
    lora_config: LoRAConfig
    
    # Quantization parameters
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: torch.dtype = torch.float16
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    
    def get_bnb_config(self) -> BitsAndBytesConfig:
        """Get BitsAndBytes configuration."""
        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type
        )

@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    remove_unused_columns: bool = False
    dataloader_pin_memory: bool = False  # Set to False for MPS
    fp16: bool = False  # Use with CUDA
    bf16: bool = False  # Use with newer CUDA cards
    gradient_checkpointing: bool = True
    report_to: Optional[str] = None  # "wandb", "tensorboard", None
    
    def to_training_args(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments."""
        return TrainingArguments(**asdict(self))

def create_lora_config(
    model_name: str,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None
) -> LoRAConfig:
    """
    Create LoRA configuration for a specific model.
    
    Args:
        model_name: Model name to determine target modules
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        target_modules: Specific target modules (auto-detected if None)
        
    Returns:
        LoRA configuration
    """
    if target_modules is None:
        # Auto-detect target modules based on model architecture
        model_name_lower = model_name.lower()
        
        if "llama" in model_name_lower or "alpaca" in model_name_lower:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "falcon" in model_name_lower:
            target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        elif "gpt" in model_name_lower:
            target_modules = ["c_attn", "c_proj", "c_fc"]
        elif "opt" in model_name_lower:
            target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
        elif "bloom" in model_name_lower:
            target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        elif "mistral" in model_name_lower or "mixtral" in model_name_lower:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "qwen" in model_name_lower:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            # Default fallback
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            warnings.warn(f"Unknown model architecture for {model_name}, using default target modules")
    
    return LoRAConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout
    )

def prepare_dataset(
    dataset_path: str,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    train_split: str = "train",
    eval_split: Optional[str] = "validation"
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Prepare dataset for training.
    
    Args:
        dataset_path: Path to dataset (JSONL file or HuggingFace dataset)
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        train_split: Training split name
        eval_split: Evaluation split name
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    # Load dataset
    if os.path.exists(dataset_path):
        # Local JSONL file
        if dataset_path.endswith('.jsonl'):
            dataset = load_dataset('json', data_files=dataset_path, split='train')
        else:
            dataset = load_dataset(dataset_path)
    else:
        # HuggingFace dataset
        dataset = load_dataset(dataset_path)
    
    # Split dataset
    if isinstance(dataset, dict):
        train_dataset = dataset[train_split]
        eval_dataset = dataset.get(eval_split) if eval_split else None
    else:
        if eval_split and eval_split in dataset:
            train_dataset = dataset[train_split]
            eval_dataset = dataset[eval_split]
        else:
            # Split training data
            split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
    
    def tokenize_function(examples):
        """Tokenize text examples."""
        # Handle different text formats
        if 'text' in examples:
            texts = examples['text']
        elif 'input' in examples and 'output' in examples:
            # Instruction format
            texts = [f"### Input:\n{inp}\n\n### Output:\n{out}" 
                    for inp, out in zip(examples['input'], examples['output'])]
        elif 'prompt' in examples and 'completion' in examples:
            # Prompt-completion format
            texts = [f"{prompt}{completion}" 
                    for prompt, completion in zip(examples['prompt'], examples['completion'])]
        else:
            raise ValueError("Dataset must contain 'text', 'input'/'output', or 'prompt'/'completion' fields")
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_overflowing_tokens=False,
        )
        
        # Set labels for causal LM
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Tokenize datasets
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset"
    )
    
    eval_dataset_tokenized = None
    if eval_dataset:
        eval_dataset_tokenized = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing eval dataset"
        )
    
    return train_dataset, eval_dataset_tokenized

def finetune_lora(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    lora_config: Optional[LoRAConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    max_length: int = 512,
    device: Optional[torch.device] = None,
    use_accelerate: bool = True
) -> str:
    """
    Fine-tune model with LoRA.
    
    Args:
        model_name: Model name or path
        dataset_path: Dataset path
        output_dir: Output directory
        lora_config: LoRA configuration
        training_config: Training configuration
        max_length: Maximum sequence length
        device: Target device
        use_accelerate: Use Accelerate for training
        
    Returns:
        Path to saved adapter
    """
    if not HF_AVAILABLE:
        raise ImportError("Required libraries not available")
    
    print(f"Starting LoRA fine-tuning of {model_name}")
    
    # Setup device
    if device is None:
        device = setup_mps_device()
    
    # Create configurations
    if lora_config is None:
        lora_config = create_lora_config(model_name)
    
    if training_config is None:
        training_config = TrainingConfig(output_dir=output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=get_optimal_dtype(device),
        device_map="auto" if device.type != "cpu" else None
    )
    
    # Apply LoRA
    print("Applying LoRA configuration...")
    peft_config = lora_config.to_peft_config()
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Prepare dataset
    print("Preparing dataset...")
    train_dataset, eval_dataset = prepare_dataset(
        dataset_path, tokenizer, max_length
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval dataset size: {len(eval_dataset)}")
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
        pad_to_multiple_of=8 if device.type != "mps" else None  # MPS doesn't like this
    )
    
    # Setup training arguments
    training_args = training_config.to_training_args()
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save adapter
    adapter_path = os.path.join(output_dir, "adapter")
    trainer.save_model(adapter_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(adapter_path)
    
    # Save training config
    config_path = os.path.join(adapter_path, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump({
            "lora_config": asdict(lora_config),
            "training_config": asdict(training_config),
            "model_name": model_name,
            "dataset_path": dataset_path,
            "max_length": max_length
        }, f, indent=2)
    
    print(f"LoRA adapter saved to: {adapter_path}")
    cleanup_memory()
    
    return adapter_path

def finetune_qlora(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    qlora_config: Optional[QLoRAConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    max_length: int = 512,
    device: Optional[torch.device] = None
) -> str:
    """
    Fine-tune model with QLoRA (Quantized LoRA).
    
    Args:
        model_name: Model name or path
        dataset_path: Dataset path
        output_dir: Output directory
        qlora_config: QLoRA configuration
        training_config: Training configuration
        max_length: Maximum sequence length
        device: Target device
        
    Returns:
        Path to saved adapter
    """
    if not HF_AVAILABLE:
        raise ImportError("Required libraries not available")
    
    print(f"Starting QLoRA fine-tuning of {model_name}")
    
    # Setup device
    if device is None:
        device = setup_mps_device()
    
    # Create configurations
    if qlora_config is None:
        lora_config = create_lora_config(model_name)
        qlora_config = QLoRAConfig(lora_config=lora_config)
    
    if training_config is None:
        training_config = TrainingConfig(output_dir=output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load quantized model
    print("Loading quantized model...")
    bnb_config = qlora_config.get_bnb_config()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    print("Applying LoRA configuration...")
    peft_config = qlora_config.lora_config.to_peft_config()
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Prepare dataset
    print("Preparing dataset...")
    train_dataset, eval_dataset = prepare_dataset(
        dataset_path, tokenizer, max_length
    )
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Setup training arguments
    training_args = training_config.to_training_args()
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save adapter
    adapter_path = os.path.join(output_dir, "qlora_adapter")
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    
    # Save config
    config_path = os.path.join(adapter_path, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump({
            "qlora_config": {
                "lora_config": asdict(qlora_config.lora_config),
                "quantization_config": {
                    "load_in_4bit": qlora_config.load_in_4bit,
                    "bnb_4bit_compute_dtype": str(qlora_config.bnb_4bit_compute_dtype),
                    "bnb_4bit_use_double_quant": qlora_config.bnb_4bit_use_double_quant,
                    "bnb_4bit_quant_type": qlora_config.bnb_4bit_quant_type
                }
            },
            "training_config": asdict(training_config),
            "model_name": model_name,
            "dataset_path": dataset_path,
            "max_length": max_length
        }, f, indent=2)
    
    print(f"QLoRA adapter saved to: {adapter_path}")
    cleanup_memory()
    
    return adapter_path

def merge_lora_adapter(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
    device: Optional[torch.device] = None
) -> str:
    """
    Merge LoRA adapter with base model.
    
    Args:
        base_model_name: Base model name
        adapter_path: Path to LoRA adapter
        output_path: Output path for merged model
        device: Target device
        
    Returns:
        Path to merged model
    """
    if not HF_AVAILABLE:
        raise ImportError("Required libraries not available")
    
    print(f"Merging LoRA adapter with {base_model_name}")
    
    if device is None:
        device = setup_mps_device()
    
    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=get_optimal_dtype(device),
        device_map="auto" if device.type != "cpu" else None
    )
    
    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Merge adapter
    print("Merging adapter...")
    model = model.merge_and_unload()
    
    # Save merged model
    print(f"Saving merged model to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    
    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)
    
    print(f"Merged model saved to: {output_path}")
    cleanup_memory()
    
    return output_path