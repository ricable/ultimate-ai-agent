#!/usr/bin/env python3
"""
Fine-tune Phi-4 for Claude Agent SDK MCP Tools
With validation and benchmarking
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

# Import dataset builder
import sys
sys.path.append(str(Path(__file__).parent))
from mcp_dataset import MCPDatasetBuilder

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MCPFineTuningConfig:
    """Configuration for MCP tool fine-tuning"""

    # Model
    model_name: str = "microsoft/Phi-4"
    model_cache_dir: str = "/app/models"
    output_dir: str = "/app/checkpoints/mcp_finetuned"

    # Training - optimized for MCP tools
    learning_rate: float = 5e-5  # Higher for tool-calling
    num_train_epochs: int = 5  # More epochs for tool learning
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 200
    max_seq_length: int = 1024  # Shorter for tool calls

    # LoRA - tuned for tool calling
    lora_r: int = 32  # Higher rank for more capacity
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None

    # Overfitting prevention
    early_stopping_patience: int = 3
    validation_split: float = 0.2  # More validation for tools
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # MCP specific
    tool_focused: bool = True
    augment_dataset: bool = True
    validation_frequency: int = 100

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class MCPFineTuner:
    """Fine-tune Phi-4 for MCP tool calling"""

    def __init__(self, config: MCPFineTuningConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def prepare_dataset(self) -> tuple:
        """Prepare MCP tool calling dataset"""
        logger.info("Preparing MCP tool dataset...")

        # Build dataset
        dataset_builder = MCPDatasetBuilder()
        dataset_examples = dataset_builder.build_dataset()

        # Augment if requested
        if self.config.augment_dataset:
            logger.info("Augmenting dataset with variations...")
            augmented = self._augment_examples(dataset_examples)
            dataset_examples.extend(augmented)

        logger.info(f"Total training examples: {len(dataset_examples)}")

        # Convert to Dataset
        texts = [ex["text"] for ex in dataset_examples]
        dataset = Dataset.from_dict({"text": texts})

        # Split
        total_size = len(dataset)
        val_size = int(total_size * self.config.validation_split)
        train_size = total_size - val_size

        splits = dataset.train_test_split(test_size=val_size, seed=42)
        train_dataset = splits["train"]
        val_dataset = splits["test"]

        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

        return train_dataset, val_dataset

    def _augment_examples(self, examples: List[Dict]) -> List[Dict]:
        """Augment dataset with variations"""
        augmented = []

        # Add paraphrased versions
        paraphrases = {
            "Read": ["Show me", "Display", "Open", "View"],
            "Create": ["Write", "Make", "Generate"],
            "Search": ["Find", "Look for", "Locate"],
            "List": ["Show all", "Display all", "Get all"],
            "Run": ["Execute", "Start", "Launch"],
        }

        for example in examples[:20]:  # Augment subset
            text = example["text"]
            for original, replacements in paraphrases.items():
                if original in text:
                    for replacement in replacements[:2]:
                        new_text = text.replace(original, replacement, 1)
                        augmented.append({
                            "text": new_text,
                            "tool_name": example.get("tool_name", ""),
                            "difficulty": "easy",
                            "context": "augmented"
                        })

        logger.info(f"Generated {len(augmented)} augmented examples")
        return augmented

    def load_model_and_tokenizer(self):
        """Load Phi-4 with LoRA for fine-tuning"""
        logger.info(f"Loading model: {self.config.model_name}")

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.model_cache_dir,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Model with 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.model_cache_dir,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            load_in_4bit=True
        )

        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)

        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model, tokenizer

    def train(self):
        """Execute fine-tuning"""
        logger.info("Starting MCP tool fine-tuning...")

        # Load model
        model, tokenizer = self.load_model_and_tokenizer()

        # Prepare datasets
        train_dataset, val_dataset = self.prepare_dataset()

        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length"
            )

        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=50,
            eval_steps=self.config.validation_frequency,
            save_steps=self.config.validation_frequency,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            gradient_checkpointing=True,
            save_total_limit=3,
            report_to="none"  # Disable wandb for now
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # Early stopping
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=self.config.early_stopping_patience
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[early_stopping]
        )

        # Train
        logger.info("Starting training...")
        trainer.train()

        # Save final model
        final_path = f"{self.config.output_dir}/final_model"
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)

        logger.info(f"Model saved to {final_path}")

        # Evaluate
        eval_results = trainer.evaluate()
        logger.info(f"Final evaluation: {eval_results}")

        # Save metrics
        metrics = {
            "eval_loss": eval_results["eval_loss"],
            "eval_perplexity": np.exp(eval_results["eval_loss"]),
            "training_completed": True
        }

        with open(f"{self.config.output_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics


def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("MCP Tool Fine-Tuning for Claude Agent SDK")
    logger.info("=" * 60)

    # Configuration
    config = MCPFineTuningConfig()

    # Create fine-tuner
    finetuner = MCPFineTuner(config)

    # Train
    metrics = finetuner.train()

    logger.info("\n" + "=" * 60)
    logger.info("Fine-tuning complete!")
    logger.info(f"Final perplexity: {metrics['eval_perplexity']:.2f}")
    logger.info(f"Model saved to: {config.output_dir}/final_model")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
