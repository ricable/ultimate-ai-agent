#!/usr/bin/env python3
"""
GPU Agentic Training System for Phi-4 Model
Optimized for Claude SDK and MCP Tools with Overfitting Prevention
"""

import os
import json
import torch
import wandb
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AgenticTrainingConfig:
    """Configuration for agentic training system"""

    # Model configuration
    model_name: str = "microsoft/Phi-4"
    model_cache_dir: str = "/app/models"
    output_dir: str = "/app/checkpoints"

    # Training hyperparameters
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_seq_length: int = 2048

    # LoRA configuration for efficient fine-tuning
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    # Overfitting prevention
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    eval_steps: int = 100
    save_steps: int = 200
    logging_steps: int = 50
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    dropout_rate: float = 0.1

    # Validation split
    validation_split: float = 0.15
    test_split: float = 0.05

    # MCP/Agentic optimization
    mcp_tool_focus: bool = True
    agentic_workflow_focus: bool = True

    # Monitoring
    use_wandb: bool = True
    wandb_project: str = "phi4-agentic-optimization"

    # Quantization
    use_8bit: bool = False
    use_4bit: bool = True


class MCPToolDatasetBuilder:
    """Build training dataset focused on MCP tool usage and agentic workflows"""

    def __init__(self, config: AgenticTrainingConfig):
        self.config = config

    def create_mcp_tool_examples(self) -> List[Dict]:
        """Create synthetic examples for MCP tool calling"""
        examples = []

        # Tool calling patterns
        tool_patterns = [
            {
                "input": "Search the codebase for authentication functions",
                "output": "I'll use the Grep tool to search for authentication functions.\n<tool_use>grep(pattern='def.*auth', path='.')</tool_use>"
            },
            {
                "input": "Create a new file for user validation",
                "output": "I'll create a user validation file.\n<tool_use>write(file_path='./validation/user.py', content='# User validation logic')</tool_use>"
            },
            {
                "input": "Read the configuration file",
                "output": "I'll read the configuration file.\n<tool_use>read(file_path='./config.json')</tool_use>"
            },
            {
                "input": "List all Python files in the project",
                "output": "I'll list all Python files.\n<tool_use>glob(pattern='**/*.py')</tool_use>"
            },
            {
                "input": "Execute the test suite",
                "output": "I'll execute the test suite.\n<tool_use>bash(command='pytest tests/', description='Run test suite')</tool_use>"
            }
        ]

        # Agentic workflow patterns
        workflow_patterns = [
            {
                "input": "Implement a new API endpoint with tests",
                "output": "I'll break this down into steps:\n1. Create endpoint file\n2. Implement logic\n3. Write tests\n4. Run validation\nStarting with step 1..."
            },
            {
                "input": "Debug the authentication error",
                "output": "Let me investigate:\n1. Check logs\n2. Review auth code\n3. Trace the error\n4. Propose fix\nAnalyzing logs first..."
            }
        ]

        examples.extend(tool_patterns)
        examples.extend(workflow_patterns)

        return examples

    def load_dataset(self) -> Dataset:
        """Load and prepare training dataset"""
        # Create MCP-focused examples
        mcp_examples = self.create_mcp_tool_examples()

        # Convert to dataset format
        data = {
            "text": [f"User: {ex['input']}\nAssistant: {ex['output']}" for ex in mcp_examples]
        }

        dataset = Dataset.from_dict(data)
        return dataset


class AgenticTrainer:
    """Main training system with overfitting prevention"""

    def __init__(self, config: AgenticTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize wandb
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                config=config.__dict__
            )

    def load_model_and_tokenizer(self):
        """Load Phi-4 model and tokenizer with quantization"""
        logger.info(f"Loading model: {self.config.model_name}")

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.model_cache_dir,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        # Model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.model_cache_dir,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            load_in_4bit=self.config.use_4bit,
            load_in_8bit=self.config.use_8bit
        )

        # Prepare for k-bit training
        if self.config.use_4bit or self.config.use_8bit:
            model = prepare_model_for_kbit_training(model)

        # Apply LoRA
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

    def prepare_datasets(self, tokenizer):
        """Prepare training and validation datasets"""
        logger.info("Preparing datasets...")

        # Build MCP-focused dataset
        dataset_builder = MCPToolDatasetBuilder(self.config)
        dataset = dataset_builder.load_dataset()

        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length"
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        # Split into train/val/test
        total_size = len(tokenized_dataset)
        val_size = int(total_size * self.config.validation_split)
        test_size = int(total_size * self.config.test_split)
        train_size = total_size - val_size - test_size

        splits = tokenized_dataset.train_test_split(test_size=val_size + test_size)
        train_dataset = splits["train"]

        val_test = splits["test"].train_test_split(test_size=test_size)
        val_dataset = val_test["train"]
        test_dataset = val_test["test"]

        logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        return train_dataset, val_dataset, test_dataset

    def train(self):
        """Execute training with overfitting prevention"""
        logger.info("Starting training...")

        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()

        # Prepare datasets
        train_dataset, val_dataset, test_dataset = self.prepare_datasets(tokenizer)

        # Training arguments with overfitting prevention
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
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            report_to="wandb" if self.config.use_wandb else "none",
            save_total_limit=3,
            gradient_checkpointing=True
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # Early stopping callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=self.config.early_stopping_patience,
            early_stopping_threshold=self.config.early_stopping_threshold
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[early_stopping]
        )

        # Train
        trainer.train()

        # Evaluate on test set
        test_results = trainer.evaluate(test_dataset)
        logger.info(f"Test results: {test_results}")

        # Save final model
        trainer.save_model(f"{self.config.output_dir}/final_model")
        tokenizer.save_pretrained(f"{self.config.output_dir}/final_model")

        # Save metrics
        metrics = {
            "test_loss": test_results["eval_loss"],
            "test_perplexity": np.exp(test_results["eval_loss"]),
            "config": self.config.__dict__
        }

        with open(f"{self.config.output_dir}/training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Training complete!")
        return metrics


def main():
    """Main entry point"""
    # Load configuration
    config = AgenticTrainingConfig()

    # Initialize trainer
    trainer = AgenticTrainer(config)

    # Train
    metrics = trainer.train()

    logger.info("Training pipeline completed successfully!")
    logger.info(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()
