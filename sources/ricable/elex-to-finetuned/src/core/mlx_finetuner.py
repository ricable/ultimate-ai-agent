"""MLX fine-tuning implementation optimized for Apple Silicon."""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, List, Any

from ..utils.logging import get_logger
from ..utils.config import MLXConfig

logger = get_logger(__name__)

# Check for MLX availability
try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
    logger.info("MLX available for Apple Silicon optimization")
except ImportError:
    HAS_MLX = False
    mx = None
    nn = None
    logger.warning("MLX not available - install with: pip install mlx>=0.12.0")


class MLXFineTuner:
    """MLX fine-tuning implementation for Apple Silicon."""
    
    def __init__(self, config: Optional[MLXConfig] = None):
        """Initialize MLX fine-tuner.
        
        Args:
            config: MLX configuration
        """
        self.config = config or MLXConfig()
        
        if not HAS_MLX:
            logger.error("MLX not available. Install with: pip install mlx>=0.12.0 mlx-lm>=0.8.0")
            return
        
        # Auto-optimize for M3 Max if enabled
        if self.config.auto_optimize_m3:
            self._optimize_for_m3_max()
    
    def _optimize_for_m3_max(self):
        """Optimize settings for M3 Max hardware."""
        if self.config.model_size in self.config.m3_max_settings:
            settings = self.config.m3_max_settings[self.config.model_size]
            self.config.batch_size = settings["batch_size"]
            self.config.learning_rate = settings["learning_rate"]
            self.config.num_iters = settings["num_iters"]
            
            logger.info(f"Optimized for M3 Max with {self.config.model_size} settings:")
            logger.info(f"  Batch size: {self.config.batch_size}")
            logger.info(f"  Learning rate: {self.config.learning_rate}")
            logger.info(f"  Iterations: {self.config.num_iters}")
    
    def finetune(self, dataset_path: str, output_dir: str) -> Optional[str]:
        """Run MLX fine-tuning on the dataset.
        
        Args:
            dataset_path: Path to training dataset
            output_dir: Output directory for fine-tuned model
            
        Returns:
            Path to fine-tuned model or None if failed
        """
        logger.info("=" * 80)
        logger.info("🚀 STARTING MLX FINE-TUNING PROCESS")
        logger.info("=" * 80)
        
        if not HAS_MLX:
            logger.error("❌ MLX not available - cannot proceed with fine-tuning")
            return None
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"📊 Fine-tuning Configuration:")
            logger.info(f"   📁 Dataset: {dataset_path}")
            logger.info(f"   📁 Output Directory: {output_dir}")
            logger.info(f"   🤖 Model: {self.config.model_name}")
            logger.info(f"   🔄 Batch Size: {self.config.batch_size}")
            logger.info(f"   📈 Learning Rate: {self.config.learning_rate}")
            logger.info(f"   ⚙️  Iterations: {self.config.num_iters}")
            logger.info(f"   🔗 LoRA Rank: {self.config.lora_rank}")
            logger.info(f"   📏 LoRA Alpha: {self.config.lora_alpha}")
            logger.info(f"   🎯 Dropout: {self.config.lora_dropout}")
            
            # Prepare dataset for MLX
            logger.info("\n🔄 Phase 1: Dataset Preparation")
            logger.info("-" * 40)
            formatted_dataset = self._format_dataset_for_mlx(dataset_path)
            if not formatted_dataset:
                logger.error("❌ Failed to format dataset for MLX")
                return None
            logger.info(f"✅ Dataset formatted successfully: {formatted_dataset}")
            
            # Run LoRA fine-tuning
            logger.info("\n🎯 Phase 2: LoRA Training")
            logger.info("-" * 40)
            adapter_path = self._run_lora_training(formatted_dataset, output_path)
            if not adapter_path:
                logger.error("❌ LoRA training failed")
                return None
            logger.info(f"✅ LoRA training completed: {adapter_path}")
            
            # Fuse model if requested
            if self.config.fuse_model:
                logger.info("\n🔗 Phase 3: Model Fusing")
                logger.info("-" * 40)
                fused_path = self._fuse_model(adapter_path, output_path)
                if fused_path:
                    logger.info("=" * 80)
                    logger.info("🎉 FINE-TUNING COMPLETED SUCCESSFULLY!")
                    logger.info(f"📁 Final Model: {fused_path}")
                    logger.info("=" * 80)
                    return fused_path
                else:
                    logger.warning("⚠️ Model fusing failed, returning LoRA adapter")
                    logger.info("=" * 80)
                    logger.info("⚠️ FINE-TUNING COMPLETED WITH WARNINGS")
                    logger.info(f"📁 LoRA Adapter: {adapter_path}")
                    logger.info("=" * 80)
                    return adapter_path
            
            logger.info("=" * 80)
            logger.info("✅ FINE-TUNING COMPLETED SUCCESSFULLY!")
            logger.info(f"📁 LoRA Adapter: {adapter_path}")
            logger.info("=" * 80)
            return adapter_path
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error("❌ FINE-TUNING FAILED")
            logger.error(f"💥 Error: {e}")
            logger.error("=" * 80)
            return None
    
    def _format_dataset_for_mlx(self, dataset_path: str) -> Optional[str]:
        """Format dataset for MLX training.
        
        Args:
            dataset_path: Path to input dataset
            
        Returns:
            Path to formatted dataset or None if failed
        """
        try:
            logger.info(f"📂 Loading dataset from: {dataset_path}")
            
            # Load dataset
            with open(dataset_path, 'r', encoding='utf-8') as f:
                if dataset_path.endswith('.jsonl'):
                    data = [json.loads(line) for line in f]
                    logger.info(f"📊 Loaded {len(data)} examples from JSONL format")
                else:
                    data = json.load(f)
                    logger.info(f"📊 Loaded data from JSON format")
            
            # Format for MLX training
            formatted_data = []
            logger.info("🔄 Converting dataset to MLX ChatML format...")
            
            if isinstance(data, dict) and 'training_data' in data:
                logger.info("📋 Detected instruction-response format")
                for item in data['training_data']:
                    formatted_item = {
                        "text": f"<|user|>\n{item['instruction']}\n{item.get('input', '')}\n<|assistant|>\n{item['output']}"
                    }
                    formatted_data.append(formatted_item)
                logger.info(f"✅ Converted {len(formatted_data)} instruction-response pairs")
            
            elif isinstance(data, dict) and 'conversations' in data:
                logger.info("💬 Detected conversation format")
                for conv in data['conversations']:
                    if 'messages' in conv:
                        text_parts = []
                        for msg in conv['messages']:
                            role = msg['role']
                            content = msg['content']
                            text_parts.append(f"<|{role}|>\n{content}")
                        
                        formatted_item = {"text": "\n".join(text_parts)}
                        formatted_data.append(formatted_item)
                logger.info(f"✅ Converted {len(formatted_data)} conversations")
            
            elif isinstance(data, list):
                logger.info("📝 Detected list format (JSONL)")
                chatgml_count = 0
                prompt_completion_count = 0
                text_count = 0
                
                for item in data:
                    if 'messages' in item:
                        # ChatML format with messages
                        text_parts = []
                        for msg in item['messages']:
                            role = msg['role']
                            content = msg['content']
                            if role == 'system':
                                text_parts.append(f"<|system|>\n{content}")
                            elif role == 'user':
                                text_parts.append(f"<|user|>\n{content}")
                            elif role == 'assistant':
                                text_parts.append(f"<|assistant|>\n{content}")
                        
                        formatted_item = {"text": "\n".join(text_parts)}
                        formatted_data.append(formatted_item)
                        chatgml_count += 1
                    elif 'prompt' in item and 'completion' in item:
                        formatted_item = {
                            "text": f"<|user|>\n{item['prompt']}\n<|assistant|>\n{item['completion']}"
                        }
                        formatted_data.append(formatted_item)
                        prompt_completion_count += 1
                    elif 'text' in item:
                        formatted_data.append({"text": item['text']})
                        text_count += 1
                
                logger.info(f"📊 Conversion summary:")
                logger.info(f"   💬 ChatML messages: {chatgml_count}")
                logger.info(f"   📝 Prompt-completion: {prompt_completion_count}")
                logger.info(f"   📄 Raw text: {text_count}")
                logger.info(f"   📋 Total examples: {len(formatted_data)}")
            
            # Create MLX dataset directory structure
            logger.info("📁 Creating MLX dataset directory structure...")
            dataset_dir = Path(dataset_path).parent / "mlx_dataset"
            dataset_dir.mkdir(exist_ok=True)
            logger.info(f"   📂 Dataset directory: {dataset_dir}")
            
            # Split data for train/validation (90/10 split)
            logger.info("🔀 Splitting data into train/validation sets...")
            split_idx = int(len(formatted_data) * 0.9)
            train_data = formatted_data[:split_idx] if split_idx > 0 else formatted_data
            valid_data = formatted_data[split_idx:] if split_idx < len(formatted_data) else formatted_data[:1]  # At least 1 validation example
            
            logger.info(f"   📊 Train examples: {len(train_data)} (90%)")
            logger.info(f"   📊 Validation examples: {len(valid_data)} (10%)")
            
            # Save train.jsonl
            logger.info("💾 Saving training dataset...")
            train_path = dataset_dir / "train.jsonl"
            with open(train_path, 'w', encoding='utf-8') as f:
                for item in train_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logger.info(f"   ✅ Saved: {train_path}")
            
            # Save valid.jsonl  
            logger.info("💾 Saving validation dataset...")
            valid_path = dataset_dir / "valid.jsonl"
            with open(valid_path, 'w', encoding='utf-8') as f:
                for item in valid_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logger.info(f"   ✅ Saved: {valid_path}")
            
            logger.info(f"✅ Dataset preparation complete!")
            logger.info(f"   📈 Ready for MLX training with {len(train_data)} training + {len(valid_data)} validation examples")
            return str(dataset_dir)
            
        except Exception as e:
            logger.error(f"Error formatting dataset: {e}")
            return None
    
    def _run_lora_training(self, dataset_path: str, output_dir: Path) -> Optional[str]:
        """Run LoRA training using MLX.
        
        Args:
            dataset_path: Path to formatted dataset
            output_dir: Output directory
            
        Returns:
            Path to trained adapter or None if failed
        """
        try:
            adapter_dir = output_dir / self.config.adapter_dir
            adapter_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📂 Adapter output directory: {adapter_dir}")
            
            # Build MLX training command (updated for new MLX CLI)
            cmd = [
                "python", "-m", "mlx_lm", "lora",
                "--model", self.config.model_name,
                "--train",
                "--data", dataset_path,  # dataset_path is now a directory containing train.jsonl and valid.jsonl
                "--batch-size", str(self.config.batch_size),
                "--learning-rate", str(self.config.learning_rate),
                "--iters", str(self.config.num_iters),
                "--steps-per-eval", str(self.config.steps_per_eval),
                "--save-every", str(self.config.steps_per_eval),
                "--adapter-path", str(adapter_dir),
                "--val-batches", "2",  # Further reduced validation batches
                "--max-seq-length", "512",  # Further reduced sequence length
                "--grad-checkpoint"  # Enable gradient checkpointing for memory efficiency
            ]
            
            logger.info("🚀 Starting LoRA training with MLX...")
            logger.info("📋 Training command:")
            logger.info(f"   {' '.join(cmd)}")
            logger.info("\n⏱️  Training in progress... (this may take a while)")
            logger.info("   📊 You should see model loading, dataset preparation, and training progress")
            logger.info("   📈 Training loss and validation metrics will be displayed")
            logger.info("   💾 Model checkpoints will be saved periodically")
            
            # Run training
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeouts.mlx_training_timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ LoRA training completed successfully!")
                logger.info("📊 Training output:")
                if result.stdout:
                    for line in result.stdout.split('\n'):
                        if line.strip():
                            logger.info(f"   {line}")
                return str(adapter_dir)
            else:
                logger.error("❌ LoRA training failed!")
                logger.error("📊 Training stdout:")
                if result.stdout:
                    for line in result.stdout.split('\n'):
                        if line.strip():
                            logger.error(f"   {line}")
                logger.error("📊 Training stderr:")
                if result.stderr:
                    for line in result.stderr.split('\n'):
                        if line.strip():
                            logger.error(f"   {line}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("LoRA training timed out")
            return None
        except Exception as e:
            logger.error(f"Error during LoRA training: {e}")
            return None
    
    def _fuse_model(self, adapter_path: str, output_dir: Path) -> Optional[str]:
        """Fuse LoRA adapter with base model.
        
        Args:
            adapter_path: Path to LoRA adapter
            output_dir: Output directory
            
        Returns:
            Path to fused model or None if failed
        """
        try:
            fused_dir = output_dir / self.config.fused_model_dir
            fused_dir.mkdir(parents=True, exist_ok=True)
            
            # Build fuse command
            cmd = [
                "python", "-m", "mlx_lm.fuse",
                "--model", self.config.model_name,
                "--adapter-path", adapter_path,
                "--save-path", str(fused_dir)
            ]
            
            logger.info("Fusing LoRA adapter with base model...")
            logger.info(f"Command: {' '.join(cmd)}")
            
            # Run fusing
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeouts.mlx_model_fusing_timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ Model fusing completed successfully")
                return str(fused_dir)
            else:
                logger.error(f"Model fusing failed:")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("Model fusing timed out")
            return None
        except Exception as e:
            logger.error(f"Error during model fusing: {e}")
            return None
    
    def generate_text(
        self, 
        model_path: str, 
        prompt: str, 
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> Optional[str]:
        """Generate text using the fine-tuned model.
        
        Args:
            model_path: Path to fine-tuned model
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text or None if failed
        """
        if not HAS_MLX:
            logger.error("MLX not available")
            return None
        
        try:
            # Build generation command
            cmd = [
                "python", "-m", "mlx_lm.generate",
                "--model", model_path,
                "--prompt", prompt,
                "--max-tokens", str(max_tokens),
                "--temp", str(temperature)
            ]
            
            # Run generation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeouts.mlx_generation_timeout
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.error(f"Text generation failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("Text generation timed out")
            return None
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            return None
    
    def interactive_chat(self, model_path: str) -> None:
        """Start interactive chat with the fine-tuned model.
        
        Args:
            model_path: Path to fine-tuned model
        """
        if not HAS_MLX:
            logger.error("MLX not available")
            return
        
        try:
            print("\n🤖 Starting interactive chat with fine-tuned model...")
            print("Type 'quit' or 'exit' to end the session.\n")
            
            while True:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print("🤖 Assistant: ", end="", flush=True)
                
                # Generate response
                response = self.generate_text(
                    model_path=model_path,
                    prompt=f"<|user|>\n{user_input}\n<|assistant|>\n",
                    max_tokens=200,
                    temperature=0.7
                )
                
                if response:
                    # Extract assistant response (remove prompt echo)
                    assistant_part = response.split("<|assistant|>")[-1].strip()
                    print(assistant_part)
                else:
                    print("Sorry, I couldn't generate a response.")
                    
        except KeyboardInterrupt:
            print("\n👋 Chat session ended.")
        except Exception as e:
            logger.error(f"Error during interactive chat: {e}")
    
    def validate_model(self, model_path: str, test_prompts: List[str]) -> Dict[str, Any]:
        """Validate the fine-tuned model with test prompts.
        
        Args:
            model_path: Path to fine-tuned model
            test_prompts: List of test prompts
            
        Returns:
            Validation results
        """
        if not HAS_MLX:
            logger.error("MLX not available")
            return {"error": "MLX not available"}
        
        results = {
            "model_path": model_path,
            "test_results": [],
            "avg_response_length": 0,
            "successful_generations": 0,
            "total_prompts": len(test_prompts)
        }
        
        response_lengths = []
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"Testing prompt {i+1}/{len(test_prompts)}")
            
            response = self.generate_text(
                model_path=model_path,
                prompt=prompt,
                max_tokens=150
            )
            
            test_result = {
                "prompt": prompt,
                "response": response,
                "success": response is not None,
                "response_length": len(response.split()) if response else 0
            }
            
            results["test_results"].append(test_result)
            
            if response:
                results["successful_generations"] += 1
                response_lengths.append(len(response.split()))
        
        # Calculate statistics
        if response_lengths:
            results["avg_response_length"] = sum(response_lengths) / len(response_lengths)
        
        results["success_rate"] = results["successful_generations"] / results["total_prompts"]
        
        logger.info(f"Validation complete: {results['success_rate']:.2%} success rate")
        return results
    
    def is_available(self) -> bool:
        """Check if MLX is available and functional.
        
        Returns:
            True if MLX is available, False otherwise
        """
        return HAS_MLX