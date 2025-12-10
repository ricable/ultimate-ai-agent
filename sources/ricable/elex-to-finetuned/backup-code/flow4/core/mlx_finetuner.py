"""MLX-based fine-tuning for document-specific language models."""

import json
import subprocess
import sys
import tempfile
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import random

from ..utils.config import MLXConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Try to import MLX with proper error handling
try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    logger.warning("MLX not available. Fine-tuning functionality disabled.")
    HAS_MLX = False


class MLXFineTuner:
    """MLX-based fine-tuning with LoRA adaptation for document-specific models."""
    
    def __init__(self, config: Optional[MLXConfig] = None):
        """Initialize the MLX fine-tuner.
        
        Args:
            config: MLX fine-tuning configuration
        """
        if not HAS_MLX:
            raise ImportError("MLX is required for fine-tuning. Install with: pip install mlx>=0.12.0 mlx-lm>=0.8.0")
        
        self.config = config or MLXConfig()
        self._auto_optimize_for_hardware()
        
    def _auto_optimize_for_hardware(self) -> None:
        """Auto-optimize settings for M3 Max hardware if detected."""
        if not self.config.auto_optimize_m3:
            return
            
        # Detect Apple Silicon
        if platform.machine() == 'arm64' and platform.system() == 'Darwin':
            logger.info("🍎 Apple Silicon detected - optimizing for M3 Max")
            
            # Apply M3 Max optimizations based on model size
            if self.config.model_size in self.config.m3_max_settings:
                settings = self.config.m3_max_settings[self.config.model_size]
                
                # Update config with optimized settings
                self.config.batch_size = settings["batch_size"]
                self.config.learning_rate = settings["learning_rate"]
                self.config.num_iters = settings["num_iters"]
                
                logger.info(f"📊 M3 Max optimized settings applied for '{self.config.model_size}' model:")
                logger.info(f"   Batch size: {self.config.batch_size}")
                logger.info(f"   Learning rate: {self.config.learning_rate}")
                logger.info(f"   Iterations: {self.config.num_iters}")
    
    def _calculate_dynamic_timeout(self, dataset_path: str, base_timeout: int = 1800) -> int:
        """Calculate dynamic timeout based on dataset size and model complexity.
        
        Args:
            dataset_path: Path to the dataset file
            base_timeout: Base timeout in seconds (default: 30 minutes)
            
        Returns:
            Calculated timeout in seconds
        """
        try:
            # Count lines in dataset
            with open(dataset_path, 'r') as f:
                num_examples = sum(1 for _ in f)
            
            # Calculate scaling factors
            size_factor = max(1.0, num_examples / 1000)  # Scale for every 1000 examples
            model_factor = {
                'small': 1.0,
                'medium': 2.0, 
                'large': 4.0,
                'max': 6.0
            }.get(self.config.model_size, 2.0)
            
            iter_factor = max(1.0, self.config.num_iters / 500)  # Scale for iterations
            
            # Calculate final timeout (minimum 30 minutes, no maximum)
            timeout = int(base_timeout * size_factor * model_factor * iter_factor)
            timeout = max(timeout, 1800)  # Minimum 30 minutes
            
            logger.info(f"⏱️ Dynamic timeout calculated: {timeout//60} minutes")
            logger.info(f"   Dataset size: {num_examples} examples (factor: {size_factor:.1f})")
            logger.info(f"   Model size: {self.config.model_size} (factor: {model_factor:.1f})")
            logger.info(f"   Iterations: {self.config.num_iters} (factor: {iter_factor:.1f})")
            
            return timeout
            
        except Exception as e:
            logger.warning(f"⚠️ Could not calculate dynamic timeout: {e}")
            logger.info(f"🔧 Using default timeout: {base_timeout//60} minutes")
            return base_timeout
    
    def test_installation(self) -> bool:
        """Test if MLX-LM is properly installed."""
        try:
            # Test MLX core
            import mlx.core as mx
            import mlx.nn as nn
            logger.info("✅ MLX core imported successfully")
            
            # Test mlx-lm import
            result = subprocess.run([sys.executable, "-c", "import mlx_lm"], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                logger.info("✅ mlx-lm imported successfully")
                return True
            else:
                logger.error(f"❌ mlx-lm import failed: {result.stderr}")
                return False
        except ImportError as e:
            logger.error(f"❌ MLX import failed: {e}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("❌ MLX test timed out")
            return False
    
    def install_requirements(self) -> bool:
        """Install required MLX packages with extended timeout."""
        logger.info("📦 Installing MLX requirements...")
        
        packages = [
            "numpy>=1.24.0,<2.0.0",  # Fix NumPy compatibility
            "mlx>=0.12.0",
            "mlx-lm>=0.8.0",
            "tqdm>=4.65.0"
        ]
        
        for package in packages:
            try:
                # Use configured timeout for package installation
                timeout_secs = self.config.timeouts.package_install_timeout
                subprocess.run([sys.executable, "-m", "pip", "install", package], 
                             check=True, capture_output=True, timeout=timeout_secs)
                logger.info(f"✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install {package}: {e}")
                return False
            except subprocess.TimeoutExpired:
                logger.error(f"❌ Timeout installing {package} ({timeout_secs//60}min limit)")
                return False
        
        return True
    
    def convert_dataset_to_mlx_format(
        self, 
        dataset_path: str, 
        output_path: str,
        val_split: Optional[float] = None,
        test_split: Optional[float] = None
    ) -> Tuple[str, str, str]:
        """Convert Flow4 dataset to MLX-LM format with train/valid/test splits."""
        val_split = val_split or self.config.val_split
        test_split = test_split or self.config.test_split
        
        logger.info(f"🔄 Converting dataset from {dataset_path} to MLX format")
        
        # Load the JSONL dataset
        data = []
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line.strip()))
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in dataset: {e}")
        
        logger.info(f"📊 Loaded {len(data)} examples")
        
        # Convert to MLX-LM format - handle different input formats
        mlx_data = []
        for example in data:
            if 'messages' in example:
                # ChatML format
                text = ""
                for msg in example['messages']:
                    if msg['role'] == 'user':
                        text += f"<|user|>{msg['content']}"
                    elif msg['role'] == 'assistant':
                        text += f"<|assistant|>{msg['content']}<|end|>"
                mlx_data.append({"text": text})
            elif 'text' in example:
                # Already in text format
                mlx_data.append({"text": example['text']})
            elif 'prompt' in example and 'completion' in example:
                # Simple prompt-completion format
                text = f"{example['prompt']}{example['completion']}"
                mlx_data.append({"text": text})
            else:
                logger.warning(f"Skipping example with unknown format: {list(example.keys())}")
                continue
        
        logger.info(f"📊 Converted {len(mlx_data)} examples to MLX format")
        
        # Shuffle data
        random.shuffle(mlx_data)
        
        # Split into train/valid/test
        total_len = len(mlx_data)
        test_idx = int(total_len * (1 - test_split))
        valid_idx = int(test_idx * (1 - val_split / (1 - test_split)))
        
        train_data = mlx_data[:valid_idx]
        valid_data = mlx_data[valid_idx:test_idx]
        test_data = mlx_data[test_idx:]
        
        # Ensure we have at least one example in each split
        if len(valid_data) == 0 and len(train_data) > 1:
            valid_data = [train_data.pop()]
        if len(test_data) == 0 and len(train_data) > 1:
            test_data = [train_data.pop()]
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save splits
        train_file = output_dir / "train.jsonl"
        valid_file = output_dir / "valid.jsonl"
        test_file = output_dir / "test.jsonl"
        
        for file_path, data_split in [(train_file, train_data), (valid_file, valid_data), (test_file, test_data)]:
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in data_split:
                    f.write(json.dumps(item) + '\n')
        
        logger.info(f"📝 Saved {len(train_data)} training examples to {train_file}")
        logger.info(f"📝 Saved {len(valid_data)} validation examples to {valid_file}")
        logger.info(f"📝 Saved {len(test_data)} test examples to {test_file}")
        
        return str(train_file), str(valid_file), str(test_file)
    
    def fine_tune(
        self, 
        dataset_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> bool:
        """Run MLX fine-tuning with LoRA."""
        dataset_path = dataset_path or self.config.dataset_path
        output_dir = output_dir or self.config.adapter_dir
        model_name = model_name or self.config.model_name
        
        logger.info(f"🚀 Starting MLX fine-tuning")
        logger.info(f"📁 Dataset: {dataset_path}")
        logger.info(f"🤖 Model: {model_name}")
        logger.info(f"💾 Output: {output_dir}")
        
        # Check if dataset exists
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Calculate dynamic timeout based on dataset size
        dynamic_timeout = self._calculate_dynamic_timeout(dataset_path)
        
        # Create temporary directory for MLX format data
        with tempfile.TemporaryDirectory() as temp_dir:
            # Convert dataset to MLX format
            train_file, valid_file, test_file = self.convert_dataset_to_mlx_format(
                dataset_path, temp_dir
            )
            
            # Prepare fine-tuning command
            cmd = [
                "python", "-m", "mlx_lm.lora",
                "--model", model_name,
                "--data", temp_dir,
                "--train",
                "--fine-tune-type", "lora",
                "--iters", str(self.config.num_iters),
                "--steps-per-eval", str(self.config.steps_per_eval),
                "--learning-rate", str(self.config.learning_rate),
                "--batch-size", str(self.config.batch_size),
                "--num-layers", str(self.config.num_layers),
                "--adapter-path", output_dir
            ]
            
            logger.info(f"🔧 Running command: {' '.join(cmd)}")
            logger.info(f"⏱️ Estimated training time: {dynamic_timeout//60} minutes (auto-calculated)")
            logger.info("⚠️ This process has NO TIMEOUT - it will run until completion")
            logger.info("💡 You can interrupt with Ctrl+C if needed")
            logger.info("📊 Progress will be shown below:")
            logger.info("-" * 60)
            
            try:
                # Run with real-time output - NO TIMEOUT for fine-tuning
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # Stream output in real-time
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        print(output.strip())
                
                # Wait for completion with no timeout limit
                return_code = process.wait()  # Changed from poll() to wait() for complete finish
                
                if return_code == 0:
                    logger.info("-" * 60)
                    logger.info("✅ Fine-tuning completed successfully!")
                    return True
                else:
                    logger.error("-" * 60)
                    logger.error("❌ Fine-tuning failed!")
                    return False
                    
            except KeyboardInterrupt:
                logger.info("\n⚠️ Fine-tuning interrupted by user")
                if process:
                    process.terminate()
                return False
            except Exception as e:
                logger.error(f"❌ Fine-tuning failed with exception: {e}")
                if process:
                    process.terminate()
                return False
    
    def fuse_model(
        self, 
        adapter_path: Optional[str] = None,
        fused_model_path: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> bool:
        """Fuse LoRA adapters with the base model."""
        adapter_path = adapter_path or self.config.adapter_dir
        fused_model_path = fused_model_path or self.config.fused_model_dir
        model_name = model_name or self.config.model_name
        
        logger.info(f"🔗 Fusing model: {model_name} + {adapter_path} -> {fused_model_path}")
        
        cmd = [
            "python", "-m", "mlx_lm.fuse",
            "--model", model_name,
            "--adapter-path", adapter_path,
            "--save-path", fused_model_path,
            "--de-quantize"
        ]
        
        try:
            # Use configured timeout for model fusing
            timeout_secs = self.config.timeouts.mlx_model_fusing_timeout
            timeout_mins = timeout_secs // 60
            logger.info(f"⏱️ Model fusing timeout set to {timeout_mins} minutes")
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout_secs)
            logger.info(f"✅ Model fused successfully to {fused_model_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Model fusing failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Model fusing timed out after {timeout_mins} minutes")
            return False
    
    def generate_sample(
        self, 
        prompt: str,
        model_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        max_tokens: int = 100
    ) -> str:
        """Generate sample output using the fine-tuned model."""
        model_path = model_path or self.config.model_name
        
        cmd = [
            "python", "-m", "mlx_lm.generate",
            "--model", model_path,
            "--prompt", prompt,
            "--max-tokens", str(max_tokens)
        ]
        
        if adapter_path:
            cmd.extend(["--adapter-path", adapter_path])
        
        try:
            # Use configured timeout for generation
            timeout_secs = self.config.timeouts.mlx_generation_timeout
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout_secs)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Generation failed: {e}")
            return f"Error: {e.stderr}"
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Generation timed out after {timeout_secs//60} minutes")
            return f"Error: Generation timed out after {timeout_secs//60} minutes"
    
    def interactive_chat(
        self,
        model_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        fused_model_path: Optional[str] = None
    ) -> None:
        """Start interactive chat with the fine-tuned model."""
        logger.info("🤖 Starting interactive chat...")
        logger.info("💬 Type 'quit' or 'exit' to stop")
        logger.info("❓ Type 'help' for commands")
        logger.info("-" * 50)
        
        # Determine which model to use
        if fused_model_path and Path(fused_model_path).exists():
            model_to_use = fused_model_path
            adapter_to_use = None
            logger.info(f"🔗 Using fused model: {fused_model_path}")
        else:
            model_to_use = model_path or self.config.model_name
            adapter_to_use = adapter_path or self.config.adapter_dir
            logger.info(f"🧩 Using base model: {model_to_use} with adapter: {adapter_to_use}")
        
        # Chat variables
        temperature = 0.7
        max_tokens = 150
        
        while True:
            try:
                user_input = input("\\n🧑 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    logger.info("👋 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    print("""
🔧 Available commands:
- help: Show this help
- quit/exit/q: Exit chat
- clear: Clear screen
- temp <value>: Set temperature (0.1-1.0)
- tokens <value>: Set max tokens (10-500)

💡 Example prompts:
- Document: combined_document.md\\nQuestion: What is NR Dynamic Power Optimizer?\\nAnswer:
- Based on the document, explain:
- Summarize this section:
                    """)
                    continue
                elif user_input.lower() == 'clear':
                    subprocess.run(['clear'])
                    continue
                elif user_input.startswith('temp '):
                    try:
                        temperature = float(user_input.split()[1])
                        if 0.1 <= temperature <= 1.0:
                            logger.info(f"🌡️ Temperature set to {temperature}")
                        else:
                            logger.error("❌ Temperature must be between 0.1 and 1.0")
                    except (ValueError, IndexError):
                        logger.error("❌ Invalid temperature value")
                    continue
                elif user_input.startswith('tokens '):
                    try:
                        max_tokens = int(user_input.split()[1])
                        if 10 <= max_tokens <= 500:
                            logger.info(f"📝 Max tokens set to {max_tokens}")
                        else:
                            logger.error("❌ Max tokens must be between 10 and 500")
                    except (ValueError, IndexError):
                        logger.error("❌ Invalid token value")
                    continue
                elif not user_input:
                    continue
                
                # Generate response
                print("🤖 Assistant: ", end="", flush=True)
                
                cmd = [
                    "python", "-m", "mlx_lm.generate",
                    "--model", model_to_use,
                    "--prompt", user_input,
                    "--max-tokens", str(max_tokens)
                ]
                
                if adapter_to_use:
                    cmd.extend(["--adapter-path", adapter_to_use])
                
                try:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True
                    )
                    
                    # Use configured timeout for interactive chat
                    timeout_secs = self.config.timeouts.mlx_chat_response_timeout
                    output, error = process.communicate(timeout=timeout_secs)
                    
                    if process.returncode == 0:
                        # Clean up the output
                        response = output.strip()
                        if response.startswith(user_input):
                            response = response[len(user_input):].strip()
                        print(response)
                    else:
                        print(f"❌ Error: {error}")
                        
                except subprocess.TimeoutExpired:
                    timeout_mins = timeout_secs // 60
                    print(f"❌ Response timed out after {timeout_mins} minutes")
                    process.kill()
                except Exception as e:
                    print(f"❌ Error: {e}")
                    
            except KeyboardInterrupt:
                logger.info("\\n👋 Chat interrupted. Goodbye!")
                break
            except EOFError:
                logger.info("\\n👋 Chat ended. Goodbye!")
                break
    
    def run_complete_workflow(
        self,
        dataset_path: Optional[str] = None,
        skip_training: bool = False,
        skip_fusing: bool = False,
        enable_chat: bool = False
    ) -> Dict[str, Any]:
        """Run the complete fine-tuning workflow."""
        results = {
            "training_success": False,
            "fusing_success": False,
            "adapter_path": None,
            "fused_model_path": None
        }
        
        logger.info("🎯 Starting complete MLX fine-tuning workflow")
        
        # Step 1: Training
        if not skip_training:
            logger.info("📚 Step 1: Fine-tuning with LoRA...")
            training_success = self.fine_tune(dataset_path)
            results["training_success"] = training_success
            
            if training_success:
                results["adapter_path"] = self.config.adapter_dir
                logger.info(f"✅ Adapters saved to: {self.config.adapter_dir}")
            else:
                logger.error("❌ Training failed! Cannot proceed with workflow.")
                return results
        else:
            logger.info("⏭️ Skipping training as requested")
            if Path(self.config.adapter_dir).exists():
                results["training_success"] = True
                results["adapter_path"] = self.config.adapter_dir
            else:
                logger.error(f"❌ No adapters found at {self.config.adapter_dir}")
                return results
        
        # Step 2: Fusing (if enabled)
        if not skip_fusing and self.config.fuse_model:
            logger.info("🔗 Step 2: Fusing model...")
            fusing_success = self.fuse_model()
            results["fusing_success"] = fusing_success
            
            if fusing_success:
                results["fused_model_path"] = self.config.fused_model_dir
                logger.info(f"✅ Fused model saved to: {self.config.fused_model_dir}")
            else:
                logger.warning("⚠️ Model fusing failed, will use adapter mode")
        
        # Step 3: Generate sample
        logger.info("🧪 Step 3: Generating sample output...")
        sample_prompt = "Document: combined_document.md\\nQuestion: What is NR Dynamic Power Optimizer?\\nAnswer:"
        
        if results["fused_model_path"]:
            sample_output = self.generate_sample(sample_prompt, results["fused_model_path"])
        else:
            sample_output = self.generate_sample(sample_prompt, adapter_path=results["adapter_path"])
        
        logger.info(f"📄 Sample output:\\n{sample_output}")
        
        # Step 4: Interactive chat (if enabled)
        if enable_chat or self.config.enable_chat:
            logger.info("💬 Step 4: Starting interactive chat...")
            self.interactive_chat(
                adapter_path=results["adapter_path"],
                fused_model_path=results["fused_model_path"]
            )
        
        logger.info("🎉 MLX fine-tuning workflow completed!")
        return results