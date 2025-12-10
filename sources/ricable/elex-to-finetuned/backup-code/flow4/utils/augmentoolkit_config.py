"""
Augmentoolkit Configuration Management for Flow4

Provides configuration management utilities for integrating Augmentoolkit
with Flow4's document processing pipeline, with special support for MLX models.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict

from .config import PipelineConfig
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class AugmentoolkitMLXConfig:
    """Configuration for MLX models in Augmentoolkit."""
    
    # Model settings
    model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    adapter_path: Optional[str] = None
    cache_dir: str = "./cache"
    
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 3000
    repetition_penalty: float = 1.1
    repetition_context_size: int = 20
    
    # Performance settings
    concurrency_limit: int = 5
    verbose: bool = True
    
    # Alternative model options for different sizes
    small_model: str = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    medium_model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    large_model: str = "mlx-community/Llama-3.1-8B-Instruct-4bit"


@dataclass
class AugmentoolkitDatasetConfig:
    """Configuration for dataset generation."""
    
    # Dataset types
    generate_factual: bool = True
    generate_rag: bool = True
    generate_conversations: bool = True
    
    # Quality settings
    skip_question_check: bool = False
    skip_answer_relevancy_check: bool = False
    skip_answer_accuracy_check: bool = False
    skip_repair_qa_tuples: bool = False
    
    # Dataset parameters
    chunk_size: int = 3000
    items_per_conversation: int = 3
    variation_generation_counts: int = 2
    
    # Subset settings for testing
    use_subset: bool = False
    subset_size: int = 100
    
    # Context and prompts
    dataset_context: str = "technical documentation"
    shared_instruction: str = "You are a helpful AI assistant specializing in technical documentation."
    
    # RAG specific settings
    rag_failure_percentage: float = 0.35
    rag_max_chunks: int = 3
    
    # Final assistant prompts
    final_assistant_prompts: List[str] = field(default_factory=lambda: [
        "You are a helpful AI assistant specializing in {context}.",
        "As an expert in {context}, provide accurate and detailed information.",
        "You are an AI assistant focused on providing accurate insights about {context}."
    ])


@dataclass  
class AugmentoolkitPipelineConfig:
    """Complete Augmentoolkit pipeline configuration."""
    
    # Input/Output paths
    input_dir: str = "./output/chunks"
    output_dir: str = "./output/augmentoolkit_datasets"
    models_dir: str = "./models"
    
    # Model configuration
    mlx_config: AugmentoolkitMLXConfig = field(default_factory=AugmentoolkitMLXConfig)
    
    # Dataset configuration
    dataset_config: AugmentoolkitDatasetConfig = field(default_factory=AugmentoolkitDatasetConfig)
    
    # Pipeline settings
    pipeline_type: str = "factual-datagen-pipeline"  # or "rag-data-pipeline"
    enable_pdf_cleaning: bool = False
    enable_correction_pipeline: bool = True
    enable_representation_variation: bool = False
    
    # System settings
    completion_mode: bool = False
    cite_sources_at_end: bool = True
    remove_system_prompt_ratio: float = 0.1
    remove_thought_process_ratio: float = 0.1


class AugmentoolkitConfigManager:
    """Manages Augmentoolkit configurations for Flow4."""
    
    def __init__(self, base_config_dir: Optional[str] = None):
        """
        Initialize the config manager.
        
        Args:
            base_config_dir: Directory containing Augmentoolkit configs
        """
        if base_config_dir is None:
            # Default to the Flow4 augmentoolkit configs directory
            self.base_config_dir = Path(__file__).parent.parent / "augmentoolkit" / "external_configs"
        else:
            self.base_config_dir = Path(base_config_dir)
        
        self.available_configs = self._discover_configs()
    
    def _discover_configs(self) -> Dict[str, Path]:
        """Discover available configuration files."""
        configs = {}
        
        if self.base_config_dir.exists():
            for config_file in self.base_config_dir.glob("flow4_*.yaml"):
                config_name = config_file.stem.replace("flow4_", "")
                configs[config_name] = config_file
        
        return configs
    
    def list_available_configs(self) -> List[str]:
        """List available configuration templates."""
        return list(self.available_configs.keys())
    
    def load_config_template(self, config_name: str) -> Dict[str, Any]:
        """
        Load a configuration template.
        
        Args:
            config_name: Name of the config (e.g., "factual_mlx", "rag_mlx")
            
        Returns:
            Configuration dictionary
        """
        if config_name not in self.available_configs:
            available = ", ".join(self.available_configs.keys())
            raise ValueError(f"Config '{config_name}' not found. Available: {available}")
        
        config_path = self.available_configs[config_name]
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded Augmentoolkit config: {config_name}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config {config_name}: {e}")
            raise
    
    def create_flow4_config(
        self,
        template_name: str = "factual_mlx",
        model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
        input_dir: str = "./output/chunks",
        output_dir: str = "./output/augmentoolkit_datasets",
        dataset_context: str = "technical documentation",
        **kwargs
    ) -> AugmentoolkitPipelineConfig:
        """
        Create a Flow4-specific Augmentoolkit configuration.
        
        Args:
            template_name: Base template to use
            model_name: MLX model name
            input_dir: Input directory path
            output_dir: Output directory path
            dataset_context: Context for dataset generation
            **kwargs: Additional configuration overrides
            
        Returns:
            Complete pipeline configuration
        """
        # Create base configuration
        config = AugmentoolkitPipelineConfig(
            input_dir=input_dir,
            output_dir=output_dir
        )
        
        # Update MLX configuration
        config.mlx_config.model_name = model_name
        
        # Update dataset configuration
        config.dataset_config.dataset_context = dataset_context
        
        # Apply template-specific settings
        if template_name == "rag_mlx":
            config.pipeline_type = "rag-data-pipeline"
            config.dataset_config.generate_factual = False
            config.dataset_config.generate_rag = True
            config.dataset_config.rag_failure_percentage = 0.40
        elif template_name == "complete_mlx":
            config.dataset_config.generate_factual = True
            config.dataset_config.generate_rag = True
            config.dataset_config.generate_conversations = True
            config.enable_representation_variation = True
        
        # Apply any additional overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config.mlx_config, key):
                setattr(config.mlx_config, key, value)
            elif hasattr(config.dataset_config, key):
                setattr(config.dataset_config, key, value)
        
        return config
    
    def save_config(self, config: AugmentoolkitPipelineConfig, output_path: str):
        """
        Save a configuration to a YAML file.
        
        Args:
            config: Configuration to save
            output_path: Path to save the config file
        """
        config_dict = asdict(config)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved Augmentoolkit config to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {output_path}: {e}")
            raise
    
    def convert_to_augmentoolkit_format(
        self,
        config: AugmentoolkitPipelineConfig
    ) -> Dict[str, Any]:
        """
        Convert Flow4 config to Augmentoolkit's expected format.
        
        Args:
            config: Flow4 Augmentoolkit configuration
            
        Returns:
            Dictionary in Augmentoolkit's expected format
        """
        augmentoolkit_config = {
            "pipeline": config.pipeline_type,
            
            "path": {
                "input_dirs": [{
                    "path": config.input_dir,
                    "variation_generation_counts": config.dataset_config.variation_generation_counts,
                    "factual_gen_subset_size_per_way": config.dataset_config.subset_size * 5,
                    "factual_gen_use_subset": config.dataset_config.use_subset,
                    "rag_subset_size": config.dataset_config.subset_size * 3,
                    "rag_use_subset": config.dataset_config.use_subset,
                }],
                "output_dir": config.output_dir,
                "models_dir": config.models_dir,
                "huggingface_cache_dir": config.mlx_config.cache_dir
            },
            
            "system": {
                "completion_mode": config.completion_mode,
                "cite_sources_at_end": config.cite_sources_at_end,
                "concurrency_limit": config.mlx_config.concurrency_limit,
                "use_stop": True,
                "chunk_size": config.dataset_config.chunk_size,
                "use_subset": config.dataset_config.use_subset,
                "subset_size": config.dataset_config.subset_size,
                "shared_instruction": config.dataset_config.shared_instruction,
                "remove_system_prompt_ratio": config.remove_system_prompt_ratio,
                "remove_thought_process_ratio": config.remove_thought_process_ratio,
            },
            
            "dataset_context": config.dataset_config.dataset_context,
            
            # MLX model configuration
            "factual_sft_settings": {
                "factual_small_model": config.mlx_config.model_name,
                "factual_large_model": config.mlx_config.model_name,
                "factual_small_mode": "mlx",
                "factual_large_mode": "mlx",
                "factual_small_api_key": "notused",
                "factual_large_api_key": "notused",
                "factual_small_base_url": "http://localhost:8000",
                "factual_large_base_url": "http://localhost:8000",
                "factual_cost_per_million_small_input": 0.0,
                "factual_cost_per_million_small_output": 0.0,
                "factual_cost_per_million_large_input": 0.0,
                "factual_cost_per_million_large_output": 0.0,
                "factual_use_stop": True,
                "factual_completion_mode": config.completion_mode,
                "factual_chunk_size": config.dataset_config.chunk_size,
                "final_assistant_prompts_no_rag": config.dataset_config.final_assistant_prompts,
                "items_per_conversation": config.dataset_config.items_per_conversation
            },
            
            # RAG configuration
            "rag_data": {
                "rag_failure_percentage": config.dataset_config.rag_failure_percentage,
                "rag_max_chunks": config.dataset_config.rag_max_chunks,
                "rag_small_model": config.mlx_config.model_name,
                "rag_large_model": config.mlx_config.model_name,
                "rag_small_mode": "mlx",
                "rag_large_mode": "mlx",
                "rag_small_api_key": "notused",
                "rag_large_api_key": "notused",
                "rag_small_base_url": "http://localhost:8000",
                "rag_large_base_url": "http://localhost:8000",
                "rag_cost_per_million_small_input": 0.0,
                "rag_cost_per_million_small_output": 0.0,
                "rag_cost_per_million_large_input": 0.0,
                "rag_cost_per_million_large_output": 0.0,
                "rag_use_stop": True,
                "rag_prompts": "prompts_local",
                "rag_default_prompts": "prompts_local",
                "final_assistant_prompts": [
                    prompt.replace("{context}", config.dataset_config.dataset_context)
                    for prompt in config.dataset_config.final_assistant_prompts
                ],
                "num_items_per_group": config.dataset_config.items_per_conversation
            },
            
            # Disable features not needed for local generation
            "model_auto_train": {"do_train": False},
            "model_auto_run": {"do_run": False},
            "do_not_use_llm_for_pdf_processing": True,
            
            # Final dataset settings
            "final_datasaving_settings": {
                "template": "chatml",
                "template_kwargs": {},
                "generic_dataset_paths": [],
                "generic_dataset_percentages": [],
                "max_samples_per_dataset": 10000,
                "minimum_generic_sft": 0
            }
        }
        
        return augmentoolkit_config


def get_recommended_mlx_models() -> Dict[str, Dict[str, Any]]:
    """Get recommended MLX models for different use cases."""
    return {
        "small": {
            "name": "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "description": "Fast, lightweight model for basic QA generation",
            "memory_usage": "~2GB",
            "recommended_for": ["testing", "simple QA", "fast iteration"]
        },
        "medium": {
            "name": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "description": "Balanced model for most use cases",
            "memory_usage": "~4GB", 
            "recommended_for": ["general use", "factual QA", "RAG datasets"]
        },
        "large": {
            "name": "mlx-community/Llama-3.1-8B-Instruct-4bit",
            "description": "High-quality model for complex reasoning",
            "memory_usage": "~8GB",
            "recommended_for": ["complex QA", "multi-turn conversations", "high quality"]
        }
    }


def create_default_config(
    model_size: str = "medium",
    dataset_type: str = "factual",
    **kwargs
) -> AugmentoolkitPipelineConfig:
    """
    Create a default configuration with recommended settings.
    
    Args:
        model_size: Model size ("small", "medium", "large")
        dataset_type: Dataset type ("factual", "rag", "complete")
        **kwargs: Additional configuration overrides
        
    Returns:
        Default pipeline configuration
    """
    models = get_recommended_mlx_models()
    
    if model_size not in models:
        model_size = "medium"
        logger.warning(f"Unknown model size, using 'medium'")
    
    model_name = models[model_size]["name"]
    
    config_manager = AugmentoolkitConfigManager()
    template_name = f"{dataset_type}_mlx"
    
    return config_manager.create_flow4_config(
        template_name=template_name,
        model_name=model_name,
        **kwargs
    )