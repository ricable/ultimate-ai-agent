"""
Augmentoolkit Generator for Flow4

Simplified interface to Augmentoolkit's advanced dataset generation capabilities,
designed to replace synthetic_dataset.py with more sophisticated QA generation,
validation, and multi-turn conversation capabilities.

Features:
- MLX model support for Apple Silicon
- Multi-stage validation (question quality, answer relevancy, answer accuracy)
- RAG training data generation
- Conversation generation for instruction tuning
- Integration with Flow4's existing chunking pipeline
"""

import os
import json
import asyncio
import traceback
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Flow4 core imports
from ..utils.config import DoclingConfig
from ..utils.logging import get_logger
from ..utils.mlx_dataset_formatter import MLXDatasetFormatter, MLXDatasetConfig, convert_existing_dataset_to_mlx_format

logger = get_logger(__name__)

# Augmentoolkit imports with progressive fallback
try:
    # Core engine functionality
    from ..augmentoolkit.augmentoolkit.generation_functions.engine_wrapper_class import EngineWrapper
    from ..augmentoolkit.augmentoolkit.generation_functions.mlx_engine_wrapper import create_mlx_engine
    from ..augmentoolkit.augmentoolkit.utils.observers import create_input_token_counter, create_output_token_counter
    HAS_AUGMENTOOLKIT_CORE = True
    logger.info("✅ Augmentoolkit core components loaded successfully")
except ImportError as e:
    HAS_AUGMENTOOLKIT_CORE = False
    logger.warning(f"❌ Augmentoolkit core components not available: {e}")

try:
    # Try the original chunking first
    from ..augmentoolkit.generation.core_components.chunking import chunk_text_list, read_and_chunk_text
    HAS_AUGMENTOOLKIT_CHUNKING = True
    logger.info("✅ Augmentoolkit chunking components loaded successfully")
except ImportError as e:
    logger.warning(f"❌ Original chunking not available: {e}")
    try:
        # Fall back to simplified chunking
        from ..augmentoolkit.generation.core_components.chunking_simple import chunk_text_list, read_and_chunk_text
        HAS_AUGMENTOOLKIT_CHUNKING = True
        logger.info("✅ Simplified Augmentoolkit chunking loaded successfully")
    except ImportError as e2:
        HAS_AUGMENTOOLKIT_CHUNKING = False
        logger.warning(f"❌ Simplified chunking also not available: {e2}")

try:
    # Use our fixed factual generation instead of the original
    from ..augmentoolkit.generation.core_pipelines.factual_generation_individual.factual_generation_fixed import generate_factual_qa_dataset
    HAS_FACTUAL_GENERATION = True
    logger.info("✅ Factual generation pipeline loaded successfully")
except ImportError as e:
    HAS_FACTUAL_GENERATION = False
    logger.warning(f"❌ Factual generation pipeline not available: {e}")

# Disable RAG and advanced pipelines for now - they have complex import issues
HAS_RAG_PIPELINE = False
logger.info("ℹ️ RAG pipeline disabled - not required for basic functionality")

HAS_AUGMENTOOLKIT_ADVANCED = False  
logger.info("ℹ️ Advanced pipelines disabled - not required for basic functionality")

# Overall availability check - we need at least core and chunking for basic functionality
HAS_AUGMENTOOLKIT = HAS_AUGMENTOOLKIT_CORE and HAS_AUGMENTOOLKIT_CHUNKING and HAS_FACTUAL_GENERATION


@dataclass
class AugmentoolkitConfig:
    """Configuration for Augmentoolkit generation."""
    
    # Model configuration
    model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    mode: str = "mlx"  # "mlx", "api", "local"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    adapter_path: Optional[str] = None
    
    # Generation settings
    max_tokens: int = 3000
    temperature: float = 0.7
    top_p: float = 0.95
    concurrency_limit: int = 10
    
    # Pipeline settings
    chunk_size: int = 3000
    use_subset: bool = False
    subset_size: int = 100
    completion_mode: bool = False
    
    # Validation settings
    skip_question_check: bool = False
    skip_answer_relevancy_check: bool = False
    skip_answer_accuracy_check: bool = False
    skip_repair_qa_tuples: bool = False
    
    # Dataset generation settings
    dataset_context: str = "technical documentation"
    use_filenames: bool = True
    items_per_conversation: int = 3
    final_assistant_prompts: List[str] = field(default_factory=lambda: [
        "You are a helpful AI assistant specializing in {context}.",
        "You are an expert assistant focused on {context}.",
        "As an AI assistant with expertise in {context}, please help with questions."
    ])
    
    # Multi-source recall settings
    multi_source_questions_per_chunk: int = 5
    multi_source_question_types: List[str] = field(default_factory=lambda: [
        "open_ended", "negative", "hallucination_check", "comparison"
    ])
    
    # Representation variation settings
    output_formats: List[str] = field(default_factory=lambda: [
        "json", "xml", "essay", "qa", "logic_chain", "stream_of_thought"
    ])
    variation_count: int = 3
    
    # Cost estimation (for API models)
    cost_per_million_small_input: float = 0.0
    cost_per_million_small_output: float = 0.0
    cost_per_million_large_input: float = 0.0
    cost_per_million_large_output: float = 0.0
    
    # Prompt configuration fields
    prompts: str = "prompts_local"
    default_prompts: str = "prompts_local"
    rag_failure_percentage: float = 0.5
    
    @classmethod
    def from_yaml(cls, yaml_path: Optional[str] = None) -> 'AugmentoolkitConfig':
        """Load configuration from YAML file."""
        return load_config_from_yaml(yaml_path)


def load_config_from_yaml(yaml_path: Optional[str] = None) -> AugmentoolkitConfig:
    """
    Load configuration from YAML file.
    
    Args:
        yaml_path: Path to YAML configuration file. If None, looks for 
                  'augmentoolkit-config.yaml' in project root.
    
    Returns:
        AugmentoolkitConfig instance with loaded or default values
    """
    # Default path if none provided
    if yaml_path is None:
        # Look for augmentoolkit-config.yaml in project root
        project_root = Path(__file__).parent.parent.parent.parent  # Go up from src/flow4/core/
        yaml_path = project_root / "augmentoolkit-config.yaml"
    else:
        yaml_path = Path(yaml_path)
    
    # Start with default configuration
    config = AugmentoolkitConfig()
    
    # Try to load YAML file
    try:
        if yaml_path.exists():
            logger.info(f"Loading Augmentoolkit configuration from: {yaml_path}")
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            if yaml_data:
                # Map YAML structure to config fields
                # Handle nested YAML structure from flow4_factual_full.yaml format
                
                # Model configuration
                if 'factual_sft_settings' in yaml_data:
                    factual_settings = yaml_data['factual_sft_settings']
                    if 'factual_small_model' in factual_settings:
                        config.model_name = factual_settings['factual_small_model']
                    if 'factual_small_mode' in factual_settings:
                        config.mode = factual_settings['factual_small_mode']
                    if 'factual_small_api_key' in factual_settings:
                        config.api_key = factual_settings['factual_small_api_key']
                    if 'factual_small_base_url' in factual_settings:
                        config.base_url = factual_settings['factual_small_base_url']
                    if 'final_assistant_prompts_no_rag' in factual_settings:
                        config.final_assistant_prompts = factual_settings['final_assistant_prompts_no_rag']
                    if 'items_per_conversation' in factual_settings:
                        config.items_per_conversation = factual_settings['items_per_conversation']
                
                # System configuration
                if 'system' in yaml_data:
                    system_settings = yaml_data['system']
                    if 'concurrency_limit' in system_settings:
                        config.concurrency_limit = system_settings['concurrency_limit']
                    if 'chunk_size' in system_settings:
                        config.chunk_size = system_settings['chunk_size']
                    if 'use_subset' in system_settings:
                        config.use_subset = system_settings['use_subset']
                    if 'subset_size' in system_settings:
                        config.subset_size = system_settings['subset_size']
                    if 'completion_mode' in system_settings:
                        config.completion_mode = system_settings['completion_mode']
                
                # RAG configuration
                if 'rag_data' in yaml_data:
                    rag_settings = yaml_data['rag_data']
                    if 'rag_failure_percentage' in rag_settings:
                        config.rag_failure_percentage = rag_settings['rag_failure_percentage']
                    if 'final_assistant_prompts' in rag_settings:
                        config.final_assistant_prompts = rag_settings['final_assistant_prompts']
                    if 'rag_prompts' in rag_settings:
                        config.prompts = rag_settings['rag_prompts']
                    if 'rag_default_prompts' in rag_settings:
                        config.default_prompts = rag_settings['rag_default_prompts']
                
                # Dataset context
                if 'dataset_context' in yaml_data:
                    config.dataset_context = yaml_data['dataset_context']
                
                # Direct field mapping for simple structure
                simple_mappings = {
                    'model_name': 'model_name',
                    'mode': 'mode',
                    'api_key': 'api_key',
                    'base_url': 'base_url',
                    'adapter_path': 'adapter_path',
                    'max_tokens': 'max_tokens',
                    'temperature': 'temperature',
                    'top_p': 'top_p',
                    'concurrency_limit': 'concurrency_limit',
                    'chunk_size': 'chunk_size',
                    'use_subset': 'use_subset',
                    'subset_size': 'subset_size',
                    'completion_mode': 'completion_mode',
                    'dataset_context': 'dataset_context',
                    'prompts': 'prompts',
                    'default_prompts': 'default_prompts',
                    'rag_failure_percentage': 'rag_failure_percentage'
                }
                
                for yaml_key, config_attr in simple_mappings.items():
                    if yaml_key in yaml_data:
                        setattr(config, config_attr, yaml_data[yaml_key])
                
                logger.info(f"✅ Configuration loaded successfully from {yaml_path}")
            else:
                logger.warning(f"⚠️ YAML file {yaml_path} is empty, using default configuration")
        else:
            logger.info(f"ℹ️ Configuration file {yaml_path} not found, using default values")
            
    except yaml.YAMLError as e:
        logger.error(f"❌ Error parsing YAML file {yaml_path}: {e}")
        logger.info("Using default configuration due to YAML parsing error")
    except Exception as e:
        logger.error(f"❌ Error loading configuration from {yaml_path}: {e}")
        logger.info("Using default configuration due to loading error")
    
    return config


class AugmentoolkitGenerator:
    """
    Advanced dataset generator using Augmentoolkit pipelines.
    
    Provides a simplified interface to Augmentoolkit's sophisticated dataset generation
    capabilities, with special support for MLX models on Apple Silicon.
    """
    
    def __init__(self, config: Optional[AugmentoolkitConfig] = None, yaml_path: Optional[str] = None):
        """
        Initialize the Augmentoolkit generator.
        
        Args:
            config: Configuration for the generator (optional if yaml_path is provided)
            yaml_path: Path to YAML configuration file (optional)
        """
        if not HAS_AUGMENTOOLKIT:
            raise ImportError(
                "Augmentoolkit is not available. Please check the installation."
            )
        
        # Load configuration from YAML if provided, otherwise use passed config
        if config is None:
            self.config = load_config_from_yaml(yaml_path)
        else:
            self.config = config
        
        self.engine = None
        self._setup_engine()
    
    def _setup_engine(self):
        """Set up the language model engine."""
        try:
            if self.config.mode == "mlx":
                logger.info(f"Initializing MLX engine with model: {self.config.model_name}")
                self.engine = create_mlx_engine(
                    model_name=self.config.model_name,
                    adapter_path=self.config.adapter_path,
                    verbose=True
                )
# Fallback mode removed - full Augmentoolkit functionality required
            else:
                logger.info(f"Initializing {self.config.mode} engine with model: {self.config.model_name}")
                self.engine = EngineWrapper(
                    model=self.config.model_name,
                    mode=self.config.mode,
                    api_key=self.config.api_key,
                    base_url=self.config.base_url
                )
            
            if self.engine:
                logger.info("✅ Engine initialized successfully")
            else:
                logger.info("✅ Fallback mode - no engine required")
            
        except Exception as e:
            logger.error(f"❌ Engine initialization failed: {e}")
            raise RuntimeError(f"Augmentoolkit engine initialization failed: {e}. Full functionality is required.")
    
    def validate_environment(self) -> Tuple[bool, List[str]]:
        """
        Validate the environment for Augmentoolkit generation.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        warnings = []
        
        # Check basic Python environment
        try:
            import json
            import asyncio
        except ImportError as e:
            issues.append(f"Missing basic Python modules: {e}")
        
        # Check MLX availability for MLX mode (but allow fallback)
        if self.config.mode == "mlx":
            try:
                import mlx
                import mlx.core as mx
                # Test basic MLX functionality
                _ = mx.array([1, 2, 3])
                logger.info("✅ MLX is available and functional")
            except ImportError:
                warnings.append("MLX is not installed - requires MLX for full functionality")
                is_valid = False
            except Exception as e:
                warnings.append(f"MLX is installed but not functional: {e} - requires functional MLX")
                is_valid = False
        
        # Check output directory permissions
        try:
            import tempfile
            test_dir = Path(".")  # Use current directory for testing
            test_file = test_dir / "test_write_permissions.tmp"
            test_file.write_text("test")
            test_file.unlink()
            logger.info("✅ Output directory is writable")
        except Exception as e:
            issues.append(f"Cannot write to output directory: {e}")
        
        # Log warnings
        for warning in warnings:
            logger.warning(f"⚠️ {warning}")
        
        # We can proceed even with warnings, only fail on critical issues
        return len(issues) == 0, issues
    
    async def generate_factual_dataset(
        self,
        input_chunks: List[Dict[str, Any]],
        output_dir: str,
        dataset_name: str = "factual_dataset"
    ) -> Dict[str, Any]:
        """
        Generate a factual Q&A dataset from input chunks.
        
        Args:
            input_chunks: List of chunk dictionaries with 'text' and metadata
            output_dir: Directory to save the generated dataset
            dataset_name: Name for the output dataset
            
        Returns:
            Dictionary containing generation results and statistics
        """
        logger.info(f"🚀 Starting factual dataset generation from {len(input_chunks)} chunks")
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"🏷️ Dataset name: {dataset_name}")
        logger.info(f"🔧 Configuration: model={self.config.model_name}, mode={self.config.mode}")
        logger.info(f"⚙️ Parameters: temperature={self.config.temperature}, concurrency={self.config.concurrency_limit}")
        
        try:
            # Prepare output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📂 Created output directory: {output_path}")
            
            # Ensure we have full Augmentoolkit functionality
            if not self.engine:
                raise RuntimeError("Augmentoolkit engine is not properly initialized")
            
            logger.info("🔄 Converting chunks to Augmentoolkit format...")
            # Convert chunks to text format expected by Augmentoolkit
            text_chunks = []
            for i, chunk in enumerate(input_chunks):
                chunk_text = chunk.get("text", "")
                chunk_metadata = chunk.get("metadata", {})
                
                # Log chunk information for debugging
                source = chunk_metadata.get("source", "unknown")
                chunk_preview = chunk_text[:100].replace('\n', ' ') + "..." if len(chunk_text) > 100 else chunk_text
                logger.debug(f"📄 Chunk {i} from {source}: {chunk_preview}")
                
                text_chunks.append({
                    "id": i,
                    "text": chunk_text,
                    "source": source,
                    "metadata": chunk_metadata
                })
            
            logger.info(f"✅ Converted {len(text_chunks)} chunks successfully")
            
            # Set up sampling parameters
            sampling_params = {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens,
                "stop": ["<|im_end|>", "Human:", "Assistant:"]
            }
            logger.info(f"⚙️ Sampling parameters: {sampling_params}")
            
            logger.info("🔥 Calling Augmentoolkit factual generation pipeline...")
            # Call Augmentoolkit's factual generation pipeline
            results = await generate_factual_qa_dataset(
                completion_mode=self.config.completion_mode,
                phase_index=0,
                work_in_phases=False,
                skip_filter_chunks=False,
                skip_repair_qa_tuples=self.config.skip_repair_qa_tuples,
                chunk_size=self.config.chunk_size,
                use_gutenberg=False,
                start_url="",
                max_books=0,
                max_failures=3,
                skip_conversation_generation=False,
                hub_path="",
                private=True,
                push_to_hub=False,
                use_filenames=self.config.use_filenames,
                input_dir="",  # We pass chunks directly
                prompts=self.config.prompts,
                default_prompts=self.config.default_prompts, 
                use_stop=True,
                skip_answer_relevancy_check=self.config.skip_answer_relevancy_check,
                skip_answer_accuracy_check=self.config.skip_answer_accuracy_check,
                conversation_instructions="",
                do_not_use_system_prompts=False,
                skip_question_check=self.config.skip_question_check,
                final_assistant_prompts_no_rag=self.config.final_assistant_prompts,
                final_assistant_prompts_rag=self.config.final_assistant_prompts,
                rag_failure_percentage=self.config.rag_failure_percentage,
                items_per_conversation=self.config.items_per_conversation,
                concurrency_limit=self.config.concurrency_limit,
                small_model=self.config.model_name,
                small_api_key=self.config.api_key or "notused",
                small_base_url=self.config.base_url or "http://localhost:8000",
                small_mode=self.config.mode,
                large_model=self.config.model_name,
                large_api_key=self.config.api_key or "notused", 
                large_base_url=self.config.base_url or "http://localhost:8000",
                large_mode=self.config.mode,
                use_subset=self.config.use_subset,
                subset_size=self.config.subset_size,
                double_check_counter=1,
                output_dir=str(output_path),
                cost_per_million_small_input=self.config.cost_per_million_small_input,
                cost_per_million_small_output=self.config.cost_per_million_small_output,
                cost_per_million_large_input=self.config.cost_per_million_large_input,
                cost_per_million_large_output=self.config.cost_per_million_large_output,
                read_files_manually=False,  # We pass chunks directly
                text_chunks_passed_in=text_chunks,
                do_meta_datagen=False,
                meta_datagen_keys=[],
                meta_datagen_extras=[],
                chunking_output_dir=None,
                task_id=None,
                seed=1048596
            )
            
            logger.info("📊 Processing Augmentoolkit generation results...")
            
            # Analyze results for debugging
            if isinstance(results, dict):
                logger.info(f"🔍 Results type: {type(results)}")
                logger.info(f"🗂️ Results keys: {list(results.keys())}")
                
                # Log basic statistics if available
                if "status" in results:
                    logger.info(f"📈 Generation status: {results['status']}")
                if "qa_pairs_generated" in results:
                    logger.info(f"💬 Q&A pairs generated: {results['qa_pairs_generated']}")
                if "chunks_processed" in results:
                    logger.info(f"📝 Chunks processed: {results['chunks_processed']}")
                if "output_files" in results:
                    logger.info(f"📁 Output files created: {list(results['output_files'].keys())}")
                    
                # Check for errors in results
                if "error" in results:
                    logger.error(f"❌ Pipeline error: {results['error']}")
                    return results
            else:
                logger.warning(f"⚠️ Unexpected results format: {type(results)}")
            
            # Save results summary
            summary = {
                "dataset_name": dataset_name,
                "input_chunks": len(input_chunks),
                "output_dir": str(output_path),
                "config": self.config.__dict__,
                "generation_results": results
            }
            
            summary_path = output_path / f"{dataset_name}_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved generation summary to: {summary_path}")
            
            # Check for output files and log their status
            output_files = []
            for pattern in ["*.json", "*.jsonl", "*.txt"]:
                matching_files = list(output_path.glob(pattern))
                output_files.extend(matching_files)
            
            if output_files:
                logger.info(f"📁 Generated output files:")
                for file_path in output_files:
                    file_size = file_path.stat().st_size
                    if file_size == 0:
                        logger.warning(f"⚠️ Empty file: {file_path.name} (0 bytes)")
                    else:
                        logger.info(f"✅ {file_path.name}: {file_size:,} bytes")
            else:
                logger.warning("⚠️ No output files found in the expected location")
            
            logger.info(f"🎉 Factual dataset generation complete!")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error generating factual dataset: {e}")
            logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            return {"error": str(e)}
    
    async def generate_rag_dataset(
        self,
        input_chunks: List[Dict[str, Any]],
        output_dir: str,
        dataset_name: str = "rag_dataset"
    ) -> Dict[str, Any]:
        """
        Generate a RAG training dataset from input chunks.
        
        Args:
            input_chunks: List of chunk dictionaries with 'text' and metadata
            output_dir: Directory to save the generated dataset
            dataset_name: Name for the output dataset
            
        Returns:
            Dictionary containing generation results and statistics
        """
        logger.info(f"Generating RAG dataset from {len(input_chunks)} chunks")
        
        try:
            # Prepare output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Ensure we have full Augmentoolkit functionality
            if not self.engine or not HAS_RAG_PIPELINE:
                raise RuntimeError("Augmentoolkit engine and RAG pipeline are required but not available")
            
            # Call Augmentoolkit's RAG data pipeline
            results = await rag_data_pipeline(
                input_dir="",  # We pass chunks directly
                output_dir=str(output_path),
                chunk_size=self.config.chunk_size,
                rag_failure_percentage=self.config.rag_failure_percentage,
                rag_max_chunks=3,
                use_subset=self.config.use_subset,
                subset_size=self.config.subset_size,
                concurrency_limit=self.config.concurrency_limit,
                final_assistant_prompts=self.config.final_assistant_prompts,
                num_items_per_group=self.config.items_per_conversation,
                rag_skip_filter_chunks=False,
                rag_small_model=self.config.model_name,
                rag_small_api_key=self.config.api_key or "notused",
                rag_small_base_url=self.config.base_url or "http://localhost:8000",
                rag_small_mode=self.config.mode,
                rag_large_model=self.config.model_name,
                rag_large_api_key=self.config.api_key or "notused",
                rag_large_base_url=self.config.base_url or "http://localhost:8000",
                rag_large_mode=self.config.mode,
                rag_cost_per_million_small_input=self.config.cost_per_million_small_input,
                rag_cost_per_million_small_output=self.config.cost_per_million_small_output,
                rag_cost_per_million_large_input=self.config.cost_per_million_large_input,
                rag_cost_per_million_large_output=self.config.cost_per_million_large_output,
                rag_use_stop=True,
                rag_prompts=self.config.prompts,
                rag_default_prompts=self.config.default_prompts,
                text_chunks_passed_in=input_chunks
            )
            
            # Save results summary
            summary = {
                "dataset_name": dataset_name,
                "input_chunks": len(input_chunks),
                "output_dir": str(output_path),
                "config": self.config.__dict__,
                "generation_results": results
            }
            
            summary_path = output_path / f"{dataset_name}_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ RAG dataset generation complete: {summary_path}")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error generating RAG dataset: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    async def generate_multi_source_dataset(
        self,
        input_chunks: List[Dict[str, Any]],
        output_dir: str,
        dataset_name: str = "multi_source_dataset"
    ) -> Dict[str, Any]:
        """
        Generate a multi-source reasoning dataset from input chunks.
        
        This pipeline creates complex questions that require information synthesis
        across multiple document chunks, testing advanced reasoning capabilities.
        
        Args:
            input_chunks: List of chunk dictionaries with 'text' and metadata
            output_dir: Directory to save the generated dataset
            dataset_name: Name for the output dataset
            
        Returns:
            Dictionary containing generation results and statistics
        """
        logger.info(f"Generating multi-source dataset from {len(input_chunks)} chunks")
        
        try:
            # Prepare output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Convert chunks to text format expected by Augmentoolkit
            text_chunks = []
            for i, chunk in enumerate(input_chunks):
                chunk_text = chunk.get("text", "")
                chunk_metadata = chunk.get("metadata", {})
                
                text_chunks.append({
                    "id": i,
                    "text": chunk_text,
                    "source": chunk_metadata.get("source", "unknown"),
                    "metadata": chunk_metadata
                })
            
            # Call Augmentoolkit's multi-source recall pipeline
            results = await generate_multi_source_dataset(
                completion_mode=self.config.completion_mode,
                use_subset=self.config.use_subset,
                subset_size=self.config.subset_size,
                concurrency_limit=self.config.concurrency_limit,
                small_model=self.config.model_name,
                small_api_key=self.config.api_key or "notused",
                small_base_url=self.config.base_url or "http://localhost:8000",
                small_mode=self.config.mode,
                large_model=self.config.model_name,
                large_api_key=self.config.api_key or "notused",
                large_base_url=self.config.base_url or "http://localhost:8000",
                large_mode=self.config.mode,
                output_dir=str(output_path),
                cost_per_million_small_input=self.config.cost_per_million_small_input,
                cost_per_million_small_output=self.config.cost_per_million_small_output,
                cost_per_million_large_input=self.config.cost_per_million_large_input,
                cost_per_million_large_output=self.config.cost_per_million_large_output,
                questions_per_chunk=self.config.multi_source_questions_per_chunk,
                question_types=self.config.multi_source_question_types,
                text_chunks_passed_in=text_chunks,
                chunk_size=self.config.chunk_size
            )
            
            # Save results summary
            summary = {
                "dataset_name": dataset_name,
                "input_chunks": len(input_chunks),
                "output_dir": str(output_path),
                "config": self.config.__dict__,
                "generation_results": results
            }
            
            summary_path = output_path / f"{dataset_name}_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Multi-source dataset generation complete: {summary_path}")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error generating multi-source dataset: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    async def generate_representation_variation_dataset(
        self,
        input_chunks: List[Dict[str, Any]],
        output_dir: str,
        dataset_name: str = "representation_variation_dataset"
    ) -> Dict[str, Any]:
        """
        Generate diverse representations of content for continued pretraining.
        
        This pipeline creates the same factual content in multiple formats
        (JSON, XML, essays, Q&A, logic chains, stream-of-thought) for
        domain adaptation and representation learning.
        
        Args:
            input_chunks: List of chunk dictionaries with 'text' and metadata
            output_dir: Directory to save the generated dataset
            dataset_name: Name for the output dataset
            
        Returns:
            Dictionary containing generation results and statistics
        """
        logger.info(f"Generating representation variation dataset from {len(input_chunks)} chunks")
        
        try:
            # Prepare output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Convert chunks to text format expected by Augmentoolkit
            text_chunks = []
            for i, chunk in enumerate(input_chunks):
                chunk_text = chunk.get("text", "")
                chunk_metadata = chunk.get("metadata", {})
                
                text_chunks.append({
                    "id": i,
                    "text": chunk_text,
                    "source": chunk_metadata.get("source", "unknown"),
                    "metadata": chunk_metadata
                })
            
            # Call Augmentoolkit's representation variation pipeline
            results = await representation_variation_pipeline(
                completion_mode=self.config.completion_mode,
                use_subset=self.config.use_subset,
                subset_size=self.config.subset_size,
                concurrency_limit=self.config.concurrency_limit,
                small_model=self.config.model_name,
                small_api_key=self.config.api_key or "notused",
                small_base_url=self.config.base_url or "http://localhost:8000",
                small_mode=self.config.mode,
                large_model=self.config.model_name,
                large_api_key=self.config.api_key or "notused",
                large_base_url=self.config.base_url or "http://localhost:8000",
                large_mode=self.config.mode,
                output_dir=str(output_path),
                cost_per_million_small_input=self.config.cost_per_million_small_input,
                cost_per_million_small_output=self.config.cost_per_million_small_output,
                cost_per_million_large_input=self.config.cost_per_million_large_input,
                cost_per_million_large_output=self.config.cost_per_million_large_output,
                output_formats=self.config.output_formats,
                variation_count=self.config.variation_count,
                text_chunks_passed_in=text_chunks,
                chunk_size=self.config.chunk_size
            )
            
            # Save results summary
            summary = {
                "dataset_name": dataset_name,
                "input_chunks": len(input_chunks),
                "output_dir": str(output_path),
                "config": self.config.__dict__,
                "generation_results": results
            }
            
            summary_path = output_path / f"{dataset_name}_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Representation variation dataset generation complete: {summary_path}")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error generating representation variation dataset: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    async def generate_complete_factual_dataset(
        self,
        input_chunks: List[Dict[str, Any]],
        output_dir: str,
        dataset_name: str = "complete_factual_dataset",
        models_dir: str = "./models"
    ) -> Dict[str, Any]:
        """
        Generate a complete factual dataset using the full Augmentoolkit composition pipeline.
        
        This is the most comprehensive pipeline that combines multiple generation
        strategies including factual QA, RAG training, multi-source reasoning,
        and representation variation.
        
        Args:
            input_chunks: List of chunk dictionaries with 'text' and metadata
            output_dir: Directory to save the generated dataset
            dataset_name: Name for the output dataset
            models_dir: Directory for model outputs and configs
            
        Returns:
            Dictionary containing generation results and statistics
        """
        logger.info(f"Generating complete factual dataset from {len(input_chunks)} chunks")
        
        try:
            # Prepare output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            models_path = Path(models_dir)
            models_path.mkdir(parents=True, exist_ok=True)
            
            # Convert chunks to input directory format expected by the pipeline
            input_dirs = [{
                "path": str(output_path / "input_chunks"),
                "variation_generation_counts": self.config.variation_count,
                "factual_gen_subset_size_per_way": self.config.subset_size,
                "rag_subset_size": max(50, self.config.subset_size // 2)
            }]
            
            # Save chunks to the expected input format
            input_chunks_dir = output_path / "input_chunks"
            input_chunks_dir.mkdir(parents=True, exist_ok=True)
            
            for i, chunk in enumerate(input_chunks):
                chunk_file = input_chunks_dir / f"chunk_{i:04d}.txt"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    f.write(chunk.get("text", ""))
            
            # For now, we'll run individual pipelines instead of the full composition
            # since the full pipeline has many complex parameters
            logger.info("Running individual pipelines for complete dataset...")
            
            results = {}
            
            # Generate factual dataset
            factual_results = await self.generate_factual_dataset(
                input_chunks, str(output_path / "factual"), "factual"
            )
            results["factual"] = factual_results
            
            # Generate RAG dataset
            rag_results = await self.generate_rag_dataset(
                input_chunks, str(output_path / "rag"), "rag"
            )
            results["rag"] = rag_results
            
            # Save results summary
            summary = {
                "dataset_name": dataset_name,
                "input_chunks": len(input_chunks),
                "output_dir": str(output_path),
                "models_dir": str(models_path),
                "config": self.config.__dict__,
                "generation_results": results
            }
            
            summary_path = output_path / f"{dataset_name}_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Complete factual dataset generation complete: {summary_path}")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error generating complete factual dataset: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
# Fallback methods removed - full Augmentoolkit functionality required
    async def _removed_fallback_factual_dataset(
        self,
        input_chunks: List[Dict[str, Any]],
        output_dir: str,
        dataset_name: str
    ) -> Dict[str, Any]:
        """Generate factual dataset using fallback method (no LLM required)."""
        logger.info("Generating factual dataset using fallback method")
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            qa_pairs = []
            for i, chunk in enumerate(input_chunks):
                chunk_text = chunk.get("text", "")
                chunk_metadata = chunk.get("metadata", {})
                source = chunk_metadata.get("source", "unknown")
                
                if not chunk_text.strip():
                    continue
                
                # Create diverse questions for each chunk
                questions = [
                    f"What is the main topic discussed in this section from {source}?",
                    f"Summarize the key points from this {source} section.",
                    f"What are the important details mentioned in this part of {source}?",
                    f"Explain the concepts described in this {source} content.",
                    f"What information is provided in this section of {source}?"
                ]
                
                for j, question in enumerate(questions):
                    # Create answer by using first part of chunk
                    answer = chunk_text[:400] + "..." if len(chunk_text) > 400 else chunk_text
                    
                    qa_pairs.append({
                        "id": f"{dataset_name}_{i}_{j}",
                        "question": question,
                        "answer": answer,
                        "source": source,
                        "chunk_id": i,
                        "metadata": chunk_metadata
                    })
            
            # Save results
            results = {
                "qa_pairs": qa_pairs,
                "total_pairs": len(qa_pairs),
                "chunks_processed": len(input_chunks),
                "generation_method": "augmentoolkit_fallback"
            }
            
            # Save as JSON
            json_file = output_path / f"{dataset_name}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Save as JSONL for training
            jsonl_file = output_path / f"{dataset_name}.jsonl"
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                for pair in qa_pairs:
                    f.write(json.dumps({
                        "prompt": pair["question"],
                        "completion": pair["answer"],
                        "metadata": {
                            "source": pair["source"],
                            "chunk_id": pair["chunk_id"]
                        }
                    }, ensure_ascii=False) + "\n")
            
            summary = {
                "dataset_name": dataset_name,
                "input_chunks": len(input_chunks),
                "output_dir": str(output_path),
                "qa_pairs_generated": len(qa_pairs),
                "files_created": [str(json_file), str(jsonl_file)],
                "generation_method": "augmentoolkit_fallback"
            }
            
            logger.info(f"✅ Generated {len(qa_pairs)} Q&A pairs using fallback method")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error in fallback factual generation: {e}")
            return {"error": str(e)}
    
    async def _removed_fallback_rag_dataset(
        self,
        input_chunks: List[Dict[str, Any]],
        output_dir: str,
        dataset_name: str
    ) -> Dict[str, Any]:
        """Generate RAG dataset using fallback method (no LLM required)."""
        logger.info("Generating RAG dataset using fallback method")
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            rag_examples = []
            for i, chunk in enumerate(input_chunks):
                chunk_text = chunk.get("text", "")
                chunk_metadata = chunk.get("metadata", {})
                source = chunk_metadata.get("source", "unknown")
                
                if not chunk_text.strip():
                    continue
                
                # Create RAG-style examples (context + question + answer)
                examples = [
                    {
                        "context": chunk_text,
                        "question": f"Based on the provided context from {source}, what are the main points discussed?",
                        "answer": f"Based on the provided context, the main points include: {chunk_text[:300]}..."
                    },
                    {
                        "context": chunk_text,
                        "question": f"What key information can be extracted from this {source} content?",
                        "answer": f"The key information extracted from this content includes: {chunk_text[:300]}..."
                    },
                    {
                        "context": chunk_text,
                        "question": f"How would you summarize the content provided from {source}?",
                        "answer": f"The content can be summarized as follows: {chunk_text[:300]}..."
                    }
                ]
                
                for j, example in enumerate(examples):
                    rag_examples.append({
                        "id": f"{dataset_name}_{i}_{j}",
                        "context": example["context"][:500] + "..." if len(example["context"]) > 500 else example["context"],
                        "question": example["question"],
                        "answer": example["answer"],
                        "source": source,
                        "chunk_id": i,
                        "metadata": chunk_metadata
                    })
            
            # Save results
            results = {
                "rag_examples": rag_examples,
                "total_examples": len(rag_examples),
                "chunks_processed": len(input_chunks),
                "generation_method": "augmentoolkit_rag_fallback"
            }
            
            # Save as JSON
            json_file = output_path / f"{dataset_name}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Save as JSONL for training (RAG format)
            jsonl_file = output_path / f"{dataset_name}.jsonl"
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                for example in rag_examples:
                    # RAG training format
                    f.write(json.dumps({
                        "prompt": f"Context: {example['context']}\\n\\nQuestion: {example['question']}\\n\\nAnswer:",
                        "completion": example["answer"],
                        "metadata": {
                            "source": example["source"],
                            "chunk_id": example["chunk_id"],
                            "type": "rag"
                        }
                    }, ensure_ascii=False) + "\n")
            
            summary = {
                "dataset_name": dataset_name,
                "input_chunks": len(input_chunks),
                "output_dir": str(output_path),
                "rag_examples_generated": len(rag_examples),
                "files_created": [str(json_file), str(jsonl_file)],
                "generation_method": "augmentoolkit_rag_fallback"
            }
            
            logger.info(f"✅ Generated {len(rag_examples)} RAG examples using fallback method")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error in fallback RAG generation: {e}")
            return {"error": str(e)}
    
    def process_chunks_from_directory(
        self,
        chunks_dir: str,
        max_chunks: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load chunks from a Flow4 pipeline output directory.
        
        Args:
            chunks_dir: Directory containing chunk JSON files
            max_chunks: Maximum number of chunks to load
            
        Returns:
            List of chunk dictionaries
        """
        chunks_path = Path(chunks_dir)
        chunk_files = sorted(chunks_path.glob("chunk_*.json"))
        
        if max_chunks:
            chunk_files = chunk_files[:max_chunks]
        
        logger.info(f"Loading {len(chunk_files)} chunks from {chunks_dir}")
        
        chunks = []
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                chunks.append(chunk_data)
            except Exception as e:
                logger.warning(f"Failed to load chunk {chunk_file}: {e}")
        
        return chunks

    def create_mlx_formatted_dataset(
        self, 
        input_chunks: List[Dict[str, Any]], 
        output_dir: str,
        dataset_name: str = "mlx_dataset"
    ) -> Dict[str, Any]:
        """
        Create properly formatted MLX dataset with all formatting issues addressed.
        
        This method integrates all the fixes for MLX dataset generation:
        1. Proper JSONL formatting with validation
        2. MLX batch size requirements (minimum 4 examples per split)
        3. Text cleaning and encoding issues
        4. Automatic train/valid/test splitting
        
        Args:
            input_chunks: List of chunk dictionaries from Flow4 pipeline
            output_dir: Directory to save the MLX-formatted dataset
            dataset_name: Name for the dataset (for logging/metadata)
            
        Returns:
            Dictionary containing generation results and file paths
        """
        logger.info(f"🔧 Creating MLX-formatted dataset from {len(input_chunks)} chunks")
        
        try:
            # Initialize the MLX dataset formatter with our config
            mlx_config = MLXDatasetConfig(
                min_examples_per_split=4,  # MLX requirement
                train_ratio=0.7,
                valid_ratio=0.2,
                test_ratio=0.1,
                max_text_length=2048,
                min_text_length=10
            )
            formatter = MLXDatasetFormatter(mlx_config)
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Convert chunks to MLX format
            logger.info("🔄 Converting chunks to MLX format...")
            success = formatter.create_mlx_dataset_from_chunks(input_chunks, output_path)
            
            if not success:
                raise RuntimeError("Failed to create MLX dataset")
            
            # Validate the created files
            train_file = output_path / "train.jsonl"
            valid_file = output_path / "valid.jsonl" 
            test_file = output_path / "test.jsonl"
            
            # Count examples in each file
            train_count = sum(1 for line in open(train_file, 'r') if line.strip())
            valid_count = sum(1 for line in open(valid_file, 'r') if line.strip())
            test_count = sum(1 for line in open(test_file, 'r') if line.strip())
            
            results = {
                "success": True,
                "dataset_name": dataset_name,
                "output_dir": str(output_path),
                "files": {
                    "train": str(train_file),
                    "valid": str(valid_file),
                    "test": str(test_file)
                },
                "counts": {
                    "train": train_count,
                    "valid": valid_count,
                    "test": test_count,
                    "total": train_count + valid_count + test_count
                },
                "validation": {
                    "train_valid": formatter.validate_jsonl_file(train_file),
                    "valid_valid": formatter.validate_jsonl_file(valid_file),
                    "test_valid": formatter.validate_jsonl_file(test_file)
                }
            }
            
            logger.info(f"✅ MLX dataset created successfully:")
            logger.info(f"   📁 Output: {output_path}")
            logger.info(f"   📊 Examples: Train={train_count}, Valid={valid_count}, Test={test_count}")
            logger.info(f"   ✅ All files validated: {all(results['validation'].values())}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to create MLX dataset: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "dataset_name": dataset_name
            }

    def convert_existing_dataset_to_mlx(
        self,
        input_file: str,
        output_dir: str,
        dataset_name: str = "converted_dataset"
    ) -> Dict[str, Any]:
        """
        Convert existing dataset files to proper MLX format.
        
        Handles common conversion issues:
        - OpenAI chat format to MLX text format
        - Prompt-completion format to MLX text format
        - JSON encoding issues
        - Proper JSONL validation
        
        Args:
            input_file: Path to existing dataset file (JSONL)
            output_dir: Directory to save converted MLX dataset
            dataset_name: Name for the converted dataset
            
        Returns:
            Dictionary containing conversion results
        """
        logger.info(f"🔄 Converting existing dataset: {input_file}")
        
        try:
            input_path = Path(input_file)
            output_path = Path(output_dir)
            
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
            
            # Use the conversion utility
            success = convert_existing_dataset_to_mlx_format(input_path, output_path)
            
            if not success:
                raise RuntimeError("Conversion failed")
            
            # Get file info
            train_file = output_path / "train.jsonl"
            valid_file = output_path / "valid.jsonl"
            test_file = output_path / "test.jsonl"
            
            train_count = sum(1 for line in open(train_file, 'r') if line.strip())
            valid_count = sum(1 for line in open(valid_file, 'r') if line.strip())
            test_count = sum(1 for line in open(test_file, 'r') if line.strip())
            
            results = {
                "success": True,
                "dataset_name": dataset_name,
                "input_file": str(input_path),
                "output_dir": str(output_path),
                "files": {
                    "train": str(train_file),
                    "valid": str(valid_file),
                    "test": str(test_file)
                },
                "counts": {
                    "train": train_count,
                    "valid": valid_count,
                    "test": test_count,
                    "total": train_count + valid_count + test_count
                }
            }
            
            logger.info(f"✅ Dataset converted successfully:")
            logger.info(f"   📁 From: {input_path}")
            logger.info(f"   📁 To: {output_path}")
            logger.info(f"   📊 Examples: Train={train_count}, Valid={valid_count}, Test={test_count}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to convert dataset: {e}")
            return {
                "success": False,
                "error": str(e),
                "input_file": input_file,
                "dataset_name": dataset_name
            }


def create_augmentoolkit_config(
    model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
    mode: str = "mlx",
    dataset_context: str = "technical documentation",
    yaml_path: Optional[str] = None,
    **kwargs
) -> AugmentoolkitConfig:
    """
    Create an Augmentoolkit configuration with sensible defaults.
    
    Args:
        model_name: Model name or path
        mode: Engine mode ("mlx", "api", "local")
        dataset_context: Context for the dataset generation
        yaml_path: Path to YAML configuration file (optional)
        **kwargs: Additional configuration options
        
    Returns:
        Configured AugmentoolkitConfig instance
    """
    # Load from YAML if path is provided, otherwise create with defaults
    if yaml_path:
        config = load_config_from_yaml(yaml_path)
        # Override with provided parameters
        config.model_name = model_name
        config.mode = mode
        config.dataset_context = dataset_context
    else:
        config = AugmentoolkitConfig(
            model_name=model_name,
            mode=mode,
            dataset_context=dataset_context
        )
    
    # Update with any additional kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


async def generate_advanced_dataset(
    input_path: str,
    output_path: str,
    dataset_type: str = "factual",
    model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
    mode: str = "mlx",
    max_chunks: Optional[int] = None,
    yaml_path: Optional[str] = None,
    **config_kwargs
) -> Dict[str, Any]:
    """
    High-level function to generate advanced datasets using Augmentoolkit.
    
    Args:
        input_path: Path to input chunks directory
        output_path: Path to save the generated dataset
        dataset_type: Type of dataset ("factual", "rag", "multi_source", "repvar", "complete")
        model_name: Model name or path
        mode: Engine mode ("mlx", "api", "local")
        max_chunks: Maximum chunks to process
        yaml_path: Path to YAML configuration file (optional)
        **config_kwargs: Additional configuration options
        
    Returns:
        Generation results dictionary
    """
    # Check if full Augmentoolkit is available
    if not HAS_AUGMENTOOLKIT:
        raise ImportError(
            "Full Augmentoolkit functionality is required but not available. "
            f"Available components: Core={HAS_AUGMENTOOLKIT_CORE}, Pipelines={HAS_AUGMENTOOLKIT_PIPELINES}, Advanced={HAS_AUGMENTOOLKIT_ADVANCED}. "
            "Please ensure all Augmentoolkit dependencies are properly installed."
        )
    
    try:
        # Create configuration
        config = create_augmentoolkit_config(
            model_name=model_name,
            mode=mode,
            yaml_path=yaml_path,
            **config_kwargs
        )
        
        # Create generator
        generator = AugmentoolkitGenerator(config)
        
        # Validate environment before proceeding
        is_valid, issues = generator.validate_environment()
        if not is_valid:
            error_msg = f"Environment validation failed: {'; '.join(issues)}"
            logger.error(f"❌ {error_msg}")
            return {"error": error_msg, "validation_issues": issues}
        
        # Load input chunks
        chunks = generator.process_chunks_from_directory(input_path, max_chunks)
        
        if not chunks:
            error_msg = f"No chunks found in {input_path}"
            logger.error(f"❌ {error_msg}")
            return {"error": error_msg}
        
        results = {}
        
        # Generate datasets based on type and availability
        if dataset_type in ["factual", "complete"]:
            logger.info("Generating factual dataset...")
            factual_results = await generator.generate_factual_dataset(
                chunks, output_path, "factual_dataset"
            )
            results["factual"] = factual_results
        
        if dataset_type in ["rag", "complete"]:
            logger.info("Generating RAG dataset...")
            rag_results = await generator.generate_rag_dataset(
                chunks, output_path, "rag_dataset"
            )
            results["rag"] = rag_results
        
        # Advanced features require additional components
        if HAS_AUGMENTOOLKIT_ADVANCED:
            if dataset_type in ["multi_source", "complete"]:
                logger.info("Generating multi-source reasoning dataset...")
                multi_source_results = await generator.generate_multi_source_dataset(
                    chunks, output_path, "multi_source_dataset"
                )
                results["multi_source"] = multi_source_results
            
            if dataset_type in ["repvar", "complete"]:
                logger.info("Generating representation variation dataset...")
                repvar_results = await generator.generate_representation_variation_dataset(
                    chunks, output_path, "representation_variation_dataset"
                )
                results["representation_variation"] = repvar_results
            
            if dataset_type == "complete":
                logger.info("Generating complete factual dataset suite...")
                complete_results = await generator.generate_complete_factual_dataset(
                    chunks, output_path, "complete_factual_dataset"
                )
                results["complete_factual"] = complete_results
        else:
            # Warn about missing advanced features
            if dataset_type in ["multi_source", "repvar"]:
                raise ImportError(f"Advanced dataset type '{dataset_type}' requires full Augmentoolkit advanced components which are not available.")
        
        # Create MLX-formatted dataset if generation was successful
        if results and not any("error" in result for result in results.values()):
            logger.info("🔄 Converting to MLX format for fine-tuning...")
            try:
                mlx_success = await _create_mlx_dataset_from_results(results, output_path)
                if mlx_success:
                    logger.info("✅ MLX dataset created at ./mlx_train.jsonl")
                else:
                    logger.warning("⚠️ MLX dataset conversion failed, but main generation succeeded")
            except Exception as mlx_error:
                logger.warning(f"⚠️ MLX conversion failed: {mlx_error}, but main generation succeeded")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in advanced dataset generation: {e}")
        raise


async def _create_mlx_dataset_from_results(results: Dict[str, Any], output_path: str) -> bool:
    """
    Create MLX-formatted dataset from Augmentoolkit generation results.
    
    This function converts the conversation-format datasets to MLX text format
    and creates a train.jsonl file in the current directory as ./mlx_train.jsonl.
    
    Args:
        results: Dictionary of generation results from Augmentoolkit
        output_path: Output path where datasets were generated
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from pathlib import Path
        import json
        
        # Initialize MLX formatter
        formatter = MLXDatasetFormatter()
        
        # Collect all conversation data from results
        all_conversations = []
        
        for dataset_name, result in results.items():
            if "error" in result:
                continue
                
            # Look for conversation files in the result output directory
            result_dir = Path(result.get("output_dir", output_path))
            
            # Common Augmentoolkit output files to check
            conversation_files = [
                "plain_qa_list.jsonl",
                "conversations.jsonl", 
                "factual_conversations.jsonl",
                "rag_conversations.jsonl"
            ]
            
            for filename in conversation_files:
                filepath = result_dir / filename
                if filepath.exists():
                    logger.info(f"📁 Processing {filepath} for MLX conversion...")
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            for line_num, line in enumerate(f, 1):
                                line = line.strip()
                                if not line:
                                    continue
                                    
                                try:
                                    entry = json.loads(line)
                                    
                                    # Handle different conversation formats
                                    if "conversations" in entry:
                                        conversations = entry["conversations"]
                                        if conversations:
                                            all_conversations.append(conversations)
                                    elif "messages" in entry:
                                        messages = entry["messages"]
                                        if messages:
                                            all_conversations.append(messages)
                                    else:
                                        logger.debug(f"Unknown format in {filepath}:{line_num}, skipping")
                                        
                                except json.JSONDecodeError as e:
                                    logger.warning(f"JSON error in {filepath}:{line_num}: {e}")
                                    continue
                                    
                    except Exception as e:
                        logger.warning(f"Error reading {filepath}: {e}")
                        continue
        
        if not all_conversations:
            logger.error("No conversation data found in Augmentoolkit results")
            return False
        
        logger.info(f"🔄 Converting {len(all_conversations)} conversations to MLX format...")
        
        # Convert conversations to MLX format
        mlx_examples = []
        for conversation in all_conversations:
            try:
                # Convert each conversation to MLX text format
                text_parts = []
                
                for message in conversation:
                    role = message.get("from", message.get("role", ""))
                    content = message.get("value", message.get("content", ""))
                    
                    if role == "system":
                        text_parts.append(f"System: {content}")
                    elif role in ["human", "user"]:
                        text_parts.append(f"Human: {content}")
                    elif role in ["gpt", "assistant"]:
                        text_parts.append(f"Assistant: {content}")
                
                if text_parts:
                    combined_text = "\n\n".join(text_parts)
                    clean_text = formatter.clean_text_for_jsonl(combined_text)
                    mlx_examples.append({"text": clean_text})
                    
            except Exception as e:
                logger.warning(f"Error converting conversation: {e}")
                continue
        
        if not mlx_examples:
            logger.error("No valid MLX examples generated from conversations")
            return False
        
        logger.info(f"✅ Generated {len(mlx_examples)} MLX training examples")
        
        # Create the MLX dataset file in current directory
        mlx_output_path = Path("./mlx_train.jsonl")
        
        # Ensure we have enough examples for MLX training
        min_examples_needed = 12  # MLX needs at least 4 per split for train/valid/test
        if len(mlx_examples) < min_examples_needed:
            logger.warning(f"Only {len(mlx_examples)} examples available, need {min_examples_needed} for proper MLX training")
            logger.info("Duplicating examples to meet MLX minimum requirements...")
            
            # Duplicate examples to reach minimum
            original_count = len(mlx_examples)
            while len(mlx_examples) < min_examples_needed:
                # Add copies of existing examples
                examples_to_add = min(original_count, min_examples_needed - len(mlx_examples))
                mlx_examples.extend(mlx_examples[:examples_to_add])
            
            logger.info(f"Extended dataset from {original_count} to {len(mlx_examples)} examples")
        
        # Write MLX dataset directly (no train/valid/test split needed for user's use case)
        success = formatter.write_jsonl_file(mlx_examples, mlx_output_path)
        
        if success:
            logger.info(f"🎯 MLX dataset created: {mlx_output_path.absolute()}")
            logger.info(f"📊 Contains {len(mlx_examples)} training examples")
            return True
        else:
            logger.error("Failed to write MLX dataset file")
            return False
            
    except Exception as e:
        logger.error(f"Error creating MLX dataset: {e}")
        return False