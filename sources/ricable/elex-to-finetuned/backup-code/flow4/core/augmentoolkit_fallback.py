"""
Fallback implementation for Augmentoolkit when full installation is not available.

This provides basic dataset generation capabilities using Flow4's existing infrastructure
when the full Augmentoolkit suite is not properly installed.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


class FallbackAugmentoolkitGenerator:
    """Fallback generator when full Augmentoolkit is not available."""
    
    def __init__(self, config=None):
        """Initialize the fallback generator."""
        self.config = config
        logger.info("Using fallback Augmentoolkit implementation")
    
    def validate_environment(self) -> tuple[bool, List[str]]:
        """Basic environment validation for fallback mode."""
        issues = []
        
        # Check basic requirements
        try:
            import json
            import asyncio
        except ImportError as e:
            issues.append(f"Missing basic Python modules: {e}")
        
        return len(issues) == 0, issues
    
    async def generate_factual_dataset(
        self,
        input_chunks: List[Dict[str, Any]],
        output_dir: str,
        dataset_name: str = "factual_dataset"
    ) -> Dict[str, Any]:
        """Generate a basic factual dataset using Flow4's infrastructure."""
        logger.info(f"Generating basic factual dataset from {len(input_chunks)} chunks")
        
        try:
            # Prepare output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create basic Q&A pairs from chunks
            qa_pairs = []
            for i, chunk in enumerate(input_chunks):
                text = chunk.get("text", "")
                source = chunk.get("metadata", {}).get("source", "unknown")
                
                # Create simple question-answer pairs
                questions = [
                    f"What is the main topic discussed in this section from {source}?",
                    f"Summarize the key points from this {source} section.",
                    f"What are the important details mentioned in this part of {source}?"
                ]
                
                for j, question in enumerate(questions):
                    qa_pairs.append({
                        "id": f"{i}_{j}",
                        "question": question,
                        "answer": text[:500] + "..." if len(text) > 500 else text,
                        "source": source,
                        "chunk_id": i
                    })
            
            # Save as JSON
            output_file = output_path / f"{dataset_name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "dataset_name": dataset_name,
                    "qa_pairs": qa_pairs,
                    "total_pairs": len(qa_pairs),
                    "generation_method": "flow4_fallback"
                }, f, indent=2, ensure_ascii=False)
            
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
            
            logger.info(f"✅ Basic factual dataset generated: {output_file}")
            
            return {
                "dataset_name": dataset_name,
                "input_chunks": len(input_chunks),
                "output_dir": str(output_path),
                "qa_pairs_generated": len(qa_pairs),
                "files_created": [str(output_file), str(jsonl_file)]
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating basic factual dataset: {e}")
            return {"error": str(e)}
    
    async def generate_rag_dataset(
        self,
        input_chunks: List[Dict[str, Any]],
        output_dir: str,
        dataset_name: str = "rag_dataset"
    ) -> Dict[str, Any]:
        """Generate a basic RAG dataset."""
        logger.info(f"Generating basic RAG dataset from {len(input_chunks)} chunks")
        
        try:
            # Prepare output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create RAG training examples
            rag_examples = []
            for i, chunk in enumerate(input_chunks):
                text = chunk.get("text", "")
                source = chunk.get("metadata", {}).get("source", "unknown")
                
                # Create context-based examples
                rag_examples.append({
                    "id": i,
                    "context": text,
                    "question": f"Based on the provided context from {source}, what are the main points?",
                    "answer": f"Based on the context, the main points are: {text[:200]}...",
                    "source": source
                })
            
            # Save as JSON
            output_file = output_path / f"{dataset_name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "dataset_name": dataset_name,
                    "rag_examples": rag_examples,
                    "total_examples": len(rag_examples),
                    "generation_method": "flow4_fallback"
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Basic RAG dataset generated: {output_file}")
            
            return {
                "dataset_name": dataset_name,
                "input_chunks": len(input_chunks),
                "output_dir": str(output_path),
                "rag_examples_generated": len(rag_examples),
                "files_created": [str(output_file)]
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating basic RAG dataset: {e}")
            return {"error": str(e)}


async def generate_fallback_dataset(
    input_path: str,
    output_path: str,
    dataset_type: str = "factual",
    **kwargs
) -> Dict[str, Any]:
    """Generate datasets using fallback implementation."""
    
    logger.info(f"Using fallback dataset generation for type: {dataset_type}")
    
    # Load chunks from input directory
    input_dir = Path(input_path)
    chunks = []
    
    if input_dir.exists():
        for chunk_file in input_dir.glob("chunk_*.json"):
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                chunks.append(chunk_data)
            except Exception as e:
                logger.warning(f"Failed to load chunk {chunk_file}: {e}")
    
    if not chunks:
        return {"error": f"No chunks found in {input_path}"}
    
    # Create fallback generator
    generator = FallbackAugmentoolkitGenerator()
    
    # Validate environment
    is_valid, issues = generator.validate_environment()
    if not is_valid:
        return {"error": f"Environment validation failed: {'; '.join(issues)}"}
    
    # Generate based on type
    results = {}
    
    if dataset_type in ["factual", "complete"]:
        factual_result = await generator.generate_factual_dataset(
            chunks, output_path, "factual_dataset"
        )
        results["factual"] = factual_result
    
    if dataset_type in ["rag", "complete"]:
        rag_result = await generator.generate_rag_dataset(
            chunks, output_path, "rag_dataset"
        )
        results["rag"] = rag_result
    
    if dataset_type in ["multi_source", "repvar"]:
        logger.warning(f"Dataset type '{dataset_type}' not available in fallback mode. Using basic factual generation.")
        factual_result = await generator.generate_factual_dataset(
            chunks, output_path, f"{dataset_type}_fallback_dataset"
        )
        results[dataset_type] = factual_result
    
    return results