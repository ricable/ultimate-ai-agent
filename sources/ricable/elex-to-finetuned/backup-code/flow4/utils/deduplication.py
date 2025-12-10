"""
Dataset Deduplication Utilities for Flow4

Provides comprehensive deduplication capabilities for RAG and fine-tuning datasets,
including exact matches, semantic similarity, and content-based deduplication.
"""

import hashlib
import json
import re
from typing import List, Dict, Any, Tuple, Set, Optional
from collections import defaultdict
from pathlib import Path
import logging
import warnings

# Suppress NumPy compatibility warnings
warnings.filterwarnings("ignore", message=".*NumPy.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*_ARRAY_API.*", category=UserWarning)

HAS_SENTENCE_TRANSFORMERS = False
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sentence_transformers import SentenceTransformer
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
    HAS_SENTENCE_TRANSFORMERS = True
except (ImportError, RuntimeError, Exception):
    HAS_SENTENCE_TRANSFORMERS = False

from .logging import get_logger

logger = get_logger(__name__)


class DatasetDeduplicator:
    """
    Comprehensive dataset deduplication with multiple strategies.
    
    Strategies:
    1. Exact match deduplication (exact string matching)
    2. Normalized deduplication (case-insensitive, whitespace normalized)
    3. Content hash deduplication (semantic content matching)
    4. Semantic similarity deduplication (embedding-based)
    """
    
    def __init__(self, similarity_threshold: float = 0.95):
        """
        Initialize deduplicator.
        
        Args:
            similarity_threshold: Threshold for semantic similarity (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold
        self.embedding_model = None
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                # Use a lightweight model for efficiency
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Semantic similarity deduplication enabled")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self.embedding_model = None
        
        if not HAS_SENTENCE_TRANSFORMERS or self.embedding_model is None:
            logger.info("📝 Using text-based deduplication only (install sentence-transformers for semantic deduplication)")
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase
        text = text.lower().strip()
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common punctuation variations
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def content_hash(self, text: str) -> str:
        """Generate content hash for text."""
        normalized = self.normalize_text(text)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def deduplicate_exact(self, dataset: List[Dict[str, Any]], key: str = 'question') -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Remove exact duplicates."""
        seen = set()
        unique_items = []
        duplicate_counts = defaultdict(int)
        
        for item in dataset:
            text = item.get(key, '')
            if text not in seen:
                seen.add(text)
                unique_items.append(item)
            else:
                duplicate_counts[text] += 1
        
        logger.info(f"Exact deduplication: {len(dataset)} → {len(unique_items)} items ({len(dataset) - len(unique_items)} duplicates)")
        return unique_items, dict(duplicate_counts)
    
    def deduplicate_normalized(self, dataset: List[Dict[str, Any]], key: str = 'question') -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Remove normalized duplicates (case-insensitive, whitespace normalized)."""
        seen_normalized = {}
        unique_items = []
        duplicate_counts = defaultdict(int)
        
        for item in dataset:
            text = item.get(key, '')
            normalized = self.normalize_text(text)
            
            if normalized not in seen_normalized:
                seen_normalized[normalized] = text
                unique_items.append(item)
            else:
                duplicate_counts[seen_normalized[normalized]] += 1
        
        logger.info(f"Normalized deduplication: {len(dataset)} → {len(unique_items)} items ({len(dataset) - len(unique_items)} duplicates)")
        return unique_items, dict(duplicate_counts)
    
    def deduplicate_content_hash(self, dataset: List[Dict[str, Any]], key: str = 'question') -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Remove duplicates based on content hash."""
        seen_hashes = {}
        unique_items = []
        duplicate_counts = defaultdict(int)
        
        for item in dataset:
            text = item.get(key, '')
            content_hash = self.content_hash(text)
            
            if content_hash not in seen_hashes:
                seen_hashes[content_hash] = text
                unique_items.append(item)
            else:
                duplicate_counts[seen_hashes[content_hash]] += 1
        
        logger.info(f"Content hash deduplication: {len(dataset)} → {len(unique_items)} items ({len(dataset) - len(unique_items)} duplicates)")
        return unique_items, dict(duplicate_counts)
    
    def deduplicate_semantic(self, dataset: List[Dict[str, Any]], key: str = 'question') -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Remove semantically similar duplicates using embeddings."""
        if not self.embedding_model:
            logger.warning("Semantic deduplication not available - falling back to content hash")
            return self.deduplicate_content_hash(dataset, key)
        
        if len(dataset) == 0:
            return dataset, {}
        
        # Extract texts
        texts = [item.get(key, '') for item in dataset]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} items...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Find duplicates using cosine similarity
        unique_indices = []
        duplicate_counts = defaultdict(int)
        seen_texts = []
        
        for i, embedding in enumerate(embeddings):
            is_duplicate = False
            
            if unique_indices:
                # Compare with existing unique items
                unique_embeddings = embeddings[unique_indices]
                similarities = cosine_similarity([embedding], unique_embeddings)[0]
                
                max_similarity = np.max(similarities)
                if max_similarity >= self.similarity_threshold:
                    # Found a duplicate
                    most_similar_idx = unique_indices[np.argmax(similarities)]
                    duplicate_counts[texts[most_similar_idx]] += 1
                    is_duplicate = True
            
            if not is_duplicate:
                unique_indices.append(i)
                seen_texts.append(texts[i])
        
        unique_items = [dataset[i] for i in unique_indices]
        
        logger.info(f"Semantic deduplication: {len(dataset)} → {len(unique_items)} items ({len(dataset) - len(unique_items)} duplicates)")
        return unique_items, dict(duplicate_counts)
    
    def deduplicate_comprehensive(
        self, 
        dataset: List[Dict[str, Any]], 
        keys: List[str] = None,
        strategy: str = "semantic"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Comprehensive deduplication with detailed reporting.
        
        Args:
            dataset: List of dataset items
            keys: Keys to check for duplication (default: ['question', 'text', 'prompt'])
            strategy: Deduplication strategy ('exact', 'normalized', 'hash', 'semantic', 'progressive')
        
        Returns:
            Tuple of (deduplicated_dataset, deduplication_report)
        """
        if not dataset:
            return dataset, {"original_size": 0, "final_size": 0, "removed": 0}
        
        if keys is None:
            keys = ['question', 'text', 'prompt', 'content']
        
        # Find the primary key that exists in the dataset
        primary_key = None
        for key in keys:
            if any(key in item for item in dataset):
                primary_key = key
                break
        
        if not primary_key:
            logger.warning("No suitable key found for deduplication")
            return dataset, {"error": "No suitable key found"}
        
        logger.info(f"Using '{primary_key}' key for deduplication")
        
        original_size = len(dataset)
        current_dataset = dataset.copy()
        report = {
            "original_size": original_size,
            "strategy": strategy,
            "primary_key": primary_key,
            "steps": []
        }
        
        if strategy == "progressive":
            # Apply multiple strategies progressively
            strategies = [
                ("exact", self.deduplicate_exact),
                ("normalized", self.deduplicate_normalized),
                ("semantic", self.deduplicate_semantic)
            ]
        else:
            # Apply single strategy
            strategy_map = {
                "exact": self.deduplicate_exact,
                "normalized": self.deduplicate_normalized,
                "hash": self.deduplicate_content_hash,
                "semantic": self.deduplicate_semantic
            }
            strategies = [(strategy, strategy_map.get(strategy, self.deduplicate_semantic))]
        
        for step_name, dedup_func in strategies:
            step_start_size = len(current_dataset)
            current_dataset, duplicates = dedup_func(current_dataset, primary_key)
            step_end_size = len(current_dataset)
            
            step_report = {
                "step": step_name,
                "input_size": step_start_size,
                "output_size": step_end_size,
                "removed": step_start_size - step_end_size,
                "duplicate_examples": list(duplicates.keys())[:5] if duplicates else []
            }
            report["steps"].append(step_report)
        
        report["final_size"] = len(current_dataset)
        report["total_removed"] = original_size - len(current_dataset)
        report["deduplication_rate"] = (report["total_removed"] / original_size * 100) if original_size > 0 else 0
        
        logger.info(f"✅ Deduplication complete: {original_size} → {len(current_dataset)} items")
        logger.info(f"📉 Removed {report['total_removed']} duplicates ({report['deduplication_rate']:.1f}% reduction)")
        
        return current_dataset, report


def deduplicate_instruction_dataset(
    input_path: str, 
    output_path: str = None, 
    strategy: str = "progressive"
) -> Dict[str, Any]:
    """
    Deduplicate an instruction dataset file.
    
    Args:
        input_path: Path to input JSON/JSONL file
        output_path: Path to save deduplicated dataset (optional)
        strategy: Deduplication strategy
    
    Returns:
        Deduplication report
    """
    input_path = Path(input_path)
    
    # Read dataset
    if input_path.suffix == '.jsonl':
        with open(input_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f if line.strip()]
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    
    # Deduplicate
    deduplicator = DatasetDeduplicator()
    deduplicated_dataset, report = deduplicator.deduplicate_comprehensive(dataset, strategy=strategy)
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        
        if output_path.suffix == '.jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in deduplicated_dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(deduplicated_dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Deduplicated dataset saved to: {output_path}")
    
    # Save report
    if output_path:
        report_path = output_path.parent / f"{output_path.stem}_deduplication_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"📊 Deduplication report saved to: {report_path}")
    
    return report


def deduplicate_rag_datasets(
    input_dir: str,
    output_dir: str = None,
    strategy: str = "progressive"
) -> Dict[str, Any]:
    """
    Deduplicate all datasets in a RAG directory.
    
    Args:
        input_dir: Directory containing RAG datasets
        output_dir: Output directory (optional)
        strategy: Deduplication strategy
    
    Returns:
        Combined deduplication report
    """
    input_dir = Path(input_dir)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    reports = {}
    dataset_files = list(input_dir.glob("*.json")) + list(input_dir.glob("*.jsonl"))
    
    for dataset_file in dataset_files:
        logger.info(f"📁 Processing {dataset_file.name}...")
        
        if output_dir:
            output_file = output_dir / dataset_file.name
        else:
            output_file = dataset_file.parent / f"deduplicated_{dataset_file.name}"
        
        try:
            report = deduplicate_instruction_dataset(
                str(dataset_file), 
                str(output_file), 
                strategy
            )
            reports[dataset_file.name] = report
        except Exception as e:
            logger.error(f"❌ Failed to deduplicate {dataset_file.name}: {e}")
            reports[dataset_file.name] = {"error": str(e)}
    
    return reports