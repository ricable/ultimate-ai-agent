"""
Resumable Augmentoolkit Generator for Flow4

Provides resumable dataset generation with progress checkpoints, timeout handling,
and robust error recovery for large-scale processing.
"""

import os
import json
import asyncio
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib
import time

from ..utils.logging import get_logger
from .augmentoolkit_generator import AugmentoolkitGenerator, AugmentoolkitConfig

logger = get_logger(__name__)


@dataclass
class GenerationProgress:
    """Progress tracking for resumable generation."""
    
    # Basic info
    session_id: str
    start_time: float
    last_update: float
    
    # Input configuration
    total_chunks: int
    input_hash: str
    config_hash: str
    
    # Progress tracking
    chunks_processed: int = 0
    chunks_failed: int = 0
    current_stage: str = "initialization"
    
    # Results tracking
    qa_pairs_generated: int = 0
    conversations_generated: int = 0
    output_files: List[str] = None
    
    # Error tracking
    errors: List[Dict[str, Any]] = None
    retry_attempts: int = 0
    
    def __post_init__(self):
        if self.output_files is None:
            self.output_files = []
        if self.errors is None:
            self.errors = []
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_chunks == 0:
            return 0.0
        return (self.chunks_processed / self.total_chunks) * 100
    
    @property
    def is_complete(self) -> bool:
        """Check if generation is complete."""
        return self.chunks_processed >= self.total_chunks
    
    def add_error(self, error: str, chunk_id: Optional[int] = None, stage: Optional[str] = None):
        """Add an error to the tracking."""
        self.errors.append({
            "timestamp": time.time(),
            "error": error,
            "chunk_id": chunk_id,
            "stage": stage or self.current_stage,
            "retry_attempt": self.retry_attempts
        })
    
    def update_stage(self, stage: str):
        """Update current processing stage."""
        self.current_stage = stage
        self.last_update = time.time()


class ResumableAugmentoolkitGenerator:
    """
    Resumable dataset generator that can save progress and recover from failures.
    
    Features:
    - Progress checkpointing with automatic saves
    - Timeout handling with configurable retries
    - Error recovery and skip-on-failure modes
    - Memory management for large datasets
    - Detailed progress reporting
    """
    
    def __init__(self, config: AugmentoolkitConfig, checkpoint_dir: Optional[str] = None):
        """
        Initialize the resumable generator.
        
        Args:
            config: Augmentoolkit configuration
            checkpoint_dir: Directory to save progress checkpoints
        """
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir or "./augmentoolkit_checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the base generator
        self.generator = AugmentoolkitGenerator(config)
        
        # Progress tracking
        self.progress: Optional[GenerationProgress] = None
        self.session_id: Optional[str] = None
        
        # Configuration
        self.save_interval = 10  # Save progress every N processed chunks
        self.max_errors_per_chunk = 3
        self.skip_failed_chunks = True
        self.memory_optimization = True
    
    def _generate_session_id(self, input_chunks: List[Dict[str, Any]]) -> str:
        """Generate a unique session ID based on inputs and config."""
        # Create a hash of the input chunks and configuration
        input_text = json.dumps([chunk.get("text", "")[:100] for chunk in input_chunks], sort_keys=True)
        config_text = json.dumps(asdict(self.config), sort_keys=True)
        
        combined = f"{input_text}:{config_text}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]
    
    def _get_checkpoint_path(self, session_id: str) -> Path:
        """Get the checkpoint file path for a session."""
        return self.checkpoint_dir / f"progress_{session_id}.json"
    
    def _save_progress(self):
        """Save current progress to checkpoint file."""
        if not self.progress:
            return
        
        checkpoint_path = self._get_checkpoint_path(self.progress.session_id)
        
        try:
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.progress), f, indent=2, ensure_ascii=False)
            
            logger.debug(f"💾 Progress saved to {checkpoint_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save progress: {e}")
    
    def _load_progress(self, session_id: str) -> Optional[GenerationProgress]:
        """Load progress from checkpoint file."""
        checkpoint_path = self._get_checkpoint_path(session_id)
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            progress = GenerationProgress(**data)
            logger.info(f"📂 Loaded progress from {checkpoint_path}")
            logger.info(f"   🔄 Progress: {progress.progress_percent:.1f}% ({progress.chunks_processed}/{progress.total_chunks})")
            logger.info(f"   📊 Generated: {progress.qa_pairs_generated} QA pairs, {progress.conversations_generated} conversations")
            logger.info(f"   ⚠️ Errors: {len(progress.errors)} errors across {progress.retry_attempts} retry attempts")
            
            return progress
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load progress: {e}")
            return None
    
    def _clean_old_checkpoints(self, keep_sessions: int = 5):
        """Clean up old checkpoint files, keeping only the most recent."""
        try:
            checkpoints = list(self.checkpoint_dir.glob("progress_*.json"))
            if len(checkpoints) <= keep_sessions:
                return
            
            # Sort by modification time and remove oldest
            checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            for old_checkpoint in checkpoints[keep_sessions:]:
                old_checkpoint.unlink()
                logger.debug(f"🗑️ Cleaned up old checkpoint: {old_checkpoint.name}")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to clean checkpoints: {e}")
    
    async def generate_factual_dataset_resumable(
        self,
        input_chunks: List[Dict[str, Any]],
        output_dir: str,
        dataset_name: str = "factual_dataset",
        resume: bool = True,
        timeout_per_chunk: int = 180,
        max_concurrent: int = 3
    ) -> Dict[str, Any]:
        """
        Generate factual dataset with resumable progress and robust error handling.
        
        Args:
            input_chunks: List of chunk dictionaries
            output_dir: Output directory
            dataset_name: Dataset name
            resume: Whether to resume from previous progress
            timeout_per_chunk: Timeout per chunk in seconds
            max_concurrent: Maximum concurrent processing
            
        Returns:
            Generation results with progress information
        """
        logger.info(f"🚀 Starting resumable factual dataset generation")
        logger.info(f"📊 Input: {len(input_chunks)} chunks")
        logger.info(f"📁 Output: {output_dir}")
        logger.info(f"⚙️ Timeout per chunk: {timeout_per_chunk}s, Max concurrent: {max_concurrent}")
        
        try:
            # Generate session ID and check for existing progress
            self.session_id = self._generate_session_id(input_chunks)
            
            if resume:
                self.progress = self._load_progress(self.session_id)
            
            # Initialize new progress if none exists
            if not self.progress:
                logger.info("🔄 Starting new generation session")
                
                # Generate hashes for validation
                input_hash = hashlib.md5(json.dumps([chunk.get("text", "")[:100] for chunk in input_chunks]).encode()).hexdigest()
                config_hash = hashlib.md5(json.dumps(asdict(self.config)).encode()).hexdigest()
                
                self.progress = GenerationProgress(
                    session_id=self.session_id,
                    start_time=time.time(),
                    last_update=time.time(),
                    total_chunks=len(input_chunks),
                    input_hash=input_hash,
                    config_hash=config_hash
                )
            else:
                logger.info(f"📂 Resuming from {self.progress.progress_percent:.1f}% completion")
            
            # Prepare output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Process chunks with concurrency control
            self.progress.update_stage("processing_chunks")
            
            # Use semaphore to control concurrency
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_chunk_with_semaphore(chunk_id: int, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                """Process a single chunk with concurrency control."""
                async with semaphore:
                    return await self._process_single_chunk(chunk_id, chunk, timeout_per_chunk)
            
            # Create tasks for unprocessed chunks
            chunks_to_process = input_chunks[self.progress.chunks_processed:]
            start_index = self.progress.chunks_processed
            
            tasks = [
                process_chunk_with_semaphore(start_index + i, chunk)
                for i, chunk in enumerate(chunks_to_process)
            ]
            
            # Process chunks with progress tracking
            results = []
            completed_count = 0
            
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    if result:
                        results.append(result)
                        self.progress.qa_pairs_generated += result.get("qa_pairs", 0)
                        self.progress.conversations_generated += result.get("conversations", 0)
                    
                    completed_count += 1
                    self.progress.chunks_processed = start_index + completed_count
                    self.progress.last_update = time.time()
                    
                    # Save progress periodically
                    if completed_count % self.save_interval == 0:
                        self._save_progress()
                        logger.info(f"🔄 Progress: {self.progress.progress_percent:.1f}% ({self.progress.chunks_processed}/{self.progress.total_chunks})")
                    
                except Exception as e:
                    logger.error(f"❌ Task failed: {e}")
                    self.progress.chunks_failed += 1
                    self.progress.add_error(str(e))
            
            # Final progress save
            self.progress.update_stage("finalizing")
            self._save_progress()
            
            # Generate final outputs
            await self._finalize_dataset(results, output_path, dataset_name)
            
            # Mark as complete
            self.progress.update_stage("completed")
            self._save_progress()
            
            # Clean up old checkpoints
            self._clean_old_checkpoints()
            
            # Prepare final results
            final_results = {
                "success": True,
                "dataset_name": dataset_name,
                "session_id": self.session_id,
                "output_dir": str(output_path),
                "progress": asdict(self.progress),
                "generation_time": time.time() - self.progress.start_time,
                "chunks_processed": self.progress.chunks_processed,
                "chunks_failed": self.progress.chunks_failed,
                "qa_pairs_generated": self.progress.qa_pairs_generated,
                "conversations_generated": self.progress.conversations_generated,
                "error_count": len(self.progress.errors)
            }
            
            logger.info(f"🎉 Dataset generation completed successfully!")
            logger.info(f"   📊 Processed: {self.progress.chunks_processed}/{self.progress.total_chunks} chunks")
            logger.info(f"   💬 Generated: {self.progress.qa_pairs_generated} QA pairs, {self.progress.conversations_generated} conversations")
            logger.info(f"   ⏱️ Time: {final_results['generation_time']:.1f} seconds")
            logger.info(f"   ⚠️ Errors: {final_results['error_count']} errors")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Fatal error in resumable generation: {e}")
            traceback.print_exc()
            
            if self.progress:
                self.progress.add_error(f"Fatal error: {str(e)}")
                self._save_progress()
            
            return {
                "success": False,
                "error": str(e),
                "session_id": self.session_id,
                "progress": asdict(self.progress) if self.progress else None
            }
    
    async def _process_single_chunk(
        self, 
        chunk_id: int, 
        chunk: Dict[str, Any], 
        timeout: int
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single chunk with timeout and retry handling.
        
        Args:
            chunk_id: Index of the chunk
            chunk: Chunk data
            timeout: Timeout in seconds
            
        Returns:
            Processing results or None if failed
        """
        chunk_text = chunk.get("text", "")
        chunk_preview = chunk_text[:100].replace('\n', ' ') + "..." if len(chunk_text) > 100 else chunk_text
        
        logger.debug(f"🔄 Processing chunk {chunk_id}: {chunk_preview}")
        
        for attempt in range(self.max_errors_per_chunk):
            try:
                # Simulate chunk processing (replace with actual Augmentoolkit call)
                async def process_chunk():
                    # Here you would call the actual Augmentoolkit processing
                    # For now, we'll create a mock result
                    await asyncio.sleep(0.1)  # Simulate processing time
                    
                    # Generate mock QA pairs based on chunk content
                    qa_pairs = []
                    if len(chunk_text.strip()) > 50:  # Only process substantial content
                        qa_pairs = [
                            {
                                "question": f"What is discussed in this section about {chunk_text[:30]}...?",
                                "answer": f"This section discusses: {chunk_text[:200]}...",
                                "source": chunk.get("metadata", {}).get("source", "unknown"),
                                "chunk_id": chunk_id
                            }
                        ]
                    
                    return {
                        "chunk_id": chunk_id,
                        "qa_pairs": len(qa_pairs),
                        "conversations": len(qa_pairs) // 2,  # Assume 2 QA pairs per conversation
                        "data": qa_pairs
                    }
                
                # Apply timeout
                result = await asyncio.wait_for(process_chunk(), timeout=timeout)
                
                logger.debug(f"✅ Chunk {chunk_id} completed: {result.get('qa_pairs', 0)} QA pairs")
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Chunk {chunk_id} timed out on attempt {attempt + 1}/{self.max_errors_per_chunk}")
                self.progress.add_error(f"Timeout on attempt {attempt + 1}", chunk_id)
                
                if attempt < self.max_errors_per_chunk - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    if self.skip_failed_chunks:
                        logger.warning(f"⚠️ Skipping chunk {chunk_id} after {self.max_errors_per_chunk} timeout attempts")
                        return None
                    else:
                        raise
                        
            except Exception as e:
                logger.warning(f"❌ Chunk {chunk_id} failed on attempt {attempt + 1}/{self.max_errors_per_chunk}: {e}")
                self.progress.add_error(str(e), chunk_id)
                
                if attempt < self.max_errors_per_chunk - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    if self.skip_failed_chunks:
                        logger.warning(f"⚠️ Skipping chunk {chunk_id} after {self.max_errors_per_chunk} failed attempts")
                        return None
                    else:
                        raise
        
        return None
    
    async def _finalize_dataset(
        self, 
        results: List[Dict[str, Any]], 
        output_path: Path, 
        dataset_name: str
    ):
        """
        Finalize the dataset by combining results and saving output files.
        
        Args:
            results: List of processing results
            output_path: Output directory path
            dataset_name: Name of the dataset
        """
        logger.info(f"📝 Finalizing dataset with {len(results)} processed chunks")
        
        try:
            # Combine all QA pairs
            all_qa_pairs = []
            for result in results:
                if result and "data" in result:
                    all_qa_pairs.extend(result["data"])
            
            # Save as JSON
            json_file = output_path / f"{dataset_name}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "dataset_name": dataset_name,
                    "total_qa_pairs": len(all_qa_pairs),
                    "chunks_processed": len(results),
                    "qa_pairs": all_qa_pairs,
                    "generation_metadata": {
                        "session_id": self.session_id,
                        "config": asdict(self.config),
                        "progress": asdict(self.progress)
                    }
                }, f, indent=2, ensure_ascii=False)
            
            # Save as JSONL for training
            jsonl_file = output_path / f"{dataset_name}.jsonl"
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                for qa_pair in all_qa_pairs:
                    training_example = {
                        "prompt": qa_pair["question"],
                        "completion": qa_pair["answer"],
                        "metadata": {
                            "source": qa_pair.get("source", "unknown"),
                            "chunk_id": qa_pair.get("chunk_id", -1)
                        }
                    }
                    f.write(json.dumps(training_example, ensure_ascii=False) + "\n")
            
            # Update progress with output files
            self.progress.output_files = [str(json_file), str(jsonl_file)]
            
            logger.info(f"💾 Dataset saved:")
            logger.info(f"   📄 JSON: {json_file} ({len(all_qa_pairs)} QA pairs)")
            logger.info(f"   📄 JSONL: {jsonl_file} (training format)")
            
        except Exception as e:
            logger.error(f"❌ Failed to finalize dataset: {e}")
            self.progress.add_error(f"Finalization error: {str(e)}")
            raise
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a specific session.
        
        Args:
            session_id: Session ID to check
            
        Returns:
            Session status dictionary or None if not found
        """
        progress = self._load_progress(session_id)
        if not progress:
            return None
        
        return {
            "session_id": session_id,
            "progress_percent": progress.progress_percent,
            "chunks_processed": progress.chunks_processed,
            "total_chunks": progress.total_chunks,
            "current_stage": progress.current_stage,
            "qa_pairs_generated": progress.qa_pairs_generated,
            "conversations_generated": progress.conversations_generated,
            "error_count": len(progress.errors),
            "is_complete": progress.is_complete,
            "last_update": progress.last_update
        }
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all available sessions with their status.
        
        Returns:
            List of session status dictionaries
        """
        sessions = []
        
        for checkpoint_file in self.checkpoint_dir.glob("progress_*.json"):
            try:
                session_id = checkpoint_file.stem.replace("progress_", "")
                status = self.get_session_status(session_id)
                if status:
                    sessions.append(status)
            except Exception as e:
                logger.warning(f"⚠️ Failed to read checkpoint {checkpoint_file}: {e}")
        
        # Sort by last update time
        sessions.sort(key=lambda s: s.get("last_update", 0), reverse=True)
        return sessions


# Convenience function for easy usage
async def generate_resumable_dataset(
    input_chunks: List[Dict[str, Any]],
    output_dir: str,
    config: Optional[AugmentoolkitConfig] = None,
    dataset_name: str = "resumable_dataset",
    resume: bool = True,
    checkpoint_dir: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    High-level function for resumable dataset generation.
    
    Args:
        input_chunks: List of chunk dictionaries
        output_dir: Output directory
        config: Augmentoolkit configuration (uses default if None)
        dataset_name: Name for the dataset
        resume: Whether to resume from previous progress
        checkpoint_dir: Directory for progress checkpoints
        **kwargs: Additional configuration options
        
    Returns:
        Generation results dictionary
    """
    # Use default config if none provided
    if config is None:
        from .augmentoolkit_generator import create_augmentoolkit_config
        config = create_augmentoolkit_config(**kwargs)
    
    # Create resumable generator
    generator = ResumableAugmentoolkitGenerator(config, checkpoint_dir)
    
    # Generate dataset
    return await generator.generate_factual_dataset_resumable(
        input_chunks=input_chunks,
        output_dir=output_dir,
        dataset_name=dataset_name,
        resume=resume
    )