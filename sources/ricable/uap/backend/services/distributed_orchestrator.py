# File: backend/services/distributed_orchestrator.py
"""
Distributed Orchestrator for UAP Platform
Coordinates distributed document processing, AI inference, and agent workloads
across Ray cluster nodes with intelligent load balancing and task optimization.
"""

import asyncio
import logging
import uuid
import time
import json
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum

# Import distributed components
from ..distributed.ray_manager import (
    ray_cluster_manager, 
    submit_distributed_task,
    get_distributed_task_status,
    DistributedTask,
    TaskStatus
)

# Import existing services
from ..processors.document_processor import DocumentProcessor, DocumentMetadata, DocumentContent
from ..processors.document_service import DocumentService
from .agent_orchestrator import UAP_AgentOrchestrator

# Import monitoring
from ..monitoring.logs.logger import uap_logger, EventType, LogLevel
from ..monitoring.metrics.performance import (
    performance_monitor,
    start_agent_request,
    finish_agent_request
)
from ..monitoring.metrics.prometheus_metrics import (
    record_agent_request, record_distributed_workload,
    update_distributed_workload_progress, update_distributed_queue_metrics
)

class WorkloadType(Enum):
    """Types of distributed workloads"""
    DOCUMENT_PROCESSING = "document_processing"
    AI_INFERENCE = "ai_inference"
    BATCH_ANALYSIS = "batch_analysis"
    MODEL_TRAINING = "model_training"
    DATA_PROCESSING = "data_processing"
    MULTI_AGENT_TASK = "multi_agent_task"

class ProcessingStrategy(Enum):
    """Processing strategies for different workloads"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    MAP_REDUCE = "map_reduce"
    PIPELINE = "pipeline"
    ADAPTIVE = "adaptive"

@dataclass
class DistributedWorkload:
    """Distributed workload configuration"""
    workload_id: str
    workload_type: WorkloadType
    strategy: ProcessingStrategy
    input_data: Dict[str, Any]
    tasks: List[str]  # Task IDs
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "queued"
    progress: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['workload_type'] = self.workload_type.value
        data['strategy'] = self.strategy.value
        data['created_at'] = self.created_at.isoformat()
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data

class DistributedOrchestrator:
    """
    Distributed Orchestrator for coordinating complex workloads across the Ray cluster.
    Integrates with existing agent orchestrator and document processing systems.
    """
    
    def __init__(self, agent_orchestrator: UAP_AgentOrchestrator):
        """
        Initialize the distributed orchestrator.
        
        Args:
            agent_orchestrator: Reference to the main agent orchestrator
        """
        self.agent_orchestrator = agent_orchestrator
        self.logger = logging.getLogger(__name__)
        
        # Initialize processors
        self.document_processor = DocumentProcessor()
        self.document_service = DocumentService()
        
        # Workload tracking
        self.workloads: Dict[str, DistributedWorkload] = {}
        self.workload_queue: List[str] = []  # Workload IDs in execution order
        
        # Performance metrics
        self.metrics = {
            'total_workloads': 0,
            'successful_workloads': 0,
            'failed_workloads': 0,
            'avg_workload_time': 0.0,
            'total_documents_processed': 0,
            'total_ai_inferences': 0
        }
        
        # Task handlers for different workload types
        self.task_handlers = {
            WorkloadType.DOCUMENT_PROCESSING: self._handle_document_processing,
            WorkloadType.AI_INFERENCE: self._handle_ai_inference,
            WorkloadType.BATCH_ANALYSIS: self._handle_batch_analysis,
            WorkloadType.DATA_PROCESSING: self._handle_data_processing,
            WorkloadType.MULTI_AGENT_TASK: self._handle_multi_agent_task
        }
    
    async def submit_workload(self,
                            workload_type: WorkloadType,
                            input_data: Dict[str, Any],
                            strategy: ProcessingStrategy = ProcessingStrategy.ADAPTIVE,
                            priority: int = 0,
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit a distributed workload for processing.
        
        Args:
            workload_type: Type of workload to process
            input_data: Input data for the workload
            strategy: Processing strategy to use
            priority: Workload priority (higher = more important)
            metadata: Additional metadata for the workload
            
        Returns:
            Workload ID for tracking
        """
        workload_id = str(uuid.uuid4())
        
        # Create workload record
        workload = DistributedWorkload(
            workload_id=workload_id,
            workload_type=workload_type,
            strategy=strategy,
            input_data=input_data,
            tasks=[],
            created_at=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.workloads[workload_id] = workload
        self.workload_queue.append(workload_id)
        
        # Sort queue by priority (would need to add priority field to workload)
        # For now, use FIFO
        
        # Log workload submission
        uap_logger.log_event(
            LogLevel.INFO,
            f"Distributed workload submitted: {workload_type.value}",
            EventType.AGENT,
            {
                "workload_id": workload_id,
                "workload_type": workload_type.value,
                "strategy": strategy.value,
                "queue_size": len(self.workload_queue)
            },
            "distributed_orchestrator"
        )
        
        # Start processing workload
        asyncio.create_task(self._process_workload(workload_id))
        
        return workload_id
    
    async def _process_workload(self, workload_id: str):
        """
        Process a distributed workload based on its type and strategy.
        
        Args:
            workload_id: Workload ID to process
        """
        workload = self.workloads[workload_id]
        start_time = time.time()
        
        try:
            # Mark workload as started
            workload.status = "running"
            workload.started_at = datetime.utcnow()
            
            # Get appropriate handler
            handler = self.task_handlers.get(workload.workload_type)
            if not handler:
                raise ValueError(f"No handler for workload type: {workload.workload_type}")
            
            # Execute workload based on strategy
            if workload.strategy == ProcessingStrategy.SEQUENTIAL:
                result = await self._execute_sequential(workload, handler)
            elif workload.strategy == ProcessingStrategy.PARALLEL:
                result = await self._execute_parallel(workload, handler)
            elif workload.strategy == ProcessingStrategy.MAP_REDUCE:
                result = await self._execute_map_reduce(workload, handler)
            elif workload.strategy == ProcessingStrategy.PIPELINE:
                result = await self._execute_pipeline(workload, handler)
            else:  # ADAPTIVE
                result = await self._execute_adaptive(workload, handler)
            
            # Mark workload as completed
            workload.status = "completed"
            workload.completed_at = datetime.utcnow()
            workload.result = result
            workload.progress = 100.0
            
            # Update metrics
            self.metrics['successful_workloads'] += 1
            execution_time = time.time() - start_time
            self._update_avg_workload_time(execution_time)
            
            # Record Prometheus metrics
            record_distributed_workload(
                workload.workload_type.value,
                workload.strategy.value,
                "completed",
                execution_time
            )
            
            self.logger.info(f"Workload {workload_id} completed successfully in {execution_time:.2f}s")
            
        except Exception as e:
            workload.status = "failed"
            workload.error = str(e)
            workload.completed_at = datetime.utcnow()
            
            self.metrics['failed_workloads'] += 1
            
            # Record Prometheus metrics
            execution_time = time.time() - start_time
            record_distributed_workload(
                workload.workload_type.value,
                workload.strategy.value,
                "failed",
                execution_time
            )
            
            self.logger.error(f"Workload {workload_id} failed: {str(e)}")
        
        finally:
            # Remove from queue
            if workload_id in self.workload_queue:
                self.workload_queue.remove(workload_id)
            
            self.metrics['total_workloads'] += 1
    
    async def _execute_sequential(self, workload: DistributedWorkload, handler: Callable) -> Any:
        """
        Execute workload tasks sequentially.
        
        Args:
            workload: Workload to execute
            handler: Task handler function
            
        Returns:
            Combined results from all tasks
        """
        results = []
        
        # Break input data into chunks if needed
        task_inputs = await handler(workload.input_data, "prepare")
        
        for i, task_input in enumerate(task_inputs):
            # Update progress
            workload.progress = (i / len(task_inputs)) * 100
            update_distributed_workload_progress(workload.workload_id, workload.progress)
            
            # Submit task to Ray cluster
            task_id = await submit_distributed_task(
                f"{workload.workload_type.value}_sequential",
                handler,
                {"input_data": task_input, "mode": "execute"}
            )
            workload.tasks.append(task_id)
            
            # Wait for task completion
            result = await self._wait_for_task(task_id)
            results.append(result)
        
        # Combine results
        return await handler({"results": results}, "combine")
    
    async def _execute_parallel(self, workload: DistributedWorkload, handler: Callable) -> Any:
        """
        Execute workload tasks in parallel.
        
        Args:
            workload: Workload to execute
            handler: Task handler function
            
        Returns:
            Combined results from all parallel tasks
        """
        # Break input data into parallel chunks
        task_inputs = await handler(workload.input_data, "prepare")
        
        # Submit all tasks in parallel
        task_ids = []
        for task_input in task_inputs:
            task_id = await submit_distributed_task(
                f"{workload.workload_type.value}_parallel",
                handler,
                {"input_data": task_input, "mode": "execute"}
            )
            task_ids.append(task_id)
            workload.tasks.append(task_id)
        
        # Wait for all tasks to complete
        results = []
        for i, task_id in enumerate(task_ids):
            result = await self._wait_for_task(task_id)
            results.append(result)
            
            # Update progress
            workload.progress = ((i + 1) / len(task_ids)) * 100
            update_distributed_workload_progress(workload.workload_id, workload.progress)
        
        # Combine results
        return await handler({"results": results}, "combine")
    
    async def _execute_map_reduce(self, workload: DistributedWorkload, handler: Callable) -> Any:
        """
        Execute workload using map-reduce pattern.
        
        Args:
            workload: Workload to execute
            handler: Task handler function
            
        Returns:
            Reduced results
        """
        # Map phase: distribute work
        map_inputs = await handler(workload.input_data, "map_prepare")
        
        map_task_ids = []
        for map_input in map_inputs:
            task_id = await submit_distributed_task(
                f"{workload.workload_type.value}_map",
                handler,
                {"input_data": map_input, "mode": "map"}
            )
            map_task_ids.append(task_id)
            workload.tasks.append(task_id)
        
        # Wait for map tasks
        map_results = []
        for i, task_id in enumerate(map_task_ids):
            result = await self._wait_for_task(task_id)
            map_results.append(result)
            workload.progress = (i / len(map_task_ids)) * 50  # 50% for map phase
        
        # Reduce phase: combine results
        reduce_inputs = await handler({"map_results": map_results}, "reduce_prepare")
        
        if len(reduce_inputs) > 1:
            # Multiple reduce tasks
            reduce_task_ids = []
            for reduce_input in reduce_inputs:
                task_id = await submit_distributed_task(
                    f"{workload.workload_type.value}_reduce",
                    handler,
                    {"input_data": reduce_input, "mode": "reduce"}
                )
                reduce_task_ids.append(task_id)
                workload.tasks.append(task_id)
            
            # Wait for reduce tasks
            reduce_results = []
            for i, task_id in enumerate(reduce_task_ids):
                result = await self._wait_for_task(task_id)
                reduce_results.append(result)
                workload.progress = 50 + ((i + 1) / len(reduce_task_ids)) * 50
            
            # Final combine
            return await handler({"reduce_results": reduce_results}, "final_combine")
        else:
            # Single reduce task
            task_id = await submit_distributed_task(
                f"{workload.workload_type.value}_reduce",
                handler,
                {"input_data": reduce_inputs[0], "mode": "reduce"}
            )
            workload.tasks.append(task_id)
            
            result = await self._wait_for_task(task_id)
            workload.progress = 100.0
            return result
    
    async def _execute_pipeline(self, workload: DistributedWorkload, handler: Callable) -> Any:
        """
        Execute workload using pipeline pattern.
        
        Args:
            workload: Workload to execute
            handler: Task handler function
            
        Returns:
            Final pipeline result
        """
        # Get pipeline stages
        stages = await handler(workload.input_data, "pipeline_prepare")
        
        current_data = workload.input_data
        
        for i, stage in enumerate(stages):
            # Update progress
            workload.progress = (i / len(stages)) * 100
            
            # Submit stage task
            task_id = await submit_distributed_task(
                f"{workload.workload_type.value}_stage_{i}",
                handler,
                {"input_data": current_data, "stage": stage, "mode": "pipeline_stage"}
            )
            workload.tasks.append(task_id)
            
            # Wait for stage completion
            stage_result = await self._wait_for_task(task_id)
            current_data = stage_result
        
        workload.progress = 100.0
        return current_data
    
    async def _execute_adaptive(self, workload: DistributedWorkload, handler: Callable) -> Any:
        """
        Execute workload using adaptive strategy based on data size and cluster capacity.
        
        Args:
            workload: Workload to execute
            handler: Task handler function
            
        Returns:
            Optimized execution result
        """
        # Analyze workload characteristics
        analysis = await handler(workload.input_data, "analyze")
        
        data_size = analysis.get("data_size", 0)
        complexity = analysis.get("complexity", "medium")
        parallelizable = analysis.get("parallelizable", True)
        
        # Get cluster capacity
        cluster_status = await ray_cluster_manager.get_cluster_status()
        available_nodes = cluster_status.get("ray_info", {}).get("nodes", 1)
        
        # Choose optimal strategy
        if data_size < 1000 or not parallelizable:
            # Small data or non-parallelizable: use sequential
            workload.strategy = ProcessingStrategy.SEQUENTIAL
            return await self._execute_sequential(workload, handler)
        
        elif available_nodes > 1 and complexity == "high":
            # Large cluster and complex data: use map-reduce
            workload.strategy = ProcessingStrategy.MAP_REDUCE
            return await self._execute_map_reduce(workload, handler)
        
        elif available_nodes > 1:
            # Multiple nodes available: use parallel
            workload.strategy = ProcessingStrategy.PARALLEL
            return await self._execute_parallel(workload, handler)
        
        else:
            # Single node: use sequential
            workload.strategy = ProcessingStrategy.SEQUENTIAL
            return await self._execute_sequential(workload, handler)
    
    async def _wait_for_task(self, task_id: str, timeout: int = 3600) -> Any:
        """
        Wait for a distributed task to complete.
        
        Args:
            task_id: Task ID to wait for
            timeout: Maximum wait time in seconds
            
        Returns:
            Task result
            
        Raises:
            TimeoutError: If task doesn't complete within timeout
            RuntimeError: If task fails
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            task = await get_distributed_task_status(task_id)
            
            if not task:
                raise RuntimeError(f"Task {task_id} not found")
            
            if task.status == TaskStatus.COMPLETED:
                return task.result
            
            elif task.status == TaskStatus.FAILED:
                raise RuntimeError(f"Task {task_id} failed: {task.error}")
            
            elif task.status == TaskStatus.CANCELLED:
                raise RuntimeError(f"Task {task_id} was cancelled")
            
            # Wait before checking again
            await asyncio.sleep(1)
        
        raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
    
    # Task handlers for different workload types
    
    async def _handle_document_processing(self, input_data: Dict[str, Any], mode: str) -> Any:
        """
        Handle document processing workloads.
        
        Args:
            input_data: Input data for document processing
            mode: Processing mode (prepare, execute, combine, etc.)
            
        Returns:
            Processing results based on mode
        """
        if mode == "prepare":
            # Prepare document processing tasks
            documents = input_data.get("documents", [])
            return [{"document": doc} for doc in documents]
        
        elif mode == "execute":
            # Process a single document
            document_info = input_data["document"]
            
            if "file_path" in document_info:
                # Process file
                metadata = await self.document_processor.process_document(
                    document_info["file_path"],
                    document_info.get("filename", "unknown")
                )
                content = await self.document_processor.get_document_content(metadata.id)
                
                self.metrics['total_documents_processed'] += 1
                
                return {
                    "document_id": metadata.id,
                    "metadata": metadata.to_dict() if hasattr(metadata, 'to_dict') else asdict(metadata),
                    "content": asdict(content) if content else None
                }
            
            elif "content" in document_info:
                # Process raw content
                # Use agent orchestrator for content analysis
                response = await self.agent_orchestrator.handle_http_chat(
                    "agno",
                    f"Analyze this document content: {document_info['content'][:1000]}...",
                    "agno",
                    {"document_content": document_info["content"]}
                )
                
                return {
                    "analysis": response.get("content", "No analysis available"),
                    "metadata": response.get("metadata", {})
                }
        
        elif mode == "combine":
            # Combine document processing results
            results = input_data["results"]
            
            combined = {
                "total_documents": len(results),
                "successful_documents": len([r for r in results if r and "error" not in r]),
                "failed_documents": len([r for r in results if not r or "error" in r]),
                "documents": results
            }
            
            return combined
        
        elif mode == "analyze":
            # Analyze workload for adaptive strategy
            documents = input_data.get("documents", [])
            
            return {
                "data_size": len(documents),
                "complexity": "high" if len(documents) > 100 else "medium",
                "parallelizable": True
            }
        
        return None
    
    async def _handle_ai_inference(self, input_data: Dict[str, Any], mode: str) -> Any:
        """
        Handle AI inference workloads.
        
        Args:
            input_data: Input data for AI inference
            mode: Processing mode
            
        Returns:
            AI inference results
        """
        if mode == "prepare":
            # Prepare AI inference tasks
            queries = input_data.get("queries", [])
            framework = input_data.get("framework", "copilot")
            
            return [{"query": query, "framework": framework} for query in queries]
        
        elif mode == "execute":
            # Execute AI inference
            query = input_data["query"]
            framework = input_data.get("framework", "copilot")
            
            # Use agent orchestrator for AI inference
            response = await self.agent_orchestrator.handle_http_chat(
                framework,
                query,
                framework,
                input_data.get("context", {})
            )
            
            self.metrics['total_ai_inferences'] += 1
            
            return {
                "query": query,
                "response": response.get("content", "No response"),
                "framework": framework,
                "metadata": response.get("metadata", {})
            }
        
        elif mode == "combine":
            # Combine AI inference results
            results = input_data["results"]
            
            return {
                "total_inferences": len(results),
                "successful_inferences": len([r for r in results if r and "error" not in r]),
                "inferences": results
            }
        
        elif mode == "analyze":
            # Analyze AI inference workload
            queries = input_data.get("queries", [])
            
            return {
                "data_size": len(queries),
                "complexity": "high" if len(queries) > 50 else "medium",
                "parallelizable": True
            }
        
        return None
    
    async def _handle_batch_analysis(self, input_data: Dict[str, Any], mode: str) -> Any:
        """
        Handle batch analysis workloads.
        
        Args:
            input_data: Input data for batch analysis
            mode: Processing mode
            
        Returns:
            Batch analysis results
        """
        if mode == "prepare":
            # Prepare batch analysis tasks
            items = input_data.get("items", [])
            analysis_type = input_data.get("analysis_type", "general")
            
            return [{"item": item, "analysis_type": analysis_type} for item in items]
        
        elif mode == "execute":
            # Execute batch analysis on single item
            item = input_data["item"]
            analysis_type = input_data.get("analysis_type", "general")
            
            # Choose appropriate framework based on analysis type
            if analysis_type in ["document", "text", "content"]:
                framework = "agno"
            elif analysis_type in ["support", "workflow"]:
                framework = "mastra"
            else:
                framework = "copilot"
            
            # Perform analysis using agent orchestrator
            query = f"Perform {analysis_type} analysis on: {str(item)[:500]}..."
            response = await self.agent_orchestrator.handle_http_chat(
                framework,
                query,
                framework,
                {"item": item, "analysis_type": analysis_type}
            )
            
            return {
                "item": item,
                "analysis_type": analysis_type,
                "analysis": response.get("content", "No analysis available"),
                "metadata": response.get("metadata", {})
            }
        
        elif mode == "combine":
            # Combine batch analysis results
            results = input_data["results"]
            
            return {
                "total_items_analyzed": len(results),
                "successful_analyses": len([r for r in results if r and "error" not in r]),
                "analyses": results
            }
        
        elif mode == "analyze":
            # Analyze batch workload
            items = input_data.get("items", [])
            
            return {
                "data_size": len(items),
                "complexity": "high" if len(items) > 200 else "medium",
                "parallelizable": True
            }
        
        return None
    
    async def _handle_data_processing(self, input_data: Dict[str, Any], mode: str) -> Any:
        """
        Handle data processing workloads.
        
        Args:
            input_data: Input data for processing
            mode: Processing mode
            
        Returns:
            Data processing results
        """
        if mode == "prepare":
            # Prepare data processing tasks
            data_chunks = input_data.get("data_chunks", [])
            processing_function = input_data.get("processing_function", "default")
            
            return [{"chunk": chunk, "function": processing_function} for chunk in data_chunks]
        
        elif mode == "execute":
            # Process data chunk
            chunk = input_data["chunk"]
            function = input_data.get("function", "default")
            
            # Simulate data processing (replace with actual logic)
            processed_data = {
                "original_size": len(str(chunk)),
                "processed_size": len(str(chunk)) * 1.1,  # Simulated processing
                "function_used": function,
                "processed_at": datetime.utcnow().isoformat(),
                "data": chunk  # In real implementation, this would be processed data
            }
            
            return processed_data
        
        elif mode == "combine":
            # Combine data processing results
            results = input_data["results"]
            
            total_original_size = sum(r.get("original_size", 0) for r in results)
            total_processed_size = sum(r.get("processed_size", 0) for r in results)
            
            return {
                "total_chunks_processed": len(results),
                "total_original_size": total_original_size,
                "total_processed_size": total_processed_size,
                "compression_ratio": total_processed_size / total_original_size if total_original_size > 0 else 1.0,
                "processed_data": [r["data"] for r in results]
            }
        
        elif mode == "analyze":
            # Analyze data processing workload
            data_chunks = input_data.get("data_chunks", [])
            
            return {
                "data_size": len(data_chunks),
                "complexity": "medium",
                "parallelizable": True
            }
        
        return None
    
    async def _handle_multi_agent_task(self, input_data: Dict[str, Any], mode: str) -> Any:
        """
        Handle multi-agent collaborative tasks.
        
        Args:
            input_data: Input data for multi-agent task
            mode: Processing mode
            
        Returns:
            Multi-agent task results
        """
        if mode == "prepare":
            # Prepare multi-agent tasks
            task_definition = input_data.get("task_definition", {})
            agents = task_definition.get("agents", ["copilot", "agno", "mastra"])
            
            return [{"agent": agent, "task_data": task_definition} for agent in agents]
        
        elif mode == "execute":
            # Execute task with specific agent
            agent = input_data["agent"]
            task_data = input_data["task_data"]
            
            query = task_data.get("query", "Process this task")
            context = task_data.get("context", {})
            
            # Execute task with specified agent
            response = await self.agent_orchestrator.handle_http_chat(
                agent,
                query,
                agent,
                context
            )
            
            return {
                "agent": agent,
                "response": response.get("content", "No response"),
                "metadata": response.get("metadata", {})
            }
        
        elif mode == "combine":
            # Combine multi-agent results
            results = input_data["results"]
            
            # Create consolidated response from all agents
            agent_responses = {r["agent"]: r["response"] for r in results}
            
            return {
                "collaborative_result": agent_responses,
                "participating_agents": list(agent_responses.keys()),
                "consensus_analysis": "Multi-agent task completed successfully",
                "individual_responses": results
            }
        
        elif mode == "analyze":
            # Analyze multi-agent workload
            task_definition = input_data.get("task_definition", {})
            agents = task_definition.get("agents", [])
            
            return {
                "data_size": len(agents),
                "complexity": "high",  # Multi-agent coordination is complex
                "parallelizable": True
            }
        
        return None
    
    def _update_avg_workload_time(self, execution_time: float):
        """
        Update average workload execution time.
        
        Args:
            execution_time: Execution time in seconds
        """
        total_workloads = self.metrics['total_workloads']
        if total_workloads == 0:
            self.metrics['avg_workload_time'] = execution_time
        else:
            current_avg = self.metrics['avg_workload_time']
            self.metrics['avg_workload_time'] = (
                (current_avg * total_workloads + execution_time) / (total_workloads + 1)
            )
    
    async def get_workload_status(self, workload_id: str) -> Optional[DistributedWorkload]:
        """
        Get status of a specific workload.
        
        Args:
            workload_id: Workload ID to check
            
        Returns:
            DistributedWorkload object or None if not found
        """
        return self.workloads.get(workload_id)
    
    async def cancel_workload(self, workload_id: str) -> bool:
        """
        Cancel a queued or running workload.
        
        Args:
            workload_id: Workload ID to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        if workload_id not in self.workloads:
            return False
        
        workload = self.workloads[workload_id]
        
        if workload.status in ["completed", "failed", "cancelled"]:
            return False
        
        # Cancel all associated tasks
        for task_id in workload.tasks:
            await ray_cluster_manager.cancel_task(task_id)
        
        workload.status = "cancelled"
        workload.completed_at = datetime.utcnow()
        workload.error = "Workload cancelled by user"
        
        # Record Prometheus metrics
        record_distributed_workload(
            workload.workload_type.value,
            workload.strategy.value,
            "cancelled"
        )
        
        # Remove from queue
        if workload_id in self.workload_queue:
            self.workload_queue.remove(workload_id)
        
        self.logger.info(f"Workload {workload_id} cancelled")
        return True
    
    async def get_orchestrator_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the distributed orchestrator.
        
        Returns:
            Status dictionary with metrics and workload information
        """
        # Get cluster status
        cluster_status = await ray_cluster_manager.get_cluster_status()
        
        # Workload statistics
        workload_stats = {
            'total_workloads': len(self.workloads),
            'queued_workloads': len([w for w in self.workloads.values() if w.status == "queued"]),
            'running_workloads': len([w for w in self.workloads.values() if w.status == "running"]),
            'completed_workloads': len([w for w in self.workloads.values() if w.status == "completed"]),
            'failed_workloads': len([w for w in self.workloads.values() if w.status == "failed"]),
            'queue_length': len(self.workload_queue)
        }
        
        # Update distributed queue metrics
        task_counts = {}
        for workload in self.workloads.values():
            workload_type = workload.workload_type.value
            if workload_type not in task_counts:
                task_counts[workload_type] = 0
            task_counts[workload_type] += len(workload.tasks)
        
        update_distributed_queue_metrics(len(self.workload_queue), task_counts)
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'cluster_status': cluster_status,
            'workload_statistics': workload_stats,
            'performance_metrics': self.metrics,
            'supported_workload_types': [wt.value for wt in WorkloadType],
            'supported_strategies': [ps.value for ps in ProcessingStrategy]
        }
    
    async def cleanup(self):
        """
        Clean up the distributed orchestrator and associated resources.
        """
        # Cancel all running workloads
        for workload_id in list(self.workload_queue):
            await self.cancel_workload(workload_id)
        
        self.logger.info("Distributed orchestrator cleanup complete")

# Global distributed orchestrator instance (initialized when agent orchestrator is available)
distributed_orchestrator: Optional[DistributedOrchestrator] = None

def initialize_distributed_orchestrator(agent_orchestrator: UAP_AgentOrchestrator) -> DistributedOrchestrator:
    """Initialize the global distributed orchestrator instance"""
    global distributed_orchestrator
    distributed_orchestrator = DistributedOrchestrator(agent_orchestrator)
    return distributed_orchestrator

# Convenience functions
async def submit_distributed_workload(workload_type: WorkloadType, 
                                     input_data: Dict[str, Any],
                                     strategy: ProcessingStrategy = ProcessingStrategy.ADAPTIVE) -> str:
    """Submit a distributed workload"""
    if not distributed_orchestrator:
        raise RuntimeError("Distributed orchestrator not initialized")
    return await distributed_orchestrator.submit_workload(workload_type, input_data, strategy)

async def get_workload_status(workload_id: str) -> Optional[DistributedWorkload]:
    """Get workload status"""
    if not distributed_orchestrator:
        return None
    return await distributed_orchestrator.get_workload_status(workload_id)

async def get_distributed_status() -> Dict[str, Any]:
    """Get distributed orchestrator status"""
    if not distributed_orchestrator:
        return {"error": "Distributed orchestrator not initialized"}
    return await distributed_orchestrator.get_orchestrator_status()
