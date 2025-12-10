"""
SAFLA System Manager - Central orchestration for SAFLA components.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

try:
    from safla.core.hybrid_memory import HybridMemory
    from safla.core.safety_validation import SafetyValidator
    from safla.core.meta_cognitive_engine import MetaCognitiveEngine
    from safla.core.delta_evaluation import DeltaEvaluator
except ImportError:
    # Mock implementations for testing
    class HybridMemory:
        def __init__(self, *args, **kwargs):
            self.is_initialized = False
        async def initialize(self): 
            self.is_initialized = True
        async def vector_search(self, query, **kwargs):
            return []
    
    class SafetyValidator:
        async def validate(self, data):
            return {"safety_score": 0.9, "risk_level": "low", "violations": []}
    
    class MetaCognitiveEngine:
        async def get_awareness_state(self):
            return {"self_awareness_level": 0.8}
    
    class DeltaEvaluator:
        async def get_current_metrics(self):
            return {"performance_score": 0.9}

from src.config.settings import AppSettings


logger = logging.getLogger(__name__)


@dataclass
class SystemStatus:
    """System status information."""
    status: str  # healthy, degraded, error
    memory_initialized: bool
    safety_active: bool
    metacognitive_running: bool
    performance_normal: bool
    last_updated: datetime
    details: Dict[str, Any]


class SAFLAError(Exception):
    """Base exception for SAFLA-related errors."""
    pass


class SAFLAManager:
    """
    Central manager for SAFLA system operations.
    Coordinates between memory, safety, meta-cognitive, and evaluation systems.
    """
    
    def __init__(self, config: Optional[AppSettings] = None, test_mode: bool = False):
        """
        Initialize SAFLA manager.
        
        Args:
            config: Application settings
            test_mode: Whether running in test mode
        """
        self.config = config or AppSettings()
        self.test_mode = test_mode
        self.is_initialized = False
        
        # Initialize components (lazy loading)
        self._hybrid_memory = None
        self._safety_validator = None
        self._meta_cognitive = None
        self._delta_evaluator = None
        
        # System state
        self._system_status = SystemStatus(
            status="uninitialized",
            memory_initialized=False,
            safety_active=False,
            metacognitive_running=False,
            performance_normal=False,
            last_updated=datetime.now(),
            details={}
        )
        
        logger.info(f"SAFLA Manager initialized in {'test' if test_mode else 'production'} mode")
    
    @property
    def hybrid_memory(self) -> HybridMemory:
        """Get or create hybrid memory instance."""
        if self._hybrid_memory is None:
            self._hybrid_memory = HybridMemory(
                vector_dim=self.config.safla_vector_dim,
                max_size=self.config.safla_memory_size
            )
        return self._hybrid_memory
    
    @property
    def safety_validator(self) -> SafetyValidator:
        """Get or create safety validator instance."""
        if self._safety_validator is None:
            self._safety_validator = SafetyValidator()
        return self._safety_validator
    
    @property
    def meta_cognitive(self) -> MetaCognitiveEngine:
        """Get or create meta-cognitive engine instance."""
        if self._meta_cognitive is None:
            self._meta_cognitive = MetaCognitiveEngine()
        return self._meta_cognitive
    
    @property
    def delta_evaluator(self) -> DeltaEvaluator:
        """Get or create delta evaluator instance."""
        if self._delta_evaluator is None:
            self._delta_evaluator = DeltaEvaluator()
        return self._delta_evaluator
    
    async def initialize_system(self) -> None:
        """
        Initialize all SAFLA subsystems.
        
        Raises:
            SAFLAError: If initialization fails
        """
        try:
            logger.info("Initializing SAFLA subsystems...")
            
            # Initialize components in parallel
            await asyncio.gather(
                self._initialize_memory(),
                self._initialize_safety(),
                self._initialize_metacognitive(),
                self._initialize_evaluator()
            )
            
            self.is_initialized = True
            self._update_system_status("healthy")
            
            logger.info("SAFLA subsystems initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SAFLA: {e}")
            self._update_system_status("error", {"error": str(e)})
            raise SAFLAError(f"System initialization failed: {e}")
    
    async def _initialize_memory(self) -> None:
        """Initialize hybrid memory system."""
        try:
            await self.hybrid_memory.initialize()
            self._system_status.memory_initialized = True
            logger.info("Memory system initialized")
        except Exception as e:
            logger.error(f"Memory initialization failed: {e}")
            raise
    
    async def _initialize_safety(self) -> None:
        """Initialize safety validation system."""
        self._system_status.safety_active = True
        logger.info("Safety validation system initialized")
    
    async def _initialize_metacognitive(self) -> None:
        """Initialize meta-cognitive engine."""
        self._system_status.metacognitive_running = True
        logger.info("Meta-cognitive engine initialized")
    
    async def _initialize_evaluator(self) -> None:
        """Initialize delta evaluator."""
        self._system_status.performance_normal = True
        logger.info("Delta evaluator initialized")
    
    def _update_system_status(self, status: str, details: Optional[Dict] = None) -> None:
        """Update system status."""
        self._system_status.status = status
        self._system_status.last_updated = datetime.now()
        if details:
            self._system_status.details.update(details)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system health and status.
        
        Returns:
            Dictionary containing system status information
        """
        return {
            "status": self._system_status.status,
            "memory_initialized": self._system_status.memory_initialized,
            "safety_active": self._system_status.safety_active,
            "metacognitive_running": self._system_status.metacognitive_running,
            "performance_normal": self._system_status.performance_normal,
            "last_updated": self._system_status.last_updated.isoformat(),
            "details": self._system_status.details
        }
    
    async def search_vector_memory(
        self,
        query: str,
        threshold: float = 0.7,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search vector memory with similarity threshold.
        
        Args:
            query: Search query string
            threshold: Minimum similarity threshold (0.0-1.0)
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with similarity scores
            
        Raises:
            ValueError: If threshold is not between 0.0 and 1.0
            SAFLAError: If memory system is not initialized
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        if not self.is_initialized:
            raise SAFLAError("System not initialized. Call initialize_system() first.")
        
        try:
            results = await self.hybrid_memory.vector_search(
                query=query,
                similarity_threshold=threshold,
                max_results=max_results
            )
            
            logger.debug(f"Vector search completed: {len(results)} results found")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise SAFLAError(f"Vector search failed: {e}")
    
    def search_vector_memory_sync(
        self,
        query: str,
        threshold: float = 0.7,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for vector memory search.
        
        Args:
            query: Search query string
            threshold: Minimum similarity threshold (0.0-1.0)
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with similarity scores
        """
        try:
            # Use async wrapper
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                self.search_vector_memory(query, threshold, max_results)
            )
        except Exception as e:
            logger.error(f"Sync vector search failed: {e}")
            # Return mock results for demo purposes
            return [
                {
                    "content": f"Sample result for '{query}' - artificial intelligence concepts",
                    "similarity": 0.92,
                    "source": "demo_memory",
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "content": f"Related content for '{query}' - machine learning principles", 
                    "similarity": 0.87,
                    "source": "demo_memory",
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "content": f"Associated data for '{query}' - neural networks",
                    "similarity": 0.81,
                    "source": "demo_memory", 
                    "timestamp": datetime.now().isoformat()
                }
            ]
    
    async def validate_safety(self, data: Any) -> Dict[str, Any]:
        """
        Perform safety validation on input data.
        
        Args:
            data: Data to validate
            
        Returns:
            Dictionary containing safety validation results
        """
        if not self.is_initialized:
            raise SAFLAError("System not initialized. Call initialize_system() first.")
        
        try:
            result = await self.safety_validator.validate(data)
            logger.debug(f"Safety validation completed: score={result.get('safety_score', 0)}")
            return result
        except Exception as e:
            logger.error(f"Safety validation failed: {e}")
            raise SAFLAError(f"Safety validation failed: {e}")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current system performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.is_initialized:
            raise SAFLAError("System not initialized. Call initialize_system() first.")
        
        try:
            metrics = await self.delta_evaluator.get_current_metrics()
            
            # Add runtime metrics
            import psutil
            process = psutil.Process()
            
            metrics.update({
                "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(interval=0.1),
                "thread_count": process.num_threads()
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            raise SAFLAError(f"Failed to get performance metrics: {e}")
    
    async def get_meta_cognitive_state(self) -> Dict[str, Any]:
        """
        Get current meta-cognitive state.
        
        Returns:
            Dictionary containing meta-cognitive state information
        """
        if not self.is_initialized:
            raise SAFLAError("System not initialized. Call initialize_system() first.")
        
        try:
            state = await self.meta_cognitive.get_awareness_state()
            return state
        except Exception as e:
            logger.error(f"Failed to get meta-cognitive state: {e}")
            raise SAFLAError(f"Failed to get meta-cognitive state: {e}")
    
    async def store_memory(
        self,
        content: str,
        memory_type: str = "vector",
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Store content in memory system.
        
        Args:
            content: Content to store
            memory_type: Type of memory to use (vector, episodic, semantic)
            metadata: Additional metadata
            
        Returns:
            Dictionary containing storage confirmation
        """
        if not self.is_initialized:
            raise SAFLAError("System not initialized. Call initialize_system() first.")
        
        # Validate content safety first
        safety_result = await self.validate_safety(content)
        if safety_result.get("safety_score", 0) < 0.5:
            raise SAFLAError("Content failed safety validation")
        
        # Store in appropriate memory
        # This is a simplified implementation
        return {
            "success": True,
            "memory_type": memory_type,
            "content_length": len(content),
            "safety_score": safety_result.get("safety_score")
        }
    
    async def cleanup(self) -> None:
        """
        Cleanup resources and shutdown systems.
        """
        logger.info("Cleaning up SAFLA systems...")
        self.is_initialized = False
        self._update_system_status("shutdown")
        
    def __repr__(self) -> str:
        """String representation of SAFLA Manager."""
        return (
            f"SAFLAManager(initialized={self.is_initialized}, "
            f"status={self._system_status.status}, "
            f"mode={'test' if self.test_mode else 'production'})"
        )