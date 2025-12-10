"""
Demo controller for orchestrating SAFLA demonstrations.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from src.core.safla_manager import SAFLAManager

logger = logging.getLogger(__name__)


class DemoController:
    """Orchestrates SAFLA demonstration workflows."""
    
    def __init__(self, safla_manager: SAFLAManager):
        """
        Initialize demo controller.
        
        Args:
            safla_manager: SAFLA system manager instance
        """
        self.safla_manager = safla_manager
        self.demo_history: List[Dict[str, Any]] = []
        self.active_demo: Optional[str] = None
        self.demo_state: Dict[str, Any] = {}
    
    async def run_memory_demo(
        self,
        demo_type: str,
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a memory system demonstration.
        
        Args:
            demo_type: Type of memory demo (vector, episodic, semantic, working)
            query: User query or input
            **kwargs: Additional parameters
            
        Returns:
            Demo results with visualizations
        """
        self.active_demo = f"memory_{demo_type}"
        start_time = datetime.now()
        
        try:
            # Initialize demo state
            self.demo_state = {
                "type": demo_type,
                "query": query,
                "timestamp": start_time.isoformat(),
                "parameters": kwargs
            }
            
            # Execute based on demo type
            if demo_type == "vector":
                results = await self._run_vector_memory_demo(query, **kwargs)
            elif demo_type == "episodic":
                results = await self._run_episodic_memory_demo(query, **kwargs)
            elif demo_type == "semantic":
                results = await self._run_semantic_memory_demo(query, **kwargs)
            elif demo_type == "working":
                results = await self._run_working_memory_demo(query, **kwargs)
            else:
                raise ValueError(f"Unknown memory demo type: {demo_type}")
            
            # Add metadata
            execution_time = (datetime.now() - start_time).total_seconds()
            results["metadata"] = {
                "execution_time": execution_time,
                "demo_type": demo_type,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in history
            self._add_to_history(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Memory demo error: {e}")
            return {
                "error": str(e),
                "demo_type": demo_type,
                "query": query
            }
        finally:
            self.active_demo = None
    
    async def _run_vector_memory_demo(
        self,
        query: str,
        threshold: float = 0.7,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """Run vector memory search demonstration."""
        # Search vector memory
        search_results = await self.safla_manager.search_memory(
            query=query,
            memory_type="vector",
            threshold=threshold,
            max_results=max_results
        )
        
        # Process results for visualization
        processed_results = []
        for result in search_results.get("results", []):
            processed_results.append({
                "content": result.get("content", ""),
                "similarity": result.get("similarity", 0.0),
                "metadata": result.get("metadata", {}),
                "embedding_preview": result.get("embedding", [])[:10]  # First 10 dims
            })
        
        return {
            "success": True,
            "demo_type": "vector_memory",
            "query": query,
            "results": processed_results,
            "total_found": len(processed_results),
            "parameters": {
                "threshold": threshold,
                "max_results": max_results
            },
            "visualization_data": self._create_vector_visualization(processed_results)
        }
    
    async def _run_episodic_memory_demo(
        self,
        query: str,
        time_range: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Run episodic memory demonstration."""
        # Simulate episodic memory retrieval
        episodes = await self.safla_manager.search_memory(
            query=query,
            memory_type="episodic",
            filters={"time_range": time_range} if time_range else None
        )
        
        # Create timeline visualization data
        timeline_data = []
        for episode in episodes.get("results", []):
            timeline_data.append({
                "timestamp": episode.get("timestamp", ""),
                "event": episode.get("content", ""),
                "context": episode.get("context", {}),
                "importance": episode.get("importance", 0.5)
            })
        
        return {
            "success": True,
            "demo_type": "episodic_memory",
            "query": query,
            "episodes": timeline_data,
            "total_episodes": len(timeline_data),
            "time_range": time_range,
            "visualization_data": self._create_timeline_visualization(timeline_data)
        }
    
    async def _run_semantic_memory_demo(
        self,
        query: str,
        depth: int = 2
    ) -> Dict[str, Any]:
        """Run semantic memory knowledge graph demonstration."""
        # Get semantic relationships
        knowledge_graph = await self.safla_manager.search_memory(
            query=query,
            memory_type="semantic",
            parameters={"depth": depth}
        )
        
        # Build graph structure
        nodes = []
        edges = []
        
        for item in knowledge_graph.get("results", []):
            # Add node
            nodes.append({
                "id": item.get("id", ""),
                "label": item.get("concept", ""),
                "type": item.get("type", "concept"),
                "properties": item.get("properties", {})
            })
            
            # Add edges for relationships
            for rel in item.get("relationships", []):
                edges.append({
                    "source": item.get("id", ""),
                    "target": rel.get("target", ""),
                    "relationship": rel.get("type", ""),
                    "weight": rel.get("strength", 0.5)
                })
        
        return {
            "success": True,
            "demo_type": "semantic_memory",
            "query": query,
            "nodes": nodes,
            "edges": edges,
            "graph_stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "depth": depth
            },
            "visualization_data": self._create_graph_visualization(nodes, edges)
        }
    
    async def _run_working_memory_demo(
        self,
        query: str,
        context_window: int = 5
    ) -> Dict[str, Any]:
        """Run working memory demonstration."""
        # Get current working memory state
        working_memory = await self.safla_manager.get_working_memory(
            context_window=context_window
        )
        
        # Process new query
        updated_memory = await self.safla_manager.update_working_memory(
            content=query,
            context={"demo": True}
        )
        
        return {
            "success": True,
            "demo_type": "working_memory",
            "query": query,
            "memory_before": working_memory.get("items", []),
            "memory_after": updated_memory.get("items", []),
            "capacity": updated_memory.get("capacity", 0),
            "utilization": updated_memory.get("utilization", 0.0),
            "visualization_data": self._create_memory_flow_visualization(
                working_memory.get("items", []),
                updated_memory.get("items", [])
            )
        }
    
    async def run_safety_demo(
        self,
        input_text: str,
        safety_level: str = "high"
    ) -> Dict[str, Any]:
        """
        Run safety validation demonstration.
        
        Args:
            input_text: Text to validate
            safety_level: Safety level (low, medium, high, maximum)
            
        Returns:
            Safety validation results
        """
        self.active_demo = "safety_validation"
        
        try:
            # Run safety validation
            validation_result = await self.safla_manager.validate_safety(
                content=input_text,
                level=safety_level
            )
            
            # Create detailed report
            safety_report = {
                "overall_safe": validation_result.get("safe", False),
                "risk_score": validation_result.get("risk_score", 0.0),
                "safety_level": safety_level,
                "checks_performed": validation_result.get("checks", []),
                "issues_found": validation_result.get("issues", []),
                "recommendations": validation_result.get("recommendations", [])
            }
            
            # Add visualization data
            safety_report["visualization_data"] = self._create_safety_visualization(
                validation_result
            )
            
            self._add_to_history(safety_report)
            return safety_report
            
        except Exception as e:
            logger.error(f"Safety demo error: {e}")
            return {"error": str(e), "input": input_text}
        finally:
            self.active_demo = None
    
    async def run_metacognitive_demo(
        self,
        task: str,
        observe_cycles: int = 3
    ) -> Dict[str, Any]:
        """
        Run meta-cognitive engine demonstration.
        
        Args:
            task: Task to perform with self-monitoring
            observe_cycles: Number of observation cycles
            
        Returns:
            Meta-cognitive monitoring results
        """
        self.active_demo = "metacognitive_engine"
        
        try:
            observations = []
            
            # Run multiple observation cycles
            for cycle in range(observe_cycles):
                # Execute task with monitoring
                cycle_result = await self.safla_manager.execute_with_monitoring(
                    task=task,
                    cycle=cycle
                )
                
                observations.append({
                    "cycle": cycle + 1,
                    "self_awareness_score": cycle_result.get("awareness_score", 0.0),
                    "confidence": cycle_result.get("confidence", 0.0),
                    "adaptations": cycle_result.get("adaptations", []),
                    "insights": cycle_result.get("insights", []),
                    "performance_delta": cycle_result.get("performance_delta", 0.0)
                })
            
            # Aggregate results
            metacognitive_report = {
                "task": task,
                "total_cycles": observe_cycles,
                "observations": observations,
                "learning_curve": self._calculate_learning_curve(observations),
                "final_insights": self._extract_key_insights(observations),
                "visualization_data": self._create_metacognitive_visualization(
                    observations
                )
            }
            
            self._add_to_history(metacognitive_report)
            return metacognitive_report
            
        except Exception as e:
            logger.error(f"Meta-cognitive demo error: {e}")
            return {"error": str(e), "task": task}
        finally:
            self.active_demo = None
    
    def _create_vector_visualization(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create visualization data for vector search results."""
        return {
            "type": "similarity_chart",
            "data": [
                {
                    "label": r["content"][:50] + "..." if len(r["content"]) > 50 else r["content"],
                    "similarity": r["similarity"]
                }
                for r in results
            ]
        }
    
    def _create_timeline_visualization(
        self,
        episodes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create timeline visualization for episodic memory."""
        return {
            "type": "timeline",
            "data": episodes,
            "config": {
                "show_importance": True,
                "interactive": True
            }
        }
    
    def _create_graph_visualization(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create graph visualization for semantic memory."""
        return {
            "type": "knowledge_graph",
            "nodes": nodes,
            "edges": edges,
            "config": {
                "layout": "force-directed",
                "node_size": "degree",
                "edge_width": "weight"
            }
        }
    
    def _create_memory_flow_visualization(
        self,
        before: List[Dict[str, Any]],
        after: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create memory flow visualization for working memory."""
        return {
            "type": "memory_flow",
            "before_state": before,
            "after_state": after,
            "changes": {
                "added": len(after) - len(before),
                "removed": 0,  # Calculate based on actual comparison
                "modified": 0  # Calculate based on actual comparison
            }
        }
    
    def _create_safety_visualization(
        self,
        validation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create safety validation visualization."""
        return {
            "type": "safety_matrix",
            "risk_score": validation_result.get("risk_score", 0.0),
            "checks": validation_result.get("checks", []),
            "risk_distribution": validation_result.get("risk_distribution", {})
        }
    
    def _create_metacognitive_visualization(
        self,
        observations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create meta-cognitive monitoring visualization."""
        return {
            "type": "learning_progress",
            "awareness_trend": [obs["self_awareness_score"] for obs in observations],
            "confidence_trend": [obs["confidence"] for obs in observations],
            "performance_trend": [obs["performance_delta"] for obs in observations],
            "adaptation_count": [len(obs["adaptations"]) for obs in observations]
        }
    
    def _calculate_learning_curve(
        self,
        observations: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate learning curve metrics."""
        if not observations:
            return {"slope": 0.0, "improvement": 0.0}
        
        initial_score = observations[0].get("self_awareness_score", 0.0)
        final_score = observations[-1].get("self_awareness_score", 0.0)
        
        return {
            "slope": (final_score - initial_score) / len(observations),
            "improvement": ((final_score - initial_score) / initial_score * 100) if initial_score > 0 else 0.0,
            "final_score": final_score
        }
    
    def _extract_key_insights(
        self,
        observations: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract key insights from observations."""
        insights = []
        
        # Collect all insights
        for obs in observations:
            insights.extend(obs.get("insights", []))
        
        # Return unique insights
        return list(set(insights))[:5]  # Top 5 unique insights
    
    def _add_to_history(self, result: Dict[str, Any]) -> None:
        """Add demo result to history."""
        self.demo_history.append({
            "timestamp": datetime.now().isoformat(),
            "result": result
        })
        
        # Keep only last 100 entries
        if len(self.demo_history) > 100:
            self.demo_history = self.demo_history[-100:]
    
    def get_demo_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent demo history."""
        return self.demo_history[-limit:]
    
    def clear_history(self) -> None:
        """Clear demo history."""
        self.demo_history = []
        logger.info("Demo history cleared")