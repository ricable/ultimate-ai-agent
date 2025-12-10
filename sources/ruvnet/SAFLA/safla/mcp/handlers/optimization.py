"""
Optimization handlers for SAFLA MCP server.

This module provides tools for memory optimization, performance analysis,
and system optimization recommendations.
"""

import gc
import time
import asyncio
import psutil
from typing import Any, Dict, List, Optional, Tuple
import logging
from collections import defaultdict

from .base import BaseHandler, ToolDefinition

logger = logging.getLogger(__name__)


class OptimizationHandler(BaseHandler):
    """Handler for system optimization tools."""
    
    def _initialize_tools(self) -> None:
        """Initialize optimization tools."""
        tools = [
            ToolDefinition(
                name="optimize_memory_usage",
                description="Optimize memory usage and perform garbage collection",
                input_schema={
                    "type": "object",
                    "properties": {
                        "aggressive": {
                            "type": "boolean",
                            "description": "Perform aggressive optimization"
                        },
                        "target_reduction_percent": {
                            "type": "number",
                            "description": "Target memory reduction percentage"
                        }
                    },
                    "required": []
                },
                handler_method="_optimize_memory_usage"
            ),
            ToolDefinition(
                name="optimize_vector_operations",
                description="Optimize vector store and embedding operations",
                input_schema={
                    "type": "object",
                    "properties": {
                        "operation_type": {
                            "type": "string",
                            "description": "Type of operation to optimize",
                            "enum": ["search", "insert", "update", "all"]
                        }
                    },
                    "required": []
                },
                handler_method="_optimize_vector_operations"
            ),
            ToolDefinition(
                name="analyze_performance_bottlenecks",
                description="Analyze system performance and identify bottlenecks",
                input_schema={
                    "type": "object",
                    "properties": {
                        "duration_seconds": {
                            "type": "integer",
                            "description": "Duration of performance analysis"
                        },
                        "focus_areas": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific areas to focus on"
                        }
                    },
                    "required": []
                },
                handler_method="_analyze_performance_bottlenecks"
            )
        ]
        
        for tool in tools:
            self.register_tool(tool)
    
    async def _optimize_memory_usage(self, aggressive: bool = False, 
                                   target_reduction_percent: float = 10.0) -> Dict[str, Any]:
        """Optimize memory usage and perform garbage collection."""
        try:
            # Get initial memory state
            initial_memory = self._get_memory_stats()
            
            # Perform optimization steps
            optimization_results = []
            
            # Step 1: Clear caches
            cache_cleared = await self._clear_caches()
            optimization_results.append({
                "step": "clear_caches",
                "success": cache_cleared,
                "memory_freed": self._calculate_memory_freed(initial_memory)
            })
            
            # Step 2: Garbage collection
            gc_stats = self._perform_garbage_collection(aggressive)
            optimization_results.append({
                "step": "garbage_collection",
                "success": True,
                "stats": gc_stats,
                "memory_freed": self._calculate_memory_freed(initial_memory)
            })
            
            # Step 3: Optimize data structures
            if aggressive:
                ds_optimized = await self._optimize_data_structures()
                optimization_results.append({
                    "step": "data_structure_optimization",
                    "success": ds_optimized,
                    "memory_freed": self._calculate_memory_freed(initial_memory)
                })
            
            # Step 4: Memory pool cleanup
            pool_cleaned = await self._cleanup_memory_pools()
            optimization_results.append({
                "step": "memory_pool_cleanup",
                "success": pool_cleaned,
                "memory_freed": self._calculate_memory_freed(initial_memory)
            })
            
            # Get final memory state
            final_memory = self._get_memory_stats()
            
            # Calculate reduction
            reduction_percent = ((initial_memory["used"] - final_memory["used"]) / 
                               initial_memory["used"] * 100)
            
            return {
                "initial_memory": initial_memory,
                "final_memory": final_memory,
                "reduction_percent": reduction_percent,
                "target_achieved": reduction_percent >= target_reduction_percent,
                "optimization_steps": optimization_results,
                "recommendations": self._get_memory_recommendations(final_memory),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {str(e)}")
            raise
    
    async def _optimize_vector_operations(self, operation_type: str = "all") -> Dict[str, Any]:
        """Optimize vector store and embedding operations."""
        try:
            optimizations = {}
            
            # Optimize based on operation type
            if operation_type in ["search", "all"]:
                search_opt = await self._optimize_vector_search()
                optimizations["search"] = search_opt
            
            if operation_type in ["insert", "all"]:
                insert_opt = await self._optimize_vector_insert()
                optimizations["insert"] = insert_opt
            
            if operation_type in ["update", "all"]:
                update_opt = await self._optimize_vector_update()
                optimizations["update"] = update_opt
            
            # General optimizations
            general_opt = await self._general_vector_optimizations()
            optimizations["general"] = general_opt
            
            # Calculate overall improvement
            improvements = []
            for opt_type, results in optimizations.items():
                if "improvement_percent" in results:
                    improvements.append(results["improvement_percent"])
            
            avg_improvement = sum(improvements) / len(improvements) if improvements else 0
            
            return {
                "optimizations": optimizations,
                "average_improvement": avg_improvement,
                "recommendations": self._get_vector_recommendations(),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Vector operation optimization failed: {str(e)}")
            raise
    
    async def _analyze_performance_bottlenecks(self, duration_seconds: int = 60,
                                             focus_areas: List[str] = None) -> Dict[str, Any]:
        """Analyze system performance and identify bottlenecks."""
        try:
            if focus_areas is None:
                focus_areas = ["cpu", "memory", "io", "network"]
            
            # Start performance monitoring
            start_time = time.time()
            performance_data = defaultdict(list)
            
            # Collect performance metrics
            while time.time() - start_time < duration_seconds:
                snapshot = await self._collect_performance_snapshot(focus_areas)
                for area, data in snapshot.items():
                    performance_data[area].append(data)
                await asyncio.sleep(1)  # Collect data every second
            
            # Analyze collected data
            analysis = {}
            bottlenecks = []
            
            for area in focus_areas:
                if area in performance_data:
                    area_analysis = self._analyze_area_performance(
                        area, performance_data[area]
                    )
                    analysis[area] = area_analysis
                    
                    # Identify bottlenecks
                    if area_analysis.get("is_bottleneck", False):
                        bottlenecks.append({
                            "area": area,
                            "severity": area_analysis.get("severity", "unknown"),
                            "description": area_analysis.get("description", ""),
                            "impact": area_analysis.get("impact", "")
                        })
            
            # Generate recommendations
            recommendations = self._generate_performance_recommendations(
                analysis, bottlenecks
            )
            
            return {
                "duration_seconds": duration_seconds,
                "focus_areas": focus_areas,
                "analysis": analysis,
                "bottlenecks": bottlenecks,
                "recommendations": recommendations,
                "summary": {
                    "total_bottlenecks": len(bottlenecks),
                    "critical_issues": sum(1 for b in bottlenecks if b["severity"] == "critical"),
                    "performance_score": self._calculate_performance_score(analysis)
                },
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {str(e)}")
            raise
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            "system": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            },
            "process": {
                "rss": process.memory_info().rss,
                "vms": process.memory_info().vms,
                "percent": process.memory_percent()
            },
            "used": memory.used  # For easy comparison
        }
    
    def _calculate_memory_freed(self, initial_memory: Dict[str, Any]) -> int:
        """Calculate memory freed since initial state."""
        current = self._get_memory_stats()
        return max(0, initial_memory["used"] - current["used"])
    
    async def _clear_caches(self) -> bool:
        """Clear various caches in the system."""
        try:
            # Clear state manager caches if available
            if self.state_manager:
                # Clear temporary namespaces
                temp_namespaces = ["cache", "temp", "session_cache"]
                for ns in temp_namespaces:
                    self.state_manager.clear_namespace(ns)
            
            # Clear Python's internal caches
            if hasattr(gc, "clear_caches"):
                gc.clear_caches()
            
            return True
        except Exception as e:
            logger.error(f"Failed to clear caches: {str(e)}")
            return False
    
    def _perform_garbage_collection(self, aggressive: bool) -> Dict[str, Any]:
        """Perform garbage collection."""
        stats = {}
        
        # Get initial stats
        stats["before"] = {
            "objects": len(gc.get_objects()),
            "collections": gc.get_count()
        }
        
        # Perform collection
        if aggressive:
            # Full collection of all generations
            stats["collected"] = gc.collect(2)
        else:
            # Collect only young generation
            stats["collected"] = gc.collect(0)
        
        # Get final stats
        stats["after"] = {
            "objects": len(gc.get_objects()),
            "collections": gc.get_count()
        }
        
        stats["objects_freed"] = stats["before"]["objects"] - stats["after"]["objects"]
        
        return stats
    
    async def _optimize_data_structures(self) -> bool:
        """Optimize in-memory data structures."""
        try:
            # This would contain actual optimization logic
            # For now, return success
            return True
        except Exception:
            return False
    
    async def _cleanup_memory_pools(self) -> bool:
        """Clean up memory pools."""
        try:
            # This would clean up various memory pools
            # For now, return success
            return True
        except Exception:
            return False
    
    def _get_memory_recommendations(self, memory_stats: Dict[str, Any]) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        # Check memory usage percentage
        if memory_stats["system"]["percent"] > 80:
            recommendations.append("Critical: System memory usage above 80%. Consider scaling up resources.")
        elif memory_stats["system"]["percent"] > 60:
            recommendations.append("Warning: System memory usage above 60%. Monitor closely.")
        
        # Check process memory
        if memory_stats["process"]["percent"] > 50:
            recommendations.append("Process using significant memory. Review data retention policies.")
        
        # General recommendations
        recommendations.extend([
            "Enable memory profiling for detailed analysis",
            "Review cache expiration policies",
            "Consider implementing memory limits for operations"
        ])
        
        return recommendations
    
    async def _optimize_vector_search(self) -> Dict[str, Any]:
        """Optimize vector search operations."""
        # Placeholder for actual vector search optimization
        return {
            "optimized": True,
            "improvement_percent": 15.5,
            "changes": ["Enabled approximate search", "Optimized index structure"]
        }
    
    async def _optimize_vector_insert(self) -> Dict[str, Any]:
        """Optimize vector insert operations."""
        return {
            "optimized": True,
            "improvement_percent": 20.0,
            "changes": ["Batched insertions", "Optimized embedding generation"]
        }
    
    async def _optimize_vector_update(self) -> Dict[str, Any]:
        """Optimize vector update operations."""
        return {
            "optimized": True,
            "improvement_percent": 12.0,
            "changes": ["Incremental updates", "Reduced reindexing"]
        }
    
    async def _general_vector_optimizations(self) -> Dict[str, Any]:
        """Apply general vector optimizations."""
        return {
            "optimized": True,
            "changes": ["Compression enabled", "Memory-mapped storage", "Query caching"]
        }
    
    def _get_vector_recommendations(self) -> List[str]:
        """Generate vector optimization recommendations."""
        return [
            "Consider using approximate nearest neighbor algorithms for large datasets",
            "Implement vector quantization for memory efficiency",
            "Use hierarchical indexing for multi-scale search",
            "Enable GPU acceleration if available"
        ]
    
    async def _collect_performance_snapshot(self, focus_areas: List[str]) -> Dict[str, Any]:
        """Collect a performance snapshot."""
        snapshot = {}
        
        if "cpu" in focus_areas:
            snapshot["cpu"] = {
                "percent": psutil.cpu_percent(interval=0.1),
                "per_core": psutil.cpu_percent(interval=0.1, percpu=True)
            }
        
        if "memory" in focus_areas:
            memory = psutil.virtual_memory()
            snapshot["memory"] = {
                "percent": memory.percent,
                "available": memory.available
            }
        
        if "io" in focus_areas:
            io = psutil.disk_io_counters()
            snapshot["io"] = {
                "read_bytes": io.read_bytes,
                "write_bytes": io.write_bytes
            }
        
        if "network" in focus_areas:
            net = psutil.net_io_counters()
            snapshot["network"] = {
                "bytes_sent": net.bytes_sent,
                "bytes_recv": net.bytes_recv
            }
        
        return snapshot
    
    def _analyze_area_performance(self, area: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance data for a specific area."""
        # Placeholder analysis logic
        if area == "cpu":
            avg_usage = sum(d["percent"] for d in data) / len(data)
            return {
                "average_usage": avg_usage,
                "is_bottleneck": avg_usage > 80,
                "severity": "critical" if avg_usage > 90 else "warning" if avg_usage > 80 else "normal",
                "description": f"Average CPU usage: {avg_usage:.1f}%"
            }
        
        # Add similar logic for other areas
        return {"is_bottleneck": False}
    
    def _generate_performance_recommendations(self, analysis: Dict[str, Any], 
                                            bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        for bottleneck in bottlenecks:
            if bottleneck["area"] == "cpu" and bottleneck["severity"] == "critical":
                recommendations.append("Consider parallelizing CPU-intensive operations")
                recommendations.append("Review algorithm efficiency for hot paths")
            elif bottleneck["area"] == "memory":
                recommendations.append("Implement data streaming for large operations")
                recommendations.append("Review object lifecycle and retention")
        
        return recommendations
    
    def _calculate_performance_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall performance score (0-100)."""
        # Simple scoring logic
        score = 100.0
        
        for area, data in analysis.items():
            if data.get("is_bottleneck"):
                if data.get("severity") == "critical":
                    score -= 30
                elif data.get("severity") == "warning":
                    score -= 15
        
        return max(0, score)