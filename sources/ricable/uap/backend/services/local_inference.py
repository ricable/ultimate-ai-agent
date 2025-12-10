# File: backend/services/local_inference.py
# Local Inference Service - Model Loading and Caching System
# Manages MLX models and provides fallback capabilities

import asyncio
import json
import logging
import os
import psutil
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta

# Import the MLX processor
from ..processors.mlx_processor import MLXProcessor

# Configure logging
logger = logging.getLogger(__name__)

class LocalInferenceService:
    """Service for managing local inference with model loading and caching.
    
    This service provides a high-level interface for local AI inference,
    handling model management, caching, and fallback strategies.
    """
    
    def __init__(self):
        self.service_name = "LocalInference"
        self.is_initialized = False
        self.status = "initializing"
        self.error_count = 0
        self.last_error = None
        
        # Initialize MLX processor
        self.mlx_processor = MLXProcessor()
        
        # Configuration
        self.fallback_enabled = os.getenv("LOCAL_INFERENCE_FALLBACK", "true").lower() == "true"
        self.auto_cleanup = os.getenv("LOCAL_INFERENCE_AUTO_CLEANUP", "true").lower() == "true"
        self.cleanup_interval = int(os.getenv("LOCAL_INFERENCE_CLEANUP_INTERVAL", "3600"))  # 1 hour
        self.max_models = int(os.getenv("LOCAL_INFERENCE_MAX_MODELS", "3"))
        self.memory_threshold = float(os.getenv("LOCAL_INFERENCE_MEMORY_THRESHOLD", "0.8"))  # 80%
        
        # Model management
        self.model_usage_stats = {}
        self.model_priorities = {}
        self.last_cleanup = datetime.utcnow()
        
        # Fallback configurations
        self.fallback_responses = {
            "code": "I can help with coding tasks. While running in fallback mode, I can provide general programming guidance and suggestions.",
            "general": "I'm your local AI assistant. While my full capabilities are loading, I can still help with many questions.",
            "technical": "I can assist with technical questions. My local inference capabilities are being prepared for optimal performance.",
            "help": "I'm here to help! I can assist with various tasks including coding, analysis, and general questions."
        }
        
        logger.info(f"Local inference service initialized.")
    
    async def initialize(self) -> None:
        """Initialize the local inference service."""
        try:
            logger.info("Initializing local inference service...")
            
            # Initialize MLX processor
            await self.mlx_processor.initialize()
            
            # Start background tasks
            if self.auto_cleanup:
                asyncio.create_task(self._background_cleanup_task())
            
            asyncio.create_task(self._monitor_system_resources())
            
            self.is_initialized = True
            self.status = "active"
            logger.info("Local inference service initialization complete.")
            
        except Exception as e:
            self.status = "error"
            self.last_error = str(e)
            logger.error(f"Local inference service initialization failed: {e}")
            raise
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a message using local inference with fallback support.
        
        Args:
            message: The user's input message
            context: Additional context for processing
            
        Returns:
            Dict containing 'content' and 'metadata' keys
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Try MLX inference first
            try:
                response = await self.mlx_processor.process_message(message, context)
                
                # Update usage statistics
                model_name = context.get("model", self.mlx_processor.default_model)
                self._update_usage_stats(model_name)
                
                # Add local inference metadata
                response["metadata"]["local_inference_service"] = True
                response["metadata"]["service"] = self.service_name
                
                return response
                
            except Exception as e:
                logger.warning(f"MLX inference failed: {e}")
                
                if self.fallback_enabled:
                    return await self._fallback_response(message, context, str(e))
                else:
                    raise
                    
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"Local inference service error: {e}")
            
            return {
                "content": f"Local inference service encountered an error: {str(e)}",
                "metadata": {
                    "source": self.service_name,
                    "error": True,
                    "error_message": str(e),
                    "fallback_used": False,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
    
    async def _fallback_response(self, message: str, context: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Generate fallback response when MLX inference fails."""
        message_lower = message.lower()
        
        # Determine response type
        if any(keyword in message_lower for keyword in ["code", "function", "programming", "script"]):
            response_type = "code"
        elif any(keyword in message_lower for keyword in ["help", "support", "assist"]):
            response_type = "help"
        elif any(keyword in message_lower for keyword in ["technical", "system", "configure", "setup"]):
            response_type = "technical"
        else:
            response_type = "general"
        
        base_response = self.fallback_responses.get(response_type, self.fallback_responses["general"])
        
        return {
            "content": f"{base_response}\n\nNote: Local inference is temporarily unavailable. Your message: '{message[:100]}{'...' if len(message) > 100 else ''}'",
            "metadata": {
                "source": self.service_name,
                "fallback_used": True,
                "fallback_reason": error,
                "response_type": response_type,
                "local_inference_available": False,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    def _update_usage_stats(self, model_name: str) -> None:
        """Update usage statistics for a model."""
        if model_name not in self.model_usage_stats:
            self.model_usage_stats[model_name] = {
                "usage_count": 0,
                "last_used": datetime.utcnow(),
                "total_time": 0.0
            }
        
        self.model_usage_stats[model_name]["usage_count"] += 1
        self.model_usage_stats[model_name]["last_used"] = datetime.utcnow()
    
    async def _background_cleanup_task(self) -> None:
        """Background task for automatic model cleanup."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._perform_cleanup()
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
    
    async def _perform_cleanup(self) -> None:
        """Perform automatic cleanup of unused models."""
        try:
            current_time = datetime.utcnow()
            
            # Skip if recently cleaned
            if (current_time - self.last_cleanup).seconds < self.cleanup_interval / 2:
                return
            
            logger.info("Performing automatic model cleanup...")
            
            loaded_models = list(self.mlx_processor.models.keys())
            
            # If we're at or near the model limit, clean up least used models
            if len(loaded_models) >= self.max_models:
                models_to_unload = self._select_models_for_cleanup(loaded_models)
                
                for model_name in models_to_unload:
                    await self.mlx_processor.unload_model(model_name)
                    logger.info(f"Automatically unloaded model: {model_name}")
            
            # Check memory usage
            memory_percent = psutil.virtual_memory().percent / 100
            if memory_percent > self.memory_threshold:
                logger.warning(f"High memory usage detected: {memory_percent:.1%}")
                await self._emergency_cleanup()
            
            self.last_cleanup = current_time
            
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")
    
    def _select_models_for_cleanup(self, loaded_models: List[str]) -> List[str]:
        """Select models for cleanup based on usage patterns."""
        models_with_stats = []
        
        for model_name in loaded_models:
            stats = self.model_usage_stats.get(model_name, {})
            usage_count = stats.get("usage_count", 0)
            last_used = stats.get("last_used", datetime.utcnow() - timedelta(days=1))
            hours_since_use = (datetime.utcnow() - last_used).total_seconds() / 3600
            
            # Calculate cleanup score (higher = more likely to be cleaned up)
            cleanup_score = hours_since_use / max(usage_count, 1)
            
            models_with_stats.append((model_name, cleanup_score))
        
        # Sort by cleanup score and select models to unload
        models_with_stats.sort(key=lambda x: x[1], reverse=True)
        
        # Unload oldest/least used models
        num_to_unload = max(1, len(loaded_models) - self.max_models + 1)
        return [model[0] for model in models_with_stats[:num_to_unload]]
    
    async def _emergency_cleanup(self) -> None:
        """Emergency cleanup when memory usage is high."""
        logger.warning("Performing emergency memory cleanup...")
        
        loaded_models = list(self.mlx_processor.models.keys())
        models_to_unload = self._select_models_for_cleanup(loaded_models)
        
        # Unload up to half of the models
        max_unload = max(1, len(loaded_models) // 2)
        for model_name in models_to_unload[:max_unload]:
            await self.mlx_processor.unload_model(model_name)
            logger.warning(f"Emergency unloaded model: {model_name}")
    
    async def _monitor_system_resources(self) -> None:
        """Monitor system resources and performance."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Get system stats
                memory = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=1)
                
                # Log if resources are high
                if memory.percent > 85:
                    logger.warning(f"High memory usage: {memory.percent:.1f}%")
                
                if cpu > 90:
                    logger.warning(f"High CPU usage: {cpu:.1f}%")
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
    
    async def load_model(self, model_identifier: str, priority: str = "normal") -> Dict[str, Any]:
        """Load a model with specified priority.
        
        Args:
            model_identifier: Model name or path
            priority: Priority level (high, normal, low)
            
        Returns:
            Dict with load result
        """
        try:
            # Set model priority
            self.model_priorities[model_identifier] = priority
            
            # Check if we need to make room
            if len(self.mlx_processor.models) >= self.max_models:
                await self._make_room_for_model(model_identifier, priority)
            
            # Load the model
            result = await self.mlx_processor.load_model(model_identifier)
            
            if result["success"]:
                logger.info(f"Model {model_identifier} loaded with priority {priority}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to load model {model_identifier}: {e}")
            return {
                "success": False,
                "error": str(e),
                "model": model_identifier
            }
    
    async def _make_room_for_model(self, new_model: str, priority: str) -> None:
        """Make room for a new model by unloading others."""
        if priority == "high":
            # For high priority models, aggressively unload others
            loaded_models = list(self.mlx_processor.models.keys())
            models_to_unload = self._select_models_for_cleanup(loaded_models)
            
            for model_name in models_to_unload[:2]:  # Unload up to 2 models
                await self.mlx_processor.unload_model(model_name)
                logger.info(f"Unloaded {model_name} to make room for high-priority {new_model}")
        else:
            # For normal/low priority, only unload if necessary
            if len(self.mlx_processor.models) >= self.max_models:
                await self._perform_cleanup()
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        mlx_status = self.mlx_processor.get_status()
        
        # Get system resource info
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        
        return {
            "service": self.service_name,
            "status": self.status,
            "initialized": self.is_initialized,
            "mlx_processor": mlx_status,
            "model_management": {
                "max_models": self.max_models,
                "loaded_count": len(self.mlx_processor.models),
                "usage_stats": self.model_usage_stats,
                "priorities": self.model_priorities,
                "last_cleanup": self.last_cleanup.isoformat()
            },
            "system_resources": {
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "cpu_percent": cpu,
                "memory_threshold": self.memory_threshold * 100
            },
            "configuration": {
                "fallback_enabled": self.fallback_enabled,
                "auto_cleanup": self.auto_cleanup,
                "cleanup_interval": self.cleanup_interval
            },
            "error_count": self.error_count,
            "last_error": self.last_error,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def list_models(self) -> Dict[str, Any]:
        """List available and loaded models."""
        available_models = await self.mlx_processor.list_available_models()
        
        return {
            "available_models": available_models,
            "loaded_models": list(self.mlx_processor.models.keys()),
            "usage_stats": self.model_usage_stats,
            "priorities": self.model_priorities,
            "capacity": {
                "max_models": self.max_models,
                "current_count": len(self.mlx_processor.models),
                "available_slots": self.max_models - len(self.mlx_processor.models)
            }
        }
    
    async def unload_model(self, model_identifier: str) -> Dict[str, Any]:
        """Unload a specific model."""
        result = await self.mlx_processor.unload_model(model_identifier)
        
        # Clean up tracking data
        if result["success"]:
            self.model_usage_stats.pop(model_identifier, None)
            self.model_priorities.pop(model_identifier, None)
        
        return result
    
    async def get_model_recommendations(self, task_type: str = "general") -> List[Dict[str, Any]]:
        """Get model recommendations based on task type.
        
        Args:
            task_type: Type of task (code, chat, analysis, etc.)
            
        Returns:
            List of recommended models
        """
        available_models = await self.mlx_processor.list_available_models()
        
        recommendations = []
        
        for model in available_models:
            score = 0
            reasoning = []
            
            # Score based on task type
            if task_type == "code":
                if "phi" in model["id"].lower():
                    score += 3
                    reasoning.append("Optimized for coding tasks")
                elif "llama" in model["id"].lower():
                    score += 2
                    reasoning.append("Good general programming support")
            
            elif task_type == "chat":
                if "instruct" in model["path"].lower():
                    score += 3
                    reasoning.append("Instruction-tuned for conversations")
                if "llama" in model["id"].lower():
                    score += 2
                    reasoning.append("Strong conversational abilities")
            
            elif task_type == "analysis":
                if "3b" in model["id"] or "7b" in model["id"]:
                    score += 3
                    reasoning.append("Larger model for complex analysis")
                elif "gemma" in model["id"].lower():
                    score += 2
                    reasoning.append("Good analytical capabilities")
            
            # Default scoring
            else:
                score = 1
                reasoning.append("General purpose model")
            
            # Adjust for memory constraints
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if model["memory_mb"] / 1024 > memory_gb * 0.5:
                score -= 1
                reasoning.append("High memory usage")
            
            # Boost if already loaded
            if model["loaded"]:
                score += 1
                reasoning.append("Already loaded")
            
            recommendations.append({
                **model,
                "recommendation_score": max(0, score),
                "reasoning": reasoning
            })
        
        # Sort by recommendation score
        recommendations.sort(key=lambda x: x["recommendation_score"], reverse=True)
        
        return recommendations
    
    async def optimize_models(self) -> Dict[str, Any]:
        """Optimize model loading based on usage patterns."""
        try:
            logger.info("Optimizing model configuration...")
            
            # Analyze usage patterns
            optimization_results = {
                "actions_taken": [],
                "recommendations": [],
                "performance_impact": {}
            }
            
            # Unload rarely used models
            for model_name, stats in self.model_usage_stats.items():
                if stats["usage_count"] < 2 and model_name in self.mlx_processor.models:
                    hours_since_use = (datetime.utcnow() - stats["last_used"]).total_seconds() / 3600
                    if hours_since_use > 24:  # Not used in 24 hours
                        await self.mlx_processor.unload_model(model_name)
                        optimization_results["actions_taken"].append(f"Unloaded rarely used model: {model_name}")
            
            # Suggest models to preload based on patterns
            # This is a simplified heuristic - could be made more sophisticated
            most_used = max(self.model_usage_stats.items(), 
                          key=lambda x: x[1]["usage_count"], 
                          default=(None, None))
            
            if most_used[0] and most_used[0] not in self.mlx_processor.models:
                optimization_results["recommendations"].append(f"Consider preloading: {most_used[0]}")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Model optimization error: {e}")
            return {
                "error": str(e),
                "actions_taken": [],
                "recommendations": []
            }
    
    async def cleanup(self) -> None:
        """Clean up the local inference service."""
        try:
            logger.info("Cleaning up local inference service...")
            
            # Clean up MLX processor
            await self.mlx_processor.cleanup()
            
            # Clear tracking data
            self.model_usage_stats.clear()
            self.model_priorities.clear()
            
            logger.info("Local inference service cleanup complete.")
            
        except Exception as e:
            logger.error(f"Local inference service cleanup error: {e}")

print("Local inference service module loaded.")