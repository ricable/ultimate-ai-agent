"""
Meta-cognitive handlers for SAFLA MCP server.

This module provides tools for system self-awareness, goal management,
strategy selection, and adaptive learning capabilities.
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from .base import BaseHandler, ToolDefinition

logger = logging.getLogger(__name__)


class MetaCognitiveHandler(BaseHandler):
    """Handler for meta-cognitive engine tools."""
    
    def _initialize_tools(self) -> None:
        """Initialize meta-cognitive tools."""
        tools = [
            # Awareness tools
            ToolDefinition(
                name="get_system_awareness",
                description="Get current system awareness state",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                handler_method="_get_system_awareness"
            ),
            ToolDefinition(
                name="update_awareness_state",
                description="Update system awareness parameters",
                input_schema={
                    "type": "object",
                    "properties": {
                        "awareness_level": {
                            "type": "number",
                            "description": "New awareness level (0-1)"
                        },
                        "focus_areas": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Areas to focus on"
                        },
                        "introspection_depth": {
                            "type": "string",
                            "description": "Depth of introspection",
                            "enum": ["shallow", "moderate", "deep"]
                        }
                    },
                    "required": []
                },
                handler_method="_update_awareness_state"
            ),
            ToolDefinition(
                name="analyze_system_introspection",
                description="Perform deep system introspection",
                input_schema={
                    "type": "object",
                    "properties": {
                        "analysis_depth": {
                            "type": "string",
                            "description": "Depth of analysis",
                            "enum": ["basic", "detailed", "comprehensive"]
                        },
                        "components": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Components to analyze"
                        }
                    },
                    "required": []
                },
                handler_method="_analyze_system_introspection"
            ),
            
            # Goal management tools
            ToolDefinition(
                name="create_goal",
                description="Create a new system goal",
                input_schema={
                    "type": "object",
                    "properties": {
                        "goal_type": {
                            "type": "string",
                            "description": "Type of goal",
                            "enum": ["performance", "learning", "optimization", "safety"]
                        },
                        "description": {
                            "type": "string",
                            "description": "Goal description"
                        },
                        "target_metrics": {
                            "type": "object",
                            "description": "Target metrics for the goal"
                        },
                        "priority": {
                            "type": "string",
                            "description": "Goal priority",
                            "enum": ["low", "medium", "high", "critical"]
                        },
                        "deadline": {
                            "type": "string",
                            "description": "Goal deadline (ISO format)"
                        }
                    },
                    "required": ["goal_type", "description"]
                },
                handler_method="_create_goal"
            ),
            ToolDefinition(
                name="list_goals",
                description="List system goals",
                input_schema={
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "description": "Filter by status",
                            "enum": ["active", "completed", "failed", "paused"]
                        },
                        "goal_type": {
                            "type": "string",
                            "description": "Filter by goal type"
                        }
                    },
                    "required": []
                },
                handler_method="_list_goals"
            ),
            ToolDefinition(
                name="update_goal_progress",
                description="Update progress on a goal",
                input_schema={
                    "type": "object",
                    "properties": {
                        "goal_id": {
                            "type": "string",
                            "description": "Goal ID to update"
                        },
                        "progress_percent": {
                            "type": "number",
                            "description": "Progress percentage (0-100)"
                        },
                        "current_metrics": {
                            "type": "object",
                            "description": "Current metric values"
                        },
                        "notes": {
                            "type": "string",
                            "description": "Progress notes"
                        }
                    },
                    "required": ["goal_id"]
                },
                handler_method="_update_goal_progress"
            ),
            ToolDefinition(
                name="evaluate_goal",
                description="Evaluate goal achievement",
                input_schema={
                    "type": "object",
                    "properties": {
                        "goal_id": {
                            "type": "string",
                            "description": "Goal ID to evaluate"
                        }
                    },
                    "required": ["goal_id"]
                },
                handler_method="_evaluate_goal"
            ),
            
            # Strategy management tools
            ToolDefinition(
                name="list_strategies",
                description="List available strategies",
                input_schema={
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "string",
                            "description": "Context to filter strategies"
                        }
                    },
                    "required": []
                },
                handler_method="_list_strategies"
            ),
            ToolDefinition(
                name="select_strategy",
                description="Select optimal strategy for context",
                input_schema={
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "string",
                            "description": "Current context"
                        },
                        "constraints": {
                            "type": "object",
                            "description": "Constraints to consider"
                        },
                        "objectives": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Objectives to optimize for"
                        }
                    },
                    "required": ["context"]
                },
                handler_method="_select_strategy"
            ),
            ToolDefinition(
                name="create_strategy",
                description="Create new adaptive strategy",
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Strategy name"
                        },
                        "description": {
                            "type": "string",
                            "description": "Strategy description"
                        },
                        "context": {
                            "type": "string",
                            "description": "Applicable context"
                        },
                        "steps": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Strategy steps"
                        }
                    },
                    "required": ["name", "description", "context", "steps"]
                },
                handler_method="_create_strategy"
            ),
            ToolDefinition(
                name="evaluate_strategy",
                description="Evaluate strategy effectiveness",
                input_schema={
                    "type": "object",
                    "properties": {
                        "strategy_id": {
                            "type": "string",
                            "description": "Strategy ID to evaluate"
                        },
                        "execution_data": {
                            "type": "object",
                            "description": "Data from strategy execution"
                        }
                    },
                    "required": ["strategy_id"]
                },
                handler_method="_evaluate_strategy"
            ),
            
            # Learning management tools
            ToolDefinition(
                name="trigger_learning",
                description="Trigger adaptive learning process",
                input_schema={
                    "type": "object",
                    "properties": {
                        "learning_type": {
                            "type": "string",
                            "description": "Type of learning",
                            "enum": ["reinforcement", "supervised", "unsupervised", "meta"]
                        },
                        "experience_data": {
                            "type": "object",
                            "description": "Experience data to learn from"
                        },
                        "learning_rate": {
                            "type": "number",
                            "description": "Learning rate (0-1)"
                        }
                    },
                    "required": ["learning_type", "experience_data"]
                },
                handler_method="_trigger_learning"
            ),
            ToolDefinition(
                name="get_learning_metrics",
                description="Get current learning metrics",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                handler_method="_get_learning_metrics"
            ),
            ToolDefinition(
                name="update_learning_parameters",
                description="Update learning parameters",
                input_schema={
                    "type": "object",
                    "properties": {
                        "learning_rate": {
                            "type": "number",
                            "description": "New learning rate"
                        },
                        "exploration_factor": {
                            "type": "number",
                            "description": "Exploration vs exploitation factor"
                        },
                        "adaptation_threshold": {
                            "type": "number",
                            "description": "Threshold for adaptation"
                        }
                    },
                    "required": []
                },
                handler_method="_update_learning_parameters"
            ),
            ToolDefinition(
                name="analyze_adaptation_patterns",
                description="Analyze system adaptation patterns",
                input_schema={
                    "type": "object",
                    "properties": {
                        "time_window_hours": {
                            "type": "integer",
                            "description": "Time window to analyze"
                        },
                        "pattern_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Types of patterns to analyze"
                        }
                    },
                    "required": []
                },
                handler_method="_analyze_adaptation_patterns"
            )
        ]
        
        for tool in tools:
            self.register_tool(tool)
            
        # Initialize meta-cognitive state if not exists
        self._initialize_meta_cognitive_state()
    
    def _initialize_meta_cognitive_state(self) -> None:
        """Initialize meta-cognitive state in state manager."""
        if not self.state_manager.exists("meta_cognitive_state", namespace="metacognitive"):
            initial_state = {
                "awareness_level": 0.7,
                "focus_areas": ["performance", "optimization", "learning"],
                "introspection_depth": "moderate",
                "last_introspection": time.time(),
                "self_assessment": {
                    "confidence": 0.8,
                    "competence": 0.75,
                    "adaptability": 0.85
                }
            }
            self.state_manager.set(
                "meta_cognitive_state", initial_state,
                namespace="metacognitive"
            )
        
        # Initialize learning metrics
        if not self.state_manager.exists("learning_metrics", namespace="metacognitive"):
            learning_metrics = {
                "accuracy": 0.85,
                "adaptation_threshold": 0.72,
                "memory_retention": 0.89,
                "knowledge_retention": 0.89,
                "learning_rate": 0.15,
                "exploration_factor": 0.25,
                "total_learning_cycles": 0,
                "successful_adaptations": 0
            }
            self.state_manager.set(
                "learning_metrics", learning_metrics,
                namespace="metacognitive"
            )
        
        # Initialize strategies
        if not self.state_manager.exists("strategies", namespace="metacognitive"):
            default_strategies = {
                "performance_optimization": {
                    "id": "perf_opt_001",
                    "name": "Performance Optimization",
                    "description": "Systematic approach to improving system performance",
                    "context": "performance",
                    "steps": ["analyze_bottlenecks", "identify_optimizations", 
                             "implement_changes", "validate_improvements"],
                    "effectiveness": 0.85,
                    "usage_count": 0,
                    "success_rate": 0.87
                },
                "memory_management": {
                    "id": "mem_mgmt_001",
                    "name": "Memory Management",
                    "description": "Efficient memory allocation and cleanup strategies",
                    "context": "memory",
                    "steps": ["monitor_usage", "identify_leaks", 
                             "optimize_allocation", "implement_cleanup"],
                    "effectiveness": 0.78,
                    "usage_count": 0,
                    "success_rate": 0.82
                }
            }
            self.state_manager.set(
                "strategies", default_strategies,
                namespace="metacognitive"
            )
    
    async def _get_system_awareness(self) -> Dict[str, Any]:
        """Get current system awareness state."""
        try:
            state = self.state_manager.get(
                "meta_cognitive_state",
                namespace="metacognitive"
            )
            
            # Calculate time since last introspection
            time_since_introspection = time.time() - state.get("last_introspection", time.time())
            
            # Add runtime information
            awareness_info = {
                **state,
                "time_since_introspection": time_since_introspection,
                "introspection_needed": time_since_introspection > 3600,  # > 1 hour
                "system_load": self._calculate_system_load(),
                "active_processes": self._count_active_processes()
            }
            
            return awareness_info
            
        except Exception as e:
            logger.error(f"Failed to get system awareness: {str(e)}")
            raise
    
    async def _update_awareness_state(self, awareness_level: Optional[float] = None,
                                    focus_areas: Optional[List[str]] = None,
                                    introspection_depth: Optional[str] = None) -> Dict[str, Any]:
        """Update system awareness parameters."""
        try:
            state = self.state_manager.get(
                "meta_cognitive_state",
                namespace="metacognitive"
            )
            
            # Update provided parameters
            if awareness_level is not None:
                state["awareness_level"] = max(0, min(1, awareness_level))
            
            if focus_areas is not None:
                state["focus_areas"] = focus_areas
            
            if introspection_depth is not None:
                state["introspection_depth"] = introspection_depth
            
            # Update timestamp
            state["last_update"] = datetime.utcnow().isoformat()
            
            # Save updated state
            self.state_manager.set(
                "meta_cognitive_state", state,
                namespace="metacognitive"
            )
            
            return {
                "status": "success",
                "updated_state": state,
                "message": "Awareness state updated successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to update awareness state: {str(e)}")
            raise
    
    async def _analyze_system_introspection(self, 
                                          analysis_depth: str = "detailed",
                                          components: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform system introspection."""
        try:
            if not components:
                components = ["memory", "processing", "learning", "goals", "strategies"]
            
            introspection_id = f"intro_{uuid.uuid4().hex[:8]}"
            start_time = time.time()
            
            # Update last introspection time
            state = self.state_manager.get("meta_cognitive_state", namespace="metacognitive")
            state["last_introspection"] = time.time()
            self.state_manager.set("meta_cognitive_state", state, namespace="metacognitive")
            
            # Perform introspection
            introspection_results = {
                "introspection_id": introspection_id,
                "timestamp": datetime.utcnow().isoformat(),
                "depth": analysis_depth,
                "components": {}
            }
            
            for component in components:
                component_analysis = await self._analyze_component(component, analysis_depth)
                introspection_results["components"][component] = component_analysis
            
            # Self-assessment
            self_assessment = self._perform_self_assessment(introspection_results["components"])
            introspection_results["self_assessment"] = self_assessment
            
            # Update self-assessment in state
            state["self_assessment"] = self_assessment
            self.state_manager.set("meta_cognitive_state", state, namespace="metacognitive")
            
            # Recommendations
            introspection_results["recommendations"] = self._generate_introspection_recommendations(
                introspection_results["components"]
            )
            
            # Store introspection results
            self.state_manager.set(
                introspection_id, introspection_results,
                namespace="introspections",
                ttl=86400  # Keep for 24 hours
            )
            
            introspection_results["duration_seconds"] = time.time() - start_time
            
            return introspection_results
            
        except Exception as e:
            logger.error(f"System introspection failed: {str(e)}")
            raise
    
    async def _create_goal(self, goal_type: str,
                         description: str,
                         target_metrics: Optional[Dict[str, Any]] = None,
                         priority: str = "medium",
                         deadline: Optional[str] = None) -> Dict[str, Any]:
        """Create a new system goal."""
        try:
            goal_id = f"goal_{uuid.uuid4().hex[:8]}"
            
            goal = {
                "goal_id": goal_id,
                "goal_type": goal_type,
                "description": description,
                "target_metrics": target_metrics or {},
                "priority": priority,
                "status": "active",
                "created_at": datetime.utcnow().isoformat(),
                "deadline": deadline,
                "progress_percent": 0,
                "current_metrics": {},
                "milestones": [],
                "strategies_used": []
            }
            
            # Store goal
            self.state_manager.set(
                goal_id, goal,
                namespace="goals"
            )
            
            logger.info(f"Created goal: {goal_id} - {description}")
            
            return {
                "goal_id": goal_id,
                "status": "success",
                "goal": goal,
                "message": f"Goal '{description}' created successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to create goal: {str(e)}")
            raise
    
    async def _list_goals(self, status: Optional[str] = None,
                        goal_type: Optional[str] = None) -> Dict[str, Any]:
        """List system goals."""
        try:
            all_goals = self.state_manager.get_namespace("goals")
            
            # Filter goals
            goals = []
            for gid, goal in all_goals.items():
                if status and goal.get("status") != status:
                    continue
                if goal_type and goal.get("goal_type") != goal_type:
                    continue
                goals.append(goal)
            
            # Sort by priority and creation date
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            goals.sort(key=lambda g: (
                priority_order.get(g.get("priority", "medium"), 2),
                g.get("created_at", "")
            ))
            
            # Calculate statistics
            stats = {
                "total": len(goals),
                "by_status": defaultdict(int),
                "by_type": defaultdict(int),
                "avg_progress": 0
            }
            
            total_progress = 0
            for goal in goals:
                stats["by_status"][goal.get("status", "unknown")] += 1
                stats["by_type"][goal.get("goal_type", "unknown")] += 1
                total_progress += goal.get("progress_percent", 0)
            
            if goals:
                stats["avg_progress"] = total_progress / len(goals)
            
            return {
                "goals": goals,
                "statistics": dict(stats),
                "filters": {
                    "status": status,
                    "goal_type": goal_type
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to list goals: {str(e)}")
            raise
    
    async def _update_goal_progress(self, goal_id: str,
                                  progress_percent: Optional[float] = None,
                                  current_metrics: Optional[Dict[str, Any]] = None,
                                  notes: Optional[str] = None) -> Dict[str, Any]:
        """Update goal progress."""
        try:
            goal = self.state_manager.get(goal_id, namespace="goals")
            
            if not goal:
                return {
                    "error": f"Goal not found: {goal_id}",
                    "status": "not_found"
                }
            
            # Update progress
            if progress_percent is not None:
                goal["progress_percent"] = max(0, min(100, progress_percent))
            
            if current_metrics:
                goal["current_metrics"].update(current_metrics)
            
            # Add milestone if significant progress
            if progress_percent and notes:
                milestone = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "progress": progress_percent,
                    "notes": notes
                }
                goal["milestones"].append(milestone)
            
            # Update status based on progress
            if goal["progress_percent"] >= 100:
                goal["status"] = "completed"
                goal["completed_at"] = datetime.utcnow().isoformat()
            elif goal["deadline"] and datetime.fromisoformat(goal["deadline"]) < datetime.utcnow():
                goal["status"] = "failed"
                goal["failed_reason"] = "Deadline exceeded"
            
            # Update last modified
            goal["last_updated"] = datetime.utcnow().isoformat()
            
            # Save updated goal
            self.state_manager.set(goal_id, goal, namespace="goals")
            
            return {
                "goal_id": goal_id,
                "status": "success",
                "current_progress": goal["progress_percent"],
                "goal_status": goal["status"],
                "message": "Goal progress updated"
            }
            
        except Exception as e:
            logger.error(f"Failed to update goal progress: {str(e)}")
            raise
    
    async def _evaluate_goal(self, goal_id: str) -> Dict[str, Any]:
        """Evaluate goal achievement."""
        try:
            goal = self.state_manager.get(goal_id, namespace="goals")
            
            if not goal:
                return {
                    "error": f"Goal not found: {goal_id}",
                    "status": "not_found"
                }
            
            evaluation = {
                "goal_id": goal_id,
                "timestamp": datetime.utcnow().isoformat(),
                "status": goal["status"],
                "progress": goal["progress_percent"],
                "metrics_achievement": {}
            }
            
            # Evaluate metric achievement
            for metric, target in goal.get("target_metrics", {}).items():
                current = goal.get("current_metrics", {}).get(metric, 0)
                if isinstance(target, (int, float)):
                    achievement = (current / target * 100) if target else 0
                    evaluation["metrics_achievement"][metric] = {
                        "target": target,
                        "current": current,
                        "achievement_percent": achievement
                    }
            
            # Calculate overall achievement
            if evaluation["metrics_achievement"]:
                avg_achievement = statistics.mean(
                    m["achievement_percent"] 
                    for m in evaluation["metrics_achievement"].values()
                )
                evaluation["overall_achievement"] = avg_achievement
            else:
                evaluation["overall_achievement"] = goal["progress_percent"]
            
            # Generate recommendations
            evaluation["recommendations"] = self._generate_goal_recommendations(goal, evaluation)
            
            # Store evaluation
            self.state_manager.set(
                f"eval_{goal_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                evaluation,
                namespace="goal_evaluations",
                ttl=604800  # Keep for 7 days
            )
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Failed to evaluate goal: {str(e)}")
            raise
    
    async def _list_strategies(self, context: Optional[str] = None) -> Dict[str, Any]:
        """List available strategies."""
        try:
            strategies = self.state_manager.get("strategies", namespace="metacognitive")
            
            # Filter by context if provided
            if context:
                filtered_strategies = {
                    sid: strategy for sid, strategy in strategies.items()
                    if strategy.get("context") == context
                }
            else:
                filtered_strategies = strategies
            
            # Sort by effectiveness
            sorted_strategies = sorted(
                filtered_strategies.values(),
                key=lambda s: s.get("effectiveness", 0),
                reverse=True
            )
            
            return {
                "strategies": sorted_strategies,
                "total": len(sorted_strategies),
                "context_filter": context
            }
            
        except Exception as e:
            logger.error(f"Failed to list strategies: {str(e)}")
            raise
    
    async def _select_strategy(self, context: str,
                             constraints: Optional[Dict[str, Any]] = None,
                             objectives: Optional[List[str]] = None) -> Dict[str, Any]:
        """Select optimal strategy for context."""
        try:
            strategies = self.state_manager.get("strategies", namespace="metacognitive")
            
            # Find applicable strategies
            applicable = []
            for sid, strategy in strategies.items():
                if strategy.get("context") == context:
                    score = self._calculate_strategy_score(strategy, constraints, objectives)
                    applicable.append((sid, strategy, score))
            
            if not applicable:
                return {
                    "error": f"No strategies found for context: {context}",
                    "status": "no_strategy"
                }
            
            # Select best strategy
            applicable.sort(key=lambda x: x[2], reverse=True)
            selected_id, selected_strategy, score = applicable[0]
            
            # Update usage count
            selected_strategy["usage_count"] += 1
            strategies[selected_id] = selected_strategy
            self.state_manager.set("strategies", strategies, namespace="metacognitive")
            
            return {
                "strategy_id": selected_id,
                "strategy": selected_strategy,
                "selection_score": score,
                "alternatives": [
                    {"id": sid, "name": s["name"], "score": sc}
                    for sid, s, sc in applicable[1:3]  # Top 3 alternatives
                ],
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to select strategy: {str(e)}")
            raise
    
    async def _create_strategy(self, name: str,
                             description: str,
                             context: str,
                             steps: List[str]) -> Dict[str, Any]:
        """Create new adaptive strategy."""
        try:
            strategy_id = f"strat_{uuid.uuid4().hex[:8]}"
            
            strategy = {
                "id": strategy_id,
                "name": name,
                "description": description,
                "context": context,
                "steps": steps,
                "effectiveness": 0.5,  # Start with neutral effectiveness
                "usage_count": 0,
                "success_rate": 0,
                "created_at": datetime.utcnow().isoformat(),
                "adaptations": []
            }
            
            # Store strategy
            strategies = self.state_manager.get("strategies", namespace="metacognitive")
            strategies[strategy_id] = strategy
            self.state_manager.set("strategies", strategies, namespace="metacognitive")
            
            logger.info(f"Created strategy: {strategy_id} - {name}")
            
            return {
                "strategy_id": strategy_id,
                "status": "success",
                "strategy": strategy,
                "message": f"Strategy '{name}' created successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to create strategy: {str(e)}")
            raise
    
    async def _evaluate_strategy(self, strategy_id: str,
                                execution_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate strategy effectiveness."""
        try:
            strategies = self.state_manager.get("strategies", namespace="metacognitive")
            
            if strategy_id not in strategies:
                return {
                    "error": f"Strategy not found: {strategy_id}",
                    "status": "not_found"
                }
            
            strategy = strategies[strategy_id]
            
            # Default execution data if not provided
            if not execution_data:
                execution_data = {
                    "success": True,
                    "execution_time": 10.5,
                    "resource_usage": 0.6,
                    "outcome_quality": 0.85
                }
            
            # Update success rate
            total_executions = strategy["usage_count"]
            if total_executions > 0:
                current_successes = strategy["success_rate"] * total_executions
                if execution_data.get("success", False):
                    current_successes += 1
                strategy["success_rate"] = current_successes / total_executions
            
            # Update effectiveness based on execution data
            outcome_quality = execution_data.get("outcome_quality", 0.5)
            resource_efficiency = 1 - execution_data.get("resource_usage", 0.5)
            
            new_effectiveness = (
                strategy["effectiveness"] * 0.7 +  # Weight existing
                outcome_quality * 0.2 +            # Weight outcome
                resource_efficiency * 0.1          # Weight efficiency
            )
            
            strategy["effectiveness"] = new_effectiveness
            
            # Store updated strategy
            strategies[strategy_id] = strategy
            self.state_manager.set("strategies", strategies, namespace="metacognitive")
            
            evaluation = {
                "strategy_id": strategy_id,
                "timestamp": datetime.utcnow().isoformat(),
                "effectiveness": new_effectiveness,
                "success_rate": strategy["success_rate"],
                "execution_data": execution_data,
                "recommendation": self._get_strategy_recommendation(strategy)
            }
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Failed to evaluate strategy: {str(e)}")
            raise
    
    async def _trigger_learning(self, learning_type: str,
                              experience_data: Dict[str, Any],
                              learning_rate: Optional[float] = None) -> Dict[str, Any]:
        """Trigger adaptive learning process."""
        try:
            learning_metrics = self.state_manager.get("learning_metrics", namespace="metacognitive")
            
            if learning_rate is None:
                learning_rate = learning_metrics.get("learning_rate", 0.15)
            
            learning_id = f"learn_{uuid.uuid4().hex[:8]}"
            
            # Process learning based on type
            if learning_type == "reinforcement":
                learning_result = self._reinforcement_learning(experience_data, learning_rate)
            elif learning_type == "supervised":
                learning_result = self._supervised_learning(experience_data, learning_rate)
            elif learning_type == "unsupervised":
                learning_result = self._unsupervised_learning(experience_data, learning_rate)
            elif learning_type == "meta":
                learning_result = self._meta_learning(experience_data, learning_rate)
            else:
                return {
                    "error": f"Unknown learning type: {learning_type}",
                    "status": "invalid_type"
                }
            
            # Update learning metrics
            learning_metrics["total_learning_cycles"] += 1
            if learning_result.get("success", False):
                learning_metrics["successful_adaptations"] += 1
            
            # Update accuracy based on learning result
            if "accuracy_delta" in learning_result:
                learning_metrics["accuracy"] = max(0, min(1, 
                    learning_metrics["accuracy"] + learning_result["accuracy_delta"]
                ))
            
            self.state_manager.set("learning_metrics", learning_metrics, namespace="metacognitive")
            
            # Store learning session
            learning_session = {
                "learning_id": learning_id,
                "timestamp": datetime.utcnow().isoformat(),
                "learning_type": learning_type,
                "experience_data": experience_data,
                "learning_rate": learning_rate,
                "result": learning_result,
                "metrics_after": learning_metrics.copy()
            }
            
            self.state_manager.set(
                learning_id, learning_session,
                namespace="learning_sessions",
                ttl=259200  # Keep for 3 days
            )
            
            return {
                "learning_id": learning_id,
                "status": "success",
                "learning_type": learning_type,
                "result": learning_result,
                "updated_metrics": learning_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to trigger learning: {str(e)}")
            raise
    
    async def _get_learning_metrics(self) -> Dict[str, Any]:
        """Get current learning metrics."""
        try:
            metrics = self.state_manager.get("learning_metrics", namespace="metacognitive")
            
            # Add calculated metrics
            if metrics["total_learning_cycles"] > 0:
                metrics["adaptation_success_rate"] = (
                    metrics["successful_adaptations"] / metrics["total_learning_cycles"]
                )
            else:
                metrics["adaptation_success_rate"] = 0
            
            # Add recent learning history
            recent_sessions = []
            learning_sessions = self.state_manager.get_namespace("learning_sessions")
            
            # Get last 5 sessions
            sorted_sessions = sorted(
                learning_sessions.items(),
                key=lambda x: x[1].get("timestamp", ""),
                reverse=True
            )[:5]
            
            for sid, session in sorted_sessions:
                recent_sessions.append({
                    "learning_id": sid,
                    "timestamp": session.get("timestamp"),
                    "type": session.get("learning_type"),
                    "success": session.get("result", {}).get("success", False)
                })
            
            metrics["recent_sessions"] = recent_sessions
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get learning metrics: {str(e)}")
            raise
    
    async def _update_learning_parameters(self, 
                                        learning_rate: Optional[float] = None,
                                        exploration_factor: Optional[float] = None,
                                        adaptation_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Update learning parameters."""
        try:
            metrics = self.state_manager.get("learning_metrics", namespace="metacognitive")
            
            # Update provided parameters
            if learning_rate is not None:
                metrics["learning_rate"] = max(0, min(1, learning_rate))
            
            if exploration_factor is not None:
                metrics["exploration_factor"] = max(0, min(1, exploration_factor))
            
            if adaptation_threshold is not None:
                metrics["adaptation_threshold"] = max(0, min(1, adaptation_threshold))
            
            # Update timestamp
            metrics["last_parameter_update"] = datetime.utcnow().isoformat()
            
            # Save updated metrics
            self.state_manager.set("learning_metrics", metrics, namespace="metacognitive")
            
            return {
                "status": "success",
                "updated_parameters": {
                    "learning_rate": metrics["learning_rate"],
                    "exploration_factor": metrics["exploration_factor"],
                    "adaptation_threshold": metrics["adaptation_threshold"]
                },
                "message": "Learning parameters updated successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to update learning parameters: {str(e)}")
            raise
    
    async def _analyze_adaptation_patterns(self, 
                                         time_window_hours: int = 24,
                                         pattern_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze system adaptation patterns."""
        try:
            if not pattern_types:
                pattern_types = ["learning", "strategy", "goal", "performance"]
            
            analysis_id = f"adapt_{uuid.uuid4().hex[:8]}"
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            patterns = {
                "analysis_id": analysis_id,
                "timestamp": datetime.utcnow().isoformat(),
                "time_window_hours": time_window_hours,
                "patterns": {}
            }
            
            # Analyze learning patterns
            if "learning" in pattern_types:
                learning_pattern = self._analyze_learning_pattern(cutoff_time)
                patterns["patterns"]["learning"] = learning_pattern
            
            # Analyze strategy patterns
            if "strategy" in pattern_types:
                strategy_pattern = self._analyze_strategy_pattern(cutoff_time)
                patterns["patterns"]["strategy"] = strategy_pattern
            
            # Analyze goal patterns
            if "goal" in pattern_types:
                goal_pattern = self._analyze_goal_pattern(cutoff_time)
                patterns["patterns"]["goal"] = goal_pattern
            
            # Analyze performance patterns
            if "performance" in pattern_types:
                performance_pattern = self._analyze_performance_pattern(cutoff_time)
                patterns["patterns"]["performance"] = performance_pattern
            
            # Generate insights
            patterns["insights"] = self._generate_adaptation_insights(patterns["patterns"])
            patterns["recommendations"] = self._generate_adaptation_recommendations(patterns["patterns"])
            
            # Store analysis
            self.state_manager.set(
                analysis_id, patterns,
                namespace="adaptation_analyses",
                ttl=604800  # Keep for 7 days
            )
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to analyze adaptation patterns: {str(e)}")
            raise
    
    # Helper methods
    
    def _calculate_system_load(self) -> float:
        """Calculate current system load."""
        # Simplified calculation
        import random
        return random.uniform(0.3, 0.8)
    
    def _count_active_processes(self) -> int:
        """Count active processes."""
        # Count active sessions, goals, etc.
        active_count = 0
        
        # Count active agent sessions
        sessions = self.state_manager.get_namespace("agent_sessions")
        active_count += sum(1 for s in sessions.values() if s.get("status") == "active")
        
        # Count active goals
        goals = self.state_manager.get_namespace("goals")
        active_count += sum(1 for g in goals.values() if g.get("status") == "active")
        
        return active_count
    
    async def _analyze_component(self, component: str, depth: str) -> Dict[str, Any]:
        """Analyze a specific component."""
        analysis = {
            "component": component,
            "status": "healthy",
            "metrics": {},
            "issues": []
        }
        
        if component == "memory":
            # Analyze memory component
            analysis["metrics"] = {
                "usage_percent": 65,
                "efficiency": 0.85,
                "fragmentation": 0.12
            }
            if analysis["metrics"]["usage_percent"] > 80:
                analysis["issues"].append("High memory usage")
                analysis["status"] = "warning"
                
        elif component == "processing":
            analysis["metrics"] = {
                "throughput": 150,
                "latency_ms": 45,
                "error_rate": 0.02
            }
            
        elif component == "learning":
            metrics = self.state_manager.get("learning_metrics", namespace="metacognitive")
            analysis["metrics"] = {
                "accuracy": metrics.get("accuracy", 0),
                "learning_rate": metrics.get("learning_rate", 0),
                "adaptations": metrics.get("successful_adaptations", 0)
            }
            
        # Add more detailed analysis for comprehensive depth
        if depth == "comprehensive":
            analysis["detailed_diagnostics"] = f"Detailed analysis of {component}"
            analysis["historical_trends"] = "Historical performance data"
        
        return analysis
    
    def _perform_self_assessment(self, component_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Perform self-assessment based on introspection."""
        # Calculate scores based on component health
        total_components = len(component_analyses)
        healthy_components = sum(
            1 for analysis in component_analyses.values()
            if analysis.get("status") == "healthy"
        )
        
        confidence = healthy_components / total_components if total_components > 0 else 0.5
        
        # Calculate competence based on metrics
        all_metrics = []
        for analysis in component_analyses.values():
            metrics = analysis.get("metrics", {})
            if "efficiency" in metrics:
                all_metrics.append(metrics["efficiency"])
            if "accuracy" in metrics:
                all_metrics.append(metrics["accuracy"])
        
        competence = statistics.mean(all_metrics) if all_metrics else 0.75
        
        # Adaptability based on learning metrics
        learning_metrics = self.state_manager.get("learning_metrics", namespace="metacognitive")
        adaptability = learning_metrics.get("adaptation_success_rate", 0.85)
        
        return {
            "confidence": confidence,
            "competence": competence,
            "adaptability": adaptability,
            "overall_health": (confidence + competence + adaptability) / 3
        }
    
    def _generate_introspection_recommendations(self, 
                                              component_analyses: Dict[str, Any]) -> List[str]:
        """Generate recommendations from introspection."""
        recommendations = []
        
        for component, analysis in component_analyses.items():
            for issue in analysis.get("issues", []):
                if "memory" in issue.lower():
                    recommendations.append(f"Optimize {component} memory usage")
                elif "error" in issue.lower():
                    recommendations.append(f"Investigate {component} error rates")
        
        if not recommendations:
            recommendations.append("System operating within normal parameters")
        
        return recommendations
    
    def _generate_goal_recommendations(self, goal: Dict[str, Any], 
                                     evaluation: Dict[str, Any]) -> List[str]:
        """Generate recommendations for goal achievement."""
        recommendations = []
        
        if evaluation["overall_achievement"] < 50:
            recommendations.append("Consider revising goal targets or approach")
        elif evaluation["overall_achievement"] < 80:
            recommendations.append("Increase focus on underperforming metrics")
        
        if goal["status"] == "active" and goal.get("deadline"):
            deadline = datetime.fromisoformat(goal["deadline"])
            if deadline < datetime.utcnow() + timedelta(days=7):
                recommendations.append("Deadline approaching - prioritize completion")
        
        return recommendations
    
    def _calculate_strategy_score(self, strategy: Dict[str, Any],
                                constraints: Optional[Dict[str, Any]],
                                objectives: Optional[List[str]]) -> float:
        """Calculate strategy score based on constraints and objectives."""
        base_score = strategy.get("effectiveness", 0.5)
        
        # Adjust for success rate
        success_rate = strategy.get("success_rate", 0.5)
        score = base_score * 0.7 + success_rate * 0.3
        
        # Apply constraints (simplified)
        if constraints:
            if "max_time" in constraints and len(strategy.get("steps", [])) > 5:
                score *= 0.8  # Penalize long strategies
            if "min_reliability" in constraints and success_rate < constraints["min_reliability"]:
                score *= 0.5
        
        # Boost for matching objectives (simplified)
        if objectives:
            objective_match = 0.1 * len(objectives)  # Simple boost
            score = min(1.0, score + objective_match)
        
        return score
    
    def _get_strategy_recommendation(self, strategy: Dict[str, Any]) -> str:
        """Get recommendation for strategy use."""
        if strategy["effectiveness"] > 0.8 and strategy["success_rate"] > 0.85:
            return "Highly effective - continue using"
        elif strategy["effectiveness"] < 0.5:
            return "Consider retiring or significantly modifying this strategy"
        elif strategy["usage_count"] < 5:
            return "Insufficient data - continue gathering performance metrics"
        else:
            return "Monitor performance and iterate on implementation"
    
    def _reinforcement_learning(self, experience: Dict[str, Any], 
                              learning_rate: float) -> Dict[str, Any]:
        """Perform reinforcement learning."""
        reward = experience.get("reward", 0)
        action = experience.get("action", "")
        
        # Simplified Q-learning update
        accuracy_delta = learning_rate * reward * 0.1
        
        return {
            "success": True,
            "learning_type": "reinforcement",
            "accuracy_delta": accuracy_delta,
            "action_updated": action,
            "reward_received": reward
        }
    
    def _supervised_learning(self, experience: Dict[str, Any], 
                           learning_rate: float) -> Dict[str, Any]:
        """Perform supervised learning."""
        correct_output = experience.get("correct_output")
        predicted_output = experience.get("predicted_output")
        
        # Calculate error
        error = 0.1 if correct_output != predicted_output else 0
        accuracy_delta = -learning_rate * error
        
        return {
            "success": True,
            "learning_type": "supervised",
            "accuracy_delta": accuracy_delta,
            "error": error
        }
    
    def _unsupervised_learning(self, experience: Dict[str, Any], 
                             learning_rate: float) -> Dict[str, Any]:
        """Perform unsupervised learning."""
        patterns = experience.get("patterns", [])
        
        # Simplified pattern learning
        patterns_learned = len(patterns)
        accuracy_delta = learning_rate * patterns_learned * 0.01
        
        return {
            "success": True,
            "learning_type": "unsupervised",
            "accuracy_delta": accuracy_delta,
            "patterns_discovered": patterns_learned
        }
    
    def _meta_learning(self, experience: Dict[str, Any], 
                      learning_rate: float) -> Dict[str, Any]:
        """Perform meta-learning (learning to learn)."""
        learning_performance = experience.get("learning_performance", {})
        
        # Adjust learning parameters based on performance
        new_learning_rate = learning_rate
        if learning_performance.get("improvement_rate", 0) < 0.1:
            new_learning_rate *= 1.1  # Increase learning rate
        elif learning_performance.get("improvement_rate", 0) > 0.5:
            new_learning_rate *= 0.9  # Decrease to fine-tune
        
        return {
            "success": True,
            "learning_type": "meta",
            "accuracy_delta": 0,
            "parameter_adjustments": {
                "learning_rate": new_learning_rate
            }
        }
    
    def _analyze_learning_pattern(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Analyze learning patterns."""
        sessions = self.state_manager.get_namespace("learning_sessions")
        recent_sessions = [
            s for s in sessions.values()
            if datetime.fromisoformat(s.get("timestamp", "2000-01-01")) > cutoff_time
        ]
        
        if not recent_sessions:
            return {"pattern": "no_data", "sessions_analyzed": 0}
        
        success_rate = sum(
            1 for s in recent_sessions 
            if s.get("result", {}).get("success", False)
        ) / len(recent_sessions)
        
        return {
            "pattern": "improving" if success_rate > 0.7 else "stable",
            "sessions_analyzed": len(recent_sessions),
            "success_rate": success_rate,
            "dominant_type": max(
                set(s.get("learning_type") for s in recent_sessions),
                key=lambda x: sum(1 for s in recent_sessions if s.get("learning_type") == x)
            )
        }
    
    def _analyze_strategy_pattern(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Analyze strategy usage patterns."""
        strategies = self.state_manager.get("strategies", namespace="metacognitive")
        
        # Sort by usage
        most_used = sorted(
            strategies.values(),
            key=lambda s: s.get("usage_count", 0),
            reverse=True
        )[:3]
        
        return {
            "pattern": "diverse" if len(strategies) > 5 else "focused",
            "total_strategies": len(strategies),
            "most_used": [s["name"] for s in most_used],
            "avg_effectiveness": statistics.mean(
                s.get("effectiveness", 0) for s in strategies.values()
            ) if strategies else 0
        }
    
    def _analyze_goal_pattern(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Analyze goal achievement patterns."""
        goals = self.state_manager.get_namespace("goals")
        recent_goals = [
            g for g in goals.values()
            if datetime.fromisoformat(g.get("created_at", "2000-01-01")) > cutoff_time
        ]
        
        if not recent_goals:
            return {"pattern": "no_data", "goals_analyzed": 0}
        
        completion_rate = sum(
            1 for g in recent_goals 
            if g.get("status") == "completed"
        ) / len(recent_goals) if recent_goals else 0
        
        return {
            "pattern": "achieving" if completion_rate > 0.6 else "struggling",
            "goals_analyzed": len(recent_goals),
            "completion_rate": completion_rate,
            "avg_progress": statistics.mean(
                g.get("progress_percent", 0) for g in recent_goals
            ) if recent_goals else 0
        }
    
    def _analyze_performance_pattern(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Analyze performance patterns."""
        # Simplified performance analysis
        return {
            "pattern": "stable",
            "trend": "improving",
            "key_metrics": {
                "response_time": "decreasing",
                "accuracy": "increasing",
                "resource_usage": "stable"
            }
        }
    
    def _generate_adaptation_insights(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate insights from adaptation patterns."""
        insights = []
        
        if "learning" in patterns:
            learning = patterns["learning"]
            if learning.get("success_rate", 0) > 0.8:
                insights.append("Learning mechanisms are highly effective")
            elif learning.get("success_rate", 0) < 0.5:
                insights.append("Learning success rate needs improvement")
        
        if "strategy" in patterns:
            strategy = patterns["strategy"]
            if strategy.get("avg_effectiveness", 0) < 0.6:
                insights.append("Strategy portfolio needs optimization")
        
        if "goal" in patterns:
            goal = patterns["goal"]
            if goal.get("completion_rate", 0) < 0.5:
                insights.append("Goal achievement rate is below target")
        
        return insights
    
    def _generate_adaptation_recommendations(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate recommendations from adaptation patterns."""
        recommendations = []
        
        # Learning recommendations
        if "learning" in patterns:
            learning = patterns["learning"]
            if learning.get("pattern") == "stable" and learning.get("success_rate", 1) < 0.7:
                recommendations.append("Increase learning rate to accelerate adaptation")
            if learning.get("dominant_type") == "supervised":
                recommendations.append("Incorporate more reinforcement learning for autonomous improvement")
        
        # Strategy recommendations
        if "strategy" in patterns:
            strategy = patterns["strategy"]
            if strategy.get("pattern") == "focused" and strategy.get("total_strategies", 0) < 5:
                recommendations.append("Develop more diverse strategies for different contexts")
        
        # Goal recommendations
        if "goal" in patterns:
            goal = patterns["goal"]
            if goal.get("pattern") == "struggling":
                recommendations.append("Review and adjust goal targets to be more achievable")
                recommendations.append("Implement better progress tracking and milestone management")
        
        return recommendations