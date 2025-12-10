"""
Goal Management and Performance Monitoring systems.

This module provides goal tracking, performance metrics collection,
and dashboard capabilities for the meta-cognitive engine.
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class Goal:
    """Represents a goal in the goal management system."""
    id: str
    description: str = ""
    priority: float = 0.5
    target_metrics: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    status: str = "active"
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)


@dataclass
class PerformanceMetrics:
    """Represents performance metrics for monitoring."""
    timestamp: float
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    throughput: float = 0.0
    latency: float = 0.0
    error_rate: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class GoalManager:
    """
    Goal Management system for dynamic goal setting, tracking, and adaptation.
    
    This module manages hierarchical goals with priority management and conflict resolution.
    """
    
    def __init__(self):
        self.active_goals: Dict[str, Goal] = {}
        self.goal_hierarchy: Dict[str, Dict[str, Any]] = {}
        self.priority_manager = self._create_priority_manager()
        self._lock = threading.Lock()
    
    def _create_priority_manager(self):
        """Create a priority manager for goal conflict resolution."""
        return type('PriorityManager', (), {
            'resolve_conflicts': lambda conflicts: self._resolve_priority_conflicts(conflicts)
        })()
    
    def create_goal(self, goal: Goal) -> str:
        """Create a new goal and add it to the active goals."""
        with self._lock:
            self.active_goals[goal.id] = goal
            
            # Update hierarchy
            if goal.dependencies:
                for dep_id in goal.dependencies:
                    if dep_id not in self.goal_hierarchy:
                        self.goal_hierarchy[dep_id] = {'children': [], 'goal': None}
                    self.goal_hierarchy[dep_id]['children'].append(goal.id)
            
            if goal.id not in self.goal_hierarchy:
                self.goal_hierarchy[goal.id] = {'children': [], 'goal': goal}
            else:
                self.goal_hierarchy[goal.id]['goal'] = goal
            
            return goal.id
    
    def get_goal_hierarchy(self) -> Dict[str, Dict[str, Any]]:
        """Get the current goal hierarchy."""
        with self._lock:
            return self.goal_hierarchy.copy()
    
    def detect_goal_conflicts(self) -> List[Dict[str, Any]]:
        """Detect conflicts between active goals."""
        conflicts = []
        
        with self._lock:
            goals = list(self.active_goals.values())
            
            for i, goal1 in enumerate(goals):
                for goal2 in goals[i+1:]:
                    # Check for resource conflicts
                    if self._has_resource_conflict(goal1, goal2):
                        conflicts.append({
                            'type': 'resource_conflict',
                            'goals': [goal1.id, goal2.id],
                            'conflicting_resources': self._get_conflicting_resources(goal1, goal2)
                        })
        
        return conflicts
    
    def _has_resource_conflict(self, goal1: Goal, goal2: Goal) -> bool:
        """Check if two goals have resource conflicts."""
        total_cpu = goal1.resource_requirements.get('cpu', 0) + goal2.resource_requirements.get('cpu', 0)
        total_memory = goal1.resource_requirements.get('memory', 0) + goal2.resource_requirements.get('memory', 0)
        
        return total_cpu > 1.0 or total_memory > 1.0
    
    def _get_conflicting_resources(self, goal1: Goal, goal2: Goal) -> List[str]:
        """Get the list of conflicting resources between two goals."""
        conflicts = []
        
        total_cpu = goal1.resource_requirements.get('cpu', 0) + goal2.resource_requirements.get('cpu', 0)
        total_memory = goal1.resource_requirements.get('memory', 0) + goal2.resource_requirements.get('memory', 0)
        
        if total_cpu > 1.0:
            conflicts.append('cpu')
        if total_memory > 1.0:
            conflicts.append('memory')
        
        return conflicts
    
    def resolve_conflicts(self, conflicts: List[Dict[str, Any]]):
        """Resolve conflicts between goals."""
        resolution = type('ConflictResolution', (), {})()
        
        resolution.resolution_strategy = 'priority_based'
        resolution.adjusted_priorities = {}
        resolution.resource_allocation = {}
        
        # Simple priority-based resolution
        for conflict in conflicts:
            if conflict['type'] == 'resource_conflict':
                goal_ids = conflict['goals']
                goals = [self.active_goals[gid] for gid in goal_ids if gid in self.active_goals]
                
                # Sort by priority
                goals.sort(key=lambda g: g.priority, reverse=True)
                
                # Adjust resource allocation
                for i, goal in enumerate(goals):
                    scale_factor = 1.0 / (i + 1)  # Higher priority gets more resources
                    resolution.resource_allocation[goal.id] = {
                        'cpu': goal.resource_requirements.get('cpu', 0) * scale_factor,
                        'memory': goal.resource_requirements.get('memory', 0) * scale_factor
                    }
        
        return resolution
    
    def _resolve_priority_conflicts(self, conflicts):
        """Internal method to resolve priority conflicts."""
        return self.resolve_conflicts(conflicts)
    
    def update_goal_progress(self, goal_id: str, progress_update: Dict[str, Any]):
        """Update the progress of a specific goal."""
        with self._lock:
            if goal_id not in self.active_goals:
                return
            
            goal = self.active_goals[goal_id]
            
            # Update progress based on target metrics
            if 'test_metric' in progress_update and 'test_metric' in goal.target_metrics:
                target_value = goal.target_metrics['test_metric']
                current_value = progress_update['test_metric']
                goal.progress = min(100.0, (current_value / target_value) * 100.0)