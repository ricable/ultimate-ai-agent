"""
Self-Awareness Module for system state monitoring and self-reflection.

This module provides introspective monitoring capabilities that allow the system
to observe its own internal processes and reflect on its behavior and performance.
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from collections import deque
import numpy as np
import psutil


logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """Represents the current state of the system for self-awareness monitoring."""
    timestamp: float
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    active_goals: List[str] = field(default_factory=list)
    current_strategies: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    introspection_data: Dict[str, Any] = field(default_factory=dict)


class SelfAwarenessModule:
    """
    Self-Awareness Module for system state monitoring and self-reflection.
    
    This module provides introspective monitoring capabilities that allow the system
    to observe its own internal processes and reflect on its behavior and performance.
    """
    
    def __init__(self):
        self.observation_points: List[str] = []
        self.introspection_enabled: bool = True
        self.introspection_data: Dict[str, Any] = {}
        self.monitoring_active: bool = False
        self.state_history: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    def get_current_state(self) -> SystemState:
        """Get the current system state."""
        with self._lock:
            # Get system resource usage
            memory_usage = psutil.virtual_memory().percent / 100.0
            cpu_usage = psutil.cpu_percent() / 100.0
            
            # Create system state
            state = SystemState(
                timestamp=time.time(),
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                introspection_data=self.introspection_data.copy()
            )
            
            # Store in history
            self.state_history.append(state)
            
            return state
    
    def configure_observation_points(self, observation_points: List[str]):
        """Configure the observation points for introspective monitoring."""
        with self._lock:
            self.observation_points = observation_points.copy()
    
    def add_observation_points(self, additional_points: List[str]):
        """Add new observation points to the existing configuration."""
        with self._lock:
            self.observation_points.extend(additional_points)
    
    def remove_observation_points(self, points_to_remove: List[str]):
        """Remove observation points from the configuration."""
        with self._lock:
            self.observation_points = [
                point for point in self.observation_points 
                if point not in points_to_remove
            ]
    
    def start_introspection(self):
        """Start introspective monitoring."""
        with self._lock:
            self.monitoring_active = True
            self.introspection_data.clear()
    
    def observe_internal_process(self, process_name: str, process_data: Dict[str, Any]):
        """Observe and record internal process data."""
        if not self.introspection_enabled or process_name not in self.observation_points:
            return
        
        with self._lock:
            if process_name not in self.introspection_data:
                self.introspection_data[process_name] = []
            
            observation = {
                'timestamp': time.time(),
                'data': process_data.copy()
            }
            self.introspection_data[process_name].append(observation)
            
            # Keep only recent observations
            if len(self.introspection_data[process_name]) > 100:
                self.introspection_data[process_name] = self.introspection_data[process_name][-100:]
    
    def get_introspection_data(self) -> Dict[str, Any]:
        """Get the current introspection data."""
        with self._lock:
            # Return the latest data for each process
            latest_data = {}
            for process_name, observations in self.introspection_data.items():
                if observations:
                    latest_data[process_name] = observations[-1]['data']
            return latest_data
    
    def reflect_on_performance(self, current_state: SystemState, historical_states: List[SystemState]):
        """Perform self-reflection on system behavior and performance."""
        reflection_result = type('ReflectionResult', (), {})()
        
        # Analyze trends
        trends_identified = []
        if len(historical_states) >= 2:
            # Memory usage trend
            memory_values = [state.memory_usage for state in historical_states] + [current_state.memory_usage]
            if len(memory_values) >= 3:
                recent_trend = np.mean(np.diff(memory_values[-3:]))
                if recent_trend > 0.05:
                    trends_identified.append('increasing_memory_usage')
                elif recent_trend < -0.05:
                    trends_identified.append('decreasing_memory_usage')
            
            # CPU usage trend
            cpu_values = [state.cpu_usage for state in historical_states] + [current_state.cpu_usage]
            if len(cpu_values) >= 3:
                recent_trend = np.mean(np.diff(cpu_values[-3:]))
                if recent_trend > 0.05:
                    trends_identified.append('increasing_cpu_usage')
                elif recent_trend < -0.05:
                    trends_identified.append('decreasing_cpu_usage')
        
        # Detect anomalies
        anomalies_detected = []
        if current_state.memory_usage > 0.9:
            anomalies_detected.append('high_memory_usage')
        if current_state.cpu_usage > 0.9:
            anomalies_detected.append('high_cpu_usage')
        
        # Identify improvement opportunities
        improvement_opportunities = []
        if current_state.memory_usage > 0.8:
            improvement_opportunities.append('memory_optimization')
        if current_state.cpu_usage > 0.8:
            improvement_opportunities.append('cpu_optimization')
        
        # Calculate confidence score
        confidence_score = min(1.0, len(historical_states) / 10.0)
        
        reflection_result.trends_identified = trends_identified
        reflection_result.anomalies_detected = anomalies_detected
        reflection_result.improvement_opportunities = improvement_opportunities
        reflection_result.confidence_score = confidence_score
        
        return reflection_result
    
    def get_state_history(self, limit: Optional[int] = None) -> List[SystemState]:
        """Get the state history with optional limit."""
        with self._lock:
            if limit:
                return list(self.state_history)[-limit:]
            return list(self.state_history)
    
    def clear_history(self):
        """Clear the state history."""
        with self._lock:
            self.state_history.clear()
    
    def enable_introspection(self):
        """Enable introspection monitoring."""
        with self._lock:
            self.introspection_enabled = True
    
    def disable_introspection(self):
        """Disable introspection monitoring."""
        with self._lock:
            self.introspection_enabled = False
    
    def is_monitoring_active(self) -> bool:
        """Check if monitoring is currently active."""
        with self._lock:
            return self.monitoring_active
    
    def stop_introspection(self):
        """Stop introspective monitoring."""
        with self._lock:
            self.monitoring_active = False
    
    def get_resource_usage_summary(self) -> Dict[str, float]:
        """Get a summary of current resource usage."""
        with self._lock:
            current_state = self.get_current_state()
            history = list(self.state_history)[-10:] if self.state_history else []
            
            if not history:
                return {
                    'current_memory': current_state.memory_usage,
                    'current_cpu': current_state.cpu_usage,
                    'avg_memory': current_state.memory_usage,
                    'avg_cpu': current_state.cpu_usage,
                    'max_memory': current_state.memory_usage,
                    'max_cpu': current_state.cpu_usage
                }
            
            memory_values = [s.memory_usage for s in history]
            cpu_values = [s.cpu_usage for s in history]
            
            return {
                'current_memory': current_state.memory_usage,
                'current_cpu': current_state.cpu_usage,
                'avg_memory': np.mean(memory_values),
                'avg_cpu': np.mean(cpu_values),
                'max_memory': max(memory_values),
                'max_cpu': max(cpu_values)
            }
    
    def analyze_observation_coverage(self) -> Dict[str, Any]:
        """Analyze the coverage of observation points."""
        with self._lock:
            total_points = len(self.observation_points)
            observed_points = len(self.introspection_data)
            
            unobserved_points = [
                point for point in self.observation_points
                if point not in self.introspection_data
            ]
            
            coverage = observed_points / total_points if total_points > 0 else 0.0
            
            return {
                'total_observation_points': total_points,
                'observed_points': observed_points,
                'unobserved_points': unobserved_points,
                'coverage_percentage': coverage * 100,
                'observation_frequency': {
                    name: len(observations)
                    for name, observations in self.introspection_data.items()
                }
            }
    
    def detect_anomalies(self, threshold: float = 0.9) -> List[str]:
        """Detect anomalies in system state."""
        with self._lock:
            anomalies = []
            current_state = self.get_current_state()
            
            if current_state.memory_usage > threshold:
                anomalies.append(f'high_memory_usage:{current_state.memory_usage:.2f}')
            if current_state.cpu_usage > threshold:
                anomalies.append(f'high_cpu_usage:{current_state.cpu_usage:.2f}')
            
            # Check for rapid changes
            if len(self.state_history) >= 2:
                prev_state = self.state_history[-2]
                memory_change = abs(current_state.memory_usage - prev_state.memory_usage)
                cpu_change = abs(current_state.cpu_usage - prev_state.cpu_usage)
                
                if memory_change > 0.3:
                    anomalies.append(f'rapid_memory_change:{memory_change:.2f}')
                if cpu_change > 0.3:
                    anomalies.append(f'rapid_cpu_change:{cpu_change:.2f}')
            
            return anomalies
    
    def get_introspection_summary(self) -> Dict[str, Any]:
        """Get a summary of introspection data."""
        with self._lock:
            summary = {
                'enabled': self.introspection_enabled,
                'active': self.monitoring_active,
                'observation_points': len(self.observation_points),
                'processes_observed': len(self.introspection_data),
                'total_observations': sum(
                    len(observations) for observations in self.introspection_data.values()
                ),
                'state_history_size': len(self.state_history)
            }
            return summary