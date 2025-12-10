# SAFLA Examples Guide

## Overview

This guide provides practical examples and use cases for implementing SAFLA (Self-Aware Feedback Loop Algorithm) in real-world applications. Each example includes complete code implementations, configuration, and explanations.

## Table of Contents

1. [Basic SAFLA System](#basic-safla-system)
2. [Machine Learning Model Optimization](#machine-learning-model-optimization)
3. [Web Service Auto-Scaling](#web-service-auto-scaling)
4. [Distributed System Coordination](#distributed-system-coordination)
5. [Real-time Data Processing](#real-time-data-processing)
6. [Autonomous Trading System](#autonomous-trading-system)
7. [IoT Device Management](#iot-device-management)
8. [Content Recommendation Engine](#content-recommendation-engine)

## Basic SAFLA System

### Simple Autonomous System

This example demonstrates a basic SAFLA implementation for a simple autonomous system.

```python
import asyncio
import time
import random
from typing import Dict, Any
from safla.core.meta_cognitive_engine import (
    MetaCognitiveEngine, 
    SystemState, 
    Goal, 
    Strategy,
    PerformanceMetrics
)
from safla.core.delta_evaluation import DeltaEvaluator, AdaptiveWeights
from safla.core.hybrid_memory import HybridMemoryArchitecture
from safla.core.safety_validation import SafetyValidationFramework

class BasicAutonomousSystem:
    """A basic autonomous system using SAFLA."""
    
    def __init__(self):
        # Initialize SAFLA components
        self.engine = MetaCognitiveEngine({
            'adaptation_threshold': 0.1,
            'confidence_threshold': 0.8,
            'max_strategies': 3
        })
        
        self.delta_evaluator = DeltaEvaluator(AdaptiveWeights(
            alpha_1=0.3,  # Performance
            alpha_2=0.25, # Efficiency  
            alpha_3=0.25, # Stability
            alpha_4=0.2   # Capability
        ))
        
        self.memory = HybridMemoryArchitecture({
            'vector_dimension': 512,
            'max_episodic_memories': 1000
        })
        
        self.safety = SafetyValidationFramework({
            'risk_tolerance': 'medium',
            'checkpoint_frequency': 60
        })
        
        self.current_metrics = None
        self.previous_metrics = None
        
    async def initialize(self):
        """Initialize the autonomous system."""
        print("Initializing Basic Autonomous System...")
        
        # Initialize components
        await self.engine.initialize()
        
        # Define system goals
        performance_goal = Goal(
            goal_id="optimize_performance",
            description="Optimize overall system performance",
            target_metrics={"accuracy": 0.95, "efficiency": 0.90},
            priority=1,
            success_criteria={"min_accuracy": 0.85, "min_efficiency": 0.80}
        )
        
        await self.engine.add_goal(performance_goal)
        
        # Create initial strategies
        strategies = [
            Strategy(
                strategy_id="conservative",
                name="Conservative Strategy",
                description="Prioritize stability over performance",
                actions=[{"type": "adjust_parameters", "risk_level": "low"}],
                expected_outcomes={"stability": 0.95, "performance": 0.80},
                confidence_score=0.9,
                resource_requirements={"cpu": 0.3, "memory": 0.2}
            ),
            Strategy(
                strategy_id="aggressive",
                name="Aggressive Strategy", 
                description="Prioritize performance over stability",
                actions=[{"type": "adjust_parameters", "risk_level": "high"}],
                expected_outcomes={"stability": 0.75, "performance": 0.95},
                confidence_score=0.7,
                resource_requirements={"cpu": 0.6, "memory": 0.4}
            ),
            Strategy(
                strategy_id="balanced",
                name="Balanced Strategy",
                description="Balance performance and stability",
                actions=[{"type": "adjust_parameters", "risk_level": "medium"}],
                expected_outcomes={"stability": 0.85, "performance": 0.87},
                confidence_score=0.85,
                resource_requirements={"cpu": 0.4, "memory": 0.3}
            )
        ]
        
        for strategy in strategies:
            await self.engine.add_strategy(strategy)
            
        print("System initialized successfully!")
        
    async def run_feedback_loop(self, duration: int = 300):
        """Run the main feedback loop for specified duration."""
        print(f"Starting feedback loop for {duration} seconds...")
        
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < duration:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            # Simulate system operation and collect metrics
            current_data = await self._simulate_system_operation()
            
            # Update performance metrics
            self.current_metrics = PerformanceMetrics(
                timestamp=time.time(),
                accuracy=current_data['accuracy'],
                efficiency=current_data['efficiency'],
                stability=current_data['stability'],
                capability_score=current_data['capability']
            )
            
            # Calculate delta if we have previous metrics
            if self.previous_metrics:
                delta_result = self.delta_evaluator.calculate_delta(
                    self.current_metrics,
                    self.previous_metrics
                )
                
                print(f"Delta: {delta_result.total_delta:.4f}")
                
                # Process feedback if delta is significant
                if abs(delta_result.total_delta) > 0.05:
                    feedback = {
                        'delta_result': delta_result,
                        'current_metrics': self.current_metrics,
                        'system_state': current_data
                    }
                    
                    adaptation_result = await self.engine.process_feedback(feedback)
                    
                    if adaptation_result.success:
                        print(f"Adaptation successful: {adaptation_result.changes_made}")
                        
                        # Store successful adaptation in memory
                        await self.memory.store_memory(
                            content=adaptation_result,
                            memory_type='episodic',
                            metadata={'iteration': iteration, 'delta': delta_result.total_delta}
                        )
                    else:
                        print(f"Adaptation failed: {adaptation_result.metadata.get('error', 'Unknown error')}")
                        
            self.previous_metrics = self.current_metrics
            
            # Wait before next iteration
            await asyncio.sleep(10)
            
        print("\nFeedback loop completed!")
        
    async def _simulate_system_operation(self) -> Dict[str, float]:
        """Simulate system operation and return performance metrics."""
        # Simulate varying performance with some randomness
        base_accuracy = 0.85 + random.uniform(-0.1, 0.1)
        base_efficiency = 0.80 + random.uniform(-0.1, 0.1)
        base_stability = 0.90 + random.uniform(-0.05, 0.05)
        base_capability = 0.82 + random.uniform(-0.08, 0.08)
        
        # Ensure values are within valid range
        return {
            'accuracy': max(0.0, min(1.0, base_accuracy)),
            'efficiency': max(0.0, min(1.0, base_efficiency)),
            'stability': max(0.0, min(1.0, base_stability)),
            'capability': max(0.0, min(1.0, base_capability))
        }

# Usage example
async def main():
    system = BasicAutonomousSystem()
    await system.initialize()
    await system.run_feedback_loop(duration=120)  # Run for 2 minutes

if __name__ == "__main__":
    asyncio.run(main())
```

## Machine Learning Model Optimization

### Adaptive Neural Network Training

This example shows how to use SAFLA to optimize neural network training dynamically.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from safla.core.meta_cognitive_engine import MetaCognitiveEngine, Goal, Strategy
from safla.core.delta_evaluation import DeltaEvaluator, PerformanceMetrics

class AdaptiveNeuralNetwork(nn.Module):
    """Neural network that adapts its training using SAFLA."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
        # SAFLA components
        self.safla_engine = MetaCognitiveEngine({
            'adaptation_threshold': 0.02,
            'confidence_threshold': 0.75
        })
        self.delta_evaluator = DeltaEvaluator()
        
        # Training parameters that can be adapted
        self.learning_rate = 0.001
        self.batch_size = 32
        self.dropout_rate = 0.2
        
        # Performance tracking
        self.previous_metrics = None
        self.adaptation_history = []
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
        
    async def initialize_safla(self):
        """Initialize SAFLA components for adaptive training."""
        await self.safla_engine.initialize()
        
        # Define training optimization goal
        optimization_goal = Goal(
            goal_id="optimize_training",
            description="Optimize neural network training performance",
            target_metrics={
                "accuracy": 0.95,
                "loss_reduction": 0.9,
                "training_speed": 0.8
            },
            priority=1,
            success_criteria={"min_accuracy": 0.85}
        )
        
        await self.safla_engine.add_goal(optimization_goal)
        
        # Define adaptation strategies
        strategies = [
            Strategy(
                strategy_id="increase_lr",
                name="Increase Learning Rate",
                description="Increase learning rate for faster convergence",
                actions=[{"type": "adjust_lr", "factor": 1.5}],
                expected_outcomes={"training_speed": 0.9, "stability": 0.7},
                confidence_score=0.8,
                resource_requirements={"computation": 0.1}
            ),
            Strategy(
                strategy_id="decrease_lr",
                name="Decrease Learning Rate",
                description="Decrease learning rate for stability",
                actions=[{"type": "adjust_lr", "factor": 0.7}],
                expected_outcomes={"training_speed": 0.6, "stability": 0.9},
                confidence_score=0.9,
                resource_requirements={"computation": 0.1}
            ),
            Strategy(
                strategy_id="adjust_dropout",
                name="Adjust Dropout Rate",
                description="Modify dropout rate to prevent overfitting",
                actions=[{"type": "adjust_dropout", "new_rate": "adaptive"}],
                expected_outcomes={"accuracy": 0.85, "generalization": 0.9},
                confidence_score=0.75,
                resource_requirements={"computation": 0.05}
            ),
            Strategy(
                strategy_id="modify_batch_size",
                name="Modify Batch Size",
                description="Adjust batch size for optimal training",
                actions=[{"type": "adjust_batch_size", "direction": "adaptive"}],
                expected_outcomes={"training_speed": 0.8, "memory_efficiency": 0.85},
                confidence_score=0.7,
                resource_requirements={"memory": 0.2}
            )
        ]
        
        for strategy in strategies:
            await self.safla_engine.add_strategy(strategy)
            
    async def adaptive_train(self, train_loader: DataLoader, val_loader: DataLoader, 
                           epochs: int, device: str = 'cpu'):
        """Train the network with SAFLA-based adaptation."""
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        print("Starting adaptive training with SAFLA...")
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = await self._train_epoch(
                train_loader, optimizer, criterion, device
            )
            
            # Validation phase
            val_loss, val_acc = await self._validate_epoch(
                val_loader, criterion, device
            )
            
            # Calculate performance metrics
            current_metrics = PerformanceMetrics(
                timestamp=time.time(),
                accuracy=val_acc,
                efficiency=1.0 / (train_loss + 1e-8),  # Inverse of loss
                stability=self._calculate_stability(train_loss, val_loss),
                capability_score=self._calculate_capability(train_acc, val_acc)
            )
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}")
            
            # SAFLA adaptation
            if self.previous_metrics:
                delta_result = self.delta_evaluator.calculate_delta(
                    current_metrics, self.previous_metrics
                )
                
                print(f"Performance Delta: {delta_result.total_delta:.4f}")
                
                # Trigger adaptation if significant change
                if abs(delta_result.total_delta) > 0.02:
                    feedback = {
                        'epoch': epoch,
                        'delta_result': delta_result,
                        'current_metrics': current_metrics,
                        'training_state': {
                            'learning_rate': self.learning_rate,
                            'batch_size': self.batch_size,
                            'dropout_rate': self.dropout_rate
                        }
                    }
                    
                    adaptation_result = await self.safla_engine.process_feedback(feedback)
                    
                    if adaptation_result.success:
                        await self._apply_adaptations(
                            adaptation_result.changes_made, optimizer
                        )
                        self.adaptation_history.append({
                            'epoch': epoch,
                            'adaptations': adaptation_result.changes_made,
                            'delta': delta_result.total_delta
                        })
                        
            self.previous_metrics = current_metrics
            
    async def _train_epoch(self, train_loader, optimizer, criterion, device):
        """Train for one epoch."""
        self.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = self(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return avg_loss, accuracy
        
    async def _validate_epoch(self, val_loader, criterion, device):
        """Validate for one epoch."""
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = self(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        return avg_loss, accuracy
        
    def _calculate_stability(self, train_loss: float, val_loss: float) -> float:
        """Calculate training stability metric."""
        # Stability based on train/validation loss ratio
        ratio = train_loss / (val_loss + 1e-8)
        return max(0.0, min(1.0, 1.0 - abs(ratio - 1.0)))
        
    def _calculate_capability(self, train_acc: float, val_acc: float) -> float:
        """Calculate model capability metric."""
        # Capability based on generalization (validation accuracy)
        generalization = val_acc / (train_acc + 1e-8)
        return max(0.0, min(1.0, generalization))
        
    async def _apply_adaptations(self, changes: list, optimizer):
        """Apply SAFLA adaptations to training parameters."""
        for change in changes:
            if change['type'] == 'adjust_lr':
                factor = change.get('factor', 1.0)
                self.learning_rate *= factor
                # Update optimizer learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
                print(f"Adjusted learning rate to {self.learning_rate:.6f}")
                
            elif change['type'] == 'adjust_dropout':
                if change.get('new_rate') == 'adaptive':
                    # Adaptive dropout based on overfitting detection
                    self.dropout_rate = min(0.5, self.dropout_rate * 1.1)
                else:
                    self.dropout_rate = change.get('new_rate', self.dropout_rate)
                # Update dropout layers
                for module in self.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = self.dropout_rate
                print(f"Adjusted dropout rate to {self.dropout_rate:.3f}")
                
            elif change['type'] == 'adjust_batch_size':
                direction = change.get('direction', 'adaptive')
                if direction == 'increase':
                    self.batch_size = min(128, int(self.batch_size * 1.5))
                elif direction == 'decrease':
                    self.batch_size = max(8, int(self.batch_size * 0.7))
                print(f"Adjusted batch size to {self.batch_size}")

# Usage example
async def train_adaptive_network():
    # Create synthetic dataset
    X = torch.randn(1000, 20)
    y = torch.randint(0, 3, (1000,))
    
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create and train adaptive network
    model = AdaptiveNeuralNetwork(input_size=20, hidden_size=64, output_size=3)
    await model.initialize_safla()
    await model.adaptive_train(train_loader, val_loader, epochs=50)
    
    print("\nAdaptation History:")
    for adaptation in model.adaptation_history:
        print(f"Epoch {adaptation['epoch']}: {adaptation['adaptations']} "
              f"(Delta: {adaptation['delta']:.4f})")

if __name__ == "__main__":
    asyncio.run(train_adaptive_network())
```

## Web Service Auto-Scaling

### Adaptive Load Balancer

This example demonstrates using SAFLA for intelligent auto-scaling of web services.

```python
import asyncio
import aiohttp
import time
from typing import Dict, List
from dataclasses import dataclass
from safla.core.meta_cognitive_engine import MetaCognitiveEngine, Goal, Strategy
from safla.core.delta_evaluation import DeltaEvaluator, PerformanceMetrics
from safla.core.safety_validation import SafetyValidationFramework

@dataclass
class ServerInstance:
    """Represents a server instance."""
    instance_id: str
    endpoint: str
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    request_count: int = 0
    response_time: float = 0.0
    is_healthy: bool = True

class AdaptiveLoadBalancer:
    """Load balancer that uses SAFLA for intelligent auto-scaling."""
    
    def __init__(self):
        self.servers: List[ServerInstance] = []
        self.target_response_time = 200  # milliseconds
        self.max_cpu_usage = 0.8
        self.max_memory_usage = 0.9
        
        # SAFLA components
        self.safla_engine = MetaCognitiveEngine({
            'adaptation_threshold': 0.05,
            'confidence_threshold': 0.8
        })
        self.delta_evaluator = DeltaEvaluator()
        self.safety = SafetyValidationFramework({
            'risk_tolerance': 'medium'
        })
        
        self.previous_metrics = None
        self.scaling_history = []
        
    async def initialize(self):
        """Initialize the adaptive load balancer."""
        await self.safla_engine.initialize()
        
        # Define performance goals
        performance_goal = Goal(
            goal_id="optimize_service_performance",
            description="Maintain optimal service performance and resource utilization",
            target_metrics={
                "response_time": 0.9,  # Normalized (lower is better)
                "resource_utilization": 0.7,
                "availability": 0.99,
                "cost_efficiency": 0.8
            },
            priority=1,
            success_criteria={
                "max_response_time": 300,  # milliseconds
                "min_availability": 0.95
            }
        )
        
        await self.safla_engine.add_goal(performance_goal)
        
        # Define scaling strategies
        strategies = [
            Strategy(
                strategy_id="scale_up",
                name="Scale Up",
                description="Add more server instances to handle increased load",
                actions=[{"type": "add_instance", "count": 1}],
                expected_outcomes={
                    "response_time": 0.8,
                    "resource_utilization": 0.6,
                    "cost_efficiency": 0.7
                },
                confidence_score=0.9,
                resource_requirements={"cost": 0.3, "time": 0.2}
            ),
            Strategy(
                strategy_id="scale_down",
                name="Scale Down", 
                description="Remove server instances to reduce costs",
                actions=[{"type": "remove_instance", "count": 1}],
                expected_outcomes={
                    "response_time": 0.7,
                    "resource_utilization": 0.8,
                    "cost_efficiency": 0.9
                },
                confidence_score=0.8,
                resource_requirements={"cost": -0.3, "time": 0.1}
            ),
            Strategy(
                strategy_id="optimize_routing",
                name="Optimize Routing",
                description="Optimize request routing to improve performance",
                actions=[{"type": "update_routing", "algorithm": "least_connections"}],
                expected_outcomes={
                    "response_time": 0.85,
                    "resource_utilization": 0.75
                },
                confidence_score=0.85,
                resource_requirements={"computation": 0.1}
            ),
            Strategy(
                strategy_id="health_check_optimization",
                name="Health Check Optimization",
                description="Optimize health check frequency and remove unhealthy instances",
                actions=[{"type": "optimize_health_checks"}],
                expected_outcomes={
                    "availability": 0.98,
                    "response_time": 0.9
                },
                confidence_score=0.9,
                resource_requirements={"monitoring": 0.1}
            )
        ]
        
        for strategy in strategies:
            await self.safla_engine.add_strategy(strategy)
            
        # Initialize with some server instances
        for i in range(3):
            await self.add_server_instance(f"server-{i}", f"http://server-{i}:8080")
            
    async def add_server_instance(self, instance_id: str, endpoint: str):
        """Add a new server instance."""
        server = ServerInstance(instance_id=instance_id, endpoint=endpoint)
        self.servers.append(server)
        print(f"Added server instance: {instance_id}")
        
    async def remove_server_instance(self, instance_id: str):
        """Remove a server instance."""
        self.servers = [s for s in self.servers if s.instance_id != instance_id]
        print(f"Removed server instance: {instance_id}")
        
    async def monitor_and_adapt(self, duration: int = 600):
        """Monitor system performance and adapt using SAFLA."""
        print(f"Starting adaptive monitoring for {duration} seconds...")
        
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < duration:
            iteration += 1
            print(f"\n--- Monitoring Iteration {iteration} ---")
            
            # Collect performance metrics
            await self._collect_server_metrics()
            
            # Calculate system-wide performance
            current_metrics = await self._calculate_system_metrics()
            
            print(f"System Metrics - Response Time: {current_metrics.accuracy:.2f}, "
                  f"Resource Utilization: {current_metrics.efficiency:.2f}, "
                  f"Availability: {current_metrics.stability:.2f}")
            
            # SAFLA adaptation
            if self.previous_metrics:
                delta_result = self.delta_evaluator.calculate_delta(
                    current_metrics, self.previous_metrics
                )
                
                print(f"Performance Delta: {delta_result.total_delta:.4f}")
                
                # Trigger adaptation if significant change
                if abs(delta_result.total_delta) > 0.05:
                    feedback = {
                        'iteration': iteration,
                        'delta_result': delta_result,
                        'current_metrics': current_metrics,
                        'system_state': {
                            'server_count': len(self.servers),
                            'avg_cpu_usage': np.mean([s.cpu_usage for s in self.servers]),
                            'avg_response_time': np.mean([s.response_time for s in self.servers])
                        }
                    }
                    
                    # Validate proposed adaptations for safety
                    adaptation_result = await self.safla_engine.process_feedback(feedback)
                    
                    if adaptation_result.success:
                        # Safety validation before applying changes
                        validation_result = await self.safety.validate_action({
                            'type': 'scaling_adaptation',
                            'changes': adaptation_result.changes_made
                        })
                        
                        if validation_result.is_valid:
                            await self._apply_scaling_adaptations(adaptation_result.changes_made)
                            self.scaling_history.append({
                                'iteration': iteration,
                                'adaptations': adaptation_result.changes_made,
                                'delta': delta_result.total_delta
                            })
                        else:
                            print(f"Adaptation rejected by safety validation: {validation_result.violations}")
                            
            self.previous_metrics = current_metrics
            
            # Wait before next monitoring cycle
            await asyncio.sleep(30)
            
        print("\nAdaptive monitoring completed!")
        self._print_scaling_history()
        
    async def _collect_server_metrics(self):
        """Collect metrics from all server instances."""
        for server in self.servers:
            # Simulate metric collection (in real implementation, this would query actual servers)
            server.cpu_usage = min(1.0, max(0.0, server.cpu_usage + np.random.normal(0, 0.05)))
            server.memory_usage = min(1.0, max(0.0, server.memory_usage + np.random.normal(0, 0.03)))
            server.response_time = max(50, server.response_time + np.random.normal(0, 20))
            server.request_count += np.random.poisson(10)
            
            # Health check
            server.is_healthy = (
                server.cpu_usage < 0.95 and 
                server.memory_usage < 0.95 and 
                server.response_time < 1000
            )
            
    async def _calculate_system_metrics(self) -> PerformanceMetrics:
        """Calculate system-wide performance metrics."""
        healthy_servers = [s for s in self.servers if s.is_healthy]
        
        if not healthy_servers:
            return PerformanceMetrics(
                timestamp=time.time(),
                accuracy=0.0,
                efficiency=0.0,
                stability=0.0,
                capability_score=0.0
            )
            
        # Response time performance (normalized, lower is better)
        avg_response_time = np.mean([s.response_time for s in healthy_servers])
        response_time_score = max(0.0, 1.0 - (avg_response_time - 100) / 400)
        
        # Resource utilization efficiency
        avg_cpu = np.mean([s.cpu_usage for s in healthy_servers])
        avg_memory = np.mean([s.memory_usage for s in healthy_servers])
        resource_efficiency = 1.0 - max(avg_cpu, avg_memory)
        
        # System availability
        availability = len(healthy_servers) / len(self.servers)
        
        # Overall capability (combination of factors)
        capability = (response_time_score + resource_efficiency + availability) / 3
        
        return PerformanceMetrics(
            timestamp=time.time(),
            accuracy=response_time_score,
            efficiency=resource_efficiency,
            stability=availability,
            capability_score=capability
        )
        
    async def _apply_scaling_adaptations(self, changes: list):
        """Apply scaling adaptations based on SAFLA recommendations."""
        for change in changes:
            if change['type'] == 'add_instance':
                count = change.get('count', 1)
                for i in range(count):
                    new_id = f"server-{len(self.servers)}"
                    await self.add_server_instance(new_id, f"http://{new_id}:8080")
                    
            elif change['type'] == 'remove_instance':
                count = change.get('count', 1)
                # Remove least utilized servers
                if len(self.servers) > 1:  # Keep at least one server
                    servers_by_utilization = sorted(
                        self.servers, 
                        key=lambda s: s.cpu_usage + s.memory_usage
                    )
                    for i in range(min(count, len(self.servers) - 1)):
                        await self.remove_server_instance(servers_by_utilization[i].instance_id)
                        
            elif change['type'] == 'update_routing':
                algorithm = change.get('algorithm', 'round_robin')
                print(f"Updated routing algorithm to: {algorithm}")
                
            elif change['type'] == 'optimize_health_checks':
                # Remove unhealthy servers
                unhealthy_servers = [s for s in self.servers if not s.is_healthy]
                for server in unhealthy_servers:
                    await self.remove_server_instance(server.instance_id)
                print(f"Removed {len(unhealthy_servers)} unhealthy servers")
                
    def _print_scaling_history(self):
        """Print the scaling adaptation history."""
        print("\nScaling Adaptation History:")
        for adaptation in self.scaling_history:
            print(f"Iteration {adaptation['iteration']}: {adaptation['adaptations']} "
                  f"(Delta: {adaptation['delta']:.4f})")

# Usage example
async def run_adaptive_load_balancer():
    load_balancer = AdaptiveLoadBalancer()
    await load_balancer.initialize()
    await load_balancer.monitor_and_adapt(duration=300)  # Run for 5 minutes

if __name__ == "__main__":
    import numpy as np
    asyncio.run(run_adaptive_load_balancer())
```

## Distributed System Coordination

### Multi-Agent SAFLA System

This example shows how to coordinate multiple SAFLA agents in a distributed system.

```python
import asyncio
import json
from typing import Dict, List, Optional
from safla.core.mcp_orchestration import (
    MCPOrchestrator, MCPServer, Agent, ContextVector
)
from safla.core.meta_cognitive_engine import MetaCognitiveEngine
from safla.core.delta_evaluation import DeltaEvaluator

class DistributedSAFLACoordinator:
    """Coordinates multiple SAFLA agents in a distributed system."""
    
    def __init__(self):
        self.orchestrator = MCPOrchestrator()
        self.agents: Dict[str, Agent] = {}
        self.agent_engines: Dict[str, MetaCognitiveEngine] = {}
        self.coordination_history = []
        
    async def initialize_distributed_system(self):
        """Initialize the distributed SAFLA system."""
        print("Initializing Distributed SAFLA System...")
        
        # Start the orchestrator
        await self.orchestrator.start()
        
        # Register MCP servers for different capabilities
        servers = [
            MCPServer(
                server_id="data_processing_server",
                name="Data Processing Server",
                endpoint="http://localhost:8001",
                capabilities=["data_analysis", "pattern_recognition", "feature_extraction"],
                status="active"
            ),
            MCPServer(
                server_id="ml_optimization_server", 
                name="ML Optimization Server",
                endpoint="http://localhost:8002",
                capabilities=["model_training", "hyperparameter_tuning", "performance_optimization"],
                status="active"
            ),
            MCPServer(
                server_id="resource_management_server",
                name="Resource Management Server",
                endpoint="http://localhost:8003", 
                capabilities=["resource_allocation", "load_balancing", "scaling"],
                status="active"
            )
        ]
        
        for server in servers:
            self.orchestrator.server_manager.register_server(server)
            
        # Register specialized agents
        agents = [
            Agent(
                agent_id="data_analyzer",
                name="Data Analysis Agent",
                agent_type="analyzer",
                capabilities=["data_analysis", "pattern_recognition"],
                status="active",
                server_id="data_processing_server",
                priority=2
            ),
            Agent(
                agent_id="ml_optimizer",
                name="ML Optimization Agent", 
                agent_type="optimizer",
                capabilities=["model_training", "hyperparameter_tuning"],
                status="active",
                server_id="ml_optimization_server",
                priority=3
            ),
            Agent(
                agent_id="resource_manager",
                name="Resource Management Agent",
                agent_type="manager",
                capabilities=["resource_allocation", "scaling"],
                status="active", 
                server_id="resource_management_server",
                priority=1
            ),
            Agent(
                agent_id="coordinator",
                name="System Coordinator Agent",
                agent_type="coordinator",
                capabilities=["coordination", "decision_making", "conflict_resolution"],
                status="active",
                server_id="data_processing_server",
                priority=4
            )
        ]
        
        for agent in agents:
            self.orchestrator.agent_coordinator.register_agent(agent)
            self.agents[agent.agent_id] = agent
            
            # Initialize SAFLA engine for each agent
            engine = MetaCognitiveEngine({
                'adaptation_threshold': 0.08,
                'confidence_threshold': 0.75,
                'max_strategies': 4
            })
            await engine.initialize()
            self.agent_engines[agent.agent_id] = engine
            
        print("Distributed system initialized successfully!")
        
    async def coordinate_distributed_task(self, task: Dict) -> Dict:
        """Coordinate a distributed task across multiple agents."""
        print(f"\nCoordinating distributed task: {task.get('task_id')}")
        
        # Decompose task into subtasks
        subtasks = await self._decompose_task(task)
        
        # Assign subtasks to appropriate agents
        assignments = await self._assign_subtasks(subtasks)
        
        # Execute subtasks in parallel with coordination
        results = await self._execute_coordinated_subtasks(assignments)
        
        # Aggregate results and adapt coordination strategy
        final_result = await self._aggregate_and_adapt(results, task)
        
        return final_result
        
    async def _decompose_task(self, task: Dict) -> List[Dict]:
        """Decompose a complex task into subtasks."""
        task_type = task.get('type', 'unknown')
        
        if task_type == 'ml_pipeline':
            return [
                {
                    'subtask_id': 'data_preprocessing',
                    'type': 'data_analysis',
                    'required_capabilities': ['data_analysis', 'feature_extraction'],
                    'input_data': task.get('raw_data'),
                    'priority': 1
                },
                {
                    'subtask_id': 'model_training',
                    'type': 'model_training', 
                    'required_capabilities': ['model_training', 'hyperparameter_tuning'],
                    'depends_on': ['data_preprocessing'],
                    'priority': 2
                },
                {
                    'subtask_id': 'performance_evaluation',
                    'type': 'evaluation',
                    'required_capabilities': ['pattern_recognition', 'performance_optimization'],
                    'depends_on': ['model_training'],
                    'priority': 3
                },
                {
                    'subtask_id': 'resource_optimization',
                    'type': 'optimization',
                    'required_capabilities': ['resource_allocation', 'scaling'],
                    'parallel_with': ['model_training'],
                    'priority': 2
                }
            ]
        elif task_type == 'system_optimization':
            return [
                {
                    'subtask_id': 'performance_analysis',
                    'type': 'analysis',
                    'required_capabilities': ['data_analysis', 'pattern_recognition'],
                    'priority': 1
                },
                {
                    'subtask_id': 'resource_reallocation',
                    'type': 'resource_management',
                    'required_capabilities': ['resource_allocation', 'load_balancing'],
                    'depends_on': ['performance_analysis'],
                    'priority': 2
                }
            ]
        else:
            # Generic decomposition
            return [task]
            
    async def _assign_subtasks(self, subtasks: List[Dict]) -> Dict[str, List[Dict]]:
        """Assign subtasks to appropriate agents."""
        assignments = {}
        
        for subtask in subtasks:
            required_capabilities = subtask.get('required_capabilities', [])
            
            # Find capable agents for each required capability
            capable_agents = []
            for capability in required_capabilities:
                agents = self.orchestrator.agent_coordinator.find_capable_agents(capability)
                capable_agents.extend(agents)
                
            if capable_agents:
                # Select best agent using conflict resolution
                selected_agent = self.orchestrator.agent_coordinator.resolve_capability_conflict(
                    required_capabilities[0]  # Use primary capability for selection
                )
                
                if selected_agent:
                    agent_id = selected_agent.agent_id
                    if agent_id not in assignments:
                        assignments[agent_id] = []
                    assignments[agent_id].append(subtask)
                    
                    print(f"Assigned subtask '{subtask['subtask_id']}' to agent '{agent_id}'")
                    
        return assignments
        
    async def _execute_coordinated_subtasks(self, assignments: Dict[str, List[Dict]]) -> Dict:
        """Execute subtasks with coordination and adaptation."""
        results = {}
        execution_tasks = []
        
        # Create execution tasks for each agent
        for agent_id, subtasks in assignments.items():
            task = asyncio.create_task(
                self._execute_agent_subtasks(agent_id, subtasks)
            )
            execution_tasks.append((agent_id, task))
            
        # Execute with coordination
        for agent_id, task in execution_tasks:
            try:
                agent_results = await task
                results[agent_id] = agent_results
                
                # Share context with other agents
                await self._share_agent_context(agent_id, agent_results)
                
            except Exception as e:
                print(f"Error executing subtasks for agent {agent_id}: {e}")
                results[agent_id] = {'error': str(e)}
                
        return results
        
    async def _execute_agent_subtasks(self, agent_id: str, subtasks: List[Dict]) -> Dict:
        """Execute subtasks for a specific agent with SAFLA adaptation."""
        agent_engine = self.agent_engines[agent_id]
        agent_results = {}
        
        for subtask in subtasks:
            print(f"Agent {agent_id} executing subtask: {subtask['subtask_id']}")
            
            # Simulate subtask execution with performance metrics
            execution_result = await self._simulate_subtask_execution(subtask)
            
            # Process through agent's SAFLA engine
            feedback = {
                'subtask_id': subtask['subtask_id'],
                'execution_result': execution_result,
                'agent_id': agent_id
            }
            
            adaptation_result = await agent_engine.process_feedback(feedback)
            
            agent_results[subtask['subtask_id']] = {
                'execution_result': execution_result,
                'adaptation_result': adaptation_result,
                'agent_id': agent_id
            }
            
        return agent_results
        
    async def _simulate_subtask_execution(self, subtask: Dict) -> Dict:
        """Simulate subtask execution (replace with actual implementation)."""
        import random
        
        # Simulate execution time and results
        execution_time = random.uniform(1.0, 5.0)
        success_rate = random.uniform(0.7, 0.95)
        
        await asyncio.sleep(execution_time / 10)  # Scaled down for demo
        
        return {
            'success': success_rate > 0.8,
            'execution_time': execution_time,
            'performance_score': success_rate,
            'output_data': f"Result for {subtask['subtask_id']}"
        }
        
    async def _share_agent_context(self, agent_id: str, results: Dict):
        """Share agent context with other agents through the orchestrator."""
        # Create context vector for sharing
        context_data = {
            'agent_id': agent_id,
            'results_summary': {
                'success_count': sum(1 for r in results.values() 
                                   if r.get('execution_result', {}).get('success', False)),
                'total_tasks': len(results),
                'avg_performance': sum(r.get('execution_result', {}).get('performance_score', 0) 
                                     for r in results.values()) / len(results) if results else 0
            },
            'timestamp': time.time()
        }
        
        # Convert to embedding (simplified - use actual embedding in production)
        embedding = [hash(str(context_data)) % 1000 / 1000.0] * 512
        
        context_vector = ContextVector(
            context_id=f"agent_{agent_id}_{int(time.time())}",
            embedding=embedding,
            metadata=context_data,
            ttl=3600  # 1 hour
        )
        
        # Store in context sharing engine
        self.orchestrator.context_engine.store_context(context_vector)
        
    async def _aggregate_and_adapt(self, results: Dict, original_task: Dict) -> Dict:
        """Aggregate results and adapt coordination strategy."""
        # Calculate overall success metrics
        total_subtasks = sum(len(agent_results) for agent_results in results.values())
        successful_subtasks = sum(
            1 for agent_results in results.values()
            for result in agent_results.values()
            if result.get('execution_result', {}).get('success', False)
        )
        
        overall_success_rate = successful_subtasks / total_subtasks if total_subtasks > 0 else 0
        
        # Aggregate performance data
        performance_scores = []
        for agent_results in results.values():
            for result in agent_results.values():
                score = result.get('execution_result', {}).get('performance_score', 0)
                performance_scores.append(score)
                
        avg_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0
        
        # Create coordination adaptation feedback
        coordination_feedback = {
            'task_id': original_task.get('task_id'),
            'overall_success_rate': overall_success_rate,
            'avg_performance': avg_performance,
            'agent_results': results,
            'coordination_metrics': {
                'task_distribution_efficiency': self._calculate_distribution_efficiency(results),
                'inter_agent_coordination': self._calculate_coordination_score(results)
            }
        }
        
        # Process through coordinator agent's SAFLA engine
        coordinator_engine = self.agent_engines.get('coordinator')
        if coordinator_engine:
            coordination_adaptation = await coordinator_engine.process_feedback(coordination_feedback)
            
            self.coordination_history.append({
                'task_id': original_task.get('task_id'),
                'coordination_adaptation': coordination_adaptation,
                'performance_metrics': coordination_feedback
            })
            
        return {
            'task_id': original_task.get('task_id'),
            'success': overall_success_rate > 0.8,
            'overall_success_rate': overall_success_rate,
            'avg_performance': avg_performance,
            'detailed_results': results,
            'coordination_feedback': coordination_feedback
        }
        
    def _calculate_distribution_efficiency(self, results: Dict) -> float:
        """Calculate task distribution efficiency."""
        # Simple efficiency calculation based on load balance
        agent_loads = [len(agent_results) for agent_results in results.values()]
        if not agent_loads:
            return 0.0
            
        max_load = max(agent_loads)
        min_load = min(agent_loads)
        
        # Efficiency is higher when load is more evenly distributed
        return 1.0 - (max_load - min_load) / max_load if max_load > 0 else 1.0
        
    def _calculate_coordination_score(self, results: Dict) -> float:
        """Calculate inter-agent coordination score."""
        # Simple coordination score based on success rate consistency
        success_rates = []
        for agent_results in results.values():
            agent_success_rate = sum(
                1 for result in agent_results.values()
                if result.get('execution_result', {}).get('success', False)
            ) / len(agent_results) if agent_results else 0
            success_rates.append(agent_success_rate)
            
        if not success_rates:
            return 0.0
            
        # Coordination is better when success rates are consistently high
        avg_success = sum(success_rates) / len(success_rates)
        variance = sum((rate - avg_success) ** 2 for rate in success_rates) / len(success_rates)
        
        return avg_success * (1.0 - variance)  # High average, low variance = good coordination

# Usage example
async def run_distributed_coordination():
    coordinator = DistributedSAFLACoordinator()
    await coordinator.initialize_distributed_system()
    
    # Example tasks
    tasks = [
        {
            'task_id': 'ml_pipeline_001',
            'type': 'ml_pipeline',
            'raw_data': 'dataset_001.csv',
            'target_accuracy': 0.9
        },
        {
            'task_id': 'system_opt_001', 
            'type': 'system_optimization',
            'target_metrics': {'latency': 100, 'throughput': 1000}
        }
    ]
    
    # Execute tasks
    for task in tasks:
        result = await coordinator.coordinate_distributed_task(task)
        print(f"\nTask {task['task_id']} completed:")
        print(f"Success: {result['success']}")
        print(f"Overall Success Rate: {result['overall_success_rate']:.2f}")
        print(f"Average Performance: {result['avg_performance']:.2f}")
        
    # Print coordination history
    print("\nCoordination History:")
    for entry in coordinator.coordination_history:
        print(f"Task {entry['task_id']}: "
              f"Success Rate: {entry['performance_metrics']['overall_success_rate']:.2f}")

if __name__ == "__main__":
    import time
    asyncio.run(run_distributed_coordination())
```

## Real-time Data Processing

### Adaptive Stream Processor

This example shows SAFLA applied to real-time data stream processing with adaptive optimization.

```python
import asyncio
import time
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import deque
import statistics
from safla.core.meta_cognitive_engine import MetaCognitiveEngine, Goal, Strategy
from safla.core.delta_evaluation import DeltaEvaluator, PerformanceMetrics
from safla.core.hybrid_memory import HybridMemoryArchitecture

@dataclass
class DataPoint:
    """Represents a single data point in the stream."""
    timestamp: float
    value: float
    metadata: Dict[str, Any]

@dataclass
class ProcessingResult:
    """Result of processing a data point."""
    input_data: DataPoint
    processed_value: float
    processing_time: float
    success: bool
    error_message: Optional[str] = None

class AdaptiveStreamProcessor:
    """Real-time data stream processor with SAFLA-based adaptation."""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.data_buffer = deque(maxlen=buffer_size)
        self.processing_results = deque(maxlen=buffer_size)
        
        # Processing parameters that can be adapted
        self.batch_size = 10
        self.processing_timeout = 1.0
        self.quality_threshold = 0.8
        self.anomaly_threshold = 2.0  # Standard deviations
        
        # SAFLA components
        self.safla_engine = MetaCognitiveEngine({
            'adaptation_threshold': 0.03,
            'confidence_threshold': 0.75
        })
        self.delta_evaluator = DeltaEvaluator()
        self.memory = HybridMemoryArchitecture({
            'vector_dimension': 256,
            'max_episodic_memories': 5000
        })
        
        # Performance tracking
        self.previous_metrics = None
        self.adaptation_history = []
        self.processing_stats = {
            'total_processed': 0,
            'successful_processed': 0,
            'total_processing_time': 0.0,
            'anomalies_detected': 0
        }
        
    async def initialize(self):
        """Initialize the adaptive stream processor."""
        print("Initializing Adaptive Stream Processor...")
        
        await self.safla_engine.initialize()
        
        # Define processing optimization goals
        throughput_goal = Goal(
            goal_id="maximize_throughput",
            description="Maximize data processing throughput while maintaining quality",
            target_metrics={
                "throughput": 0.9,
                "latency": 0.8,  # Lower is better
                "quality": 0.9,
                "resource_efficiency": 0.85
            },
            priority=1,
            success_criteria={
                "min_throughput": 100,  # items per second
                "max_latency": 2.0,     # seconds
                "min_quality": 0.8
            }
        )
        
        await self.safla_engine.add_goal(throughput_goal)
        
        # Define adaptation strategies
        strategies = [
            Strategy(
                strategy_id="increase_batch_size",
                name="Increase Batch Size",
                description="Increase batch size to improve throughput",
                actions=[{"type": "adjust_batch_size", "factor": 1.5}],
                expected_outcomes={
                    "throughput": 0.9,
                    "latency": 0.7,
                    "resource_efficiency": 0.8
                },
                confidence_score=0.85,
                resource_requirements={"memory": 0.2, "cpu": 0.1}
            ),
            Strategy(
                strategy_id="decrease_batch_size",
                name="Decrease Batch Size",
                description="Decrease batch size to reduce latency",
                actions=[{"type": "adjust_batch_size", "factor": 0.7}],
                expected_outcomes={
                    "throughput": 0.7,
                    "latency": 0.9,
                    "resource_efficiency": 0.9
                },
                confidence_score=0.9,
                resource_requirements={"memory": -0.1, "cpu": -0.05}
            ),
            Strategy(
                strategy_id="adjust_timeout",
                name="Adjust Processing Timeout",
                description="Optimize processing timeout for better performance",
                actions=[{"type": "adjust_timeout", "direction": "adaptive"}],
                expected_outcomes={
                    "throughput": 0.85,
                    "quality": 0.9
                },
                confidence_score=0.8,
                resource_requirements={"cpu": 0.05}
            ),
            Strategy(
                strategy_id="optimize_quality_threshold",
                name="Optimize Quality Threshold",
                description="Adjust quality threshold for optimal processing",
                actions=[{"type": "adjust_quality_threshold", "direction": "adaptive"}],
                expected_outcomes={
                    "quality": 0.95,
                    "throughput": 0.8
                },
                confidence_score=0.75,
                resource_requirements={"computation": 0.1}
            ),
            Strategy(
                strategy_id="anomaly_detection_tuning",
                name="Anomaly Detection Tuning",
                description="Tune anomaly detection sensitivity",
                actions=[{"type": "adjust_anomaly_threshold", "direction": "adaptive"}],
                expected_outcomes={
                    "quality": 0.9,
                    "false_positive_rate": 0.1
                },
                confidence_score=0.8,
                resource_requirements={"computation": 0.15}
            )
        ]
        
        for strategy in strategies:
            await self.safla_engine.add_strategy(strategy)
            
        print("Stream processor initialized successfully!")
        
    async def start_processing(self, data_generator, duration: int = 300):
        """Start processing data stream with adaptive optimization."""
        print(f"Starting adaptive stream processing for {duration} seconds...")
        
        # Start data ingestion task
        ingestion_task = asyncio.create_task(
            self._ingest_data_stream(data_generator, duration)
        )
        
        # Start processing task
        processing_task = asyncio.create_task(
            self._process_data_stream(duration)
        )
        
        # Start adaptation monitoring task
        adaptation_task = asyncio.create_task(
            self._monitor_and_adapt(duration)
        )
        
        # Wait for all tasks to complete
        await asyncio.gather(ingestion_task, processing_task, adaptation_task)
        
        print("Stream processing completed!")
        self._print_final_statistics()
        
    async def _ingest_data_stream(self, data_generator, duration: int):
        """Ingest data from the stream."""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                # Generate or receive data point
                data_point = await data_generator()
                self.data_buffer.append(data_point)
                
                # Small delay to simulate real-time streaming
                await asyncio.sleep(0.01)
                
            except Exception as e:
                print(f"Error in data ingestion: {e}")
                await asyncio.sleep(0.1)
                
    async def _process_data_stream(self, duration: int):
        """Process data from the buffer."""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            if len(self.data_buffer) >= self.batch_size:
                # Extract batch for processing
                batch = []
                for _ in range(min(self.batch_size, len(self.data_buffer))):
                    if self.data_buffer:
                        batch.append(self.data_buffer.popleft())
                        
                if batch:
                    # Process batch
                    batch_results = await self._process_batch(batch)
                    self.processing_results.extend(batch_results)
                    
                    # Update statistics
                    self._update_processing_stats(batch_results)
                    
            else:
                # Wait for more data
                await asyncio.sleep(0.05)
                
    async def _process_batch(self, batch: List[DataPoint]) -> List[ProcessingResult]:
        """Process a batch of data points."""
        results = []
        batch_start_time = time.time()
        
        for data_point in batch:
            processing_start = time.time()
            
            try:
                # Simulate data processing with timeout
                processed_value = await asyncio.wait_for(
                    self._process_single_point(data_point),
                    timeout=self.processing_timeout
                )
                
                processing_time = time.time() - processing_start
                
                # Quality check
                quality_score = self._calculate_quality_score(data_point, processed_value)
                success = quality_score >= self.quality_threshold
                
                result = ProcessingResult(
                    input_data=data_point,
                    processed_value=processed_value,
                    processing_time=processing_time,
                    success=success
                )
                
                results.append(result)
                
            except asyncio.TimeoutError:
                result = ProcessingResult(
                    input_data=data_point,
                    processed_value=0.0,
                    processing_time=self.processing_timeout,
                    success=False,
                    error_message="Processing timeout"
                )
                results.append(result)
                
            except Exception as e:
                result = ProcessingResult(
                    input_data=data_point,
                    processed_value=0.0,
                    processing_time=time.time() - processing_start,
                    success=False,
                    error_message=str(e)
                )
                results.append(result)
                
        return results
        
    async def _process_single_point(self, data_point: DataPoint) -> float:
        """Process a single data point (implement your processing logic here)."""
        # Simulate processing with some computation
        await asyncio.sleep(random.uniform(0.01, 0.1))
        
        # Example processing: apply some transformation
        processed_value = data_point.value * 1.5 + random.uniform(-0.1, 0.1)
        
        # Simulate anomaly detection
        if abs(data_point.value) > self.anomaly_threshold:
            # Handle anomaly
            processed_value = self._handle_anomaly(data_point)
            
        return processed_value
        
    def _handle_anomaly(self, data_point: DataPoint) -> float:
        """Handle detected anomalies."""
        self.processing_stats['anomalies_detected'] += 1
        
        # Simple anomaly handling: cap the value
        max_normal_value = self.anomaly_threshold * 0.9
        return max(-max_normal_value, min(max_normal_value, data_point.value))
        
    def _calculate_quality_score(self, input_data: DataPoint, processed_value: float) -> float:
        """Calculate quality score for processed data."""
        # Simple quality metric based on processing consistency
        expected_range = abs(input_data.value) * 2
        actual_deviation = abs(processed_value - input_data.value * 1.5)
        
        quality = max(0.0, 1.0 - (actual_deviation / expected_range))
        return quality
        
    def _update_processing_stats(self, results: List[ProcessingResult]):
        """Update processing statistics."""
        for result in results:
            self.processing_stats['total_processed'] += 1
            self.processing_stats['total_processing_time'] += result.processing_time
            
            if result.success:
                self.processing_stats['successful_processed'] += 1
                
    async def _monitor_and_adapt(self, duration: int):
        """Monitor performance and trigger SAFLA adaptations."""
        start_time = time.time()
        monitoring_interval = 30  # seconds
        
        while time.time() - start_time < duration:
            await asyncio.sleep(monitoring_interval)
            
            # Calculate current performance metrics
            current_metrics = self._calculate_performance_metrics()
            
            print(f"\nPerformance Metrics:")
            print(f"Throughput: {current_metrics.accuracy:.3f}")
            print(f"Latency: {current_metrics.efficiency:.3f}")
            print(f"Quality: {current_metrics.stability:.3f}")
            print(f"Resource Efficiency: {current_metrics.capability_score:.3f}")
            
            # SAFLA adaptation
            if self.previous_metrics:
                delta_result = self.delta_evaluator.calculate_delta(
                    current_metrics, self.previous_metrics
                )
                
                print(f"Performance Delta: {delta_result.total_delta:.4f}")
                
                # Trigger adaptation if significant change
                if abs(delta_result.total_delta) > 0.03:
                    feedback = {
                        'delta_result': delta_result,
                        'current_metrics': current_metrics,
                        'processing_stats': self.processing_stats.copy(),
                        'system_state': {
                            'batch_size': self.batch_size,
                            'processing_timeout': self.processing_timeout,
                            'quality_threshold': self.quality_threshold,
                            'buffer_utilization': len(self.data_buffer) / self.buffer_size
                        }
                    }
                    
                    adaptation_result = await self.safla_engine.process_feedback(feedback)
                    
                    if adaptation_result.success:
                        await self._apply_processing_adaptations(adaptation_result.changes_made)
                        
                        # Store adaptation in memory
                        await self.memory.store_memory(
                            content=adaptation_result,
                            memory_type='episodic',
                            metadata={
                                'timestamp': time.time(),
                                'delta': delta_result.total_delta,
                                'processing_stats': self.processing_stats.copy()
                            }
                        )
                        
                        self.adaptation_history.append({
                            'timestamp': time.time(),
                            'adaptations': adaptation_result.changes_made,
                            'delta': delta_result.total_delta,
                            'metrics': current_metrics
                        })
                        
            self.previous_metrics = current_metrics
            
    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate current performance metrics."""
        if self.processing_stats['total_processed'] == 0:
            return PerformanceMetrics(
                timestamp=time.time(),
                accuracy=0.0,
                efficiency=0.0,
                stability=0.0,
                capability_score=0.0
            )
            
        # Throughput (items per second)
        avg_processing_time = (
            self.processing_stats['total_processing_time'] / 
            self.processing_stats['total_processed']
        )
        throughput_score = min(1.0, 1.0 / (avg_processing_time * 10))  # Normalized
        
        # Latency (inverse of processing time)
        latency_score = min(1.0, 1.0 / (avg_processing_time + 0.1))
        
        # Quality (success rate)
        quality_score = (
            self.processing_stats['successful_processed'] / 
            self.processing_stats['total_processed']
        )
        
        # Resource efficiency (based on buffer utilization and processing efficiency)
        buffer_efficiency = 1.0 - (len(self.data_buffer) / self.buffer_size)
        processing_efficiency = quality_score * latency_score
        resource_efficiency = (buffer_efficiency + processing_efficiency) / 2
        
        return PerformanceMetrics(
            timestamp=time.time(),
            accuracy=throughput_score,
            efficiency=latency_score,
            stability=quality_score,
            capability_score=resource_efficiency
        )
        
    async def _apply_processing_adaptations(self, changes: list):
        """Apply SAFLA adaptations to processing parameters."""
        for change in changes:
            if change['type'] == 'adjust_batch_size':
                factor = change.get('factor', 1.0)
                old_batch_size = self.batch_size
                self.batch_size = max(1, min(100, int(self.batch_size * factor)))
                print(f"Adjusted batch size from {old_batch_size} to {self.batch_size}")
                
            elif change['type'] == 'adjust_timeout':
                direction = change.get('direction', 'adaptive')
                old_timeout = self.processing_timeout
                
                if direction == 'increase':
                    self.processing_timeout = min(5.0, self.processing_timeout * 1.2)
                elif direction == 'decrease':
                    self.processing_timeout = max(0.1, self.processing_timeout * 0.8)
                elif direction == 'adaptive':
                    # Adaptive timeout based on recent processing times
                    recent_times = [r.processing_time for r in list(self.processing_results)[-100:] if r.success]
                    if recent_times:
                        avg_time = statistics.mean(recent_times)
                        self.processing_timeout = avg_time * 2.0  # 2x average time
                        
                print(f"Adjusted timeout from {old_timeout:.3f} to {self.processing_timeout:.3f}")
                
            elif change['type'] == 'adjust_quality_threshold':
                direction = change.get('direction', 'adaptive')
                old_threshold = self.quality_threshold
                
                if direction == 'increase':
                    self.quality_threshold = min(0.95, self.quality_threshold + 0.05)
                elif direction == 'decrease':
                    self.quality_threshold = max(0.5, self.quality_threshold - 0.05)
                elif direction == 'adaptive':
                    # Adaptive threshold based on recent quality scores
                    recent_results = list(self.processing_results)[-100:]
                    if recent_results:
                        success_rate = sum(1 for r in recent_results if r.success) / len(recent_results)
                        if success_rate > 0.9:
                            self.quality_threshold = min(0.95, self.quality_threshold + 0.02)
                        elif success_rate < 0.7:
                            self.quality_threshold = max(0.5, self.quality_threshold - 0.02)
                            
                print(f"Adjusted quality threshold from {old_threshold:.3f} to {self.quality_threshold:.3f}")
                
            elif change['type'] == 'adjust_anomaly_threshold':
                direction = change.get('direction', 'adaptive')
                old_threshold = self.anomaly_threshold
                
                if direction == 'increase':
                    self.anomaly_threshold = min(5.0, self.anomaly_threshold * 1.1)
                elif direction == 'decrease':
                    self.anomaly_threshold = max(1.0, self.anomaly_threshold * 0.9)
                elif direction == 'adaptive':
                    # Adaptive threshold based on anomaly detection rate
                    anomaly_rate = (
                        self.processing_stats['anomalies_detected'] / 
                        max(1, self.processing_stats['total_processed'])
                    )
                    if anomaly_rate > 0.1:  # Too many anomalies
                        self.anomaly_threshold = min(5.0, self.anomaly_threshold * 1.1)
                    elif anomaly_rate < 0.01:  # Too few anomalies
                        self.anomaly_threshold = max(1.0, self.anomaly_threshold * 0.9)
                        
                print(f"Adjusted anomaly threshold from {old_threshold:.3f} to {self.anomaly_threshold:.3f}")
                
    def _print_final_statistics(self):
        """Print final processing statistics."""
        print("\n" + "="*50)
        print("FINAL PROCESSING STATISTICS")
        print("="*50)
        
        total = self.processing_stats['total_processed']
        successful = self.processing_stats['successful_processed']
        
        print(f"Total Data Points Processed: {total}")
        print(f"Successfully Processed: {successful}")
        print(f"Success Rate: {successful/total*100:.1f}%" if total > 0 else "Success Rate: 0%")
        
        if total > 0:
            avg_time = self.processing_stats['total_processing_time'] / total
            print(f"Average Processing Time: {avg_time:.3f} seconds")
            print(f"Throughput: {1/avg_time:.1f} items/second")
            
        print(f"Anomalies Detected: {self.processing_stats['anomalies_detected']}")
        
        print(f"\nFinal Parameters:")
        print(f"Batch Size: {self.batch_size}")
        print(f"Processing Timeout: {self.processing_timeout:.3f} seconds")
        print(f"Quality Threshold: {self.quality_threshold:.3f}")
        print(f"Anomaly Threshold: {self.anomaly_threshold:.3f}")
        
        print(f"\nAdaptations Applied: {len(self.adaptation_history)}")
        for i, adaptation in enumerate(self.adaptation_history):
            print(f"  {i+1}. {adaptation['adaptations']} (Delta: {adaptation['delta']:.4f})")

# Data generator for testing
async def generate_data_stream():
    """Generate synthetic data stream for testing."""
    timestamp = time.time()
    
    # Generate data with some patterns and occasional anomalies
    if random.random() < 0.05:  # 5% chance of anomaly
        value = random.uniform(-5, 5)  # Anomalous value
    else:
        value = random.gauss(0, 1)  # Normal value
        
    return DataPoint(
        timestamp=timestamp,
        value=value,
        metadata={
            'source': 'sensor_001',
            'quality': random.uniform(0.8, 1.0)
        }
    )

# Usage example
async def run_adaptive_stream_processor():
    processor = AdaptiveStreamProcessor(buffer_size=500)
    await processor.initialize()
    await processor.start_processing(generate_data_stream, duration=180)  # 3 minutes

if __name__ == "__main__":
    asyncio.run(run_adaptive_stream_processor())
```

## Next Steps

These examples demonstrate the versatility and power of SAFLA across different domains. To implement SAFLA in your own applications:

1. **Start Simple**: Begin with the basic autonomous system example and gradually add complexity
2. **Identify Adaptation Points**: Determine what parameters in your system can be adapted
3. **Define Clear Goals**: Establish measurable goals and success criteria
4. **Implement Safety**: Always include safety validation for critical systems
5. **Monitor and Learn**: Use the memory system to learn from past adaptations

For more detailed information:
- Review the [API Reference](09-api-reference.md) for complete method documentation
- Check the [SDK Guide](10-sdk-guide.md) for integration patterns
- Study the [Configuration Guide](11-configuration.md) for optimal settings
- Explore the [Algorithms Guide](13-algorithms.md) for mathematical foundations