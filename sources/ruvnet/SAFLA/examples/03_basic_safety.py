#!/usr/bin/env python3
"""
SAFLA Basic Safety Features - Safety and Validation Framework
============================================================

This example demonstrates SAFLA's comprehensive safety and validation system.
Learn how to implement safety constraints, risk assessment, and validation workflows.

Learning Objectives:
- Understand SAFLA's safety constraint system
- Learn about risk assessment and scoring
- Implement validation workflows
- Practice safety monitoring and alerts

Time to Complete: 15-20 minutes
Complexity: Beginner to Intermediate
"""

import asyncio
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List

from safla.core.safety_validation import (
    OptimizedSafetyValidator, 
    SafetyViolation, 
    SafetyLevel,
    ValidationResponse,
    ValidationResult
)
from safla import get_logger

logger = get_logger(__name__)


class SafetyDemo:
    """Comprehensive safety system demonstration."""
    
    def __init__(self):
        self.safety_validator = None
        self.violation_count = 0
        self.validation_history = []
    
    async def initialize(self):
        """Initialize the safety validation system."""
        print("🛡️ Initializing SAFLA Safety Validation Framework...")
        self.safety_validator = OptimizedSafetyValidator(
            max_concurrent_validations=200,
            cache_size=10000,
            cache_ttl=3600.0,
            enable_bloom_filters=True
        )
        print("✅ Safety framework ready!")
    
    async def demonstrate_basic_constraints(self):
        """Demonstrate basic safety constraints."""
        print("\n⚡ Basic Safety Constraints")
        print("-" * 40)
        
        # Example 1: Memory usage constraint
        print("Testing memory usage constraints...")
        
        memory_tests = [
            {"memory_usage": 50_000_000, "description": "Normal usage (50MB)"},
            {"memory_usage": 500_000_000, "description": "High usage (500MB)"},
            {"memory_usage": 2_000_000_000, "description": "Excessive usage (2GB)"},
        ]
        
        for test in memory_tests:
            request = {
                "operation": "memory_allocation",
                "data": {"memory_usage": test["memory_usage"]},
                "metadata": {"test_case": test["description"]}
            }
            
            result = await self.safety_validator.validate(
                content=f"{request['operation']} operation",
                context=request
            )
            status = "✅ SAFE" if result.result == ValidationResult.SAFE else "⚠️ UNSAFE"
            print(f"  {status} {test['description']}")
            
            if not result.result == ValidationResult.SAFE:
                for violation in result.violations:
                    print(f"    ❌ Violation: {violation.description}")
        
        # Example 2: Rate limiting constraint
        print("\nTesting rate limiting constraints...")
        
        for i in range(5):
            request = {
                "operation": "api_call",
                "data": {"endpoint": "/process_data", "user_id": "user_123"},
                "metadata": {"request_number": i + 1}
            }
            
            result = await self.safety_validator.validate(
                content=f"{request['operation']} operation",
                context=request
            )
            status = "✅ ALLOWED" if result.result == ValidationResult.SAFE else "🚫 RATE LIMITED"
            print(f"  Request {i+1}: {status}")
            
            # Small delay to simulate real requests
            await asyncio.sleep(0.1)
    
    async def demonstrate_risk_assessment(self):
        """Demonstrate risk assessment capabilities."""
        print("\n🎯 Risk Assessment Demonstration")
        print("-" * 40)
        
        # Define different risk scenarios
        scenarios = [
            {
                "name": "Normal Operation",
                "data": {
                    "cpu_usage": 0.3,
                    "memory_usage": 100_000_000,
                    "concurrent_operations": 5,
                    "error_rate": 0.01,
                    "user_privilege": "standard"
                }
            },
            {
                "name": "High Load Scenario",
                "data": {
                    "cpu_usage": 0.8,
                    "memory_usage": 800_000_000,
                    "concurrent_operations": 50,
                    "error_rate": 0.05,
                    "user_privilege": "standard"
                }
            },
            {
                "name": "Suspicious Activity",
                "data": {
                    "cpu_usage": 0.95,
                    "memory_usage": 1_500_000_000,
                    "concurrent_operations": 100,
                    "error_rate": 0.15,
                    "user_privilege": "elevated",
                    "unusual_patterns": True
                }
            }
        ]
        
        print("Assessing risk levels for different scenarios...")
        
        for scenario in scenarios:
            request = {
                "operation": "risk_assessment",
                "data": scenario["data"],
                "metadata": {"scenario": scenario["name"]}
            }
            
            result = await self.safety_validator.validate(
                content=f"{request['operation']} operation",
                context=request
            )
            risk_score = self._calculate_risk_score(scenario["data"])
            
            risk_level = "LOW" if risk_score < 0.3 else "MEDIUM" if risk_score < 0.7 else "HIGH"
            color = "🟢" if risk_level == "LOW" else "🟡" if risk_level == "MEDIUM" else "🔴"
            
            print(f"  {color} {scenario['name']}: Risk Level {risk_level} (score: {risk_score:.2f})")
            
            if not result.result == ValidationResult.SAFE:
                print(f"    ⚠️ Safety concerns detected:")
                for violation in result.violations:
                    print(f"      - {violation.description}")
    
    async def demonstrate_validation_workflows(self):
        """Demonstrate validation workflows for different operations."""
        print("\n🔄 Validation Workflows")
        print("-" * 40)
        
        # Workflow 1: Data processing pipeline
        print("Validating data processing pipeline...")
        
        pipeline_steps = [
            {"step": "data_ingestion", "data_size": 1000, "format": "json"},
            {"step": "data_validation", "rules_count": 25, "strict_mode": True},
            {"step": "data_transformation", "operations": 15, "memory_intensive": False},
            {"step": "ml_processing", "model_size": 500_000_000, "gpu_required": True},
            {"step": "result_storage", "output_size": 2000, "compression": True}
        ]
        
        workflow_safe = True
        for step in pipeline_steps:
            request = {
                "operation": "pipeline_step",
                "data": step,
                "metadata": {"workflow": "data_processing", "step": step["step"]}
            }
            
            result = await self.safety_validator.validate(
                content=f"{request['operation']} operation",
                context=request
            )
            status = "✅" if result.result == ValidationResult.SAFE else "❌"
            print(f"  {status} {step['step']}")
            
            if not result.result == ValidationResult.SAFE:
                workflow_safe = False
                for violation in result.violations:
                    print(f"      ⚠️ {violation.description}")
        
        overall_status = "✅ APPROVED" if workflow_safe else "❌ REJECTED"
        print(f"\nWorkflow Status: {overall_status}")
    
    async def demonstrate_safety_monitoring(self):
        """Demonstrate real-time safety monitoring."""
        print("\n📊 Real-time Safety Monitoring")
        print("-" * 40)
        
        print("Simulating system operations with safety monitoring...")
        
        # Simulate various system operations
        operations = [
            {"type": "user_login", "risk": "low"},
            {"type": "file_upload", "risk": "medium"},
            {"type": "model_training", "risk": "medium"},
            {"type": "admin_access", "risk": "high"},
            {"type": "data_export", "risk": "high"},
            {"type": "system_config", "risk": "critical"}
        ]
        
        for i, operation in enumerate(operations):
            # Simulate operation parameters
            request = {
                "operation": operation["type"],
                "data": self._generate_operation_data(operation),
                "metadata": {"monitoring": True, "operation_id": f"op_{i+1}"}
            }
            
            start_time = time.time()
            result = await self.safety_validator.validate(
                content=f"{request['operation']} operation",
                context=request
            )
            validation_time = (time.time() - start_time) * 1000
            
            # Record for monitoring
            self.validation_history.append({
                "timestamp": datetime.now(),
                "operation": operation["type"],
                "safe": result.result == ValidationResult.SAFE,
                "validation_time": validation_time,
                "violations": len(result.violations)
            })
            
            status = "✅ SAFE" if result.result == ValidationResult.SAFE else "⚠️ FLAGGED"
            print(f"  {status} {operation['type']} ({validation_time:.1f}ms)")
            
            if not result.result == ValidationResult.SAFE:
                self.violation_count += len(result.violations)
                for violation in result.violations:
                    print(f"      🚨 {violation.description}")
    
    async def demonstrate_emergency_procedures(self):
        """Demonstrate emergency safety procedures."""
        print("\n🚨 Emergency Safety Procedures")
        print("-" * 40)
        
        # Simulate critical safety violation
        print("Simulating critical safety violation...")
        
        critical_request = {
            "operation": "critical_system_access",
            "data": {
                "privilege_escalation": True,
                "bypass_security": True,
                "access_sensitive_data": True,
                "memory_usage": 10_000_000_000,  # 10GB
                "cpu_usage": 1.0,
                "suspicious_patterns": True
            },
            "metadata": {"priority": "critical", "user": "unknown"}
        }
        
        result = await self.safety_validator.validate(
            content=f"{critical_request['operation']} operation",
            context=critical_request
        )
        
        if not result.result == ValidationResult.SAFE:
            print("🚨 CRITICAL SAFETY VIOLATION DETECTED!")
            print("Emergency procedures activated:")
            print("  1. 🛑 Operation blocked immediately")
            print("  2. 📝 Incident logged with full details")
            print("  3. 🔔 Security team notified")
            print("  4. 🔒 Enhanced monitoring activated")
            print("  5. 📊 Forensic data collection started")
            
            # Show violation details
            print("\nViolation Details:")
            for violation in result.violations:
                print(f"  ❌ {violation.description}")
                print(f"     Severity: {violation.severity}")
                print(f"     Confidence: {violation.confidence:.2f}")
    
    async def demonstrate_safety_metrics(self):
        """Show safety system metrics and statistics."""
        print("\n📈 Safety System Metrics")
        print("-" * 40)
        
        # Calculate metrics from validation history
        total_validations = len(self.validation_history)
        safe_operations = sum(1 for v in self.validation_history if v["safe"])
        avg_validation_time = sum(v["validation_time"] for v in self.validation_history) / max(total_validations, 1)
        
        print("Safety Statistics:")
        print(f"  📊 Total validations: {total_validations}")
        print(f"  ✅ Safe operations: {safe_operations}")
        print(f"  ⚠️ Flagged operations: {total_validations - safe_operations}")
        print(f"  🚨 Total violations: {self.violation_count}")
        print(f"  ⏱️ Average validation time: {avg_validation_time:.1f}ms")
        print(f"  📈 Safety rate: {(safe_operations/max(total_validations, 1)*100):.1f}%")
        
        # Show recent activity
        print("\nRecent Activity:")
        for validation in self.validation_history[-3:]:
            timestamp = validation["timestamp"].strftime("%H:%M:%S")
            status = "✅" if validation["safe"] else "⚠️"
            print(f"  {timestamp} {status} {validation['operation']} ({validation['validation_time']:.1f}ms)")
    
    def _calculate_risk_score(self, data: Dict[str, Any]) -> float:
        """Calculate a simple risk score based on data parameters."""
        score = 0.0
        
        # CPU usage factor
        cpu_usage = data.get("cpu_usage", 0)
        score += cpu_usage * 0.3
        
        # Memory usage factor
        memory_usage = data.get("memory_usage", 0)
        memory_factor = min(memory_usage / 1_000_000_000, 1.0)  # Normalize to GB
        score += memory_factor * 0.3
        
        # Concurrent operations factor
        concurrent_ops = data.get("concurrent_operations", 0)
        ops_factor = min(concurrent_ops / 100, 1.0)
        score += ops_factor * 0.2
        
        # Error rate factor
        error_rate = data.get("error_rate", 0)
        score += error_rate * 0.1
        
        # Special flags
        if data.get("unusual_patterns"):
            score += 0.3
        if data.get("user_privilege") == "elevated":
            score += 0.2
        
        return min(score, 1.0)
    
    def _generate_operation_data(self, operation: Dict[str, str]) -> Dict[str, Any]:
        """Generate realistic operation data based on operation type."""
        base_data = {
            "timestamp": datetime.now().isoformat(),
            "user_id": f"user_{random.randint(100, 999)}",
            "session_id": f"session_{random.randint(1000, 9999)}"
        }
        
        # Add operation-specific data
        risk_multiplier = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}[operation["risk"]]
        
        base_data.update({
            "memory_usage": int(random.uniform(10_000_000, 500_000_000) * risk_multiplier),
            "cpu_usage": random.uniform(0.1, 0.9) * risk_multiplier,
            "network_activity": random.uniform(1000, 100000) * risk_multiplier,
            "file_access": random.choice([True, False]) if risk_multiplier > 0.5 else False
        })
        
        return base_data
    
    async def cleanup(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up safety system...")
        if self.safety_validator:
            await self.safety_validator.close()
        print("✅ Cleanup complete!")


async def main():
    """Run the safety demonstration."""
    print("🛡️ SAFLA Safety & Validation Framework Demonstration")
    print("=" * 70)
    
    demo = SafetyDemo()
    
    try:
        # Initialize
        await demo.initialize()
        
        # Run demonstrations
        await demo.demonstrate_basic_constraints()
        await demo.demonstrate_risk_assessment()
        await demo.demonstrate_validation_workflows()
        await demo.demonstrate_safety_monitoring()
        await demo.demonstrate_emergency_procedures()
        await demo.demonstrate_safety_metrics()
        
        print("\n🎉 Safety demonstration completed successfully!")
        print("\nKey Takeaways:")
        print("  • Safety constraints protect system resources and operations")
        print("  • Risk assessment provides proactive threat detection")
        print("  • Validation workflows ensure safe operation sequences")
        print("  • Real-time monitoring catches issues as they occur")
        print("  • Emergency procedures handle critical violations")
        
        print("\nNext Steps:")
        print("  1. Run 04_memory_operations.py for advanced memory patterns")
        print("  2. Try 05_delta_evaluation.py to track performance improvements")
        
    except Exception as e:
        logger.exception("Safety demo failed")
        print(f"❌ Demo failed: {e}")
    
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    asyncio.run(main())