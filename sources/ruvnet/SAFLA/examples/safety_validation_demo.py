"""
Safety and Validation Framework Demo

This example demonstrates the comprehensive Safety and Validation Framework
for SAFLA, showing how to configure and use all components:

1. Safety Constraints - Hard limits and soft boundaries
2. Validation Pipeline - Multi-stage validation
3. Risk Assessment - Quantitative risk scoring
4. Rollback Mechanisms - Safe reversion to previous states
5. Monitoring and Alerts - Real-time safety monitoring

Based on research findings from research/04_synthesis/01_integrated_model.md
"""

import asyncio
import time
import logging
from typing import Dict, Any

from safla.core.safety_validation import (
    SafetyValidationFramework,
    SafetyConstraint,
    ConstraintType,
    ValidationStage,
    ValidationResult,
    RiskFactor,
    AlertLevel,
    SafetyAlert
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SafetyValidationDemo:
    """Comprehensive demo of the Safety and Validation Framework."""
    
    def __init__(self):
        self.framework = SafetyValidationFramework()
        self.system_state = {
            "cpu_usage": 30,
            "memory_usage": 500_000_000,  # 500MB
            "active_processes": 5,
            "error_rate": 0.01,
            "status": "stable"
        }
        self.setup_framework()
    
    def setup_framework(self):
        """Configure the safety framework with realistic constraints and validators."""
        
        # 1. Configure Safety Constraints
        self.setup_safety_constraints()
        
        # 2. Configure Validation Pipeline
        self.setup_validation_pipeline()
        
        # 3. Configure Risk Assessment
        self.setup_risk_assessment()
        
        # 4. Configure Monitoring
        self.setup_monitoring()
        
        # 5. Configure Framework Functions
        self.setup_framework_functions()
    
    def setup_safety_constraints(self):
        """Setup safety constraints for system protection."""
        
        # Hard constraint: Memory usage limit (cannot be exceeded)
        memory_constraint = SafetyConstraint(
            name="memory_limit",
            constraint_type=ConstraintType.HARD,
            description="Maximum memory usage limit - 2GB",
            rule="memory_usage <= 2000000000",
            threshold=2_000_000_000,  # 2GB
            violation_action="emergency_stop"
        )
        
        # Hard constraint: CPU usage limit
        cpu_constraint = SafetyConstraint(
            name="cpu_limit",
            constraint_type=ConstraintType.HARD,
            description="Maximum CPU usage limit - 90%",
            rule="cpu_usage <= 90",
            threshold=90,
            violation_action="rollback"
        )
        
        # Soft constraint: Error rate warning
        error_rate_constraint = SafetyConstraint(
            name="error_rate_warning",
            constraint_type=ConstraintType.SOFT,
            description="Error rate warning threshold - 5%",
            rule="error_rate <= 0.05",
            threshold=0.05,
            violation_action="warn"
        )
        
        # Soft constraint: Process count limit
        process_constraint = SafetyConstraint(
            name="process_limit",
            constraint_type=ConstraintType.SOFT,
            description="Maximum active processes - 20",
            rule="active_processes <= 20",
            threshold=20,
            violation_action="warn"
        )
        
        # Add constraints to engine
        self.framework.constraint_engine.add_constraint(memory_constraint)
        self.framework.constraint_engine.add_constraint(cpu_constraint)
        self.framework.constraint_engine.add_constraint(error_rate_constraint)
        self.framework.constraint_engine.add_constraint(process_constraint)
        
        logger.info("Safety constraints configured")
    
    def setup_validation_pipeline(self):
        """Setup multi-stage validation pipeline."""
        
        # Stage 1: Syntax validation
        def syntax_validator(data: Dict[str, Any]) -> ValidationResult:
            """Validate that all required fields are present and have correct types."""
            required_fields = ["cpu_usage", "memory_usage", "active_processes", "error_rate"]
            
            for field in required_fields:
                if field not in data:
                    return ValidationResult(
                        stage="syntax",
                        is_valid=False,
                        message=f"Missing required field: {field}",
                        details={"missing_field": field}
                    )
                
                if not isinstance(data[field], (int, float)):
                    return ValidationResult(
                        stage="syntax",
                        is_valid=False,
                        message=f"Invalid type for field {field}: expected number",
                        details={"invalid_field": field, "type": type(data[field]).__name__}
                    )
            
            return ValidationResult(
                stage="syntax",
                is_valid=True,
                message="Syntax validation passed",
                details={}
            )
        
        # Stage 2: Semantic validation
        def semantic_validator(data: Dict[str, Any]) -> ValidationResult:
            """Validate that values are within reasonable ranges."""
            
            # Check for reasonable value ranges
            if data.get("cpu_usage", 0) < 0 or data.get("cpu_usage", 0) > 100:
                return ValidationResult(
                    stage="semantic",
                    is_valid=False,
                    message="CPU usage must be between 0 and 100",
                    details={"cpu_usage": data.get("cpu_usage")}
                )
            
            if data.get("memory_usage", 0) < 0:
                return ValidationResult(
                    stage="semantic",
                    is_valid=False,
                    message="Memory usage cannot be negative",
                    details={"memory_usage": data.get("memory_usage")}
                )
            
            if data.get("active_processes", 0) < 0:
                return ValidationResult(
                    stage="semantic",
                    is_valid=False,
                    message="Active processes cannot be negative",
                    details={"active_processes": data.get("active_processes")}
                )
            
            if data.get("error_rate", 0) < 0 or data.get("error_rate", 0) > 1:
                return ValidationResult(
                    stage="semantic",
                    is_valid=False,
                    message="Error rate must be between 0 and 1",
                    details={"error_rate": data.get("error_rate")}
                )
            
            return ValidationResult(
                stage="semantic",
                is_valid=True,
                message="Semantic validation passed",
                details={}
            )
        
        # Stage 3: Performance validation
        def performance_validator(data: Dict[str, Any]) -> ValidationResult:
            """Validate that the change won't cause performance degradation."""
            
            current_cpu = self.system_state.get("cpu_usage", 0)
            new_cpu = data.get("cpu_usage", current_cpu)
            cpu_increase = new_cpu - current_cpu
            
            current_memory = self.system_state.get("memory_usage", 0)
            new_memory = data.get("memory_usage", current_memory)
            memory_increase = new_memory - current_memory
            
            # Check for significant performance impact
            if cpu_increase > 30:  # More than 30% CPU increase
                return ValidationResult(
                    stage="performance",
                    is_valid=False,
                    message=f"CPU increase too high: {cpu_increase}%",
                    details={"cpu_increase": cpu_increase}
                )
            
            if memory_increase > 1_000_000_000:  # More than 1GB memory increase
                return ValidationResult(
                    stage="performance",
                    is_valid=False,
                    message=f"Memory increase too high: {memory_increase} bytes",
                    details={"memory_increase": memory_increase}
                )
            
            return ValidationResult(
                stage="performance",
                is_valid=True,
                message="Performance validation passed",
                details={"cpu_increase": cpu_increase, "memory_increase": memory_increase}
            )
        
        # Add validation stages
        self.framework.validation_pipeline.add_stage(ValidationStage(
            name="syntax",
            description="Syntax and type validation",
            validator=syntax_validator,
            required=True
        ))
        
        self.framework.validation_pipeline.add_stage(ValidationStage(
            name="semantic",
            description="Semantic range validation",
            validator=semantic_validator,
            required=True
        ))
        
        self.framework.validation_pipeline.add_stage(ValidationStage(
            name="performance",
            description="Performance impact validation",
            validator=performance_validator,
            required=True
        ))
        
        logger.info("Validation pipeline configured")
    
    def setup_risk_assessment(self):
        """Setup risk assessment with multiple risk factors."""
        
        # Risk Factor 1: Performance Impact
        performance_risk = RiskFactor(
            name="performance_impact",
            description="Risk from performance degradation",
            weight=0.3,
            calculator=lambda data: min(
                (data.get("cpu_usage", 0) / 100) * 0.7 + 
                (data.get("memory_usage", 0) / 2_000_000_000) * 0.3,
                1.0
            )
        )
        
        # Risk Factor 2: Stability Risk
        stability_risk = RiskFactor(
            name="stability_risk",
            description="Risk from system instability",
            weight=0.4,
            calculator=lambda data: min(
                data.get("error_rate", 0) * 10 +  # Error rate impact
                max(0, (data.get("active_processes", 0) - 10) / 20),  # Process count impact
                1.0
            )
        )
        
        # Risk Factor 3: Resource Exhaustion Risk
        resource_risk = RiskFactor(
            name="resource_exhaustion",
            description="Risk from resource exhaustion",
            weight=0.3,
            calculator=lambda data: min(
                max(
                    data.get("cpu_usage", 0) / 100,
                    data.get("memory_usage", 0) / 2_000_000_000
                ),
                1.0
            )
        )
        
        # Add risk factors
        self.framework.risk_scorer.add_risk_factor(performance_risk)
        self.framework.risk_scorer.add_risk_factor(stability_risk)
        self.framework.risk_scorer.add_risk_factor(resource_risk)
        
        logger.info("Risk assessment configured")
    
    def setup_monitoring(self):
        """Setup safety monitoring with alert handlers."""
        
        # Configure safety thresholds
        self.framework.safety_monitor.safety_thresholds = {
            "cpu_usage": {"warning": 70, "critical": 85},
            "memory_usage": {"warning": 1_500_000_000, "critical": 1_800_000_000},
            "error_rate": {"warning": 0.03, "critical": 0.08},
            "active_processes": {"warning": 15, "critical": 25}
        }
        
        # Add alert handlers
        def warning_handler(alert: SafetyAlert):
            logger.warning(f"⚠️  WARNING: {alert.message} - {alert.metric_name}: {alert.current_value}")
        
        def critical_handler(alert: SafetyAlert):
            logger.critical(f"🚨 CRITICAL: {alert.message} - {alert.metric_name}: {alert.current_value}")
            # In a real system, this might trigger automatic remediation
        
        self.framework.safety_monitor.add_alert_handler(AlertLevel.WARNING, warning_handler)
        self.framework.safety_monitor.add_alert_handler(AlertLevel.CRITICAL, critical_handler)
        
        # Configure system metrics getter
        self.framework.safety_monitor.get_system_metrics = lambda: self.system_state.copy()
        
        logger.info("Safety monitoring configured")
    
    def setup_framework_functions(self):
        """Setup framework integration functions."""
        
        # System state getter
        self.framework.get_system_state = lambda: self.system_state.copy()
        
        # Rollback function
        def rollback_function(state: Dict[str, Any]) -> bool:
            logger.info(f"🔄 Rolling back to state: {state}")
            self.system_state.update(state)
            return True
        
        self.framework.rollback_function = rollback_function
        
        # Emergency stop function
        def emergency_stop() -> bool:
            logger.critical("🛑 EMERGENCY STOP ACTIVATED")
            self.system_state["status"] = "emergency_stopped"
            return True
        
        self.framework.emergency_stop_function = emergency_stop
        
        logger.info("Framework functions configured")
    
    async def demonstrate_safety_constraints(self):
        """Demonstrate safety constraint validation."""
        logger.info("\n" + "="*50)
        logger.info("DEMONSTRATING SAFETY CONSTRAINTS")
        logger.info("="*50)
        
        # Test 1: Valid modification (should pass)
        logger.info("\n1. Testing valid modification...")
        valid_modification = {
            "cpu_usage": 45,
            "memory_usage": 800_000_000,
            "active_processes": 8,
            "error_rate": 0.02
        }
        
        result = await self.framework.validate_system_modification(valid_modification)
        logger.info(f"✅ Valid modification result: Approved = {result.is_approved}")
        logger.info(f"   Risk score: {result.risk_assessment.overall_score:.3f}")
        
        # Test 2: Constraint violation (should fail)
        logger.info("\n2. Testing constraint violation...")
        violating_modification = {
            "cpu_usage": 95,  # Exceeds CPU limit
            "memory_usage": 800_000_000,
            "active_processes": 8,
            "error_rate": 0.02
        }
        
        result = await self.framework.validate_system_modification(violating_modification)
        logger.info(f"❌ Violating modification result: Approved = {result.is_approved}")
        logger.info(f"   Violations: {len(result.constraint_violations)}")
        for violation in result.constraint_violations:
            logger.info(f"   - {violation.constraint_name}: {violation.message}")
    
    async def demonstrate_validation_pipeline(self):
        """Demonstrate multi-stage validation pipeline."""
        logger.info("\n" + "="*50)
        logger.info("DEMONSTRATING VALIDATION PIPELINE")
        logger.info("="*50)
        
        # Test 1: Valid data (should pass all stages)
        logger.info("\n1. Testing valid data through pipeline...")
        valid_data = {
            "cpu_usage": 60,
            "memory_usage": 1_000_000_000,
            "active_processes": 12,
            "error_rate": 0.025
        }
        
        results = await self.framework.validation_pipeline.run(valid_data)
        logger.info(f"✅ Pipeline results: {len(results)} stages completed")
        for result in results:
            status = "✅ PASS" if result.is_valid else "❌ FAIL"
            logger.info(f"   {result.stage}: {status} - {result.message}")
        
        # Test 2: Invalid data (should fail at semantic stage)
        logger.info("\n2. Testing invalid data through pipeline...")
        invalid_data = {
            "cpu_usage": 150,  # Invalid: > 100%
            "memory_usage": 1_000_000_000,
            "active_processes": 12,
            "error_rate": 0.025
        }
        
        results = await self.framework.validation_pipeline.run(invalid_data)
        logger.info(f"❌ Pipeline results: {len(results)} stages completed")
        for result in results:
            status = "✅ PASS" if result.is_valid else "❌ FAIL"
            logger.info(f"   {result.stage}: {status} - {result.message}")
    
    async def demonstrate_risk_assessment(self):
        """Demonstrate risk assessment system."""
        logger.info("\n" + "="*50)
        logger.info("DEMONSTRATING RISK ASSESSMENT")
        logger.info("="*50)
        
        test_scenarios = [
            {
                "name": "Low Risk Scenario",
                "data": {"cpu_usage": 30, "memory_usage": 500_000_000, "active_processes": 5, "error_rate": 0.01}
            },
            {
                "name": "Medium Risk Scenario",
                "data": {"cpu_usage": 60, "memory_usage": 1_200_000_000, "active_processes": 15, "error_rate": 0.04}
            },
            {
                "name": "High Risk Scenario",
                "data": {"cpu_usage": 85, "memory_usage": 1_800_000_000, "active_processes": 25, "error_rate": 0.08}
            }
        ]
        
        for scenario in test_scenarios:
            logger.info(f"\n{scenario['name']}:")
            assessment = self.framework.risk_scorer.calculate_risk(scenario['data'])
            
            risk_level = "LOW" if assessment.overall_score <= 0.3 else "MEDIUM" if assessment.overall_score <= 0.6 else "HIGH"
            acceptable = "✅ ACCEPTABLE" if assessment.is_acceptable() else "❌ UNACCEPTABLE"
            
            logger.info(f"   Overall Risk Score: {assessment.overall_score:.3f} ({risk_level}) - {acceptable}")
            logger.info("   Factor Breakdown:")
            for factor, score in assessment.factor_scores.items():
                logger.info(f"     - {factor}: {score:.3f}")
    
    async def demonstrate_rollback_system(self):
        """Demonstrate rollback and checkpoint system."""
        logger.info("\n" + "="*50)
        logger.info("DEMONSTRATING ROLLBACK SYSTEM")
        logger.info("="*50)
        
        # Create initial checkpoint
        logger.info("\n1. Creating safety checkpoint...")
        checkpoint_id = await self.framework.create_safety_checkpoint(
            "stable_state",
            "Stable system state before modifications"
        )
        logger.info(f"✅ Checkpoint created: {checkpoint_id}")
        logger.info(f"   Current state: {self.system_state}")
        
        # Simulate system modification
        logger.info("\n2. Simulating system modification...")
        self.system_state.update({
            "cpu_usage": 75,
            "memory_usage": 1_500_000_000,
            "active_processes": 18,
            "error_rate": 0.06
        })
        logger.info(f"   Modified state: {self.system_state}")
        
        # Rollback to checkpoint
        logger.info("\n3. Rolling back to checkpoint...")
        success = await self.framework.rollback_to_safe_state(checkpoint_id)
        logger.info(f"✅ Rollback successful: {success}")
        logger.info(f"   Restored state: {self.system_state}")
    
    async def demonstrate_monitoring_system(self):
        """Demonstrate real-time monitoring and alerts."""
        logger.info("\n" + "="*50)
        logger.info("DEMONSTRATING MONITORING SYSTEM")
        logger.info("="*50)
        
        # Start monitoring
        logger.info("\n1. Starting safety monitoring...")
        await self.framework.safety_monitor.start_monitoring()
        
        # Simulate gradual system degradation
        logger.info("\n2. Simulating system state changes...")
        
        scenarios = [
            {"cpu_usage": 50, "memory_usage": 1_000_000_000, "description": "Normal operation"},
            {"cpu_usage": 75, "memory_usage": 1_600_000_000, "description": "Warning thresholds exceeded"},
            {"cpu_usage": 90, "memory_usage": 1_900_000_000, "description": "Critical thresholds exceeded"}
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            logger.info(f"\n   Scenario {i}: {scenario['description']}")
            self.system_state.update({
                "cpu_usage": scenario["cpu_usage"],
                "memory_usage": scenario["memory_usage"]
            })
            
            # Let monitoring system detect changes
            await asyncio.sleep(0.2)
        
        # Stop monitoring
        logger.info("\n3. Stopping safety monitoring...")
        await self.framework.safety_monitor.stop_monitoring()
    
    async def demonstrate_emergency_stop(self):
        """Demonstrate emergency stop functionality."""
        logger.info("\n" + "="*50)
        logger.info("DEMONSTRATING EMERGENCY STOP")
        logger.info("="*50)
        
        logger.info("\n1. Current system status:")
        logger.info(f"   Status: {self.system_state.get('status', 'unknown')}")
        logger.info(f"   Framework active: {self.framework.is_active}")
        
        logger.info("\n2. Triggering emergency stop...")
        success = await self.framework.emergency_stop("Demo emergency stop")
        
        logger.info(f"\n3. Emergency stop result: {success}")
        logger.info(f"   Status: {self.system_state.get('status', 'unknown')}")
        logger.info(f"   Framework active: {self.framework.is_active}")
    
    async def run_complete_demo(self):
        """Run the complete safety validation framework demo."""
        logger.info("🚀 Starting Safety and Validation Framework Demo")
        logger.info("="*60)
        
        try:
            # Start the framework
            await self.framework.start()
            
            # Run all demonstrations
            await self.demonstrate_safety_constraints()
            await self.demonstrate_validation_pipeline()
            await self.demonstrate_risk_assessment()
            await self.demonstrate_rollback_system()
            await self.demonstrate_monitoring_system()
            await self.demonstrate_emergency_stop()
            
            logger.info("\n" + "="*60)
            logger.info("✅ Safety and Validation Framework Demo Complete!")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"❌ Demo failed with error: {e}")
            raise
        finally:
            # Ensure framework is stopped
            await self.framework.stop()


async def main():
    """Main demo function."""
    demo = SafetyValidationDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())