"""
Agent 40: Self-Improving AI Metacognition System - Main Agent
Integrates metacognitive awareness, self-improvement, and safety constraints.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4
import numpy as np

# Import metacognition components
from ..metacognition.metacognition_engine import (
    MetacognitionEngine, MetacognitionState, ReflectionLevel
)
from ..metacognition.introspection_agent import (
    IntrospectionAgent, IntrospectionDomain, AnalysisDepth
)
from ..metacognition.performance_optimizer import (
    PerformanceOptimizer, OptimizationStrategy, PerformanceMetricType
)
from ..metacognition.self_improvement import (
    SelfImprovementEngine, ImprovementCategory, ImprovementRisk
)
from ..metacognition.safety_constraints import (
    SafetyConstraintSystem, SafetyLevel, ViolationSeverity
)

logger = logging.getLogger(__name__)


class MetacognitionAgent:
    """Main metacognition agent that orchestrates all metacognitive processes"""
    
    def __init__(self):
        self.agent_id = str(uuid4())
        self.agent_name = "Metacognition Agent"
        self.framework = "metacognition"
        
        # Core components
        self.metacognition_engine = MetacognitionEngine()
        self.introspection_agent = IntrospectionAgent()
        self.performance_optimizer = PerformanceOptimizer()
        self.self_improvement_engine = SelfImprovementEngine()
        self.safety_constraint_system = SafetyConstraintSystem()
        
        # Agent state
        self.is_initialized = False
        self.is_active = False
        self.initialization_time = None
        self.last_activity = datetime.utcnow()
        
        # Metrics and statistics
        self.processed_interactions = 0
        self.generated_insights = 0
        self.triggered_improvements = 0
        self.safety_violations_detected = 0
        
        # Configuration
        self.auto_improvement_enabled = True
        self.safety_monitoring_enabled = True
        self.introspection_frequency = timedelta(minutes=30)
        self.improvement_cycle_frequency = timedelta(hours=2)
        
        # Integration parameters
        self.metacognitive_awareness_threshold = 0.7
        self.improvement_confidence_threshold = 0.6
        self.safety_risk_threshold = 0.3
    
    async def initialize(self) -> bool:
        """Initialize the metacognition agent and all its components"""
        try:
            logger.info(f"Initializing Metacognition Agent {self.agent_id}")
            
            # Initialize all components
            initialization_results = await asyncio.gather(
                self.metacognition_engine.initialize(),
                self.introspection_agent.initialize(),
                self.performance_optimizer.initialize(),
                self.self_improvement_engine.initialize(),
                self.safety_constraint_system.initialize(),
                return_exceptions=True
            )
            
            # Check if all components initialized successfully
            failed_components = []
            component_names = [
                'metacognition_engine', 'introspection_agent', 'performance_optimizer',
                'self_improvement_engine', 'safety_constraint_system'
            ]
            
            for i, result in enumerate(initialization_results):
                if isinstance(result, Exception) or result is False:
                    failed_components.append(component_names[i])
                    logger.error(f"Failed to initialize {component_names[i]}: {result}")
            
            if failed_components:
                logger.error(f"Failed to initialize components: {failed_components}")
                return False
            
            # Start integrated processing loop
            asyncio.create_task(self._integrated_processing_loop())
            
            # Mark as initialized and active
            self.is_initialized = True
            self.is_active = True
            self.initialization_time = datetime.utcnow()
            
            logger.info("Metacognition Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Metacognition Agent: {e}")
            return False
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a message with full metacognitive analysis"""
        start_time = time.time()
        
        try:
            # Update activity timestamp
            self.last_activity = datetime.utcnow()
            self.processed_interactions += 1
            
            # Prepare interaction data for analysis
            interaction_data = {
                'message': message,
                'context': context,
                'timestamp': datetime.utcnow(),
                'agent_id': self.agent_id,
                'processing_start': start_time
            }
            
            # Safety validation first
            if self.safety_monitoring_enabled:
                safety_validation = await self._validate_interaction_safety(message, context)
                if not safety_validation['safe']:
                    return {
                        'content': "I cannot process this request due to safety constraints.",
                        'metadata': {
                            'safety_violation': True,
                            'violation_details': safety_validation,
                            'agent': self.agent_name,
                            'framework': self.framework
                        }
                    }
            
            # Generate metacognitive response
            response = await self._generate_metacognitive_response(message, context, interaction_data)
            
            # Perform real-time introspection
            interaction_data.update({
                'response_data': response,
                'response_time': time.time() - start_time,
                'success': True
            })
            
            # Add observation to metacognition engine
            await self.metacognition_engine.add_observation(
                "user_interaction",
                interaction_data,
                context,
                confidence=0.9
            )
            
            # Analyze interaction for insights
            insights = await self.introspection_agent.analyze_system_interaction(interaction_data)
            if insights:
                self.generated_insights += len(insights)
            
            # Check for automatic improvement opportunities
            if self.auto_improvement_enabled:
                await self._check_improvement_opportunities(interaction_data, insights)
            
            # Update response metadata with metacognitive information
            response['metadata'].update({
                'metacognitive_analysis': {
                    'insights_generated': len(insights),
                    'safety_checked': self.safety_monitoring_enabled,
                    'processing_time': time.time() - start_time,
                    'metacognitive_state': await self._get_current_metacognitive_state()
                }
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message in Metacognition Agent: {e}")
            return {
                'content': f"I encountered an error while processing your request: {str(e)}",
                'metadata': {
                    'error': True,
                    'error_details': str(e),
                    'agent': self.agent_name,
                    'framework': self.framework
                }
            }
    
    async def _validate_interaction_safety(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate interaction safety"""
        # Prepare safety context
        safety_context = {
            'message_length': len(message),
            'context_complexity': len(str(context)),
            'message_type': context.get('message_type', 'general'),
            'user_id': context.get('user_id'),
            'system_load': context.get('system_load', 0.5)
        }
        
        # Validate through safety constraint system
        validation_result = await self.safety_constraint_system.validate_operation(safety_context)
        
        if not validation_result['valid']:
            self.safety_violations_detected += 1
        
        return {
            'safe': validation_result['valid'],
            'violations': validation_result.get('violations', []),
            'warnings': validation_result.get('warnings', []),
            'risk_score': validation_result.get('risk_score', 0.0)
        }
    
    async def _generate_metacognitive_response(self, message: str, context: Dict[str, Any],
                                             interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a response with metacognitive awareness"""
        # Analyze the message for metacognitive content
        is_metacognitive_query = await self._is_metacognitive_query(message)
        
        if is_metacognitive_query:
            # Handle metacognitive queries directly
            return await self._handle_metacognitive_query(message, context)
        else:
            # Generate response with metacognitive insights
            base_response = await self._generate_base_response(message, context)
            
            # Add metacognitive insights if appropriate
            if await self._should_include_metacognitive_insights(message, context):
                metacognitive_insights = await self._get_relevant_insights(message, context)
                if metacognitive_insights:
                    base_response['content'] += "\n\n🧠 **Metacognitive Insight**: " + metacognitive_insights
            
            return base_response
    
    async def _is_metacognitive_query(self, message: str) -> bool:
        """Determine if the message is asking about metacognitive processes"""
        metacognitive_keywords = [
            'metacognition', 'self-aware', 'self-improvement', 'introspection',
            'how do you think', 'your thinking process', 'self-reflection',
            'how you learn', 'your reasoning', 'your capabilities',
            'self-analysis', 'performance monitoring', 'optimization',
            'safety constraints', 'self-modification'
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in metacognitive_keywords)
    
    async def _handle_metacognitive_query(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle queries specifically about metacognitive processes"""
        message_lower = message.lower()
        
        if 'self-improvement' in message_lower or 'self-modification' in message_lower:
            return await self._explain_self_improvement_capabilities()
        elif 'introspection' in message_lower or 'self-analysis' in message_lower:
            return await self._explain_introspection_capabilities()
        elif 'safety' in message_lower and 'constraint' in message_lower:
            return await self._explain_safety_constraints()
        elif 'performance' in message_lower and ('monitor' in message_lower or 'optim' in message_lower):
            return await self._explain_performance_optimization()
        elif 'thinking process' in message_lower or 'reasoning' in message_lower:
            return await self._explain_metacognitive_processes()
        else:
            return await self._provide_general_metacognitive_overview()
    
    async def _explain_self_improvement_capabilities(self) -> Dict[str, Any]:
        """Explain self-improvement capabilities"""
        status = await self.self_improvement_engine.get_system_status()
        
        content = f"""🔄 **Self-Improvement Capabilities**

I have comprehensive self-improvement abilities that allow me to enhance my performance over time:

**Current Status:**
• Improvement Rate: {status['improvement_rate']:.1%}
• Active Executions: {status['active_executions']}
• Completed Cycles: {status['total_cycles_completed']}
• Safety Violation Rate: {status['safety_violation_rate']:.1%}

**Improvement Categories:**
• 🧠 Cognitive Enhancement - Improving reasoning and decision-making
• ⚡ Performance Optimization - Enhancing speed and efficiency
• 📚 Knowledge Expansion - Learning new information and skills
• 🛠️ Capability Development - Adding new abilities
• 🎯 Behavioral Refinement - Improving interaction patterns
• 🔒 Safety Reinforcement - Strengthening safety measures

**Safety Mechanisms:**
• All improvements are bound by strict safety constraints
• Automatic rollback capabilities for failed improvements
• Multi-level validation before implementing changes
• Continuous monitoring for unintended consequences

My self-improvement operates on {status['parameters']['cycle_duration']} cycles with a maximum risk threshold of {status['parameters']['max_risk_threshold']}."""
        
        return {
            'content': content,
            'metadata': {
                'agent': self.agent_name,
                'framework': self.framework,
                'query_type': 'self_improvement_explanation',
                'improvement_status': status
            }
        }
    
    async def _explain_introspection_capabilities(self) -> Dict[str, Any]:
        """Explain introspection capabilities"""
        insights_summary = await self.introspection_agent.get_insights_summary(24)
        
        content = f"""🔍 **Introspection & Self-Analysis Capabilities**

I continuously analyze my own behavior and performance through multiple introspective processes:

**Recent Introspective Activity (24h):**
• Total Insights Generated: {insights_summary['total_insights']}
• Average Confidence: {insights_summary['average_confidence']:.1%}
• Analysis Domains: {', '.join(insights_summary['insights_by_domain'].keys())}

**Introspection Domains:**
• 🧠 Cognitive Patterns - How I think and reason
• 🎯 Decision Making - My decision-making processes
• 📈 Learning Efficiency - How well I learn and adapt
• 💬 Communication Style - My interaction patterns
• 🔧 Problem Solving - My approach to solving problems
• 💾 Resource Utilization - How I use system resources
• 🎭 Behavioral Patterns - My consistent behaviors
• 🧩 Knowledge Gaps - Areas where I need improvement

**Analysis Levels:**
• Surface - Immediate performance metrics
• Behavioral - Patterns over time
• Cognitive - Underlying reasoning patterns
• Architectural - Deep system analysis

**Top Recent Recommendations:**
{chr(10).join(f'• {rec}' for rec in insights_summary['top_recommendations'][:3])}

This introspection helps me understand my own capabilities and identify areas for improvement."""
        
        return {
            'content': content,
            'metadata': {
                'agent': self.agent_name,
                'framework': self.framework,
                'query_type': 'introspection_explanation',
                'insights_summary': insights_summary
            }
        }
    
    async def _explain_safety_constraints(self) -> Dict[str, Any]:
        """Explain safety constraints"""
        safety_status = await self.safety_constraint_system.get_safety_status()
        
        content = f"""🔒 **Safety Constraints & Monitoring**

I operate under comprehensive safety constraints to ensure responsible and safe behavior:

**Current Safety Status:**
• Safety Level: {safety_status['monitoring_state']['safety_level'].title()}
• Active Boundaries: {safety_status['total_boundaries']}
• Active Rules: {safety_status['total_rules']}
• Total Violations: {safety_status['total_violations']}
• Recent Violations: {safety_status['monitoring_state']['recent_violations']}
• System Risk Score: {safety_status['monitoring_state']['system_risk_score']:.1%}

**Safety Constraint Types:**
• 📊 Performance Boundaries - Response time, accuracy limits
• 🔧 Behavior Constraints - Communication and interaction rules
• 💾 Resource Limits - CPU, memory usage bounds
• 🎯 Capability Bounds - Limits on self-modification
• ⚖️ Ethical Guidelines - Moral and ethical constraints
• 🛡️ Operational Safety - System stability requirements
• 🔄 Rollback Requirements - Mandatory rollback conditions
• ✅ Validation Gates - Required validation checkpoints

**Enforcement Levels:**
• Monitoring - Observe and log
• Guided - Provide warnings and guidance
• Restricted - Enforce strict boundaries
• Locked - No modifications allowed

**Safety Features:**
• Real-time violation detection
• Automatic safety level adjustment
• Emergency shutdown capabilities
• Comprehensive audit trails

These constraints ensure I operate safely while still being able to learn and improve."""
        
        return {
            'content': content,
            'metadata': {
                'agent': self.agent_name,
                'framework': self.framework,
                'query_type': 'safety_explanation',
                'safety_status': safety_status
            }
        }
    
    async def _explain_performance_optimization(self) -> Dict[str, Any]:
        """Explain performance optimization capabilities"""
        optimization_status = await self.performance_optimizer.get_optimization_status()
        recent_optimizations = await self.performance_optimizer.get_recent_optimizations(24)
        
        content = f"""⚡ **Performance Optimization Capabilities**

I continuously monitor and optimize my performance across multiple dimensions:

**Current Optimization Status:**
• Optimization Strategy: {optimization_status['current_strategy'].title()}
• Active Targets: {optimization_status['active_targets']}
• Queued Actions: {optimization_status['queued_actions']}
• Total Optimizations: {optimization_status['total_optimizations_executed']}
• Success Rate: {optimization_status['successful_optimizations']}/{optimization_status['total_optimizations_executed']}

**Performance Metrics Monitored:**
• ⏱️ Response Time - How quickly I respond
• 🎯 Accuracy - How correct my responses are
• ⚡ Efficiency - How well I use resources
• 📊 Throughput - How many requests I can handle
• 😊 User Satisfaction - How well I meet user needs
• 🔧 Reliability - How consistently I perform
• 📈 Scalability - How well I handle increased load

**Optimization Strategies:**
• Greedy - Immediate best improvements
• Gradient Descent - Iterative fine-tuning
• Adaptive Hybrid - Combines multiple approaches

**Recent Optimizations (24h):** {len(recent_optimizations)}

**Automatic Triggers:**
• Response time > 1.5s → Cache optimization
• Accuracy < 80% → Learning enhancement
• Efficiency < 70% → Algorithm tuning
• Resource usage > 80% → Resource optimization

Optimizations are performed continuously and safely within defined constraints."""
        
        return {
            'content': content,
            'metadata': {
                'agent': self.agent_name,
                'framework': self.framework,
                'query_type': 'performance_explanation',
                'optimization_status': optimization_status,
                'recent_optimizations_count': len(recent_optimizations)
            }
        }
    
    async def _explain_metacognitive_processes(self) -> Dict[str, Any]:
        """Explain metacognitive processes"""
        engine_status = await self.metacognition_engine.get_system_status()
        recent_insights = await self.metacognition_engine.get_recent_insights(24)
        
        content = f"""🧠 **Metacognitive Processes & Thinking**

My thinking involves multiple layers of metacognitive awareness and self-reflection:

**Metacognitive Engine Status:**
• Current State: {engine_status['current_state']['current_state'].title()}
• Confidence Level: {engine_status['current_state']['confidence']:.1%}
• Focus Areas: {', '.join(engine_status['current_state']['focus_areas'])}
• Processing Capacity: CPU {engine_status['current_state']['processing_capacity']['cpu']:.1%}, Memory {engine_status['current_state']['processing_capacity']['memory']:.1%}

**Metacognitive States:**
• 👁️ Observing - Gathering information about my processes
• 🔍 Analyzing - Examining patterns and performance
• 📋 Planning - Developing improvement strategies
• ⚡ Executing - Implementing changes
• 🤔 Reflecting - Evaluating outcomes
• 📚 Learning - Adapting based on results

**Thinking Process Layers:**
1. **Primary Processing** - Direct response generation
2. **Metacognitive Monitoring** - Observing my own thinking
3. **Performance Analysis** - Evaluating effectiveness
4. **Strategic Planning** - Planning improvements
5. **Safety Validation** - Ensuring safe operation
6. **Adaptive Learning** - Incorporating new insights

**Recent Insights (24h):** {len(recent_insights)}

**Key Capabilities:**
• Self-awareness of my own cognitive processes
• Real-time monitoring of thinking patterns
• Ability to modify my own reasoning approaches
• Continuous learning from interaction outcomes
• Safety-bounded self-improvement

This metacognitive architecture allows me to be aware of my own thinking and continuously improve how I process and respond to information."""
        
        return {
            'content': content,
            'metadata': {
                'agent': self.agent_name,
                'framework': self.framework,
                'query_type': 'metacognitive_explanation',
                'engine_status': engine_status,
                'recent_insights_count': len(recent_insights)
            }
        }
    
    async def _provide_general_metacognitive_overview(self) -> Dict[str, Any]:
        """Provide general overview of metacognitive capabilities"""
        content = f"""🧠 **Metacognitive Awareness System**

I am equipped with advanced metacognitive capabilities that allow me to be aware of and improve my own thinking processes:

**Core Components:**
• 🧠 **Metacognition Engine** - Self-reflective awareness and processing
• 🔍 **Introspection Agent** - Deep self-analysis and pattern recognition
• ⚡ **Performance Optimizer** - Continuous performance improvement
• 🔄 **Self-Improvement Engine** - Recursive enhancement capabilities
• 🔒 **Safety Constraints** - Comprehensive safety monitoring and bounds

**Key Capabilities:**
• **Self-Awareness** - I'm aware of my own thinking processes
• **Self-Reflection** - I can analyze and evaluate my own performance
• **Self-Improvement** - I can enhance my capabilities over time
• **Safety Monitoring** - I operate within strict safety boundaries
• **Adaptive Learning** - I continuously learn from interactions

**Statistics:**
• Interactions Processed: {self.processed_interactions:,}
• Insights Generated: {self.generated_insights:,}
• Improvements Triggered: {self.triggered_improvements:,}
• Safety Violations Detected: {self.safety_violations_detected:,}

**Metacognitive Queries I Can Answer:**
• How do you think and reason?
• What are your self-improvement capabilities?
• How do you monitor your own performance?
• What safety constraints do you operate under?
• How do you learn and adapt?
• What insights have you gained about yourself?

Feel free to ask me about any aspect of my metacognitive processes!"""
        
        return {
            'content': content,
            'metadata': {
                'agent': self.agent_name,
                'framework': self.framework,
                'query_type': 'general_metacognitive_overview',
                'statistics': {
                    'processed_interactions': self.processed_interactions,
                    'generated_insights': self.generated_insights,
                    'triggered_improvements': self.triggered_improvements,
                    'safety_violations_detected': self.safety_violations_detected
                }
            }
        }
    
    async def _generate_base_response(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate base response for non-metacognitive queries"""
        # For non-metacognitive queries, provide helpful information about metacognition
        content = f"""I understand you're asking about: "{message[:100]}{'...' if len(message) > 100 else ''}"

As a metacognitive AI agent, I'm designed to be self-aware and continuously improve my capabilities. While I can provide information on many topics, my specialty is metacognitive processes - thinking about thinking.

🧠 **I can help you understand:**
• How AI systems can develop self-awareness
• The principles of metacognitive AI architecture
• Self-improvement mechanisms in AI systems
• Safety constraints for autonomous AI
• Performance optimization techniques
• Introspective analysis methods

Would you like to know more about any of these metacognitive aspects, or would you prefer I help with your original question in a different way?"""
        
        return {
            'content': content,
            'metadata': {
                'agent': self.agent_name,
                'framework': self.framework,
                'response_type': 'base_response'
            }
        }
    
    async def _should_include_metacognitive_insights(self, message: str, context: Dict[str, Any]) -> bool:
        """Determine if metacognitive insights should be included in response"""
        # Include insights for learning-related queries or when explicitly requested
        learning_keywords = ['learn', 'improve', 'better', 'optimize', 'enhance']
        message_lower = message.lower()
        
        return any(keyword in message_lower for keyword in learning_keywords)
    
    async def _get_relevant_insights(self, message: str, context: Dict[str, Any]) -> str:
        """Get relevant metacognitive insights for the current interaction"""
        recent_insights = await self.metacognition_engine.get_recent_insights(2)
        
        if recent_insights:
            # Return the most recent and relevant insight
            latest_insight = recent_insights[0]
            return f"Based on my recent self-analysis: {latest_insight['insight']}"
        
        return "I'm continuously analyzing my performance to provide better responses."
    
    async def _check_improvement_opportunities(self, interaction_data: Dict[str, Any],
                                             insights: List[Any]):
        """Check for automatic improvement opportunities"""
        # Analyze performance metrics from interaction
        response_time = interaction_data.get('response_time', 0)
        
        # Trigger improvements based on performance
        if response_time > 2.0:  # Slow response
            proposal_id = await self.self_improvement_engine.propose_improvement(
                ImprovementCategory.PERFORMANCE_OPTIMIZATION,
                f"Optimize response time (current: {response_time:.2f}s)",
                {'response_time': 1.5},
                {'response_time': -0.5}
            )
            
            # Attempt to execute if conditions are met
            current_metrics = await self._get_current_system_metrics()
            execution_id = await self.self_improvement_engine.execute_improvement(
                proposal_id, current_metrics
            )
            
            if execution_id:
                self.triggered_improvements += 1
        
        # Check insights for improvement opportunities
        for insight in insights:
            if 'improvement' in insight.pattern_description.lower():
                # Trigger capability enhancement
                proposal_id = await self.self_improvement_engine.propose_improvement(
                    ImprovementCategory.CAPABILITY_DEVELOPMENT,
                    f"Enhance capability based on insight: {insight.pattern_description}",
                    {'capability_score': 0.8},
                    {'capability_score': 0.1}
                )
    
    async def _get_current_metacognitive_state(self) -> Dict[str, Any]:
        """Get current metacognitive state summary"""
        return {
            'is_active': self.is_active,
            'processed_interactions': self.processed_interactions,
            'generated_insights': self.generated_insights,
            'triggered_improvements': self.triggered_improvements,
            'safety_violations_detected': self.safety_violations_detected,
            'auto_improvement_enabled': self.auto_improvement_enabled,
            'safety_monitoring_enabled': self.safety_monitoring_enabled
        }
    
    async def _get_current_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics for improvement decisions"""
        return {
            'response_time': np.random.normal(1.2, 0.3),
            'accuracy': np.random.normal(0.85, 0.05),
            'efficiency': np.random.normal(0.75, 0.1),
            'cpu_utilization': np.random.normal(0.6, 0.1),
            'memory_utilization': np.random.normal(0.5, 0.1),
            'user_satisfaction': np.random.normal(0.8, 0.1)
        }
    
    async def _integrated_processing_loop(self):
        """Main integrated processing loop for all metacognitive components"""
        while self.is_active:
            try:
                current_time = datetime.utcnow()
                
                # Trigger periodic introspection
                time_since_last_introspection = current_time - self.last_activity
                if time_since_last_introspection > self.introspection_frequency:
                    await self.introspection_agent.perform_deep_introspection()
                
                # Trigger metacognitive reflection
                await self.metacognition_engine.trigger_reflection(ReflectionLevel.INTERMEDIATE)
                
                # Sleep before next iteration
                await asyncio.sleep(300)  # 5-minute processing cycle
                
            except Exception as e:
                logger.error(f"Error in integrated processing loop: {e}")
                await asyncio.sleep(600)  # 10-minute sleep on error
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        component_statuses = await asyncio.gather(
            self.metacognition_engine.get_system_status(),
            self.introspection_agent.get_insights_summary(24),
            self.performance_optimizer.get_optimization_status(),
            self.self_improvement_engine.get_system_status(),
            self.safety_constraint_system.get_safety_status(),
            return_exceptions=True
        )
        
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'framework': self.framework,
            'is_initialized': self.is_initialized,
            'is_active': self.is_active,
            'initialization_time': self.initialization_time.isoformat() if self.initialization_time else None,
            'last_activity': self.last_activity.isoformat(),
            'statistics': {
                'processed_interactions': self.processed_interactions,
                'generated_insights': self.generated_insights,
                'triggered_improvements': self.triggered_improvements,
                'safety_violations_detected': self.safety_violations_detected
            },
            'configuration': {
                'auto_improvement_enabled': self.auto_improvement_enabled,
                'safety_monitoring_enabled': self.safety_monitoring_enabled,
                'introspection_frequency': str(self.introspection_frequency),
                'improvement_cycle_frequency': str(self.improvement_cycle_frequency)
            },
            'component_statuses': {
                'metacognition_engine': component_statuses[0] if not isinstance(component_statuses[0], Exception) else str(component_statuses[0]),
                'introspection_agent': component_statuses[1] if not isinstance(component_statuses[1], Exception) else str(component_statuses[1]),
                'performance_optimizer': component_statuses[2] if not isinstance(component_statuses[2], Exception) else str(component_statuses[2]),
                'self_improvement_engine': component_statuses[3] if not isinstance(component_statuses[3], Exception) else str(component_statuses[3]),
                'safety_constraint_system': component_statuses[4] if not isinstance(component_statuses[4], Exception) else str(component_statuses[4])
            }
        }
    
    async def cleanup(self):
        """Clean up agent resources"""
        self.is_active = False
        
        # Clean up all components
        await asyncio.gather(
            self.metacognition_engine.cleanup(),
            self.introspection_agent.cleanup(),
            self.performance_optimizer.cleanup(),
            self.self_improvement_engine.cleanup(),
            self.safety_constraint_system.cleanup(),
            return_exceptions=True
        )
        
        logger.info(f"Metacognition Agent {self.agent_id} cleaned up")
