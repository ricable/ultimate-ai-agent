"""
Basic Workflow Example

This example demonstrates how to create and orchestrate multiple agents in workflows.
"""

import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime
from uap_sdk import UAPAgent, UAPClient, Configuration, CustomAgentBuilder, AgentFramework


class WorkflowOrchestrator:
    """Orchestrates multiple agents in workflows."""
    
    def __init__(self, config: Configuration):
        self.config = config
        self.agents: Dict[str, UAPAgent] = {}
        self.workflow_history: List[Dict[str, Any]] = []
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
    
    async def register_agent(self, agent_id: str, agent: UAPAgent) -> None:
        """Register an agent with the orchestrator."""
        self.agents[agent_id] = agent
        await agent.start()
        print(f"Agent '{agent_id}' registered and started")
    
    async def execute_workflow(self, workflow_id: str, workflow_definition: Dict[str, Any], 
                              initial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a workflow with multiple agents."""
        
        workflow_start = datetime.utcnow()
        workflow_data = initial_data or {}
        
        # Track workflow execution
        workflow_execution = {
            "workflow_id": workflow_id,
            "start_time": workflow_start.isoformat(),
            "steps": [],
            "data": workflow_data.copy(),
            "status": "running"
        }
        
        self.active_workflows[workflow_id] = workflow_execution
        
        try:
            steps = workflow_definition.get("steps", [])
            
            for step_idx, step in enumerate(steps):
                step_start = datetime.utcnow()
                step_id = step.get("id", f"step_{step_idx}")
                
                print(f"Executing step {step_idx + 1}/{len(steps)}: {step_id}")
                
                step_result = await self._execute_step(step, workflow_data)
                
                step_execution = {
                    "step_id": step_id,
                    "step_index": step_idx,
                    "start_time": step_start.isoformat(),
                    "end_time": datetime.utcnow().isoformat(),
                    "duration_ms": (datetime.utcnow() - step_start).total_seconds() * 1000,
                    "result": step_result,
                    "success": step_result.get("success", True)
                }
                
                workflow_execution["steps"].append(step_execution)
                
                # Update workflow data with step results
                if step_result.get("data"):
                    workflow_data.update(step_result["data"])
                
                # Handle step failure
                if not step_result.get("success", True):
                    workflow_execution["status"] = "failed"
                    workflow_execution["error"] = step_result.get("error", "Step failed")
                    break
                
                # Check for early termination conditions
                if step_result.get("terminate_workflow"):
                    workflow_execution["status"] = "terminated"
                    workflow_execution["termination_reason"] = step_result.get("termination_reason", "Workflow terminated by step")
                    break
            
            if workflow_execution["status"] == "running":
                workflow_execution["status"] = "completed"
            
        except Exception as e:
            workflow_execution["status"] = "error"
            workflow_execution["error"] = str(e)
            print(f"Workflow error: {e}")
        
        finally:
            workflow_execution["end_time"] = datetime.utcnow().isoformat()
            workflow_execution["total_duration_ms"] = (datetime.utcnow() - workflow_start).total_seconds() * 1000
            workflow_execution["final_data"] = workflow_data
            
            # Move to history
            self.workflow_history.append(workflow_execution)
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
        
        return workflow_execution
    
    async def _execute_step(self, step: Dict[str, Any], workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step."""
        step_type = step.get("type", "agent")
        
        if step_type == "agent":
            return await self._execute_agent_step(step, workflow_data)
        elif step_type == "condition":
            return await self._execute_condition_step(step, workflow_data)
        elif step_type == "parallel":
            return await self._execute_parallel_step(step, workflow_data)
        elif step_type == "transform":
            return await self._execute_transform_step(step, workflow_data)
        else:
            return {
                "success": False,
                "error": f"Unknown step type: {step_type}"
            }
    
    async def _execute_agent_step(self, step: Dict[str, Any], workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a step that calls an agent."""
        agent_id = step.get("agent_id")
        message_template = step.get("message", "")
        
        if agent_id not in self.agents:
            return {
                "success": False,
                "error": f"Agent '{agent_id}' not found"
            }
        
        # Format message with workflow data
        try:
            message = message_template.format(**workflow_data)
        except KeyError as e:
            return {
                "success": False,
                "error": f"Message template error: Missing variable {e}"
            }
        
        # Call agent
        agent = self.agents[agent_id]
        response = await agent.process_message(message, {"workflow_data": workflow_data})
        
        # Extract data from response
        result_data = {}
        output_mapping = step.get("output_mapping", {})
        
        for output_key, source_path in output_mapping.items():
            try:
                # Simple path extraction (e.g., "metadata.result")
                value = response
                for path_part in source_path.split("."):
                    value = value[path_part]
                result_data[output_key] = value
            except (KeyError, TypeError):
                print(f"Warning: Could not extract {source_path} from agent response")
        
        return {
            "success": True,
            "agent_response": response,
            "data": result_data
        }
    
    async def _execute_condition_step(self, step: Dict[str, Any], workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a conditional step."""
        condition = step.get("condition", "")
        true_action = step.get("true_action", {})
        false_action = step.get("false_action", {})
        
        try:
            # Simple condition evaluation (in production, use a safe evaluator)
            # This is a simplified example - real implementation should be more secure
            result = eval(condition, {"__builtins__": {}}, workflow_data)
            
            action = true_action if result else false_action
            
            if action:
                return await self._execute_step(action, workflow_data)
            else:
                return {
                    "success": True,
                    "condition_result": result,
                    "action_taken": "none"
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Condition evaluation error: {str(e)}"
            }
    
    async def _execute_parallel_step(self, step: Dict[str, Any], workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multiple steps in parallel."""
        parallel_steps = step.get("steps", [])
        
        if not parallel_steps:
            return {
                "success": True,
                "results": []
            }
        
        # Execute all steps concurrently
        tasks = [self._execute_step(parallel_step, workflow_data.copy()) for parallel_step in parallel_steps]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        success = True
        processed_results = []
        combined_data = {}
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "step_index": i
                })
                success = False
            else:
                processed_results.append(result)
                if result.get("data"):
                    combined_data.update(result["data"])
                if not result.get("success", True):
                    success = False
        
        return {
            "success": success,
            "results": processed_results,
            "data": combined_data
        }
    
    async def _execute_transform_step(self, step: Dict[str, Any], workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a data transformation step."""
        transformations = step.get("transformations", {})
        
        result_data = {}
        
        for output_key, transformation in transformations.items():
            try:
                # Simple transformation (in production, use a safe evaluator)
                result = eval(transformation, {"__builtins__": {}}, workflow_data)
                result_data[output_key] = result
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Transformation error for '{output_key}': {str(e)}"
                }
        
        return {
            "success": True,
            "data": result_data
        }
    
    async def cleanup(self) -> None:
        """Clean up all agents."""
        for agent in self.agents.values():
            await agent.stop()


# Example workflow definitions
CUSTOMER_SUPPORT_WORKFLOW = {
    "name": "Customer Support Workflow",
    "description": "Handle customer inquiries with multiple specialized agents",
    "steps": [
        {
            "id": "classify_inquiry",
            "type": "agent",
            "agent_id": "classifier",
            "message": "Classify this customer inquiry: {customer_message}",
            "output_mapping": {
                "category": "metadata.classification",
                "priority": "metadata.priority"
            }
        },
        {
            "id": "route_by_category",
            "type": "condition",
            "condition": "category == 'technical'",
            "true_action": {
                "type": "agent",
                "agent_id": "technical_support",
                "message": "Provide technical support for: {customer_message}",
                "output_mapping": {
                    "solution": "content",
                    "escalate": "metadata.needs_escalation"
                }
            },
            "false_action": {
                "type": "agent",
                "agent_id": "general_support",
                "message": "Provide general support for: {customer_message}",
                "output_mapping": {
                    "response": "content",
                    "satisfaction_likely": "metadata.satisfaction_score"
                }
            }
        },
        {
            "id": "generate_response",
            "type": "agent",
            "agent_id": "response_generator",
            "message": "Generate a professional customer response based on: {solution}{response}",
            "output_mapping": {
                "final_response": "content"
            }
        }
    ]
}

DATA_PROCESSING_WORKFLOW = {
    "name": "Data Processing Pipeline",
    "description": "Process and analyze data through multiple stages",
    "steps": [
        {
            "id": "validate_data",
            "type": "agent",
            "agent_id": "data_validator",
            "message": "Validate this data: {input_data}",
            "output_mapping": {
                "is_valid": "metadata.valid",
                "cleaned_data": "metadata.cleaned_data"
            }
        },
        {
            "id": "check_validation",
            "type": "condition",
            "condition": "is_valid == True",
            "true_action": {
                "type": "parallel",
                "steps": [
                    {
                        "type": "agent",
                        "agent_id": "statistical_analyzer",
                        "message": "Perform statistical analysis on: {cleaned_data}",
                        "output_mapping": {
                            "statistics": "metadata.stats"
                        }
                    },
                    {
                        "type": "agent",
                        "agent_id": "pattern_detector",
                        "message": "Detect patterns in: {cleaned_data}",
                        "output_mapping": {
                            "patterns": "metadata.patterns"
                        }
                    }
                ]
            },
            "false_action": {
                "type": "transform",
                "transformations": {
                    "error_message": "'Data validation failed'",
                    "terminate_workflow": "True",
                    "termination_reason": "'Invalid input data'"
                }
            }
        },
        {
            "id": "generate_report",
            "type": "agent",
            "agent_id": "report_generator",
            "message": "Generate a report with statistics: {statistics} and patterns: {patterns}",
            "output_mapping": {
                "report": "content"
            }
        }
    ]
}


# Mock agents for demonstration
class MockClassifierAgent(AgentFramework):
    """Mock agent that classifies customer inquiries."""
    
    def __init__(self, config: Configuration = None):
        super().__init__("classifier", config)
    
    async def initialize(self) -> None:
        self.is_initialized = True
        self.status = "active"
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Simple classification logic
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["bug", "error", "crash", "technical", "not working"]):
            category = "technical"
            priority = "high"
        elif any(word in message_lower for word in ["billing", "payment", "invoice", "charge"]):
            category = "billing"
            priority = "medium"
        else:
            category = "general"
            priority = "low"
        
        return {
            "content": f"Classified as: {category} (Priority: {priority})",
            "metadata": {
                "classification": category,
                "priority": priority,
                "confidence": 0.85
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        return {"status": self.status, "framework": self.framework_name}


class MockSupportAgent(AgentFramework):
    """Mock support agent."""
    
    def __init__(self, agent_type: str, config: Configuration = None):
        super().__init__(agent_type, config)
        self.agent_type = agent_type
    
    async def initialize(self) -> None:
        self.is_initialized = True
        self.status = "active"
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if self.agent_type == "technical_support":
            return {
                "content": "I've analyzed your technical issue and found a solution: Please try restarting the application and clearing your cache.",
                "metadata": {
                    "needs_escalation": False,
                    "solution_confidence": 0.9
                }
            }
        elif self.agent_type == "general_support":
            return {
                "content": "Thank you for contacting us. I understand your concern and I'm here to help you resolve this issue.",
                "metadata": {
                    "satisfaction_score": 0.8
                }
            }
        elif self.agent_type == "response_generator":
            return {
                "content": "Dear Customer,\n\nThank you for reaching out to us. Based on our analysis, we have identified a solution for your inquiry. Please try restarting the application and clearing your cache. If the issue persists, please don't hesitate to contact us again.\n\nBest regards,\nCustomer Support Team"
            }
        else:
            return {"content": f"Response from {self.agent_type}"}
    
    def get_status(self) -> Dict[str, Any]:
        return {"status": self.status, "framework": self.framework_name}


async def demo_customer_support_workflow():
    """Demonstrate customer support workflow."""
    print("=== Customer Support Workflow Demo ===\n")
    
    config = Configuration()
    orchestrator = WorkflowOrchestrator(config)
    
    # Create and register mock agents
    classifier = UAPAgent("classifier", MockClassifierAgent(config), config)
    technical_support = UAPAgent("technical_support", MockSupportAgent("technical_support", config), config)
    general_support = UAPAgent("general_support", MockSupportAgent("general_support", config), config)
    response_generator = UAPAgent("response_generator", MockSupportAgent("response_generator", config), config)
    
    await orchestrator.register_agent("classifier", classifier)
    await orchestrator.register_agent("technical_support", technical_support)
    await orchestrator.register_agent("general_support", general_support)
    await orchestrator.register_agent("response_generator", response_generator)
    
    # Test different types of customer inquiries
    test_inquiries = [
        "The application keeps crashing when I try to save my work",
        "I have a question about my billing statement",
        "How do I change my password?"
    ]
    
    for i, inquiry in enumerate(test_inquiries, 1):
        print(f"Processing inquiry {i}: {inquiry}")
        
        result = await orchestrator.execute_workflow(
            f"support_inquiry_{i}",
            CUSTOMER_SUPPORT_WORKFLOW,
            {"customer_message": inquiry}
        )
        
        print(f"Workflow status: {result['status']}")
        print(f"Total duration: {result['total_duration_ms']:.2f}ms")
        print(f"Steps executed: {len(result['steps'])}")
        
        if result['status'] == 'completed':
            final_response = result['final_data'].get('final_response', 'No response generated')
            print(f"Final response:\n{final_response}")
        
        print("\n" + "="*60 + "\n")
    
    await orchestrator.cleanup()


async def demo_parallel_workflow():
    """Demonstrate parallel workflow execution."""
    print("=== Parallel Workflow Demo ===\n")
    
    config = Configuration()
    orchestrator = WorkflowOrchestrator(config)
    
    # Create agents for parallel processing
    agent1 = UAPAgent("agent1", MockSupportAgent("analyzer1", config), config)
    agent2 = UAPAgent("agent2", MockSupportAgent("analyzer2", config), config)
    agent3 = UAPAgent("agent3", MockSupportAgent("summarizer", config), config)
    
    await orchestrator.register_agent("agent1", agent1)
    await orchestrator.register_agent("agent2", agent2)
    await orchestrator.register_agent("agent3", agent3)
    
    # Define a workflow with parallel execution
    parallel_workflow = {
        "name": "Parallel Analysis",
        "steps": [
            {
                "id": "parallel_analysis",
                "type": "parallel",
                "steps": [
                    {
                        "type": "agent",
                        "agent_id": "agent1",
                        "message": "Analyze aspect 1 of: {data}",
                        "output_mapping": {"analysis1": "content"}
                    },
                    {
                        "type": "agent",
                        "agent_id": "agent2",
                        "message": "Analyze aspect 2 of: {data}",
                        "output_mapping": {"analysis2": "content"}
                    }
                ]
            },
            {
                "id": "combine_results",
                "type": "agent",
                "agent_id": "agent3",
                "message": "Combine analyses: {analysis1} and {analysis2}",
                "output_mapping": {"final_result": "content"}
            }
        ]
    }
    
    result = await orchestrator.execute_workflow(
        "parallel_demo",
        parallel_workflow,
        {"data": "Sample data for analysis"}
    )
    
    print(f"Parallel workflow completed in {result['total_duration_ms']:.2f}ms")
    print(f"Status: {result['status']}")
    
    # Show step details
    for step in result['steps']:
        print(f"Step '{step['step_id']}': {step['duration_ms']:.2f}ms")
    
    await orchestrator.cleanup()


async def main():
    """Main function demonstrating workflows."""
    print("=== UAP Workflow Examples ===\n")
    
    await demo_customer_support_workflow()
    await demo_parallel_workflow()


if __name__ == "__main__":
    asyncio.run(main())