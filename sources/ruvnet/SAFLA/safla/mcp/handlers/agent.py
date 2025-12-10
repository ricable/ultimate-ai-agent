"""
Agent interaction handlers for SAFLA MCP server.

This module provides tools for managing agent sessions, interactions,
and multi-agent coordination.
"""

import asyncio
import uuid
import time
from typing import Any, Dict, List, Optional, Set
import logging
from datetime import datetime, timedelta
from collections import defaultdict

from .base import BaseHandler, ToolDefinition

logger = logging.getLogger(__name__)


class AgentHandler(BaseHandler):
    """Handler for agent interaction and coordination tools."""
    
    def _initialize_tools(self) -> None:
        """Initialize agent interaction tools."""
        tools = [
            ToolDefinition(
                name="create_agent_session",
                description="Create a new agent session",
                input_schema={
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "description": "Name of the agent"
                        },
                        "agent_type": {
                            "type": "string",
                            "description": "Type of agent",
                            "enum": ["assistant", "specialist", "coordinator", "evaluator"]
                        },
                        "capabilities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Agent capabilities"
                        },
                        "config": {
                            "type": "object",
                            "description": "Agent configuration"
                        }
                    },
                    "required": ["agent_name", "agent_type"]
                },
                handler_method="_create_agent_session"
            ),
            ToolDefinition(
                name="interact_with_agent",
                description="Send a message to an agent and get response",
                input_schema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Agent session ID"
                        },
                        "message": {
                            "type": "string",
                            "description": "Message to send"
                        },
                        "context": {
                            "type": "object",
                            "description": "Additional context"
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "description": "Response timeout"
                        }
                    },
                    "required": ["session_id", "message"]
                },
                handler_method="_interact_with_agent"
            ),
            ToolDefinition(
                name="list_agent_sessions",
                description="List all active agent sessions",
                input_schema={
                    "type": "object",
                    "properties": {
                        "agent_type": {
                            "type": "string",
                            "description": "Filter by agent type"
                        },
                        "status": {
                            "type": "string",
                            "description": "Filter by status",
                            "enum": ["active", "idle", "busy", "terminated"]
                        }
                    },
                    "required": []
                },
                handler_method="_list_agent_sessions"
            ),
            ToolDefinition(
                name="terminate_agent_session",
                description="Terminate an agent session",
                input_schema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID to terminate"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Termination reason"
                        }
                    },
                    "required": ["session_id"]
                },
                handler_method="_terminate_agent_session"
            ),
            ToolDefinition(
                name="coordinate_agents",
                description="Coordinate multiple agents for a task",
                input_schema={
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Task description"
                        },
                        "agent_sessions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Agent session IDs to coordinate"
                        },
                        "coordination_type": {
                            "type": "string",
                            "description": "Type of coordination",
                            "enum": ["sequential", "parallel", "hierarchical", "consensus"]
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "description": "Overall timeout"
                        }
                    },
                    "required": ["task", "agent_sessions"]
                },
                handler_method="_coordinate_agents"
            ),
            ToolDefinition(
                name="get_agent_capabilities",
                description="Get capabilities of available agents",
                input_schema={
                    "type": "object",
                    "properties": {
                        "capability_filter": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by specific capabilities"
                        }
                    },
                    "required": []
                },
                handler_method="_get_agent_capabilities"
            )
        ]
        
        for tool in tools:
            self.register_tool(tool)
    
    async def _create_agent_session(self, agent_name: str,
                                  agent_type: str,
                                  capabilities: Optional[List[str]] = None,
                                  config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new agent session."""
        try:
            session_id = f"agent_{uuid.uuid4().hex[:12]}"
            
            # Create agent session
            session_data = {
                "session_id": session_id,
                "agent_name": agent_name,
                "agent_type": agent_type,
                "capabilities": capabilities or [],
                "config": config or {},
                "status": "active",
                "created_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat(),
                "interactions": 0,
                "context": {},
                "performance_metrics": {
                    "response_time_avg": 0,
                    "success_rate": 1.0,
                    "tokens_processed": 0
                }
            }
            
            # Store session
            self.state_manager.set(
                session_id, session_data,
                namespace="agent_sessions"
            )
            
            # Register agent capabilities
            self._register_agent_capabilities(session_id, capabilities)
            
            logger.info(f"Created agent session: {session_id} for {agent_name}")
            
            return {
                "session_id": session_id,
                "agent_name": agent_name,
                "agent_type": agent_type,
                "status": "active",
                "message": f"Agent session created successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to create agent session: {str(e)}")
            raise
    
    async def _interact_with_agent(self, session_id: str,
                                 message: str,
                                 context: Optional[Dict[str, Any]] = None,
                                 timeout_seconds: int = 30) -> Dict[str, Any]:
        """Interact with an agent."""
        try:
            # Get agent session
            session = self.state_manager.get(session_id, namespace="agent_sessions")
            
            if not session:
                return {
                    "error": f"Agent session not found: {session_id}",
                    "status": "not_found"
                }
            
            if session["status"] != "active":
                return {
                    "error": f"Agent is not active: {session['status']}",
                    "status": "unavailable"
                }
            
            # Update session status
            session["status"] = "busy"
            session["last_activity"] = datetime.utcnow().isoformat()
            self.state_manager.set(session_id, session, namespace="agent_sessions")
            
            # Simulate agent processing
            interaction_id = f"int_{uuid.uuid4().hex[:8]}"
            start_time = time.time()
            
            try:
                # Process based on agent type
                response = await self._process_agent_message(
                    session, message, context or {}
                )
                
                # Update metrics
                processing_time = time.time() - start_time
                session["interactions"] += 1
                session["performance_metrics"]["response_time_avg"] = (
                    (session["performance_metrics"]["response_time_avg"] * 
                     (session["interactions"] - 1) + processing_time) / 
                    session["interactions"]
                )
                session["performance_metrics"]["tokens_processed"] += len(message.split())
                
                # Store interaction
                interaction_data = {
                    "interaction_id": interaction_id,
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": message,
                    "response": response,
                    "context": context,
                    "processing_time": processing_time
                }
                
                self.state_manager.set(
                    interaction_id, interaction_data,
                    namespace="agent_interactions",
                    ttl=86400  # Keep for 24 hours
                )
                
                # Update session
                session["status"] = "active"
                self.state_manager.set(session_id, session, namespace="agent_sessions")
                
                return {
                    "interaction_id": interaction_id,
                    "session_id": session_id,
                    "response": response,
                    "processing_time": processing_time,
                    "status": "success"
                }
                
            except asyncio.TimeoutError:
                session["status"] = "active"
                self.state_manager.set(session_id, session, namespace="agent_sessions")
                return {
                    "error": "Agent response timeout",
                    "status": "timeout"
                }
                
        except Exception as e:
            logger.error(f"Agent interaction failed: {str(e)}")
            raise
    
    async def _list_agent_sessions(self, agent_type: Optional[str] = None,
                                 status: Optional[str] = None) -> Dict[str, Any]:
        """List agent sessions."""
        try:
            all_sessions = self.state_manager.get_namespace("agent_sessions")
            
            # Filter sessions
            sessions = []
            for sid, session in all_sessions.items():
                if agent_type and session.get("agent_type") != agent_type:
                    continue
                if status and session.get("status") != status:
                    continue
                
                # Add summary info
                session_info = {
                    "session_id": sid,
                    "agent_name": session.get("agent_name"),
                    "agent_type": session.get("agent_type"),
                    "status": session.get("status"),
                    "created_at": session.get("created_at"),
                    "last_activity": session.get("last_activity"),
                    "interactions": session.get("interactions", 0),
                    "capabilities": session.get("capabilities", [])
                }
                sessions.append(session_info)
            
            # Sort by last activity
            sessions.sort(key=lambda s: s.get("last_activity", ""), reverse=True)
            
            return {
                "sessions": sessions,
                "total": len(sessions),
                "filters": {
                    "agent_type": agent_type,
                    "status": status
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to list agent sessions: {str(e)}")
            raise
    
    async def _terminate_agent_session(self, session_id: str,
                                     reason: Optional[str] = None) -> Dict[str, Any]:
        """Terminate an agent session."""
        try:
            session = self.state_manager.get(session_id, namespace="agent_sessions")
            
            if not session:
                return {
                    "error": f"Agent session not found: {session_id}",
                    "status": "not_found"
                }
            
            # Update session
            session["status"] = "terminated"
            session["terminated_at"] = datetime.utcnow().isoformat()
            session["termination_reason"] = reason or "User requested"
            
            # Keep terminated sessions for audit
            self.state_manager.set(
                session_id, session,
                namespace="agent_sessions",
                ttl=3600  # Keep for 1 hour
            )
            
            # Unregister capabilities
            self._unregister_agent_capabilities(session_id)
            
            logger.info(f"Terminated agent session: {session_id}")
            
            return {
                "session_id": session_id,
                "status": "success",
                "message": f"Agent session terminated",
                "reason": reason
            }
            
        except Exception as e:
            logger.error(f"Failed to terminate agent session: {str(e)}")
            raise
    
    async def _coordinate_agents(self, task: str,
                               agent_sessions: List[str],
                               coordination_type: str = "parallel",
                               timeout_seconds: int = 300) -> Dict[str, Any]:
        """Coordinate multiple agents."""
        try:
            coordination_id = f"coord_{uuid.uuid4().hex[:8]}"
            start_time = time.time()
            
            # Validate agents
            valid_agents = []
            for session_id in agent_sessions:
                session = self.state_manager.get(session_id, namespace="agent_sessions")
                if session and session["status"] == "active":
                    valid_agents.append(session_id)
            
            if not valid_agents:
                return {
                    "error": "No valid active agents found",
                    "status": "no_agents"
                }
            
            # Create coordination context
            coordination_context = {
                "coordination_id": coordination_id,
                "task": task,
                "type": coordination_type,
                "agents": valid_agents,
                "started_at": datetime.utcnow().isoformat(),
                "status": "running"
            }
            
            self.state_manager.set(
                coordination_id, coordination_context,
                namespace="agent_coordination"
            )
            
            # Execute coordination
            results = await self._execute_coordination(
                task, valid_agents, coordination_type, timeout_seconds
            )
            
            # Update coordination status
            coordination_context["status"] = "completed"
            coordination_context["completed_at"] = datetime.utcnow().isoformat()
            coordination_context["duration_seconds"] = time.time() - start_time
            coordination_context["results"] = results
            
            self.state_manager.set(
                coordination_id, coordination_context,
                namespace="agent_coordination"
            )
            
            return {
                "coordination_id": coordination_id,
                "task": task,
                "agents_coordinated": len(valid_agents),
                "coordination_type": coordination_type,
                "duration_seconds": coordination_context["duration_seconds"],
                "results": results,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Agent coordination failed: {str(e)}")
            raise
    
    async def _get_agent_capabilities(self, 
                                    capability_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get agent capabilities."""
        try:
            capabilities_map = self.state_manager.get(
                "agent_capabilities_map",
                namespace="agent_registry",
                default={}
            )
            
            if capability_filter:
                # Filter capabilities
                filtered_map = {}
                for cap in capability_filter:
                    if cap in capabilities_map:
                        filtered_map[cap] = capabilities_map[cap]
                capabilities_map = filtered_map
            
            # Get agent details for each capability
            capability_details = {}
            for capability, agent_ids in capabilities_map.items():
                agents = []
                for agent_id in agent_ids:
                    session = self.state_manager.get(
                        agent_id, namespace="agent_sessions"
                    )
                    if session and session["status"] == "active":
                        agents.append({
                            "session_id": agent_id,
                            "agent_name": session["agent_name"],
                            "agent_type": session["agent_type"]
                        })
                
                if agents:
                    capability_details[capability] = agents
            
            return {
                "capabilities": capability_details,
                "total_capabilities": len(capability_details),
                "filter_applied": capability_filter is not None
            }
            
        except Exception as e:
            logger.error(f"Failed to get agent capabilities: {str(e)}")
            raise
    
    async def _process_agent_message(self, session: Dict[str, Any],
                                   message: str,
                                   context: Dict[str, Any]) -> str:
        """Process message based on agent type."""
        agent_type = session["agent_type"]
        
        # Simulate different agent behaviors
        if agent_type == "assistant":
            # General purpose assistant
            response = f"I'll help you with: {message[:50]}..."
            await asyncio.sleep(0.5)  # Simulate processing
            
        elif agent_type == "specialist":
            # Domain specialist
            domain = session.get("config", {}).get("domain", "general")
            response = f"As a {domain} specialist, regarding '{message[:30]}...': [specialized response]"
            await asyncio.sleep(0.8)
            
        elif agent_type == "coordinator":
            # Task coordinator
            response = f"Coordinating task: {message[:40]}... [delegation plan]"
            await asyncio.sleep(0.3)
            
        elif agent_type == "evaluator":
            # Quality evaluator
            response = f"Evaluating: {message[:30]}... Score: 8.5/10 [detailed feedback]"
            await asyncio.sleep(0.6)
            
        else:
            response = f"Processing: {message[:50]}..."
            await asyncio.sleep(0.4)
        
        return response
    
    def _register_agent_capabilities(self, session_id: str, 
                                   capabilities: Optional[List[str]]) -> None:
        """Register agent capabilities for discovery."""
        if not capabilities:
            return
        
        # Get capabilities map
        cap_map = self.state_manager.get(
            "agent_capabilities_map",
            namespace="agent_registry",
            default=defaultdict(set)
        )
        
        # Register each capability
        for capability in capabilities:
            if capability not in cap_map:
                cap_map[capability] = set()
            cap_map[capability].add(session_id)
        
        # Convert sets to lists for JSON serialization
        cap_map_serializable = {k: list(v) for k, v in cap_map.items()}
        
        # Store updated map
        self.state_manager.set(
            "agent_capabilities_map",
            cap_map_serializable,
            namespace="agent_registry"
        )
    
    def _unregister_agent_capabilities(self, session_id: str) -> None:
        """Unregister agent capabilities."""
        # Get capabilities map
        cap_map = self.state_manager.get(
            "agent_capabilities_map",
            namespace="agent_registry",
            default={}
        )
        
        # Remove agent from all capabilities
        updated_map = {}
        for capability, agents in cap_map.items():
            if isinstance(agents, list):
                agents = set(agents)
            agents.discard(session_id)
            if agents:
                updated_map[capability] = list(agents)
        
        # Store updated map
        self.state_manager.set(
            "agent_capabilities_map",
            updated_map,
            namespace="agent_registry"
        )
    
    async def _execute_coordination(self, task: str,
                                  agents: List[str],
                                  coord_type: str,
                                  timeout: int) -> Dict[str, Any]:
        """Execute agent coordination based on type."""
        results = {
            "agents": {},
            "consensus": None,
            "summary": None
        }
        
        if coord_type == "sequential":
            # Sequential execution
            current_context = {"task": task}
            for agent_id in agents:
                result = await self._interact_with_agent(
                    agent_id, task, current_context, timeout // len(agents)
                )
                results["agents"][agent_id] = result
                if result.get("status") == "success":
                    current_context["previous_response"] = result["response"]
            
        elif coord_type == "parallel":
            # Parallel execution
            tasks = []
            for agent_id in agents:
                task_coro = self._interact_with_agent(
                    agent_id, task, {"task": task}, timeout
                )
                tasks.append(task_coro)
            
            agent_results = await asyncio.gather(*tasks, return_exceptions=True)
            for agent_id, result in zip(agents, agent_results):
                if isinstance(result, Exception):
                    results["agents"][agent_id] = {
                        "error": str(result),
                        "status": "error"
                    }
                else:
                    results["agents"][agent_id] = result
        
        elif coord_type == "hierarchical":
            # Hierarchical execution (coordinator first)
            coordinator = agents[0]  # First agent is coordinator
            coord_result = await self._interact_with_agent(
                coordinator, f"Coordinate: {task}", {"task": task}, timeout // 2
            )
            results["agents"][coordinator] = coord_result
            
            # Then workers in parallel
            if len(agents) > 1:
                worker_tasks = []
                for agent_id in agents[1:]:
                    context = {
                        "task": task,
                        "coordinator_guidance": coord_result.get("response", "")
                    }
                    task_coro = self._interact_with_agent(
                        agent_id, task, context, timeout // 2
                    )
                    worker_tasks.append(task_coro)
                
                worker_results = await asyncio.gather(*worker_tasks, return_exceptions=True)
                for agent_id, result in zip(agents[1:], worker_results):
                    if isinstance(result, Exception):
                        results["agents"][agent_id] = {
                            "error": str(result),
                            "status": "error"
                        }
                    else:
                        results["agents"][agent_id] = result
        
        elif coord_type == "consensus":
            # Consensus building
            # First round - gather opinions
            opinion_tasks = []
            for agent_id in agents:
                task_coro = self._interact_with_agent(
                    agent_id, f"Opinion on: {task}", {"task": task}, timeout // 3
                )
                opinion_tasks.append(task_coro)
            
            opinions = await asyncio.gather(*opinion_tasks, return_exceptions=True)
            
            # Second round - build consensus
            consensus_context = {
                "task": task,
                "opinions": [
                    op.get("response", "") for op in opinions 
                    if not isinstance(op, Exception)
                ]
            }
            
            consensus_tasks = []
            for agent_id in agents:
                task_coro = self._interact_with_agent(
                    agent_id, "Build consensus", consensus_context, timeout // 3
                )
                consensus_tasks.append(task_coro)
            
            consensus_results = await asyncio.gather(*consensus_tasks, return_exceptions=True)
            
            for i, agent_id in enumerate(agents):
                results["agents"][agent_id] = {
                    "opinion": opinions[i] if not isinstance(opinions[i], Exception) else {"error": str(opinions[i])},
                    "consensus": consensus_results[i] if not isinstance(consensus_results[i], Exception) else {"error": str(consensus_results[i])}
                }
            
            # Synthesize consensus
            results["consensus"] = "Consensus achieved through multi-agent deliberation"
        
        # Generate summary
        successful_responses = [
            r["response"] for r in results["agents"].values()
            if r.get("status") == "success"
        ]
        
        results["summary"] = f"Coordinated {len(agents)} agents. {len(successful_responses)} successful responses."
        
        return results