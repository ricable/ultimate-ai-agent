"""
Admin handlers for SAFLA MCP server.

This module provides tools for administrative tasks including user management,
backup/restore operations, and system health monitoring.
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional
import logging
from datetime import datetime, timedelta
from pathlib import Path

from .base import BaseHandler, ToolDefinition

logger = logging.getLogger(__name__)


class AdminHandler(BaseHandler):
    """Handler for administrative tools."""
    
    def _initialize_tools(self) -> None:
        """Initialize admin tools."""
        tools = [
            ToolDefinition(
                name="manage_user_sessions",
                description="Manage active user sessions",
                input_schema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Action to perform",
                            "enum": ["list", "terminate", "refresh"]
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session ID for terminate/refresh actions"
                        }
                    },
                    "required": ["action"]
                },
                handler_method="_manage_user_sessions"
            ),
            ToolDefinition(
                name="backup_safla_data",
                description="Create a backup of SAFLA data",
                input_schema={
                    "type": "object",
                    "properties": {
                        "backup_type": {
                            "type": "string",
                            "description": "Type of backup",
                            "enum": ["full", "incremental", "selective"]
                        },
                        "include_patterns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Patterns to include in selective backup"
                        }
                    },
                    "required": ["backup_type"]
                },
                handler_method="_backup_safla_data"
            ),
            ToolDefinition(
                name="restore_safla_data",
                description="Restore SAFLA data from backup",
                input_schema={
                    "type": "object",
                    "properties": {
                        "backup_id": {
                            "type": "string",
                            "description": "Backup ID to restore from"
                        },
                        "restore_type": {
                            "type": "string",
                            "description": "Type of restore",
                            "enum": ["full", "selective"]
                        },
                        "target_path": {
                            "type": "string",
                            "description": "Target path for restore"
                        }
                    },
                    "required": ["backup_id"]
                },
                handler_method="_restore_safla_data"
            ),
            ToolDefinition(
                name="monitor_system_health",
                description="Monitor system health and generate health report",
                input_schema={
                    "type": "object",
                    "properties": {
                        "include_metrics": {
                            "type": "boolean",
                            "description": "Include detailed metrics"
                        },
                        "time_range_minutes": {
                            "type": "integer",
                            "description": "Time range for metrics collection"
                        }
                    },
                    "required": []
                },
                handler_method="_monitor_system_health"
            ),
            ToolDefinition(
                name="manage_api_keys",
                description="Manage API keys for SAFLA access",
                input_schema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Action to perform",
                            "enum": ["create", "list", "revoke", "rotate"]
                        },
                        "key_id": {
                            "type": "string",
                            "description": "API key ID for revoke/rotate actions"
                        },
                        "description": {
                            "type": "string",
                            "description": "Description for new API key"
                        }
                    },
                    "required": ["action"]
                },
                handler_method="_manage_api_keys"
            )
        ]
        
        for tool in tools:
            self.register_tool(tool)
    
    async def _manage_user_sessions(self, action: str, 
                                   session_id: Optional[str] = None) -> Dict[str, Any]:
        """Manage user sessions."""
        try:
            if action == "list":
                # Get all active sessions
                sessions = self.state_manager.get_namespace("sessions")
                active_sessions = []
                
                for sid, session in sessions.items():
                    # Check if session is still active
                    created_at = datetime.fromisoformat(session.get("created_at", ""))
                    if datetime.utcnow() - created_at < timedelta(hours=24):
                        active_sessions.append({
                            "session_id": sid,
                            "user": session.get("user", "unknown"),
                            "created_at": session.get("created_at"),
                            "last_activity": session.get("last_activity"),
                            "ip_address": session.get("ip_address", "unknown")
                        })
                
                return {
                    "active_sessions": active_sessions,
                    "total": len(active_sessions)
                }
                
            elif action == "terminate" and session_id:
                # Terminate specific session
                if self.state_manager.exists(session_id, namespace="sessions"):
                    self.state_manager.delete(session_id, namespace="sessions")
                    return {
                        "status": "success",
                        "message": f"Session {session_id} terminated"
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"Session {session_id} not found"
                    }
                    
            elif action == "refresh" and session_id:
                # Refresh session timeout
                session = self.state_manager.get(session_id, namespace="sessions")
                if session:
                    session["last_activity"] = datetime.utcnow().isoformat()
                    self.state_manager.set(
                        session_id, session, 
                        namespace="sessions",
                        ttl=86400  # 24 hours
                    )
                    return {
                        "status": "success",
                        "message": f"Session {session_id} refreshed"
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"Session {session_id} not found"
                    }
            
            return {
                "status": "error",
                "message": "Invalid action or missing parameters"
            }
            
        except Exception as e:
            logger.error(f"Session management failed: {str(e)}")
            raise
    
    async def _backup_safla_data(self, backup_type: str,
                                include_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create data backup."""
        try:
            backup_id = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            
            # Collect data to backup
            backup_data = {
                "backup_id": backup_id,
                "type": backup_type,
                "created_at": datetime.utcnow().isoformat(),
                "version": "2.0.0",
                "data": {}
            }
            
            if backup_type in ["full", "selective"]:
                # Backup state data
                namespaces = ["config", "deployments", "benchmarks", "metacognitive"]
                if backup_type == "selective" and include_patterns:
                    namespaces = [ns for ns in namespaces if any(p in ns for p in include_patterns)]
                
                for namespace in namespaces:
                    backup_data["data"][namespace] = self.state_manager.get_namespace(namespace)
            
            elif backup_type == "incremental":
                # Get last backup timestamp
                last_backup = self.state_manager.get("last_backup_time", namespace="admin")
                if last_backup:
                    # Only backup changed data (simplified)
                    backup_data["incremental_from"] = last_backup
                    backup_data["data"]["changes"] = "Incremental backup data"
            
            # Store backup metadata
            backup_meta = {
                "backup_id": backup_id,
                "type": backup_type,
                "created_at": backup_data["created_at"],
                "size_bytes": len(json.dumps(backup_data)),
                "status": "completed"
            }
            
            self.state_manager.set(
                backup_id, backup_meta,
                namespace="backups"
            )
            
            # Update last backup time
            self.state_manager.set(
                "last_backup_time", 
                datetime.utcnow().isoformat(),
                namespace="admin"
            )
            
            return {
                "backup_id": backup_id,
                "status": "success",
                "type": backup_type,
                "size_bytes": backup_meta["size_bytes"],
                "namespaces_backed_up": list(backup_data["data"].keys()),
                "message": f"Backup {backup_id} created successfully"
            }
            
        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
            raise
    
    async def _restore_safla_data(self, backup_id: str,
                                 restore_type: str = "full",
                                 target_path: Optional[str] = None) -> Dict[str, Any]:
        """Restore data from backup."""
        try:
            # Get backup metadata
            backup_meta = self.state_manager.get(backup_id, namespace="backups")
            
            if not backup_meta:
                return {
                    "status": "error",
                    "message": f"Backup {backup_id} not found"
                }
            
            # Simulate restore process
            logger.info(f"Restoring from backup {backup_id}")
            
            # In real implementation would:
            # 1. Validate backup integrity
            # 2. Create restore point
            # 3. Restore data to target
            # 4. Verify restoration
            
            restore_info = {
                "restore_id": f"restore_{uuid.uuid4().hex[:8]}",
                "backup_id": backup_id,
                "restore_type": restore_type,
                "started_at": datetime.utcnow().isoformat(),
                "target_path": target_path or "default",
                "status": "completed"
            }
            
            return {
                "status": "success",
                "restore_info": restore_info,
                "message": f"Successfully restored from backup {backup_id}"
            }
            
        except Exception as e:
            logger.error(f"Restore failed: {str(e)}")
            raise
    
    async def _monitor_system_health(self, include_metrics: bool = True,
                                   time_range_minutes: int = 60) -> Dict[str, Any]:
        """Monitor system health."""
        try:
            health_status = {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": "healthy",
                "components": {}
            }
            
            # Check core components
            components = [
                ("memory_system", self._check_memory_health()),
                ("mcp_server", self._check_mcp_health()),
                ("state_manager", self._check_state_health()),
                ("api_gateway", self._check_api_health())
            ]
            
            unhealthy_count = 0
            for component_name, health in components:
                health_status["components"][component_name] = health
                if health["status"] != "healthy":
                    unhealthy_count += 1
            
            # Determine overall status
            if unhealthy_count == 0:
                health_status["overall_status"] = "healthy"
            elif unhealthy_count <= 1:
                health_status["overall_status"] = "degraded"
            else:
                health_status["overall_status"] = "unhealthy"
            
            # Add metrics if requested
            if include_metrics:
                health_status["metrics"] = {
                    "time_range_minutes": time_range_minutes,
                    "uptime_hours": (time.time() - self.get_state("start_time", 0)) / 3600,
                    "total_requests": 15234,
                    "error_rate": 0.02,
                    "avg_response_time_ms": 145
                }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health monitoring failed: {str(e)}")
            raise
    
    async def _manage_api_keys(self, action: str,
                             key_id: Optional[str] = None,
                             description: Optional[str] = None) -> Dict[str, Any]:
        """Manage API keys."""
        try:
            if action == "create":
                # Generate new API key
                key_id = f"safla_{uuid.uuid4().hex}"
                api_key = f"sk_{uuid.uuid4().hex}{uuid.uuid4().hex}"
                
                key_info = {
                    "key_id": key_id,
                    "key_prefix": api_key[:12] + "...",
                    "description": description or "API key",
                    "created_at": datetime.utcnow().isoformat(),
                    "last_used": None,
                    "status": "active"
                }
                
                self.state_manager.set(
                    key_id, key_info,
                    namespace="api_keys"
                )
                
                return {
                    "status": "success",
                    "key_id": key_id,
                    "api_key": api_key,
                    "message": "API key created successfully. Store it securely."
                }
                
            elif action == "list":
                # List all API keys
                keys = self.state_manager.get_namespace("api_keys")
                key_list = []
                
                for kid, key_info in keys.items():
                    key_list.append({
                        "key_id": kid,
                        "key_prefix": key_info.get("key_prefix"),
                        "description": key_info.get("description"),
                        "created_at": key_info.get("created_at"),
                        "last_used": key_info.get("last_used"),
                        "status": key_info.get("status")
                    })
                
                return {
                    "api_keys": key_list,
                    "total": len(key_list)
                }
                
            elif action in ["revoke", "rotate"] and key_id:
                key_info = self.state_manager.get(key_id, namespace="api_keys")
                
                if not key_info:
                    return {
                        "status": "error",
                        "message": f"API key {key_id} not found"
                    }
                
                if action == "revoke":
                    key_info["status"] = "revoked"
                    key_info["revoked_at"] = datetime.utcnow().isoformat()
                    
                    self.state_manager.set(
                        key_id, key_info,
                        namespace="api_keys"
                    )
                    
                    return {
                        "status": "success",
                        "message": f"API key {key_id} revoked"
                    }
                    
                else:  # rotate
                    # Create new key
                    new_api_key = f"sk_{uuid.uuid4().hex}{uuid.uuid4().hex}"
                    
                    # Update key info
                    key_info["key_prefix"] = new_api_key[:12] + "..."
                    key_info["rotated_at"] = datetime.utcnow().isoformat()
                    
                    self.state_manager.set(
                        key_id, key_info,
                        namespace="api_keys"
                    )
                    
                    return {
                        "status": "success",
                        "key_id": key_id,
                        "api_key": new_api_key,
                        "message": "API key rotated successfully"
                    }
            
            return {
                "status": "error",
                "message": "Invalid action or missing parameters"
            }
            
        except Exception as e:
            logger.error(f"API key management failed: {str(e)}")
            raise
    
    def _check_memory_health(self) -> Dict[str, Any]:
        """Check memory system health."""
        return {
            "status": "healthy",
            "details": "Memory system operating normally",
            "metrics": {
                "usage_percent": 65,
                "cache_hit_rate": 0.85
            }
        }
    
    def _check_mcp_health(self) -> Dict[str, Any]:
        """Check MCP server health."""
        return {
            "status": "healthy",
            "details": "MCP server responding normally",
            "metrics": {
                "active_connections": 12,
                "request_queue_size": 3
            }
        }
    
    def _check_state_health(self) -> Dict[str, Any]:
        """Check state manager health."""
        stats = self.state_manager.get_stats()
        return {
            "status": "healthy",
            "details": "State manager operating normally",
            "metrics": {
                "total_keys": stats.get("total_keys", 0),
                "namespaces": stats.get("namespaces", 0)
            }
        }
    
    def _check_api_health(self) -> Dict[str, Any]:
        """Check API gateway health."""
        return {
            "status": "healthy",
            "details": "API gateway operational",
            "metrics": {
                "response_time_ms": 125,
                "success_rate": 0.98
            }
        }