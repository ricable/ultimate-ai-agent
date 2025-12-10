"""
Agent Orchestration Module
Manages AI agent lifecycle, deployment, and coordination
Supports SpinKube (Wasm), WasmEdge, and Container runtimes on Kubernetes
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
import uuid
import json
import os
import httpx
from abc import ABC, abstractmethod


class AgentRuntime(Enum):
    """Agent runtime environments."""
    WASM_SPIN = "wasm-spin"  # SpinKube WebAssembly (wasmtime)
    WASM_EDGE = "wasm-edge"  # WasmEdge with QuickJS
    CONTAINER = "container"  # Docker/K8s container
    E2B = "e2b"  # E2B Firecracker sandbox


class AgentStatus(Enum):
    """Agent lifecycle status."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    PAUSED = "paused"
    FAILED = "failed"
    TERMINATED = "terminated"
    SCALING = "scaling"


@dataclass
class AgentConfig:
    """Agent configuration."""
    id: str
    name: str
    description: str
    capabilities: List[str] = field(default_factory=list)
    model: str = "claude-3-5-sonnet-20241022"
    tools: List[str] = field(default_factory=list)
    runtime: AgentRuntime = AgentRuntime.WASM_SPIN
    owner_id: Optional[str] = None
    system_prompt: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7
    replicas: int = 1
    resources: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "limits": {"cpu": "200m", "memory": "128Mi"},
        "requests": {"cpu": "50m", "memory": "32Mi"}
    })
    allowed_hosts: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)


@dataclass
class Agent:
    """Agent instance."""
    config: AgentConfig
    status: AgentStatus = AgentStatus.PENDING
    endpoint: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    kubernetes_resource: Optional[Dict[str, Any]] = None


class KubernetesClient:
    """
    Kubernetes API client for managing Wasm workloads.
    Supports SpinApp CRDs, WasmEdge Deployments, and standard containers.
    """

    def __init__(self):
        # In-cluster configuration
        self.api_server = os.getenv("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
        self.api_port = os.getenv("KUBERNETES_SERVICE_PORT", "443")
        self.namespace = os.getenv("POD_NAMESPACE", "agents")
        self.token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
        self.ca_path = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
        self._token: Optional[str] = None

    @property
    def token(self) -> str:
        """Get service account token."""
        if self._token is None:
            try:
                with open(self.token_path, "r") as f:
                    self._token = f.read().strip()
            except FileNotFoundError:
                # Running outside cluster, use kubeconfig
                self._token = os.getenv("KUBERNETES_TOKEN", "")
        return self._token

    @property
    def base_url(self) -> str:
        """Get Kubernetes API base URL."""
        return f"https://{self.api_server}:{self.api_port}"

    def _headers(self) -> Dict[str, str]:
        """Get request headers with auth."""
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    async def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make authenticated request to Kubernetes API."""
        async with httpx.AsyncClient(verify=self.ca_path if os.path.exists(self.ca_path) else False) as client:
            response = await client.request(
                method=method,
                url=f"{self.base_url}{path}",
                headers=self._headers(),
                json=body,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json() if response.content else {}

    async def create_spinapp(self, config: AgentConfig) -> Dict[str, Any]:
        """Create SpinApp custom resource."""
        spinapp = {
            "apiVersion": "core.spinoperator.dev/v1alpha1",
            "kind": "SpinApp",
            "metadata": {
                "name": f"agent-{config.id}",
                "namespace": self.namespace,
                "labels": {
                    "app": "spin-agent",
                    "agent-id": config.id,
                    "owner-id": config.owner_id or "system"
                }
            },
            "spec": {
                "image": f"ghcr.io/ruvnet/spin-agent:{config.id}",
                "replicas": config.replicas,
                "executor": "containerd-shim-spin",
                "resources": config.resources,
                "variables": [
                    {"name": "agent_id", "value": config.id},
                    {"name": "agent_model", "value": config.model},
                    {
                        "name": "anthropic_api_key",
                        "valueFrom": {"secretKeyRef": {"name": "agent-secrets", "key": "ANTHROPIC_API_KEY"}}
                    }
                ],
                "runtimeConfig": {
                    "allowedHosts": config.allowed_hosts or [
                        "https://*.anthropic.com",
                        "http://litellm.default.svc.cluster.local:4000",
                        "http://ruvector.default.svc.cluster.local:8765"
                    ],
                    "keyValueStores": [f"agent-{config.id}-state"]
                }
            }
        }

        path = f"/apis/core.spinoperator.dev/v1alpha1/namespaces/{self.namespace}/spinapps"
        return await self._request("POST", path, spinapp)

    async def create_wasmedge_deployment(self, config: AgentConfig) -> Dict[str, Any]:
        """Create WasmEdge Deployment with RuntimeClass."""
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"wasmedge-agent-{config.id}",
                "namespace": self.namespace,
                "labels": {
                    "app": "wasmedge-agent",
                    "agent-id": config.id,
                    "owner-id": config.owner_id or "system"
                }
            },
            "spec": {
                "replicas": config.replicas,
                "selector": {
                    "matchLabels": {"agent-id": config.id}
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "wasmedge-agent",
                            "agent-id": config.id
                        }
                    },
                    "spec": {
                        "runtimeClassName": "wasmedge",
                        "serviceAccountName": "wasmedge-workload",
                        "containers": [{
                            "name": "agent",
                            "image": f"ghcr.io/ruvnet/wasmedge-agent:{config.id}",
                            "ports": [{"containerPort": 8080}],
                            "env": [
                                {"name": "AGENT_ID", "value": config.id},
                                {"name": "AGENT_MODEL", "value": config.model},
                                {
                                    "name": "ANTHROPIC_API_KEY",
                                    "valueFrom": {"secretKeyRef": {"name": "agent-secrets", "key": "ANTHROPIC_API_KEY"}}
                                }
                            ],
                            "resources": config.resources,
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": 8080},
                                "initialDelaySeconds": 5,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/health", "port": 8080},
                                "initialDelaySeconds": 3,
                                "periodSeconds": 5
                            }
                        }],
                        "nodeSelector": {"wasm.runtime/wasmedge": "true"}
                    }
                }
            }
        }

        path = f"/apis/apps/v1/namespaces/{self.namespace}/deployments"
        return await self._request("POST", path, deployment)

    async def create_service(self, config: AgentConfig, runtime: str) -> Dict[str, Any]:
        """Create Service for agent."""
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"agent-{config.id}",
                "namespace": self.namespace,
                "labels": {
                    "app": f"{runtime}-agent",
                    "agent-id": config.id
                }
            },
            "spec": {
                "type": "ClusterIP",
                "ports": [{"port": 8080, "targetPort": 8080, "protocol": "TCP"}],
                "selector": {"agent-id": config.id}
            }
        }

        path = f"/api/v1/namespaces/{self.namespace}/services"
        return await self._request("POST", path, service)

    async def delete_spinapp(self, agent_id: str) -> Dict[str, Any]:
        """Delete SpinApp."""
        path = f"/apis/core.spinoperator.dev/v1alpha1/namespaces/{self.namespace}/spinapps/agent-{agent_id}"
        return await self._request("DELETE", path)

    async def delete_deployment(self, agent_id: str) -> Dict[str, Any]:
        """Delete Deployment."""
        path = f"/apis/apps/v1/namespaces/{self.namespace}/deployments/wasmedge-agent-{agent_id}"
        return await self._request("DELETE", path)

    async def delete_service(self, agent_id: str) -> Dict[str, Any]:
        """Delete Service."""
        path = f"/api/v1/namespaces/{self.namespace}/services/agent-{agent_id}"
        return await self._request("DELETE", path)

    async def get_spinapp_status(self, agent_id: str) -> Dict[str, Any]:
        """Get SpinApp status."""
        path = f"/apis/core.spinoperator.dev/v1alpha1/namespaces/{self.namespace}/spinapps/agent-{agent_id}"
        return await self._request("GET", path)

    async def get_deployment_status(self, agent_id: str) -> Dict[str, Any]:
        """Get Deployment status."""
        path = f"/apis/apps/v1/namespaces/{self.namespace}/deployments/wasmedge-agent-{agent_id}"
        return await self._request("GET", path)

    async def scale_spinapp(self, agent_id: str, replicas: int) -> Dict[str, Any]:
        """Scale SpinApp."""
        patch = [{"op": "replace", "path": "/spec/replicas", "value": replicas}]
        path = f"/apis/core.spinoperator.dev/v1alpha1/namespaces/{self.namespace}/spinapps/agent-{agent_id}"

        async with httpx.AsyncClient(verify=self.ca_path if os.path.exists(self.ca_path) else False) as client:
            response = await client.patch(
                url=f"{self.base_url}{path}",
                headers={**self._headers(), "Content-Type": "application/json-patch+json"},
                json=patch,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()


class AgentOrchestrator:
    """
    Agent Orchestrator for SpinKube, WasmEdge, and Container deployments.

    Manages:
    - SpinApp deployments for Wasm agents (Spin/wasmtime)
    - WasmEdge deployments for QuickJS agents
    - Container deployments for full Python agents
    - E2B sandbox sessions for code execution
    """

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.k8s = KubernetesClient()

    async def deploy_agent(self, config: AgentConfig) -> Agent:
        """Deploy agent based on runtime type."""
        if config.runtime == AgentRuntime.WASM_SPIN:
            return await self.deploy_spinapp(config)
        elif config.runtime == AgentRuntime.WASM_EDGE:
            return await self.deploy_wasmedge(config)
        elif config.runtime == AgentRuntime.CONTAINER:
            return await self.deploy_container(config)
        elif config.runtime == AgentRuntime.E2B:
            return await self.deploy_e2b(config)
        else:
            raise ValueError(f"Unknown runtime: {config.runtime}")

    async def deploy_spinapp(self, config: AgentConfig) -> Agent:
        """Deploy agent as SpinKube SpinApp."""
        agent = Agent(config=config, status=AgentStatus.DEPLOYING)
        self.agents[config.id] = agent

        try:
            # Create SpinApp CRD
            spinapp_result = await self.k8s.create_spinapp(config)
            agent.kubernetes_resource = spinapp_result

            # Create Service
            await self.k8s.create_service(config, "spin")

            agent.status = AgentStatus.RUNNING
            agent.endpoint = f"http://agent-{config.id}.{self.k8s.namespace}.svc.cluster.local:8080"

        except Exception as e:
            agent.status = AgentStatus.FAILED
            agent.metrics["error"] = str(e)

        return agent

    async def deploy_wasmedge(self, config: AgentConfig) -> Agent:
        """Deploy agent with WasmEdge RuntimeClass."""
        agent = Agent(config=config, status=AgentStatus.DEPLOYING)
        self.agents[config.id] = agent

        try:
            # Create Deployment with WasmEdge RuntimeClass
            deployment_result = await self.k8s.create_wasmedge_deployment(config)
            agent.kubernetes_resource = deployment_result

            # Create Service
            await self.k8s.create_service(config, "wasmedge")

            agent.status = AgentStatus.RUNNING
            agent.endpoint = f"http://agent-{config.id}.{self.k8s.namespace}.svc.cluster.local:8080"

        except Exception as e:
            agent.status = AgentStatus.FAILED
            agent.metrics["error"] = str(e)

        return agent

    async def deploy_container(self, config: AgentConfig) -> Agent:
        """Deploy agent as standard K8s container."""
        agent = Agent(config=config, status=AgentStatus.DEPLOYING)
        self.agents[config.id] = agent

        # Standard container deployment (similar to WasmEdge but without RuntimeClass)
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"container-agent-{config.id}",
                "namespace": self.k8s.namespace,
                "labels": {"agent-id": config.id}
            },
            "spec": {
                "replicas": config.replicas,
                "selector": {"matchLabels": {"agent-id": config.id}},
                "template": {
                    "metadata": {"labels": {"agent-id": config.id}},
                    "spec": {
                        "containers": [{
                            "name": "agent",
                            "image": f"ghcr.io/ruvnet/container-agent:{config.id}",
                            "ports": [{"containerPort": 8080}],
                            "resources": config.resources
                        }]
                    }
                }
            }
        }

        try:
            path = f"/apis/apps/v1/namespaces/{self.k8s.namespace}/deployments"
            result = await self.k8s._request("POST", path, deployment)
            agent.kubernetes_resource = result
            await self.k8s.create_service(config, "container")

            agent.status = AgentStatus.RUNNING
            agent.endpoint = f"http://agent-{config.id}.{self.k8s.namespace}.svc.cluster.local:8080"

        except Exception as e:
            agent.status = AgentStatus.FAILED
            agent.metrics["error"] = str(e)

        return agent

    async def deploy_e2b(self, config: AgentConfig) -> Agent:
        """Deploy agent in E2B Firecracker sandbox."""
        agent = Agent(config=config, status=AgentStatus.DEPLOYING)
        self.agents[config.id] = agent

        try:
            # E2B deployment would go through their API
            e2b_api_key = os.getenv("E2B_API_KEY")
            if not e2b_api_key:
                raise ValueError("E2B_API_KEY not configured")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.e2b.dev/v1/sandboxes",
                    headers={"Authorization": f"Bearer {e2b_api_key}"},
                    json={
                        "template": "python-agent",
                        "metadata": {"agent_id": config.id}
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                sandbox = response.json()

            agent.kubernetes_resource = sandbox
            agent.status = AgentStatus.RUNNING
            agent.endpoint = sandbox.get("url")

        except Exception as e:
            agent.status = AgentStatus.FAILED
            agent.metrics["error"] = str(e)

        return agent

    async def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID."""
        return self.agents.get(agent_id)

    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed agent status from Kubernetes."""
        agent = self.agents.get(agent_id)
        if not agent:
            return {"error": "Agent not found"}

        try:
            if agent.config.runtime == AgentRuntime.WASM_SPIN:
                k8s_status = await self.k8s.get_spinapp_status(agent_id)
            elif agent.config.runtime in (AgentRuntime.WASM_EDGE, AgentRuntime.CONTAINER):
                k8s_status = await self.k8s.get_deployment_status(agent_id)
            else:
                k8s_status = {}

            return {
                "agent_id": agent_id,
                "status": agent.status.value,
                "endpoint": agent.endpoint,
                "runtime": agent.config.runtime.value,
                "kubernetes": k8s_status.get("status", {})
            }

        except Exception as e:
            return {
                "agent_id": agent_id,
                "status": agent.status.value,
                "error": str(e)
            }

    async def list_agents(self, owner_id: Optional[str] = None) -> List[Agent]:
        """List all agents, optionally filtered by owner."""
        agents = list(self.agents.values())
        if owner_id:
            agents = [a for a in agents if a.config.owner_id == owner_id]
        return agents

    async def scale_agent(self, agent_id: str, replicas: int) -> bool:
        """Scale agent replicas."""
        agent = self.agents.get(agent_id)
        if not agent:
            return False

        try:
            agent.status = AgentStatus.SCALING

            if agent.config.runtime == AgentRuntime.WASM_SPIN:
                await self.k8s.scale_spinapp(agent_id, replicas)
            else:
                # Scale deployment
                patch = [{"op": "replace", "path": "/spec/replicas", "value": replicas}]
                path = f"/apis/apps/v1/namespaces/{self.k8s.namespace}/deployments/wasmedge-agent-{agent_id}"

                async with httpx.AsyncClient(verify=False) as client:
                    response = await client.patch(
                        url=f"{self.k8s.base_url}{path}",
                        headers={**self.k8s._headers(), "Content-Type": "application/json-patch+json"},
                        json=patch,
                        timeout=30.0
                    )
                    response.raise_for_status()

            agent.config.replicas = replicas
            agent.status = AgentStatus.RUNNING
            return True

        except Exception as e:
            agent.status = AgentStatus.FAILED
            agent.metrics["scale_error"] = str(e)
            return False

    async def terminate_agent(self, agent_id: str) -> bool:
        """Terminate an agent and clean up resources."""
        agent = self.agents.get(agent_id)
        if not agent:
            return False

        try:
            # Delete Kubernetes resources
            if agent.config.runtime == AgentRuntime.WASM_SPIN:
                await self.k8s.delete_spinapp(agent_id)
            elif agent.config.runtime in (AgentRuntime.WASM_EDGE, AgentRuntime.CONTAINER):
                await self.k8s.delete_deployment(agent_id)

            # Delete Service
            await self.k8s.delete_service(agent_id)

            agent.status = AgentStatus.TERMINATED
            return True

        except Exception as e:
            agent.metrics["termination_error"] = str(e)
            return False

    async def execute_task(self, agent_id: str, task: str, context: List[Dict] = None) -> Dict[str, Any]:
        """Execute task on agent."""
        agent = self.agents.get(agent_id)
        if not agent or agent.status != AgentStatus.RUNNING:
            return {"error": "Agent not available"}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{agent.endpoint}/execute",
                    json={"task": task, "context": context or []},
                    timeout=120.0
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            return {"error": str(e)}
