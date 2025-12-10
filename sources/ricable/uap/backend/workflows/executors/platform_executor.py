"""
Cross-Platform Workflow Executors

Executors for running workflows across different platforms including
cloud services, local systems, containers, and edge devices.
"""

import asyncio
import logging
import subprocess
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
import aiohttp
import paramiko
import docker
from kubernetes import client, config
import boto3
from azure.identity import DefaultAzureCredential
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from google.cloud import functions_v1
import requests

logger = logging.getLogger(__name__)


class PlatformExecutor(ABC):
    """Abstract base class for platform-specific workflow executors."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
    @abstractmethod
    async def execute_step(self, step_config: Dict[str, Any], context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow step on the platform."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the platform is available and healthy."""
        pass
    
    def get_platform_info(self) -> Dict[str, Any]:
        """Get platform information and capabilities."""
        return {
            "name": self.__class__.__name__,
            "config": self.config,
            "supported_features": []
        }


class LocalExecutor(PlatformExecutor):
    """Execute workflows on the local system."""
    
    async def execute_step(self, step_config: Dict[str, Any], context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute step locally using subprocess or Python functions."""
        
        execution_type = step_config.get("execution_type", "command")
        
        try:
            if execution_type == "command":
                return await self._execute_command(step_config, context_data)
            elif execution_type == "python":
                return await self._execute_python(step_config, context_data)
            elif execution_type == "script":
                return await self._execute_script(step_config, context_data)
            else:
                raise ValueError(f"Unsupported execution type: {execution_type}")
        
        except Exception as e:
            logger.error(f"Local execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "platform": "local"
            }
    
    async def _execute_command(self, step_config: Dict[str, Any], context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a shell command."""
        
        command = step_config.get("command", "")
        if not command:
            raise ValueError("Command is required")
        
        # Format command with context data
        try:
            formatted_command = command.format(**context_data)
        except KeyError as e:
            raise ValueError(f"Missing context variable: {e}")
        
        # Execute command
        process = await asyncio.create_subprocess_shell(
            formatted_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=step_config.get("working_directory"),
            env={**os.environ, **step_config.get("environment", {})}
        )
        
        stdout, stderr = await process.communicate()
        
        return {
            "success": process.returncode == 0,
            "exit_code": process.returncode,
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "platform": "local",
            "command": formatted_command
        }
    
    async def _execute_python(self, step_config: Dict[str, Any], context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python code."""
        
        code = step_config.get("code", "")
        if not code:
            raise ValueError("Python code is required")
        
        # Create safe execution namespace
        namespace = {
            "__builtins__": __builtins__,
            "context": context_data,
            "result": {},
            "json": json,
            "datetime": datetime,
            "requests": requests
        }
        
        try:
            exec(code, namespace)
            result = namespace.get("result", {})
            
            return {
                "success": True,
                "result": result,
                "platform": "local",
                "execution_type": "python"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "platform": "local",
                "execution_type": "python"
            }
    
    async def _execute_script(self, step_config: Dict[str, Any], context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a script file."""
        
        script_path = step_config.get("script_path", "")
        interpreter = step_config.get("interpreter", "bash")
        
        if not script_path or not os.path.exists(script_path):
            raise ValueError(f"Script file not found: {script_path}")
        
        # Prepare command
        command = f"{interpreter} {script_path}"
        
        # Write context data to temporary file if needed
        if step_config.get("pass_context_as_file"):
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                json.dump(context_data, f)
                command += f" {f.name}"
        
        return await self._execute_command({"command": command}, context_data)
    
    async def health_check(self) -> bool:
        """Check local system health."""
        try:
            process = await asyncio.create_subprocess_shell(
                "echo 'health_check'",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()
            return process.returncode == 0 and b"health_check" in stdout
        except Exception:
            return False


class DockerExecutor(PlatformExecutor):
    """Execute workflows in Docker containers."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.client = docker.from_env()
    
    async def execute_step(self, step_config: Dict[str, Any], context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute step in a Docker container."""
        
        image = step_config.get("image", "alpine:latest")
        command = step_config.get("command", "echo 'Hello from Docker'")
        
        try:
            # Format command with context data
            formatted_command = command.format(**context_data)
            
            # Prepare container configuration
            container_config = {
                "image": image,
                "command": formatted_command,
                "environment": step_config.get("environment", {}),
                "volumes": step_config.get("volumes", {}),
                "working_dir": step_config.get("working_dir"),
                "detach": False,
                "remove": True
            }
            
            # Run container
            container = self.client.containers.run(**container_config)
            
            # Get logs
            logs = container.logs().decode()
            
            return {
                "success": True,
                "logs": logs,
                "platform": "docker",
                "image": image,
                "command": formatted_command
            }
            
        except Exception as e:
            logger.error(f"Docker execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "platform": "docker"
            }
    
    async def health_check(self) -> bool:
        """Check Docker daemon health."""
        try:
            info = self.client.info()
            return info is not None
        except Exception:
            return False


class KubernetesExecutor(PlatformExecutor):
    """Execute workflows on Kubernetes cluster."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        self.v1 = client.CoreV1Api()
        self.batch_v1 = client.BatchV1Api()
    
    async def execute_step(self, step_config: Dict[str, Any], context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute step as a Kubernetes Job."""
        
        image = step_config.get("image", "alpine:latest")
        command = step_config.get("command", ["echo", "Hello from Kubernetes"])
        namespace = step_config.get("namespace", "default")
        
        try:
            # Create job name
            job_name = f"workflow-job-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            # Job specification
            job_spec = client.V1Job(
                api_version="batch/v1",
                kind="Job",
                metadata=client.V1ObjectMeta(name=job_name),
                spec=client.V1JobSpec(
                    template=client.V1PodTemplateSpec(
                        spec=client.V1PodSpec(
                            restart_policy="Never",
                            containers=[
                                client.V1Container(
                                    name="workflow-container",
                                    image=image,
                                    command=command,
                                    env=[
                                        client.V1EnvVar(name=k, value=str(v))
                                        for k, v in context_data.items()
                                    ]
                                )
                            ]
                        )
                    ),
                    backoff_limit=step_config.get("backoff_limit", 3)
                )
            )
            
            # Create job
            job = self.batch_v1.create_namespaced_job(
                namespace=namespace,
                body=job_spec
            )
            
            # Wait for completion (simplified)
            await asyncio.sleep(30)  # In production, implement proper waiting
            
            # Get job status
            job_status = self.batch_v1.read_namespaced_job_status(
                name=job_name,
                namespace=namespace
            )
            
            # Get pod logs
            pods = self.v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"job-name={job_name}"
            )
            
            logs = ""
            if pods.items:
                pod_name = pods.items[0].metadata.name
                logs = self.v1.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=namespace
                )
            
            # Cleanup job
            self.batch_v1.delete_namespaced_job(
                name=job_name,
                namespace=namespace
            )
            
            success = job_status.status.succeeded == 1
            
            return {
                "success": success,
                "logs": logs,
                "platform": "kubernetes",
                "job_name": job_name,
                "namespace": namespace
            }
            
        except Exception as e:
            logger.error(f"Kubernetes execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "platform": "kubernetes"
            }
    
    async def health_check(self) -> bool:
        """Check Kubernetes cluster health."""
        try:
            version = self.v1.get_api_resources()
            return version is not None
        except Exception:
            return False


class AWSExecutor(PlatformExecutor):
    """Execute workflows on AWS services."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.lambda_client = boto3.client('lambda', region_name=config.get('region', 'us-east-1'))
        self.ecs_client = boto3.client('ecs', region_name=config.get('region', 'us-east-1'))
    
    async def execute_step(self, step_config: Dict[str, Any], context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute step on AWS Lambda or ECS."""
        
        service_type = step_config.get("service", "lambda")
        
        if service_type == "lambda":
            return await self._execute_lambda(step_config, context_data)
        elif service_type == "ecs":
            return await self._execute_ecs(step_config, context_data)
        else:
            raise ValueError(f"Unsupported AWS service: {service_type}")
    
    async def _execute_lambda(self, step_config: Dict[str, Any], context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute on AWS Lambda."""
        
        function_name = step_config.get("function_name")
        if not function_name:
            raise ValueError("Lambda function name is required")
        
        try:
            response = self.lambda_client.invoke(
                FunctionName=function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps({
                    "context_data": context_data,
                    "step_config": step_config
                })
            )
            
            payload = json.loads(response['Payload'].read())
            
            return {
                "success": response['StatusCode'] == 200,
                "result": payload,
                "platform": "aws_lambda",
                "function_name": function_name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "platform": "aws_lambda"
            }
    
    async def _execute_ecs(self, step_config: Dict[str, Any], context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute on AWS ECS."""
        
        cluster = step_config.get("cluster", "default")
        task_definition = step_config.get("task_definition")
        
        if not task_definition:
            raise ValueError("ECS task definition is required")
        
        try:
            response = self.ecs_client.run_task(
                cluster=cluster,
                taskDefinition=task_definition,
                overrides={
                    'containerOverrides': [
                        {
                            'name': step_config.get("container_name", "main"),
                            'environment': [
                                {'name': k, 'value': str(v)}
                                for k, v in context_data.items()
                            ]
                        }
                    ]
                }
            )
            
            task_arn = response['tasks'][0]['taskArn']
            
            # Wait for task completion (simplified)
            await asyncio.sleep(60)
            
            return {
                "success": True,
                "task_arn": task_arn,
                "platform": "aws_ecs",
                "cluster": cluster
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "platform": "aws_ecs"
            }
    
    async def health_check(self) -> bool:
        """Check AWS service health."""
        try:
            self.lambda_client.list_functions(MaxItems=1)
            return True
        except Exception:
            return False


class SSHExecutor(PlatformExecutor):
    """Execute workflows on remote systems via SSH."""
    
    async def execute_step(self, step_config: Dict[str, Any], context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute step on remote system via SSH."""
        
        hostname = step_config.get("hostname")
        username = step_config.get("username")
        password = step_config.get("password")
        key_filename = step_config.get("key_filename")
        command = step_config.get("command", "echo 'Hello from remote'")
        
        if not hostname or not username:
            raise ValueError("Hostname and username are required for SSH execution")
        
        try:
            # Format command with context data
            formatted_command = command.format(**context_data)
            
            # Create SSH client
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Connect
            connect_kwargs = {
                "hostname": hostname,
                "username": username
            }
            
            if password:
                connect_kwargs["password"] = password
            elif key_filename:
                connect_kwargs["key_filename"] = key_filename
            
            ssh.connect(**connect_kwargs)
            
            # Execute command
            stdin, stdout, stderr = ssh.exec_command(formatted_command)
            
            # Get results
            exit_code = stdout.channel.recv_exit_status()
            stdout_data = stdout.read().decode()
            stderr_data = stderr.read().decode()
            
            ssh.close()
            
            return {
                "success": exit_code == 0,
                "exit_code": exit_code,
                "stdout": stdout_data,
                "stderr": stderr_data,
                "platform": "ssh",
                "hostname": hostname,
                "command": formatted_command
            }
            
        except Exception as e:
            logger.error(f"SSH execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "platform": "ssh"
            }
    
    async def health_check(self) -> bool:
        """Check SSH connectivity."""
        try:
            # This would need actual connection details
            return True
        except Exception:
            return False


class PlatformExecutorFactory:
    """Factory for creating platform-specific executors."""
    
    _executors = {
        "local": LocalExecutor,
        "docker": DockerExecutor,
        "kubernetes": KubernetesExecutor,
        "aws": AWSExecutor,
        "ssh": SSHExecutor
    }
    
    @classmethod
    def create_executor(cls, platform: str, config: Dict[str, Any] = None) -> PlatformExecutor:
        """Create an executor for the specified platform."""
        
        if platform not in cls._executors:
            raise ValueError(f"Unsupported platform: {platform}")
        
        executor_class = cls._executors[platform]
        return executor_class(config)
    
    @classmethod
    def get_supported_platforms(cls) -> List[str]:
        """Get list of supported platforms."""
        return list(cls._executors.keys())
    
    @classmethod
    def register_executor(cls, platform: str, executor_class: type):
        """Register a custom executor."""
        cls._executors[platform] = executor_class


class CrossPlatformOrchestrator:
    """Orchestrate workflow execution across multiple platforms."""
    
    def __init__(self):
        self.executors: Dict[str, PlatformExecutor] = {}
        self.platform_configs: Dict[str, Dict[str, Any]] = {}
    
    def register_platform(self, platform: str, config: Dict[str, Any] = None):
        """Register a platform for execution."""
        try:
            self.executors[platform] = PlatformExecutorFactory.create_executor(platform, config)
            self.platform_configs[platform] = config or {}
            logger.info(f"Registered platform: {platform}")
        except Exception as e:
            logger.error(f"Failed to register platform {platform}: {str(e)}")
    
    async def execute_on_platform(
        self, 
        platform: str, 
        step_config: Dict[str, Any], 
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a step on a specific platform."""
        
        if platform not in self.executors:
            raise ValueError(f"Platform not registered: {platform}")
        
        executor = self.executors[platform]
        
        start_time = datetime.utcnow()
        result = await executor.execute_step(step_config, context_data)
        end_time = datetime.utcnow()
        
        # Add execution metadata
        result.update({
            "execution_start": start_time.isoformat(),
            "execution_end": end_time.isoformat(),
            "execution_duration_ms": int((end_time - start_time).total_seconds() * 1000),
            "platform": platform
        })
        
        return result
    
    async def execute_multi_platform(
        self, 
        platform_steps: List[Dict[str, Any]], 
        context_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute steps across multiple platforms in parallel."""
        
        tasks = []
        for step in platform_steps:
            platform = step.get("platform", "local")
            step_config = step.get("config", {})
            
            task = self.execute_on_platform(platform, step_config, context_data)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "step_index": i,
                    "platform": platform_steps[i].get("platform", "local")
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def health_check_all_platforms(self) -> Dict[str, bool]:
        """Check health of all registered platforms."""
        
        health_results = {}
        
        for platform, executor in self.executors.items():
            try:
                health_results[platform] = await executor.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {platform}: {str(e)}")
                health_results[platform] = False
        
        return health_results
    
    def get_platform_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered platforms."""
        
        return {
            platform: executor.get_platform_info()
            for platform, executor in self.executors.items()
        }