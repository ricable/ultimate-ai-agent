"""
SAFLA CLI Benchmarks
====================

This module provides specific benchmarks for testing SAFLA CLI performance,
including command execution times, memory usage, and throughput metrics.
"""

import asyncio
import subprocess
import time
import json
import tempfile
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from .core import Benchmark

logger = logging.getLogger(__name__)


class CLICommandBenchmark(Benchmark):
    """Base class for CLI command benchmarks."""
    
    def __init__(self, name: str, command: List[str], description: str = "",
                 expected_exit_code: int = 0, timeout: float = 30.0):
        super().__init__(name, description)
        self.command = command
        self.expected_exit_code = expected_exit_code
        self.timeout = timeout
        self.temp_files: List[Path] = []
    
    async def setup(self) -> None:
        """Setup for CLI command execution."""
        # Ensure SAFLA is available
        try:
            result = await asyncio.create_subprocess_exec(
                'python', '-m', 'safla.cli', '--help',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()
            if result.returncode != 0:
                raise RuntimeError("SAFLA CLI not available")
        except Exception as e:
            raise RuntimeError(f"Failed to verify SAFLA CLI: {e}")
    
    async def run(self) -> Dict[str, Any]:
        """Execute CLI command and measure performance."""
        start_time = time.perf_counter()
        
        try:
            # Execute command
            process = await asyncio.create_subprocess_exec(
                *self.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=self.timeout
            )
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Decode output
            stdout_text = stdout.decode('utf-8') if stdout else ""
            stderr_text = stderr.decode('utf-8') if stderr else ""
            
            # Check exit code
            success = process.returncode == self.expected_exit_code
            
            return {
                'execution_time': execution_time,
                'exit_code': process.returncode,
                'success': success,
                'stdout_length': len(stdout_text),
                'stderr_length': len(stderr_text),
                'stdout': stdout_text[:1000],  # First 1000 chars for debugging
                'stderr': stderr_text[:1000],
                'latency': execution_time * 1000  # Convert to ms
            }
            
        except asyncio.TimeoutError:
            return {
                'execution_time': self.timeout,
                'exit_code': -1,
                'success': False,
                'error': 'Command timeout',
                'latency': self.timeout * 1000
            }
        except Exception as e:
            end_time = time.perf_counter()
            return {
                'execution_time': end_time - start_time,
                'exit_code': -1,
                'success': False,
                'error': str(e),
                'latency': (end_time - start_time) * 1000
            }
    
    async def teardown(self) -> None:
        """Cleanup temporary files."""
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")


class CLIHelpBenchmark(CLICommandBenchmark):
    """Benchmark for CLI help command performance."""
    
    def __init__(self):
        super().__init__(
            name="cli_help_command",
            command=['python', '-m', 'safla.cli', '--help'],
            description="SAFLA CLI help command execution time"
        )


class CLIValidateBenchmark(CLICommandBenchmark):
    """Benchmark for CLI validate command performance."""
    
    def __init__(self):
        super().__init__(
            name="cli_validate_command",
            command=['python', '-m', 'safla.cli', 'validate'],
            description="SAFLA CLI validate command execution time",
            expected_exit_code=1,  # May fail due to missing dependencies
            timeout=60.0
        )


class CLIInfoBenchmark(CLICommandBenchmark):
    """Benchmark for CLI info command performance."""
    
    def __init__(self):
        super().__init__(
            name="cli_info_command",
            command=['python', '-m', 'safla.cli', 'info'],
            description="SAFLA CLI info command execution time"
        )


class CLIStatusBenchmark(CLICommandBenchmark):
    """Benchmark for CLI status command performance."""
    
    def __init__(self):
        super().__init__(
            name="cli_status_command",
            command=['python', '-m', 'safla.cli', 'status'],
            description="SAFLA CLI status command execution time"
        )


class CLIConfigInitBenchmark(CLICommandBenchmark):
    """Benchmark for CLI config initialization performance."""
    
    def __init__(self):
        super().__init__(
            name="cli_config_init_command",
            command=[],  # Will be set in setup
            description="SAFLA CLI config init command execution time"
        )
        self.config_file = None
    
    async def setup(self) -> None:
        """Setup temporary config file."""
        await super().setup()
        
        # Create temporary config file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.json', prefix='safla_config_')
        os.close(temp_fd)
        self.config_file = Path(temp_path)
        self.temp_files.append(self.config_file)
        
        # Set command with temp file
        self.command = [
            'python', '-m', 'safla.cli', 'init-config',
            '--output', str(self.config_file),
            '--template', 'minimal'
        ]


class CLIFastMCPDiscoveryBenchmark(CLICommandBenchmark):
    """Benchmark for FastMCP discovery performance."""
    
    def __init__(self):
        super().__init__(
            name="cli_fastmcp_discover_command",
            command=[
                'python', '-m', 'safla.cli', 'fastmcp', 'discover',
                '--urls', 'http://localhost:8000',
                '--timeout', '5'
            ],
            description="SAFLA CLI FastMCP discover command execution time",
            expected_exit_code=1,  # Expected to fail without server
            timeout=10.0
        )


class CLIBatchCommandBenchmark(Benchmark):
    """Benchmark for executing multiple CLI commands in sequence."""
    
    def __init__(self, commands: List[List[str]], name: str = "cli_batch_commands"):
        super().__init__(name, f"Batch execution of {len(commands)} CLI commands")
        self.commands = commands
        self.temp_files: List[Path] = []
    
    async def setup(self) -> None:
        """Setup for batch command execution."""
        # Verify SAFLA CLI is available
        try:
            result = await asyncio.create_subprocess_exec(
                'python', '-m', 'safla.cli', '--help',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()
            if result.returncode != 0:
                raise RuntimeError("SAFLA CLI not available")
        except Exception as e:
            raise RuntimeError(f"Failed to verify SAFLA CLI: {e}")
    
    async def run(self) -> Dict[str, Any]:
        """Execute all commands and measure total performance."""
        start_time = time.perf_counter()
        
        results = []
        successful_commands = 0
        total_stdout_length = 0
        total_stderr_length = 0
        
        for i, command in enumerate(self.commands):
            try:
                cmd_start = time.perf_counter()
                
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=30.0
                )
                
                cmd_end = time.perf_counter()
                cmd_time = cmd_end - cmd_start
                
                stdout_text = stdout.decode('utf-8') if stdout else ""
                stderr_text = stderr.decode('utf-8') if stderr else ""
                
                total_stdout_length += len(stdout_text)
                total_stderr_length += len(stderr_text)
                
                if process.returncode == 0:
                    successful_commands += 1
                
                results.append({
                    'command': ' '.join(command),
                    'execution_time': cmd_time,
                    'exit_code': process.returncode,
                    'success': process.returncode == 0
                })
                
            except Exception as e:
                cmd_end = time.perf_counter()
                results.append({
                    'command': ' '.join(command),
                    'execution_time': cmd_end - cmd_start,
                    'exit_code': -1,
                    'success': False,
                    'error': str(e)
                })
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        return {
            'total_execution_time': total_time,
            'command_count': len(self.commands),
            'successful_commands': successful_commands,
            'success_rate': successful_commands / len(self.commands),
            'average_command_time': total_time / len(self.commands),
            'throughput': len(self.commands) / total_time,  # commands per second
            'total_stdout_length': total_stdout_length,
            'total_stderr_length': total_stderr_length,
            'command_results': results,
            'latency': total_time * 1000  # Convert to ms
        }
    
    async def teardown(self) -> None:
        """Cleanup temporary files."""
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")


class CLIMemoryUsageBenchmark(Benchmark):
    """Benchmark for measuring CLI memory usage during execution."""
    
    def __init__(self, command: List[str], name: str = "cli_memory_usage"):
        super().__init__(name, f"Memory usage benchmark for: {' '.join(command)}")
        self.command = command
        self.monitoring_interval = 0.1  # seconds
    
    async def setup(self) -> None:
        """Setup for memory monitoring."""
        import psutil
        self.psutil = psutil
    
    async def run(self) -> Dict[str, Any]:
        """Execute command while monitoring memory usage."""
        import asyncio
        
        start_time = time.perf_counter()
        memory_samples = []
        
        try:
            # Start the process
            process = await asyncio.create_subprocess_exec(
                *self.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Monitor memory usage
            async def monitor_memory():
                try:
                    psutil_process = self.psutil.Process(process.pid)
                    while process.returncode is None:
                        try:
                            memory_info = psutil_process.memory_info()
                            memory_samples.append({
                                'timestamp': time.perf_counter() - start_time,
                                'rss': memory_info.rss / 1024 / 1024,  # MB
                                'vms': memory_info.vms / 1024 / 1024   # MB
                            })
                            await asyncio.sleep(self.monitoring_interval)
                        except self.psutil.NoSuchProcess:
                            break
                except Exception as e:
                    logger.warning(f"Memory monitoring error: {e}")
            
            # Start monitoring task
            monitor_task = asyncio.create_task(monitor_memory())
            
            # Wait for process completion
            stdout, stderr = await process.communicate()
            
            # Stop monitoring
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Calculate memory statistics
            if memory_samples:
                rss_values = [s['rss'] for s in memory_samples]
                vms_values = [s['vms'] for s in memory_samples]
                
                max_rss = max(rss_values)
                avg_rss = sum(rss_values) / len(rss_values)
                max_vms = max(vms_values)
                avg_vms = sum(vms_values) / len(vms_values)
            else:
                max_rss = avg_rss = max_vms = avg_vms = 0.0
            
            return {
                'execution_time': execution_time,
                'exit_code': process.returncode,
                'success': process.returncode == 0,
                'max_memory_rss': max_rss,
                'avg_memory_rss': avg_rss,
                'max_memory_vms': max_vms,
                'avg_memory_vms': avg_vms,
                'memory_samples': len(memory_samples),
                'memory_efficiency': avg_rss / max_rss if max_rss > 0 else 1.0,
                'latency': execution_time * 1000
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            return {
                'execution_time': end_time - start_time,
                'exit_code': -1,
                'success': False,
                'error': str(e),
                'max_memory_rss': 0.0,
                'avg_memory_rss': 0.0,
                'latency': (end_time - start_time) * 1000
            }
    
    async def teardown(self) -> None:
        """No cleanup needed for memory benchmark."""
        pass


class CLIBenchmarkSuite:
    """Complete benchmark suite for SAFLA CLI."""
    
    def __init__(self):
        self.benchmarks = self._create_benchmarks()
    
    def _create_benchmarks(self) -> List[Benchmark]:
        """Create all CLI benchmarks."""
        benchmarks = [
            # Basic command benchmarks
            CLIHelpBenchmark(),
            CLIInfoBenchmark(),
            CLIStatusBenchmark(),
            CLIConfigInitBenchmark(),
            
            # Validation benchmark (may fail, but measures performance)
            CLIValidateBenchmark(),
            
            # Batch command benchmark
            CLIBatchCommandBenchmark([
                ['python', '-m', 'safla.cli', '--help'],
                ['python', '-m', 'safla.cli', 'info'],
                ['python', '-m', 'safla.cli', 'status'],
                ['python', '-m', 'safla.cli', 'status', '--format', 'json']
            ], "cli_batch_basic_commands"),
            
            # Memory usage benchmarks
            CLIMemoryUsageBenchmark(
                ['python', '-m', 'safla.cli', 'info'],
                "cli_info_memory_usage"
            ),
            CLIMemoryUsageBenchmark(
                ['python', '-m', 'safla.cli', 'validate'],
                "cli_validate_memory_usage"
            )
        ]
        
        return benchmarks
    
    def get_benchmark_names(self) -> List[str]:
        """Get list of all benchmark names."""
        return [b.name for b in self.benchmarks]
    
    def get_benchmark(self, name: str) -> Optional[Benchmark]:
        """Get specific benchmark by name."""
        for benchmark in self.benchmarks:
            if benchmark.name == name:
                return benchmark
        return None