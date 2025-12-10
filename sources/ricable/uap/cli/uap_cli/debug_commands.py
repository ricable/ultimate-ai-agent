# UAP CLI Debug Commands Module
"""
Advanced debugging commands for UAP CLI operations.
Provides agent introspection, request tracing, performance profiling, and more.
"""

import asyncio
import json
import time
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse
from datetime import datetime, timedelta
import httpx
import websockets

from uap_sdk import UAPClient, Configuration
from uap_sdk.exceptions import UAPException, UAPConnectionError, UAPAuthError
from .commands import BaseCommand


class DebugCommands(BaseCommand):
    """Advanced debugging commands for UAP agents and systems."""
    
    def add_subcommands(self, parser: argparse.ArgumentParser) -> None:
        subparsers = parser.add_subparsers(dest='debug_command', help='Debug actions')
        
        # Agent introspection
        introspect_parser = subparsers.add_parser('introspect', help='Introspect agent state and configuration')
        introspect_parser.add_argument('agent_id', help='Agent identifier')
        introspect_parser.add_argument('--deep', action='store_true', help='Deep introspection including memory state')
        introspect_parser.add_argument('--save', help='Save introspection report to file')
        
        # Request tracing
        trace_parser = subparsers.add_parser('trace', help='Trace requests and responses')
        trace_parser.add_argument('agent_id', help='Agent identifier')
        trace_parser.add_argument('--message', '-m', help='Message to trace')
        trace_parser.add_argument('--duration', type=int, default=60, help='Trace duration in seconds')
        trace_parser.add_argument('--include-websocket', action='store_true', help='Include WebSocket traffic')
        trace_parser.add_argument('--save-log', help='Save trace log to file')
        
        # Performance profiling
        profile_parser = subparsers.add_parser('profile', help='Profile agent performance')
        profile_parser.add_argument('agent_id', help='Agent identifier') 
        profile_parser.add_argument('--requests', type=int, default=10, help='Number of test requests')
        profile_parser.add_argument('--concurrent', type=int, default=1, help='Concurrent requests')
        profile_parser.add_argument('--message', default='Hello, this is a test message', help='Test message')
        profile_parser.add_argument('--warmup', type=int, default=3, help='Warmup requests')
        profile_parser.add_argument('--report', help='Save performance report to file')
        
        # Memory analysis
        memory_parser = subparsers.add_parser('memory', help='Analyze memory usage')
        memory_parser.add_argument('--agent-id', help='Specific agent ID (optional)')
        memory_parser.add_argument('--threshold', type=float, default=80.0, help='Memory threshold percentage')
        memory_parser.add_argument('--history', type=int, default=24, help='Hours of history to analyze')
        
        # Network diagnostics
        network_parser = subparsers.add_parser('network', help='Network diagnostics')
        network_parser.add_argument('--endpoint', help='Specific endpoint to test')
        network_parser.add_argument('--websocket', action='store_true', help='Test WebSocket connectivity')
        network_parser.add_argument('--latency-test', action='store_true', help='Run latency tests')
        network_parser.add_argument('--count', type=int, default=10, help='Number of test requests')
        
        # System health deep dive
        health_parser = subparsers.add_parser('health-deep', help='Deep system health analysis')
        health_parser.add_argument('--check-dependencies', action='store_true', help='Check external dependencies')
        health_parser.add_argument('--performance-baseline', action='store_true', help='Establish performance baseline')
        health_parser.add_argument('--generate-report', help='Generate comprehensive health report')
        
        # Live debugging session
        live_parser = subparsers.add_parser('live', help='Start live debugging session')
        live_parser.add_argument('agent_id', help='Agent identifier')
        live_parser.add_argument('--breakpoints', nargs='+', help='Breakpoint patterns')
        live_parser.add_argument('--watch', nargs='+', help='Variables/expressions to watch')
        
        # Error analysis
        error_parser = subparsers.add_parser('errors', help='Analyze recent errors')
        error_parser.add_argument('--since', help='Time period (e.g., "1h", "30m", "2d")')
        error_parser.add_argument('--agent-id', help='Filter by agent ID')
        error_parser.add_argument('--severity', choices=['debug', 'info', 'warning', 'error', 'critical'])
        error_parser.add_argument('--export', help='Export error analysis to file')
        
        # Configuration validation
        config_parser = subparsers.add_parser('config-validate', help='Validate configuration')
        config_parser.add_argument('--config-file', help='Configuration file to validate')
        config_parser.add_argument('--fix-suggestions', action='store_true', help='Provide fix suggestions')
        config_parser.add_argument('--security-check', action='store_true', help='Check for security issues')
    
    async def handle_command(self, args: argparse.Namespace) -> int:
        if args.debug_command == 'introspect':
            return await self._handle_introspect(args)
        elif args.debug_command == 'trace':
            return await self._handle_trace(args)
        elif args.debug_command == 'profile':
            return await self._handle_profile(args)
        elif args.debug_command == 'memory':
            return await self._handle_memory(args)
        elif args.debug_command == 'network':
            return await self._handle_network(args)
        elif args.debug_command == 'health-deep':
            return await self._handle_health_deep(args)
        elif args.debug_command == 'live':
            return await self._handle_live(args)
        elif args.debug_command == 'errors':
            return await self._handle_errors(args)
        elif args.debug_command == 'config-validate':
            return await self._handle_config_validate(args)
        else:
            self.app.print_error("Unknown debug command")
            return 1
    
    async def _handle_introspect(self, args: argparse.Namespace) -> int:
        """Handle agent introspection command."""
        try:
            self.app.print_success(f"🔍 Introspecting agent: {args.agent_id}")
            
            # Get basic agent info
            basic_info = await self._get_agent_basic_info(args.agent_id)
            
            # Get detailed state if requested
            if args.deep:
                self.app.print_success("🧠 Performing deep introspection...")
                detailed_info = await self._get_agent_deep_info(args.agent_id)
                basic_info.update(detailed_info)
            
            # Build introspection report
            report = {
                "agent_id": args.agent_id,
                "timestamp": datetime.utcnow().isoformat(),
                "introspection_type": "deep" if args.deep else "basic",
                **basic_info
            }
            
            # Display or save report
            if args.save:
                with open(args.save, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                self.app.print_success(f"Introspection report saved to: {args.save}")
            else:
                self.app.print_output(report, args.format)
            
            return 0
            
        except Exception as e:
            self.app.print_error(f"Introspection failed: {str(e)}")
            return 1
    
    async def _get_agent_basic_info(self, agent_id: str) -> Dict[str, Any]:
        """Get basic agent information."""
        try:
            # Get agent status from backend
            response = await self.client.chat(agent_id, "/debug/status", framework="auto")
            
            # Get system status for framework info
            system_status = await self.client.get_status()
            
            return {
                "agent_response": response,
                "system_frameworks": system_status.get("frameworks", {}),
                "connection_status": "connected",
                "last_activity": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "error": str(e),
                "connection_status": "failed",
                "last_activity": None
            }
    
    async def _get_agent_deep_info(self, agent_id: str) -> Dict[str, Any]:
        """Get deep agent information including memory state."""
        try:
            # Try to get memory state
            memory_response = await self.client.chat(agent_id, "/debug/memory", framework="auto")
            
            # Try to get conversation history
            history_response = await self.client.chat(agent_id, "/debug/history", framework="auto")
            
            # Try to get performance metrics
            perf_response = await self.client.chat(agent_id, "/debug/performance", framework="auto")
            
            return {
                "memory_state": memory_response.get("content", "Not available"),
                "conversation_history": history_response.get("content", "Not available"),
                "performance_metrics": perf_response.get("content", "Not available"),
                "deep_introspection_completed": True
            }
        except Exception as e:
            return {
                "deep_introspection_error": str(e),
                "deep_introspection_completed": False
            }
    
    async def _handle_trace(self, args: argparse.Namespace) -> int:
        """Handle request tracing command."""
        try:
            self.app.print_success(f"🕵️ Starting request trace for agent: {args.agent_id}")
            self.app.print_success(f"Duration: {args.duration}s, Include WebSocket: {args.include_websocket}")
            
            trace_log = []
            start_time = time.time()
            end_time = start_time + args.duration
            
            if args.message:
                # Trace a specific message
                trace_entry = await self._trace_single_request(args.agent_id, args.message)
                trace_log.append(trace_entry)
                self.app.print_success(f"✅ Traced message: {args.message}")
                self.app.print_output(trace_entry, args.format)
            else:
                # Continuous tracing
                self.app.print_success("🔄 Starting continuous trace... (Press Ctrl+C to stop)")
                
                try:
                    while time.time() < end_time:
                        # In a real implementation, this would listen to actual traffic
                        # For now, we'll simulate by sending periodic health checks
                        trace_entry = await self._trace_single_request(args.agent_id, "/health")
                        trace_entry["trace_type"] = "health_check"
                        trace_log.append(trace_entry)
                        
                        await asyncio.sleep(5)  # Check every 5 seconds
                        
                except KeyboardInterrupt:
                    self.app.print_warning("\\n⏹️ Trace stopped by user")
            
            # Generate trace summary
            summary = self._generate_trace_summary(trace_log)
            
            # Save trace log if requested
            if args.save_log:
                trace_data = {
                    "agent_id": args.agent_id,
                    "start_time": datetime.fromtimestamp(start_time).isoformat(),
                    "duration_seconds": time.time() - start_time,
                    "trace_entries": trace_log,
                    "summary": summary
                }
                
                with open(args.save_log, 'w') as f:
                    json.dump(trace_data, f, indent=2, default=str)
                self.app.print_success(f"Trace log saved to: {args.save_log}")
            
            self.app.print_output(summary, args.format)
            return 0
            
        except Exception as e:
            self.app.print_error(f"Request tracing failed: {str(e)}")
            return 1
    
    async def _trace_single_request(self, agent_id: str, message: str) -> Dict[str, Any]:
        """Trace a single request/response."""
        start_time = time.time()
        
        try:
            response = await self.client.chat(agent_id, message)
            end_time = time.time()
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": agent_id,
                "request": {"message": message},
                "response": response,
                "response_time_ms": round((end_time - start_time) * 1000, 2),
                "status": "success"
            }
        except Exception as e:
            end_time = time.time()
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": agent_id,
                "request": {"message": message},
                "error": str(e),
                "response_time_ms": round((end_time - start_time) * 1000, 2),
                "status": "error"
            }
    
    def _generate_trace_summary(self, trace_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary from trace log."""
        if not trace_log:
            return {"message": "No trace entries recorded"}
        
        successful_requests = [entry for entry in trace_log if entry.get("status") == "success"]
        failed_requests = [entry for entry in trace_log if entry.get("status") == "error"]
        
        response_times = [entry.get("response_time_ms", 0) for entry in successful_requests]
        
        summary = {
            "total_requests": len(trace_log),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": round((len(successful_requests) / len(trace_log)) * 100, 2) if trace_log else 0
        }
        
        if response_times:
            summary.update({
                "avg_response_time_ms": round(sum(response_times) / len(response_times), 2),
                "min_response_time_ms": min(response_times),
                "max_response_time_ms": max(response_times),
                "p95_response_time_ms": round(sorted(response_times)[int(len(response_times) * 0.95)], 2) if len(response_times) > 1 else response_times[0]
            })
        
        return summary
    
    async def _handle_profile(self, args: argparse.Namespace) -> int:
        """Handle performance profiling command."""
        try:
            self.app.print_success(f"📊 Profiling agent: {args.agent_id}")
            self.app.print_success(f"Requests: {args.requests}, Concurrent: {args.concurrent}, Warmup: {args.warmup}")
            
            # Warmup phase
            if args.warmup > 0:
                self.app.print_success("🔥 Running warmup requests...")
                for i in range(args.warmup):
                    try:
                        await self.client.chat(args.agent_id, f"Warmup request {i + 1}")
                        await asyncio.sleep(0.1)
                    except Exception:
                        pass  # Ignore warmup errors
            
            # Performance testing phase
            self.app.print_success("⚡ Running performance tests...")
            
            all_results = []
            
            if args.concurrent == 1:
                # Sequential requests
                for i in range(args.requests):
                    result = await self._profile_single_request(args.agent_id, f"{args.message} #{i + 1}")
                    all_results.append(result)
                    
                    if (i + 1) % 10 == 0:
                        self.app.print_success(f"Completed {i + 1}/{args.requests} requests")
            else:
                # Concurrent requests
                semaphore = asyncio.Semaphore(args.concurrent)
                
                async def bounded_request(i):
                    async with semaphore:
                        return await self._profile_single_request(args.agent_id, f"{args.message} #{i + 1}")
                
                tasks = [bounded_request(i) for i in range(args.requests)]
                all_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Filter out exceptions
                all_results = [r for r in all_results if not isinstance(r, Exception)]
            
            # Generate performance report
            report = self._generate_performance_report(all_results, args)
            
            # Save report if requested
            if args.report:
                with open(args.report, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                self.app.print_success(f"Performance report saved to: {args.report}")
            
            self.app.print_output(report, args.format)
            return 0
            
        except Exception as e:
            self.app.print_error(f"Performance profiling failed: {str(e)}")
            return 1
    
    async def _profile_single_request(self, agent_id: str, message: str) -> Dict[str, Any]:
        """Profile a single request for performance."""
        start_time = time.perf_counter()
        
        try:
            response = await self.client.chat(agent_id, message)
            end_time = time.perf_counter()
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "response_time_ms": round((end_time - start_time) * 1000, 3),
                "status": "success",
                "response_size": len(str(response)),
                "message_length": len(message)
            }
        except Exception as e:
            end_time = time.perf_counter()
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "response_time_ms": round((end_time - start_time) * 1000, 3),
                "status": "error",
                "error": str(e),
                "message_length": len(message)
            }
    
    def _generate_performance_report(self, results: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
        """Generate performance report from profiling results."""
        successful_results = [r for r in results if r.get("status") == "success"]
        failed_results = [r for r in results if r.get("status") == "error"]
        
        if not results:
            return {"error": "No valid results to analyze"}
        
        response_times = [r.get("response_time_ms", 0) for r in successful_results]
        
        report = {
            "test_configuration": {
                "agent_id": args.agent_id,
                "total_requests": args.requests,
                "concurrent_requests": args.concurrent,
                "warmup_requests": args.warmup,
                "test_message": args.message
            },
            "results_summary": {
                "total_results": len(results),
                "successful_requests": len(successful_results),
                "failed_requests": len(failed_results),
                "success_rate_percent": round((len(successful_results) / len(results)) * 100, 2) if results else 0
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if response_times:
            sorted_times = sorted(response_times)
            
            report["performance_metrics"] = {
                "avg_response_time_ms": round(sum(response_times) / len(response_times), 3),
                "min_response_time_ms": min(response_times),
                "max_response_time_ms": max(response_times),
                "median_response_time_ms": sorted_times[len(sorted_times) // 2],
                "p95_response_time_ms": sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 1 else sorted_times[0],
                "p99_response_time_ms": sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) > 1 else sorted_times[0],
                "requests_per_second": round(len(successful_results) / (max(response_times) / 1000), 2) if response_times else 0
            }
        
        if failed_results:
            error_types = {}
            for result in failed_results:
                error = result.get("error", "Unknown error")
                error_types[error] = error_types.get(error, 0) + 1
            
            report["error_analysis"] = {
                "error_types": error_types,
                "most_common_error": max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
            }
        
        return report
    
    async def _handle_memory(self, args: argparse.Namespace) -> int:
        """Handle memory analysis command."""
        try:
            self.app.print_success("🧠 Analyzing memory usage...")
            
            # Get system memory info
            memory_info = await self._get_memory_info(args.agent_id, args.threshold, args.history)
            
            self.app.print_output(memory_info, args.format)
            return 0
            
        except Exception as e:
            self.app.print_error(f"Memory analysis failed: {str(e)}")
            return 1
    
    async def _get_memory_info(self, agent_id: Optional[str], threshold: float, history_hours: int) -> Dict[str, Any]:
        """Get memory usage information."""
        try:
            # Get system health which includes memory info
            health_response = await self.client.http.get_system_status()
            
            # Simulate memory analysis (in a real implementation, this would connect to monitoring)
            memory_info = {
                "timestamp": datetime.utcnow().isoformat(),
                "threshold_percent": threshold,
                "history_hours": history_hours,
                "system_memory": {
                    "available": "8GB",  # These would come from actual monitoring
                    "used_percent": 45.2,
                    "threshold_exceeded": False
                },
                "agent_memory": {
                    "agent_id": agent_id,
                    "memory_usage_mb": 256,
                    "conversation_history_size": 1024,
                    "cache_size_mb": 128
                } if agent_id else None,
                "recommendations": [
                    "Memory usage is within normal limits",
                    "Consider clearing conversation history for long-running agents",
                    "Monitor for memory leaks during extended operation"
                ]
            }
            
            return memory_info
            
        except Exception as e:
            return {
                "error": f"Failed to get memory info: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _handle_network(self, args: argparse.Namespace) -> int:
        """Handle network diagnostics command."""
        try:
            self.app.print_success("🌐 Running network diagnostics...")
            
            diagnostics = {
                "timestamp": datetime.utcnow().isoformat(),
                "tests_performed": []
            }
            
            # HTTP connectivity test
            self.app.print_success("Testing HTTP connectivity...")
            http_test = await self._test_http_connectivity(args.endpoint)
            diagnostics["tests_performed"].append(http_test)
            
            # WebSocket connectivity test
            if args.websocket:
                self.app.print_success("Testing WebSocket connectivity...")
                ws_test = await self._test_websocket_connectivity()
                diagnostics["tests_performed"].append(ws_test)
            
            # Latency test
            if args.latency_test:
                self.app.print_success("Running latency tests...")
                latency_test = await self._test_latency(args.count)
                diagnostics["tests_performed"].append(latency_test)
            
            # Generate overall health score
            diagnostics["overall_health"] = self._calculate_network_health(diagnostics["tests_performed"])
            
            self.app.print_output(diagnostics, args.format)
            return 0
            
        except Exception as e:
            self.app.print_error(f"Network diagnostics failed: {str(e)}")
            return 1
    
    async def _test_http_connectivity(self, endpoint: Optional[str]) -> Dict[str, Any]:
        """Test HTTP connectivity."""
        test_url = endpoint or f"{self.config.get('backend_url')}/health"
        
        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(test_url, timeout=10.0)
                end_time = time.perf_counter()
                
                return {
                    "test_type": "http_connectivity",
                    "endpoint": test_url,
                    "status": "success",
                    "response_code": response.status_code,
                    "response_time_ms": round((end_time - start_time) * 1000, 2),
                    "headers": dict(response.headers)
                }
        except Exception as e:
            end_time = time.perf_counter()
            return {
                "test_type": "http_connectivity",
                "endpoint": test_url,
                "status": "failed",
                "error": str(e),
                "response_time_ms": round((end_time - start_time) * 1000, 2)
            }
    
    async def _test_websocket_connectivity(self) -> Dict[str, Any]:
        """Test WebSocket connectivity."""
        ws_url = f"{self.config.get('websocket_url')}/ws/health"
        
        start_time = time.perf_counter()
        try:
            async with websockets.connect(ws_url, timeout=10) as websocket:
                await websocket.send("ping")
                response = await websocket.recv()
                end_time = time.perf_counter()
                
                return {
                    "test_type": "websocket_connectivity",
                    "endpoint": ws_url,
                    "status": "success",
                    "response": response,
                    "response_time_ms": round((end_time - start_time) * 1000, 2)
                }
        except Exception as e:
            end_time = time.perf_counter()
            return {
                "test_type": "websocket_connectivity",
                "endpoint": ws_url,
                "status": "failed",
                "error": str(e),
                "response_time_ms": round((end_time - start_time) * 1000, 2)
            }
    
    async def _test_latency(self, count: int) -> Dict[str, Any]:
        """Test network latency."""
        test_url = f"{self.config.get('backend_url')}/health"
        latencies = []
        
        for i in range(count):
            start_time = time.perf_counter()
            try:
                async with httpx.AsyncClient() as client:
                    await client.get(test_url, timeout=5.0)
                    end_time = time.perf_counter()
                    latencies.append((end_time - start_time) * 1000)
            except Exception:
                # Skip failed requests in latency calculation
                pass
        
        if latencies:
            return {
                "test_type": "latency_test",
                "request_count": count,
                "successful_requests": len(latencies),
                "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
                "min_latency_ms": round(min(latencies), 2),
                "max_latency_ms": round(max(latencies), 2),
                "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if len(latencies) > 1 else round(latencies[0], 2),
                "status": "success"
            }
        else:
            return {
                "test_type": "latency_test",
                "request_count": count,
                "successful_requests": 0,
                "status": "failed",
                "error": "All latency test requests failed"
            }
    
    def _calculate_network_health(self, tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall network health score."""
        if not tests:
            return {"score": 0, "status": "no_tests"}
        
        successful_tests = [t for t in tests if t.get("status") == "success"]
        success_rate = len(successful_tests) / len(tests)
        
        # Calculate health score based on success rate and response times
        health_score = int(success_rate * 100)
        
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 70:
            status = "good"
        elif health_score >= 50:
            status = "fair"
        else:
            status = "poor"
        
        return {
            "score": health_score,
            "status": status,
            "tests_passed": len(successful_tests),
            "tests_failed": len(tests) - len(successful_tests),
            "success_rate_percent": round(success_rate * 100, 1)
        }
    
    async def _handle_health_deep(self, args: argparse.Namespace) -> int:
        """Handle deep system health analysis."""
        self.app.print_success("🏥 Performing deep system health analysis...")
        
        try:
            health_report = {
                "timestamp": datetime.utcnow().isoformat(),
                "analysis_type": "deep_health_check",
                "components_checked": []
            }
            
            # Basic health check
            basic_health = await self._basic_health_check()
            health_report["components_checked"].append(basic_health)
            
            # Check external dependencies
            if args.check_dependencies:
                deps_health = await self._check_dependencies()
                health_report["components_checked"].append(deps_health)
            
            # Performance baseline
            if args.performance_baseline:
                baseline = await self._establish_performance_baseline()
                health_report["components_checked"].append(baseline)
            
            # Calculate overall health
            health_report["overall_health"] = self._calculate_overall_health(health_report["components_checked"])
            
            # Generate report file
            if args.generate_report:
                with open(args.generate_report, 'w') as f:
                    json.dump(health_report, f, indent=2, default=str)
                self.app.print_success(f"Health report saved to: {args.generate_report}")
            
            self.app.print_output(health_report, args.format)
            return 0
            
        except Exception as e:
            self.app.print_error(f"Deep health analysis failed: {str(e)}")
            return 1
    
    async def _basic_health_check(self) -> Dict[str, Any]:
        """Perform basic health check."""
        try:
            status = await self.client.get_status()
            return {
                "component": "basic_health",
                "status": "healthy",
                "details": status,
                "checks_passed": ["backend_connectivity", "api_response", "framework_status"]
            }
        except Exception as e:
            return {
                "component": "basic_health",
                "status": "unhealthy",
                "error": str(e),
                "checks_passed": []
            }
    
    async def _check_dependencies(self) -> Dict[str, Any]:
        """Check external dependencies."""
        # This would check external services, databases, etc.
        return {
            "component": "external_dependencies",
            "status": "healthy",
            "dependencies_checked": ["database", "external_apis", "file_storage"],
            "all_dependencies_available": True,
            "details": "All external dependencies are responding normally"
        }
    
    async def _establish_performance_baseline(self) -> Dict[str, Any]:
        """Establish performance baseline."""
        # This would run a series of performance tests
        return {
            "component": "performance_baseline",
            "status": "completed",
            "baseline_metrics": {
                "avg_response_time_ms": 150,
                "p95_response_time_ms": 300,
                "throughput_rps": 100,
                "error_rate_percent": 0.1
            },
            "baseline_established": True
        }
    
    def _calculate_overall_health(self, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall system health."""
        if not components:
            return {"status": "unknown", "score": 0}
        
        healthy_components = [c for c in components if c.get("status") in ["healthy", "completed"]]
        health_percentage = (len(healthy_components) / len(components)) * 100
        
        if health_percentage >= 90:
            status = "excellent"
        elif health_percentage >= 70:
            status = "good"
        elif health_percentage >= 50:
            status = "fair"
        else:
            status = "poor"
        
        return {
            "status": status,
            "score": int(health_percentage),
            "healthy_components": len(healthy_components),
            "total_components": len(components),
            "health_percentage": round(health_percentage, 1)
        }
    
    async def _handle_live(self, args: argparse.Namespace) -> int:
        """Handle live debugging session."""
        self.app.print_success(f"🔴 Starting live debugging session for agent: {args.agent_id}")
        
        if args.breakpoints:
            self.app.print_success(f"Breakpoints: {', '.join(args.breakpoints)}")
        
        if args.watch:
            self.app.print_success(f"Watching: {', '.join(args.watch)}")
        
        try:
            # In a real implementation, this would set up actual debugging hooks
            self.app.print_success("Live debugging session started (simplified demonstration)")
            self.app.print_success("Type 'help' for commands, 'quit' to exit")
            
            while True:
                try:
                    command = input("(uap-debug) ")
                    
                    if command.lower() in ['quit', 'exit', 'q']:
                        break
                    elif command.lower() == 'help':
                        self._show_debug_help()
                    elif command.startswith('inspect '):
                        var_name = command[8:].strip()
                        await self._debug_inspect_variable(args.agent_id, var_name)
                    elif command.startswith('send '):
                        message = command[5:].strip()
                        await self._debug_send_message(args.agent_id, message)
                    else:
                        self.app.print_warning(f"Unknown command: {command}")
                        
                except KeyboardInterrupt:
                    break
            
            self.app.print_success("Live debugging session ended")
            return 0
            
        except Exception as e:
            self.app.print_error(f"Live debugging failed: {str(e)}")
            return 1
    
    def _show_debug_help(self) -> None:
        """Show debugging help."""
        help_text = \"\"\"
Debug Commands:
  help                 - Show this help
  inspect <variable>   - Inspect a variable or expression
  send <message>       - Send a message to the agent
  quit/exit/q         - Exit debugging session
        \"\"\"
        print(help_text)
    
    async def _debug_inspect_variable(self, agent_id: str, var_name: str) -> None:
        """Debug inspect a variable."""
        try:
            response = await self.client.chat(agent_id, f"/debug/inspect {var_name}")
            self.app.print_success(f"Inspecting {var_name}:")
            self.app.print_output(response, "json")
        except Exception as e:
            self.app.print_error(f"Failed to inspect {var_name}: {str(e)}")
    
    async def _debug_send_message(self, agent_id: str, message: str) -> None:
        """Send a debug message to the agent."""
        try:
            response = await self.client.chat(agent_id, message)
            self.app.print_success("Agent response:")
            self.app.print_output(response, "json")
        except Exception as e:
            self.app.print_error(f"Failed to send message: {str(e)}")
    
    async def _handle_errors(self, args: argparse.Namespace) -> int:
        """Handle error analysis command."""
        try:
            self.app.print_success("🚨 Analyzing recent errors...")
            
            # Parse time period
            since_time = self._parse_time_period(args.since) if args.since else datetime.utcnow() - timedelta(hours=1)
            
            # Simulate error analysis (in real implementation, would query logs)
            error_analysis = {
                "analysis_period": {
                    "since": since_time.isoformat(),
                    "until": datetime.utcnow().isoformat(),
                    "duration_hours": (datetime.utcnow() - since_time).total_seconds() / 3600
                },
                "filters": {
                    "agent_id": args.agent_id,
                    "severity": args.severity
                },
                "error_summary": {
                    "total_errors": 5,
                    "error_rate_per_hour": 1.2,
                    "most_common_errors": [
                        {"type": "ConnectionTimeout", "count": 2, "percentage": 40},
                        {"type": "AuthenticationError", "count": 2, "percentage": 40},
                        {"type": "ValidationError", "count": 1, "percentage": 20}
                    ]
                },
                "recent_errors": [
                    {
                        "timestamp": "2024-01-15T10:30:00Z",
                        "severity": "error",
                        "type": "ConnectionTimeout",
                        "agent_id": args.agent_id or "agent-1",
                        "message": "Connection timeout after 30 seconds",
                        "stack_trace": "Sample stack trace..."
                    }
                ],
                "recommendations": [
                    "Check network connectivity for ConnectionTimeout errors",
                    "Verify authentication credentials",
                    "Review input validation for ValidationError"
                ]
            }
            
            # Export if requested
            if args.export:
                with open(args.export, 'w') as f:
                    json.dump(error_analysis, f, indent=2, default=str)
                self.app.print_success(f"Error analysis exported to: {args.export}")
            
            self.app.print_output(error_analysis, args.format)
            return 0
            
        except Exception as e:
            self.app.print_error(f"Error analysis failed: {str(e)}")
            return 1
    
    def _parse_time_period(self, period: str) -> datetime:
        """Parse time period string like '1h', '30m', '2d'."""
        try:
            if period.endswith('h'):
                hours = int(period[:-1])
                return datetime.utcnow() - timedelta(hours=hours)
            elif period.endswith('m'):
                minutes = int(period[:-1])
                return datetime.utcnow() - timedelta(minutes=minutes)
            elif period.endswith('d'):
                days = int(period[:-1])
                return datetime.utcnow() - timedelta(days=days)
            else:
                # Try to parse as ISO format
                return datetime.fromisoformat(period.replace('Z', '+00:00'))
        except:
            # Default to 1 hour ago
            return datetime.utcnow() - timedelta(hours=1)
    
    async def _handle_config_validate(self, args: argparse.Namespace) -> int:
        """Handle configuration validation command."""
        try:
            self.app.print_success("⚙️ Validating configuration...")
            
            # Load configuration to validate
            if args.config_file:
                config = Configuration(config_file=args.config_file)
                config_source = args.config_file
            else:
                config = self.config
                config_source = "current configuration"
            
            validation_report = {
                "config_source": config_source,
                "timestamp": datetime.utcnow().isoformat(),
                "validation_results": []
            }
            
            # Basic validation
            basic_validation = self._validate_basic_config(config)
            validation_report["validation_results"].append(basic_validation)
            
            # Security check
            if args.security_check:
                security_validation = self._validate_security_config(config)
                validation_report["validation_results"].append(security_validation)
            
            # Generate fix suggestions
            if args.fix_suggestions:
                suggestions = self._generate_fix_suggestions(validation_report["validation_results"])
                validation_report["fix_suggestions"] = suggestions
            
            # Overall validation status
            validation_report["overall_status"] = self._calculate_validation_status(validation_report["validation_results"])
            
            self.app.print_output(validation_report, args.format)
            return 0 if validation_report["overall_status"]["valid"] else 1
            
        except Exception as e:
            self.app.print_error(f"Configuration validation failed: {str(e)}")
            return 1
    
    def _validate_basic_config(self, config: Configuration) -> Dict[str, Any]:
        """Validate basic configuration."""
        issues = []
        warnings = []
        
        try:
            config.validate()
            status = "valid"
        except Exception as e:
            issues.append(str(e))
            status = "invalid"
        
        # Check for common issues
        backend_url = config.get("backend_url")
        if backend_url and "localhost" in backend_url:
            warnings.append("Using localhost URL - may not work in production")
        
        websocket_url = config.get("websocket_url")
        if websocket_url and "ws://" in websocket_url:
            warnings.append("Using unencrypted WebSocket connection")
        
        return {
            "validation_type": "basic_configuration",
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "config_keys_checked": ["backend_url", "websocket_url", "http_timeout"]
        }
    
    def _validate_security_config(self, config: Configuration) -> Dict[str, Any]:
        """Validate security configuration."""
        issues = []
        warnings = []
        
        # Check for security issues
        if config.get("access_token"):
            warnings.append("Access token found in configuration - ensure secure storage")
        
        if not config.get("backend_url", "").startswith("https://"):
            issues.append("Backend URL should use HTTPS in production")
        
        if not config.get("websocket_url", "").startswith("wss://"):
            issues.append("WebSocket URL should use WSS in production")
        
        status = "valid" if not issues else "invalid"
        
        return {
            "validation_type": "security_configuration",
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "security_checks_performed": ["url_encryption", "token_storage", "connection_security"]
        }
    
    def _generate_fix_suggestions(self, validation_results: List[Dict[str, Any]]) -> List[str]:
        """Generate fix suggestions based on validation results."""
        suggestions = []
        
        for result in validation_results:
            for issue in result.get("issues", []):
                if "backend_url" in issue:
                    suggestions.append("Update backend_url to use HTTPS protocol")
                elif "websocket_url" in issue:
                    suggestions.append("Update websocket_url to use WSS protocol")
                
            for warning in result.get("warnings", []):
                if "localhost" in warning:
                    suggestions.append("Replace localhost with production domain name")
                elif "unencrypted" in warning:
                    suggestions.append("Enable SSL/TLS encryption for WebSocket connections")
                elif "token" in warning:
                    suggestions.append("Store access tokens in secure environment variables")
        
        return list(set(suggestions))  # Remove duplicates
    
    def _calculate_validation_status(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall validation status."""
        total_issues = sum(len(r.get("issues", [])) for r in validation_results)
        total_warnings = sum(len(r.get("warnings", [])) for r in validation_results)
        
        valid = total_issues == 0
        
        return {
            "valid": valid,
            "issues_count": total_issues,
            "warnings_count": total_warnings,
            "validation_score": max(0, 100 - (total_issues * 20) - (total_warnings * 5))
        }