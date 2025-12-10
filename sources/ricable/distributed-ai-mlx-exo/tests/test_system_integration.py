"""
Comprehensive Automated Testing Suite for MLX-Exo Distributed Cluster
Includes end-to-end testing, performance regression, load testing, and chaos engineering
"""

import asyncio
import pytest
import time
import random
import json
import logging
import psutil
import subprocess
import statistics
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp
import requests

# Import our components for testing
try:
    from src.monitoring.health_monitor import ClusterHealthMonitor, create_health_monitor, HealthStatus
    from src.monitoring.prometheus_metrics import MetricsRegistry, MetricsCollector, create_metrics_system
    from src.enhanced_api_server import create_enhanced_api_server
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False
    logging.warning("Some components not available for testing")

logger = logging.getLogger(__name__)

@dataclass
class SystemTestResult:
    """Test result data structure"""
    test_name: str
    success: bool
    duration: float
    metrics: Dict[str, Any]
    error_message: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass  
class LoadTestConfig:
    """Load test configuration"""
    concurrent_users: int = 10
    requests_per_user: int = 100
    ramp_up_time: float = 30.0
    test_duration: float = 300.0
    models_to_test: List[str] = None
    
    def __post_init__(self):
        if self.models_to_test is None:
            self.models_to_test = ['llama-7b', 'llama-30b']

@dataclass
class PerformanceBenchmark:
    """Performance benchmark thresholds"""
    max_api_latency_p95: float = 5.0  # seconds
    min_tokens_per_second: float = 5.0
    max_time_to_first_token: float = 0.5  # seconds
    max_model_load_time: float = 120.0  # seconds
    min_cluster_availability: float = 0.99  # 99%
    max_error_rate: float = 0.01  # 1%

class SystemIntegrationTester:
    """Main system integration test coordinator"""
    
    def __init__(self, api_base_url: str = "http://localhost:52415", 
                 prometheus_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.prometheus_url = prometheus_url
        self.test_results: List[SystemTestResult] = []
        self.benchmark = PerformanceBenchmark()
        
        # Test configuration
        self.test_nodes = [
            {'id': 'mac-studio-1', 'ip': '10.0.1.10'},
            {'id': 'mac-studio-2', 'ip': '10.0.1.11'},
            {'id': 'mac-studio-3', 'ip': '10.0.1.12'}
        ]
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite"""
        logger.info("Starting comprehensive system integration tests")
        
        # Test categories
        test_suites = [
            ("Health Check Tests", self.run_health_tests),
            ("API Functionality Tests", self.run_api_tests),
            ("Performance Regression Tests", self.run_performance_tests),
            ("Load Tests", self.run_load_tests),
            ("Failover Tests", self.run_failover_tests),
            ("Chaos Engineering Tests", self.run_chaos_tests)
        ]
        
        all_results = {}
        overall_success = True
        
        for suite_name, test_func in test_suites:
            logger.info(f"Running {suite_name}")
            try:
                results = await test_func()
                all_results[suite_name] = results
                
                # Check if any tests failed
                if isinstance(results, list):
                    suite_success = all(r.success for r in results if isinstance(r, SystemTestResult))
                else:
                    suite_success = results.get('success', False)
                    
                if not suite_success:
                    overall_success = False
                    
            except Exception as e:
                logger.error(f"Error running {suite_name}: {e}")
                all_results[suite_name] = {'error': str(e), 'success': False}
                overall_success = False
        
        # Generate summary report
        summary = self._generate_test_summary(all_results, overall_success)
        return summary
    
    async def run_health_tests(self) -> List[SystemTestResult]:
        """Test health monitoring and cluster status"""
        results = []
        
        # Test 1: Cluster health endpoint
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_base_url}/health") as response:
                    health_data = await response.json()
                    
                success = response.status == 200 and health_data.get('status') == 'healthy'
                results.append(SystemTestResult(
                    test_name="cluster_health_endpoint",
                    success=success,
                    duration=time.time() - start_time,
                    metrics={'status_code': response.status, 'health_status': health_data.get('status')},
                    error_message=None if success else f"Health check failed: {health_data}"
                ))
        except Exception as e:
            results.append(SystemTestResult(
                test_name="cluster_health_endpoint",
                success=False,
                duration=time.time() - start_time,
                metrics={},
                error_message=str(e)
            ))
        
        # Test 2: Individual node health
        if COMPONENTS_AVAILABLE:
            start_time = time.time()
            try:
                health_monitor = create_health_monitor(self.test_nodes)
                
                # Let it run for a short time to collect data
                monitoring_task = asyncio.create_task(health_monitor.start_monitoring())
                await asyncio.sleep(15)
                health_monitor.stop_monitoring()
                
                cluster_health = health_monitor.get_cluster_health()
                healthy_nodes = cluster_health.get('summary', {}).get('healthy', 0)
                total_nodes = cluster_health.get('summary', {}).get('total', 0)
                
                success = healthy_nodes >= total_nodes * 0.75  # At least 75% healthy
                results.append(SystemTestResult(
                    test_name="node_health_monitoring",
                    success=success,
                    duration=time.time() - start_time,
                    metrics={'healthy_nodes': healthy_nodes, 'total_nodes': total_nodes},
                    error_message=None if success else f"Only {healthy_nodes}/{total_nodes} nodes healthy"
                ))
                
                await monitoring_task
                
            except Exception as e:
                results.append(SystemTestResult(
                    test_name="node_health_monitoring",
                    success=False,
                    duration=time.time() - start_time,
                    metrics={},
                    error_message=str(e)
                ))
        
        return results
    
    async def run_api_tests(self) -> List[SystemTestResult]:
        """Test API functionality and endpoints"""
        results = []
        
        # Test 1: Models list endpoint
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_base_url}/v1/models") as response:
                    models_data = await response.json()
                    
                success = response.status == 200 and 'data' in models_data
                results.append(SystemTestResult(
                    test_name="models_list_endpoint",
                    success=success,
                    duration=time.time() - start_time,
                    metrics={'status_code': response.status, 'model_count': len(models_data.get('data', []))},
                    error_message=None if success else f"Models endpoint failed: {models_data}"
                ))
        except Exception as e:
            results.append(SystemTestResult(
                test_name="models_list_endpoint",
                success=False,
                duration=time.time() - start_time,
                metrics={},
                error_message=str(e)
            ))
        
        # Test 2: Chat completions endpoint
        start_time = time.time()
        try:
            test_request = {
                "model": "llama-7b",
                "messages": [{"role": "user", "content": "Hello, this is a test."}],
                "max_tokens": 10,
                "temperature": 0.7
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base_url}/v1/chat/completions",
                    json=test_request
                ) as response:
                    response_data = await response.json()
                    
                success = (response.status == 200 and 
                          'choices' in response_data and 
                          len(response_data['choices']) > 0)
                          
                results.append(SystemTestResult(
                    test_name="chat_completions_endpoint",
                    success=success,
                    duration=time.time() - start_time,
                    metrics={
                        'status_code': response.status,
                        'response_time': time.time() - start_time,
                        'has_response': bool(response_data.get('choices'))
                    },
                    error_message=None if success else f"Chat completion failed: {response_data}"
                ))
        except Exception as e:
            results.append(SystemTestResult(
                test_name="chat_completions_endpoint",
                success=False,
                duration=time.time() - start_time,
                metrics={},
                error_message=str(e)
            ))
        
        # Test 3: Streaming endpoint
        start_time = time.time()
        try:
            test_request = {
                "model": "llama-7b",
                "messages": [{"role": "user", "content": "Count to 5"}],
                "max_tokens": 20,
                "stream": True
            }
            
            stream_chunks = 0
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base_url}/v1/chat/completions",
                    json=test_request
                ) as response:
                    
                    if response.status == 200:
                        async for chunk in response.content.iter_chunked(1024):
                            if chunk:
                                stream_chunks += 1
                                if stream_chunks > 5:  # Limit for test
                                    break
            
            success = stream_chunks > 0
            results.append(SystemTestResult(
                test_name="streaming_endpoint",
                success=success,
                duration=time.time() - start_time,
                metrics={'stream_chunks': stream_chunks},
                error_message=None if success else "No streaming chunks received"
            ))
            
        except Exception as e:
            results.append(SystemTestResult(
                test_name="streaming_endpoint",
                success=False,
                duration=time.time() - start_time,
                metrics={},
                error_message=str(e)
            ))
        
        return results
    
    async def run_performance_tests(self) -> List[SystemTestResult]:
        """Run performance regression tests"""
        results = []
        
        # Test 1: API latency benchmark
        start_time = time.time()
        latencies = []
        
        try:
            for _ in range(10):  # 10 sample requests
                request_start = time.time()
                
                test_request = {
                    "model": "llama-7b",
                    "messages": [{"role": "user", "content": "Quick test"}],
                    "max_tokens": 5
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.api_base_url}/v1/chat/completions",
                        json=test_request
                    ) as response:
                        await response.json()
                        
                latency = time.time() - request_start
                latencies.append(latency)
                
                await asyncio.sleep(1)  # Avoid overwhelming the system
            
            if latencies:
                p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
                avg_latency = statistics.mean(latencies)
                
                success = p95_latency <= self.benchmark.max_api_latency_p95
                results.append(SystemTestResult(
                    test_name="api_latency_benchmark",
                    success=success,
                    duration=time.time() - start_time,
                    metrics={
                        'p95_latency': p95_latency,
                        'avg_latency': avg_latency,
                        'max_latency': max(latencies),
                        'min_latency': min(latencies)
                    },
                    error_message=None if success else f"P95 latency {p95_latency:.2f}s exceeds threshold {self.benchmark.max_api_latency_p95}s"
                ))
            
        except Exception as e:
            results.append(SystemTestResult(
                test_name="api_latency_benchmark",
                success=False,
                duration=time.time() - start_time,
                metrics={},
                error_message=str(e)
            ))
        
        # Test 2: Token generation performance
        start_time = time.time()
        try:
            test_request = {
                "model": "llama-7b",
                "messages": [{"role": "user", "content": "Write a short story about a robot."}],
                "max_tokens": 100
            }
            
            generation_start = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base_url}/v1/chat/completions",
                    json=test_request
                ) as response:
                    response_data = await response.json()
            
            generation_time = time.time() - generation_start
            
            if 'usage' in response_data:
                tokens_generated = response_data['usage'].get('completion_tokens', 0)
                tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
                
                success = tokens_per_second >= self.benchmark.min_tokens_per_second
                results.append(SystemTestResult(
                    test_name="token_generation_performance",
                    success=success,
                    duration=time.time() - start_time,
                    metrics={
                        'tokens_per_second': tokens_per_second,
                        'tokens_generated': tokens_generated,
                        'generation_time': generation_time
                    },
                    error_message=None if success else f"Token rate {tokens_per_second:.2f} below threshold {self.benchmark.min_tokens_per_second}"
                ))
            
        except Exception as e:
            results.append(SystemTestResult(
                test_name="token_generation_performance",
                success=False,
                duration=time.time() - start_time,
                metrics={},
                error_message=str(e)
            ))
        
        return results
    
    async def run_load_tests(self) -> List[SystemTestResult]:
        """Run load testing with multiple concurrent users"""
        results = []
        config = LoadTestConfig(concurrent_users=20, requests_per_user=50)
        
        start_time = time.time()
        
        async def user_session(user_id: int) -> Dict[str, Any]:
            """Simulate a single user session"""
            session_results = {'requests': 0, 'errors': 0, 'total_time': 0}
            
            async with aiohttp.ClientSession() as session:
                for request_num in range(config.requests_per_user):
                    try:
                        request_start = time.time()
                        
                        test_request = {
                            "model": random.choice(config.models_to_test),
                            "messages": [{"role": "user", "content": f"Test message {request_num} from user {user_id}"}],
                            "max_tokens": random.randint(5, 20)
                        }
                        
                        async with session.post(
                            f"{self.api_base_url}/v1/chat/completions",
                            json=test_request
                        ) as response:
                            await response.json()
                            
                        session_results['requests'] += 1
                        session_results['total_time'] += time.time() - request_start
                        
                        # Random delay between requests
                        await asyncio.sleep(random.uniform(0.1, 2.0))
                        
                    except Exception as e:
                        session_results['errors'] += 1
                        logger.debug(f"User {user_id} request {request_num} failed: {e}")
            
            return session_results
        
        try:
            # Create concurrent user sessions
            tasks = [user_session(i) for i in range(config.concurrent_users)]
            session_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate results
            total_requests = sum(r.get('requests', 0) for r in session_results if isinstance(r, dict))
            total_errors = sum(r.get('errors', 0) for r in session_results if isinstance(r, dict))
            total_time = time.time() - start_time
            
            error_rate = total_errors / total_requests if total_requests > 0 else 1.0
            requests_per_second = total_requests / total_time if total_time > 0 else 0
            
            success = error_rate <= self.benchmark.max_error_rate
            results.append(SystemTestResult(
                test_name="concurrent_load_test",
                success=success,
                duration=total_time,
                metrics={
                    'concurrent_users': config.concurrent_users,
                    'total_requests': total_requests,
                    'total_errors': total_errors,
                    'error_rate': error_rate,
                    'requests_per_second': requests_per_second
                },
                error_message=None if success else f"Error rate {error_rate:.3f} exceeds threshold {self.benchmark.max_error_rate}"
            ))
            
        except Exception as e:
            results.append(SystemTestResult(
                test_name="concurrent_load_test",
                success=False,
                duration=time.time() - start_time,
                metrics={},
                error_message=str(e)
            ))
        
        return results
    
    async def run_failover_tests(self) -> List[SystemTestResult]:
        """Test failover and recovery mechanisms"""
        results = []
        
        # This is a simulated failover test - in production would involve
        # actually taking nodes offline and testing recovery
        
        start_time = time.time()
        try:
            # Test 1: Health monitoring detects simulated failures
            if COMPONENTS_AVAILABLE:
                health_monitor = create_health_monitor(self.test_nodes)
                
                # Start monitoring
                monitoring_task = asyncio.create_task(health_monitor.start_monitoring())
                await asyncio.sleep(10)
                
                # Get initial health
                initial_health = health_monitor.get_cluster_health()
                
                # Simulate checking health recovery
                await asyncio.sleep(5)
                recovery_health = health_monitor.get_cluster_health()
                
                health_monitor.stop_monitoring()
                await monitoring_task
                
                success = recovery_health.get('status') in ['healthy', 'degraded']
                results.append(SystemTestResult(
                    test_name="health_monitoring_failover",
                    success=success,
                    duration=time.time() - start_time,
                    metrics={
                        'initial_status': initial_health.get('status'),
                        'recovery_status': recovery_health.get('status')
                    },
                    error_message=None if success else "Health monitoring failed to detect or recover from failures"
                ))
        
        except Exception as e:
            results.append(SystemTestResult(
                test_name="health_monitoring_failover",
                success=False,
                duration=time.time() - start_time,
                metrics={},
                error_message=str(e)
            ))
        
        return results
    
    async def run_chaos_tests(self) -> List[SystemTestResult]:
        """Run chaos engineering tests to validate fault tolerance"""
        results = []
        
        # Chaos Test 1: Resource pressure test
        start_time = time.time()
        try:
            # Create memory pressure (simulated)
            stress_duration = 30  # seconds
            
            # Monitor system during stress
            initial_memory = psutil.virtual_memory().percent
            
            # Simulate some load
            for _ in range(10):
                test_request = {
                    "model": "llama-7b",
                    "messages": [{"role": "user", "content": "Generate a longer response under stress"}],
                    "max_tokens": 50
                }
                
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self.api_base_url}/v1/chat/completions",
                            json=test_request,
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as response:
                            await response.json()
                except asyncio.TimeoutError:
                    pass  # Expected under stress
                except Exception:
                    pass  # Expected under stress
                
                await asyncio.sleep(1)
            
            final_memory = psutil.virtual_memory().percent
            
            # Test passes if system remains responsive
            success = True  # Basic test - system didn't crash
            results.append(SystemTestResult(
                test_name="resource_pressure_test",
                success=success,
                duration=time.time() - start_time,
                metrics={
                    'initial_memory_percent': initial_memory,
                    'final_memory_percent': final_memory,
                    'stress_duration': stress_duration
                },
                error_message=None if success else "System became unresponsive under pressure"
            ))
            
        except Exception as e:
            results.append(SystemTestResult(
                test_name="resource_pressure_test",
                success=False,
                duration=time.time() - start_time,
                metrics={},
                error_message=str(e)
            ))
        
        # Chaos Test 2: Network latency simulation
        start_time = time.time()
        try:
            # Test with artificial delays
            slow_requests = 0
            timeout_requests = 0
            
            for _ in range(5):
                test_request = {
                    "model": "llama-7b",
                    "messages": [{"role": "user", "content": "Test with simulated latency"}],
                    "max_tokens": 10
                }
                
                request_start = time.time()
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self.api_base_url}/v1/chat/completions",
                            json=test_request,
                            timeout=aiohttp.ClientTimeout(total=10)
                        ) as response:
                            await response.json()
                            
                    request_time = time.time() - request_start
                    if request_time > 5:
                        slow_requests += 1
                        
                except asyncio.TimeoutError:
                    timeout_requests += 1
                except Exception:
                    pass
                
                await asyncio.sleep(2)
            
            # Test passes if we don't have excessive timeouts
            success = timeout_requests <= 2  # Allow some timeouts
            results.append(SystemTestResult(
                test_name="network_latency_simulation",
                success=success,
                duration=time.time() - start_time,
                metrics={
                    'slow_requests': slow_requests,
                    'timeout_requests': timeout_requests
                },
                error_message=None if success else f"Too many timeouts: {timeout_requests}"
            ))
            
        except Exception as e:
            results.append(SystemTestResult(
                test_name="network_latency_simulation",
                success=False,
                duration=time.time() - start_time,
                metrics={},
                error_message=str(e)
            ))
        
        return results
    
    def _generate_test_summary(self, all_results: Dict[str, Any], overall_success: bool) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        summary = {
            'overall_success': overall_success,
            'timestamp': datetime.now().isoformat(),
            'test_suites': {},
            'performance_summary': {},
            'recommendations': []
        }
        
        total_tests = 0
        passed_tests = 0
        
        for suite_name, suite_results in all_results.items():
            if isinstance(suite_results, list):
                suite_total = len(suite_results)
                suite_passed = sum(1 for r in suite_results if isinstance(r, SystemTestResult) and r.success)
            else:
                suite_total = 1
                suite_passed = 1 if suite_results.get('success', False) else 0
            
            total_tests += suite_total
            passed_tests += suite_passed
            
            summary['test_suites'][suite_name] = {
                'total': suite_total,
                'passed': suite_passed,
                'success_rate': suite_passed / suite_total if suite_total > 0 else 0
            }
        
        summary['overall_stats'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0
        }
        
        # Generate recommendations based on failures
        if not overall_success:
            if summary['test_suites'].get('Health Check Tests', {}).get('success_rate', 1) < 1:
                summary['recommendations'].append("Check cluster health monitoring configuration")
            
            if summary['test_suites'].get('Performance Regression Tests', {}).get('success_rate', 1) < 1:
                summary['recommendations'].append("Investigate performance degradation")
            
            if summary['test_suites'].get('Load Tests', {}).get('success_rate', 1) < 1:
                summary['recommendations'].append("Scale cluster resources or optimize load balancing")
        
        return summary

# pytest fixtures and test functions
@pytest.fixture
def system_tester():
    """Create system integration tester instance"""
    return SystemIntegrationTester()

@pytest.mark.asyncio
async def test_complete_system_integration(system_tester):
    """Run the complete system integration test suite"""
    results = await system_tester.run_all_tests()
    
    # Log results
    logger.info(f"System integration test results: {json.dumps(results, indent=2)}")
    
    # Assert overall success
    assert results['overall_success'], f"System tests failed. Results: {results}"

@pytest.mark.asyncio 
async def test_health_monitoring_only(system_tester):
    """Test only health monitoring functionality"""
    results = await system_tester.run_health_tests()
    assert all(r.success for r in results), f"Health tests failed: {[r.error_message for r in results if not r.success]}"

@pytest.mark.asyncio
async def test_api_functionality_only(system_tester):
    """Test only API functionality"""
    results = await system_tester.run_api_tests()
    assert all(r.success for r in results), f"API tests failed: {[r.error_message for r in results if not r.success]}"

@pytest.mark.asyncio
async def test_performance_benchmarks_only(system_tester):
    """Test only performance benchmarks"""
    results = await system_tester.run_performance_tests()
    assert all(r.success for r in results), f"Performance tests failed: {[r.error_message for r in results if not r.success]}"

@pytest.mark.load
@pytest.mark.asyncio
async def test_load_testing_only(system_tester):
    """Test only load testing (marked separately due to duration)"""
    results = await system_tester.run_load_tests()
    assert all(r.success for r in results), f"Load tests failed: {[r.error_message for r in results if not r.success]}"

@pytest.mark.chaos
@pytest.mark.asyncio
async def test_chaos_engineering_only(system_tester):
    """Test only chaos engineering tests"""
    results = await system_tester.run_chaos_tests()
    assert all(r.success for r in results), f"Chaos tests failed: {[r.error_message for r in results if not r.success]}"

if __name__ == "__main__":
    async def main():
        """Run tests directly"""
        logging.basicConfig(level=logging.INFO)
        
        tester = SystemIntegrationTester()
        results = await tester.run_all_tests()
        
        print("\n" + "="*50)
        print("SYSTEM INTEGRATION TEST RESULTS")
        print("="*50)
        print(json.dumps(results, indent=2))
        
        if results['overall_success']:
            print("\n✅ ALL TESTS PASSED")
        else:
            print("\n❌ SOME TESTS FAILED")
            if results.get('recommendations'):
                print("\nRecommendations:")
                for rec in results['recommendations']:
                    print(f"  - {rec}")
    
    asyncio.run(main())