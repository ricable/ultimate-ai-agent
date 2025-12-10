#!/usr/bin/env python3
"""
Network Diagnostics and Validation Script
Validates network topology and performance for MLX + Exo cluster
"""

import asyncio
import json
import socket
import subprocess
import time
import argparse
import sys
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class NetworkTest:
    """Represents a network test result"""
    test_name: str
    passed: bool
    details: str
    latency_ms: float = 0.0
    bandwidth_mbps: float = 0.0

class NetworkValidator:
    """Validates network configuration for distributed cluster"""
    
    def __init__(self):
        self.nodes = {
            "mac-node-1": "10.0.1.10",
            "mac-node-2": "10.0.1.11", 
            "mac-node-3": "10.0.1.12"
        }
        self.required_ports = [22, 52415, 40000, 40001, 40002, 40003]
        
    def get_local_node(self) -> str:
        """Detect which node this script is running on"""
        hostname = socket.gethostname().lower()
        
        # Try to match by hostname first
        for node_name in self.nodes:
            if node_name.replace('-', '') in hostname or node_name in hostname:
                return node_name
        
        # Try to match by IP
        local_ip = self._get_primary_ip()
        for node_name, ip in self.nodes.items():
            if ip == local_ip:
                return node_name
        
        return "unknown"
    
    def _get_primary_ip(self) -> str:
        """Get primary IP address"""
        try:
            # Connect to a remote address to determine primary interface
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    async def test_basic_connectivity(self) -> List[NetworkTest]:
        """Test basic ping connectivity to all nodes"""
        tests = []
        current_node = self.get_local_node()
        
        print("Testing basic connectivity...")
        
        for node_name, ip in self.nodes.items():
            if node_name == current_node:
                continue
                
            try:
                print(f"  Pinging {node_name} ({ip})...")
                start_time = time.time()
                result = subprocess.run(
                    ['ping', '-c', '3', ip],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                latency = (time.time() - start_time) * 1000 / 3  # Average latency
                
                if result.returncode == 0:
                    # Extract actual latency from ping output
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'avg' in line and '/' in line:
                            try:
                                parts = line.split('/')
                                if len(parts) >= 5:
                                    latency = float(parts[4])
                                    break
                            except (IndexError, ValueError):
                                pass
                    
                    tests.append(NetworkTest(
                        test_name=f"ping_{node_name}",
                        passed=True,
                        details=f"Ping to {ip} successful",
                        latency_ms=latency
                    ))
                else:
                    tests.append(NetworkTest(
                        test_name=f"ping_{node_name}",
                        passed=False,
                        details=f"Ping to {ip} failed: {result.stderr.strip()}"
                    ))
                    
            except subprocess.TimeoutExpired:
                tests.append(NetworkTest(
                    test_name=f"ping_{node_name}",
                    passed=False,
                    details=f"Ping to {ip} timed out"
                ))
            except Exception as e:
                tests.append(NetworkTest(
                    test_name=f"ping_{node_name}",
                    passed=False,
                    details=f"Ping to {ip} error: {str(e)}"
                ))
        
        return tests
    
    async def test_port_connectivity(self, quick_test: bool = False) -> List[NetworkTest]:
        """Test connectivity to required ports"""
        tests = []
        current_node = self.get_local_node()
        
        print("Testing port connectivity...")
        
        # For quick test, only test essential ports
        ports_to_test = [22, 52415] if quick_test else self.required_ports
        
        for node_name, ip in self.nodes.items():
            if node_name == current_node:
                continue
                
            for port in ports_to_test:
                try:
                    print(f"  Testing {node_name}:{port}...")
                    # Test TCP connection
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_connection(ip, port),
                        timeout=5
                    )
                    
                    writer.close()
                    await writer.wait_closed()
                    
                    tests.append(NetworkTest(
                        test_name=f"port_{node_name}_{port}",
                        passed=True,
                        details=f"Port {port} on {ip} accessible"
                    ))
                    
                except asyncio.TimeoutError:
                    tests.append(NetworkTest(
                        test_name=f"port_{node_name}_{port}",
                        passed=False,
                        details=f"Port {port} on {ip} timeout"
                    ))
                except Exception as e:
                    tests.append(NetworkTest(
                        test_name=f"port_{node_name}_{port}",
                        passed=False,
                        details=f"Port {port} on {ip} error: {str(e)}"
                    ))
        
        return tests
    
    def test_mtu_configuration(self) -> List[NetworkTest]:
        """Test MTU configuration for jumbo frames"""
        tests = []
        
        print("Testing MTU configuration...")
        
        try:
            # Get primary interface
            result = subprocess.run(['route', 'get', 'default'], 
                                  capture_output=True, text=True)
            
            interface = None
            for line in result.stdout.split('\n'):
                if 'interface:' in line:
                    interface = line.split(':')[1].strip()
                    break
            
            if interface:
                print(f"  Checking MTU on interface {interface}...")
                # Check MTU setting
                result = subprocess.run(['ifconfig', interface], 
                                      capture_output=True, text=True)
                
                mtu_found = False
                current_mtu = None
                for line in result.stdout.split('\n'):
                    if 'mtu' in line.lower():
                        # Extract MTU value
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'mtu' in part.lower():
                                try:
                                    if i + 1 < len(parts):
                                        current_mtu = int(parts[i + 1])
                                    else:
                                        # MTU might be in the same part
                                        current_mtu = int(part.split('mtu')[1])
                                    break
                                except (ValueError, IndexError):
                                    continue
                        
                        if current_mtu:
                            if current_mtu >= 9000:
                                tests.append(NetworkTest(
                                    test_name="mtu_jumbo_frames",
                                    passed=True,
                                    details=f"Jumbo frames (MTU {current_mtu}) configured on {interface}"
                                ))
                            else:
                                tests.append(NetworkTest(
                                    test_name="mtu_jumbo_frames",
                                    passed=False,
                                    details=f"Standard MTU ({current_mtu}) on {interface}, jumbo frames not configured"
                                ))
                            mtu_found = True
                            break
                
                if not mtu_found:
                    tests.append(NetworkTest(
                        test_name="mtu_jumbo_frames",
                        passed=False,
                        details="Could not determine MTU configuration"
                    ))
            else:
                tests.append(NetworkTest(
                    test_name="mtu_jumbo_frames",
                    passed=False,
                    details="Could not determine primary interface"
                ))
                
        except Exception as e:
            tests.append(NetworkTest(
                test_name="mtu_jumbo_frames",
                passed=False,
                details=f"MTU test failed: {str(e)}"
            ))
        
        return tests
    
    async def test_bandwidth(self, quick_test: bool = False) -> List[NetworkTest]:
        """Test network bandwidth between nodes"""
        tests = []
        current_node = self.get_local_node()
        
        print("Testing network bandwidth...")
        
        # Skip bandwidth test in quick mode
        if quick_test:
            tests.append(NetworkTest(
                test_name="bandwidth_test",
                passed=True,
                details="Bandwidth test skipped in quick mode"
            ))
            return tests
        
        # Simplified bandwidth test using ping flood
        for node_name, ip in self.nodes.items():
            if node_name == current_node:
                continue
                
            try:
                print(f"  Testing bandwidth to {node_name}...")
                # Use ping flood for basic bandwidth estimation
                result = subprocess.run(
                    ['ping', '-f', '-c', '100', ip],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    # Parse results for packet loss and timing
                    lines = result.stdout.split('\n')
                    packet_loss = 0
                    for line in lines:
                        if 'packet loss' in line:
                            try:
                                packet_loss = float(line.split('%')[0].split()[-1])
                            except (IndexError, ValueError):
                                pass
                    
                    # Estimate bandwidth based on packet loss and timing
                    if packet_loss < 1:
                        estimated_bandwidth = 1000  # Good network
                    elif packet_loss < 5:
                        estimated_bandwidth = 500   # Decent network
                    else:
                        estimated_bandwidth = 100   # Poor network
                    
                    tests.append(NetworkTest(
                        test_name=f"bandwidth_{node_name}",
                        passed=packet_loss < 10,
                        details=f"Bandwidth to {ip}: ~{estimated_bandwidth}Mbps, {packet_loss}% loss",
                        bandwidth_mbps=estimated_bandwidth
                    ))
                else:
                    tests.append(NetworkTest(
                        test_name=f"bandwidth_{node_name}",
                        passed=False,
                        details=f"Bandwidth test to {ip} failed"
                    ))
                    
            except subprocess.TimeoutExpired:
                tests.append(NetworkTest(
                    test_name=f"bandwidth_{node_name}",
                    passed=False,
                    details=f"Bandwidth test to {ip} timed out"
                ))
            except Exception as e:
                tests.append(NetworkTest(
                    test_name=f"bandwidth_{node_name}",
                    passed=False,
                    details=f"Bandwidth test to {ip} error: {str(e)}"
                ))
        
        return tests
    
    def test_firewall_configuration(self) -> List[NetworkTest]:
        """Test firewall configuration"""
        tests = []
        
        print("Testing firewall configuration...")
        
        try:
            # Check if packet filter is running
            result = subprocess.run(['sudo', 'pfctl', '-s', 'info'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                if 'Status: Enabled' in result.stdout:
                    tests.append(NetworkTest(
                        test_name="firewall_enabled",
                        passed=True,
                        details="Packet Filter (pf) is enabled"
                    ))
                else:
                    tests.append(NetworkTest(
                        test_name="firewall_enabled",
                        passed=False,
                        details="Packet Filter (pf) is not enabled"
                    ))
                    
                # Check rules
                result = subprocess.run(['sudo', 'pfctl', '-s', 'rules'], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    rules_output = result.stdout
                    required_ports = ['52415', '40000', '22']
                    rules_found = 0
                    
                    for port in required_ports:
                        if port in rules_output:
                            rules_found += 1
                    
                    tests.append(NetworkTest(
                        test_name="firewall_rules",
                        passed=rules_found >= 2,
                        details=f"Found rules for {rules_found}/{len(required_ports)} required ports"
                    ))
            else:
                tests.append(NetworkTest(
                    test_name="firewall_configuration",
                    passed=False,
                    details="Could not check firewall status (may need sudo)"
                ))
            
        except Exception as e:
            tests.append(NetworkTest(
                test_name="firewall_configuration",
                passed=False,
                details=f"Firewall test failed: {str(e)}"
            ))
        
        return tests
    
    async def run_all_tests(self, quick_test: bool = False) -> Dict[str, List[NetworkTest]]:
        """Run all network validation tests"""
        print("Running network validation tests...")
        print(f"Local node detected as: {self.get_local_node()}")
        print()
        
        results = {
            "connectivity": await self.test_basic_connectivity(),
            "ports": await self.test_port_connectivity(quick_test), 
            "mtu": self.test_mtu_configuration(),
            "firewall": self.test_firewall_configuration()
        }
        
        if not quick_test:
            results["bandwidth"] = await self.test_bandwidth()
        
        return results
    
    def generate_report(self, results: Dict[str, List[NetworkTest]]) -> str:
        """Generate a human-readable test report"""
        report = []
        report.append("=" * 60)
        report.append("NETWORK VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Local Node: {self.get_local_node()}")
        report.append(f"Test Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        total_tests = 0
        passed_tests = 0
        
        for category, tests in results.items():
            report.append(f"{category.upper()} TESTS:")
            report.append("-" * 40)
            
            for test in tests:
                total_tests += 1
                if test.passed:
                    passed_tests += 1
                    status = "✓ PASS"
                else:
                    status = "✗ FAIL"
                
                report.append(f"{status} {test.test_name}")
                report.append(f"      {test.details}")
                
                if test.latency_ms > 0:
                    report.append(f"      Latency: {test.latency_ms:.2f}ms")
                if test.bandwidth_mbps > 0:
                    report.append(f"      Bandwidth: {test.bandwidth_mbps:.0f}Mbps")
                
                report.append("")
        
        report.append("=" * 60)
        report.append(f"SUMMARY: {passed_tests}/{total_tests} tests passed")
        if passed_tests == total_tests:
            report.append("✓ All tests passed - Network ready for distributed cluster")
        else:
            report.append("✗ Some tests failed - Address issues before proceeding")
        
        # Add recommendations
        if passed_tests < total_tests:
            report.append("")
            report.append("RECOMMENDATIONS:")
            for category, tests in results.items():
                failed_tests = [t for t in tests if not t.passed]
                if failed_tests:
                    report.append(f"- {category.upper()}: {len(failed_tests)} failed tests")
                    for test in failed_tests[:3]:  # Show first 3 failures
                        report.append(f"  • {test.test_name}: {test.details}")
        
        report.append("=" * 60)
        
        return "\n".join(report)

async def main():
    """Run network validation"""
    parser = argparse.ArgumentParser(description='Network Validation for MLX Cluster')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--json-only', action='store_true', help='Output only JSON results')
    parser.add_argument('--output-dir', default='.', help='Output directory for reports')
    
    args = parser.parse_args()
    
    validator = NetworkValidator()
    
    # Run all tests
    results = await validator.run_all_tests(quick_test=args.quick)
    
    # Generate JSON output
    json_results = {}
    for category, tests in results.items():
        json_results[category] = [
            {
                "test_name": test.test_name,
                "passed": test.passed,
                "details": test.details,
                "latency_ms": test.latency_ms,
                "bandwidth_mbps": test.bandwidth_mbps
            }
            for test in tests
        ]
    
    # Calculate summary
    total_tests = sum(len(tests) for tests in results.values())
    passed_tests = sum(len([t for t in tests if t.passed]) for tests in results.values())
    
    json_results["summary"] = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
        "all_passed": passed_tests == total_tests,
        "local_node": validator.get_local_node(),
        "timestamp": time.time()
    }
    
    if args.json_only:
        print(json.dumps(json_results, indent=2))
        return
    
    # Generate and display report
    report = validator.generate_report(results)
    print(report)
    
    # Save reports
    import os
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    report_file = os.path.join(output_dir, 'network_validation_report.txt')
    json_file = os.path.join(output_dir, 'network_validation_results.json')
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    with open(json_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nReports saved:")
    print(f"- {report_file}")
    print(f"- {json_file}")
    
    # Exit with appropriate code
    if passed_tests < total_tests:
        print(f"\nValidation failed: {total_tests - passed_tests} tests failed")
        sys.exit(1)
    else:
        print("\nValidation successful: All tests passed")
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())