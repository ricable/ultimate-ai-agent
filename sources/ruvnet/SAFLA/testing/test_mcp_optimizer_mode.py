#!/usr/bin/env python3
"""
Test suite for MCP Optimizer mode implementation
Tests the complete workflow and integration patterns
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

def test_file_structure():
    """Test that all required files exist with proper structure"""
    print("🔍 Testing MCP Optimizer file structure...")
    
    base_path = Path(".roo/mcp-modes/rules-mcp-optimizer")
    required_files = [
        "rules.md",
        "examples.md", 
        "integration_guide.md",
        "README.md"
    ]
    
    results = []
    for file_name in required_files:
        file_path = base_path / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            results.append(f"✅ {file_name}: {size:,} bytes")
        else:
            results.append(f"❌ {file_name}: Missing")
    
    return results

def test_modes_json_registration():
    """Test that MCP Optimizer is properly registered in modes.json"""
    print("🔍 Testing modes.json registration...")
    
    modes_path = Path(".roo/mcp-modes/modes.json")
    if not modes_path.exists():
        return ["❌ modes.json not found"]
    
    try:
        with open(modes_path, 'r') as f:
            modes_config = json.load(f)
        
        results = []
        
        # Check if mcp-optimizer exists
        if "mcp-optimizer" in modes_config.get("mcpOptimizedModes", {}):
            optimizer_config = modes_config["mcpOptimizedModes"]["mcp-optimizer"]
            results.append("✅ MCP Optimizer mode registered")
            
            # Validate configuration structure
            required_fields = ["name", "slug", "description", "capabilities", "requiredMcpServers"]
            for field in required_fields:
                if field in optimizer_config:
                    results.append(f"✅ {field}: {optimizer_config[field]}")
                else:
                    results.append(f"❌ Missing field: {field}")
                    
            # Check capabilities
            expected_capabilities = [
                "performance_profiling",
                "benchmarking", 
                "resource_optimization",
                "bottleneck_identification"
            ]
            
            actual_capabilities = optimizer_config.get("capabilities", [])
            for cap in expected_capabilities:
                if cap in actual_capabilities:
                    results.append(f"✅ Capability: {cap}")
                else:
                    results.append(f"❌ Missing capability: {cap}")
                    
        else:
            results.append("❌ MCP Optimizer mode not found in modes.json")
            
        return results
        
    except json.JSONDecodeError as e:
        return [f"❌ JSON parsing error: {e}"]
    except Exception as e:
        return [f"❌ Error reading modes.json: {e}"]

def test_mcp_server_integration():
    """Test MCP server configuration for optimizer requirements"""
    print("🔍 Testing MCP server integration...")
    
    mcp_path = Path(".roo/mcp.json")
    if not mcp_path.exists():
        return ["❌ mcp.json not found"]
    
    try:
        with open(mcp_path, 'r') as f:
            mcp_config = json.load(f)
        
        results = []
        
        # Check required servers
        required_servers = ["safla"]
        optional_servers = ["context7", "perplexity"]
        
        mcp_servers = mcp_config.get("mcpServers", {})
        
        for server in required_servers:
            if server in mcp_servers:
                results.append(f"✅ Required server: {server}")
                
                # Check specific tools for optimization
                if server == "safla":
                    optimization_tools = [
                        "analyze_performance_bottlenecks",
                        "optimize_memory_usage",
                        "optimize_vector_operations",
                        "benchmark_vector_operations",
                        "benchmark_memory_performance"
                    ]
                    
                    allowed_tools = mcp_servers[server].get("alwaysAllow", [])
                    for tool in optimization_tools:
                        if tool in allowed_tools:
                            results.append(f"✅ Optimization tool: {tool}")
                        else:
                            results.append(f"❌ Missing tool: {tool}")
            else:
                results.append(f"❌ Missing required server: {server}")
        
        for server in optional_servers:
            if server in mcp_servers:
                results.append(f"✅ Optional server: {server}")
            else:
                results.append(f"⚠️  Optional server not configured: {server}")
                
        return results
        
    except json.JSONDecodeError as e:
        return [f"❌ JSON parsing error: {e}"]
    except Exception as e:
        return [f"❌ Error reading mcp.json: {e}"]

def test_workflow_templates():
    """Test workflow template integration"""
    print("🔍 Testing workflow templates...")
    
    modes_path = Path(".roo/mcp-modes/modes.json")
    if not modes_path.exists():
        return ["❌ modes.json not found"]
    
    try:
        with open(modes_path, 'r') as f:
            modes_config = json.load(f)
        
        results = []
        
        workflow_templates = modes_config.get("workflowTemplates", {})
        
        # Check for optimization-related workflows
        optimization_workflows = [
            "analyze_and_optimize",
            "research_and_implement"
        ]
        
        for workflow in optimization_workflows:
            if workflow in workflow_templates:
                template = workflow_templates[workflow]
                results.append(f"✅ Workflow template: {workflow}")
                
                # Check if mcp-optimizer is used in the workflow
                steps = template.get("steps", [])
                optimizer_used = any(step.get("mode") == "mcp-optimizer" for step in steps)
                
                if optimizer_used:
                    results.append(f"✅ MCP Optimizer used in {workflow}")
                else:
                    results.append(f"⚠️  MCP Optimizer not used in {workflow}")
            else:
                results.append(f"❌ Missing workflow template: {workflow}")
                
        return results
        
    except Exception as e:
        return [f"❌ Error testing workflow templates: {e}"]

def test_documentation_content():
    """Test documentation content quality and completeness"""
    print("🔍 Testing documentation content...")
    
    results = []
    base_path = Path(".roo/mcp-modes/rules-mcp-optimizer")
    
    # Test rules.md content
    rules_path = base_path / "rules.md"
    if rules_path.exists():
        with open(rules_path, 'r') as f:
            rules_content = f.read()
        
        # Check for key sections
        key_sections = [
            "# MCP Optimizer Mode",
            "## Overview",
            "## Workflow Phases",
            "## MCP Integration Patterns",
            "## Optimization Methodologies",
            "## Performance Measurement Framework",
            "## Technology-Specific Optimizations"
        ]
        
        for section in key_sections:
            if section in rules_content:
                results.append(f"✅ Rules section: {section}")
            else:
                results.append(f"❌ Missing rules section: {section}")
                
        # Check for code examples
        if "```typescript" in rules_content or "```javascript" in rules_content:
            results.append("✅ TypeScript/JavaScript examples present")
        else:
            results.append("❌ Missing TypeScript/JavaScript examples")
            
        if "```sql" in rules_content:
            results.append("✅ SQL examples present")
        else:
            results.append("❌ Missing SQL examples")
    else:
        results.append("❌ rules.md not found")
    
    # Test examples.md content
    examples_path = base_path / "examples.md"
    if examples_path.exists():
        with open(examples_path, 'r') as f:
            examples_content = f.read()
        
        # Check for performance metrics
        if "improvement" in examples_content.lower() and "%" in examples_content:
            results.append("✅ Performance improvement metrics present")
        else:
            results.append("❌ Missing performance improvement metrics")
            
        # Check for before/after examples
        if "before:" in examples_content.lower() and "after:" in examples_content.lower():
            results.append("✅ Before/after examples present")
        else:
            results.append("❌ Missing before/after examples")
    else:
        results.append("❌ examples.md not found")
    
    return results

def test_integration_patterns():
    """Test integration guide patterns and completeness"""
    print("🔍 Testing integration patterns...")
    
    results = []
    integration_path = Path(".roo/mcp-modes/rules-mcp-optimizer/integration_guide.md")
    
    if integration_path.exists():
        with open(integration_path, 'r') as f:
            integration_content = f.read()
        
        # Check for key integration patterns
        integration_patterns = [
            "MCP Tool Integration",
            "External Tool Integration", 
            "Workflow Integration",
            "Error Handling",
            "Monitoring Integration"
        ]
        
        for pattern in integration_patterns:
            if pattern in integration_content:
                results.append(f"✅ Integration pattern: {pattern}")
            else:
                results.append(f"❌ Missing integration pattern: {pattern}")
                
        # Check for code examples
        if "use_mcp_tool" in integration_content:
            results.append("✅ MCP tool usage examples present")
        else:
            results.append("❌ Missing MCP tool usage examples")
            
        # Check for workflow examples
        if "new_task" in integration_content:
            results.append("✅ Workflow integration examples present")
        else:
            results.append("❌ Missing workflow integration examples")
    else:
        results.append("❌ integration_guide.md not found")
    
    return results

def run_comprehensive_test():
    """Run all tests and generate comprehensive report"""
    print("🚀 Starting MCP Optimizer Mode Comprehensive Test Suite")
    print("=" * 60)
    
    test_functions = [
        ("File Structure", test_file_structure),
        ("Modes.json Registration", test_modes_json_registration),
        ("MCP Server Integration", test_mcp_server_integration),
        ("Workflow Templates", test_workflow_templates),
        ("Documentation Content", test_documentation_content),
        ("Integration Patterns", test_integration_patterns)
    ]
    
    all_results = {}
    total_tests = 0
    passed_tests = 0
    
    for test_name, test_func in test_functions:
        print(f"\n{test_name}")
        print("-" * 40)
        
        try:
            results = test_func()
            all_results[test_name] = results
            
            for result in results:
                print(result)
                total_tests += 1
                if result.startswith("✅"):
                    passed_tests += 1
                    
        except Exception as e:
            error_msg = f"❌ Test failed with error: {e}"
            print(error_msg)
            all_results[test_name] = [error_msg]
            total_tests += 1
    
    # Generate summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎉 EXCELLENT: MCP Optimizer mode is well implemented!")
    elif success_rate >= 75:
        print("✅ GOOD: MCP Optimizer mode is properly implemented with minor issues")
    elif success_rate >= 50:
        print("⚠️  FAIR: MCP Optimizer mode has some implementation issues")
    else:
        print("❌ POOR: MCP Optimizer mode needs significant improvements")
    
    # Generate detailed report
    print("\n" + "=" * 60)
    print("📋 DETAILED REPORT")
    print("=" * 60)
    
    for test_name, results in all_results.items():
        print(f"\n{test_name}:")
        for result in results:
            print(f"  {result}")
    
    return success_rate >= 75

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)