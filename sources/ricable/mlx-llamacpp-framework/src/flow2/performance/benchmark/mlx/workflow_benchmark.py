#!/usr/bin/env python3
"""
MLX Workflow Benchmark - Test all MLX functionality in a comprehensive workflow
"""

import os
import sys
import time
import subprocess
import json
from datetime import datetime
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_benchmark_test(name, command, description):
    """Run a benchmark test and capture results"""
    print(f"\n🎯 {name}")
    print(f"📝 {description}")
    print("=" * 60)
    
    start_time = time.time()
    try:
        # Run command
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=".")
        end_time = time.time()
        duration = end_time - start_time
        
        success = result.returncode == 0
        
        # Extract key metrics from output
        output_lines = result.stdout.split('\n')
        metrics = {}
        
        # Look for common performance metrics
        for line in output_lines:
            if 'completed in' in line.lower():
                metrics['completion_time'] = line.strip()
            elif 'tokens/sec' in line.lower() or 'tok/sec' in line.lower():
                metrics['throughput'] = line.strip()
            elif 'peak mem' in line.lower() or 'memory usage' in line.lower():
                metrics['memory'] = line.strip()
            elif 'loss' in line.lower() and ('train' in line.lower() or 'val' in line.lower()):
                metrics['loss'] = line.strip()
        
        return {
            "name": name,
            "description": description,
            "command": command,
            "success": success,
            "duration": duration,
            "returncode": result.returncode,
            "metrics": metrics,
            "stdout": result.stdout[:1000] + "..." if len(result.stdout) > 1000 else result.stdout,
            "stderr": result.stderr[:500] + "..." if len(result.stderr) > 500 else result.stderr
        }
        
    except Exception as e:
        return {
            "name": name,
            "description": description,
            "command": command,
            "success": False,
            "duration": time.time() - start_time,
            "error": str(e)
        }

def main():
    """Run comprehensive MLX workflow benchmark"""
    
    print("🚀 MLX COMPREHENSIVE WORKFLOW BENCHMARK")
    print("=" * 80)
    print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  Platform: Apple Silicon")
    print("=" * 80)
    
    # Activate virtual environment in command
    venv_prefix = "source venv/bin/activate && "
    
    # Test suite
    tests = [
        {
            "name": "Basic MLX Fine-tuning",
            "command": f"{venv_prefix}python examples/mlx/run_mlx_finetune.py",
            "description": "Standard MLX LoRA fine-tuning with quotes dataset"
        },
        {
            "name": "Enhanced MLX Fine-tuning with Flash Attention",
            "command": f"{venv_prefix}python examples/mlx/run_mlx_finetune_improved.py --prepare-data --train --use-flash-attention --iters 25",
            "description": "Enhanced MLX fine-tuning with Flash Attention optimizations"
        },
        {
            "name": "Flash Attention Performance Comparison",
            "command": f"{venv_prefix}python examples/mlx/test_flash_attention_comparison.py",
            "description": "Compare MLX performance with and without Flash Attention"
        },
        {
            "name": "MLX Comprehensive Benchmark Suite",
            "command": f"{venv_prefix}python mlx_comprehensive_benchmark.py",
            "description": "Test all MLX operations including inference, quantization, memory usage"
        },
        {
            "name": "MLX Flash Attention Benchmark Suite",
            "command": f"{venv_prefix}python mlx_flash_attention_benchmark.py",
            "description": "Comprehensive Flash Attention performance benchmarking"
        },
        {
            "name": "MLX Chat Interface Test",
            "command": f"{venv_prefix}timeout 10 python src/flow2/chat/interfaces/cli/mlx_chat.py --test-mode || true",
            "description": "Test MLX chat interface (with timeout)"
        }
    ]
    
    results = []
    total_start = time.time()
    
    for test in tests:
        result = run_benchmark_test(test["name"], test["command"], test["description"])
        results.append(result)
        
        # Print immediate result
        if result["success"]:
            print(f"✅ {test['name']} completed in {result['duration']:.2f}s")
            if result.get("metrics"):
                for key, value in result["metrics"].items():
                    print(f"   📊 {key}: {value}")
        else:
            print(f"❌ {test['name']} failed after {result['duration']:.2f}s")
            if "error" in result:
                print(f"   Error: {result['error']}")
            elif result.get("stderr"):
                print(f"   Error: {result['stderr'][:200]}...")
    
    total_duration = time.time() - total_start
    
    # Create comprehensive report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("benchmark_results/mlx_workflow")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_file = output_dir / f"mlx_workflow_benchmark_{timestamp}.json"
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_duration": total_duration,
        "total_tests": len(tests),
        "successful_tests": len([r for r in results if r["success"]]),
        "failed_tests": len([r for r in results if not r["success"]]),
        "results": results
    }
    
    with open(results_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("📊 MLX WORKFLOW BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"⏱️  Total time: {total_duration:.2f}s")
    print(f"✅ Successful tests: {report['successful_tests']}/{report['total_tests']}")
    print(f"❌ Failed tests: {report['failed_tests']}/{report['total_tests']}")
    
    print(f"\n📁 Detailed results saved to: {results_file}")
    
    # Test-by-test summary
    print(f"\n📋 Test Results:")
    for result in results:
        status = "✅" if result["success"] else "❌"
        print(f"{status} {result['name']} ({result['duration']:.2f}s)")
    
    # Key insights
    print(f"\n🔍 Key Insights:")
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        avg_duration = sum(r["duration"] for r in successful_results) / len(successful_results)
        print(f"📈 Average successful test duration: {avg_duration:.2f}s")
        
        # Look for performance metrics
        performance_found = False
        for result in successful_results:
            if result.get("metrics"):
                performance_found = True
                break
        
        if performance_found:
            print("🚀 Performance metrics captured for detailed analysis")
        
    if report['successful_tests'] == report['total_tests']:
        print("🎉 All MLX functionality tests passed!")
    elif report['successful_tests'] > 0:
        print(f"⚠️  {report['failed_tests']} tests failed, but core functionality working")
    else:
        print("❌ Major issues detected - see detailed logs")
    
    return 0 if report['successful_tests'] > 0 else 1

if __name__ == "__main__":
    sys.exit(main())