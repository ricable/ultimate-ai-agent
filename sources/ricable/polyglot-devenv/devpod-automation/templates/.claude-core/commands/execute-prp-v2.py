#!/usr/bin/env python3
"""
Enhanced PRP Execution Command with Version Control and Scalability
Uses the integrated PRP system with Memento/Observer and Mediator/Factory patterns.

Addresses Issues #6 and #8: Version Control and Scalability
"""

import sys
import asyncio
import argparse
from pathlib import Path

# Add the context-engineering lib to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "context-engineering" / "lib"))

from integrated_prp_system import IntegratedPRPSystem


async def main():
    """Enhanced PRP execution with version control and rollback capability."""
    parser = argparse.ArgumentParser(description="Execute PRP with version control and scalability")
    parser.add_argument("prp_file", help="Path to PRP file to execute")
    parser.add_argument("--validate", action="store_true", help="Run validation after execution")
    parser.add_argument("--monitor", action="store_true", help="Enable performance monitoring")
    parser.add_argument("--dry-run", action="store_true", help="Show execution plan without running")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--no-rollback", action="store_true", help="Disable automatic rollback")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--timeout", type=int, default=300, help="Execution timeout in seconds")
    
    args = parser.parse_args()
    
    # Initialize integrated system
    system = IntegratedPRPSystem(max_workers=args.workers)
    await system.initialize()
    
    try:
        # Validate PRP file exists
        prp_path = Path(args.prp_file)
        if not prp_path.exists():
            print(f"❌ PRP file not found: {args.prp_file}")
            return 1
        
        # Extract environment from PRP file or path
        environment = "python-env"  # Default
        if "python" in prp_path.name:
            environment = "python-env"
        elif "typescript" in prp_path.name:
            environment = "typescript-env"
        elif "rust" in prp_path.name:
            environment = "rust-env"
        elif "go" in prp_path.name:
            environment = "go-env"
        elif "nushell" in prp_path.name:
            environment = "nushell-env"
        
        # Try to extract environment from PRP content
        prp_content = prp_path.read_text()
        if "Environment:" in prp_content:
            for line in prp_content.split('\n'):
                if line.strip().startswith("Environment:"):
                    env_line = line.split(":", 1)[1].strip()
                    if env_line:
                        environment = env_line
                        break
        
        execution_options = {
            "validate": args.validate,
            "monitor": args.monitor,
            "dry_run": args.dry_run,
            "timeout": args.timeout
        }
        
        print(f"🚀 Executing PRP: {prp_path.name}")
        print(f"   Environment: {environment}")
        print(f"   Options: {execution_options}")
        print(f"   Auto-rollback: {'enabled' if not args.no_rollback else 'disabled'}")
        
        if args.dry_run:
            print(f"   📋 DRY RUN - Would execute PRP with above settings")
            return 0
        
        # Show execution history
        prp_name = prp_path.stem
        history = system.get_execution_history(prp_name, limit=3)
        if history:
            print(f"   📚 Previous executions: {len(history)}")
            for i, exec_record in enumerate(history[:2]):
                print(f"     {i+1}. {exec_record['status']} ({exec_record['timestamp'][:19]})")
        
        # Execute PRP with integrated system
        result = await system.execute_prp_with_rollback(
            str(prp_path),
            environment,
            execution_options,
            auto_rollback=not args.no_rollback
        )
        
        if result and result.success:
            print(f"✅ PRP execution completed successfully")
            exec_result = result.result
            print(f"   Status: {exec_result.get('status', 'unknown')}")
            print(f"   Tasks executed: {exec_result.get('tasks_executed', 0)}")
            print(f"   Execution time: {result.execution_time:.2f}s")
            
            if exec_result.get('validation_passed'):
                print(f"   ✅ Validation: PASSED")
            else:
                print(f"   ⚠️  Validation: Check required")
            
            # Run additional validation if requested
            if args.validate:
                print(f"\n🔍 Running additional validation...")
                validation_gates = [
                    "devbox run lint",
                    "devbox run test",
                    "devbox run format"
                ]
                
                validation_result = await system.validate_with_history(
                    environment,
                    validation_gates,
                    compare_with_previous=True
                )
                
                if validation_result and validation_result.success:
                    val_data = validation_result.result
                    print(f"   Overall validation: {val_data.get('overall_status', 'unknown')}")
                    print(f"   Gates executed: {val_data.get('total_gates', 0)}")
                else:
                    print(f"   ❌ Validation failed")
            
            # Show system status
            if args.debug:
                status = system.get_system_status()
                print(f"\n📊 System Status:")
                print(f"   Session: {status['session_id']}")
                print(f"   Tasks completed: {status['performance_metrics']['tasks_completed']}")
                print(f"   Total execution time: {status['performance_metrics']['total_execution_time']:.2f}s")
                print(f"   Version saves: {status['performance_metrics']['version_saves']}")
                print(f"   Version restores: {status['performance_metrics']['version_restores']}")
        
        else:
            print(f"❌ PRP execution failed")
            if result and result.error:
                print(f"   Error: {result.error}")
            
            # Show rollback information if it occurred
            if not args.no_rollback:
                print(f"   🔄 Automatic rollback was attempted")
            
            return 1
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    finally:
        system.shutdown()
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)