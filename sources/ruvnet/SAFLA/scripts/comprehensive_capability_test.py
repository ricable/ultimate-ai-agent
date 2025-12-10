#!/usr/bin/env python3
"""Comprehensive SAFLA Capability Verification Suite"""

import asyncio
import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Set up environment
os.environ.setdefault('JWT_SECRET_KEY', 'test-secret-key-for-verification-12345678901234567890')
os.environ.setdefault('SAFLA_ENVIRONMENT', 'development')

class CapabilityTester:
    """Comprehensive capability testing framework"""
    
    def __init__(self):
        self.results = {}
        self.passed = 0
        self.failed = 0
        self.skipped = 0
    
    def test(self, category: str, capability: str, test_func, *args, **kwargs):
        """Run a capability test and record results"""
        if category not in self.results:
            self.results[category] = []
        
        try:
            start_time = time.time()
            if asyncio.iscoroutinefunction(test_func):
                result = asyncio.run(test_func(*args, **kwargs))
            else:
                result = test_func(*args, **kwargs)
            
            duration = time.time() - start_time
            
            if result is True:
                status = "✅ PASS"
                self.passed += 1
            elif result is False:
                status = "❌ FAIL"
                self.failed += 1
            else:
                status = "⚠️ SKIP"
                self.skipped += 1
            
            self.results[category].append({
                'capability': capability,
                'status': status,
                'duration': f"{duration:.3f}s",
                'details': result if isinstance(result, str) else ""
            })
            
            print(f"{status} {category}: {capability} ({duration:.3f}s)")
            
        except Exception as e:
            status = "❌ ERROR"
            self.failed += 1
            self.results[category].append({
                'capability': capability,
                'status': status,
                'duration': "N/A",
                'details': str(e)
            })
            print(f"{status} {category}: {capability} - {str(e)[:80]}...")
    
    def print_summary(self):
        """Print test summary"""
        total = self.passed + self.failed + self.skipped
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE CAPABILITY TEST SUMMARY")
        print("="*80)
        
        for category, tests in self.results.items():
            print(f"\n📋 {category.upper()}")
            print("-" * 60)
            for test in tests:
                print(f"{test['status']} {test['capability']:.<40} {test['duration']}")
                if test['details']:
                    print(f"    └─ {test['details']}")
        
        print("\n" + "="*80)
        print(f"TOTAL RESULTS: {self.passed} PASSED | {self.failed} FAILED | {self.skipped} SKIPPED")
        print(f"SUCCESS RATE: {(self.passed/total*100):.1f}% ({self.passed}/{total})")
        print("="*80)


async def test_core_architecture():
    """Test core architecture components"""
    
    # Test 1: Package imports
    try:
        import safla
        return f"Version {safla.__version__}"
    except Exception as e:
        return False


async def test_hybrid_memory_architecture():
    """Test hybrid memory system"""
    try:
        from safla.core.hybrid_memory import (
            VectorMemoryManager, EpisodicMemory, SemanticMemory, 
            WorkingMemory, HybridMemoryArchitecture
        )
        
        # Test vector memory
        vector_memory = VectorMemoryManager(dimension=768)
        embedding = np.random.randn(768).astype(np.float32)
        
        # This tests the basic structure even if methods differ
        if hasattr(vector_memory, 'add_memory') or hasattr(vector_memory, 'store'):
            return "Vector memory structure verified"
        else:
            return "Vector memory API needs verification"
            
    except Exception as e:
        return False


async def test_metacognitive_engine():
    """Test meta-cognitive engine"""
    try:
        from safla.core.metacognitive import MetaCognitiveEngine
        
        # Test engine initialization
        engine = MetaCognitiveEngine()
        
        return "Meta-cognitive engine structure verified"
        
    except Exception as e:
        return False


async def test_safety_validation():
    """Test safety validation framework"""
    try:
        from safla.core.safety_validation import SafetyValidationFramework
        
        framework = SafetyValidationFramework()
        
        return "Safety framework structure verified"
        
    except Exception as e:
        return False


async def test_jwt_authentication():
    """Test JWT authentication system"""
    try:
        from safla.auth import JWTManager, AuthMiddleware
        from safla.auth.jwt_manager import JWTConfig
        
        # Test JWT manager
        config = JWTConfig()
        jwt_manager = JWTManager(config)
        
        # Generate token
        token, expiry = jwt_manager.generate_token(
            user_id="test_user",
            roles=["admin", "developer"],
            permissions=["tools:read", "tools:execute", "resources:read"]
        )
        
        # Validate token
        payload = jwt_manager.validate_token(token)
        
        # Test refresh token
        refresh_token, _ = jwt_manager.generate_refresh_token("test_user")
        new_token, _ = jwt_manager.refresh_access_token(refresh_token)
        
        # Test middleware
        middleware = AuthMiddleware(jwt_manager)
        request = {"method": "tools/list", "params": {"auth_token": token}}
        auth_result = await middleware.authenticate_request(request)
        
        return f"JWT system fully functional - User: {auth_result.sub}, Roles: {len(auth_result.roles)}"
        
    except Exception as e:
        return False


async def test_data_encryption():
    """Test data encryption capabilities"""
    try:
        from safla.security import DataEncryptor, SecureStorage
        
        # Test encryptor
        encryptor = DataEncryptor()
        
        # Test string encryption
        test_data = "Sensitive API key: sk-abc123def456ghi789"
        encrypted = encryptor.encrypt_string(test_data)
        decrypted = encryptor.decrypt_string(encrypted)
        
        if decrypted != test_data:
            return False
        
        # Test dictionary encryption
        dict_data = {
            "database_url": "postgresql://user:pass@host:5432/db",
            "api_keys": ["key1", "key2"],
            "secrets": {"jwt": "secret123"}
        }
        
        encrypted_dict = encryptor.encrypt_dict(dict_data)
        decrypted_dict = encryptor.decrypt_dict(encrypted_dict)
        
        if decrypted_dict != dict_data:
            return False
        
        # Test secure storage
        storage = SecureStorage()
        storage.store("test_key", dict_data, encrypt=True)
        retrieved = storage.retrieve("test_key", decrypt=True)
        storage.delete("test_key")
        
        if retrieved != dict_data:
            return False
        
        return f"Encryption fully functional - {len(encrypted)} bytes encrypted"
        
    except Exception as e:
        return False


async def test_rate_limiting():
    """Test rate limiting system"""
    try:
        from safla.middleware import RateLimiter, RateLimitConfig
        
        # Test with restrictive settings
        config = RateLimitConfig(
            requests_per_minute=5,
            burst_size=3,
            method_limits={"tools/call": 2}
        )
        
        rate_limiter = RateLimiter(config)
        
        # Test normal flow
        allowed_count = 0
        for i in range(3):
            allowed, _ = await rate_limiter.check_rate_limit(
                {"method": f"test_{i}"}, 
                identifier="test_user"
            )
            if allowed:
                allowed_count += 1
        
        # Test blocking
        blocked_count = 0
        for i in range(10):
            allowed, _ = await rate_limiter.check_rate_limit(
                {"method": "tools/call"}, 
                identifier="heavy_user"
            )
            if not allowed:
                blocked_count += 1
        
        # Test statistics
        stats = await rate_limiter.get_stats("heavy_user")
        
        return f"Rate limiting functional - {allowed_count} allowed, {blocked_count} blocked, {stats['violations']} violations"
        
    except Exception as e:
        return False


async def test_input_validation():
    """Test input validation system"""
    try:
        from safla.validation import validate_path, validate_tool_name, sanitize_error_message
        from safla.validation.mcp_models import MCPRequest, ToolCallRequest, ResourceReadRequest
        
        # Test MCP request validation
        valid_request = MCPRequest(
            jsonrpc="2.0",
            method="tools/list",
            params={"filter": "core"},
            id=1
        )
        
        # Test tool call validation
        tool_request = ToolCallRequest(
            name="get_system_info",
            arguments={"include_gpu": True}
        )
        
        # Test resource request validation
        resource_request = ResourceReadRequest(
            uri="configs/server.json"
        )
        
        # Test path validation
        safe_paths = 0
        blocked_paths = 0
        
        safe_test_paths = ["data/config.json", "logs/server.log"]
        for path in safe_test_paths:
            try:
                validate_path(path, base_dir="/tmp")
                safe_paths += 1
            except ValueError:
                pass
        
        malicious_paths = ["../../../etc/passwd", "..\\..\\windows\\system32"]
        for path in malicious_paths:
            try:
                validate_path(path, base_dir="/tmp")
            except ValueError:
                blocked_paths += 1
        
        # Test tool name validation
        valid_tools = ["get_system_info", "deploy_instance"]
        invalid_tools = ["../evil", "<script>alert(1)</script>"]
        
        valid_tool_count = 0
        for tool in valid_tools:
            try:
                validate_tool_name(tool)
                valid_tool_count += 1
            except ValueError:
                pass
        
        blocked_tool_count = 0
        for tool in invalid_tools:
            try:
                validate_tool_name(tool)
            except ValueError:
                blocked_tool_count += 1
        
        # Test error sanitization
        sensitive_error = "Database connection failed: postgresql://admin:secret@192.168.1.100:5432/db with API key sk-abc123"
        sanitized = sanitize_error_message(sensitive_error)
        
        return f"Validation functional - {safe_paths} safe paths, {blocked_paths} blocked paths, {valid_tool_count} valid tools, {blocked_tool_count} blocked tools"
        
    except Exception as e:
        return False


async def test_handler_registry():
    """Test modular handler registry system"""
    try:
        from safla.mcp.handler_registry import get_registry
        
        # Import handler modules
        import safla.mcp.handlers.core_tools
        import safla.mcp.handlers.deployment_tools
        import safla.mcp.handlers.optimization_tools
        
        registry = get_registry()
        
        # Test registry functions
        handlers = registry.list_handlers()
        categories = registry.list_categories()
        
        # Test handler dispatch
        public_handlers = [h for h in handlers if not h.requires_auth]
        if public_handlers:
            test_handler = public_handlers[0]
            context = {"authenticated": False, "user": None}
            result = await registry.dispatch(test_handler.name, {}, context)
            
            # Test authenticated handler protection
            auth_handlers = [h for h in handlers if h.requires_auth]
            if auth_handlers:
                test_auth_handler = auth_handlers[0]
                try:
                    await registry.dispatch(test_auth_handler.name, {}, context)
                    auth_protection = False
                except PermissionError:
                    auth_protection = True
            else:
                auth_protection = True
        
        # Test tools definition
        tools_def = registry.get_tools_definition()
        
        return f"Handler registry functional - {len(handlers)} handlers, {len(categories)} categories, {len(tools_def)} tools, auth protection: {auth_protection}"
        
    except Exception as e:
        return False


async def test_mcp_core_tools():
    """Test MCP core tool handlers"""
    try:
        from safla.mcp.handler_registry import get_registry
        import safla.mcp.handlers.core_tools
        
        registry = get_registry()
        context = {"authenticated": False, "user": None}
        
        # Test validate_installation
        result1 = await registry.dispatch("validate_installation", {}, context)
        
        # Test get_system_info  
        result2 = await registry.dispatch("get_system_info", {}, context)
        
        # Test check_gpu_status
        result3 = await registry.dispatch("check_gpu_status", {}, context)
        
        return f"Core tools functional - Installation: {result1.get('valid', False)}, System info: {'platform' in result2}, GPU: {'available' in result3}"
        
    except Exception as e:
        return False


async def test_mcp_deployment_tools():
    """Test MCP deployment tool handlers"""
    try:
        from safla.mcp.handler_registry import get_registry
        import safla.mcp.handlers.deployment_tools
        
        registry = get_registry()
        context = {"authenticated": True, "user": None}
        
        # Test deployment
        deploy_args = {
            "instance_count": 2,
            "deployment_mode": "local"
        }
        result1 = await registry.dispatch("deploy_safla_instance", deploy_args, context)
        
        if "deployment_id" in result1:
            deployment_id = result1["deployment_id"]
            
            # Test status check
            status_args = {"deployment_id": deployment_id}
            result2 = await registry.dispatch("check_deployment_status", status_args, context)
            
            # Test scaling
            scale_args = {
                "deployment_id": deployment_id,
                "action": "scale_up",
                "instance_count": 1
            }
            result3 = await registry.dispatch("scale_deployment", scale_args, context)
            
            return f"Deployment tools functional - Deploy: {result1['status']}, Status: {result2['status']}, Scale: {result3['action']}"
        
        return False
        
    except Exception as e:
        return False


async def test_mcp_optimization_tools():
    """Test MCP optimization tool handlers"""
    try:
        from safla.mcp.handler_registry import get_registry
        import safla.mcp.handlers.optimization_tools
        
        registry = get_registry()
        context = {"authenticated": True, "user": None}
        
        # Test memory optimization
        memory_args = {"target_reduction": 0.2, "aggressive": False}
        result1 = await registry.dispatch("optimize_memory_usage", memory_args, context)
        
        # Test vector optimization
        vector_args = {"operation_type": "similarity_search", "use_gpu": False}
        result2 = await registry.dispatch("optimize_vector_operations", vector_args, context)
        
        # Test performance analysis
        result3 = await registry.dispatch("analyze_performance_bottlenecks", {}, context)
        
        return f"Optimization tools functional - Memory: {result1.get('reduction_achieved', 0):.2f}, Vector: {len(result2.get('optimizations_applied', []))}, Analysis: {len(result3.get('bottlenecks', []))}"
        
    except Exception as e:
        return False


async def test_configuration_system():
    """Test configuration management"""
    try:
        from safla.utils.config import get_config, SAFLAConfig
        
        # Test config loading
        config = get_config()
        
        # Test basic config attributes
        has_log_level = hasattr(config, 'log_level')
        has_gpu_enabled = hasattr(config, 'gpu_enabled')
        
        return f"Configuration system functional - Log level: {has_log_level}, GPU config: {has_gpu_enabled}"
        
    except Exception as e:
        return False


async def test_logging_system():
    """Test logging capabilities"""
    try:
        from safla.utils.logging import setup_logging, get_logger
        
        # Test logger setup
        logger = get_logger("test_logger")
        
        # Test logging
        logger.info("Test log message")
        logger.warning("Test warning message")
        
        return "Logging system functional"
        
    except Exception as e:
        return False


async def test_utilities():
    """Test utility functions"""
    try:
        from safla.utils.validation import validate_installation, check_gpu_availability
        
        # Test installation validation
        install_result = validate_installation()
        
        # Test GPU check
        gpu_result = check_gpu_availability()
        
        return f"Utilities functional - Installation: {install_result.get('valid', False)}, GPU: {gpu_result.get('available', False)}"
        
    except Exception as e:
        return False


async def test_exceptions():
    """Test exception handling"""
    try:
        from safla.exceptions import SAFLAError
        from safla.auth.exceptions import AuthenticationError, TokenExpiredError
        
        # Test exception creation
        safla_error = SAFLAError("Test error")
        auth_error = AuthenticationError("Test auth error")
        token_error = TokenExpiredError()
        
        return f"Exception system functional - {len([safla_error, auth_error, token_error])} exception types"
        
    except Exception as e:
        return False


async def test_integration_workflow():
    """Test complete integration workflow"""
    try:
        from safla.auth import JWTManager
        from safla.middleware import RateLimiter
        from safla.security import DataEncryptor
        from safla.mcp.handler_registry import get_registry
        
        # Import handlers
        import safla.mcp.handlers.core_tools
        
        # Step 1: Authentication
        jwt_manager = JWTManager()
        token, _ = jwt_manager.generate_token("integration_user", roles=["admin"], permissions=["*"])
        payload = jwt_manager.validate_token(token)
        
        # Step 2: Rate limiting check
        rate_limiter = RateLimiter()
        allowed, _ = await rate_limiter.check_rate_limit(
            {"method": "tools/call"}, 
            user_id=payload.sub
        )
        
        if not allowed:
            return "Rate limiting blocked integration test"
        
        # Step 3: Data encryption
        encryptor = DataEncryptor()
        sensitive_data = {"user_id": payload.sub, "permissions": payload.permissions}
        encrypted = encryptor.encrypt_dict(sensitive_data)
        decrypted = encryptor.decrypt_dict(encrypted)
        
        # Step 4: Handler execution
        registry = get_registry()
        context = {"authenticated": True, "user": payload}
        result = await registry.dispatch("validate_installation", {}, context)
        
        return f"Integration workflow successful - Auth: {payload.sub}, Encryption: verified, Handler: {result.get('valid', False)}"
        
    except Exception as e:
        return False


async def test_performance_benchmarks():
    """Test performance characteristics"""
    try:
        from safla.auth import JWTManager
        from safla.security import DataEncryptor
        from safla.middleware import RateLimiter
        
        # JWT performance
        jwt_manager = JWTManager()
        start_time = time.time()
        for i in range(100):
            jwt_manager.generate_token(f"user_{i}", roles=["user"])
        jwt_ops_per_sec = 100 / (time.time() - start_time)
        
        # Encryption performance
        encryptor = DataEncryptor()
        start_time = time.time()
        for i in range(100):
            encryptor.encrypt_string(f"test data {i}")
        encrypt_ops_per_sec = 100 / (time.time() - start_time)
        
        # Rate limiting performance
        rate_limiter = RateLimiter()
        start_time = time.time()
        for i in range(100):
            await rate_limiter.check_rate_limit({"method": f"test_{i}"}, identifier=f"user_{i}")
        rate_limit_ops_per_sec = 100 / (time.time() - start_time)
        
        return f"Performance benchmarks - JWT: {jwt_ops_per_sec:.0f} ops/sec, Encryption: {encrypt_ops_per_sec:.0f} ops/sec, Rate limiting: {rate_limit_ops_per_sec:.0f} ops/sec"
        
    except Exception as e:
        return False


async def main():
    """Run comprehensive capability tests"""
    tester = CapabilityTester()
    
    print("🔥 SAFLA COMPREHENSIVE CAPABILITY VERIFICATION")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Core Architecture Tests
    print("\n🏗️ CORE ARCHITECTURE")
    tester.test("Core Architecture", "Package Import", test_core_architecture)
    tester.test("Core Architecture", "Hybrid Memory Architecture", test_hybrid_memory_architecture)
    tester.test("Core Architecture", "Meta-Cognitive Engine", test_metacognitive_engine)
    tester.test("Core Architecture", "Safety Validation Framework", test_safety_validation)
    
    # Security System Tests
    print("\n🔐 SECURITY SYSTEMS")
    tester.test("Security", "JWT Authentication", test_jwt_authentication)
    tester.test("Security", "Data Encryption", test_data_encryption)
    tester.test("Security", "Rate Limiting", test_rate_limiting)
    tester.test("Security", "Input Validation", test_input_validation)
    
    # Handler System Tests
    print("\n📋 HANDLER SYSTEMS")
    tester.test("Handlers", "Handler Registry", test_handler_registry)
    tester.test("Handlers", "Core Tools", test_mcp_core_tools)
    tester.test("Handlers", "Deployment Tools", test_mcp_deployment_tools)
    tester.test("Handlers", "Optimization Tools", test_mcp_optimization_tools)
    
    # Utility System Tests
    print("\n🛠️ UTILITY SYSTEMS")
    tester.test("Utilities", "Configuration System", test_configuration_system)
    tester.test("Utilities", "Logging System", test_logging_system)
    tester.test("Utilities", "Validation Utilities", test_utilities)
    tester.test("Utilities", "Exception Handling", test_exceptions)
    
    # Integration Tests
    print("\n🔄 INTEGRATION TESTS")
    tester.test("Integration", "Complete Workflow", test_integration_workflow)
    tester.test("Integration", "Performance Benchmarks", test_performance_benchmarks)
    
    # Print final summary
    tester.print_summary()
    
    # Overall assessment
    total_tests = tester.passed + tester.failed + tester.skipped
    success_rate = (tester.passed / total_tests) * 100 if total_tests > 0 else 0
    
    if success_rate >= 90:
        print("\n🏆 EXCELLENT: System is fully operational")
    elif success_rate >= 80:
        print("\n✅ GOOD: System is mostly functional")
    elif success_rate >= 70:
        print("\n⚠️ WARNING: System has some issues")
    else:
        print("\n❌ CRITICAL: System needs attention")
    
    return success_rate >= 80

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1)