#!/usr/bin/env python3
"""Comprehensive SAFLA System Verification Script"""

import asyncio
import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Set up environment
os.environ.setdefault('SAFLA_ENVIRONMENT', 'development')
os.environ.setdefault('JWT_SECRET_KEY', 'test-secret-key-for-verification')
os.environ.setdefault('SAFLA_ENCRYPTION_KEY', 'test-encryption-key')

from safla.core.hybrid_memory import HybridMemoryArchitecture as HybridMemory
from safla.core.meta_cognitive_engine import MetaCognitiveEngine
from safla.core.safety_validation import SafetyValidationFramework
from safla.auth import JWTManager, AuthMiddleware
from safla.security import DataEncryptor, SecureStorage
from safla.middleware import RateLimiter
from safla.validation import validate_path, validate_tool_name
from safla.mcp.handler_registry import get_registry
from safla.utils.config import get_config


class SystemVerifier:
    """Comprehensive system verification"""
    
    def __init__(self):
        self.results = {}
        self.passed = 0
        self.failed = 0
    
    def report(self, component: str, test: str, success: bool, details: str = ""):
        """Report test result"""
        if component not in self.results:
            self.results[component] = []
        
        self.results[component].append({
            'test': test,
            'success': success,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
        
        if success:
            self.passed += 1
            print(f"✅ {component}: {test}")
        else:
            self.failed += 1
            print(f"❌ {component}: {test} - {details}")
    
    async def verify_hybrid_memory(self):
        """Verify Hybrid Memory System"""
        print("\n🧠 Testing Hybrid Memory System...")
        
        try:
            # Initialize memory
            memory = HybridMemory()
            self.report("HybridMemory", "Initialization", True)
            
            # Test vector memory
            embedding = np.random.randn(768).astype(np.float32)
            metadata = {"type": "test", "content": "test vector"}
            
            vector_id = await memory.add_vector_memory(embedding, metadata)
            self.report("HybridMemory", "Add Vector Memory", True, f"ID: {vector_id}")
            
            # Test similarity search
            results = await memory.vector_memory.similarity_search(embedding, k=5)
            self.report("HybridMemory", "Similarity Search", len(results) > 0, f"Found {len(results)} results")
            
            # Test episodic memory
            episode_id = await memory.add_episodic_memory(
                event_type="test_event",
                content="Test episode content",
                importance=0.8
            )
            self.report("HybridMemory", "Add Episodic Memory", True, f"ID: {episode_id}")
            
            # Test semantic memory
            node_id = await memory.add_semantic_memory(
                concept="test_concept",
                properties={"category": "test"},
                embedding=embedding
            )
            self.report("HybridMemory", "Add Semantic Memory", True, f"ID: {node_id}")
            
            # Test working memory
            context_id = await memory.update_working_memory(
                context_type="test_context",
                content="Test working memory",
                attention_score=0.9
            )
            self.report("HybridMemory", "Update Working Memory", True, f"ID: {context_id}")
            
            # Test memory consolidation
            consolidated = await memory.consolidate_memories(similarity_threshold=0.7)
            self.report("HybridMemory", "Memory Consolidation", True, f"Consolidated {consolidated} concepts")
            
            # Test save/load with encryption
            save_path = Path("/tmp/safla_memory_test.enc")
            await memory.save_to_disk(save_path, encrypt=True)
            self.report("HybridMemory", "Save to Disk (Encrypted)", save_path.exists())
            
            # Clear and reload
            await memory.clear_all()
            await memory.load_from_disk(save_path)
            self.report("HybridMemory", "Load from Disk (Encrypted)", True)
            
            # Clean up
            save_path.unlink(missing_ok=True)
            
        except Exception as e:
            self.report("HybridMemory", "General Test", False, str(e))
    
    async def verify_metacognitive_engine(self):
        """Verify Meta-Cognitive Engine"""
        print("\n🤖 Testing Meta-Cognitive Engine...")
        
        try:
            # Initialize engine
            config = get_config()
            engine = MetaCognitiveEngine(config)
            await engine.initialize()
            self.report("MetaCognitiveEngine", "Initialization", True)
            
            # Test self-awareness
            awareness = await engine.assess_self_awareness()
            self.report("MetaCognitiveEngine", "Self-Awareness Assessment", 
                       awareness['awareness_level'] > 0, 
                       f"Level: {awareness['awareness_level']}")
            
            # Test goal management
            goal_id = await engine.set_goal(
                goal_type="optimization",
                description="Optimize memory usage",
                priority=0.8,
                metrics={"memory_reduction": 0.2}
            )
            self.report("MetaCognitiveEngine", "Set Goal", True, f"ID: {goal_id}")
            
            # Test strategy selection
            context = {
                "task": "memory_optimization",
                "constraints": {"time_limit": 60},
                "resources": {"cpu": 0.8, "memory": 0.5}
            }
            strategy = await engine.select_strategy(context)
            self.report("MetaCognitiveEngine", "Strategy Selection", 
                       strategy is not None, 
                       f"Selected: {strategy.get('name', 'Unknown')}")
            
            # Test performance monitoring
            await engine.update_performance_metrics({
                "response_time": 100,
                "accuracy": 0.95,
                "resource_usage": 0.6
            })
            status = await engine.get_performance_status()
            self.report("MetaCognitiveEngine", "Performance Monitoring", True, 
                       f"Health: {status['health_status']}")
            
            # Test adaptation
            adaptations = await engine.adapt_behavior({"performance_drop": True})
            self.report("MetaCognitiveEngine", "Behavior Adaptation", 
                       len(adaptations) > 0, 
                       f"Made {len(adaptations)} adaptations")
            
        except Exception as e:
            self.report("MetaCognitiveEngine", "General Test", False, str(e))
    
    async def verify_safety_framework(self):
        """Verify Safety & Validation Framework"""
        print("\n🛡️ Testing Safety & Validation Framework...")
        
        try:
            # Initialize framework
            framework = SafetyValidationFramework()
            self.report("SafetyFramework", "Initialization", True)
            
            # Test safety constraints
            safe_action = {"action": "read_memory", "target": "vector_memory"}
            unsafe_action = {"action": "delete_all", "target": "system"}
            
            safe_valid = await framework.validate_action(safe_action)
            self.report("SafetyFramework", "Validate Safe Action", 
                       safe_valid['valid'], 
                       f"Risk: {safe_valid.get('risk_level', 0)}")
            
            unsafe_valid = await framework.validate_action(unsafe_action)
            self.report("SafetyFramework", "Block Unsafe Action", 
                       not unsafe_valid['valid'], 
                       "Correctly blocked dangerous action")
            
            # Test validation pipeline
            data = {"input": "test data", "operation": "process"}
            validation_result = await framework.run_validation_pipeline(data)
            self.report("SafetyFramework", "Validation Pipeline", 
                       validation_result['passed'], 
                       f"Stages: {validation_result.get('stages_passed', 0)}")
            
            # Test risk assessment
            risk = await framework.assess_risk({
                "action": "modify_config",
                "scope": "limited",
                "reversible": True
            })
            self.report("SafetyFramework", "Risk Assessment", 
                       0 <= risk['risk_score'] <= 1, 
                       f"Score: {risk['risk_score']:.2f}")
            
            # Test rollback mechanism
            checkpoint = await framework.create_checkpoint("test_checkpoint")
            self.report("SafetyFramework", "Create Checkpoint", True, 
                       f"ID: {checkpoint['checkpoint_id']}")
            
            # Test monitoring
            await framework.start_monitoring()
            self.report("SafetyFramework", "Safety Monitoring", True, "Monitoring active")
            
        except Exception as e:
            self.report("SafetyFramework", "General Test", False, str(e))
    
    async def verify_authentication(self):
        """Verify JWT Authentication System"""
        print("\n🔐 Testing Authentication System...")
        
        try:
            # Initialize JWT manager
            jwt_manager = JWTManager()
            auth_middleware = AuthMiddleware(jwt_manager)
            self.report("Authentication", "JWT Manager Init", True)
            
            # Generate tokens
            token, expiry = jwt_manager.generate_token(
                user_id="test_user",
                roles=["developer"],
                permissions=["tools:read", "tools:execute"]
            )
            self.report("Authentication", "Token Generation", len(token) > 0, 
                       f"Expires: {expiry.isoformat()}")
            
            # Validate token
            payload = jwt_manager.validate_token(token)
            self.report("Authentication", "Token Validation", 
                       payload.sub == "test_user", 
                       f"User: {payload.sub}")
            
            # Test refresh token
            refresh_token, _ = jwt_manager.generate_refresh_token("test_user")
            new_token, _ = jwt_manager.refresh_access_token(refresh_token)
            self.report("Authentication", "Token Refresh", len(new_token) > 0)
            
            # Test middleware
            request = {
                "method": "tools/list",
                "params": {"auth_token": token}
            }
            auth_result = await auth_middleware.authenticate_request(request)
            self.report("Authentication", "Middleware Authentication", 
                       auth_result is not None, 
                       "Request authenticated")
            
        except Exception as e:
            self.report("Authentication", "General Test", False, str(e))
    
    async def verify_encryption(self):
        """Verify Data Encryption"""
        print("\n🔒 Testing Data Encryption...")
        
        try:
            # Initialize encryptor
            encryptor = DataEncryptor()
            self.report("Encryption", "Encryptor Init", True)
            
            # Test string encryption
            test_data = "Sensitive information"
            encrypted = encryptor.encrypt_string(test_data)
            decrypted = encryptor.decrypt_string(encrypted)
            self.report("Encryption", "String Encryption/Decryption", 
                       decrypted == test_data, 
                       "Data integrity verified")
            
            # Test dictionary encryption
            test_dict = {"user": "admin", "password": "secret123"}
            encrypted_dict = encryptor.encrypt_dict(test_dict)
            decrypted_dict = encryptor.decrypt_dict(encrypted_dict)
            self.report("Encryption", "Dictionary Encryption", 
                       decrypted_dict == test_dict, 
                       "Complex data encrypted")
            
            # Test secure storage
            storage = SecureStorage()
            storage.store("test_key", test_dict, encrypt=True)
            retrieved = storage.retrieve("test_key", decrypt=True)
            self.report("Encryption", "Secure Storage", 
                       retrieved == test_dict, 
                       "Secure storage working")
            
            # Clean up
            storage.delete("test_key")
            
        except Exception as e:
            self.report("Encryption", "General Test", False, str(e))
    
    async def verify_rate_limiting(self):
        """Verify Rate Limiting"""
        print("\n⏱️ Testing Rate Limiting...")
        
        try:
            # Initialize rate limiter
            rate_limiter = RateLimiter()
            self.report("RateLimiting", "Rate Limiter Init", True)
            
            # Test normal requests
            for i in range(5):
                request = {"method": f"test_{i}"}
                allowed, error = await rate_limiter.check_rate_limit(request)
                if not allowed:
                    break
            
            self.report("RateLimiting", "Normal Request Flow", 
                       i >= 4, 
                       f"Processed {i+1} requests")
            
            # Test rate limit enforcement
            requests_blocked = 0
            for i in range(100):
                request = {"method": "tools/call"}
                allowed, error = await rate_limiter.check_rate_limit(
                    request, 
                    identifier="test_user"
                )
                if not allowed:
                    requests_blocked += 1
            
            self.report("RateLimiting", "Rate Limit Enforcement", 
                       requests_blocked > 0, 
                       f"Blocked {requests_blocked} requests")
            
            # Test stats
            stats = await rate_limiter.get_stats("test_user")
            self.report("RateLimiting", "Statistics Tracking", 
                       stats['requests_last_minute'] > 0, 
                       f"Tracked {stats['requests_last_minute']} requests")
            
        except Exception as e:
            self.report("RateLimiting", "General Test", False, str(e))
    
    async def verify_validation(self):
        """Verify Input Validation"""
        print("\n✅ Testing Input Validation...")
        
        try:
            # Test path validation
            try:
                safe_path = validate_path("/tmp/test.txt", base_dir="/tmp")
                self.report("Validation", "Safe Path Validation", True, safe_path)
            except ValueError:
                self.report("Validation", "Safe Path Validation", False, "Rejected valid path")
            
            # Test path traversal prevention
            try:
                validate_path("../../../etc/passwd", base_dir="/tmp")
                self.report("Validation", "Path Traversal Prevention", False, 
                           "Failed to block traversal")
            except ValueError:
                self.report("Validation", "Path Traversal Prevention", True, 
                           "Correctly blocked traversal")
            
            # Test tool name validation
            try:
                valid_name = validate_tool_name("get_system_info")
                self.report("Validation", "Valid Tool Name", True, valid_name)
            except ValueError:
                self.report("Validation", "Valid Tool Name", False, "Rejected valid name")
            
            # Test invalid tool name
            try:
                validate_tool_name("../../evil")
                self.report("Validation", "Invalid Tool Name", False, 
                           "Failed to block invalid name")
            except ValueError:
                self.report("Validation", "Invalid Tool Name", True, 
                           "Correctly blocked invalid name")
            
        except Exception as e:
            self.report("Validation", "General Test", False, str(e))
    
    async def verify_handler_registry(self):
        """Verify Handler Registry"""
        print("\n📋 Testing Handler Registry...")
        
        try:
            # Get registry
            registry = get_registry()
            
            # Import handlers to register them
            import safla.mcp.handlers.core_tools
            import safla.mcp.handlers.deployment_tools
            import safla.mcp.handlers.optimization_tools
            
            handlers = registry.list_handlers()
            self.report("HandlerRegistry", "Handler Registration", 
                       len(handlers) > 0, 
                       f"Registered {len(handlers)} handlers")
            
            # Test handler categories
            categories = registry.list_categories()
            self.report("HandlerRegistry", "Handler Categories", 
                       len(categories) > 0, 
                       f"Categories: {', '.join(categories)}")
            
            # Test handler dispatch
            if handlers:
                test_handler = handlers[0]
                try:
                    context = {"authenticated": True}
                    result = await registry.dispatch(
                        test_handler.name, 
                        {}, 
                        context
                    )
                    self.report("HandlerRegistry", "Handler Dispatch", True, 
                               f"Dispatched: {test_handler.name}")
                except Exception as e:
                    self.report("HandlerRegistry", "Handler Dispatch", False, str(e))
            
        except Exception as e:
            self.report("HandlerRegistry", "General Test", False, str(e))
    
    async def run_all_tests(self):
        """Run all verification tests"""
        print("🚀 SAFLA System Verification Starting...\n")
        
        # Run all component tests
        await self.verify_hybrid_memory()
        await self.verify_metacognitive_engine()
        await self.verify_safety_framework()
        await self.verify_authentication()
        await self.verify_encryption()
        await self.verify_rate_limiting()
        await self.verify_validation()
        await self.verify_handler_registry()
        
        # Print summary
        print("\n" + "="*60)
        print("📊 VERIFICATION SUMMARY")
        print("="*60)
        
        for component, tests in self.results.items():
            passed = sum(1 for t in tests if t['success'])
            total = len(tests)
            print(f"\n{component}:")
            print(f"  ✅ Passed: {passed}/{total}")
            
            # Show failed tests
            failed_tests = [t for t in tests if not t['success']]
            if failed_tests:
                print("  ❌ Failed:")
                for test in failed_tests:
                    print(f"    - {test['test']}: {test['details']}")
        
        print("\n" + "="*60)
        print(f"TOTAL: ✅ {self.passed} passed, ❌ {self.failed} failed")
        print("="*60)
        
        return self.failed == 0


async def main():
    """Main verification entry point"""
    verifier = SystemVerifier()
    success = await verifier.run_all_tests()
    
    # Save results
    with open('/tmp/safla_verification_results.json', 'w') as f:
        json.dump(verifier.results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: /tmp/safla_verification_results.json")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())