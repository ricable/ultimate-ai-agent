#!/usr/bin/env python3
"""Test if enhanced endpoints can be imported."""

try:
    from safla.api.enhanced_endpoints import EnhancedSAFLAEndpoints
    print("✅ Successfully imported EnhancedSAFLAEndpoints")
    
    # Test instantiation
    from types import SimpleNamespace
    import asyncio
    
    async def mock_generate_embeddings(texts):
        return [[0.1] * 384 for _ in texts]
    
    safla_mock = SimpleNamespace(
        neural_engine=SimpleNamespace(
            generate_embeddings=mock_generate_embeddings
        ),
        memory=SimpleNamespace(
            store=lambda k, v: None,
            retrieve=lambda k: None
        )
    )
    
    enhanced = EnhancedSAFLAEndpoints(safla_mock)
    print("✅ Successfully created EnhancedSAFLAEndpoints instance")
    
    # Test that the analyze_text method exists
    if hasattr(enhanced, 'analyze_text'):
        print("✅ analyze_text method exists")
    else:
        print("❌ analyze_text method not found")
        
    # List all methods
    methods = [method for method in dir(enhanced) if not method.startswith('_')]
    print(f"Available methods: {methods}")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Error: {e}")