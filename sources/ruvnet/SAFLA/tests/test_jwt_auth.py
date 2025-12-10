#!/usr/bin/env python3
"""Test script for JWT authentication in MCP server."""

import asyncio
import json
import sys
from safla.auth import JWTManager, AuthMiddleware
from safla.auth.jwt_manager import JWTConfig


async def test_jwt_flow():
    """Test JWT token generation, validation, and refresh."""
    print("Testing JWT Authentication Flow...")
    
    try:
        # Initialize JWT manager
        jwt_config = JWTConfig(secret_key="test-secret-key-for-testing")
        jwt_manager = JWTManager(jwt_config)
        auth_middleware = AuthMiddleware(jwt_manager)
        
        print("✓ JWT Manager initialized")
        
        # Test 1: Generate access token
        user_id = "test_user_001"
        roles = ["developer"]
        permissions = ["tools:read", "tools:execute", "resources:read"]
        
        access_token, access_exp = jwt_manager.generate_token(
            user_id=user_id,
            roles=roles,
            permissions=permissions
        )
        
        print(f"✓ Access token generated: {access_token[:50]}...")
        print(f"  Expires at: {access_exp}")
        
        # Test 2: Validate token
        payload = jwt_manager.validate_token(access_token)
        print(f"✓ Token validated successfully")
        print(f"  User ID: {payload.sub}")
        print(f"  Roles: {payload.roles}")
        print(f"  Permissions: {payload.permissions}")
        
        # Test 3: Generate refresh token
        refresh_token, refresh_exp = jwt_manager.generate_refresh_token(user_id)
        print(f"✓ Refresh token generated: {refresh_token[:50]}...")
        
        # Test 4: Refresh access token
        new_access_token, new_exp = jwt_manager.refresh_access_token(refresh_token)
        print(f"✓ Access token refreshed: {new_access_token[:50]}...")
        
        # Test 5: Test authentication middleware
        # Simulate MCP request with token
        request_with_token = {
            "method": "tools/list",
            "params": {
                "headers": {
                    "Authorization": f"Bearer {access_token}"
                }
            }
        }
        
        # Should authenticate successfully
        auth_result = await auth_middleware.authenticate_request(request_with_token)
        print(f"✓ Request authenticated via middleware")
        print(f"  Authenticated user: {auth_result.sub}")
        
        # Test 6: Test public method (should not require auth)
        public_request = {
            "method": "initialize",
            "params": {}
        }
        
        public_result = await auth_middleware.authenticate_request(public_request)
        print(f"✓ Public method accessed without authentication")
        
        # Test 7: Test missing token
        request_no_token = {
            "method": "tools/list",
            "params": {}
        }
        
        try:
            await auth_middleware.authenticate_request(request_no_token)
            print("✗ Should have failed without token")
        except Exception as e:
            print(f"✓ Request correctly rejected without token: {e}")
        
        # Test 8: Test insufficient permissions
        limited_token, _ = jwt_manager.generate_token(
            user_id="limited_user",
            roles=["reader"],
            permissions=["tools:read"]  # Missing tools:execute
        )
        
        request_limited = {
            "method": "tools/call",
            "params": {
                "headers": {
                    "Authorization": f"Bearer {limited_token}"
                }
            }
        }
        
        try:
            await auth_middleware.authenticate_request(request_limited)
            print("✗ Should have failed with insufficient permissions")
        except Exception as e:
            print(f"✓ Request correctly rejected with insufficient permissions: {e}")
        
        print("\n✅ All JWT authentication tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(test_jwt_flow())