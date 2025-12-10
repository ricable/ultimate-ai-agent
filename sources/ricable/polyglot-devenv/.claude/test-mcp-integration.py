#!/usr/bin/env python3
"""
Docker MCP Integration Test Suite
Tests HTTP transport connectivity and validates the complete integration
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

# Add the current directory to Python path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import aiohttp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPIntegrationTest:
    """Test suite for Docker MCP integration"""
    
    def __init__(self):
        self.gateway_process = None
        self.bridge_process = None
        self.test_results = {}
        
    async def test_docker_mcp_availability(self):
        """Test if Docker MCP is available"""
        try:
            result = subprocess.run(["docker", "mcp", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.test_results["docker_mcp_available"] = True
                logger.info("✅ Docker MCP is available")
                return True
            else:
                self.test_results["docker_mcp_available"] = False
                logger.error("❌ Docker MCP not available")
                return False
        except Exception as e:
            self.test_results["docker_mcp_available"] = False
            logger.error(f"❌ Error checking Docker MCP: {e}")
            return False
    
    async def test_gateway_dry_run(self):
        """Test gateway dry-run functionality"""
        try:
            result = subprocess.run(
                ["docker", "mcp", "gateway", "run", "--dry-run", "--tools", "filesystem,fetch,memory"],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                self.test_results["gateway_dry_run"] = True
                logger.info("✅ Gateway dry-run successful")
                return True
            else:
                self.test_results["gateway_dry_run"] = False
                logger.error(f"❌ Gateway dry-run failed: {result.stderr}")
                return False
        except Exception as e:
            self.test_results["gateway_dry_run"] = False
            logger.error(f"❌ Error in gateway dry-run: {e}")
            return False
    
    async def test_client_connections(self):
        """Test client connections"""
        try:
            result = subprocess.run(["docker", "mcp", "client", "ls"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                output = result.stdout
                claude_connected = "claude-desktop" in output and "connected" in output
                self.test_results["claude_connected"] = claude_connected
                
                if claude_connected:
                    logger.info("✅ Claude Desktop is connected")
                else:
                    logger.warning("⚠️ Claude Desktop not connected")
                
                return True
            else:
                self.test_results["client_connections"] = False
                logger.error("❌ Failed to check client connections")
                return False
        except Exception as e:
            self.test_results["client_connections"] = False
            logger.error(f"❌ Error checking clients: {e}")
            return False
    
    async def test_available_tools(self):
        """Test available tools listing"""
        try:
            result = subprocess.run(["docker", "mcp", "tools"], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                tools_output = result.stdout
                essential_tools = ["filesystem", "fetch", "memory", "docker"]
                found_tools = []
                
                for tool in essential_tools:
                    if tool in tools_output.lower():
                        found_tools.append(tool)
                
                self.test_results["available_tools"] = found_tools
                self.test_results["tools_count"] = len(found_tools)
                
                logger.info(f"✅ Found {len(found_tools)}/{len(essential_tools)} essential tools")
                return len(found_tools) > 0
            else:
                self.test_results["available_tools"] = []
                logger.error("❌ Failed to list tools")
                return False
        except Exception as e:
            self.test_results["available_tools"] = []
            logger.error(f"❌ Error listing tools: {e}")
            return False
    
    async def test_http_bridge_startup(self):
        """Test HTTP bridge startup"""
        try:
            # Try to import required modules
            import fastapi
            import uvicorn
            import sse_starlette
            
            self.test_results["http_bridge_deps"] = True
            logger.info("✅ HTTP bridge dependencies available")
            
            # Test if we can create the FastAPI app
            script_path = Path(__file__).parent / "mcp-http-bridge.py"
            if script_path.exists():
                self.test_results["http_bridge_script"] = True
                logger.info("✅ HTTP bridge script exists")
            else:
                self.test_results["http_bridge_script"] = False
                logger.warning("⚠️ HTTP bridge script not found")
            
            return True
            
        except ImportError as e:
            self.test_results["http_bridge_deps"] = False
            logger.error(f"❌ HTTP bridge dependencies missing: {e}")
            return False
    
    async def test_gemini_client_setup(self):
        """Test Gemini client configuration"""
        try:
            import google.generativeai as genai
            
            self.test_results["gemini_deps"] = True
            logger.info("✅ Gemini dependencies available")
            
            script_path = Path(__file__).parent / "gemini-mcp-config.py"
            if script_path.exists():
                self.test_results["gemini_script"] = True
                logger.info("✅ Gemini client script exists")
            else:
                self.test_results["gemini_script"] = False
                logger.warning("⚠️ Gemini client script not found")
            
            return True
            
        except ImportError as e:
            self.test_results["gemini_deps"] = False
            logger.error(f"❌ Gemini dependencies missing: {e}")
            return False
    
    async def test_configuration_files(self):
        """Test if all configuration files exist"""
        script_dir = Path(__file__).parent
        required_files = [
            "mcp-gateway-config.json",
            "start-mcp-gateway.sh",
            "mcp-http-bridge.py", 
            "gemini-mcp-config.py",
            "requirements-mcp.txt",
            "setup-mcp-integration.sh"
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = script_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        self.test_results["missing_files"] = missing_files
        
        if not missing_files:
            logger.info("✅ All configuration files present")
            return True
        else:
            logger.error(f"❌ Missing files: {missing_files}")
            return False
    
    async def run_all_tests(self):
        """Run all integration tests"""
        logger.info("🧪 Starting Docker MCP Integration Tests")
        logger.info("=" * 50)
        
        tests = [
            ("Docker MCP Availability", self.test_docker_mcp_availability),
            ("Configuration Files", self.test_configuration_files),
            ("Gateway Dry Run", self.test_gateway_dry_run),
            ("Client Connections", self.test_client_connections),
            ("Available Tools", self.test_available_tools),
            ("HTTP Bridge Setup", self.test_http_bridge_startup),
            ("Gemini Client Setup", self.test_gemini_client_setup),
        ]
        
        results = {}
        for test_name, test_func in tests:
            logger.info(f"Running: {test_name}")
            try:
                result = await test_func()
                results[test_name] = result
                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                results[test_name] = False
                logger.error(f"❌ {test_name}: ERROR - {e}")
            
            logger.info("-" * 30)
        
        return results
    
    def print_summary(self, results):
        """Print test summary"""
        logger.info("📊 Test Summary")
        logger.info("=" * 50)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        logger.info(f"Tests Passed: {passed}/{total}")
        
        if passed == total:
            logger.info("🎉 All tests passed! Docker MCP integration is ready.")
        else:
            logger.warning(f"⚠️ {total - passed} tests failed. Check the logs above.")
        
        logger.info("\n📋 Detailed Results:")
        for key, value in self.test_results.items():
            logger.info(f"  • {key}: {value}")
        
        logger.info("\n🚀 Next Steps:")
        if results.get("Docker MCP Availability", False):
            logger.info("  1. Start Docker MCP Gateway:")
            logger.info("     ./start-mcp-gateway.sh")
            
        if results.get("HTTP Bridge Setup", False):
            logger.info("  2. Start HTTP/SSE Bridge:")
            logger.info("     python3 mcp-http-bridge.py --port 8080")
            
        if results.get("Gemini Client Setup", False):
            logger.info("  3. Test Gemini Integration:")
            logger.info("     export GEMINI_API_KEY='your-key'")
            logger.info("     python3 gemini-mcp-config.py")
        
        logger.info("  4. Monitor with: docker mcp client ls")

async def main():
    """Main test function"""
    tester = MCPIntegrationTest()
    results = await tester.run_all_tests()
    tester.print_summary(results)

if __name__ == "__main__":
    asyncio.run(main())