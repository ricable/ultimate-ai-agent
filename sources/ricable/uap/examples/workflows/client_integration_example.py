"""
Client Integration Example

This example demonstrates how to integrate with the UAP backend using the SDK client.
"""

import asyncio
import json
from typing import Dict, Any, Optional
from uap_sdk import UAPClient, Configuration
from uap_sdk.exceptions import UAPException, UAPConnectionError, UAPAuthError


class UAP客户端示例:
    """Example demonstrating UAP client integration."""
    
    def __init__(self):
        self.client: Optional[UAPClient] = None
        self.config: Optional[Configuration] = None
    
    async def initialize(self) -> bool:
        """Initialize the UAP client with configuration."""
        try:
            # Create configuration
            self.config = Configuration({
                "backend_url": "http://localhost:8000",
                "websocket_url": "ws://localhost:8000",
                "http_timeout": 30,
                "websocket_timeout": 30,
                "log_level": "INFO"
            })
            
            # Create client
            self.client = UAPClient(self.config)
            
            print("UAP Client initialized successfully")
            return True
            
        except Exception as e:
            print(f"Failed to initialize UAP client: {e}")
            return False
    
    async def authenticate(self, username: str = "admin", password: str = "admin123!") -> bool:
        """Authenticate with the UAP backend."""
        try:
            result = await self.client.login(username, password)
            print(f"Authentication successful: {result.get('message', 'Logged in')}")
            return True
            
        except UAPAuthError as e:
            print(f"Authentication failed: {e.message}")
            return False
        except UAPConnectionError as e:
            print(f"Connection failed: {e.message}")
            print("Make sure the UAP backend is running on http://localhost:8000")
            return False
        except Exception as e:
            print(f"Unexpected authentication error: {e}")
            return False
    
    async def check_system_status(self) -> Dict[str, Any]:
        """Check the status of the UAP system."""
        try:
            status = await self.client.get_status()
            print("System Status:")
            print(f"  Overall Status: {status.get('status', 'unknown')}")
            
            frameworks = status.get('frameworks', {})
            print(f"  Available Frameworks: {len(frameworks)}")
            
            for framework_name, framework_info in frameworks.items():
                print(f"    {framework_name}: {framework_info.get('status', 'unknown')}")
            
            return status
            
        except Exception as e:
            print(f"Failed to get system status: {e}")
            return {}
    
    async def chat_with_agents(self) -> None:
        """Demonstrate chatting with different agents."""
        print("\n=== Agent Communication Examples ===")
        
        # Test messages for different frameworks
        test_scenarios = [
            {
                "agent_id": "copilot-agent",
                "framework": "copilot",
                "messages": [
                    "Hello! Can you help me with coding?",
                    "How do I create a Python function?",
                    "What's the best way to handle errors in Python?"
                ]
            },
            {
                "agent_id": "document-agent", 
                "framework": "agno",
                "messages": [
                    "Can you analyze documents?",
                    "What types of documents can you process?",
                    "How do you extract information from PDFs?"
                ]
            },
            {
                "agent_id": "workflow-agent",
                "framework": "mastra", 
                "messages": [
                    "Help me create a workflow",
                    "What workflow capabilities do you have?",
                    "How do I automate a business process?"
                ]
            },
            {
                "agent_id": "auto-agent",
                "framework": "auto",
                "messages": [
                    "What's the weather like today?",
                    "Can you help me solve a math problem: 15 * 23?",
                    "Tell me about artificial intelligence"
                ]
            }
        ]
        
        for scenario in test_scenarios:
            agent_id = scenario["agent_id"]
            framework = scenario["framework"]
            messages = scenario["messages"]
            
            print(f"\n--- Testing {framework.upper()} Framework ({agent_id}) ---")
            
            for message in messages:
                try:
                    print(f"User: {message}")
                    
                    # Send message via HTTP
                    response = await self.client.chat(
                        agent_id=agent_id,
                        message=message,
                        framework=framework,
                        use_websocket=False
                    )
                    
                    # Display response
                    content = response.get('content', '')
                    if len(content) > 200:
                        content = content[:200] + "..."
                    
                    print(f"Agent: {content}")
                    
                    # Show metadata if available
                    metadata = response.get('metadata', {})
                    if metadata:
                        print(f"Metadata: Framework={metadata.get('framework', 'unknown')}, "
                              f"Response time={metadata.get('response_time', 'N/A')}")
                    
                    print()
                    
                except Exception as e:
                    print(f"Error: {e}")
                    print()
    
    async def demonstrate_websocket_communication(self) -> None:
        """Demonstrate real-time WebSocket communication."""
        print("\n=== WebSocket Communication Example ===")
        
        try:
            # Connect to WebSocket
            agent_id = "websocket-demo"
            await self.client.connect_websocket(agent_id)
            print(f"Connected to WebSocket for agent: {agent_id}")
            
            # Set up message handler
            received_messages = []
            
            def handle_response(event: Dict[str, Any]) -> None:
                """Handle WebSocket responses."""
                event_type = event.get('type', 'unknown')
                content = event.get('content', '')
                
                print(f"WebSocket Event [{event_type}]: {content}")
                received_messages.append(event)
            
            # Register handlers
            self.client.websocket.on_message("text_message_content", handle_response)
            self.client.websocket.on_message("tool_call_start", handle_response)
            self.client.websocket.on_message("tool_call_end", handle_response)
            self.client.websocket.on_message("state_delta", handle_response)
            
            # Send some messages
            test_messages = [
                "Hello via WebSocket!",
                "How are you today?",
                "Can you process this in real-time?"
            ]
            
            for message in test_messages:
                print(f"Sending: {message}")
                await self.client.websocket.send_message(
                    message=message,
                    metadata={"framework": "auto", "realtime": True}
                )
                
                # Wait a bit for response
                await asyncio.sleep(1)
            
            print(f"Received {len(received_messages)} WebSocket events")
            
        except Exception as e:
            print(f"WebSocket communication error: {e}")
    
    async def document_processing_example(self) -> None:
        """Demonstrate document processing capabilities."""
        print("\n=== Document Processing Example ===")
        
        # Create a sample document
        sample_content = """
        # Sample Document
        
        This is a sample document for testing UAP document processing capabilities.
        
        ## Features
        - Document upload
        - Text extraction
        - Content analysis
        - Metadata extraction
        
        ## Data
        - Date: 2024-01-01
        - Author: UAP SDK Example
        - Version: 1.0
        """
        
        # Save sample document
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(sample_content)
            temp_file_path = f.name
        
        try:
            # Upload document
            print(f"Uploading document: {temp_file_path}")
            result = await self.client.upload_document(
                file_path=temp_file_path,
                process_immediately=True
            )
            
            print(f"Upload result: {result.get('status', 'unknown')}")
            
            if 'document_id' in result:
                document_id = result['document_id']
                print(f"Document ID: {document_id}")
                
                # You could then query the document or use it in agent conversations
                response = await self.client.chat(
                    agent_id="document-agent",
                    message=f"Please analyze the document with ID: {document_id}",
                    framework="agno"
                )
                
                print(f"Document analysis: {response.get('content', '')[:200]}...")
            
        except Exception as e:
            print(f"Document processing error: {e}")
        
        finally:
            # Clean up
            try:
                os.unlink(temp_file_path)
            except:
                pass
    
    async def error_handling_examples(self) -> None:
        """Demonstrate proper error handling."""
        print("\n=== Error Handling Examples ===")
        
        # Test authentication errors
        print("1. Testing authentication error...")
        try:
            fake_client = UAPClient(self.config)
            await fake_client.login("invalid_user", "wrong_password")
        except UAPAuthError as e:
            print(f"Caught authentication error: {e.message}")
        
        # Test connection errors
        print("\n2. Testing connection error...")
        try:
            offline_config = Configuration({"backend_url": "http://localhost:9999"})
            offline_client = UAPClient(offline_config)
            await offline_client.get_status()
        except UAPConnectionError as e:
            print(f"Caught connection error: {e.message}")
        
        # Test invalid agent requests
        print("\n3. Testing invalid agent request...")
        try:
            response = await self.client.chat(
                agent_id="nonexistent-agent",
                message="Hello",
                framework="invalid-framework"
            )
        except UAPException as e:
            print(f"Caught UAP error: {e.message}")
        except Exception as e:
            print(f"Caught general error: {e}")
    
    async def performance_testing(self) -> None:
        """Demonstrate performance testing with multiple requests."""
        print("\n=== Performance Testing Example ===")
        
        import time
        
        # Test multiple sequential requests
        print("Testing sequential requests...")
        start_time = time.time()
        
        for i in range(5):
            try:
                response = await self.client.chat(
                    agent_id="perf-test",
                    message=f"Test message {i+1}",
                    framework="auto"
                )
                print(f"Request {i+1}: {response.get('metadata', {}).get('response_time', 'N/A')}ms")
            except Exception as e:
                print(f"Request {i+1} failed: {e}")
        
        sequential_time = time.time() - start_time
        print(f"Sequential requests took: {sequential_time:.2f}s")
        
        # Test concurrent requests
        print("\nTesting concurrent requests...")
        start_time = time.time()
        
        async def make_request(request_id: int) -> Dict[str, Any]:
            try:
                return await self.client.chat(
                    agent_id="perf-test",
                    message=f"Concurrent test message {request_id}",
                    framework="auto"
                )
            except Exception as e:
                return {"error": str(e), "request_id": request_id}
        
        # Make 5 concurrent requests
        tasks = [make_request(i) for i in range(1, 6)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        concurrent_time = time.time() - start_time
        print(f"Concurrent requests took: {concurrent_time:.2f}s")
        
        successful_requests = len([r for r in results if not isinstance(r, Exception) and "error" not in r])
        print(f"Successful requests: {successful_requests}/5")
    
    async def cleanup(self) -> None:
        """Clean up client resources."""
        if self.client:
            await self.client.cleanup()
            print("Client cleaned up successfully")


async def main():
    """Main function demonstrating UAP client integration."""
    print("=== UAP Client Integration Example ===\n")
    
    example = UAP客户端示例()
    
    # Initialize
    if not await example.initialize():
        print("Failed to initialize. Exiting.")
        return
    
    # Check if backend is available
    print("Checking system status...")
    status = await example.check_system_status()
    
    if not status:
        print("\nBackend appears to be unavailable.")
        print("To run this example:")
        print("1. Start the UAP backend: uap deploy start")
        print("2. Or run: devbox run dev")
        print("3. Then run this example again")
        return
    
    # Try to authenticate (optional for status checks)
    print("\nAttempting authentication...")
    auth_success = await example.authenticate()
    
    if auth_success:
        # Run various examples
        await example.chat_with_agents()
        
        # Only run WebSocket example if authentication succeeded
        # (WebSocket may require authentication)
        await example.demonstrate_websocket_communication()
        
        await example.document_processing_example()
        
        await example.performance_testing()
    
    # Always run error handling examples
    await example.error_handling_examples()
    
    # Cleanup
    await example.cleanup()
    
    print("\n=== Example completed ===")


if __name__ == "__main__":
    asyncio.run(main())