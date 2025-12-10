"""
Example Tool Plugin

This plugin demonstrates how to create tool plugins that provide functions for agents to use.
"""

import asyncio
import json
import os
import subprocess
from typing import Dict, Any, Callable, List
from datetime import datetime
from pathlib import Path
from uap_sdk.plugin import ToolPlugin, Configuration


class FileSystemToolPlugin(ToolPlugin):
    """Plugin that provides file system operations as tools."""
    
    PLUGIN_NAME = "filesystem"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_DESCRIPTION = "Provides file system operations like read, write, list directories"
    
    def __init__(self):
        super().__init__(self.PLUGIN_NAME, self.PLUGIN_VERSION)
        self.allowed_paths = []
        self.operation_count = 0
    
    async def initialize(self, config: Configuration) -> None:
        """Initialize the filesystem plugin."""
        # Set allowed paths from config
        self.allowed_paths = config.get("filesystem_allowed_paths", [str(Path.cwd())])
        print(f"Filesystem plugin initialized. Allowed paths: {self.allowed_paths}")
    
    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        print(f"Filesystem plugin cleaned up. Performed {self.operation_count} operations.")
    
    def get_tools(self) -> Dict[str, Callable]:
        """Get tools provided by this plugin."""
        return {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "list_directory": self._list_directory,
            "file_exists": self._file_exists,
            "create_directory": self._create_directory,
            "get_file_info": self._get_file_info
        }
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool function."""
        tools = self.get_tools()
        if tool_name not in tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        self.operation_count += 1
        tool_func = tools[tool_name]
        
        # Execute tool (handle both sync and async functions)
        if asyncio.iscoroutinefunction(tool_func):
            return await tool_func(**kwargs)
        else:
            return tool_func(**kwargs)
    
    def _is_path_allowed(self, path: str) -> bool:
        """Check if a path is within allowed directories."""
        abs_path = os.path.abspath(path)
        return any(abs_path.startswith(os.path.abspath(allowed)) for allowed in self.allowed_paths)
    
    def _read_file(self, file_path: str, max_size: int = 1024*1024) -> Dict[str, Any]:
        """Read contents of a file."""
        if not self._is_path_allowed(file_path):
            return {"error": f"Path not allowed: {file_path}"}
        
        try:
            path = Path(file_path)
            if not path.exists():
                return {"error": f"File not found: {file_path}"}
            
            if path.stat().st_size > max_size:
                return {"error": f"File too large (max {max_size} bytes)"}
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                "success": True,
                "content": content,
                "size": len(content),
                "path": str(path.absolute())
            }
        except Exception as e:
            return {"error": f"Failed to read file: {str(e)}"}
    
    def _write_file(self, file_path: str, content: str, create_dirs: bool = True) -> Dict[str, Any]:
        """Write content to a file."""
        if not self._is_path_allowed(file_path):
            return {"error": f"Path not allowed: {file_path}"}
        
        try:
            path = Path(file_path)
            
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True,
                "bytes_written": len(content.encode('utf-8')),
                "path": str(path.absolute())
            }
        except Exception as e:
            return {"error": f"Failed to write file: {str(e)}"}
    
    def _list_directory(self, dir_path: str = ".", include_hidden: bool = False) -> Dict[str, Any]:
        """List contents of a directory."""
        if not self._is_path_allowed(dir_path):
            return {"error": f"Path not allowed: {dir_path}"}
        
        try:
            path = Path(dir_path)
            if not path.exists():
                return {"error": f"Directory not found: {dir_path}"}
            
            if not path.is_dir():
                return {"error": f"Path is not a directory: {dir_path}"}
            
            items = []
            for item in path.iterdir():
                if not include_hidden and item.name.startswith('.'):
                    continue
                
                items.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                    "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                })
            
            return {
                "success": True,
                "path": str(path.absolute()),
                "items": sorted(items, key=lambda x: (x["type"], x["name"])),
                "count": len(items)
            }
        except Exception as e:
            return {"error": f"Failed to list directory: {str(e)}"}
    
    def _file_exists(self, file_path: str) -> Dict[str, Any]:
        """Check if a file exists."""
        if not self._is_path_allowed(file_path):
            return {"error": f"Path not allowed: {file_path}"}
        
        try:
            path = Path(file_path)
            return {
                "success": True,
                "exists": path.exists(),
                "is_file": path.is_file() if path.exists() else None,
                "is_directory": path.is_dir() if path.exists() else None,
                "path": str(path.absolute())
            }
        except Exception as e:
            return {"error": f"Failed to check file existence: {str(e)}"}
    
    def _create_directory(self, dir_path: str, parents: bool = True) -> Dict[str, Any]:
        """Create a directory."""
        if not self._is_path_allowed(dir_path):
            return {"error": f"Path not allowed: {dir_path}"}
        
        try:
            path = Path(dir_path)
            path.mkdir(parents=parents, exist_ok=True)
            
            return {
                "success": True,
                "path": str(path.absolute()),
                "created": True
            }
        except Exception as e:
            return {"error": f"Failed to create directory: {str(e)}"}
    
    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get detailed information about a file."""
        if not self._is_path_allowed(file_path):
            return {"error": f"Path not allowed: {file_path}"}
        
        try:
            path = Path(file_path)
            if not path.exists():
                return {"error": f"File not found: {file_path}"}
            
            stat = path.stat()
            
            return {
                "success": True,
                "path": str(path.absolute()),
                "name": path.name,
                "size": stat.st_size,
                "type": "directory" if path.is_dir() else "file",
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
                "permissions": oct(stat.st_mode)[-3:]
            }
        except Exception as e:
            return {"error": f"Failed to get file info: {str(e)}"}


class HTTPToolPlugin(ToolPlugin):
    """Plugin that provides HTTP request tools."""
    
    PLUGIN_NAME = "http"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_DESCRIPTION = "Provides HTTP request capabilities (GET, POST, etc.)"
    
    def __init__(self):
        super().__init__(self.PLUGIN_NAME, self.PLUGIN_VERSION)
        self.request_count = 0
        self.allowed_domains = []
    
    async def initialize(self, config: Configuration) -> None:
        """Initialize the HTTP plugin."""
        self.allowed_domains = config.get("http_allowed_domains", [])
        print(f"HTTP plugin initialized. Allowed domains: {self.allowed_domains}")
    
    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        print(f"HTTP plugin cleaned up. Made {self.request_count} requests.")
    
    def get_tools(self) -> Dict[str, Callable]:
        """Get tools provided by this plugin."""
        return {
            "http_get": self._http_get,
            "http_post": self._http_post,
            "check_url": self._check_url,
            "download_file": self._download_file
        }
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool function."""
        tools = self.get_tools()
        if tool_name not in tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        self.request_count += 1
        tool_func = tools[tool_name]
        
        if asyncio.iscoroutinefunction(tool_func):
            return await tool_func(**kwargs)
        else:
            return tool_func(**kwargs)
    
    def _is_domain_allowed(self, url: str) -> bool:
        """Check if a domain is allowed."""
        if not self.allowed_domains:  # If no restrictions, allow all
            return True
        
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return any(domain.endswith(allowed) for allowed in self.allowed_domains)
    
    async def _http_get(self, url: str, headers: Dict[str, str] = None, timeout: int = 30) -> Dict[str, Any]:
        """Make an HTTP GET request."""
        if not self._is_domain_allowed(url):
            return {"error": f"Domain not allowed: {url}"}
        
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, headers=headers or {})
                
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "content": response.text,
                    "url": str(response.url)
                }
        except Exception as e:
            return {"error": f"HTTP GET failed: {str(e)}"}
    
    async def _http_post(self, url: str, data: Dict[str, Any] = None, json_data: Dict[str, Any] = None, 
                         headers: Dict[str, str] = None, timeout: int = 30) -> Dict[str, Any]:
        """Make an HTTP POST request."""
        if not self._is_domain_allowed(url):
            return {"error": f"Domain not allowed: {url}"}
        
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                if json_data:
                    response = await client.post(url, json=json_data, headers=headers or {})
                else:
                    response = await client.post(url, data=data or {}, headers=headers or {})
                
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "content": response.text,
                    "url": str(response.url)
                }
        except Exception as e:
            return {"error": f"HTTP POST failed: {str(e)}"}
    
    async def _check_url(self, url: str, timeout: int = 10) -> Dict[str, Any]:
        """Check if a URL is accessible."""
        if not self._is_domain_allowed(url):
            return {"error": f"Domain not allowed: {url}"}
        
        try:
            import httpx
            import time
            
            start_time = time.time()
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.head(url)
                
                response_time = time.time() - start_time
                
                return {
                    "success": True,
                    "accessible": True,
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "url": str(response.url)
                }
        except Exception as e:
            return {
                "success": True,
                "accessible": False,
                "error": str(e),
                "url": url
            }
    
    async def _download_file(self, url: str, file_path: str, max_size: int = 10*1024*1024) -> Dict[str, Any]:
        """Download a file from a URL."""
        if not self._is_domain_allowed(url):
            return {"error": f"Domain not allowed: {url}"}
        
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    
                    # Check content length
                    content_length = response.headers.get("content-length")
                    if content_length and int(content_length) > max_size:
                        return {"error": f"File too large (max {max_size} bytes)"}
                    
                    with open(file_path, "wb") as f:
                        downloaded = 0
                        async for chunk in response.aiter_bytes():
                            downloaded += len(chunk)
                            if downloaded > max_size:
                                return {"error": f"File too large (max {max_size} bytes)"}
                            f.write(chunk)
                    
                    return {
                        "success": True,
                        "file_path": file_path,
                        "size": downloaded,
                        "url": url
                    }
        except Exception as e:
            return {"error": f"Download failed: {str(e)}"}


class UtilityToolPlugin(ToolPlugin):
    """Plugin that provides utility tools."""
    
    PLUGIN_NAME = "utilities"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_DESCRIPTION = "Provides utility functions like encoding, hashing, etc."
    
    def __init__(self):
        super().__init__(self.PLUGIN_NAME, self.PLUGIN_VERSION)
        self.operation_count = 0
    
    async def initialize(self, config: Configuration) -> None:
        """Initialize the utilities plugin."""
        print("Utilities plugin initialized")
    
    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        print(f"Utilities plugin cleaned up. Performed {self.operation_count} operations.")
    
    def get_tools(self) -> Dict[str, Callable]:
        """Get tools provided by this plugin."""
        return {
            "encode_base64": self._encode_base64,
            "decode_base64": self._decode_base64,
            "hash_text": self._hash_text,
            "generate_uuid": self._generate_uuid,
            "format_json": self._format_json,
            "validate_email": self._validate_email,
            "generate_password": self._generate_password
        }
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool function."""
        tools = self.get_tools()
        if tool_name not in tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        self.operation_count += 1
        tool_func = tools[tool_name]
        
        if asyncio.iscoroutinefunction(tool_func):
            return await tool_func(**kwargs)
        else:
            return tool_func(**kwargs)
    
    def _encode_base64(self, text: str) -> Dict[str, Any]:
        """Encode text to base64."""
        try:
            import base64
            encoded = base64.b64encode(text.encode('utf-8')).decode('utf-8')
            return {
                "success": True,
                "original": text,
                "encoded": encoded
            }
        except Exception as e:
            return {"error": f"Base64 encoding failed: {str(e)}"}
    
    def _decode_base64(self, encoded_text: str) -> Dict[str, Any]:
        """Decode base64 text."""
        try:
            import base64
            decoded = base64.b64decode(encoded_text).decode('utf-8')
            return {
                "success": True,
                "encoded": encoded_text,
                "decoded": decoded
            }
        except Exception as e:
            return {"error": f"Base64 decoding failed: {str(e)}"}
    
    def _hash_text(self, text: str, algorithm: str = "sha256") -> Dict[str, Any]:
        """Hash text using specified algorithm."""
        try:
            import hashlib
            
            if algorithm not in hashlib.algorithms_available:
                return {"error": f"Unsupported hash algorithm: {algorithm}"}
            
            hash_obj = hashlib.new(algorithm)
            hash_obj.update(text.encode('utf-8'))
            hashed = hash_obj.hexdigest()
            
            return {
                "success": True,
                "original": text,
                "algorithm": algorithm,
                "hash": hashed
            }
        except Exception as e:
            return {"error": f"Hashing failed: {str(e)}"}
    
    def _generate_uuid(self, version: int = 4) -> Dict[str, Any]:
        """Generate a UUID."""
        try:
            import uuid
            
            if version == 1:
                generated_uuid = str(uuid.uuid1())
            elif version == 4:
                generated_uuid = str(uuid.uuid4())
            else:
                return {"error": f"Unsupported UUID version: {version}"}
            
            return {
                "success": True,
                "uuid": generated_uuid,
                "version": version
            }
        except Exception as e:
            return {"error": f"UUID generation failed: {str(e)}"}
    
    def _format_json(self, json_text: str, indent: int = 2) -> Dict[str, Any]:
        """Format JSON text."""
        try:
            parsed = json.loads(json_text)
            formatted = json.dumps(parsed, indent=indent, sort_keys=True)
            
            return {
                "success": True,
                "original": json_text,
                "formatted": formatted
            }
        except Exception as e:
            return {"error": f"JSON formatting failed: {str(e)}"}
    
    def _validate_email(self, email: str) -> Dict[str, Any]:
        """Validate email address format."""
        import re
        
        # Basic email regex pattern
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        is_valid = bool(re.match(email_pattern, email))
        
        return {
            "success": True,
            "email": email,
            "valid": is_valid,
            "reason": "Valid email format" if is_valid else "Invalid email format"
        }
    
    def _generate_password(self, length: int = 12, include_symbols: bool = True) -> Dict[str, Any]:
        """Generate a random password."""
        try:
            import random
            import string
            
            if length < 4 or length > 128:
                return {"error": "Password length must be between 4 and 128 characters"}
            
            characters = string.ascii_letters + string.digits
            if include_symbols:
                characters += "!@#$%^&*"
            
            password = ''.join(random.choice(characters) for _ in range(length))
            
            return {
                "success": True,
                "password": password,
                "length": length,
                "includes_symbols": include_symbols
            }
        except Exception as e:
            return {"error": f"Password generation failed: {str(e)}"}


# Example usage and testing
async def test_tool_plugins():
    """Test the tool plugins."""
    print("=== Testing Tool Plugins ===\n")
    
    config = Configuration({
        "filesystem_allowed_paths": [str(Path.cwd()), "/tmp"],
        "http_allowed_domains": ["httpbin.org", "api.github.com"]
    })
    
    # Create and initialize plugins
    fs_plugin = FileSystemToolPlugin()
    http_plugin = HTTPToolPlugin()
    util_plugin = UtilityToolPlugin()
    
    await fs_plugin.enable(config)
    await http_plugin.enable(config)
    await util_plugin.enable(config)
    
    print("All plugins initialized\n")
    
    # Test filesystem tools
    print("=== Filesystem Tools ===")
    
    # List current directory
    result = await fs_plugin.execute_tool("list_directory", dir_path=".")
    print(f"List directory: {result.get('count', 0)} items found")
    
    # Create a test file
    test_content = "This is a test file created by the filesystem plugin."
    result = await fs_plugin.execute_tool("write_file", file_path="test_plugin_file.txt", content=test_content)
    print(f"Write file: {'Success' if result.get('success') else 'Failed'}")
    
    # Read the test file
    result = await fs_plugin.execute_tool("read_file", file_path="test_plugin_file.txt")
    print(f"Read file: {'Success' if result.get('success') else 'Failed'}")
    
    # Get file info
    result = await fs_plugin.execute_tool("get_file_info", file_path="test_plugin_file.txt")
    print(f"File info: {result.get('size', 0)} bytes")
    
    print()
    
    # Test HTTP tools (if network is available)
    print("=== HTTP Tools ===")
    
    try:
        # Check a URL
        result = await http_plugin.execute_tool("check_url", url="https://httpbin.org/status/200")
        print(f"URL check: {'Accessible' if result.get('accessible') else 'Not accessible'}")
        
        # Make a GET request
        result = await http_plugin.execute_tool("http_get", url="https://httpbin.org/json")
        print(f"HTTP GET: Status {result.get('status_code', 'N/A')}")
    except Exception as e:
        print(f"HTTP tests skipped (no network): {e}")
    
    print()
    
    # Test utility tools
    print("=== Utility Tools ===")
    
    # Base64 encoding
    result = await util_plugin.execute_tool("encode_base64", text="Hello, World!")
    print(f"Base64 encode: {result.get('encoded', 'Failed')[:20]}...")
    
    # Generate UUID
    result = await util_plugin.execute_tool("generate_uuid")
    print(f"Generate UUID: {result.get('uuid', 'Failed')}")
    
    # Hash text
    result = await util_plugin.execute_tool("hash_text", text="test", algorithm="sha256")
    print(f"SHA256 hash: {result.get('hash', 'Failed')[:16]}...")
    
    # Generate password
    result = await util_plugin.execute_tool("generate_password", length=16)
    print(f"Generate password: {result.get('password', 'Failed')}")
    
    # Validate email
    result = await util_plugin.execute_tool("validate_email", email="test@example.com")
    print(f"Email validation: {'Valid' if result.get('valid') else 'Invalid'}")
    
    print()
    
    # Cleanup
    await fs_plugin.disable()
    await http_plugin.disable()
    await util_plugin.disable()
    
    # Clean up test file
    try:
        Path("test_plugin_file.txt").unlink()
        print("Test file cleaned up")
    except:
        pass
    
    print("\nAll tool plugins tested successfully!")


if __name__ == "__main__":
    asyncio.run(test_tool_plugins())