# JWT Authentication for SAFLA MCP Server

## Overview

The SAFLA MCP Server now includes JWT (JSON Web Token) authentication to secure access to tools and resources. This authentication system provides:

- Token-based authentication
- Role-based access control (RBAC)
- Permission-based authorization
- Session management
- Token refresh mechanism

## Configuration

### Environment Variables

Configure JWT authentication by setting the following environment variables:

```bash
# Required - JWT secret key for token signing
JWT_SECRET_KEY=your-secret-key-here-change-in-production

# Optional - Token expiration time in minutes (default: 60)
JWT_EXPIRATION_TIME=60

# Optional - Refresh token expiration in days (default: 7)
JWT_REFRESH_EXPIRATION_DAYS=7

# Optional - JWT algorithm (default: HS256)
JWT_ALGORITHM=HS256

# Optional - Enable/disable authentication (default: true)
MCP_AUTH_ENABLED=true
```

### Generating a Secure Secret Key

For production use, generate a secure secret key:

```bash
# Using OpenSSL
openssl rand -base64 32

# Using Python
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Authentication Flow

### 1. Login

Authenticate with username and password to receive tokens:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "auth/login",
  "params": {
    "username": "developer",
    "password": "dev123"
  }
}
```

Response:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "success": true,
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
    "expires_in": 3600,
    "token_type": "Bearer",
    "user": {
      "id": "user_002",
      "username": "developer",
      "roles": ["developer"],
      "permissions": ["tools:read", "tools:execute", "resources:read", "resources:write"]
    }
  }
}
```

### 2. Using the Access Token

Include the access token in subsequent requests:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "auth_token": "eyJhbGciOiJIUzI1NiIs...",
    "name": "validate_installation",
    "arguments": {}
  }
}
```

Alternatively, use the Authorization header format:
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "headers": {
      "Authorization": "Bearer eyJhbGciOiJIUzI1NiIs..."
    },
    "name": "validate_installation",
    "arguments": {}
  }
}
```

### 3. Refreshing Tokens

When the access token expires, use the refresh token to get a new one:

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "auth/refresh",
  "params": {
    "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
  }
}
```

### 4. Logout

Invalidate the current session:

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "auth/logout",
  "params": {
    "token": "eyJhbGciOiJIUzI1NiIs..."
  }
}
```

## User Roles and Permissions

### Default Users (Demo)

The system includes demo users for testing:

| Username | Password | Role | Permissions |
|----------|----------|------|-------------|
| admin | admin123 | admin | * (all permissions) |
| developer | dev123 | developer | tools:read, tools:execute, resources:read, resources:write |
| reader | read123 | reader | tools:read, resources:read |

### Permissions

- `tools:read` - List and view available tools
- `tools:execute` - Execute tools
- `resources:read` - List and read resources
- `resources:write` - Create and modify resources
- `resources:delete` - Delete resources
- `*` - All permissions (admin only)

### Public Methods

The following methods do not require authentication:

- `initialize` - Initialize MCP connection
- `auth/login` - Login endpoint
- `auth/refresh` - Token refresh endpoint
- `health/check` - Health check endpoint

## Integration Guide

### Python Client Example

```python
import json
import sys

class MCPAuthClient:
    def __init__(self):
        self.access_token = None
        self.refresh_token = None
    
    def send_request(self, method, params=None):
        """Send a request to the MCP server."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {}
        }
        
        # Add auth token if available and not a public method
        if self.access_token and method not in ["initialize", "auth/login", "auth/refresh"]:
            request["params"]["auth_token"] = self.access_token
        
        # Send to stdout
        print(json.dumps(request))
        sys.stdout.flush()
        
        # Read response from stdin
        response = json.loads(input())
        return response
    
    def login(self, username, password):
        """Login and store tokens."""
        response = self.send_request("auth/login", {
            "username": username,
            "password": password
        })
        
        if response.get("result", {}).get("success"):
            self.access_token = response["result"]["access_token"]
            self.refresh_token = response["result"]["refresh_token"]
            return True
        return False
    
    def call_tool(self, tool_name, arguments):
        """Call a tool with authentication."""
        return self.send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })

# Usage
client = MCPAuthClient()
if client.login("developer", "dev123"):
    result = client.call_tool("validate_installation", {})
    print(result)
```

### Error Handling

Authentication errors return specific error codes:

- `-32000` - Authentication required or failed
- `-32001` - Token expired
- `-32002` - Invalid token
- `-32003` - Insufficient permissions

Example error response:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32000,
    "message": "Authentication required: No authentication token provided"
  }
}
```

## Security Best Practices

1. **Secret Key Management**
   - Never commit the JWT secret key to version control
   - Use environment variables or secure key management services
   - Rotate keys periodically

2. **Token Expiration**
   - Keep access token expiration short (15-60 minutes)
   - Use refresh tokens for long-term access
   - Implement token revocation for logout

3. **HTTPS/TLS**
   - Always use encrypted connections in production
   - Never transmit tokens over unencrypted channels

4. **Permission Validation**
   - Implement least privilege principle
   - Validate permissions on every request
   - Log authentication attempts and failures

5. **Production Deployment**
   - Use strong, randomly generated secret keys
   - Enable rate limiting for login attempts
   - Monitor for suspicious authentication patterns
   - Implement account lockout policies

## Troubleshooting

### Common Issues

1. **"JWT_SECRET_KEY must be set"**
   - Ensure the JWT_SECRET_KEY environment variable is set
   - Check .env file is loaded correctly

2. **"Token has expired"**
   - Use the refresh token to get a new access token
   - Check system time synchronization

3. **"Invalid token"**
   - Ensure token is correctly formatted
   - Verify the secret key matches between token generation and validation

4. **"Insufficient permissions"**
   - Check user roles and permissions
   - Verify the method requires the permissions you have

### Debug Mode

Enable debug logging for authentication:

```bash
SAFLA_LOG_LEVEL=DEBUG
SAFLA_DEBUG=true
```

This will log detailed authentication information including:
- Token validation attempts
- Permission checks
- Session management operations

## API Reference

### Authentication Methods

#### auth/login
Authenticate user and receive tokens.

**Parameters:**
- `username` (string, required): User's username
- `password` (string, required): User's password

**Returns:**
- `access_token`: JWT access token
- `refresh_token`: JWT refresh token
- `expires_in`: Token expiration time in seconds
- `user`: User information with roles and permissions

#### auth/refresh
Refresh an expired access token.

**Parameters:**
- `refresh_token` (string, required): Valid refresh token

**Returns:**
- `access_token`: New JWT access token
- `expires_in`: Token expiration time in seconds

#### auth/logout
Logout and invalidate session.

**Parameters:**
- `token` (string, required): Current access token

**Returns:**
- `success`: Boolean indicating logout success
- `message`: Logout status message

#### auth/validate
Validate a token and check session status.

**Parameters:**
- `token` (string, required): Token to validate

**Returns:**
- `valid`: Boolean indicating token validity
- `user_id`: User identifier
- `roles`: User roles
- `permissions`: User permissions
- `expires_at`: Token expiration timestamp
- `session_active`: Session status