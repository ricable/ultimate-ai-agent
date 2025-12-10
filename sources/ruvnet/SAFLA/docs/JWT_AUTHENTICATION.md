# JWT Authentication for SAFLA MCP Server

## Overview

The SAFLA MCP Server implements JWT (JSON Web Token) authentication to secure access to tools and resources. This authentication system provides:

- Token-based authentication
- Role-based access control (RBAC)
- Refresh token support
- Configurable token expiration
- Public endpoint exceptions

## Configuration

### Environment Variables

```bash
# Required for JWT authentication
JWT_SECRET_KEY=your-secret-key-here

# Optional (defaults shown)
JWT_EXPIRATION_TIME=3600  # Access token expiration in seconds (1 hour)
```

### Disabling Authentication

If `JWT_SECRET_KEY` is not set, the server will start with authentication disabled and log a warning.

## Authentication Flow

### 1. Login

Send credentials to receive access and refresh tokens:

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
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
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

### 2. Authenticated Requests

Include the access token in request headers:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {
    "headers": {
      "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
    }
  }
}
```

Alternative: Include token in params:
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {
    "auth_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
  }
}
```

### 3. Token Refresh

Use the refresh token to get a new access token:

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "auth/refresh",
  "params": {
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
  }
}
```

Response:
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "success": true,
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "expires_in": 3600,
    "token_type": "Bearer"
  }
}
```

## Demo Users

For development and testing, the following demo users are available:

| Username | Password | Role | Permissions |
|----------|----------|------|-------------|
| admin | admin123 | admin | * (all permissions) |
| developer | dev123 | developer | tools:read, tools:execute, resources:read, resources:write |
| reader | read123 | reader | tools:read, resources:read |

## Public Endpoints

The following endpoints do not require authentication:

- `initialize` - MCP protocol initialization
- `auth/login` - User login
- `auth/refresh` - Token refresh
- `health/check` - Health check endpoint

## Role-Based Permissions

### Permission Model

Permissions follow the format `resource:action`:

- `tools:read` - List available tools
- `tools:execute` - Execute tools
- `resources:read` - Read resources
- `resources:write` - Write/modify resources
- `resources:delete` - Delete resources

### Role Permissions

- **admin**: Full access (`*`)
- **developer**: Read/execute tools, read/write resources
- **reader**: Read-only access to tools and resources

## Token Structure

### Access Token Payload

```json
{
  "sub": "user_002",              // User ID
  "exp": 1735920000,              // Expiration timestamp
  "iat": 1735916400,              // Issued at timestamp
  "jti": "550e8400-e29b-41d4-a716-446655440000",  // Token ID
  "roles": ["developer"],         // User roles
  "permissions": ["tools:read", "tools:execute"],  // Explicit permissions
  "metadata": {                   // Custom metadata
    "username": "developer"
  }
}
```

### Refresh Token

Refresh tokens have a longer expiration (7 days by default) and include:
```json
{
  "metadata": {
    "token_type": "refresh"
  }
}
```

## Error Handling

### Authentication Errors

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32000,
    "message": "Authentication required: No authorization header provided"
  }
}
```

Common error codes:
- `-32000`: Authentication error
- `-32601`: Method not found
- `-32603`: Internal error

### Token Expiration

When a token expires, you'll receive:
```json
{
  "error": {
    "code": -32000,
    "message": "Authentication required: Token has expired"
  }
}
```

Use the refresh token to get a new access token.

## Security Best Practices

1. **Secret Key Management**
   - Use a strong, random JWT secret key
   - Never commit the secret key to version control
   - Rotate keys periodically

2. **Token Storage**
   - Store tokens securely on the client side
   - Never log or expose tokens
   - Clear tokens on logout

3. **HTTPS Usage**
   - Always use HTTPS in production
   - Implement additional transport security for stdio communication

4. **Token Expiration**
   - Keep access token expiration short (1 hour default)
   - Use refresh tokens for long-lived sessions
   - Implement token revocation for compromised tokens

## Integration Example

```python
import json
import sys

# Login
login_request = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "auth/login",
    "params": {
        "username": "developer",
        "password": "dev123"
    }
}

# Send to MCP server via stdio
print(json.dumps(login_request))
response = json.loads(sys.stdin.readline())

# Extract token
access_token = response["result"]["access_token"]

# Use token in subsequent requests
tools_request = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list",
    "params": {
        "headers": {
            "Authorization": f"Bearer {access_token}"
        }
    }
}

print(json.dumps(tools_request))
```

## Troubleshooting

### Authentication Not Working

1. Check if `JWT_SECRET_KEY` is set:
   ```bash
   echo $JWT_SECRET_KEY
   ```

2. Verify server logs for authentication status:
   ```
   JWT authentication enabled
   # or
   JWT authentication disabled: JWT_SECRET_KEY must be set
   ```

3. Ensure token format is correct:
   - Use `Bearer` prefix in Authorization header
   - No extra spaces or newlines in token

### Token Validation Failures

1. Check token expiration
2. Verify JWT secret key matches between server restarts
3. Ensure token hasn't been modified

### Permission Denied

1. Check user roles and permissions
2. Verify method requires the permissions user has
3. Check server logs for detailed permission errors