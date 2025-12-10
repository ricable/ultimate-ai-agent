# Test-Swarm API Specification

## API Overview

**Base URL**: `https://api.test-swarm.com/v1`  
**Authentication**: Bearer token (JWT)  
**Content-Type**: `application/json`  
**API Version**: 1.0.0

## Authentication

All API requests (except public endpoints) require a valid JWT token in the Authorization header:

```
Authorization: Bearer <jwt-token>
```

## Common Response Headers

```http
X-Request-ID: uuid-v4
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 99
X-RateLimit-Reset: 1640995200
X-Response-Time: 125ms
```

## Status Codes

| Code | Meaning |
|------|---------|
| 200 | OK - Request succeeded |
| 201 | Created - Resource created successfully |
| 204 | No Content - Request succeeded with no response body |
| 400 | Bad Request - Invalid request data |
| 401 | Unauthorized - Missing or invalid authentication |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource not found |
| 409 | Conflict - Resource already exists |
| 422 | Unprocessable Entity - Validation failed |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

## Endpoints

### Authentication Endpoints

#### 1. Register User

Create a new user account.

**Endpoint**: `POST /auth/register`  
**Authentication**: Not required  
**Rate Limit**: 5 requests per hour per IP

**Request Body**:
```json
{
    "username": "johndoe",
    "email": "john@example.com",
    "password": "SecurePass123!"
}
```

**Validation Rules**:
- `username`: 3-50 characters, alphanumeric and underscore only
- `email`: Valid email format
- `password`: Minimum 8 characters, must contain uppercase, lowercase, number, and special character

**Success Response** (201 Created):
```json
{
    "success": true,
    "data": {
        "user": {
            "id": 1,
            "username": "johndoe",
            "email": "john@example.com",
            "createdAt": "2024-01-01T00:00:00Z"
        },
        "tokens": {
            "accessToken": "eyJhbGciOiJSUzI1NiIs...",
            "refreshToken": "eyJhbGciOiJSUzI1NiIs...",
            "expiresIn": 900
        }
    },
    "meta": {
        "timestamp": "2024-01-01T00:00:00Z",
        "requestId": "550e8400-e29b-41d4-a716-446655440000"
    }
}
```

**Error Response** (409 Conflict):
```json
{
    "success": false,
    "error": {
        "code": "USER_EXISTS",
        "message": "Username or email already registered"
    }
}
```

#### 2. Login

Authenticate user and receive access tokens.

**Endpoint**: `POST /auth/login`  
**Authentication**: Not required  
**Rate Limit**: 10 requests per hour per IP

**Request Body**:
```json
{
    "username": "johndoe",
    "password": "SecurePass123!",
    "deviceFingerprint": "optional-device-id"
}
```

**Success Response** (200 OK):
```json
{
    "success": true,
    "data": {
        "user": {
            "id": 1,
            "username": "johndoe",
            "email": "john@example.com",
            "lastLoginAt": "2024-01-01T00:00:00Z"
        },
        "tokens": {
            "accessToken": "eyJhbGciOiJSUzI1NiIs...",
            "refreshToken": "eyJhbGciOiJSUzI1NiIs...",
            "expiresIn": 900
        }
    }
}
```

**Error Response** (401 Unauthorized):
```json
{
    "success": false,
    "error": {
        "code": "INVALID_CREDENTIALS",
        "message": "Invalid username or password",
        "remainingAttempts": 3
    }
}
```

#### 3. Refresh Token

Exchange refresh token for new access token.

**Endpoint**: `POST /auth/refresh`  
**Authentication**: Not required  
**Rate Limit**: 30 requests per hour per token

**Request Body**:
```json
{
    "refreshToken": "eyJhbGciOiJSUzI1NiIs..."
}
```

**Success Response** (200 OK):
```json
{
    "success": true,
    "data": {
        "tokens": {
            "accessToken": "eyJhbGciOiJSUzI1NiIs...",
            "refreshToken": "eyJhbGciOiJSUzI1NiIs...",
            "expiresIn": 900
        }
    }
}
```

#### 4. Logout

Invalidate current session and tokens.

**Endpoint**: `POST /auth/logout`  
**Authentication**: Required  
**Rate Limit**: Standard

**Request Body**:
```json
{
    "refreshToken": "eyJhbGciOiJSUzI1NiIs..."
}
```

**Success Response** (204 No Content):
```
(empty body)
```

### User Management Endpoints

#### 1. Get Current User

Retrieve authenticated user's profile.

**Endpoint**: `GET /users/me`  
**Authentication**: Required  
**Rate Limit**: Standard

**Success Response** (200 OK):
```json
{
    "success": true,
    "data": {
        "id": 1,
        "username": "johndoe",
        "email": "john@example.com",
        "createdAt": "2024-01-01T00:00:00Z",
        "updatedAt": "2024-01-01T00:00:00Z",
        "isActive": true,
        "lastLoginAt": "2024-01-01T00:00:00Z"
    }
}
```

#### 2. Update User Profile

Update authenticated user's information.

**Endpoint**: `PUT /users/me`  
**Authentication**: Required  
**Rate Limit**: Standard

**Request Body**:
```json
{
    "email": "newemail@example.com",
    "currentPassword": "CurrentPass123!",
    "newPassword": "NewSecurePass456!"
}
```

**Validation Rules**:
- `currentPassword`: Required when changing password or email
- `newPassword`: Same rules as registration password

**Success Response** (200 OK):
```json
{
    "success": true,
    "data": {
        "id": 1,
        "username": "johndoe",
        "email": "newemail@example.com",
        "updatedAt": "2024-01-01T00:00:00Z"
    }
}
```

#### 3. Delete User Account

Permanently delete user account.

**Endpoint**: `DELETE /users/me`  
**Authentication**: Required  
**Rate Limit**: 1 request per day

**Request Body**:
```json
{
    "password": "CurrentPass123!",
    "confirmation": "DELETE_MY_ACCOUNT"
}
```

**Success Response** (204 No Content):
```
(empty body)
```

#### 4. List Users (Admin Only)

Retrieve paginated list of users.

**Endpoint**: `GET /users`  
**Authentication**: Required (Admin role)  
**Rate Limit**: Standard

**Query Parameters**:
- `cursor`: Base64 encoded cursor for pagination
- `limit`: Number of results (1-100, default: 20)
- `sort`: Sort field and direction (e.g., `createdAt:desc`)
- `filter`: JSON encoded filter object

**Example Request**:
```
GET /users?limit=20&sort=createdAt:desc&filter={"isActive":true}
```

**Success Response** (200 OK):
```json
{
    "success": true,
    "data": [
        {
            "id": 1,
            "username": "johndoe",
            "email": "john@example.com",
            "createdAt": "2024-01-01T00:00:00Z",
            "isActive": true
        }
    ],
    "pagination": {
        "cursor": "eyJpZCI6MTAsImNyZWF0ZWRBdCI6IjIwMjQtMDEtMDEifQ==",
        "hasMore": true,
        "total": 150
    }
}
```

### Session Management Endpoints

#### 1. List Active Sessions

Get all active sessions for authenticated user.

**Endpoint**: `GET /sessions`  
**Authentication**: Required  
**Rate Limit**: Standard

**Success Response** (200 OK):
```json
{
    "success": true,
    "data": [
        {
            "id": 1,
            "deviceFingerprint": "device-123",
            "ipAddress": "192.168.1.1",
            "userAgent": "Mozilla/5.0...",
            "createdAt": "2024-01-01T00:00:00Z",
            "lastUsedAt": "2024-01-01T00:00:00Z",
            "isCurrent": true
        }
    ]
}
```

#### 2. Revoke Session

Terminate a specific session.

**Endpoint**: `DELETE /sessions/:sessionId`  
**Authentication**: Required  
**Rate Limit**: Standard

**Success Response** (204 No Content):
```
(empty body)
```

#### 3. Revoke All Sessions

Terminate all sessions except current.

**Endpoint**: `POST /sessions/revoke-all`  
**Authentication**: Required  
**Rate Limit**: 5 requests per hour

**Success Response** (200 OK):
```json
{
    "success": true,
    "data": {
        "revokedCount": 3
    }
}
```

### Health & Monitoring Endpoints

#### 1. Health Check

Basic health check endpoint.

**Endpoint**: `GET /health`  
**Authentication**: Not required  
**Rate Limit**: Exempt

**Success Response** (200 OK):
```json
{
    "success": true,
    "data": {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "version": "1.0.0",
        "uptime": 86400
    }
}
```

#### 2. Detailed Health Status

Comprehensive system health information.

**Endpoint**: `GET /health/detailed`  
**Authentication**: Required (Admin role)  
**Rate Limit**: Standard

**Success Response** (200 OK):
```json
{
    "success": true,
    "data": {
        "status": "healthy",
        "components": {
            "database": {
                "status": "healthy",
                "responseTime": 5,
                "connections": {
                    "active": 10,
                    "idle": 40,
                    "max": 50
                }
            },
            "cache": {
                "status": "healthy",
                "hitRate": 0.95,
                "memoryUsage": "124MB"
            }
        },
        "metrics": {
            "requestsPerMinute": 150,
            "averageResponseTime": 125,
            "errorRate": 0.002,
            "cpuUsage": 45.2,
            "memoryUsage": 512
        }
    }
}
```

#### 3. Performance Metrics

Retrieve performance metrics.

**Endpoint**: `GET /metrics`  
**Authentication**: Required (Admin role)  
**Rate Limit**: Standard

**Query Parameters**:
- `startTime`: ISO 8601 timestamp
- `endTime`: ISO 8601 timestamp
- `interval`: Aggregation interval (1m, 5m, 1h, 1d)
- `endpoint`: Filter by specific endpoint

**Success Response** (200 OK):
```json
{
    "success": true,
    "data": {
        "summary": {
            "totalRequests": 10000,
            "averageResponseTime": 125,
            "p50ResponseTime": 100,
            "p95ResponseTime": 250,
            "p99ResponseTime": 500,
            "errorRate": 0.002
        },
        "timeSeries": [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "requests": 150,
                "avgResponseTime": 120,
                "errors": 0
            }
        ],
        "topEndpoints": [
            {
                "endpoint": "GET /users/me",
                "requests": 2500,
                "avgResponseTime": 50
            }
        ]
    }
}
```

### Rate Limiting

Rate limits are enforced per user (authenticated) or per IP (unauthenticated):

| Category | Authenticated | Unauthenticated |
|----------|---------------|-----------------|
| Standard | 1000/hour | 100/hour |
| Auth | 100/hour | 10/hour |
| Sensitive | 10/hour | 5/hour |

When rate limited, the API returns:

**429 Too Many Requests**:
```json
{
    "success": false,
    "error": {
        "code": "RATE_LIMIT_EXCEEDED",
        "message": "Too many requests",
        "retryAfter": 3600
    }
}
```

## WebSocket Events (Future)

### Connection

```javascript
const ws = new WebSocket('wss://api.test-swarm.com/v1/ws');
ws.send(JSON.stringify({
    type: 'auth',
    token: 'jwt-token'
}));
```

### Event Types

| Event | Direction | Description |
|-------|-----------|-------------|
| `auth` | Client→Server | Authenticate connection |
| `ping` | Client↔Server | Keep connection alive |
| `user.updated` | Server→Client | User profile changed |
| `session.created` | Server→Client | New session created |
| `session.revoked` | Server→Client | Session terminated |

## Error Codes Reference

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Input validation failed |
| `AUTHENTICATION_REQUIRED` | Missing authentication |
| `INVALID_CREDENTIALS` | Wrong username/password |
| `TOKEN_EXPIRED` | JWT token expired |
| `TOKEN_INVALID` | Malformed or invalid token |
| `INSUFFICIENT_PERMISSIONS` | User lacks required role |
| `RESOURCE_NOT_FOUND` | Requested resource doesn't exist |
| `RESOURCE_EXISTS` | Resource already exists |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `INTERNAL_ERROR` | Server error |
| `SERVICE_UNAVAILABLE` | Service temporarily down |

## SDK Examples

### JavaScript/TypeScript

```typescript
import { TestSwarmClient } from '@test-swarm/sdk';

const client = new TestSwarmClient({
    baseUrl: 'https://api.test-swarm.com/v1',
    timeout: 5000
});

// Login
const { tokens } = await client.auth.login({
    username: 'johndoe',
    password: 'SecurePass123!'
});

// Set auth token
client.setAuthToken(tokens.accessToken);

// Get user profile
const user = await client.users.getMe();

// Update profile
await client.users.updateMe({
    email: 'newemail@example.com'
});
```

### cURL Examples

```bash
# Register
curl -X POST https://api.test-swarm.com/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"johndoe","email":"john@example.com","password":"SecurePass123!"}'

# Login
curl -X POST https://api.test-swarm.com/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"johndoe","password":"SecurePass123!"}'

# Get profile
curl -X GET https://api.test-swarm.com/v1/users/me \
  -H "Authorization: Bearer <jwt-token>"
```

## API Versioning

The API uses URL versioning. The current version is `v1`. When breaking changes are introduced:

1. New version endpoint created (e.g., `/v2`)
2. Previous version maintained for 12 months
3. Deprecation warnings added to headers
4. Migration guide published

## Security Best Practices

1. **Always use HTTPS** in production
2. **Store tokens securely** (never in localStorage for web apps)
3. **Implement token refresh** before expiration
4. **Validate all inputs** on client side
5. **Handle rate limits** gracefully
6. **Monitor for suspicious activity**
7. **Rotate tokens regularly**
8. **Use device fingerprinting** for enhanced security