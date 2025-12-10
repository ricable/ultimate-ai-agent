# OAuth2 Authentication Flow - CRISPR-Cas13 Pipeline

## Overview

The CRISPR-Cas13 pipeline uses OAuth2 with JWT (JSON Web Tokens) for authentication and authorization. This document describes the complete authentication flow, token management, and security best practices.

## Authentication Architecture

```mermaid
sequenceDiagram
    actor User
    participant Client as Web/CLI Client
    participant API as API Gateway
    participant DB as PostgreSQL
    participant Redis as Redis Cache

    User->>Client: Enter credentials
    Client->>API: POST /api/v1/auth/login<br/>{email, password}

    API->>DB: SELECT user WHERE email=?
    DB-->>API: User record

    API->>API: Verify password (bcrypt)

    alt Password Valid
        API->>API: Generate JWT access token (24h TTL)
        API->>API: Generate refresh token (7d TTL)
        API->>Redis: Store session<br/>SET session:{token_hash}
        Redis-->>API: OK

        API->>DB: INSERT INTO audit_log (login)

        API-->>Client: 200 OK<br/>{token, refresh_token, user}
        Client->>Client: Store tokens
        Client-->>User: Redirect to dashboard
    else Password Invalid
        API-->>Client: 401 Unauthorized
        Client-->>User: Show error
    end
```

---

## Token Types

### 1. Access Token (JWT)

**Purpose**: Short-lived token for API authentication
**Lifetime**: 24 hours
**Storage**: Client-side (localStorage or secure cookie)

**Token Structure**:
```json
{
  "header": {
    "alg": "RS256",
    "typ": "JWT",
    "kid": "key-2025-01"
  },
  "payload": {
    "sub": "550e8400-e29b-41d4-a716-446655440000",
    "email": "researcher@example.com",
    "role": "researcher",
    "permissions": [
      "experiments:read",
      "experiments:create",
      "experiments:update:own",
      "jobs:submit",
      "results:read"
    ],
    "iat": 1705401600,
    "exp": 1705488000,
    "iss": "crispr-api.example.com",
    "aud": "crispr-clients"
  },
  "signature": "..."
}
```

**Signing Algorithm**: RS256 (RSA with SHA-256)
- **Public Key**: Distributed via JWKS endpoint (`/api/v1/auth/jwks.json`)
- **Private Key**: Stored in Kubernetes Secret, rotated quarterly

---

### 2. Refresh Token

**Purpose**: Long-lived token for obtaining new access tokens
**Lifetime**: 7 days
**Storage**: HttpOnly cookie (more secure than localStorage)

**Token Structure** (Opaque token):
```
{
  "jti": "unique-token-id",
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "issued_at": 1705401600,
  "expires_at": 1706006400
}
```

**Storage**: Redis with TTL
```redis
SET refresh_token:{token_hash} "{user_id: ..., issued_at: ...}" EX 604800
```

---

## Authentication Flows

### 1. Initial Login

```http
POST /api/v1/auth/login HTTP/1.1
Host: api.crispr.example.com
Content-Type: application/json

{
  "email": "researcher@example.com",
  "password": "SecurePassword123!"
}
```

**Success Response (200 OK)**:
```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIs...",
  "refresh_token": "550e8400-e29b-41d4-a716-446655440000",
  "token_type": "Bearer",
  "expires_in": 86400,
  "user": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "email": "researcher@example.com",
    "full_name": "Dr. Jane Smith",
    "role": "researcher",
    "institution": "Stanford University"
  }
}
```

**Error Response (401 Unauthorized)**:
```json
{
  "error": "invalid_credentials",
  "error_description": "Invalid email or password"
}
```

---

### 2. Authenticated API Request

```http
GET /api/v1/experiments HTTP/1.1
Host: api.crispr.example.com
Authorization: Bearer eyJhbGciOiJSUzI1NiIs...
```

**Validation Steps** (in API Gateway middleware):
1. Extract token from `Authorization: Bearer` header
2. Verify JWT signature using public key
3. Check token expiration (`exp` claim)
4. Validate issuer (`iss`) and audience (`aud`)
5. Extract user ID and permissions from token
6. Check required permission for route (e.g., `experiments:read`)
7. Inject user claims into request context
8. Forward request to route handler

---

### 3. Token Refresh

When access token expires (24 hours), client uses refresh token to obtain new access token:

```http
POST /api/v1/auth/refresh HTTP/1.1
Host: api.crispr.example.com
Content-Type: application/json

{
  "refresh_token": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Success Response (200 OK)**:
```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIs...",
  "refresh_token": "new-refresh-token-id",
  "token_type": "Bearer",
  "expires_in": 86400
}
```

**Error Response (401 Unauthorized)**:
```json
{
  "error": "invalid_token",
  "error_description": "Refresh token has expired or been revoked"
}
```

---

### 4. Logout

```http
POST /api/v1/auth/logout HTTP/1.1
Host: api.crispr.example.com
Authorization: Bearer eyJhbGciOiJSUzI1NiIs...
```

**Backend Actions**:
1. Extract user ID from JWT
2. Delete refresh token from Redis
3. Add access token to blacklist (Redis) with TTL = remaining token lifetime
4. Log logout event in audit log

**Client Actions**:
1. Delete access token from localStorage
2. Clear refresh token cookie
3. Redirect to login page

---

## Role-Based Access Control (RBAC)

### Roles & Permissions

| Role | Permissions | Use Case |
|------|-------------|----------|
| **Guest** | `experiments:read:public`<br/>`results:read:public` | Public demo accounts |
| **Researcher** | `experiments:*:own`<br/>`samples:*:own`<br/>`jobs:submit`<br/>`results:*:own` | Standard lab members |
| **Bioinformatician** | All researcher permissions +<br/>`pipelines:configure`<br/>`data:export`<br/>`experiments:read:all` | Data analysts |
| **Admin** | `*:*` (all permissions) | System administrators |

### Permission Naming Convention

```
resource:action:scope
```

**Examples**:
- `experiments:read:own` - Read own experiments
- `experiments:create` - Create experiments (implicit own scope)
- `jobs:submit` - Submit analysis jobs
- `users:update:all` - Update all users (admin only)

---

## Security Implementation

### Password Hashing

**Algorithm**: bcrypt with cost factor 12

```rust
// Hash password on registration
use bcrypt::{hash, DEFAULT_COST};
let password_hash = hash(password, DEFAULT_COST)?;

// Verify password on login
use bcrypt::verify;
let valid = verify(password, &stored_hash)?;
```

**Password Requirements**:
- Minimum 12 characters
- At least 1 uppercase letter
- At least 1 lowercase letter
- At least 1 number
- At least 1 special character
- Not in common password list (10,000 most common)

---

### JWT Signature Verification

**Public Key Distribution** (JWKS):
```http
GET /api/v1/auth/jwks.json HTTP/1.1
Host: api.crispr.example.com
```

**Response**:
```json
{
  "keys": [
    {
      "kty": "RSA",
      "use": "sig",
      "kid": "key-2025-01",
      "alg": "RS256",
      "n": "0vx7agoebGcQSu...",
      "e": "AQAB"
    }
  ]
}
```

**Key Rotation Policy**:
- New key generated every 90 days
- Old key valid for additional 30 days (grace period)
- Clients fetch JWKS on startup and cache for 24 hours

---

### Token Blacklist (Logout & Revocation)

**Redis Storage**:
```redis
# Add token to blacklist on logout
SET token_blacklist:{token_hash} "revoked" EX {remaining_ttl}

# Check blacklist on every request (fast O(1) lookup)
EXISTS token_blacklist:{token_hash}
```

**Performance**: Redis GET operation <1ms, negligible impact on request latency

---

### Rate Limiting (Brute Force Protection)

**Login Endpoint** (`POST /api/v1/auth/login`):
- **Limit**: 5 failed attempts per 15 minutes per email
- **Lockout**: After 5 failures, block for 15 minutes
- **Storage**: Redis with sliding window

**Implementation**:
```rust
async fn rate_limit_login(email: &str, redis: &redis::Client) -> Result<(), ApiError> {
    let key = format!("rate_limit:login:{}", email);
    let mut conn = redis.get_async_connection().await?;

    let count: u32 = conn.incr(&key, 1).await?;

    if count == 1 {
        conn.expire(&key, 900).await?; // 15 minutes
    }

    if count > 5 {
        return Err(ApiError::TooManyRequests("Too many login attempts. Try again in 15 minutes."));
    }

    Ok(())
}
```

---

## Security Best Practices

### 1. Token Storage

**✅ Recommended**:
- **Access Token**: localStorage (for SPA) or memory (for native apps)
- **Refresh Token**: HttpOnly cookie with `Secure` and `SameSite=Strict` flags

**❌ Not Recommended**:
- Storing refresh tokens in localStorage (vulnerable to XSS)
- Storing tokens in URL parameters (visible in logs)

---

### 2. HTTPS Only

**All API endpoints require TLS 1.3**:
- Certificate: Let's Encrypt wildcard cert for `*.crispr.example.com`
- Renewal: Automated via cert-manager in Kubernetes
- HSTS Header: `Strict-Transport-Security: max-age=31536000; includeSubDomains`

---

### 3. CORS Configuration

```rust
use tower_http::cors::{CorsLayer, Any};

let cors = CorsLayer::new()
    .allow_origin(["https://app.crispr.example.com".parse().unwrap()])
    .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
    .allow_headers([AUTHORIZATION, CONTENT_TYPE])
    .allow_credentials(true);
```

---

### 4. Token Expiration Times

| Token Type | Lifetime | Rationale |
|------------|----------|-----------|
| Access Token | 24 hours | Balance security vs. UX |
| Refresh Token | 7 days | Weekly re-authentication |
| API Key (automation) | 90 days | For CI/CD pipelines |

---

### 5. Audit Logging

**All authentication events are logged**:
```sql
INSERT INTO audit_log (user_id, action, ip_address, user_agent, metadata)
VALUES (
  '550e8400-e29b-41d4-a716-446655440000',
  'login',
  '203.0.113.42',
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
  '{"success": true, "method": "password"}'
);
```

**Monitored Events**:
- Successful login
- Failed login attempts
- Token refresh
- Logout
- Permission denied (403)
- Password changes
- Account deletions

---

## OAuth2 External Providers (Future)

**Planned Integrations**:
- **Google OAuth2**: For institutional accounts
- **GitHub OAuth2**: For open-source collaborators
- **ORCID**: For academic identity verification

**Flow** (Authorization Code with PKCE):
```http
GET /api/v1/auth/oauth/google/authorize HTTP/1.1
Host: api.crispr.example.com

# Redirect to Google OAuth consent screen
# After approval, redirect back with authorization code
# Exchange code for tokens
```

---

## Compliance & Standards

**Standards Followed**:
- ✅ **RFC 6749**: OAuth 2.0 Authorization Framework
- ✅ **RFC 7519**: JSON Web Token (JWT)
- ✅ **RFC 7517**: JSON Web Key (JWK)
- ✅ **OWASP Authentication Cheat Sheet**
- ✅ **NIST SP 800-63B**: Digital Identity Guidelines

**HIPAA Compliance**:
- All PHI (Personal Health Information) encrypted at rest and in transit
- Access logs retained for 7 years
- User sessions automatically expire after 24 hours of inactivity
- Multi-factor authentication (MFA) required for admin accounts

---

## Troubleshooting

### Issue: "Invalid token signature"
**Cause**: JWT signed with old key after rotation
**Solution**: Client should fetch new JWKS from `/api/v1/auth/jwks.json`

### Issue: "Token expired"
**Cause**: Access token TTL exceeded
**Solution**: Use refresh token to obtain new access token

### Issue: "Insufficient permissions"
**Cause**: User lacks required permission for resource
**Solution**: Admin must update user role or grant specific permission

---

**Document Version**: 1.0
**Last Updated**: 2025-10-12
**See Also**: [API Documentation](../api/README.md), [Security Audit Report](./security-audit.md)
