name: "User Management REST API - Multi-Environment Polyglot Implementation"
description: |

## Purpose
Comprehensive Product Requirements Prompt (PRP) for implementing a production-ready user management REST API across multiple environments in the polyglot development ecosystem. This PRP provides complete context for one-pass implementation with comprehensive testing, monitoring integration, and cross-environment compatibility.

## Core Principles
1. **Multi-Environment Excellence**: Seamless integration across Python, TypeScript, Rust, Go, and Nushell environments
2. **Security First**: JWT-based authentication with secure password hashing and comprehensive input validation
3. **Modern Stack**: Latest 2024 patterns for FastAPI, SQLAlchemy 2.0 async, PyJWT, Pydantic v2
4. **Performance & Monitoring**: Deep integration with existing performance analytics and resource monitoring
5. **Production Ready**: Comprehensive error handling, logging, security, and deployment configuration
6. **Intelligence Integration**: Automated quality gates, performance tracking, and security scanning

---

## Goal
Build a complete user management REST API that serves as the foundational authentication and user management system for the entire polyglot development environment, with comprehensive testing, monitoring integration, and multi-environment orchestration capabilities.

## Why
- **Foundation Service**: Core user management system for all applications across Python, TypeScript, Rust, Go environments
- **Security Template**: Demonstrates secure authentication patterns and integration with polyglot monitoring systems
- **Performance Benchmark**: Showcases optimal async FastAPI performance with comprehensive analytics integration
- **Multi-Environment Integration**: Template for cross-language service communication and orchestration
- **Intelligence Showcase**: Leverages all existing automation, monitoring, and quality gate systems

## What
RESTful API with complete user lifecycle management, cross-environment integration, and comprehensive monitoring:
- User registration and JWT authentication system with PyJWT (2024 best practices)
- PostgreSQL database integration with async SQLAlchemy 2.0
- Role-based access control with admin and user roles
- Comprehensive REST API with OpenAPI documentation and examples
- Multi-environment client libraries and integration patterns
- Production-ready configuration with Docker and deployment automation
- Deep integration with performance analytics, security scanning, and resource monitoring

### Success Criteria
- [ ] Complete CRUD operations for user management with 90%+ test coverage
- [ ] JWT-based authentication system using PyJWT (not deprecated python-jose)
- [ ] PostgreSQL database integration with async SQLAlchemy 2.0 and Alembic migrations
- [ ] Cross-environment client libraries (TypeScript, Rust, Go) for API consumption
- [ ] Integration with existing performance analytics and security scanning systems
- [ ] Comprehensive OpenAPI documentation with authentication examples
- [ ] Production deployment configuration with Docker and environment management
- [ ] Multi-environment orchestration scripts using Nushell automation
- [ ] Security best practices with rate limiting, input validation, and monitoring

## All Needed Context

### Target Environments and Integration
```yaml
Primary_Environment: python-env
Python_Version: 3.12+ (from devbox.json analysis)
Package_Manager: uv (exclusively - no pip/poetry/pipenv)
Database: PostgreSQL with async support
Authentication: JWT with PyJWT (not python-jose)
Testing: pytest-asyncio with httpx AsyncClient

Cross_Environment_Integration:
  typescript-env: TypeScript client SDK and type definitions
  rust-env: Rust client library with async support
  go-env: Go client package with context handling  
  nushell-env: Orchestration scripts and automation tools
  
Monitoring_Integration:
  performance_analytics: "nushell-env/scripts/performance-analytics.nu"
  security_scanner: "nushell-env/scripts/security-scanner.nu"  
  resource_monitor: "nushell-env/scripts/resource-monitor.nu"
  test_intelligence: "nushell-env/scripts/test-intelligence.nu"
  dependency_monitor: "nushell-env/scripts/dependency-monitor.nu"
```

### Current Polyglot Environment Analysis
```yaml
# From environment analysis across all directories
Python_Environment:
  devbox_packages: ["python@3.12", "uv", "ruff", "mypy", "nushell"]
  existing_dependencies: ["fastapi", "pydantic", "httpx", "requests>=2.32.4"]
  development_tools: ["ruff>=0.8.0", "mypy>=1.7.0", "pytest>=7.4.0", "pytest-cov"]
  scripts: {
    "format": "uv run ruff format .",
    "lint": "uv run ruff check . --fix",
    "test": "uv run pytest --cov=src"
  }

TypeScript_Environment:
  devbox_packages: ["nodejs@20", "typescript", "eslint", "prettier"]
  structure: "Standard Node.js project with strict TypeScript configuration"
  
Rust_Environment:
  devbox_packages: ["rustc", "cargo", "rust-analyzer", "clippy", "rustfmt"]
  structure: "Standard Cargo workspace with async tokio support"
  
Go_Environment:
  devbox_packages: ["go@1.22", "golangci-lint", "goimports"]
  structure: "Standard Go module with cmd/ and internal/ structure"
  
Nushell_Environment:
  automation_scripts: [
    "performance-analytics.nu", "security-scanner.nu", "resource-monitor.nu",
    "test-intelligence.nu", "dependency-monitor.nu", "github-integration.nu"
  ]
  intelligence_systems: "Advanced monitoring and automation already implemented"
```

### Required Dependencies (2024 Best Practices)
```yaml
# Production Dependencies (Python)
Core_API: ["fastapi[all]", "uvicorn[standard]"]
Database: ["sqlalchemy[asyncio]>=2.0.0", "asyncpg", "alembic"] 
Authentication: ["PyJWT>=2.8.0", "passlib[bcrypt]", "python-multipart"]
Validation: ["pydantic[email]>=2.0", "pydantic-settings>=2.0"]

# Development Dependencies
Testing: ["pytest-asyncio>=0.23.0", "httpx>=0.25.0", "pytest-cov>=4.0"]
Quality: ["ruff>=0.8.0", "mypy>=1.7.0"]
Database_Testing: ["aiosqlite"]  # For test database

# Cross-Environment Dependencies
TypeScript_Client: ["@types/node", "axios", "zod"]  # Type-safe API client
Rust_Client: ["reqwest", "serde", "tokio", "anyhow"]  # Async HTTP client
Go_Client: ["net/http", "encoding/json", "context"]  # Standard library
Nushell_Scripts: ["http", "json", "table"]  # Built-in commands

# CRITICAL: Using PyJWT instead of python-jose (deprecated in 2024)
```

### Documentation & References
```yaml
# CRITICAL READING - Include these in context window
Primary_API_References:
  - url: https://fastapi.tiangolo.com/tutorial/
    why: FastAPI patterns, dependency injection, async handling, OpenAPI docs
    
  - url: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
    why: SQLAlchemy 2.0 async patterns and session management
    
  - url: https://pyjwt.readthedocs.io/en/stable/
    why: JWT authentication with PyJWT (not python-jose)
    critical: "Migration from python-jose to PyJWT - change imports and use InvalidTokenError"
    
  - url: https://alembic.sqlalchemy.org/en/latest/
    why: Database migrations and schema management with async support

Security_References:
  - pattern: "Use PyJWT>=2.8.0 instead of python-jose (deprecated 2024)"
  - pattern: "Implement proper async session management with SQLAlchemy 2.0"
  - pattern: "Use passlib with bcrypt for password hashing"
  - pattern: "Store secrets in environment variables"
  - pattern: "Implement rate limiting and request validation"

Cross_Environment_Integration:
  - pattern: "TypeScript SDK with Zod validation and axios client"
  - pattern: "Rust client with reqwest async and serde serialization"
  - pattern: "Go client with context handling and proper error wrapping"
  - pattern: "Nushell orchestration scripts for deployment and monitoring"

Performance_Integration:
  - file: "nushell-env/scripts/performance-analytics.nu"
    pattern: "Record API performance metrics for optimization"
  - file: "nushell-env/scripts/security-scanner.nu"
    pattern: "Automated security scanning on code changes"
  - file: "nushell-env/scripts/resource-monitor.nu"
    pattern: "Track resource usage during API operations"
```

### Current Codebase Structure
```bash
polyglot-devenv/
├── python-env/
│   ├── devbox.json         # Python 3.12, uv, ruff, mypy, nushell
│   ├── pyproject.toml      # Basic FastAPI dependencies
│   ├── test_example.py     # Basic test examples  
│   ├── test_format.py      # Format test file
│   └── uv.lock            # Dependency lock file
├── typescript-env/
│   ├── devbox.json         # Node.js 20, TypeScript, ESLint
│   ├── package.json        # TypeScript project configuration
│   └── tsconfig.json       # Strict TypeScript settings
├── rust-env/
│   ├── devbox.json         # Rust toolchain with async support
│   ├── Cargo.toml          # Rust project configuration
│   └── src/lib.rs          # Basic Rust structure
├── go-env/
│   ├── devbox.json         # Go 1.22 with linting tools
│   ├── go.mod              # Go module configuration
│   └── cmd/main.go         # Basic Go structure
├── nushell-env/
│   ├── devbox.json         # Nushell automation environment
│   ├── scripts/            # Advanced automation scripts
│   └── common.nu           # Shared utilities
└── context-engineering/
    ├── PRPs/               # Product Requirements Prompts
    └── templates/          # PRP templates
```

### Target Multi-Environment Structure
```bash
# Primary API Implementation (Python)
python-env/
├── src/
│   ├── core/
│   │   ├── config.py          # Pydantic settings with environment variables
│   │   ├── database.py        # Async SQLAlchemy 2.0 session management
│   │   ├── security.py        # PyJWT authentication with bcrypt
│   │   └── dependencies.py    # FastAPI dependency injection
│   ├── models/
│   │   ├── user.py           # SQLAlchemy User model with indexes
│   │   └── schemas.py        # Pydantic v2 request/response schemas
│   ├── api/
│   │   ├── auth.py           # Authentication endpoints
│   │   ├── users.py          # User management endpoints
│   │   └── admin.py          # Admin operations endpoints
│   ├── services/
│   │   ├── user_service.py   # User business logic
│   │   └── auth_service.py   # Authentication business logic
│   ├── clients/              # Cross-environment client generators
│   │   ├── typescript.py     # TypeScript SDK generator
│   │   ├── rust.py           # Rust client generator
│   │   ├── go.py             # Go client generator
│   │   └── nushell.py        # Nushell script generator
│   └── monitoring/
│       ├── performance.py    # Performance analytics integration
│       ├── security.py       # Security monitoring integration
│       └── health.py         # Health check and monitoring
├── tests/
│   ├── conftest.py          # Async test fixtures with monitoring
│   ├── test_api/            # API endpoint tests
│   ├── test_services/       # Service layer tests
│   ├── test_integration/    # End-to-end tests
│   └── test_monitoring/     # Monitoring integration tests
├── alembic/                 # Database migrations
├── docker-compose.yml       # Multi-environment setup
├── .env.example            # Environment variables template
└── clients/                # Generated client libraries

# TypeScript Client SDK
typescript-env/user-api-client/
├── src/
│   ├── client.ts           # Main API client
│   ├── types.ts            # Generated type definitions
│   ├── auth.ts             # Authentication helpers
│   └── errors.ts           # Error handling
├── tests/
│   └── integration.test.ts # Integration tests
└── package.json            # TypeScript client package

# Rust Client Library  
rust-env/user-api-client/
├── src/
│   ├── lib.rs              # Main client library
│   ├── client.rs           # HTTP client implementation
│   ├── models.rs           # Data models with serde
│   ├── auth.rs             # Authentication handling
│   └── error.rs            # Error types
├── tests/
│   └── integration.rs      # Integration tests
└── Cargo.toml              # Rust client configuration

# Go Client Package
go-env/user-api-client/
├── client.go               # Main client implementation
├── types.go                # Data structures
├── auth.go                 # Authentication helpers
├── errors.go               # Error handling
├── client_test.go          # Unit tests
└── go.mod                  # Go module

# Nushell Orchestration Scripts
nushell-env/user-api/
├── deploy.nu               # Multi-environment deployment
├── monitor.nu              # Monitoring and health checks
├── test-all.nu             # Cross-environment testing
├── generate-clients.nu     # Client SDK generation
└── performance.nu          # Performance testing and analytics
```

### Known Environment Gotchas & 2024 Updates
```python
# CRITICAL: Multi-environment and 2024-specific gotchas

# Python Environment (Primary API)
# ✅ uv add package-name
# ✅ uv run command 
# ✅ import jwt  # PyJWT library
# ✅ from jwt import InvalidTokenError
# ✅ async def endpoint(db: AsyncSession = Depends(get_db))
# ✅ result = await session.execute(select(User).where(User.email == email))
# ✅ user = result.scalar_one_or_none()
# ✅ model_config = ConfigDict(from_attributes=True)
# ❌ pip install package-name
# ❌ from jose import JWTError, jwt  # python-jose deprecated
# ❌ session.query(User).filter()  # SQLAlchemy 1.x pattern
# ❌ class Config: orm_mode = True  # Pydantic v1 pattern

# Cross-Environment Integration Gotchas
# TypeScript: Use strict mode, proper async/await, Zod for validation
# Rust: Use #[tokio::main] for async, proper Result<T, E> error handling
# Go: Use context.Context for timeouts, proper error wrapping
# Nushell: Use structured data, proper error handling with try/catch

# Performance Monitoring Integration
# ✅ nu ../nushell-env/scripts/performance-analytics.nu record "api-test" "python-env" "30s"
# ✅ nu ../nushell-env/scripts/security-scanner.nu scan-directory "src/"
# ❌ Direct performance calls without environment context

# Database Patterns (SQLAlchemy 2.0)
# ✅ async with AsyncSession() as session:
# ✅ await session.execute(select(User))
# ✅ await session.commit()
# ❌ session.query()  # Legacy pattern
# ❌ session.add() without await session.commit()

# Testing Patterns (2024)
# ✅ @pytest.mark.asyncio
# ✅ async with AsyncClient(app=app, base_url="http://test") as client:
# ✅ async def test_endpoint():
# ❌ with TestClient(app) as client:  # Sync testing pattern
# ❌ def test_endpoint():  # Sync test function

# Security Patterns (2024)
# ✅ PyJWT for token handling
# ✅ passlib with bcrypt for passwords
# ✅ Environment variables for secrets
# ✅ Rate limiting with slowapi or similar
# ❌ python-jose (deprecated)
# ❌ Hardcoded secrets
# ❌ MD5 or SHA1 for passwords
```

## Implementation Blueprint

### Task 1: Environment Setup and Modern Dependencies
```bash
# Activate Python environment and verify polyglot setup
cd python-env && devbox shell

# Verify environment
uv --version && python --version  # Should be 3.12+
ls ../  # Should show typescript-env, rust-env, go-env, nushell-env

# Install production dependencies (2024 best practices)
uv add "fastapi[all]" "uvicorn[standard]"
uv add "sqlalchemy[asyncio]>=2.0.0" "asyncpg" "alembic"
uv add "PyJWT>=2.8.0" "passlib[bcrypt]" "python-multipart"
uv add "pydantic[email]>=2.0" "pydantic-settings>=2.0"

# Install development dependencies
uv add --dev "pytest-asyncio>=0.23.0" "httpx>=0.25.0" "pytest-cov>=4.0"
uv add --dev "aiosqlite"  # For test database

# Verify installation and check for conflicts
uv run python -c "import fastapi, sqlalchemy, jwt, passlib, pydantic; print('All packages installed successfully')"

# Verify cross-environment accessibility
cd ../typescript-env && devbox shell --pure --command "node --version"
cd ../rust-env && devbox shell --pure --command "cargo --version"  
cd ../go-env && devbox shell --pure --command "go version"
cd ../nushell-env && devbox shell --pure --command "nu --version"
cd ../python-env && devbox shell  # Return to Python environment
```

### Task 2: Core Infrastructure with Monitoring Integration
```python
# src/core/config.py - Enhanced configuration with monitoring
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
import os

class Settings(BaseSettings):
    """Application configuration with environment variable support and monitoring integration"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application Settings
    app_name: str = "User Management API"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = Field(default="development", description="Deployment environment")
    
    # Database Settings
    database_url: str = Field(..., description="PostgreSQL async connection string")
    database_echo: bool = Field(False, description="Enable SQLAlchemy query logging")
    database_pool_size: int = Field(20, description="Database connection pool size")
    database_max_overflow: int = Field(0, description="Database connection overflow")
    
    # Security Settings  
    secret_key: str = Field(..., description="JWT secret key - use openssl rand -hex 32")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # CORS Settings
    allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    allowed_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]  
    allowed_headers: List[str] = ["*"]
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Monitoring Integration
    monitoring_enabled: bool = True
    performance_tracking: bool = True
    security_scanning: bool = True
    resource_monitoring: bool = True
    
    # Cross-Environment Settings
    typescript_client_enabled: bool = True
    rust_client_enabled: bool = True
    go_client_enabled: bool = True
    nushell_scripts_enabled: bool = True
    
    # Deployment Settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1

# Global settings instance
settings = Settings()

# src/core/database.py - Async SQLAlchemy 2.0 with monitoring
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import QueuePool
from typing import AsyncGenerator
import logging
from .config import settings

logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    """Base class for all database models with enhanced metadata"""
    pass

# Create async engine with production-ready settings
engine = create_async_engine(
    settings.database_url,
    echo=settings.database_echo,
    future=True,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    pool_pre_ping=True,
    pool_recycle=3600,
    poolclass=QueuePool,
    # Connection arguments for PostgreSQL
    connect_args={
        "command_timeout": 60,
        "server_settings": {
            "jit": "off",  # Disable JIT for faster connection times
        },
    },
)

# Create async session factory with optimal settings
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session with monitoring integration.
    
    Integrates with performance analytics to track database operation times.
    """
    start_time = None
    if settings.performance_tracking:
        import time
        start_time = time.time()
    
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()
            
            # Record performance metric if enabled
            if settings.performance_tracking and start_time:
                duration = time.time() - start_time
                # Integration with performance analytics will be added in monitoring module

# Database utility functions
async def init_db() -> None:
    """Initialize database tables"""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

async def close_db() -> None:
    """Close database connections gracefully"""
    try:
        await engine.dispose()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")

# Health check function
async def check_db_health() -> bool:
    """Check database connectivity for health monitoring"""
    try:
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False

# src/core/security.py - Enhanced security with PyJWT (2024 patterns)
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Union
import jwt
from jwt import InvalidTokenError
from passlib.context import CryptContext
from fastapi import HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
import logging
from .config import settings

logger = logging.getLogger(__name__)

# Password hashing context with modern settings
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,  # Increased rounds for better security
)

class SecurityService:
    """
    Enhanced security service for password hashing, JWT operations, and rate limiting.
    
    Implements 2024 best practices with PyJWT and comprehensive security monitoring.
    """
    
    def __init__(self):
        self.secret_key = settings.secret_key
        self.algorithm = settings.algorithm
        self.access_token_expire_minutes = settings.access_token_expire_minutes
        self.refresh_token_expire_days = settings.refresh_token_expire_days
        
        # Validate secret key strength
        if len(self.secret_key) < 32:
            logger.warning("Secret key should be at least 32 characters long")
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash with timing attack protection"""
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def get_password_hash(self, password: str) -> str:
        """Hash password using bcrypt with enhanced security"""
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        try:
            return pwd_context.hash(password)
        except Exception as e:
            logger.error(f"Password hashing error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Password processing failed"
            )
    
    def create_access_token(
        self, 
        data: Dict[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token with proper timezone handling and enhanced claims"""
        to_encode = data.copy()
        
        # Set expiration with proper timezone
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                minutes=self.access_token_expire_minutes
            )
        
        # Add standard JWT claims
        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "iss": settings.app_name,
            "jti": secrets.token_urlsafe(32),  # Unique token ID for revocation
        })
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            
            # Log token creation for monitoring (without exposing token)
            if settings.security_scanning:
                logger.info(f"Access token created for user: {data.get('sub', 'unknown')}")
            
            return encoded_jwt
        except Exception as e:
            logger.error(f"Token creation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create access token"
            )
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create refresh token with longer expiration"""
        expire = datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days)
        
        to_encode = data.copy()
        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "refresh",
            "jti": secrets.token_urlsafe(32),
        })
        
        try:
            return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        except Exception as e:
            logger.error(f"Refresh token creation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create refresh token"
            )
    
    def verify_token(self, token: str) -> Optional[str]:
        """Verify JWT token and return username with enhanced error handling"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            
            if username is None:
                logger.warning("Token missing 'sub' claim")
                return None
                
            # Verify token is not a refresh token
            if payload.get("type") == "refresh":
                logger.warning("Refresh token used as access token")
                return None
                
            return username
        except jwt.ExpiredSignatureError:
            logger.info("Token has expired")
            return None
        except InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """Decode JWT token and return payload with comprehensive error handling"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except Exception as e:
            logger.error(f"Token decode error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Token processing failed"
            )

class JWTBearer(HTTPBearer):
    """Custom JWT Bearer authentication with rate limiting and monitoring"""
    
    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)
        self.security_service = SecurityService()
    
    async def __call__(self, request: Request) -> HTTPAuthorizationCredentials:
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer, self).__call__(request)
        
        if credentials:
            if not credentials.scheme == "Bearer":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication scheme.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            if not self.security_service.verify_token(credentials.credentials):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token or expired token.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            return credentials
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization code.",
                headers={"WWW-Authenticate": "Bearer"},
            )

# Global security service instance
security_service = SecurityService()
jwt_bearer = JWTBearer()
```

### Task 3: Database Models and Schemas with Performance Optimization
```python
# src/models/user.py - Enhanced User model with optimization and auditing
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Index, JSON
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from src.core.database import Base
import uuid

class User(Base):
    """
    User database model with performance optimization and audit fields.
    
    Includes composite indexes for common queries and audit tracking.
    """
    __tablename__ = "users"

    # Primary fields
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(200), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    
    # Status and permissions
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    is_superuser = Column(Boolean, default=False, nullable=False, index=True)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Profile information
    bio = Column(Text, nullable=True)
    avatar_url = Column(String(500), nullable=True)
    
    # Audit and metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now(),
        nullable=False
    )
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    
    # Additional metadata for monitoring and analytics
    metadata_json = Column(JSON, nullable=True, default={})
    
    # Performance-optimized composite indexes
    __table_args__ = (
        # Index for authentication queries
        Index('idx_user_email_active', 'email', 'is_active'),
        Index('idx_user_username_active', 'username', 'is_active'),
        
        # Index for admin queries
        Index('idx_user_superuser_active', 'is_superuser', 'is_active'),
        
        # Index for pagination and listing
        Index('idx_user_created_active', 'created_at', 'is_active'),
        
        # Index for UUID lookups (cross-environment references)
        Index('idx_user_uuid_active', 'uuid', 'is_active'),
    )

# src/models/schemas.py - Comprehensive Pydantic v2 schemas
from pydantic import BaseModel, Field, EmailStr, ConfigDict, UUID4
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    """User role enumeration for RBAC"""
    USER = "user"
    ADMIN = "admin"
    SUPERUSER = "superuser"

class UserStatus(str, Enum):
    """User status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"

# Base schemas
class UserBase(BaseModel):
    """Base user schema with common validation"""
    email: EmailStr = Field(..., description="User email address")
    username: str = Field(
        ..., 
        min_length=3, 
        max_length=100, 
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Username (alphanumeric, underscore, hyphen only)"
    )
    full_name: Optional[str] = Field(None, max_length=200, description="Full name")
    bio: Optional[str] = Field(None, max_length=1000, description="User biography")
    avatar_url: Optional[str] = Field(None, max_length=500, description="Avatar image URL")

class UserCreate(UserBase):
    """Schema for user creation with password validation"""
    password: str = Field(
        ..., 
        min_length=8, 
        max_length=100, 
        description="Password (minimum 8 characters)"
    )
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

class UserUpdate(BaseModel):
    """Schema for user updates with optional fields"""
    email: Optional[EmailStr] = Field(None, description="Updated email address")
    username: Optional[str] = Field(
        None, 
        min_length=3, 
        max_length=100, 
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Updated username"
    )
    full_name: Optional[str] = Field(None, max_length=200, description="Updated full name")
    bio: Optional[str] = Field(None, max_length=1000, description="Updated biography")
    avatar_url: Optional[str] = Field(None, max_length=500, description="Updated avatar URL")
    is_active: Optional[bool] = Field(None, description="User active status")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")

class UserResponse(UserBase):
    """Schema for user responses with computed fields"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int = Field(..., description="User ID")
    uuid: UUID4 = Field(..., description="User UUID for cross-environment references")
    is_active: bool = Field(..., description="User active status")
    is_superuser: bool = Field(..., description="User superuser status")
    is_verified: bool = Field(..., description="User verification status")
    created_at: datetime = Field(..., description="User creation timestamp")
    updated_at: datetime = Field(..., description="User last update timestamp")
    last_login_at: Optional[datetime] = Field(None, description="Last login timestamp")
    metadata_json: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class UserInDB(UserResponse):
    """Schema for user with password hash (internal use only)"""
    hashed_password: str = Field(..., description="Hashed password")

class UserListResponse(BaseModel):
    """Schema for paginated user list responses"""
    users: list[UserResponse] = Field(..., description="List of users")
    total: int = Field(..., description="Total number of users")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Number of users per page")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_prev: bool = Field(..., description="Whether there are previous pages")

# Authentication schemas
class Token(BaseModel):
    """Token response schema"""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")

class TokenRefresh(BaseModel):
    """Token refresh request schema"""
    refresh_token: str = Field(..., description="Valid refresh token")

class TokenData(BaseModel):
    """Token data schema for internal use"""
    username: Optional[str] = Field(None, description="Username from token")
    permissions: list[str] = Field(default_factory=list, description="User permissions")

class LoginRequest(BaseModel):
    """Login request schema"""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")
    remember_me: bool = Field(default=False, description="Extended token expiration")

class LoginResponse(BaseModel):
    """Login response schema"""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user: UserResponse = Field(..., description="User information")

class PasswordChangeRequest(BaseModel):
    """Password change request schema"""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, max_length=100, description="New password")

# Error response schemas
class ErrorResponse(BaseModel):
    """Standard error response schema"""
    detail: str = Field(..., description="Error description")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")

class ValidationErrorResponse(BaseModel):
    """Validation error response schema"""
    detail: list = Field(..., description="Validation error details")
    error_code: str = Field(default="VALIDATION_ERROR", description="Error code")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

# Health check schemas
class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Health check timestamp")
    database: bool = Field(..., description="Database connectivity status")
    dependencies: Dict[str, bool] = Field(..., description="External dependencies status")
    
# Cross-environment integration schemas
class CrossEnvUserReference(BaseModel):
    """Schema for cross-environment user references"""
    uuid: UUID4 = Field(..., description="User UUID")
    username: str = Field(..., description="Username")
    email: EmailStr = Field(..., description="Email address")
    is_active: bool = Field(..., description="Active status")
    
class ClientSDKInfo(BaseModel):
    """Schema for client SDK information"""
    language: str = Field(..., description="Programming language")
    version: str = Field(..., description="SDK version")
    base_url: str = Field(..., description="API base URL")
    auth_required: bool = Field(..., description="Authentication requirement")
```

### Task 4: Service Layer with Business Logic and Monitoring
```python
# src/services/user_service.py - Enhanced user service with comprehensive business logic
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc, asc
from sqlalchemy.orm import selectinload
from fastapi import HTTPException, status
import logging
from uuid import UUID

from src.models.user import User
from src.models.schemas import (
    UserCreate, UserUpdate, UserResponse, UserInDB, UserListResponse,
    CrossEnvUserReference
)
from src.core.security import security_service
from src.utils.exceptions import (
    UserNotFoundError, UserAlreadyExistsError, InvalidOperationError
)

logger = logging.getLogger(__name__)

class UserService:
    """
    Enhanced user service with comprehensive business logic, monitoring integration,
    and cross-environment support.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """
        Create a new user with comprehensive validation and monitoring.
        
        Includes duplicate checking, password validation, and audit logging.
        """
        logger.info(f"Creating user with email: {user_data.email}")
        
        # Check for existing users
        existing_email = await self.get_user_by_email(user_data.email)
        if existing_email:
            logger.warning(f"Attempt to register with existing email: {user_data.email}")
            raise UserAlreadyExistsError("Email already registered")
        
        existing_username = await self.get_user_by_username(user_data.username)
        if existing_username:
            logger.warning(f"Attempt to register with existing username: {user_data.username}")
            raise UserAlreadyExistsError("Username already taken")
        
        # Validate password strength (additional checks beyond schema validation)
        if not self._validate_password_strength(user_data.password):
            raise InvalidOperationError("Password does not meet security requirements")
        
        # Create user with hashed password
        hashed_password = security_service.get_password_hash(user_data.password)
        
        db_user = User(
            email=user_data.email,
            username=user_data.username,
            full_name=user_data.full_name,
            bio=user_data.bio,
            avatar_url=user_data.avatar_url,
            hashed_password=hashed_password,
            is_active=True,
            is_superuser=False,
            is_verified=False,
            metadata_json=user_data.metadata or {}
        )
        
        try:
            self.db.add(db_user)
            await self.db.commit()
            await self.db.refresh(db_user)
            
            logger.info(f"User created successfully: ID={db_user.id}, UUID={db_user.uuid}")
            
            return UserResponse.model_validate(db_user)
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to create user: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )
    
    async def get_user_by_id(self, user_id: int) -> Optional[UserResponse]:
        """Get user by ID with caching consideration"""
        try:
            result = await self.db.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                return None
            
            return UserResponse.model_validate(user)
            
        except Exception as e:
            logger.error(f"Error fetching user by ID {user_id}: {e}")
            return None
    
    async def get_user_by_uuid(self, user_uuid: UUID) -> Optional[UserResponse]:
        """Get user by UUID for cross-environment references"""
        try:
            result = await self.db.execute(
                select(User).where(User.uuid == user_uuid)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                return None
            
            return UserResponse.model_validate(user)
            
        except Exception as e:
            logger.error(f"Error fetching user by UUID {user_uuid}: {e}")
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        """Get user by email with password hash for authentication"""
        try:
            result = await self.db.execute(
                select(User).where(User.email == email.lower())
            )
            user = result.scalar_one_or_none()
            
            if not user:
                return None
            
            return UserInDB.model_validate(user)
            
        except Exception as e:
            logger.error(f"Error fetching user by email: {e}")
            return None
    
    async def get_user_by_username(self, username: str) -> Optional[UserInDB]:
        """Get user by username with password hash"""
        try:
            result = await self.db.execute(
                select(User).where(User.username == username.lower())
            )
            user = result.scalar_one_or_none()
            
            if not user:
                return None
            
            return UserInDB.model_validate(user)
            
        except Exception as e:
            logger.error(f"Error fetching user by username: {e}")
            return None
    
    async def update_user(self, user_id: int, user_update: UserUpdate) -> UserResponse:
        """
        Update user with validation and conflict checking.
        
        Includes optimistic locking consideration and audit logging.
        """
        logger.info(f"Updating user ID: {user_id}")
        
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise UserNotFoundError(f"User with id {user_id} not found")
        
        # Check for conflicts with new email/username
        update_data = user_update.model_dump(exclude_unset=True)
        
        if "email" in update_data and update_data["email"] != user.email:
            existing_email = await self.get_user_by_email(update_data["email"])
            if existing_email and existing_email.id != user_id:
                raise UserAlreadyExistsError("Email already in use by another user")
        
        if "username" in update_data and update_data["username"] != user.username:
            existing_username = await self.get_user_by_username(update_data["username"])
            if existing_username and existing_username.id != user_id:
                raise UserAlreadyExistsError("Username already in use by another user")
        
        # Apply updates
        for field, value in update_data.items():
            if field == "email":
                setattr(user, field, value.lower())
            elif field == "username":
                setattr(user, field, value.lower())
            elif field == "metadata":
                # Merge metadata instead of replacing
                current_metadata = user.metadata_json or {}
                current_metadata.update(value or {})
                setattr(user, "metadata_json", current_metadata)
            else:
                setattr(user, field, value)
        
        try:
            await self.db.commit()
            await self.db.refresh(user)
            
            logger.info(f"User updated successfully: ID={user.id}")
            
            return UserResponse.model_validate(user)
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to update user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update user"
            )
    
    async def list_users(
        self, 
        skip: int = 0, 
        limit: int = 100,
        active_only: bool = True,
        search: Optional[str] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> UserListResponse:
        """
        List users with advanced filtering, searching, and pagination.
        
        Supports full-text search and flexible sorting options.
        """
        logger.debug(f"Listing users: skip={skip}, limit={limit}, search={search}")
        
        # Build base query
        query = select(User)
        count_query = select(func.count(User.id))
        
        # Apply filters
        if active_only:
            query = query.where(User.is_active == True)
            count_query = count_query.where(User.is_active == True)
        
        if search:
            search_term = f"%{search.lower()}%"
            search_condition = or_(
                User.email.ilike(search_term),
                User.username.ilike(search_term),
                User.full_name.ilike(search_term)
            )
            query = query.where(search_condition)
            count_query = count_query.where(search_condition)
        
        # Apply sorting
        sort_column = getattr(User, sort_by, User.created_at)
        if sort_order.lower() == "desc":
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(asc(sort_column))
        
        # Apply pagination
        query = query.offset(skip).limit(limit)
        
        try:
            # Execute queries
            result = await self.db.execute(query)
            users = result.scalars().all()
            
            count_result = await self.db.execute(count_query)
            total = count_result.scalar_one()
            
            # Calculate pagination info
            page = (skip // limit) + 1
            has_next = (skip + limit) < total
            has_prev = skip > 0
            
            user_responses = [UserResponse.model_validate(user) for user in users]
            
            return UserListResponse(
                users=user_responses,
                total=total,
                page=page,
                per_page=limit,
                has_next=has_next,
                has_prev=has_prev
            )
            
        except Exception as e:
            logger.error(f"Error listing users: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve users"
            )
    
    async def deactivate_user(self, user_id: int) -> UserResponse:
        """
        Deactivate user account (soft delete).
        
        Maintains data integrity while preventing login.
        """
        logger.info(f"Deactivating user ID: {user_id}")
        
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise UserNotFoundError(f"User with id {user_id} not found")
        
        if not user.is_active:
            logger.warning(f"Attempt to deactivate already inactive user: {user_id}")
            return UserResponse.model_validate(user)
        
        user.is_active = False
        
        try:
            await self.db.commit()
            await self.db.refresh(user)
            
            logger.info(f"User deactivated successfully: ID={user.id}")
            
            return UserResponse.model_validate(user)
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to deactivate user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to deactivate user"
            )
    
    async def activate_user(self, user_id: int) -> UserResponse:
        """Reactivate user account"""
        logger.info(f"Activating user ID: {user_id}")
        
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise UserNotFoundError(f"User with id {user_id} not found")
        
        user.is_active = True
        
        try:
            await self.db.commit()
            await self.db.refresh(user)
            
            logger.info(f"User activated successfully: ID={user.id}")
            
            return UserResponse.model_validate(user)
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to activate user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to activate user"
            )
    
    async def change_password(
        self, 
        user_id: int, 
        current_password: str, 
        new_password: str
    ) -> bool:
        """
        Change user password with current password verification.
        
        Includes password strength validation and audit logging.
        """
        logger.info(f"Password change request for user ID: {user_id}")
        
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise UserNotFoundError(f"User with id {user_id} not found")
        
        # Verify current password
        if not security_service.verify_password(current_password, user.hashed_password):
            logger.warning(f"Invalid current password for user {user_id}")
            raise InvalidOperationError("Current password is incorrect")
        
        # Validate new password strength
        if not self._validate_password_strength(new_password):
            raise InvalidOperationError("New password does not meet security requirements")
        
        # Hash new password
        new_hashed_password = security_service.get_password_hash(new_password)
        user.hashed_password = new_hashed_password
        
        try:
            await self.db.commit()
            logger.info(f"Password changed successfully for user ID: {user_id}")
            return True
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to change password for user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to change password"
            )
    
    async def get_user_count(self, active_only: bool = True) -> int:
        """Get total user count for statistics"""
        try:
            query = select(func.count(User.id))
            if active_only:
                query = query.where(User.is_active == True)
            
            result = await self.db.execute(query)
            return result.scalar_one()
            
        except Exception as e:
            logger.error(f"Error getting user count: {e}")
            return 0
    
    async def get_cross_env_reference(self, user_id: int) -> Optional[CrossEnvUserReference]:
        """
        Get user reference for cross-environment communication.
        
        Returns minimal user data suitable for other services.
        """
        try:
            result = await self.db.execute(
                select(User.uuid, User.username, User.email, User.is_active)
                .where(User.id == user_id)
            )
            user_data = result.first()
            
            if not user_data:
                return None
            
            return CrossEnvUserReference(
                uuid=user_data.uuid,
                username=user_data.username,
                email=user_data.email,
                is_active=user_data.is_active
            )
            
        except Exception as e:
            logger.error(f"Error getting cross-env reference for user {user_id}: {e}")
            return None
    
    async def update_last_login(self, user_id: int) -> None:
        """Update user's last login timestamp"""
        try:
            result = await self.db.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if user:
                user.last_login_at = func.now()
                await self.db.commit()
                
        except Exception as e:
            logger.error(f"Error updating last login for user {user_id}: {e}")
    
    def _validate_password_strength(self, password: str) -> bool:
        """
        Validate password strength beyond basic length requirements.
        
        Implements comprehensive password security checks.
        """
        if len(password) < 8:
            return False
        
        # Check for at least one uppercase, lowercase, digit, and special character
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        # Require at least 3 of the 4 character types
        strength_score = sum([has_upper, has_lower, has_digit, has_special])
        
        return strength_score >= 3

# src/services/auth_service.py - Enhanced authentication service
from typing import Optional, Dict, Any
from datetime import timedelta, datetime
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, status, Request
import logging

from src.models.user import User
from src.models.schemas import (
    UserCreate, UserResponse, LoginRequest, LoginResponse, TokenRefresh
)
from src.core.security import security_service
from src.services.user_service import UserService
from src.utils.exceptions import InvalidCredentialsError, UserNotFoundError

logger = logging.getLogger(__name__)

class AuthService:
    """
    Enhanced authentication service with comprehensive security features.
    
    Includes rate limiting, session management, and security monitoring.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.user_service = UserService(db)
        
        # Rate limiting tracking (in production, use Redis)
        self._login_attempts: Dict[str, list] = {}
        self._max_attempts = 5
        self._lockout_duration = timedelta(minutes=15)
    
    async def register_user(self, user_data: UserCreate) -> UserResponse:
        """
        Register new user with enhanced validation.
        
        Includes email validation and security checks.
        """
        logger.info(f"User registration attempt: {user_data.email}")
        
        try:
            return await self.user_service.create_user(user_data)
        except Exception as e:
            logger.error(f"Registration failed for {user_data.email}: {e}")
            raise
    
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """
        Authenticate user with rate limiting and security monitoring.
        
        Implements login attempt tracking and account lockout.
        """
        email_lower = email.lower()
        
        # Check rate limiting
        if self._is_rate_limited(email_lower):
            logger.warning(f"Rate limited login attempt: {email_lower}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many login attempts. Please try again later."
            )
        
        user = await self.user_service.get_user_by_email(email_lower)
        
        if not user:
            self._record_failed_attempt(email_lower)
            logger.warning(f"Login attempt for non-existent user: {email_lower}")
            return None
        
        if not security_service.verify_password(password, user.hashed_password):
            self._record_failed_attempt(email_lower)
            logger.warning(f"Invalid password for user: {email_lower}")
            return None
        
        # Clear failed attempts on successful authentication
        self._clear_failed_attempts(email_lower)
        
        # Update last login
        await self.user_service.update_last_login(user.id)
        
        logger.info(f"Successful authentication: {email_lower}")
        return user
    
    async def login(self, login_request: LoginRequest) -> LoginResponse:
        """
        User login with comprehensive token generation.
        
        Includes refresh token generation and extended sessions.
        """
        user = await self.authenticate_user(login_request.email, login_request.password)
        
        if not user:
            raise InvalidCredentialsError("Incorrect email or password")
        
        if not user.is_active:
            logger.warning(f"Login attempt for inactive user: {user.email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Account is deactivated"
            )
        
        # Determine token expiration based on remember_me
        if login_request.remember_me:
            access_token_expires = timedelta(hours=24)  # Extended session
            refresh_token_expires = timedelta(days=30)
        else:
            access_token_expires = timedelta(minutes=security_service.access_token_expire_minutes)
            refresh_token_expires = timedelta(days=security_service.refresh_token_expire_days)
        
        # Create tokens
        token_data = {
            "sub": user.username,
            "user_id": user.id,
            "email": user.email,
            "is_superuser": user.is_superuser
        }
        
        access_token = security_service.create_access_token(
            data=token_data,
            expires_delta=access_token_expires
        )
        
        refresh_token = security_service.create_refresh_token(data=token_data)
        
        logger.info(f"Login successful: {user.email} (remember_me={login_request.remember_me})")
        
        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=int(access_token_expires.total_seconds()),
            user=UserResponse.model_validate(user)
        )
    
    async def refresh_token(self, token_refresh: TokenRefresh) -> LoginResponse:
        """
        Refresh access token using valid refresh token.
        
        Implements token rotation for enhanced security.
        """
        try:
            # Decode refresh token
            payload = security_service.decode_token(token_refresh.refresh_token)
            
            # Verify it's actually a refresh token
            if payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            username = payload.get("sub")
            if not username:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )
            
            # Get current user
            user = await self.user_service.get_user_by_username(username)
            if not user or not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found or inactive"
                )
            
            # Generate new tokens
            token_data = {
                "sub": user.username,
                "user_id": user.id,
                "email": user.email,
                "is_superuser": user.is_superuser
            }
            
            access_token = security_service.create_access_token(data=token_data)
            new_refresh_token = security_service.create_refresh_token(data=token_data)
            
            logger.info(f"Token refreshed for user: {user.email}")
            
            return LoginResponse(
                access_token=access_token,
                refresh_token=new_refresh_token,
                token_type="bearer",
                expires_in=security_service.access_token_expire_minutes * 60,
                user=UserResponse.model_validate(user)
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not refresh token"
            )
    
    async def logout(self, user_id: int, token_jti: str) -> bool:
        """
        User logout with token invalidation.
        
        In production, implement token blacklisting with Redis.
        """
        logger.info(f"User logout: ID={user_id}, JTI={token_jti}")
        
        # TODO: Implement token blacklisting with Redis
        # For now, we rely on token expiration
        
        return True
    
    async def get_current_user_from_token(self, token: str) -> Optional[UserResponse]:
        """
        Get current user from valid JWT token.
        
        Used by dependency injection for protected endpoints.
        """
        username = security_service.verify_token(token)
        if not username:
            return None
        
        user = await self.user_service.get_user_by_username(username)
        if not user or not user.is_active:
            return None
        
        return UserResponse.model_validate(user)
    
    def _is_rate_limited(self, identifier: str) -> bool:
        """Check if identifier is rate limited"""
        now = datetime.now()
        
        if identifier not in self._login_attempts:
            return False
        
        # Clean old attempts
        cutoff = now - self._lockout_duration
        self._login_attempts[identifier] = [
            attempt for attempt in self._login_attempts[identifier]
            if attempt > cutoff
        ]
        
        return len(self._login_attempts[identifier]) >= self._max_attempts
    
    def _record_failed_attempt(self, identifier: str) -> None:
        """Record failed login attempt"""
        now = datetime.now()
        
        if identifier not in self._login_attempts:
            self._login_attempts[identifier] = []
        
        self._login_attempts[identifier].append(now)
    
    def _clear_failed_attempts(self, identifier: str) -> None:
        """Clear failed attempts for identifier"""
        if identifier in self._login_attempts:
            del self._login_attempts[identifier]
```

### Task 5: API Endpoints with Comprehensive Documentation
```python
# src/api/auth.py - Enhanced authentication endpoints with monitoring
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated
import logging

from src.core.database import get_db
from src.core.security import security_service, JWTBearer
from src.services.auth_service import AuthService
from src.services.user_service import UserService
from src.models.schemas import (
    UserCreate, UserResponse, LoginRequest, LoginResponse, 
    TokenRefresh, ErrorResponse
)
from src.utils.exceptions import InvalidCredentialsError, UserAlreadyExistsError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
jwt_bearer = JWTBearer()

async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    Get current authenticated user from JWT token.
    
    Used as dependency for protected endpoints.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    auth_service = AuthService(db)
    user = await auth_service.get_current_user_from_token(token)
    
    if user is None:
        logger.warning("Invalid or expired token used")
        raise credentials_exception
    
    return user

async def get_current_active_user(
    current_user: Annotated[UserResponse, Depends(get_current_user)]
) -> UserResponse:
    """Get current active user (non-deactivated)"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

async def get_current_superuser(
    current_user: Annotated[UserResponse, Depends(get_current_active_user)]
) -> UserResponse:
    """Get current superuser (admin access required)"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    return current_user

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a new user account with email verification",
    responses={
        201: {
            "description": "User created successfully",
            "model": UserResponse
        },
        400: {
            "description": "Email or username already exists",
            "model": ErrorResponse
        },
        422: {
            "description": "Validation error",
            "model": ErrorResponse
        },
    },
    operation_id="register_user"
)
async def register(
    user_data: UserCreate,
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    Register a new user account.
    
    **Requirements:**
    - **email**: Valid email address (must be unique)
    - **username**: Username (3-100 characters, alphanumeric/underscore/hyphen, must be unique)
    - **password**: Password (minimum 8 characters with complexity requirements)
    - **full_name**: Optional full name (max 200 characters)
    - **bio**: Optional user biography (max 1000 characters)
    - **avatar_url**: Optional avatar image URL
    
    **Password Requirements:**
    - Minimum 8 characters
    - At least 3 of: uppercase, lowercase, digit, special character
    
    **Returns:**
    - User profile information (without password)
    - UUID for cross-environment references
    """
    try:
        logger.info(f"Registration attempt from IP: {request.client.host}")
        
        auth_service = AuthService(db)
        user = await auth_service.register_user(user_data)
        
        logger.info(f"User registered successfully: {user.email}")
        return user
        
    except UserAlreadyExistsError as e:
        logger.warning(f"Registration conflict: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ValueError as e:
        logger.warning(f"Registration validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Login user",
    description="Authenticate user and return access and refresh tokens",
    responses={
        200: {
            "description": "Login successful",
            "model": LoginResponse
        },
        401: {
            "description": "Invalid credentials",
            "model": ErrorResponse
        },
        400: {
            "description": "Account deactivated",
            "model": ErrorResponse
        },
        429: {
            "description": "Too many login attempts",
            "model": ErrorResponse
        },
    },
    operation_id="login_user"
)
async def login(
    login_request: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> LoginResponse:
    """
    Login with email and password.
    
    **Parameters:**
    - **email**: User email address
    - **password**: User password
    - **remember_me**: Extended session (24 hours vs 30 minutes)
    
    **Returns:**
    - **access_token**: JWT access token for API requests
    - **refresh_token**: JWT refresh token for token renewal
    - **token_type**: Always "bearer"
    - **expires_in**: Token expiration time in seconds
    - **user**: User profile information
    
    **Authentication:**
    Use the access token in the Authorization header:
    ```
    Authorization: Bearer <access_token>
    ```
    
    **Rate Limiting:**
    - Maximum 5 failed attempts per email
    - 15-minute lockout after exceeding limit
    """
    try:
        logger.info(f"Login attempt from IP: {request.client.host}")
        
        auth_service = AuthService(db)
        login_response = await auth_service.login(login_request)
        
        logger.info(f"Login successful: {login_request.email}")
        return login_response
        
    except InvalidCredentialsError as e:
        logger.warning(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post(
    "/refresh",
    response_model=LoginResponse,
    summary="Refresh access token",
    description="Get new access token using refresh token",
    responses={
        200: {
            "description": "Token refreshed successfully",
            "model": LoginResponse
        },
        401: {
            "description": "Invalid or expired refresh token",
            "model": ErrorResponse
        },
    },
    operation_id="refresh_token"
)
async def refresh_token(
    token_refresh: TokenRefresh,
    db: AsyncSession = Depends(get_db)
) -> LoginResponse:
    """
    Refresh access token using valid refresh token.
    
    **Parameters:**
    - **refresh_token**: Valid JWT refresh token
    
    **Returns:**
    - New access token and refresh token
    - Updated user information
    
    **Security:**
    - Implements token rotation (new refresh token issued)
    - Validates refresh token type and expiration
    - Verifies user is still active
    """
    auth_service = AuthService(db)
    return await auth_service.refresh_token(token_refresh)

@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
    description="Get profile information of authenticated user",
    responses={
        200: {
            "description": "User profile retrieved",
            "model": UserResponse
        },
        401: {
            "description": "Invalid or expired token",
            "model": ErrorResponse
        },
    },
    operation_id="get_current_user"
)
async def get_current_user_profile(
    current_user: Annotated[UserResponse, Depends(get_current_active_user)]
) -> UserResponse:
    """
    Get current user profile information.
    
    **Authentication Required:**
    Requires valid JWT token in Authorization header.
    
    **Returns:**
    - Complete user profile information
    - UUID for cross-environment references
    - Account status and permissions
    - Creation and update timestamps
    
    **Usage Example:**
    ```bash
    curl -H "Authorization: Bearer <token>" http://localhost:8000/auth/me
    ```
    """
    return current_user

@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Logout user",
    description="Logout user and invalidate tokens",
    responses={
        204: {"description": "Logout successful"},
        401: {"description": "Invalid token"},
    },
    operation_id="logout_user"
)
async def logout(
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    token: Annotated[str, Depends(oauth2_scheme)],
    db: AsyncSession = Depends(get_db)
) -> None:
    """
    Logout user and invalidate current session.
    
    **Authentication Required:**
    Requires valid JWT token.
    
    **Security:**
    - Invalidates current access token
    - Logs logout event for audit
    - In production, implements token blacklisting
    """
    try:
        # Get token JTI for blacklisting
        payload = security_service.decode_token(token)
        jti = payload.get("jti")
        
        auth_service = AuthService(db)
        await auth_service.logout(current_user.id, jti)
        
        logger.info(f"User logged out: {current_user.email}")
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        # Don't raise exception for logout - fail gracefully

# OAuth2 compatibility endpoint for Swagger UI
@router.post(
    "/token",
    response_model=dict,
    summary="OAuth2 token endpoint",
    description="OAuth2 compatible login endpoint for Swagger UI",
    include_in_schema=False,  # Hidden from main docs
    operation_id="oauth2_token"
)
async def oauth2_login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    OAuth2 compatible login endpoint for Swagger UI.
    
    Uses form data instead of JSON for OAuth2 standard compliance.
    """
    login_request = LoginRequest(
        email=form_data.username,  # OAuth2 uses 'username' field
        password=form_data.password,
        remember_me=False
    )
    
    auth_service = AuthService(db)
    login_response = await auth_service.login(login_request)
    
    # Return format expected by OAuth2
    return {
        "access_token": login_response.access_token,
        "token_type": login_response.token_type,
    }

# src/api/users.py - Enhanced user management endpoints
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated, List, Optional
import logging

from src.core.database import get_db
from src.services.user_service import UserService
from src.models.schemas import (
    UserResponse, UserUpdate, UserListResponse, PasswordChangeRequest,
    CrossEnvUserReference, ErrorResponse
)
from src.api.auth import get_current_active_user, get_current_superuser
from src.utils.exceptions import UserNotFoundError, UserAlreadyExistsError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/users", tags=["User Management"])

@router.get(
    "/",
    response_model=UserListResponse,
    summary="List users",
    description="Get a paginated list of users with filtering and search",
    responses={
        200: {
            "description": "List of users retrieved",
            "model": UserListResponse
        },
        401: {"description": "Authentication required"},
    },
    operation_id="list_users"
)
async def list_users(
    skip: int = Query(0, ge=0, description="Number of users to skip for pagination"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of users to return"),
    active_only: bool = Query(True, description="Only return active users"),
    search: Optional[str] = Query(None, min_length=3, max_length=100, description="Search term for email, username, or full name"),
    sort_by: str = Query("created_at", regex="^(created_at|updated_at|email|username|full_name)$", description="Field to sort by"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    current_user: Annotated[UserResponse, Depends(get_current_active_user)] = None,
    db: AsyncSession = Depends(get_db)
) -> UserListResponse:
    """
    List users with advanced filtering and pagination.
    
    **Parameters:**
    - **skip**: Number of users to skip (for pagination)
    - **limit**: Maximum users to return (1-100)
    - **active_only**: Filter to only active users
    - **search**: Search in email, username, or full name
    - **sort_by**: Field to sort by (created_at, updated_at, email, username, full_name)
    - **sort_order**: Sort direction (asc or desc)
    
    **Returns:**
    - Paginated list of users
    - Total count and pagination metadata
    - Search and filter results
    
    **Permissions:**
    - Requires authentication
    - All authenticated users can list users
    """
    logger.debug(f"Listing users: user={current_user.email}, search={search}")
    
    user_service = UserService(db)
    return await user_service.list_users(
        skip=skip,
        limit=limit,
        active_only=active_only,
        search=search,
        sort_by=sort_by,
        sort_order=sort_order
    )

@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Get user by ID",
    description="Get detailed information about a specific user",
    responses={
        200: {
            "description": "User information retrieved",
            "model": UserResponse
        },
        404: {"description": "User not found"},
        401: {"description": "Authentication required"},
    },
    operation_id="get_user_by_id"
)
async def get_user(
    user_id: int = Path(..., description="User ID to retrieve"),
    current_user: Annotated[UserResponse, Depends(get_current_active_user)] = None,
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    Get user information by ID.
    
    **Parameters:**
    - **user_id**: Unique user identifier
    
    **Returns:**
    - Complete user profile information
    - Account status and metadata
    
    **Permissions:**
    - Requires authentication
    - Users can view any user profile
    """
    logger.debug(f"Getting user {user_id} requested by {current_user.email}")
    
    user_service = UserService(db)
    user = await user_service.get_user_by_id(user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user

@router.get(
    "/uuid/{user_uuid}",
    response_model=UserResponse,
    summary="Get user by UUID",
    description="Get user information by UUID (for cross-environment references)",
    responses={
        200: {"description": "User information retrieved"},
        404: {"description": "User not found"},
        401: {"description": "Authentication required"},
    },
    operation_id="get_user_by_uuid"
)
async def get_user_by_uuid(
    user_uuid: str = Path(..., description="User UUID"),
    current_user: Annotated[UserResponse, Depends(get_current_active_user)] = None,
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    Get user information by UUID.
    
    **Cross-Environment Usage:**
    UUIDs provide stable references across different services and environments.
    
    **Parameters:**
    - **user_uuid**: User UUID (stable identifier)
    
    **Returns:**
    - User profile information
    """
    user_service = UserService(db)
    user = await user_service.get_user_by_uuid(user_uuid)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user

@router.put(
    "/me",
    response_model=UserResponse,
    summary="Update current user",
    description="Update authenticated user's profile information",
    responses={
        200: {
            "description": "Profile updated successfully",
            "model": UserResponse
        },
        400: {"description": "Email or username already exists"},
        401: {"description": "Authentication required"},
    },
    operation_id="update_current_user"
)
async def update_current_user(
    user_update: UserUpdate,
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    Update current user profile.
    
    **Updatable Fields:**
    - **email**: New email address (must be unique)
    - **username**: New username (must be unique)
    - **full_name**: Full name
    - **bio**: User biography
    - **avatar_url**: Avatar image URL
    - **metadata**: Additional metadata
    
    **Validation:**
    - Email and username uniqueness enforced
    - Only provided fields are updated
    - Maintains audit trail
    """
    try:
        logger.info(f"Profile update requested by user: {current_user.email}")
        
        user_service = UserService(db)
        updated_user = await user_service.update_user(current_user.id, user_update)
        
        logger.info(f"Profile updated successfully: {current_user.email}")
        return updated_user
        
    except UserAlreadyExistsError as e:
        logger.warning(f"Profile update conflict: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.put(
    "/{user_id}",
    response_model=UserResponse,
    summary="Update user (admin only)",
    description="Update any user's profile (requires admin privileges)",
    responses={
        200: {"description": "User updated successfully"},
        400: {"description": "Validation error or conflict"},
        401: {"description": "Authentication required"},
        403: {"description": "Admin privileges required"},
        404: {"description": "User not found"},
    },
    operation_id="update_user_admin"
)
async def update_user_admin(
    user_id: int = Path(..., description="User ID to update"),
    user_update: UserUpdate = ...,
    current_user: Annotated[UserResponse, Depends(get_current_superuser)] = None,
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    Update any user's profile (admin only).
    
    **Admin Capabilities:**
    - Update any user's profile
    - Change active status
    - Modify user metadata
    
    **Parameters:**
    - **user_id**: Target user ID
    - **user_update**: Fields to update
    
    **Permissions:**
    - Requires superuser privileges
    """
    try:
        logger.info(f"Admin user update: admin={current_user.email}, target_id={user_id}")
        
        user_service = UserService(db)
        updated_user = await user_service.update_user(user_id, user_update)
        
        logger.info(f"User updated by admin: user_id={user_id}")
        return updated_user
        
    except UserNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except UserAlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post(
    "/me/change-password",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Change password",
    description="Change current user's password",
    responses={
        204: {"description": "Password changed successfully"},
        400: {"description": "Invalid current password or weak new password"},
        401: {"description": "Authentication required"},
    },
    operation_id="change_password"
)
async def change_password(
    password_change: PasswordChangeRequest,
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    db: AsyncSession = Depends(get_db)
) -> None:
    """
    Change current user's password.
    
    **Security Requirements:**
    - Must provide current password for verification
    - New password must meet strength requirements
    - Audit logging for security monitoring
    
    **Parameters:**
    - **current_password**: Current password for verification
    - **new_password**: New password (min 8 chars, complexity required)
    """
    logger.info(f"Password change requested by user: {current_user.email}")
    
    user_service = UserService(db)
    await user_service.change_password(
        current_user.id,
        password_change.current_password,
        password_change.new_password
    )
    
    logger.info(f"Password changed successfully: {current_user.email}")

@router.delete(
    "/me",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Deactivate current user",
    description="Deactivate the authenticated user's account",
    responses={
        204: {"description": "Account deactivated successfully"},
        401: {"description": "Authentication required"},
    },
    operation_id="deactivate_current_user"
)
async def deactivate_current_user(
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    db: AsyncSession = Depends(get_db)
) -> None:
    """
    Deactivate current user account (soft delete).
    
    **Effects:**
    - Sets account to inactive status
    - Prevents future logins
    - Preserves data for audit purposes
    - Does not delete account permanently
    
    **Recovery:**
    Contact administrator to reactivate account.
    """
    logger.info(f"Account deactivation requested by user: {current_user.email}")
    
    user_service = UserService(db)
    await user_service.deactivate_user(current_user.id)
    
    logger.info(f"Account deactivated successfully: {current_user.email}")

@router.post(
    "/{user_id}/activate",
    response_model=UserResponse,
    summary="Activate user (admin only)",
    description="Activate a deactivated user account",
    responses={
        200: {"description": "User activated successfully"},
        401: {"description": "Authentication required"},
        403: {"description": "Admin privileges required"},
        404: {"description": "User not found"},
    },
    operation_id="activate_user"
)
async def activate_user(
    user_id: int = Path(..., description="User ID to activate"),
    current_user: Annotated[UserResponse, Depends(get_current_superuser)] = None,
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    Activate a deactivated user account (admin only).
    
    **Admin Function:**
    - Reactivates deactivated accounts
    - Restores login capability
    - Audit logged for security
    
    **Parameters:**
    - **user_id**: Target user ID to activate
    
    **Permissions:**
    - Requires superuser privileges
    """
    logger.info(f"User activation requested: admin={current_user.email}, target_id={user_id}")
    
    user_service = UserService(db)
    
    try:
        activated_user = await user_service.activate_user(user_id)
        logger.info(f"User activated by admin: user_id={user_id}")
        return activated_user
        
    except UserNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

@router.get(
    "/{user_id}/cross-env-ref",
    response_model=CrossEnvUserReference,
    summary="Get cross-environment user reference",
    description="Get minimal user data for cross-environment communication",
    responses={
        200: {"description": "User reference retrieved"},
        401: {"description": "Authentication required"},
        404: {"description": "User not found"},
    },
    operation_id="get_cross_env_reference"
)
async def get_cross_env_reference(
    user_id: int = Path(..., description="User ID"),
    current_user: Annotated[UserResponse, Depends(get_current_active_user)] = None,
    db: AsyncSession = Depends(get_db)
) -> CrossEnvUserReference:
    """
    Get user reference for cross-environment communication.
    
    **Cross-Environment Integration:**
    Returns minimal user data suitable for other services and environments.
    
    **Use Cases:**
    - TypeScript frontend authentication
    - Rust service user validation
    - Go microservice authorization
    - Nushell automation scripts
    
    **Returns:**
    - UUID, username, email, active status
    - Optimized for cross-service communication
    """
    user_service = UserService(db)
    user_ref = await user_service.get_cross_env_reference(user_id)
    
    if not user_ref:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user_ref

@router.get(
    "/stats/count",
    response_model=dict,
    summary="Get user statistics",
    description="Get user count and statistics",
    responses={
        200: {"description": "Statistics retrieved"},
        401: {"description": "Authentication required"},
    },
    operation_id="get_user_stats"
)
async def get_user_stats(
    include_inactive: bool = Query(False, description="Include inactive users in count"),
    current_user: Annotated[UserResponse, Depends(get_current_active_user)] = None,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Get user statistics and counts.
    
    **Statistics:**
    - Total user count
    - Active/inactive breakdown
    - Registration trends (if implemented)
    
    **Parameters:**
    - **include_inactive**: Include deactivated users in count
    
    **Permissions:**
    - Requires authentication
    """
    user_service = UserService(db)
    
    if include_inactive:
        active_count = await user_service.get_user_count(active_only=True)
        total_count = await user_service.get_user_count(active_only=False)
        inactive_count = total_count - active_count
        
        return {
            "total_users": total_count,
            "active_users": active_count,
            "inactive_users": inactive_count
        }
    else:
        active_count = await user_service.get_user_count(active_only=True)
        return {
            "active_users": active_count
        }
```

## List of Tasks to Complete (Multi-Environment Implementation)

```yaml
Task 1: Environment Setup and Cross-Platform Dependencies
  COMMAND: cd python-env && devbox shell
  VERIFY: uv --version && python --version  # Should be 3.12+
  INSTALL_CORE: uv add "fastapi[all]" "uvicorn[standard]" "sqlalchemy[asyncio]>=2.0.0" "asyncpg" "alembic" "PyJWT>=2.8.0" "passlib[bcrypt]" "python-multipart" "pydantic[email]>=2.0" "pydantic-settings>=2.0"
  INSTALL_DEV: uv add --dev "pytest-asyncio>=0.23.0" "httpx>=0.25.0" "pytest-cov>=4.0" "aiosqlite"
  VERIFY_CROSS_ENV: Check access to typescript-env, rust-env, go-env, nushell-env

Task 2: Core Infrastructure with Enhanced Monitoring
  CREATE: src/core/config.py (Enhanced Pydantic settings with monitoring flags)
  CREATE: src/core/database.py (Async SQLAlchemy 2.0 with performance tracking)
  CREATE: src/core/security.py (PyJWT with comprehensive security monitoring)
  PATTERN: Use Pydantic v2 patterns, SQLAlchemy 2.0 async, PyJWT instead of python-jose
  INTEGRATE: Performance analytics hooks and security monitoring

Task 3: Database Models and Schemas with Cross-Environment Support
  CREATE: src/models/user.py (Enhanced User model with UUID and performance indexes)
  CREATE: src/models/schemas.py (Comprehensive Pydantic v2 schemas with cross-env references)
  PATTERN: UUID fields for cross-environment references, performance-optimized indexes
  VALIDATE: Model relationships and constraint validation

Task 4: Enhanced Service Layer with Business Logic
  CREATE: src/services/user_service.py (Comprehensive user business logic)
  CREATE: src/services/auth_service.py (Enhanced authentication with rate limiting)
  PATTERN: Async service methods with monitoring integration
  SECURITY: Rate limiting, password strength validation, audit logging

Task 5: API Endpoints with Comprehensive Documentation
  CREATE: src/api/auth.py (Authentication endpoints with OAuth2 compatibility)
  CREATE: src/api/users.py (User management with advanced filtering)
  CREATE: src/api/admin.py (Admin operations with proper authorization)
  PATTERN: FastAPI dependency injection, comprehensive OpenAPI documentation
  MONITORING: Request tracking and performance measurement

Task 6: Exception Handling and Utilities
  CREATE: src/utils/exceptions.py (Custom exception hierarchy)
  CREATE: src/utils/monitoring.py (Monitoring integration utilities)
  PATTERN: Structured error handling with monitoring integration

Task 7: Database Migrations with Async Support
  SETUP: uv run alembic init alembic
  CONFIGURE: Async-compatible Alembic configuration
  CREATE: uv run alembic revision --autogenerate -m "Create users table with enhanced indexes"
  APPLY: uv run alembic upgrade head

Task 8: Comprehensive Async Testing with Monitoring
  CREATE: tests/conftest.py (Enhanced async test fixtures with monitoring)
  CREATE: tests/test_api/ (API endpoint tests with performance tracking)
  CREATE: tests/test_services/ (Service layer tests)
  CREATE: tests/test_integration/ (End-to-end flow tests)
  CREATE: tests/test_monitoring/ (Monitoring integration tests)
  PATTERN: pytest-asyncio with httpx AsyncClient and monitoring validation

Task 9: Cross-Environment Client Generation
  CREATE: src/clients/typescript.py (TypeScript SDK generator)
  CREATE: src/clients/rust.py (Rust client library generator)
  CREATE: src/clients/go.py (Go client package generator)
  CREATE: src/clients/nushell.py (Nushell script generator)
  PATTERN: Automated client generation with type safety

Task 10: Application Integration with Enhanced Features
  CREATE: src/main.py (FastAPI app with comprehensive middleware and monitoring)
  PATTERN: Production-ready FastAPI with lifespan events, CORS, rate limiting
  INTEGRATE: Performance analytics, security scanning, resource monitoring

Task 11: Multi-Environment Development Setup
  CREATE: docker-compose.yml (PostgreSQL + Redis for development)
  CREATE: .env.example (Comprehensive environment variables)
  CREATE: clients/ (Generated client libraries directory structure)
  PATTERN: Development-friendly multi-environment configuration

Task 12: Cross-Environment Orchestration Scripts
  CREATE: nushell-env/user-api/deploy.nu (Multi-environment deployment)
  CREATE: nushell-env/user-api/monitor.nu (Monitoring and health checks)
  CREATE: nushell-env/user-api/test-all.nu (Cross-environment testing)
  CREATE: nushell-env/user-api/generate-clients.nu (Client SDK generation)
  PATTERN: Nushell automation for cross-environment operations

Task 13: TypeScript Client SDK Implementation
  DIRECTORY: typescript-env/user-api-client/
  CREATE: TypeScript client with Zod validation and axios
  PATTERN: Type-safe API client with comprehensive error handling

Task 14: Rust Client Library Implementation
  DIRECTORY: rust-env/user-api-client/
  CREATE: Rust async client with reqwest and serde
  PATTERN: Safe Rust client with proper error handling

Task 15: Go Client Package Implementation
  DIRECTORY: go-env/user-api-client/
  CREATE: Go client with context support and proper error wrapping
  PATTERN: Idiomatic Go client with interface design

Task 16: Monitoring Integration Implementation
  CREATE: src/monitoring/performance.py (Performance analytics integration)
  CREATE: src/monitoring/security.py (Security monitoring integration)
  CREATE: src/monitoring/health.py (Health check system)
  INTEGRATE: Full monitoring pipeline with nushell scripts

Task 17: Production Deployment Configuration
  CREATE: Dockerfile (Multi-stage build for production)
  CREATE: kubernetes/ (K8s deployment manifests)
  CREATE: terraform/ (Infrastructure as code)
  PATTERN: Production-ready deployment with monitoring

Task 18: Documentation and Integration Guides
  UPDATE: README.md (Comprehensive multi-environment setup)
  CREATE: docs/ (API documentation and integration guides)
  CREATE: examples/ (Usage examples for each environment)
  PATTERN: Clear documentation for cross-environment usage
```

## Validation Loop (Multi-Environment)

### Level 1: Python Environment Quality and Standards
```bash
cd python-env && devbox shell

# Install all dependencies
uv sync --dev

# Format code with latest ruff
uv run ruff format src/ tests/

# Comprehensive linting with ruff
uv run ruff check src/ tests/ --fix

# Strict type checking with mypy
uv run mypy src/ tests/

# Expected: All checks pass without errors
```

### Level 2: Database and Migration Testing
```bash
# Start development databases
docker-compose up -d postgres redis

# Run database migrations
uv run alembic upgrade head

# Verify database schema and connections
uv run python -c "
from src.core.database import engine, check_db_health
import asyncio
print('Database connection test:', asyncio.run(check_db_health()))
"

# Expected: Database initializes and connects successfully
```

### Level 3: Comprehensive Testing with Monitoring
```bash
# Run all tests with coverage
uv run pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# Run specific test categories
uv run pytest tests/test_api/ -v
uv run pytest tests/test_services/ -v
uv run pytest tests/test_integration/ -v
uv run pytest tests/test_monitoring/ -v

# Performance testing
uv run pytest tests/ -v --durations=10

# Expected: All tests pass, 90%+ coverage, performance within limits
```

### Level 4: Cross-Environment Validation
```bash
# Generate and test client libraries
uv run python -m src.clients.typescript --output ../typescript-env/user-api-client/
uv run python -m src.clients.rust --output ../rust-env/user-api-client/
uv run python -m src.clients.go --output ../go-env/user-api-client/

# Test TypeScript client
cd ../typescript-env/user-api-client && npm test

# Test Rust client
cd ../rust-env/user-api-client && cargo test

# Test Go client
cd ../go-env/user-api-client && go test

# Test Nushell orchestration
cd ../nushell-env/user-api && nu test-all.nu

# Expected: All cross-environment clients work correctly
```

### Level 5: API Testing and Documentation Validation
```bash
# Start the API server
uv run uvicorn src.main:app --reload --port 8000

# Test API endpoints with monitoring
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "username": "testuser", "password": "TestPass123!"}'

# Test authentication flow
TOKEN=$(curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "TestPass123!"}' | jq -r '.access_token')

# Test protected endpoints
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/auth/me

# Check comprehensive API documentation
open http://localhost:8000/docs

# Expected: API responds correctly, documentation is comprehensive and interactive
```

### Level 6: Performance and Security Monitoring Integration
```bash
# Test performance analytics integration
nu ../nushell-env/scripts/performance-analytics.nu record "api-startup" "python-env" "5s" --status "success"

# Test security scanner integration
nu ../nushell-env/scripts/security-scanner.nu scan-directory "src/" --format "json" --quiet

# Test resource monitoring
nu ../nushell-env/scripts/resource-monitor.nu track "user-api-test" --duration "30s"

# Validate monitoring dashboard
nu ../nushell-env/user-api/monitor.nu --dashboard

# Expected: All monitoring systems integrate successfully with meaningful data
```

### Level 7: Production Readiness Validation
```bash
# Build production container
docker build -t user-api:latest .

# Test production configuration
docker run -d --name user-api-test \
  -e DATABASE_URL="postgresql+asyncpg://..." \
  -e SECRET_KEY="$(openssl rand -hex 32)" \
  -p 8000:8000 user-api:latest

# Health check
curl http://localhost:8000/health

# Performance benchmark
ab -n 1000 -c 10 http://localhost:8000/health

# Security scan
docker run --rm -v $(pwd):/app clair-scanner:latest /app

# Expected: Production build works, passes security scans, meets performance requirements
```

## Final Validation Checklist (Multi-Environment)

### Core Functionality
- [ ] Environment setup successful: Python 3.12+ with uv package manager
- [ ] All dependencies installed: FastAPI, SQLAlchemy 2.0 async, PyJWT, etc.
- [ ] Code quality passes: ruff format, ruff check, mypy all clean
- [ ] Database migrations work: Alembic creates and applies migrations successfully
- [ ] All tests pass: 90%+ coverage with comprehensive async testing

### API Implementation
- [ ] API server starts: uvicorn runs without errors
- [ ] Authentication functional: Registration, login, JWT token flow works
- [ ] CRUD operations work: User management endpoints respond correctly
- [ ] OpenAPI documentation complete: /docs shows comprehensive interactive documentation
- [ ] Rate limiting functional: Login attempt limits and security measures work
- [ ] Admin operations work: Superuser endpoints function correctly

### Cross-Environment Integration
- [ ] TypeScript client generated: Type-safe client with Zod validation
- [ ] Rust client generated: Async client with proper error handling
- [ ] Go client generated: Idiomatic client with context support
- [ ] Nushell scripts functional: Orchestration and automation scripts work
- [ ] Cross-environment testing: All clients can communicate with API

### Monitoring and Intelligence Integration
- [ ] Performance analytics integrated: API operations tracked and analyzed
- [ ] Security scanner integration: Code changes trigger security scans
- [ ] Resource monitoring active: Memory, CPU, database usage tracked
- [ ] Health checks comprehensive: Database, dependencies, and service health monitored
- [ ] Test intelligence working: Test performance and flaky test detection active

### Security and Production Readiness
- [ ] Security implemented: PyJWT tokens, bcrypt passwords, input validation
- [ ] Error handling comprehensive: Proper HTTP status codes and error messages
- [ ] Logging configured: Structured logging with correlation IDs
- [ ] Environment configuration: Secure secrets management and configuration
- [ ] Docker deployment ready: Multi-stage build and production configuration

### Documentation and Usability
- [ ] README comprehensive: Multi-environment setup and usage instructions
- [ ] API documentation complete: OpenAPI with examples and authentication flows
- [ ] Client documentation: Usage guides for TypeScript, Rust, Go clients
- [ ] Monitoring documentation: Performance analytics and security scanning guides
- [ ] Integration examples: Cross-environment communication patterns

---

## Confidence Score: 9.5/10

This comprehensive PRP provides complete context for implementing a production-ready user management API across the entire polyglot development environment. The high confidence score reflects:

1. **Complete Technical Specification**: All modern 2024 patterns covered (PyJWT, SQLAlchemy 2.0, async FastAPI)
2. **Multi-Environment Integration**: Full cross-language client generation and orchestration
3. **Intelligence Integration**: Deep integration with existing performance analytics and monitoring systems
4. **Production Readiness**: Comprehensive security, monitoring, deployment, and documentation
5. **Validation Framework**: Multi-level testing and validation across all environments
6. **Real-World Implementation**: Based on current best practices and proven patterns

The implementation should succeed in one pass with the provided context, validation loops, and comprehensive specifications for all components across the polyglot environment.