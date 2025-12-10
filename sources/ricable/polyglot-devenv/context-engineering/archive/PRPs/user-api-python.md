name: "User Management REST API - Production-Ready FastAPI with JWT Authentication"
description: |

## Purpose
Comprehensive Product Requirements Prompt (PRP) for implementing a production-ready user management REST API in the python-env environment. This PRP provides complete context for one-pass implementation of a secure, scalable, and maintainable user management system with modern FastAPI patterns, async SQLAlchemy database integration, JWT authentication, and comprehensive testing.

## Core Principles
1. **Security First**: JWT-based authentication with secure password hashing and input validation
2. **Async Excellence**: Full async/await implementation throughout the stack
3. **Type Safety**: Strict mypy compliance and comprehensive Pydantic models
4. **Testing Excellence**: 90%+ test coverage with async testing patterns
5. **Production Ready**: Comprehensive error handling, logging, and monitoring integration
6. **Modern Standards**: Latest 2024 patterns for FastAPI, SQLAlchemy 2.0, and PyJWT

---

## Goal
Build a complete user management REST API with user registration, JWT authentication, profile management, admin operations, and comprehensive testing. The API should integrate seamlessly with the existing polyglot environment's automation and monitoring systems.

## Why
- **Foundation Service**: Core user management system for all applications in the polyglot environment
- **Security Template**: Demonstrates secure authentication patterns for other services
- **Performance Benchmark**: Async implementation showcases optimal FastAPI performance
- **Testing Example**: Comprehensive testing patterns for complex async FastAPI applications
- **Monitoring Integration**: Leverages existing performance analytics and security scanning systems

## What
RESTful API with complete user lifecycle management including:
- User registration and authentication system
- JWT-based authentication with secure token management
- PostgreSQL database integration with async SQLAlchemy 2.0
- Role-based access control (user/admin roles)
- Comprehensive REST API with OpenAPI documentation
- Production-ready configuration and deployment setup

### Success Criteria
- [ ] Complete CRUD operations for user management
- [ ] JWT-based authentication system using PyJWT (not python-jose)
- [ ] PostgreSQL database integration with async SQLAlchemy 2.0
- [ ] Comprehensive test coverage (90%+) with async testing patterns
- [ ] OpenAPI documentation with security examples
- [ ] Integration with existing polyglot monitoring systems
- [ ] Production deployment configuration
- [ ] Security best practices implementation

## All Needed Context

### Target Environment
```yaml
Environment: python-env
Python_Version: 3.12+ (from devbox.json)
Package_Manager: uv (exclusively - no pip/poetry/pipenv)
Database: PostgreSQL with async support
Authentication: JWT with PyJWT (not python-jose)
Testing: pytest-asyncio with httpx AsyncClient
```

### Current Environment Analysis
```yaml
# From python-env/devbox.json analysis
Existing_Packages: ["python@3.12", "uv", "ruff", "mypy", "nushell"]
Existing_Scripts: {
  "format": "uv run ruff format .",
  "lint": "uv run ruff check . --fix", 
  "type-check": "uv run mypy .",
  "test": "uv run pytest --cov=src"
}
Project_Structure: {
  "pyproject.toml": "Basic FastAPI, Pydantic, httpx dependencies",
  "src/": "Not yet created - will be main application directory",
  "tests/": "Basic test files exist"
}
```

### Required Dependencies (2024 Best Practices)
```yaml
# Production Dependencies
Core_API: ["fastapi[all]", "uvicorn[standard]"]
Database: ["sqlalchemy[asyncio]>=2.0", "asyncpg", "alembic"]
Authentication: ["PyJWT", "passlib[bcrypt]", "python-multipart"]
Validation: ["pydantic[email]", "pydantic-settings"]

# Development Dependencies  
Testing: ["pytest-asyncio", "httpx", "pytest-cov"]
Quality: ["ruff>=0.8.0", "mypy>=1.7.0"]
Database_Testing: ["aiosqlite"]  # For test database

# Security Note: Using PyJWT instead of python-jose (deprecated in 2024)
```

### Documentation & References
```yaml
# CRITICAL READING - Include these in context window
Primary_References:
  - url: https://fastapi.tiangolo.com/tutorial/
    why: FastAPI patterns, dependency injection, async handling
    
  - url: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
    why: SQLAlchemy async patterns and session management
    
  - url: https://pyjwt.readthedocs.io/en/stable/
    why: JWT authentication with PyJWT (not python-jose)
    
  - url: https://alembic.sqlalchemy.org/en/latest/
    why: Database migrations and schema management

Security_References:
  - pattern: "Use PyJWT instead of python-jose (deprecated 2024)"
  - pattern: "Implement proper async session management"
  - pattern: "Use passlib with bcrypt for password hashing"
  - pattern: "Store secrets in environment variables"

Testing_References:
  - pattern: "Use pytest-asyncio with httpx AsyncClient"
  - pattern: "Implement proper async database testing"
  - pattern: "Use dependency overrides for testing"
```

### Polyglot Environment Integration
```yaml
# Existing automation systems to integrate with
Performance_Analytics: "nushell-env/scripts/performance-analytics.nu"
Security_Scanner: "nushell-env/scripts/security-scanner.nu"  
Resource_Monitor: "nushell-env/scripts/resource-monitor.nu"
Test_Intelligence: "nushell-env/scripts/test-intelligence.nu"

# Integration patterns from environment analysis
Monitoring_Pattern: "All operations tracked via performance analytics"
Security_Pattern: "Automated security scanning on code changes"
Quality_Pattern: "Automated linting and formatting via hooks"
```

### Current Codebase Structure
```bash
python-env/
├── devbox.json         # Python 3.12, uv, ruff, mypy, nushell
├── pyproject.toml      # Basic FastAPI dependencies
├── test_example.py     # Basic test examples  
├── test_format.py      # Format test file
└── uv.lock            # Dependency lock file
```

### Target Codebase Structure
```bash
python-env/
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py          # Pydantic settings with environment variables
│   │   ├── database.py        # Async SQLAlchemy session management
│   │   ├── security.py        # JWT and password hashing utilities
│   │   └── dependencies.py    # FastAPI dependency injection
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py           # SQLAlchemy User model
│   │   └── schemas.py        # Pydantic request/response schemas
│   ├── api/
│   │   ├── __init__.py
│   │   ├── auth.py           # Authentication endpoints (register, login)
│   │   ├── users.py          # User management endpoints
│   │   └── admin.py          # Admin operations endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── user_service.py   # User business logic
│   │   └── auth_service.py   # Authentication business logic
│   └── utils/
│       ├── __init__.py
│       └── exceptions.py     # Custom exception classes
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Pytest configuration and fixtures
│   ├── test_api/
│   │   ├── __init__.py
│   │   ├── test_auth.py     # Authentication endpoint tests
│   │   ├── test_users.py    # User management tests
│   │   └── test_admin.py    # Admin operations tests
│   ├── test_services/
│   │   ├── __init__.py
│   │   ├── test_user_service.py
│   │   └── test_auth_service.py
│   └── test_integration/
│       ├── __init__.py
│       └── test_user_flow.py # End-to-end user flow tests
├── alembic/
│   ├── versions/            # Database migration files
│   ├── env.py              # Alembic environment configuration
│   └── script.py.mako      # Migration template
├── alembic.ini             # Alembic configuration
├── docker-compose.yml      # PostgreSQL for development
├── .env.example           # Environment variables template
├── requirements.txt        # Alternative dependency specification
└── README.md              # API documentation and setup guide
```

### Known Environment Gotchas & 2024 Updates
```python
# CRITICAL: Python environment-specific gotchas and 2024 updates

# Package Management
# ✅ uv add package-name
# ✅ uv run command
# ❌ pip install package-name
# ❌ poetry add package-name

# Authentication Library Changes (2024)
# ✅ import jwt  # PyJWT library
# ✅ from jwt import InvalidTokenError
# ❌ from jose import JWTError, jwt  # python-jose deprecated
# ❌ from jose.exceptions import JWTError

# SQLAlchemy 2.0 Patterns
# ✅ from sqlalchemy import select, and_, or_
# ✅ result = await session.execute(select(User).where(User.email == email))
# ✅ user = result.scalar_one_or_none()
# ❌ session.query(User).filter(User.email == email).first()

# FastAPI Async Patterns
# ✅ async def endpoint(db: AsyncSession = Depends(get_db)):
# ✅ async with AsyncClient(app=app, base_url="http://test") as client:
# ❌ def sync_endpoint():  # Don't use sync endpoints
# ❌ with TestClient(app) as client:  # Use for sync only

# Pydantic v2 Patterns
# ✅ model_config = ConfigDict(from_attributes=True)
# ✅ return UserResponse.model_validate(user)
# ❌ class Config: orm_mode = True  # Pydantic v1 pattern

# Testing Patterns (2024)
# ✅ @pytest.mark.asyncio
# ✅ async def test_endpoint():
# ✅ async with AsyncClient(app=app, base_url="http://test") as client:
# ❌ def test_endpoint():  # Use sync only when needed
```

## Implementation Blueprint

### Task 1: Environment Setup and Modern Dependencies
```bash
# Activate Python environment
cd python-env && devbox shell

# Verify environment
uv --version  # Should show uv version
python --version  # Should be 3.12+

# Install production dependencies (2024 best practices)
uv add "fastapi[all]" "uvicorn[standard]"
uv add "sqlalchemy[asyncio]>=2.0.0" "asyncpg" "alembic"
uv add "PyJWT" "passlib[bcrypt]" "python-multipart"
uv add "pydantic[email]" "pydantic-settings"

# Install development dependencies
uv add --dev "pytest-asyncio" "httpx" "pytest-cov"
uv add --dev "aiosqlite"  # For test database

# Verify installation
uv run python -c "import fastapi, sqlalchemy, jwt, passlib; print('All packages installed')"
```

### Task 2: Core Configuration and Database Setup
```python
# src/core/config.py
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

class Settings(BaseSettings):
    """Application configuration with environment variable support"""
    
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
    
    # Database Settings
    database_url: str = Field(..., description="Database connection string")
    database_echo: bool = Field(False, description="Enable SQLAlchemy query logging")
    
    # Security Settings
    secret_key: str = Field(..., description="JWT secret key")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # CORS Settings
    allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    allowed_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allowed_headers: List[str] = ["*"]
    
    # Security Headers
    include_security_headers: bool = True

# Global settings instance
settings = Settings()

# src/core/database.py
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from typing import AsyncGenerator
from .config import settings

class Base(DeclarativeBase):
    """Base class for all database models"""
    pass

# Create async engine with optimal settings
engine = create_async_engine(
    settings.database_url,
    echo=settings.database_echo,
    future=True,
    pool_size=20,
    max_overflow=0,
    pool_pre_ping=True,
    pool_recycle=3600,
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# Database utility functions
async def init_db() -> None:
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def close_db() -> None:
    """Close database connections"""
    await engine.dispose()
```

### Task 3: Security Implementation with PyJWT (2024 Best Practices)
```python
# src/core/security.py
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import jwt
from jwt import InvalidTokenError
from passlib.context import CryptContext
from passlib.hash import bcrypt
from fastapi import HTTPException, status
from .config import settings

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class SecurityService:
    """Security service for password hashing and JWT operations"""
    
    def __init__(self):
        self.secret_key = settings.secret_key
        self.algorithm = settings.algorithm
        self.access_token_expire_minutes = settings.access_token_expire_minutes
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash password using bcrypt"""
        return pwd_context.hash(password)
    
    def create_access_token(
        self, 
        data: Dict[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token with proper timezone handling"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                minutes=self.access_token_expire_minutes
            )
        
        to_encode.update({"exp": expire})
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create access token"
            )
    
    def verify_token(self, token: str) -> Optional[str]:
        """Verify JWT token and return username"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            
            if username is None:
                return None
                
            return username
        except InvalidTokenError:
            return None
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """Decode JWT token and return payload"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

# Global security service instance
security_service = SecurityService()
```

### Task 4: Database Models and Schemas
```python
# src/models/user.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Index
from sqlalchemy.sql import func
from src.core.database import Base

class User(Base):
    """User database model with optimized indexing"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(200), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    bio = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now(),
        nullable=False
    )

    # Composite indexes for common queries
    __table_args__ = (
        Index('idx_user_email_active', 'email', 'is_active'),
        Index('idx_user_username_active', 'username', 'is_active'),
    )

# src/models/schemas.py
from pydantic import BaseModel, Field, EmailStr, ConfigDict
from typing import Optional
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    """User role enumeration"""
    USER = "user"
    ADMIN = "admin"

class UserBase(BaseModel):
    """Base user schema with common fields"""
    email: EmailStr = Field(..., description="User email address")
    username: str = Field(..., min_length=3, max_length=100, description="Username")
    full_name: Optional[str] = Field(None, max_length=200, description="Full name")
    bio: Optional[str] = Field(None, max_length=1000, description="User biography")

class UserCreate(UserBase):
    """Schema for user creation"""
    password: str = Field(..., min_length=8, max_length=100, description="User password")

class UserUpdate(BaseModel):
    """Schema for user updates"""
    email: Optional[EmailStr] = Field(None, description="Updated email address")
    username: Optional[str] = Field(None, min_length=3, max_length=100, description="Updated username")
    full_name: Optional[str] = Field(None, max_length=200, description="Updated full name")
    bio: Optional[str] = Field(None, max_length=1000, description="Updated biography")
    is_active: Optional[bool] = Field(None, description="User active status")

class UserResponse(UserBase):
    """Schema for user responses"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int = Field(..., description="User ID")
    is_active: bool = Field(..., description="User active status")
    is_superuser: bool = Field(..., description="User superuser status")
    created_at: datetime = Field(..., description="User creation timestamp")
    updated_at: datetime = Field(..., description="User last update timestamp")

class UserInDB(UserResponse):
    """Schema for user with password hash (internal use)"""
    hashed_password: str = Field(..., description="Hashed password")

class Token(BaseModel):
    """Token response schema"""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")

class TokenData(BaseModel):
    """Token data schema for internal use"""
    username: Optional[str] = Field(None, description="Username from token")

class LoginResponse(BaseModel):
    """Login response schema"""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user: UserResponse = Field(..., description="User information")

# Error response schemas
class ErrorResponse(BaseModel):
    """Error response schema"""
    detail: str = Field(..., description="Error description")
    error_code: Optional[str] = Field(None, description="Error code")

class ValidationErrorResponse(BaseModel):
    """Validation error response schema"""
    detail: list = Field(..., description="Validation error details")
    error_code: str = Field(default="VALIDATION_ERROR", description="Error code")
```

### Task 5: Service Layer Implementation
```python
# src/services/user_service.py
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from fastapi import HTTPException, status
from src.models.user import User
from src.models.schemas import UserCreate, UserUpdate, UserResponse, UserInDB
from src.core.security import security_service
from src.utils.exceptions import UserNotFoundError, UserAlreadyExistsError

class UserService:
    """User service for business logic operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create a new user with validation"""
        # Check if user already exists
        existing_user = await self.get_user_by_email(user_data.email)
        if existing_user:
            raise UserAlreadyExistsError("Email already registered")
        
        existing_username = await self.get_user_by_username(user_data.username)
        if existing_username:
            raise UserAlreadyExistsError("Username already taken")
        
        # Create user
        hashed_password = security_service.get_password_hash(user_data.password)
        db_user = User(
            email=user_data.email,
            username=user_data.username,
            full_name=user_data.full_name,
            bio=user_data.bio,
            hashed_password=hashed_password,
            is_active=True,
            is_superuser=False
        )
        
        self.db.add(db_user)
        await self.db.commit()
        await self.db.refresh(db_user)
        
        return UserResponse.model_validate(db_user)
    
    async def get_user_by_id(self, user_id: int) -> Optional[UserResponse]:
        """Get user by ID"""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            return None
        
        return UserResponse.model_validate(user)
    
    async def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        """Get user by email with password hash"""
        result = await self.db.execute(
            select(User).where(User.email == email)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            return None
        
        return UserInDB.model_validate(user)
    
    async def get_user_by_username(self, username: str) -> Optional[UserInDB]:
        """Get user by username with password hash"""
        result = await self.db.execute(
            select(User).where(User.username == username)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            return None
        
        return UserInDB.model_validate(user)
    
    async def update_user(self, user_id: int, user_update: UserUpdate) -> UserResponse:
        """Update user information"""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise UserNotFoundError(f"User with id {user_id} not found")
        
        # Update fields if provided
        update_data = user_update.model_dump(exclude_unset=True)
        
        # Check for email/username conflicts
        if "email" in update_data and update_data["email"] != user.email:
            existing_email = await self.get_user_by_email(update_data["email"])
            if existing_email:
                raise UserAlreadyExistsError("Email already in use")
        
        if "username" in update_data and update_data["username"] != user.username:
            existing_username = await self.get_user_by_username(update_data["username"])
            if existing_username:
                raise UserAlreadyExistsError("Username already in use")
        
        # Update user
        for field, value in update_data.items():
            setattr(user, field, value)
        
        await self.db.commit()
        await self.db.refresh(user)
        
        return UserResponse.model_validate(user)
    
    async def list_users(
        self, 
        skip: int = 0, 
        limit: int = 100,
        active_only: bool = True
    ) -> List[UserResponse]:
        """List users with pagination"""
        query = select(User)
        
        if active_only:
            query = query.where(User.is_active == True)
        
        query = query.offset(skip).limit(limit).order_by(User.created_at.desc())
        
        result = await self.db.execute(query)
        users = result.scalars().all()
        
        return [UserResponse.model_validate(user) for user in users]
    
    async def deactivate_user(self, user_id: int) -> UserResponse:
        """Deactivate user account"""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise UserNotFoundError(f"User with id {user_id} not found")
        
        user.is_active = False
        await self.db.commit()
        await self.db.refresh(user)
        
        return UserResponse.model_validate(user)
    
    async def get_user_count(self) -> int:
        """Get total user count"""
        result = await self.db.execute(select(func.count(User.id)))
        return result.scalar_one()

# src/services/auth_service.py
from typing import Optional
from datetime import timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, status
from src.models.user import User
from src.models.schemas import UserCreate, UserResponse, LoginResponse
from src.core.security import security_service
from src.services.user_service import UserService
from src.utils.exceptions import InvalidCredentialsError

class AuthService:
    """Authentication service for login and registration"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.user_service = UserService(db)
    
    async def register_user(self, user_data: UserCreate) -> UserResponse:
        """Register new user"""
        return await self.user_service.create_user(user_data)
    
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        user = await self.user_service.get_user_by_email(email)
        
        if not user:
            return None
        
        if not security_service.verify_password(password, user.hashed_password):
            return None
        
        return user
    
    async def login(self, email: str, password: str) -> LoginResponse:
        """Login user and return access token"""
        user = await self.authenticate_user(email, password)
        
        if not user:
            raise InvalidCredentialsError("Incorrect email or password")
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Account is deactivated"
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=security_service.access_token_expire_minutes)
        access_token = security_service.create_access_token(
            data={"sub": user.username}, 
            expires_delta=access_token_expires
        )
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=security_service.access_token_expire_minutes * 60,
            user=UserResponse.model_validate(user)
        )
```

### Task 6: API Endpoints with OpenAPI Documentation
```python
# src/api/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated
from src.core.database import get_db
from src.core.security import security_service
from src.services.auth_service import AuthService
from src.services.user_service import UserService
from src.models.schemas import UserCreate, UserResponse, LoginResponse, Token
from src.utils.exceptions import InvalidCredentialsError, UserAlreadyExistsError

router = APIRouter(prefix="/auth", tags=["Authentication"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    username = security_service.verify_token(token)
    if username is None:
        raise credentials_exception
    
    user_service = UserService(db)
    user = await user_service.get_user_by_username(username)
    if user is None:
        raise credentials_exception
    
    return UserResponse.model_validate(user)

async def get_current_active_user(
    current_user: Annotated[UserResponse, Depends(get_current_user)]
) -> UserResponse:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a new user account with email, username, and password",
    responses={
        201: {"description": "User created successfully"},
        400: {"description": "Email or username already exists"},
        422: {"description": "Validation error"},
    }
)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    Register a new user account.
    
    - **email**: Valid email address (must be unique)
    - **username**: Username (3-100 characters, must be unique)
    - **password**: Password (minimum 8 characters)
    - **full_name**: Optional full name
    - **bio**: Optional user biography
    """
    try:
        auth_service = AuthService(db)
        return await auth_service.register_user(user_data)
    except UserAlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Login user",
    description="Authenticate user and return access token",
    responses={
        200: {"description": "Login successful"},
        401: {"description": "Invalid credentials"},
        400: {"description": "Account deactivated"},
    }
)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
) -> LoginResponse:
    """
    Login with email and password.
    
    - **username**: Email address (OAuth2 standard uses 'username' field)
    - **password**: User password
    
    Returns JWT access token for subsequent requests.
    """
    try:
        auth_service = AuthService(db)
        return await auth_service.login(form_data.username, form_data.password)
    except InvalidCredentialsError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
    description="Get profile information of authenticated user",
    responses={
        200: {"description": "User profile retrieved"},
        401: {"description": "Invalid or expired token"},
    }
)
async def get_current_user_profile(
    current_user: Annotated[UserResponse, Depends(get_current_active_user)]
) -> UserResponse:
    """
    Get current user profile information.
    
    Returns the authenticated user's profile data.
    Requires valid JWT token in Authorization header.
    """
    return current_user

# src/api/users.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated, List
from src.core.database import get_db
from src.services.user_service import UserService
from src.models.schemas import UserResponse, UserUpdate
from src.api.auth import get_current_active_user
from src.utils.exceptions import UserNotFoundError, UserAlreadyExistsError

router = APIRouter(prefix="/users", tags=["Users"])

@router.get(
    "/",
    response_model=List[UserResponse],
    summary="List users",
    description="Get a paginated list of users",
    responses={
        200: {"description": "List of users retrieved"},
        401: {"description": "Authentication required"},
    }
)
async def list_users(
    skip: int = Query(0, ge=0, description="Number of users to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of users to return"),
    active_only: bool = Query(True, description="Only return active users"),
    current_user: Annotated[UserResponse, Depends(get_current_active_user)] = None,
    db: AsyncSession = Depends(get_db)
) -> List[UserResponse]:
    """
    List users with pagination.
    
    - **skip**: Number of users to skip (default: 0)
    - **limit**: Maximum number of users to return (default: 100, max: 1000)
    - **active_only**: Only return active users (default: true)
    
    Requires authentication.
    """
    user_service = UserService(db)
    return await user_service.list_users(skip=skip, limit=limit, active_only=active_only)

@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Get user by ID",
    description="Get detailed information about a specific user",
    responses={
        200: {"description": "User information retrieved"},
        404: {"description": "User not found"},
        401: {"description": "Authentication required"},
    }
)
async def get_user(
    user_id: int,
    current_user: Annotated[UserResponse, Depends(get_current_active_user)] = None,
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    Get user information by ID.
    
    - **user_id**: User ID to retrieve
    
    Returns detailed user information.
    """
    user_service = UserService(db)
    user = await user_service.get_user_by_id(user_id)
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
        200: {"description": "Profile updated successfully"},
        400: {"description": "Email or username already exists"},
        401: {"description": "Authentication required"},
    }
)
async def update_current_user(
    user_update: UserUpdate,
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    Update current user profile.
    
    - **email**: New email address (optional)
    - **username**: New username (optional)
    - **full_name**: New full name (optional)
    - **bio**: New biography (optional)
    
    Updates only the fields provided.
    """
    try:
        user_service = UserService(db)
        return await user_service.update_user(current_user.id, user_update)
    except UserAlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.delete(
    "/me",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Deactivate current user",
    description="Deactivate the authenticated user's account",
    responses={
        204: {"description": "Account deactivated successfully"},
        401: {"description": "Authentication required"},
    }
)
async def deactivate_current_user(
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    db: AsyncSession = Depends(get_db)
) -> None:
    """
    Deactivate current user account.
    
    This will set the user's account to inactive status.
    The user will not be able to login after deactivation.
    """
    user_service = UserService(db)
    await user_service.deactivate_user(current_user.id)
```

### Task 7: Exception Handling and Utilities
```python
# src/utils/exceptions.py
from fastapi import HTTPException, status

class UserManagementException(Exception):
    """Base exception for user management operations"""
    pass

class UserNotFoundError(UserManagementException):
    """Exception raised when user is not found"""
    pass

class UserAlreadyExistsError(UserManagementException):
    """Exception raised when user already exists"""
    pass

class InvalidCredentialsError(UserManagementException):
    """Exception raised when credentials are invalid"""
    pass

class InsufficientPermissionsError(UserManagementException):
    """Exception raised when user lacks required permissions"""
    pass

# HTTP exception mapping
def map_exception_to_http(exception: Exception) -> HTTPException:
    """Map internal exceptions to HTTP exceptions"""
    if isinstance(exception, UserNotFoundError):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exception)
        )
    elif isinstance(exception, UserAlreadyExistsError):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exception)
        )
    elif isinstance(exception, InvalidCredentialsError):
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exception),
            headers={"WWW-Authenticate": "Bearer"},
        )
    elif isinstance(exception, InsufficientPermissionsError):
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exception)
        )
    else:
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
```

### Task 8: Comprehensive Async Testing (2024 Patterns)
```python
# tests/conftest.py
import pytest
import asyncio
from typing import AsyncGenerator
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool
from src.main import app
from src.core.database import get_db, Base
from src.models.user import User
from src.core.security import security_service

# Test database URL (in-memory SQLite)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for the session"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine"""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    await engine.dispose()

@pytest.fixture
async def test_db(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session"""
    async_session_maker = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with async_session_maker() as session:
        yield session

@pytest.fixture
async def client(test_db: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create test client with test database"""
    app.dependency_overrides[get_db] = lambda: test_db
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
    
    app.dependency_overrides.clear()

@pytest.fixture
async def test_user(test_db: AsyncSession) -> User:
    """Create test user"""
    user = User(
        email="test@example.com",
        username="testuser",
        full_name="Test User",
        hashed_password=security_service.get_password_hash("testpassword123"),
        is_active=True
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    return user

@pytest.fixture
async def admin_user(test_db: AsyncSession) -> User:
    """Create admin user"""
    user = User(
        email="admin@example.com",
        username="admin",
        full_name="Admin User",
        hashed_password=security_service.get_password_hash("adminpassword123"),
        is_active=True,
        is_superuser=True
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    return user

@pytest.fixture
async def auth_headers(client: AsyncClient, test_user: User) -> dict:
    """Create authentication headers for test user"""
    login_response = await client.post("/auth/login", data={
        "username": test_user.email,
        "password": "testpassword123"
    })
    token = login_response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

# tests/test_api/test_auth.py
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.user import User

@pytest.mark.asyncio
async def test_register_user_success(client: AsyncClient):
    """Test successful user registration"""
    user_data = {
        "email": "newuser@example.com",
        "username": "newuser",
        "full_name": "New User",
        "password": "newpassword123"
    }
    
    response = await client.post("/auth/register", json=user_data)
    
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == user_data["email"]
    assert data["username"] == user_data["username"]
    assert data["full_name"] == user_data["full_name"]
    assert "id" in data
    assert "created_at" in data
    assert "hashed_password" not in data

@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient, test_user: User):
    """Test registration with duplicate email"""
    user_data = {
        "email": test_user.email,
        "username": "differentuser",
        "password": "password123"
    }
    
    response = await client.post("/auth/register", json=user_data)
    
    assert response.status_code == 400
    assert "already" in response.json()["detail"].lower()

@pytest.mark.asyncio
async def test_register_invalid_email(client: AsyncClient):
    """Test registration with invalid email"""
    user_data = {
        "email": "invalid-email",
        "username": "testuser",
        "password": "password123"
    }
    
    response = await client.post("/auth/register", json=user_data)
    
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_register_weak_password(client: AsyncClient):
    """Test registration with weak password"""
    user_data = {
        "email": "test@example.com",
        "username": "testuser",
        "password": "weak"
    }
    
    response = await client.post("/auth/register", json=user_data)
    
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_login_success(client: AsyncClient, test_user: User):
    """Test successful login"""
    login_data = {
        "username": test_user.email,
        "password": "testpassword123"
    }
    
    response = await client.post("/auth/login", data=login_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert "expires_in" in data
    assert "user" in data
    assert data["user"]["email"] == test_user.email

@pytest.mark.asyncio
async def test_login_invalid_credentials(client: AsyncClient, test_user: User):
    """Test login with invalid credentials"""
    login_data = {
        "username": test_user.email,
        "password": "wrongpassword"
    }
    
    response = await client.post("/auth/login", data=login_data)
    
    assert response.status_code == 401
    assert "incorrect" in response.json()["detail"].lower()

@pytest.mark.asyncio
async def test_login_nonexistent_user(client: AsyncClient):
    """Test login with nonexistent user"""
    login_data = {
        "username": "nonexistent@example.com",
        "password": "password123"
    }
    
    response = await client.post("/auth/login", data=login_data)
    
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_get_current_user_success(client: AsyncClient, auth_headers: dict):
    """Test getting current user info with valid token"""
    response = await client.get("/auth/me", headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    assert "email" in data
    assert "username" in data
    assert "id" in data
    assert "is_active" in data

@pytest.mark.asyncio
async def test_get_current_user_invalid_token(client: AsyncClient):
    """Test getting current user with invalid token"""
    response = await client.get(
        "/auth/me",
        headers={"Authorization": "Bearer invalid_token"}
    )
    
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_get_current_user_no_token(client: AsyncClient):
    """Test getting current user without token"""
    response = await client.get("/auth/me")
    
    assert response.status_code == 401

# tests/test_integration/test_user_flow.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_complete_user_flow(client: AsyncClient):
    """Test complete user registration and management flow"""
    # 1. Register user
    user_data = {
        "email": "flowtest@example.com",
        "username": "flowtest",
        "full_name": "Flow Test User",
        "password": "flowpassword123"
    }
    
    register_response = await client.post("/auth/register", json=user_data)
    assert register_response.status_code == 201
    user_id = register_response.json()["id"]
    
    # 2. Login
    login_response = await client.post("/auth/login", data={
        "username": user_data["email"],
        "password": user_data["password"]
    })
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 3. Get current user
    me_response = await client.get("/auth/me", headers=headers)
    assert me_response.status_code == 200
    assert me_response.json()["email"] == user_data["email"]
    
    # 4. Update profile
    update_data = {
        "full_name": "Updated Flow Test User",
        "bio": "This is my updated bio"
    }
    update_response = await client.put("/users/me", json=update_data, headers=headers)
    assert update_response.status_code == 200
    assert update_response.json()["full_name"] == update_data["full_name"]
    assert update_response.json()["bio"] == update_data["bio"]
    
    # 5. Get user by ID
    user_response = await client.get(f"/users/{user_id}", headers=headers)
    assert user_response.status_code == 200
    assert user_response.json()["full_name"] == update_data["full_name"]
    
    # 6. List users
    list_response = await client.get("/users/", headers=headers)
    assert list_response.status_code == 200
    users = list_response.json()
    assert len(users) >= 1
    assert any(user["id"] == user_id for user in users)
    
    # 7. Deactivate account
    deactivate_response = await client.delete("/users/me", headers=headers)
    assert deactivate_response.status_code == 204
    
    # 8. Verify login fails after deactivation
    login_after_deactivation = await client.post("/auth/login", data={
        "username": user_data["email"],
        "password": user_data["password"]
    })
    assert login_after_deactivation.status_code == 400

@pytest.mark.asyncio
async def test_user_flow_with_validation_errors(client: AsyncClient):
    """Test user flow with various validation scenarios"""
    # Invalid email format
    invalid_email_data = {
        "email": "not-an-email",
        "username": "testuser",
        "password": "password123"
    }
    
    response = await client.post("/auth/register", json=invalid_email_data)
    assert response.status_code == 422
    
    # Password too short
    short_password_data = {
        "email": "test@example.com",
        "username": "testuser",
        "password": "short"
    }
    
    response = await client.post("/auth/register", json=short_password_data)
    assert response.status_code == 422
    
    # Username too short
    short_username_data = {
        "email": "test@example.com",
        "username": "ab",
        "password": "password123"
    }
    
    response = await client.post("/auth/register", json=short_username_data)
    assert response.status_code == 422
```

### Task 9: Database Migrations with Alembic
```bash
# Initialize Alembic
cd python-env && devbox shell
uv run alembic init alembic

# Create initial migration
uv run alembic revision --autogenerate -m "Create users table"

# Apply migrations
uv run alembic upgrade head
```

```python
# alembic/env.py (key modifications)
from sqlalchemy import engine_from_config, pool
from alembic import context
from src.core.database import Base
from src.models.user import User  # Import all models

# Set target metadata
target_metadata = Base.metadata

# Configure for async
def run_migrations_online():
    """Run migrations in 'online' mode with async support"""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()
```

### Task 10: Application Integration and Configuration
```python
# src/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from src.core.config import settings
from src.core.database import init_db, close_db
from src.api import auth, users
from src.utils.exceptions import UserManagementException, map_exception_to_http

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting User Management API...")
    await init_db()
    logger.info("Database initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down User Management API...")
    await close_db()
    logger.info("Database connections closed")

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-ready user management REST API with JWT authentication",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)

# Global exception handler
@app.exception_handler(UserManagementException)
async def user_management_exception_handler(request: Request, exc: UserManagementException):
    """Handle custom user management exceptions"""
    http_exc = map_exception_to_http(exc)
    return JSONResponse(
        status_code=http_exc.status_code,
        content={"detail": http_exc.detail}
    )

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/users", tags=["Users"])

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": settings.app_version}

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "User Management API",
        "version": settings.app_version,
        "docs_url": "/docs",
        "health_url": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

### Task 11: Development Configuration
```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: user_management_dev
      POSTGRES_USER: dev_user
      POSTGRES_PASSWORD: dev_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U dev_user -d user_management_dev"]
      interval: 10s
      timeout: 5s
      retries: 5

  postgres_test:
    image: postgres:15
    environment:
      POSTGRES_DB: user_management_test
      POSTGRES_USER: test_user
      POSTGRES_PASSWORD: test_password
    ports:
      - "5433:5432"
    volumes:
      - postgres_test_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U test_user -d user_management_test"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
  postgres_test_data:
```

```bash
# .env.example
# Application Configuration
APP_NAME=User Management API
APP_VERSION=1.0.0
DEBUG=false

# Database Configuration
DATABASE_URL=postgresql+asyncpg://dev_user:dev_password@localhost:5432/user_management_dev
DATABASE_ECHO=false

# Security Configuration
SECRET_KEY=your-super-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS Configuration
ALLOWED_ORIGINS=["http://localhost:3000", "http://localhost:8000"]
ALLOWED_METHODS=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
ALLOWED_HEADERS=["*"]

# Security Headers
INCLUDE_SECURITY_HEADERS=true
```

### Task 12: Documentation and Integration
```markdown
# README.md
# User Management API

Production-ready FastAPI application with JWT authentication, async PostgreSQL database, and comprehensive testing.

## Features

- 🔐 **JWT Authentication**: Secure user authentication with PyJWT
- 📊 **Async Database**: PostgreSQL with SQLAlchemy 2.0 async support
- 🧪 **Comprehensive Testing**: 90%+ test coverage with pytest-asyncio
- 📝 **API Documentation**: Interactive OpenAPI/Swagger documentation
- 🔒 **Security**: Password hashing, input validation, SQL injection protection
- 🚀 **Performance**: Async/await throughout, connection pooling
- 🔧 **Modern Stack**: FastAPI, Pydantic v2, SQLAlchemy 2.0, PyJWT

## Quick Start

### Prerequisites
- Python 3.12+
- PostgreSQL 15+
- uv package manager

### Installation

1. **Setup environment:**
   ```bash
   cd python-env && devbox shell
   ```

2. **Install dependencies:**
   ```bash
   uv sync --dev
   ```

3. **Start PostgreSQL:**
   ```bash
   docker-compose up -d postgres
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run database migrations:**
   ```bash
   uv run alembic upgrade head
   ```

6. **Start the API:**
   ```bash
   uv run uvicorn src.main:app --reload
   ```

### API Documentation

Visit http://localhost:8000/docs for interactive API documentation.

## API Endpoints

### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - Login user
- `GET /auth/me` - Get current user

### Users
- `GET /users/` - List users (paginated)
- `GET /users/{id}` - Get user by ID
- `PUT /users/me` - Update current user
- `DELETE /users/me` - Deactivate account

### Health
- `GET /health` - Health check
- `GET /` - API information

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=src --cov-report=html

# Run specific test categories
uv run pytest tests/test_api/ -v
uv run pytest tests/test_integration/ -v
```

## Development

### Code Quality

```bash
# Format code
uv run ruff format src/ tests/

# Lint code
uv run ruff check src/ tests/ --fix

# Type checking
uv run mypy src/ tests/
```

### Database Operations

```bash
# Create migration
uv run alembic revision --autogenerate -m "Description"

# Apply migrations
uv run alembic upgrade head

# Rollback migration
uv run alembic downgrade -1
```

## Deployment

### Environment Variables

Required environment variables for production:

- `DATABASE_URL`: PostgreSQL connection string
- `SECRET_KEY`: JWT secret key (generate with `openssl rand -hex 32`)
- `ALLOWED_ORIGINS`: Comma-separated list of allowed origins

### Security Considerations

- Generate strong `SECRET_KEY` for production
- Use environment variables for all secrets
- Enable HTTPS in production
- Configure proper CORS settings
- Implement rate limiting (optional)
- Set up monitoring and logging

## Performance Monitoring

This API integrates with the polyglot environment's monitoring systems:

- Performance analytics via `nushell-env/scripts/performance-analytics.nu`
- Security scanning via `nushell-env/scripts/security-scanner.nu`
- Resource monitoring via `nushell-env/scripts/resource-monitor.nu`

## Architecture

```
src/
├── core/          # Configuration, database, security
├── models/        # Database models and schemas
├── api/           # API route handlers
├── services/      # Business logic
└── utils/         # Utilities and exceptions
```

## Contributing

1. Follow the existing code style and patterns
2. Add tests for new functionality
3. Update documentation
4. Run quality checks before committing
```

## List of Tasks to Complete

```yaml
Task 1: Environment Setup and Dependencies
  COMMAND: cd python-env && devbox shell
  INSTALL: uv add "fastapi[all]" "uvicorn[standard]" "sqlalchemy[asyncio]>=2.0.0" "asyncpg" "alembic" "PyJWT" "passlib[bcrypt]" "python-multipart" "pydantic[email]" "pydantic-settings"
  INSTALL_DEV: uv add --dev "pytest-asyncio" "httpx" "pytest-cov" "aiosqlite"
  VERIFY: uv run python -c "import fastapi, sqlalchemy, jwt, passlib; print('Success')"

Task 2: Core Infrastructure
  CREATE: src/core/config.py (Pydantic settings with environment variables)
  CREATE: src/core/database.py (Async SQLAlchemy session management)
  CREATE: src/core/security.py (PyJWT authentication with bcrypt)
  PATTERN: Use Pydantic v2 patterns and SQLAlchemy 2.0 async patterns

Task 3: Database Models and Schemas
  CREATE: src/models/user.py (SQLAlchemy User model with indexes)
  CREATE: src/models/schemas.py (Pydantic request/response schemas)
  PATTERN: Use SQLAlchemy 2.0 declarative base and Pydantic v2 ConfigDict

Task 4: Service Layer
  CREATE: src/services/user_service.py (User business logic)
  CREATE: src/services/auth_service.py (Authentication business logic)
  PATTERN: Async service methods with proper error handling

Task 5: API Endpoints
  CREATE: src/api/auth.py (Authentication endpoints)
  CREATE: src/api/users.py (User management endpoints)
  PATTERN: FastAPI dependency injection with comprehensive OpenAPI docs

Task 6: Exception Handling
  CREATE: src/utils/exceptions.py (Custom exception classes)
  PATTERN: Map internal exceptions to HTTP status codes

Task 7: Database Migrations
  COMMAND: uv run alembic init alembic
  COMMAND: uv run alembic revision --autogenerate -m "Create users table"
  COMMAND: uv run alembic upgrade head
  PATTERN: Async-compatible Alembic configuration

Task 8: Comprehensive Testing
  CREATE: tests/conftest.py (Async test fixtures)
  CREATE: tests/test_api/ (API endpoint tests)
  CREATE: tests/test_integration/ (End-to-end flow tests)
  PATTERN: pytest-asyncio with httpx AsyncClient

Task 9: Application Integration
  CREATE: src/main.py (FastAPI app with lifespan, CORS, exception handlers)
  PATTERN: Production-ready FastAPI configuration

Task 10: Development Environment
  CREATE: docker-compose.yml (PostgreSQL development database)
  CREATE: .env.example (Environment variables template)
  PATTERN: Development-friendly configuration

Task 11: Documentation
  CREATE: README.md (Comprehensive API documentation)
  PATTERN: Clear setup instructions and API overview

Task 12: Monitoring Integration
  INTEGRATE: Performance analytics tracking
  INTEGRATE: Security scanning compatibility
  PATTERN: Use existing polyglot monitoring systems
```

## Validation Loop

### Level 1: Code Quality and Standards
```bash
cd python-env && devbox shell

# Install all dependencies
uv sync --dev

# Format code
uv run ruff format src/ tests/

# Lint code
uv run ruff check src/ tests/ --fix

# Type checking
uv run mypy src/ tests/

# Expected: All checks pass without errors
```

### Level 2: Database and Migration Testing
```bash
# Start test database
docker-compose up -d postgres

# Run migrations
uv run alembic upgrade head

# Verify database schema
uv run python -c "from src.core.database import engine; print('Database connection successful')"

# Expected: Database initializes without errors
```

### Level 3: Comprehensive Testing
```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# Run specific test categories
uv run pytest tests/test_api/ -v
uv run pytest tests/test_services/ -v
uv run pytest tests/test_integration/ -v

# Expected: All tests pass, 90%+ coverage
```

### Level 4: API Testing and Documentation
```bash
# Start the API server
uv run uvicorn src.main:app --reload --port 8000

# Test endpoints manually
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "username": "testuser", "password": "testpass123"}'

# Check API documentation
open http://localhost:8000/docs

# Expected: API responds correctly, documentation is comprehensive
```

### Level 5: Performance and Security Validation
```bash
# Run performance analytics
nu ../nushell-env/scripts/performance-analytics.nu record "api-test" "python-env" "30s" --status "success"

# Run security scan
nu ../nushell-env/scripts/security-scanner.nu scan-directory "src/" --quiet

# Expected: Integration with monitoring systems successful
```

## Final Validation Checklist

- [ ] Environment setup successful: `devbox shell` activates Python 3.12+
- [ ] All dependencies installed: PyJWT, FastAPI, SQLAlchemy async, etc.
- [ ] Code quality passes: ruff format, ruff check, mypy all clean
- [ ] Database migrations work: Alembic creates and applies migrations
- [ ] All tests pass: 90%+ coverage with async testing patterns
- [ ] API server starts: uvicorn runs without errors
- [ ] API endpoints respond: Registration, login, user management work
- [ ] JWT authentication functional: Token creation and validation work
- [ ] OpenAPI documentation complete: /docs shows comprehensive API info
- [ ] Database operations work: CRUD operations execute successfully
- [ ] Security implemented: Password hashing, input validation, JWT tokens
- [ ] Error handling comprehensive: Proper HTTP status codes and messages
- [ ] Performance monitoring integrated: Analytics and scanning systems work
- [ ] Production configuration ready: Environment variables, Docker compose

## Integration with Polyglot Environment

### Performance Analytics Integration
```bash
# The API automatically integrates with performance monitoring
# Performance data is tracked via:
nu ../nushell-env/scripts/performance-analytics.nu record "user-api" "python-env" "$duration" --status "$status"
```

### Security Scanning Integration
```bash
# Security scanning is performed on code changes
nu ../nushell-env/scripts/security-scanner.nu scan-directory "src/" --format "json"
```

### Resource Monitoring Integration
```bash
# Resource usage is tracked during operations
nu ../nushell-env/scripts/resource-monitor.nu track "user-api" --interval "30s"
```

---

## Production Readiness Features

### Security
- PyJWT for secure token management (replaces deprecated python-jose)
- Bcrypt password hashing with salt
- Input validation with Pydantic v2
- SQL injection protection via SQLAlchemy ORM
- CORS configuration
- Security headers middleware

### Performance
- Async/await throughout the entire stack
- Connection pooling with SQLAlchemy async engine
- Efficient database queries with proper indexing
- Pagination for list endpoints
- Query optimization patterns

### Reliability
- Comprehensive error handling with custom exceptions
- Database transaction management
- Proper session handling with async context managers
- Health check endpoints
- Graceful shutdown with lifespan events

### Monitoring & Observability
- Structured logging with correlation IDs
- Performance metrics tracking
- Integration with existing polyglot monitoring systems
- Comprehensive test coverage with performance tracking
- Resource usage monitoring

### Developer Experience
- Interactive OpenAPI documentation
- Comprehensive type hints for IDE support
- Clear error messages with detailed responses
- Development environment with Docker Compose
- Hot reload during development
- Comprehensive README with setup instructions

This PRP provides complete context for implementing a production-ready user management API that follows 2024 best practices, integrates seamlessly with the polyglot environment's monitoring systems, and provides a solid foundation for additional microservices in the ecosystem.