name: "Python FastAPI REST API with Database Integration"
description: |

## Purpose
Example PRP demonstrating a complete Python FastAPI REST API implementation with PostgreSQL database integration, comprehensive testing, and deployment configuration.

## Core Principles
1. **FastAPI Best Practices**: Async endpoints, dependency injection, comprehensive documentation
2. **Database Integration**: SQLAlchemy async ORM with Alembic migrations
3. **Production Ready**: Error handling, logging, monitoring, security
4. **Comprehensive Testing**: Unit, integration, and end-to-end tests
5. **Type Safety**: Strict mypy validation and Pydantic models

---

## Goal
Build a production-ready user management REST API with CRUD operations, authentication, and comprehensive testing in the python-env environment.

## Why
- **Business value**: Foundational user management system for applications
- **Integration**: Template for other microservices in the polyglot environment
- **Learning**: Demonstrates FastAPI patterns and testing strategies

## What
RESTful API with user registration, authentication, profile management, and admin operations.

### Success Criteria
- [ ] Complete CRUD operations for user management
- [ ] JWT-based authentication system
- [ ] PostgreSQL database integration with migrations
- [ ] Comprehensive test coverage (90%+)
- [ ] OpenAPI documentation with examples
- [ ] Production deployment configuration

## All Needed Context

### Target Environment
```yaml
Environment: python-env
Devbox_Config: python-env/devbox.json
Dependencies: [fastapi, uvicorn, sqlalchemy, alembic, asyncpg, python-jose, passlib, pytest-asyncio]
Python_Version: 3.12+ (as specified in devbox.json)
Package_Manager: uv (exclusively - no pip/poetry/pipenv)
```

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://fastapi.tiangolo.com/tutorial/
  why: FastAPI patterns, dependency injection, async handling
  
- url: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
  why: SQLAlchemy async patterns and session management
  
- file: python-env/src/main.py
  why: Existing FastAPI application structure
  
- doc: https://alembic.sqlalchemy.org/en/latest/
  section: Database migrations and schema management
  critical: Use Alembic for all schema changes
  
- doc: https://docs.pydantic.dev/
  section: Model validation and serialization patterns
  critical: Use Pydantic v2 syntax and features
```

### Current Codebase tree
```bash
python-env/
├── devbox.json         # Python 3.12, uv, ruff, mypy, pytest
├── src/
│   ├── __init__.py
│   ├── main.py         # FastAPI app entry point
│   ├── api/            # API routes and endpoints
│   ├── models/         # Pydantic models and schemas
│   ├── services/       # Business logic layer
│   └── utils/          # Shared utilities
├── tests/
│   ├── __init__.py
│   ├── conftest.py     # pytest configuration
│   ├── test_api/       # API endpoint tests
│   └── test_services/  # Service layer tests
├── pyproject.toml      # uv dependencies and tool config
└── README.md
```

### Desired Codebase tree with files to be added
```bash
python-env/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # Application configuration
│   │   ├── database.py         # Database connection and session
│   │   ├── security.py         # Authentication and authorization
│   │   └── dependencies.py     # FastAPI dependencies
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py            # User SQLAlchemy model
│   │   └── schemas.py         # Pydantic schemas
│   ├── api/
│   │   ├── __init__.py
│   │   ├── auth.py            # Authentication endpoints
│   │   ├── users.py           # User management endpoints
│   │   └── admin.py           # Admin operations
│   ├── services/
│   │   ├── __init__.py
│   │   ├── user_service.py    # User business logic
│   │   └── auth_service.py    # Authentication logic
│   └── utils/
│       ├── __init__.py
│       └── exceptions.py      # Custom exception classes
├── alembic/
│   ├── versions/              # Database migrations
│   ├── env.py                 # Alembic configuration
│   └── script.py.mako         # Migration template
├── tests/
│   ├── test_api/
│   │   ├── test_auth.py
│   │   ├── test_users.py
│   │   └── test_admin.py
│   ├── test_services/
│   │   ├── test_user_service.py
│   │   └── test_auth_service.py
│   └── test_integration/
│       └── test_user_flow.py
├── alembic.ini             # Alembic configuration
├── docker-compose.yml      # PostgreSQL for development
└── .env.example           # Environment variables template
```

### Known Gotchas of Python Environment
```python
# CRITICAL: Python environment-specific gotchas
# uv: Use 'uv add package' not 'pip install package'
# uv: Run commands with 'uv run command' for proper environment
# FastAPI: All endpoints must be async def for proper performance
# SQLAlchemy: Use async session and await all database operations
# Pydantic: Use v2 syntax - Field() instead of field parameters
# Alembic: Generate migrations with proper async support
# Testing: Use pytest-asyncio for FastAPI testing
# JWT: Store secret keys securely, never in code

# Example patterns:
# ✅ uv add fastapi[all] sqlalchemy[asyncio] alembic
# ✅ async def create_user(user_data: UserCreate, db: AsyncSession = Depends(get_db))
# ✅ result = await db.execute(select(User).where(User.email == email))
# ✅ from pydantic import BaseModel, Field, ConfigDict
# ❌ pip install anything
# ❌ def sync_endpoint():  # Use async def
# ❌ db.query(User)  # Use async session methods
```

## Implementation Blueprint

### Environment Setup
```bash
# Activate Python environment
cd python-env && devbox shell

# Verify environment
uv --version
python --version  # Should be 3.12+

# Install additional dependencies
uv add fastapi[all] sqlalchemy[asyncio] alembic asyncpg python-jose[cryptography] passlib[bcrypt] pytest-asyncio httpx
```

### Database Models and Configuration
```python
# src/core/config.py
from pydantic import BaseSettings, Field
from typing import Optional

class Settings(BaseSettings):
    """Application configuration"""
    
    # Application
    app_name: str = "User Management API"
    debug: bool = False
    version: str = "1.0.0"
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    database_echo: bool = Field(False, env="DATABASE_ECHO")
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # CORS
    allowed_origins: list[str] = ["http://localhost:3000"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()

# src/core/database.py
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from typing import AsyncGenerator
from .config import settings

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.database_echo,
    future=True
)

# Create async session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for models
Base = declarative_base()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session"""
    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# src/models/user.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.sql import func
from src.core.database import Base

class User(Base):
    """User database model"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(200), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    bio = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

# src/models/schemas.py
from pydantic import BaseModel, Field, EmailStr, ConfigDict
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    """Base user schema"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)
    full_name: Optional[str] = Field(None, max_length=200)
    bio: Optional[str] = Field(None, max_length=1000)

class UserCreate(UserBase):
    """Schema for user creation"""
    password: str = Field(..., min_length=8, max_length=100)

class UserUpdate(BaseModel):
    """Schema for user updates"""
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=100)
    full_name: Optional[str] = Field(None, max_length=200)
    bio: Optional[str] = Field(None, max_length=1000)
    is_active: Optional[bool] = None

class UserResponse(UserBase):
    """Schema for user responses"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime

class UserInDB(UserResponse):
    """Schema for user with password hash"""
    hashed_password: str

class Token(BaseModel):
    """Token response schema"""
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    """Token data schema"""
    username: Optional[str] = None
```

### List of tasks to be completed
```yaml
Task 1: Environment Setup and Dependencies
  COMMAND: cd python-env && devbox shell
  VERIFY: uv --version && python --version
  INSTALL: uv add fastapi[all] sqlalchemy[asyncio] alembic asyncpg python-jose[cryptography] passlib[bcrypt] pytest-asyncio httpx

Task 2: Database Configuration
  CREATE: python-env/src/core/config.py
  CREATE: python-env/src/core/database.py
  PATTERN: Async SQLAlchemy with proper session management
  VALIDATE: Database connection and configuration

Task 3: Database Models and Schemas
  CREATE: python-env/src/models/user.py
  CREATE: python-env/src/models/schemas.py
  PATTERN: SQLAlchemy models with Pydantic schemas
  VALIDATE: Model definitions and relationships

Task 4: Authentication System
  CREATE: python-env/src/core/security.py
  CREATE: python-env/src/services/auth_service.py
  PATTERN: JWT-based authentication with password hashing
  SECURITY: Secure password handling and token management

Task 5: User Service Layer
  CREATE: python-env/src/services/user_service.py
  PATTERN: Business logic with dependency injection
  ASYNC: All service methods async with proper error handling
  VALIDATION: Input validation and business rules

Task 6: API Endpoints
  CREATE: python-env/src/api/auth.py (login, register)
  CREATE: python-env/src/api/users.py (CRUD operations)
  CREATE: python-env/src/api/admin.py (admin operations)
  PATTERN: FastAPI routers with proper documentation
  DEPS: Dependency injection for authentication and database

Task 7: Database Migrations
  SETUP: alembic init alembic
  CREATE: Initial migration for user table
  PATTERN: Proper migration workflow with async support
  VALIDATE: Migration up and down operations

Task 8: Comprehensive Testing
  CREATE: Test fixtures and factories
  CREATE: Unit tests for services
  CREATE: Integration tests for API endpoints
  CREATE: End-to-end user workflow tests
  PATTERN: pytest-asyncio with proper test database
  COVERAGE: 90%+ test coverage with meaningful tests

Task 9: Application Integration
  MODIFY: python-env/src/main.py
  PATTERN: FastAPI app with CORS, middleware, exception handlers
  ROUTERS: Include all API routers with proper prefixes
  DOCS: Comprehensive OpenAPI documentation

Task 10: Development Environment
  CREATE: docker-compose.yml for PostgreSQL
  CREATE: .env.example with all required variables
  SCRIPTS: Development scripts for common tasks
  DOCS: Update README.md with setup and usage instructions
```

### Per task pseudocode

**Task 4: Authentication System**
```python
# src/core/security.py
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from .config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt

def verify_token(token: str) -> Optional[str]:
    """Verify JWT token and return username"""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except JWTError:
        return None

# src/services/auth_service.py
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException, status
from src.models.user import User
from src.models.schemas import UserCreate, UserResponse
from src.core.security import verify_password, get_password_hash, create_access_token
from src.services.user_service import UserService

class AuthService:
    """Authentication service"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.user_service = UserService(db)
    
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        result = await self.db.execute(
            select(User).where(User.email == email)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            return None
        
        if not verify_password(password, user.hashed_password):
            return None
        
        return user
    
    async def register_user(self, user_data: UserCreate) -> UserResponse:
        """Register new user"""
        # Check if user already exists
        existing_user = await self.user_service.get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        existing_username = await self.user_service.get_user_by_username(user_data.username)
        if existing_username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        
        # Create user
        hashed_password = get_password_hash(user_data.password)
        user = User(
            email=user_data.email,
            username=user_data.username,
            full_name=user_data.full_name,
            bio=user_data.bio,
            hashed_password=hashed_password
        )
        
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        
        return UserResponse.model_validate(user)
    
    async def login(self, email: str, password: str) -> dict:
        """Login user and return access token"""
        user = await self.authenticate_user(email, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user"
            )
        
        access_token = create_access_token(data={"sub": user.username})
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": UserResponse.model_validate(user)
        }
```

**Task 6: API Endpoints**
```python
# src/api/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated

from src.core.database import get_db
from src.core.security import verify_token
from src.services.auth_service import AuthService
from src.services.user_service import UserService
from src.models.schemas import UserCreate, UserResponse, Token

router = APIRouter(prefix="/auth", tags=["authentication"])
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
    
    username = verify_token(token)
    if username is None:
        raise credentials_exception
    
    user_service = UserService(db)
    user = await user_service.get_user_by_username(username)
    if user is None:
        raise credentials_exception
    
    return user

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

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    Register a new user.
    
    Creates a new user account with the provided information.
    Email and username must be unique.
    """
    auth_service = AuthService(db)
    return await auth_service.register_user(user_data)

@router.post("/login", response_model=dict)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Login with email and password.
    
    Returns access token for authentication.
    Use the token in Authorization header: Bearer <token>
    """
    auth_service = AuthService(db)
    return await auth_service.login(form_data.username, form_data.password)

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: Annotated[UserResponse, Depends(get_current_active_user)]
) -> UserResponse:
    """
    Get current user information.
    
    Returns the profile information of the authenticated user.
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

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/", response_model=List[UserResponse])
async def list_users(
    skip: int = Query(0, ge=0, description="Number of users to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of users to return"),
    current_user: Annotated[UserResponse, Depends(get_current_active_user)] = None,
    db: AsyncSession = Depends(get_db)
) -> List[UserResponse]:
    """
    List users with pagination.
    
    Returns a list of users. Requires authentication.
    """
    user_service = UserService(db)
    return await user_service.list_users(skip=skip, limit=limit)

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_user: Annotated[UserResponse, Depends(get_current_active_user)] = None,
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    Get user by ID.
    
    Returns user information for the specified ID.
    """
    user_service = UserService(db)
    user = await user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user

@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    Update current user profile.
    
    Updates the authenticated user's profile information.
    """
    user_service = UserService(db)
    return await user_service.update_user(current_user.id, user_update)

@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def deactivate_current_user(
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    db: AsyncSession = Depends(get_db)
) -> None:
    """
    Deactivate current user account.
    
    Deactivates the authenticated user's account.
    """
    user_service = UserService(db)
    await user_service.deactivate_user(current_user.id)
```

**Task 8: Comprehensive Testing**
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
from src.core.security import get_password_hash

# Test database URL (in-memory SQLite)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for the session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine"""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={
            "check_same_thread": False,
        },
        poolclass=StaticPool,
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
        hashed_password=get_password_hash("testpassword123"),
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
        hashed_password=get_password_hash("adminpassword123"),
        is_active=True,
        is_superuser=True
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    return user

# tests/test_api/test_auth.py
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.user import User

@pytest.mark.asyncio
async def test_register_user(client: AsyncClient):
    """Test user registration"""
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
    assert "already registered" in response.json()["detail"]

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
    assert "user" in data

@pytest.mark.asyncio
async def test_login_invalid_credentials(client: AsyncClient, test_user: User):
    """Test login with invalid credentials"""
    login_data = {
        "username": test_user.email,
        "password": "wrongpassword"
    }
    
    response = await client.post("/auth/login", data=login_data)
    
    assert response.status_code == 401
    assert "Incorrect email or password" in response.json()["detail"]

@pytest.mark.asyncio
async def test_get_current_user(client: AsyncClient, test_user: User):
    """Test getting current user info"""
    # First login to get token
    login_response = await client.post("/auth/login", data={
        "username": test_user.email,
        "password": "testpassword123"
    })
    token = login_response.json()["access_token"]
    
    # Get current user info
    response = await client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == test_user.email
    assert data["username"] == test_user.username

@pytest.mark.asyncio
async def test_get_current_user_invalid_token(client: AsyncClient):
    """Test getting current user with invalid token"""
    response = await client.get(
        "/auth/me",
        headers={"Authorization": "Bearer invalid_token"}
    )
    
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
```

### Integration Points
```yaml
MAIN_APPLICATION:
  - modify: python-env/src/main.py
  - pattern: Include auth and user routers, setup CORS and middleware
  
DEPENDENCIES:
  - add to: python-env/pyproject.toml
  - pattern: All required packages with version constraints
  
DATABASE:
  - setup: docker-compose.yml with PostgreSQL
  - pattern: Local development database with proper configuration
  
ENVIRONMENT:
  - create: .env.example with all required variables
  - pattern: Secure defaults and clear documentation
  
MIGRATIONS:
  - setup: Alembic with async support
  - pattern: Initial migration and proper versioning
```

## Validation Loop

### Level 1: Python Syntax & Style
```bash
cd python-env && devbox shell

# Format code with ruff
uv run ruff format src/ tests/

# Lint with ruff
uv run ruff check src/ tests/ --fix

# Type checking with mypy
uv run mypy src/ tests/

# Expected: No errors. If errors, READ and fix before proceeding.
```

### Level 2: Unit and Integration Tests
```bash
# Run all tests
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# Run specific test categories
uv run pytest tests/test_api/ -v
uv run pytest tests/test_services/ -v
uv run pytest tests/test_integration/ -v

# Expected: All tests pass, 90%+ coverage
```

### Level 3: Database and API Testing
```bash
# Start PostgreSQL (if using docker-compose)
docker-compose up -d postgres

# Run database migrations
uv run alembic upgrade head

# Start the API server
uv run uvicorn src.main:app --reload --port 8000

# Test API endpoints manually
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "username": "testuser", "password": "testpass123"}'

# Check API documentation
open http://localhost:8000/docs

# Expected: API responds correctly, documentation is comprehensive
```

## Final Validation Checklist
- [ ] Environment setup successful: `devbox shell` works
- [ ] Dependencies installed: `uv add` commands successful
- [ ] Code formatted: `uv run ruff format` clean
- [ ] Linting passed: `uv run ruff check` clean
- [ ] Type checking passed: `uv run mypy` clean
- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] Test coverage 90%+: Coverage report shows adequate coverage
- [ ] Database migrations work: `alembic upgrade head` successful
- [ ] API server starts: `uvicorn` runs without errors
- [ ] API endpoints respond: Manual tests successful
- [ ] OpenAPI docs complete: `/docs` shows comprehensive documentation
- [ ] Authentication works: Login/register flow functional
- [ ] CRUD operations work: User management endpoints functional
- [ ] Error handling comprehensive: Proper error responses and logging
- [ ] Security implemented: Password hashing, JWT tokens, input validation

---

## Production Readiness Checklist
- [ ] Environment variables properly configured
- [ ] Database connection pooling optimized
- [ ] Logging configured for production
- [ ] Error monitoring integrated (optional)
- [ ] Rate limiting implemented (optional)
- [ ] Input validation comprehensive
- [ ] SQL injection protection (SQLAlchemy ORM)
- [ ] CORS properly configured
- [ ] API versioning strategy implemented
- [ ] Performance benchmarks established

This example demonstrates a complete, production-ready FastAPI application with all the essential components needed for a modern web API, following the polyglot environment patterns and best practices.