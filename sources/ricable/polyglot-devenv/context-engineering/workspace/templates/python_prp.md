name: "Python PRP Template - FastAPI with uv Package Management"
description: |

## Purpose
Template optimized for AI agents to implement Python features in the python-env using FastAPI, uv package management, and strict type checking.

## Core Principles
1. **uv-first**: Use uv exclusively for package management
2. **Type Safety**: Strict mypy and comprehensive type hints
3. **FastAPI Patterns**: Async-first, dependency injection, Pydantic models
4. **Testing Excellence**: pytest with coverage, clear test structure
5. **Code Quality**: ruff formatting/linting, clear error handling

---

## Goal
[What needs to be built - be specific about the Python feature and FastAPI integration]

## Why
- [Business value and user impact]
- [Integration with existing Python services]
- [Problems this solves and for whom]

## What
[User-visible behavior and API endpoints, technical requirements]

### Success Criteria
- [ ] [Specific measurable outcomes for Python implementation]
- [ ] FastAPI endpoints working with proper validation
- [ ] All type hints and mypy checks passing
- [ ] Comprehensive test coverage (80%+)

## All Needed Context

### Target Environment
```yaml
Environment: python-env
Devbox_Config: python-env/devbox.json
Dependencies: [List required Python packages for uv]
Python_Version: 3.12+ (as specified in devbox.json)
Package_Manager: uv (exclusively - no pip/poetry/pipenv)
```

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://fastapi.tiangolo.com/
  why: FastAPI patterns, dependency injection, async handling
  
- file: python-env/src/[existing_similar_module].py
  why: Existing patterns to follow
  
- file: python-env/pyproject.toml
  why: Dependency management and tool configuration
  
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
│   ├── models/
│   │   └── feature_models.py    # Pydantic models for new feature
│   ├── api/
│   │   └── feature_router.py    # FastAPI router with endpoints
│   ├── services/
│   │   └── feature_service.py   # Business logic implementation
│   └── utils/
│       └── feature_utils.py     # Helper functions (if needed)
├── tests/
│   ├── test_models/
│   │   └── test_feature_models.py
│   ├── test_api/
│   │   └── test_feature_router.py
│   └── test_services/
│       └── test_feature_service.py
```

### Known Gotchas of Python Environment
```python
# CRITICAL: Python environment-specific gotchas
# uv: Use 'uv add package' not 'pip install package'
# uv: Run commands with 'uv run command' for proper environment
# FastAPI: All endpoints must be async def for proper performance
# Pydantic: Use v2 syntax - Field() instead of field parameters
# mypy: Strict mode enabled - all functions need return type hints
# pytest: Use async tests with pytest-asyncio for FastAPI testing

# Example patterns:
# ✅ uv add fastapi uvicorn
# ✅ uv run pytest tests/ -v
# ✅ async def endpoint() -> ResponseModel:
# ✅ from pydantic import BaseModel, Field
# ❌ pip install anything
# ❌ def sync_endpoint():  # Use async def
```

## Implementation Blueprint

### Environment Setup
```bash
# Activate Python environment
cd python-env && devbox shell

# Verify environment
uv --version
python --version  # Should be 3.12+

# Install dependencies if needed
uv add [new-package-name]
```

### Data Models (Pydantic v2)
```python
# src/models/feature_models.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime
from enum import Enum

class FeatureStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"

class FeatureBase(BaseModel):
    """Base model for feature data"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    status: FeatureStatus = Field(default=FeatureStatus.PENDING)

class FeatureCreate(FeatureBase):
    """Model for creating new features"""
    pass

class FeatureUpdate(BaseModel):
    """Model for updating existing features"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    status: Optional[FeatureStatus] = None

class FeatureResponse(FeatureBase):
    """Model for feature API responses"""
    id: int = Field(..., gt=0)
    created_at: datetime
    updated_at: datetime
```

### List of tasks to be completed
```yaml
Task 1: Environment Setup
  COMMAND: cd python-env && devbox shell
  VERIFY: uv --version && python --version
  INSTALL: uv add [required-packages]

Task 2: Pydantic Models
  CREATE: python-env/src/models/feature_models.py
  PATTERN: Follow existing model patterns in src/models/
  VALIDATE: uv run mypy src/models/feature_models.py

Task 3: Service Layer
  CREATE: python-env/src/services/feature_service.py
  PATTERN: Follow dependency injection patterns
  ASYNC: All service methods must be async
  ERRORS: Use specific exception types

Task 4: API Router
  CREATE: python-env/src/api/feature_router.py
  PATTERN: Follow FastAPI router patterns
  DEPS: Use dependency injection for services
  DOCS: Add comprehensive OpenAPI documentation

Task 5: Integration
  MODIFY: python-env/src/main.py
  PATTERN: app.include_router(feature_router, prefix="/api/v1")
  VALIDATE: FastAPI server starts without errors

Task 6: Comprehensive Testing
  CREATE: Test files for models, services, and API
  PATTERN: Follow pytest-asyncio patterns for FastAPI
  COVERAGE: Ensure 80%+ test coverage
  MOCKING: Mock external dependencies properly

Task 7: Documentation
  UPDATE: README.md with new endpoints
  DOCSTRINGS: Google-style docstrings for all functions
```

### Per task pseudocode

**Task 3: Service Layer**
```python
# src/services/feature_service.py
from typing import List, Optional
from ..models.feature_models import FeatureCreate, FeatureUpdate, FeatureResponse
from ..utils.exceptions import FeatureNotFoundError, ValidationError

class FeatureService:
    """Service for feature business logic"""
    
    def __init__(self, db_session: AsyncSession) -> None:
        self.db = db_session
    
    async def create_feature(self, feature_data: FeatureCreate) -> FeatureResponse:
        """Create a new feature with validation"""
        # PATTERN: Validate business rules first
        await self._validate_feature_name(feature_data.name)
        
        # PATTERN: Use async database operations
        db_feature = await self._save_feature(feature_data)
        
        # PATTERN: Return response model
        return FeatureResponse.model_validate(db_feature)
    
    async def get_feature(self, feature_id: int) -> FeatureResponse:
        """Get feature by ID with error handling"""
        # PATTERN: Check existence first
        db_feature = await self._get_feature_by_id(feature_id)
        if not db_feature:
            raise FeatureNotFoundError(f"Feature {feature_id} not found")
        
        return FeatureResponse.model_validate(db_feature)
```

**Task 4: API Router**
```python
# src/api/feature_router.py
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from ..models.feature_models import FeatureCreate, FeatureUpdate, FeatureResponse
from ..services.feature_service import FeatureService
from ..utils.dependencies import get_feature_service

router = APIRouter(tags=["features"])

@router.post("/features/", 
             response_model=FeatureResponse,
             status_code=status.HTTP_201_CREATED,
             summary="Create a new feature")
async def create_feature(
    feature_data: FeatureCreate,
    service: FeatureService = Depends(get_feature_service)
) -> FeatureResponse:
    """
    Create a new feature with validation.
    
    Args:
        feature_data: Feature creation data
        service: Injected feature service
        
    Returns:
        Created feature data
        
    Raises:
        HTTPException: If validation fails or creation error
    """
    try:
        return await service.create_feature(feature_data)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
```

### Integration Points
```yaml
MAIN_APP:
  - modify: python-env/src/main.py
  - pattern: "app.include_router(feature_router, prefix='/api/v1')"
  
DEPENDENCIES:
  - add to: python-env/pyproject.toml
  - pattern: Use uv add for new dependencies
  
DATABASE:
  - migration: Add feature table if using database
  - models: SQLAlchemy models if needed
  
CONFIG:
  - add to: python-env/src/config/settings.py
  - pattern: Environment variable configuration
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

### Level 2: Unit Tests with pytest
```python
# tests/test_services/test_feature_service.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.services.feature_service import FeatureService
from src.models.feature_models import FeatureCreate, FeatureStatus
from src.utils.exceptions import FeatureNotFoundError

@pytest.fixture
def mock_db_session():
    return AsyncMock()

@pytest.fixture
def feature_service(mock_db_session):
    return FeatureService(mock_db_session)

@pytest.mark.asyncio
async def test_create_feature_success(feature_service):
    """Test successful feature creation"""
    feature_data = FeatureCreate(
        name="Test Feature",
        description="Test description",
        status=FeatureStatus.ACTIVE
    )
    
    result = await feature_service.create_feature(feature_data)
    
    assert result.name == "Test Feature"
    assert result.status == FeatureStatus.ACTIVE

@pytest.mark.asyncio
async def test_get_feature_not_found(feature_service):
    """Test feature not found error"""
    with pytest.raises(FeatureNotFoundError):
        await feature_service.get_feature(999)
```

```bash
# Run tests with coverage
uv run pytest tests/ -v --cov=src --cov-report=term-missing

# Expected: All tests pass, 80%+ coverage
```

### Level 3: FastAPI Integration Tests
```python
# tests/test_api/test_feature_router.py
import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from src.main import app

@pytest.mark.asyncio
async def test_create_feature_endpoint():
    """Test feature creation endpoint"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/features/",
            json={
                "name": "Test Feature",
                "description": "Test description"
            }
        )
    
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Feature"
    assert "id" in data

@pytest.mark.asyncio
async def test_get_feature_endpoint():
    """Test feature retrieval endpoint"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Create a feature first
        create_response = await client.post(
            "/api/v1/features/",
            json={"name": "Test Feature"}
        )
        feature_id = create_response.json()["id"]
        
        # Get the feature
        response = await client.get(f"/api/v1/features/{feature_id}")
        
    assert response.status_code == 200
    assert response.json()["name"] == "Test Feature"
```

```bash
# Run integration tests
uv run pytest tests/test_api/ -v

# Start server for manual testing
uv run uvicorn src.main:app --reload --port 8000

# Test endpoints manually
curl -X POST http://localhost:8000/api/v1/features/ \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Feature", "description": "Test description"}'
```

## Final Validation Checklist
- [ ] Environment setup successful: `devbox shell` works
- [ ] All dependencies installed: `uv add` commands successful
- [ ] Code formatted: `uv run ruff format` clean
- [ ] Linting passed: `uv run ruff check` clean
- [ ] Type checking passed: `uv run mypy` clean
- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] Test coverage 80%+: `--cov-report` shows adequate coverage
- [ ] FastAPI server starts: `uv run uvicorn src.main:app` works
- [ ] API endpoints respond: Manual curl tests successful
- [ ] Error cases handled gracefully
- [ ] OpenAPI docs updated: Check /docs endpoint
- [ ] Documentation updated if needed

---

## Python-Specific Anti-Patterns to Avoid
- ❌ Don't use pip, poetry, or pipenv - use uv exclusively
- ❌ Don't create sync endpoints - use async def
- ❌ Don't skip type hints - mypy strict mode requires them
- ❌ Don't use Pydantic v1 patterns - use v2 syntax
- ❌ Don't ignore uv run prefix for commands
- ❌ Don't mix async/sync code inappropriately
- ❌ Don't skip input validation with Pydantic
- ❌ Don't hardcode configuration - use environment variables

## Python Best Practices
- ✅ Use uv for all package management operations
- ✅ Activate devbox environment before all operations
- ✅ Use async/await consistently throughout FastAPI code
- ✅ Leverage Pydantic v2 for data validation and serialization
- ✅ Follow dependency injection patterns for services
- ✅ Use specific exception types for different error cases
- ✅ Write comprehensive docstrings in Google style
- ✅ Use type hints everywhere for mypy compatibility
- ✅ Structure tests to mirror source code organization