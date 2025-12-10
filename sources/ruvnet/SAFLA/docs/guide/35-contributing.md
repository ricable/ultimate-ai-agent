# Contributing to SAFLA

## Welcome Contributors! 🎉

Thank you for your interest in contributing to SAFLA (Self-Aware Feedback Loop Algorithm)! This guide will help you get started with contributing to our project, whether you're fixing bugs, adding features, improving documentation, or helping with community support.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Security Contributions](#security-contributions)
- [Community Support](#community-support)
- [Recognition](#recognition)

## Code of Conduct

### Our Pledge

We are committed to making participation in SAFLA a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Positive behaviors include:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behaviors include:**
- Harassment, trolling, or discriminatory comments
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported to the project maintainers at [conduct@safla.dev](mailto:conduct@safla.dev). All complaints will be reviewed and investigated promptly and fairly.

## Getting Started

### Ways to Contribute

1. **Code Contributions**
   - Bug fixes
   - Feature implementations
   - Performance improvements
   - Security enhancements

2. **Documentation**
   - API documentation
   - Tutorials and guides
   - Code examples
   - Translation

3. **Testing**
   - Writing test cases
   - Manual testing
   - Performance testing
   - Security testing

4. **Community Support**
   - Answering questions in discussions
   - Helping with issue triage
   - Code reviews
   - Mentoring new contributors

### Prerequisites

- **Programming Languages**: Python 3.9+, TypeScript/JavaScript (Node.js 18+)
- **Tools**: Git, Docker, VS Code (recommended)
- **Knowledge**: Basic understanding of AI/ML concepts, async programming
- **Optional**: Experience with vector databases, MCP protocol

### First-Time Contributors

If you're new to open source or SAFLA:

1. Start with issues labeled `good-first-issue`
2. Read our [architecture overview](./02-architecture.md)
3. Join our [Discord community](https://discord.gg/safla)
4. Introduce yourself in the `#new-contributors` channel

## Development Setup

### Local Development Environment

#### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/safla.git
cd safla

# Add upstream remote
git remote add upstream https://github.com/safla-org/safla.git
```

#### 2. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Install Node.js dependencies (for frontend/docs)
npm install
```

#### 3. Configuration

```bash
# Copy example configuration
cp config/example.yaml config/development.yaml

# Set environment variables
export SAFLA_ENV=development
export SAFLA_CONFIG_PATH=config/development.yaml
```

#### 4. Database Setup

```bash
# Start development services with Docker
docker-compose -f docker-compose.dev.yml up -d

# Initialize database
safla db init
safla db migrate
```

#### 5. Verify Setup

```bash
# Run tests
pytest

# Start development server
safla dev --reload

# Check health
curl http://localhost:8080/health
```

### Development Tools

#### Recommended VS Code Extensions

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.black-formatter",
    "ms-python.isort",
    "ms-python.pylint",
    "bradlc.vscode-tailwindcss",
    "esbenp.prettier-vscode",
    "ms-vscode.vscode-typescript-next"
  ]
}
```

#### Development Scripts

```bash
# Code formatting
make format

# Linting
make lint

# Type checking
make typecheck

# Run all checks
make check

# Build documentation
make docs

# Run specific test suite
make test-memory
make test-agents
make test-integration
```

## Contributing Guidelines

### Branch Strategy

We use **Git Flow** with the following branches:

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Critical production fixes
- `release/*`: Release preparation

#### Branch Naming Convention

```bash
# Features
feature/memory-optimization
feature/agent-coordination-v2

# Bug fixes
bugfix/memory-leak-vector-store
bugfix/auth-token-validation

# Hotfixes
hotfix/security-vulnerability-cve-2024-001

# Documentation
docs/api-reference-update
docs/migration-guide-v1.1
```

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements
- `security`: Security improvements

#### Examples

```bash
feat(memory): add vector similarity search optimization

fix(agents): resolve coordination deadlock in high-load scenarios

docs(api): update authentication examples

test(memory): add integration tests for episodic memory

security(auth): implement rate limiting for login attempts
```

### Development Workflow

#### 1. Create Feature Branch

```bash
# Update develop branch
git checkout develop
git pull upstream develop

# Create feature branch
git checkout -b feature/your-feature-name
```

#### 2. Development Process

```bash
# Make changes
# ... code, test, commit ...

# Run checks before committing
make check

# Commit with conventional format
git commit -m "feat(memory): add vector similarity optimization"
```

#### 3. Keep Branch Updated

```bash
# Regularly sync with upstream
git fetch upstream
git rebase upstream/develop
```

#### 4. Prepare for PR

```bash
# Final checks
make check
make test

# Push to your fork
git push origin feature/your-feature-name
```

## Code Standards

### Python Code Style

We follow **PEP 8** with some modifications:

```python
# Line length: 88 characters (Black default)
# Use Black for formatting
# Use isort for import sorting
# Use pylint for linting

# Example function
async def store_vector_memory(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    embedding_model: str = "text-embedding-ada-002"
) -> VectorMemoryResult:
    """Store content in vector memory with optional metadata.
    
    Args:
        content: Text content to store
        metadata: Optional metadata dictionary
        embedding_model: Model to use for embeddings
        
    Returns:
        VectorMemoryResult containing storage details
        
    Raises:
        MemoryStorageError: If storage fails
        ValidationError: If content is invalid
    """
    if not content.strip():
        raise ValidationError("Content cannot be empty")
    
    try:
        embedding = await self._generate_embedding(content, embedding_model)
        result = await self._store_vector(embedding, content, metadata)
        return result
    except Exception as e:
        logger.error(f"Failed to store vector memory: {e}")
        raise MemoryStorageError(f"Storage failed: {e}") from e
```

### TypeScript/JavaScript Code Style

We use **Prettier** and **ESLint**:

```typescript
// Example TypeScript code
interface VectorMemoryOptions {
  embeddingModel?: string;
  metadata?: Record<string, unknown>;
  similarity?: number;
}

class VectorMemory {
  private readonly client: VectorClient;
  
  constructor(private readonly config: VectorMemoryConfig) {
    this.client = new VectorClient(config);
  }
  
  async store(
    content: string,
    options: VectorMemoryOptions = {}
  ): Promise<VectorMemoryResult> {
    const { embeddingModel = 'text-embedding-ada-002', metadata } = options;
    
    if (!content.trim()) {
      throw new ValidationError('Content cannot be empty');
    }
    
    try {
      const embedding = await this.generateEmbedding(content, embeddingModel);
      return await this.storeVector(embedding, content, metadata);
    } catch (error) {
      logger.error('Failed to store vector memory:', error);
      throw new MemoryStorageError(`Storage failed: ${error.message}`);
    }
  }
}
```

### Documentation Standards

#### Code Documentation

```python
class MemorySystem:
    """Central memory management system for SAFLA.
    
    The MemorySystem coordinates between different memory types (vector, episodic,
    semantic, working) and provides a unified interface for memory operations.
    
    Attributes:
        vector_memory: Vector-based similarity search memory
        episodic_memory: Sequential event memory
        semantic_memory: Structured knowledge memory
        working_memory: Temporary processing memory
        
    Example:
        >>> memory = MemorySystem()
        >>> await memory.initialize()
        >>> result = await memory.store("Hello world", memory_type="vector")
        >>> print(result.id)
        'vec_123456789'
    """
    
    def __init__(self, config: MemoryConfig) -> None:
        """Initialize memory system with configuration.
        
        Args:
            config: Memory system configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
```

#### API Documentation

```python
@router.post("/memory/vector/store")
async def store_vector_memory(
    request: VectorStoreRequest,
    current_user: User = Depends(get_current_user)
) -> VectorStoreResponse:
    """Store content in vector memory.
    
    This endpoint stores text content in the vector memory system, generating
    embeddings and enabling similarity-based retrieval.
    
    **Required Permissions**: `memory:write`
    
    **Rate Limits**: 100 requests per minute per user
    
    Args:
        request: Vector storage request containing content and metadata
        current_user: Authenticated user making the request
        
    Returns:
        VectorStoreResponse with storage details and generated ID
        
    Raises:
        HTTPException: 400 if content is invalid
        HTTPException: 403 if user lacks permissions
        HTTPException: 429 if rate limit exceeded
        HTTPException: 500 if storage fails
        
    Example:
        ```bash
        curl -X POST "http://localhost:8080/api/v1/memory/vector/store" \
             -H "Authorization: Bearer $TOKEN" \
             -H "Content-Type: application/json" \
             -d '{
               "content": "Important information to remember",
               "metadata": {"source": "user_input", "priority": "high"}
             }'
        ```
        
        Response:
        ```json
        {
          "id": "vec_123456789",
          "status": "stored",
          "embedding_model": "text-embedding-ada-002",
          "dimensions": 1536,
          "created_at": "2025-01-01T12:00:00Z"
        }
        ```
    """
```

## Testing Requirements

### Test Coverage

We maintain **minimum 80% test coverage** with the following requirements:

- **Unit tests**: All public methods and functions
- **Integration tests**: API endpoints and service interactions
- **End-to-end tests**: Critical user workflows
- **Performance tests**: Memory operations and agent coordination
- **Security tests**: Authentication and authorization

### Test Structure

```python
# tests/unit/memory/test_vector_memory.py
import pytest
from unittest.mock import AsyncMock, patch
from safla.memory import VectorMemory
from safla.exceptions import MemoryStorageError

class TestVectorMemory:
    """Test suite for VectorMemory class."""
    
    @pytest.fixture
    async def vector_memory(self):
        """Create VectorMemory instance for testing."""
        config = VectorMemoryConfig(
            provider="faiss",
            dimensions=768,
            similarity_metric="cosine"
        )
        memory = VectorMemory(config)
        await memory.initialize()
        yield memory
        await memory.cleanup()
    
    @pytest.mark.asyncio
    async def test_store_valid_content(self, vector_memory):
        """Test storing valid content in vector memory."""
        # Arrange
        content = "Test content for storage"
        metadata = {"source": "test"}
        
        # Act
        result = await vector_memory.store(content, metadata=metadata)
        
        # Assert
        assert result.id is not None
        assert result.status == "stored"
        assert result.content == content
        assert result.metadata == metadata
    
    @pytest.mark.asyncio
    async def test_store_empty_content_raises_error(self, vector_memory):
        """Test that storing empty content raises ValidationError."""
        # Arrange
        content = ""
        
        # Act & Assert
        with pytest.raises(ValidationError, match="Content cannot be empty"):
            await vector_memory.store(content)
    
    @pytest.mark.asyncio
    async def test_search_returns_similar_content(self, vector_memory):
        """Test that search returns semantically similar content."""
        # Arrange
        await vector_memory.store("Python programming language")
        await vector_memory.store("JavaScript web development")
        
        # Act
        results = await vector_memory.search("Python coding", limit=5)
        
        # Assert
        assert len(results) > 0
        assert any("Python" in result.content for result in results)
        assert results[0].similarity_score > 0.7
```

### Integration Tests

```python
# tests/integration/test_memory_api.py
import pytest
from httpx import AsyncClient
from safla.main import app

@pytest.mark.asyncio
async def test_store_and_retrieve_vector_memory():
    """Test complete store and retrieve workflow via API."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Store content
        store_response = await client.post(
            "/api/v1/memory/vector/store",
            json={
                "content": "Integration test content",
                "metadata": {"test": True}
            },
            headers={"Authorization": "Bearer test-token"}
        )
        assert store_response.status_code == 200
        store_data = store_response.json()
        
        # Search for content
        search_response = await client.post(
            "/api/v1/memory/vector/search",
            json={
                "query": "Integration test",
                "limit": 5
            },
            headers={"Authorization": "Bearer test-token"}
        )
        assert search_response.status_code == 200
        search_data = search_response.json()
        
        # Verify results
        assert len(search_data["results"]) > 0
        assert search_data["results"][0]["id"] == store_data["id"]
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=safla --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run tests with specific markers
pytest -m "not slow"
pytest -m "security"
pytest -m "performance"

# Run tests in parallel
pytest -n auto

# Run tests with verbose output
pytest -v --tb=short
```

## Documentation

### Documentation Types

1. **API Documentation**: Auto-generated from code docstrings
2. **User Guides**: Step-by-step tutorials and how-tos
3. **Developer Documentation**: Architecture and contribution guides
4. **Examples**: Code samples and use cases

### Writing Documentation

#### User Guides

```markdown
# How to Set Up Vector Memory

Vector memory enables semantic search and similarity-based retrieval in SAFLA.

## Prerequisites

- SAFLA 1.0+ installed
- Vector database (Qdrant, Pinecone, or FAISS)
- OpenAI API key (for embeddings)

## Step 1: Configure Vector Memory

Create a configuration file:

```yaml
memory:
  vector:
    provider: "qdrant"
    url: "http://localhost:6333"
    collection_name: "safla_vectors"
    embedding_model: "text-embedding-ada-002"
```

## Step 2: Initialize Memory System

```python
from safla import SAFLA

safla = SAFLA(config_path="config.yaml")
await safla.initialize()
```

## Step 3: Store and Search Content

```python
# Store content
result = await safla.memory.vector.store(
    "SAFLA is a self-aware AI system",
    metadata={"category": "definition"}
)

# Search for similar content
results = await safla.memory.vector.search(
    "What is SAFLA?",
    limit=5
)
```

## Troubleshooting

**Issue**: Connection refused to vector database
**Solution**: Ensure the vector database is running and accessible
```

#### Code Examples

```python
# examples/memory/vector_search.py
"""
Example: Advanced Vector Memory Search

This example demonstrates advanced vector memory features including
metadata filtering, similarity thresholds, and result ranking.
"""

import asyncio
from safla import SAFLA
from safla.memory import VectorMemoryConfig

async def main():
    """Demonstrate advanced vector memory search capabilities."""
    
    # Initialize SAFLA with vector memory
    config = VectorMemoryConfig(
        provider="qdrant",
        url="http://localhost:6333",
        embedding_model="text-embedding-ada-002"
    )
    
    safla = SAFLA()
    await safla.initialize()
    
    # Store documents with metadata
    documents = [
        {
            "content": "SAFLA uses vector memory for semantic search",
            "metadata": {"type": "documentation", "section": "memory"}
        },
        {
            "content": "Agent coordination enables distributed processing",
            "metadata": {"type": "documentation", "section": "agents"}
        },
        {
            "content": "Safety mechanisms prevent harmful outputs",
            "metadata": {"type": "documentation", "section": "safety"}
        }
    ]
    
    for doc in documents:
        await safla.memory.vector.store(
            doc["content"],
            metadata=doc["metadata"]
        )
    
    # Advanced search with filters
    results = await safla.memory.vector.search(
        query="How does SAFLA handle memory?",
        limit=10,
        similarity_threshold=0.7,
        metadata_filter={"type": "documentation"},
        include_metadata=True
    )
    
    # Display results
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.content}")
        print(f"   Similarity: {result.similarity_score:.3f}")
        print(f"   Section: {result.metadata.get('section', 'unknown')}")
        print()

if __name__ == "__main__":
    asyncio.run(main())
```

## Pull Request Process

### Before Submitting

1. **Ensure your branch is up to date**:
   ```bash
   git fetch upstream
   git rebase upstream/develop
   ```

2. **Run all checks**:
   ```bash
   make check
   make test
   ```

3. **Update documentation** if needed

4. **Add tests** for new functionality

### PR Template

When creating a pull request, use this template:

```markdown
## Description

Brief description of changes and motivation.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Security enhancement

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or properly documented)
- [ ] Commit messages follow conventional format

## Related Issues

Closes #123
Related to #456

## Screenshots (if applicable)

## Additional Notes

Any additional information for reviewers.
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and checks
2. **Code Review**: At least one maintainer reviews the code
3. **Testing**: Manual testing if needed
4. **Approval**: Maintainer approval required for merge
5. **Merge**: Squash and merge to develop branch

### Review Criteria

Reviewers will check for:

- **Functionality**: Does the code work as intended?
- **Code Quality**: Is the code clean, readable, and maintainable?
- **Testing**: Are there adequate tests?
- **Documentation**: Is documentation updated?
- **Performance**: Any performance implications?
- **Security**: Any security considerations?
- **Breaking Changes**: Are breaking changes properly handled?

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
**Bug Description**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
 - OS: [e.g. Ubuntu 20.04]
 - Python Version: [e.g. 3.9.7]
 - SAFLA Version: [e.g. 1.0.0]
 - Vector Database: [e.g. Qdrant 1.7.0]

**Additional Context**
Add any other context about the problem here.

**Logs**
```
Paste relevant log output here
```
```

### Feature Requests

Use the feature request template:

```markdown
**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Use Case**
Describe the specific use case for this feature.

**Additional context**
Add any other context or screenshots about the feature request here.
```

### Issue Labels

We use the following labels:

- **Type**: `bug`, `feature`, `documentation`, `question`
- **Priority**: `critical`, `high`, `medium`, `low`
- **Difficulty**: `good-first-issue`, `easy`, `medium`, `hard`
- **Component**: `memory`, `agents`, `api`, `security`, `docs`
- **Status**: `needs-triage`, `in-progress`, `blocked`, `ready-for-review`

## Security Contributions

### Security Policy

- **Report security vulnerabilities** to [security@safla.dev](mailto:security@safla.dev)
- **Do not** create public issues for security vulnerabilities
- **Follow responsible disclosure** practices
- **Expect acknowledgment** within 48 hours

### Security Review Process

1. **Initial Assessment**: Security team reviews the report
2. **Verification**: Reproduce and verify the vulnerability
3. **Impact Analysis**: Assess severity and impact
4. **Fix Development**: Develop and test fix
5. **Disclosure**: Coordinate disclosure timeline
6. **Recognition**: Credit reporter (if desired)

### Security Testing

```python
# Example security test
import pytest
from safla.security import AuthenticationService
from safla.exceptions import AuthenticationError

class TestAuthenticationSecurity:
    """Security tests for authentication system."""
    
    @pytest.mark.security
    async def test_brute_force_protection(self):
        """Test that brute force attacks are prevented."""
        auth_service = AuthenticationService()
        
        # Attempt multiple failed logins
        for _ in range(10):
            with pytest.raises(AuthenticationError):
                await auth_service.authenticate("user", "wrong_password")
        
        # Verify account is locked
        with pytest.raises(AuthenticationError, match="Account locked"):
            await auth_service.authenticate("user", "correct_password")
    
    @pytest.mark.security
    async def test_sql_injection_prevention(self):
        """Test that SQL injection is prevented."""
        # Test with malicious input
        malicious_input = "'; DROP TABLE users; --"
        
        # Should not raise database errors
        result = await auth_service.find_user(malicious_input)
        assert result is None
```

## Community Support

### Communication Channels

- **GitHub Discussions**: General questions and discussions
- **Discord**: Real-time chat and community support
- **Stack Overflow**: Technical questions (tag: `safla`)
- **Twitter**: Updates and announcements (@safla_dev)

### Community Guidelines

1. **Be respectful** and inclusive
2. **Search before asking** - check existing issues/discussions
3. **Provide context** when asking questions
4. **Help others** when you can
5. **Follow up** on your questions and issues

### Mentorship Program

We offer mentorship for new contributors:

- **Mentor Assignment**: Experienced contributors guide newcomers
- **Regular Check-ins**: Weekly progress discussions
- **Pair Programming**: Collaborative coding sessions
- **Code Reviews**: Detailed feedback and learning

To join the mentorship program, comment on a `good-first-issue` or reach out in Discord.

## Recognition

### Contributor Recognition

We recognize contributors through:

1. **Contributors File**: Listed in CONTRIBUTORS.md
2. **Release Notes**: Mentioned in changelog
3. **Social Media**: Highlighted on Twitter
4. **Swag**: SAFLA stickers and merchandise
5. **Conference Talks**: Speaking opportunities

### Contribution Levels

- **First-time Contributor**: First merged PR
- **Regular Contributor**: 5+ merged PRs
- **Core Contributor**: 20+ merged PRs + ongoing involvement
- **Maintainer**: Trusted with repository access

### Hall of Fame

Outstanding contributors are featured in our Hall of Fame:

- **Innovation Award**: Most creative feature
- **Quality Award**: Best code quality and testing
- **Community Award**: Most helpful in community support
- **Documentation Award**: Best documentation contributions

## Release Process

### Release Schedule

- **Major releases**: Every 6 months
- **Minor releases**: Monthly
- **Patch releases**: As needed for bugs/security

### Release Roles

- **Release Manager**: Coordinates release process
- **QA Lead**: Oversees testing and validation
- **Documentation Lead**: Ensures docs are updated
- **Community Manager**: Handles announcements

### Contributing to Releases

1. **Feature Freeze**: 2 weeks before major releases
2. **Release Candidates**: Test pre-release versions
3. **Documentation Review**: Update guides and examples
4. **Migration Testing**: Verify upgrade paths

## Getting Help

### For Contributors

- **Documentation**: Check our comprehensive guides
- **Discord**: Ask in #contributors channel
- **Office Hours**: Weekly maintainer office hours
- **1:1 Sessions**: Schedule with maintainers

### For Maintainers

- **Maintainer Guide**: Internal documentation
- **Security Protocols**: Incident response procedures
- **Release Procedures**: Step-by-step release guide
- **Community Management**: Guidelines for community interaction

---

## Thank You! 🙏

Your contributions make SAFLA better for everyone. Whether you're fixing a typo, adding a feature, or helping other users, every contribution matters.

**Questions?** Reach out to us:
- Email: [contributors@safla.dev](mailto:contributors@safla.dev)
- Discord: [SAFLA Community](https://discord.gg/safla)
- GitHub: [@safla-org](https://github.com/safla-org)

**Happy Contributing!** 🚀

---

**Last Updated**: January 2025  
**Version**: 1.0.0  
**Maintained by**: SAFLA Core Team