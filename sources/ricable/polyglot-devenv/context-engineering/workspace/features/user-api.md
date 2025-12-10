# Feature: User Management REST API

## FEATURE:
Build a complete user management REST API with CRUD operations, JWT authentication, and database integration. The API should include user registration, login, profile management, and admin operations with comprehensive validation, security, and testing.

Key requirements:
- User registration and authentication system
- JWT-based authentication with secure password hashing
- PostgreSQL database integration with async SQLAlchemy
- Comprehensive REST API with OpenAPI documentation
- Role-based access control (user/admin roles)
- Input validation and error handling
- Comprehensive test coverage
- Production-ready configuration and deployment setup

## EXAMPLES:
- Reference existing FastAPI patterns in the polyglot environment
- Follow async/await patterns for database operations
- Use existing validation and error handling patterns
- Integrate with the polyglot automation and monitoring systems

## DOCUMENTATION:
- FastAPI: https://fastapi.tiangolo.com/tutorial/
- SQLAlchemy async: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
- Pydantic v2: https://docs.pydantic.dev/
- JWT authentication: https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/
- Alembic migrations: https://alembic.sqlalchemy.org/en/latest/

## OTHER CONSIDERATIONS:
- Environment: python-env
- Database: PostgreSQL with async SQLAlchemy and Alembic migrations
- Authentication: JWT tokens with secure password hashing (bcrypt)
- Testing: pytest-asyncio with comprehensive test coverage (90%+)
- Security: Input validation, SQL injection prevention, secure password storage
- Performance: Database connection pooling, async operations throughout
- Monitoring: Integration with existing performance analytics and security scanning
- Deployment: Docker-ready with environment configuration
- API Documentation: Comprehensive OpenAPI/Swagger documentation with examples