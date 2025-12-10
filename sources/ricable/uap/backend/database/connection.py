# File: backend/database/connection.py
import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from databases import Database
import logging

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+asyncpg://uap_user:uap_password@localhost:5432/uap_db"
)

# For databases library (used for raw queries)
DATABASE_URL_SYNC = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")

# SQLAlchemy async engine
async_engine = create_async_engine(
    DATABASE_URL,
    echo=bool(os.getenv("DB_ECHO", "false").lower() == "true"),
    pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
    pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
    pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600")),
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=True,
    autocommit=False,
)

# Databases instance for raw queries
database = Database(DATABASE_URL_SYNC)

# Base class for SQLAlchemy models
class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models"""
    pass

async def init_database():
    """Initialize database connection"""
    try:
        await database.connect()
        logger.info("Database connection established")
        
        # Test connection
        await database.execute("SELECT 1")
        logger.info("Database connection test successful")
        
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

async def close_database():
    """Close database connection"""
    try:
        await database.disconnect()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Error closing database connection: {e}")
        raise

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for dependency injection"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()

async def create_all_tables():
    """Create all database tables"""
    from ..models import Base
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        logger.info("All database tables created")

async def drop_all_tables():
    """Drop all database tables (use with caution!)"""
    from ..models import Base
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        logger.info("All database tables dropped")

def get_database_url() -> str:
    """Get database URL for external tools"""
    return DATABASE_URL

def get_sync_database_url() -> str:
    """Get synchronous database URL for external tools"""
    return DATABASE_URL_SYNC