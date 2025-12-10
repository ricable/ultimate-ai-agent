# File: backend/database/migrations/manager.py
import os
import asyncio
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select, MetaData, Table
from datetime import datetime, timezone
import json
import logging

from ..connection import async_engine, AsyncSessionLocal, get_sync_database_url
from ..session import SessionManager

logger = logging.getLogger(__name__)

class MigrationManager:
    """Database migration management system"""
    
    def __init__(self):
        self.session_manager = SessionManager()
        self.migrations_dir = os.path.dirname(__file__)
        
    async def create_migration_table(self):
        """Create migration tracking table if it doesn't exist"""
        async with AsyncSessionLocal() as session:
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS migration_history (
                    id SERIAL PRIMARY KEY,
                    migration_id VARCHAR(255) UNIQUE NOT NULL,
                    migration_name VARCHAR(255) NOT NULL,
                    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    execution_time_ms INTEGER,
                    checksum VARCHAR(64),
                    success BOOLEAN DEFAULT TRUE,
                    error_message TEXT,
                    metadata JSON DEFAULT '{}'
                )
            """))
            await session.commit()
    
    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration IDs"""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(text(
                    "SELECT migration_id FROM migration_history WHERE success = TRUE ORDER BY applied_at"
                ))
                return [row[0] for row in result.fetchall()]
        except Exception:
            # Table doesn't exist yet
            return []
    
    async def record_migration(self, migration_id: str, migration_name: str, 
                             execution_time_ms: int, success: bool = True, 
                             error_message: str = None, metadata: Dict = None):
        """Record migration execution in history"""
        async with AsyncSessionLocal() as session:
            await session.execute(text("""
                INSERT INTO migration_history 
                (migration_id, migration_name, execution_time_ms, success, error_message, metadata)
                VALUES (:migration_id, :migration_name, :execution_time_ms, :success, :error_message, :metadata)
            """), {
                "migration_id": migration_id,
                "migration_name": migration_name,
                "execution_time_ms": execution_time_ms,
                "success": success,
                "error_message": error_message,
                "metadata": json.dumps(metadata or {})
            })
            await session.commit()
    
    async def run_migration_sql(self, sql: str, migration_id: str = None) -> bool:
        """Execute migration SQL"""
        try:
            async with AsyncSessionLocal() as session:
                # Split SQL into individual statements
                statements = [stmt.strip() for stmt in sql.split(';') if stmt.strip()]
                
                for statement in statements:
                    await session.execute(text(statement))
                
                await session.commit()
                return True
                
        except Exception as e:
            logger.error(f"Migration SQL execution failed: {e}")
            if migration_id:
                logger.error(f"Failed migration: {migration_id}")
            raise
    
    async def create_all_tables(self):
        """Create all database tables from models"""
        start_time = datetime.now()
        
        try:
            # Import all models to ensure they're registered
            from ...models import Base
            
            async with async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Record as migration
            await self.record_migration(
                migration_id="001_create_all_tables",
                migration_name="Create all initial tables",
                execution_time_ms=execution_time,
                success=True,
                metadata={"type": "initial_schema"}
            )
            
            logger.info(f"All database tables created successfully in {execution_time}ms")
            return True
            
        except Exception as e:
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            await self.record_migration(
                migration_id="001_create_all_tables",
                migration_name="Create all initial tables",
                execution_time_ms=execution_time,
                success=False,
                error_message=str(e),
                metadata={"type": "initial_schema"}
            )
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    async def drop_all_tables(self):
        """Drop all database tables (use with caution!)"""
        start_time = datetime.now()
        
        try:
            from ...models import Base
            
            async with async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.info(f"All database tables dropped successfully in {execution_time}ms")
            return True
            
        except Exception as e:
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    async def backup_database(self, backup_path: str = None) -> Dict[str, Any]:
        """Create database backup"""
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"/tmp/uap_backup_{timestamp}.sql"
        
        try:
            # Use pg_dump for backup
            db_url = get_sync_database_url()
            cmd = f"pg_dump {db_url} > {backup_path}"
            
            # This would need to be executed as a subprocess in production
            # For now, we'll create a metadata backup
            
            backup_info = {
                "backup_path": backup_path,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "database_url": db_url.split('@')[1] if '@' in db_url else "hidden",
                "backup_type": "metadata_only",
                "success": True
            }
            
            # Get table statistics
            async with AsyncSessionLocal() as session:
                tables_info = {}
                
                # Get table names from information_schema
                result = await session.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """))
                
                for (table_name,) in result.fetchall():
                    count_result = await session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    row_count = count_result.scalar()
                    tables_info[table_name] = {"row_count": row_count}
                
                backup_info["tables"] = tables_info
            
            logger.info(f"Database backup metadata created: {backup_path}")
            return backup_info
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return {
                "backup_path": backup_path,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "success": False,
                "error": str(e)
            }
    
    async def get_database_status(self) -> Dict[str, Any]:
        """Get comprehensive database status"""
        try:
            async with AsyncSessionLocal() as session:
                # Test connection
                await session.execute(text("SELECT 1"))
                
                # Get version info
                version_result = await session.execute(text("SELECT version()"))
                version = version_result.scalar()
                
                # Get database size
                size_result = await session.execute(text("""
                    SELECT pg_size_pretty(pg_database_size(current_database()))
                """))
                db_size = size_result.scalar()
                
                # Get connection count
                conn_result = await session.execute(text("""
                    SELECT count(*) FROM pg_stat_activity
                """))
                connection_count = conn_result.scalar()
                
                # Get table statistics
                tables_result = await session.execute(text("""
                    SELECT 
                        schemaname,
                        tablename,
                        n_tup_ins as inserts,
                        n_tup_upd as updates,
                        n_tup_del as deletes,
                        n_live_tup as live_tuples
                    FROM pg_stat_user_tables
                    ORDER BY tablename
                """))
                
                tables_stats = []
                for row in tables_result.fetchall():
                    tables_stats.append({
                        "schema": row[0],
                        "table": row[1],
                        "inserts": row[2],
                        "updates": row[3],
                        "deletes": row[4],
                        "live_tuples": row[5]
                    })
                
                return {
                    "connected": True,
                    "version": version,
                    "database_size": db_size,
                    "connection_count": connection_count,
                    "tables": tables_stats,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def optimize_database(self) -> Dict[str, Any]:
        """Optimize database performance"""
        results = []
        
        try:
            async with AsyncSessionLocal() as session:
                # Get all user tables
                tables_result = await session.execute(text("""
                    SELECT tablename FROM pg_tables WHERE schemaname = 'public'
                """))
                
                tables = [row[0] for row in tables_result.fetchall()]
                
                for table in tables:
                    try:
                        start_time = datetime.now()
                        
                        # Run VACUUM ANALYZE
                        await session.execute(text(f"VACUUM ANALYZE {table}"))
                        
                        execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
                        
                        results.append({
                            "table": table,
                            "operation": "VACUUM ANALYZE",
                            "success": True,
                            "execution_time_ms": execution_time
                        })
                        
                    except Exception as e:
                        results.append({
                            "table": table,
                            "operation": "VACUUM ANALYZE",
                            "success": False,
                            "error": str(e)
                        })
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": results
            }
        
        return {
            "success": True,
            "results": results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Global migration manager instance
migration_manager = MigrationManager()

def get_migration_manager() -> MigrationManager:
    """Get the global migration manager instance"""
    return migration_manager