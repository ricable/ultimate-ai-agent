# File: backend/database/session.py
from typing import Optional, Any, Dict, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select, delete, update
from sqlalchemy.orm import selectinload
from datetime import datetime, timezone, timedelta
import logging
import asyncio
from contextlib import asynccontextmanager

from .connection import AsyncSessionLocal, database

logger = logging.getLogger(__name__)

class SessionManager:
    """Advanced database session manager with utilities"""
    
    def __init__(self):
        self.active_sessions: Dict[str, AsyncSession] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def start_cleanup_task(self):
        """Start background task for session cleanup"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            
    async def stop_cleanup_task(self):
        """Stop background cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
    
    @asynccontextmanager
    async def get_session(self, session_id: Optional[str] = None):
        """Get managed database session"""
        session = AsyncSessionLocal()
        if session_id:
            self.active_sessions[session_id] = session
        
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            if session_id and session_id in self.active_sessions:
                del self.active_sessions[session_id]
            await session.close()
    
    async def execute_raw_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute raw SQL query and return results"""
        try:
            if parameters:
                result = await database.fetch_all(query=query, values=parameters)
            else:
                result = await database.fetch_all(query=query)
            
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Raw query execution failed: {e}")
            raise
    
    async def execute_raw_command(self, command: str, parameters: Optional[Dict] = None) -> bool:
        """Execute raw SQL command (INSERT, UPDATE, DELETE)"""
        try:
            if parameters:
                await database.execute(query=command, values=parameters)
            else:
                await database.execute(query=command)
            return True
        except Exception as e:
            logger.error(f"Raw command execution failed: {e}")
            raise
    
    async def bulk_insert(self, model_class, data: List[Dict]) -> bool:
        """Bulk insert data for better performance"""
        if not data:
            return True
            
        async with self.get_session() as session:
            try:
                # Use bulk_insert_mappings for better performance
                await session.execute(
                    model_class.__table__.insert(),
                    data
                )
                return True
            except Exception as e:
                logger.error(f"Bulk insert failed: {e}")
                raise
    
    async def bulk_update(self, model_class, data: List[Dict], key_field: str = "id") -> bool:
        """Bulk update data for better performance"""
        if not data:
            return True
            
        async with self.get_session() as session:
            try:
                for item in data:
                    await session.execute(
                        update(model_class).where(
                            getattr(model_class, key_field) == item[key_field]
                        ).values(item)
                    )
                return True
            except Exception as e:
                logger.error(f"Bulk update failed: {e}")
                raise
    
    async def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """Get statistics for a database table"""
        query = f"""
        SELECT 
            COUNT(*) as row_count,
            pg_size_pretty(pg_total_relation_size('{table_name}')) as table_size,
            pg_size_pretty(pg_relation_size('{table_name}')) as data_size,
            pg_size_pretty(pg_total_relation_size('{table_name}') - pg_relation_size('{table_name}')) as index_size
        """
        
        result = await self.execute_raw_query(query)
        return result[0] if result else {}
    
    async def vacuum_analyze_table(self, table_name: str) -> bool:
        """Run VACUUM ANALYZE on a specific table"""
        try:
            await self.execute_raw_command(f"VACUUM ANALYZE {table_name}")
            logger.info(f"VACUUM ANALYZE completed for table: {table_name}")
            return True
        except Exception as e:
            logger.error(f"VACUUM ANALYZE failed for table {table_name}: {e}")
            return False
    
    async def get_active_connections(self) -> List[Dict]:
        """Get information about active database connections"""
        query = """
        SELECT 
            pid,
            usename,
            application_name,
            client_addr,
            backend_start,
            state,
            query
        FROM pg_stat_activity
        WHERE state = 'active'
        """
        
        return await self.execute_raw_query(query)
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of stale sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Check for stale sessions
                current_time = datetime.now(timezone.utc)
                stale_sessions = []
                
                for session_id, session in self.active_sessions.items():
                    # Add your stale session detection logic here
                    # For example, check if session has been idle too long
                    pass
                
                # Clean up stale sessions
                for session_id in stale_sessions:
                    if session_id in self.active_sessions:
                        await self.active_sessions[session_id].close()
                        del self.active_sessions[session_id]
                        logger.info(f"Cleaned up stale session: {session_id}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")

# Global session manager instance
session_manager = SessionManager()

def get_session_manager() -> SessionManager:
    """Get the global session manager instance"""
    return session_manager