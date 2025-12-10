# File: backend/database/retention.py
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from sqlalchemy import text, delete, select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from .connection import AsyncSessionLocal
from .session import SessionManager
from ..models.conversation import Conversation, Message, ConversationStatus
from ..models.document import Document, DocumentStatus
from ..models.analytics import UserSession, AgentUsage, SystemMetrics, AuditLog
from ..models.user import RefreshToken

logger = logging.getLogger(__name__)

class DataRetentionManager:
    """Manage data retention policies and cleanup procedures"""
    
    def __init__(self):
        self.session_manager = SessionManager()
        self.retention_policies = self._get_default_policies()
        
    def _get_default_policies(self) -> Dict[str, Any]:
        """Get default retention policies"""
        return {
            "conversations": {
                "delete_after_days": 365,  # 1 year
                "archive_after_days": 90,  # 3 months
                "cleanup_deleted_after_days": 30  # 30 days after deletion
            },
            "documents": {
                "delete_after_days": 730,  # 2 years
                "archive_after_days": 180,  # 6 months
                "cleanup_deleted_after_days": 30
            },
            "user_sessions": {
                "delete_after_days": 90,  # 3 months
                "cleanup_inactive_after_hours": 24  # 24 hours inactive
            },
            "agent_usage": {
                "delete_after_days": 365,  # 1 year
                "aggregate_after_days": 30  # Aggregate to daily summaries after 30 days
            },
            "system_metrics": {
                "delete_after_days": 90,  # 3 months
                "aggregate_after_days": 7  # Aggregate to hourly after 7 days
            },
            "audit_logs": {
                "delete_after_days": 2555,  # 7 years (compliance requirement)
                "archive_after_days": 365  # 1 year
            },
            "refresh_tokens": {
                "delete_expired_after_hours": 1  # Clean up expired tokens hourly
            }
        }
    
    async def apply_retention_policies(self) -> Dict[str, Any]:
        """Apply all retention policies"""
        results = {}
        
        try:
            # Clean up conversations
            results["conversations"] = await self._cleanup_conversations()
            
            # Clean up documents
            results["documents"] = await self._cleanup_documents()
            
            # Clean up user sessions
            results["user_sessions"] = await self._cleanup_user_sessions()
            
            # Clean up agent usage data
            results["agent_usage"] = await self._cleanup_agent_usage()
            
            # Clean up system metrics
            results["system_metrics"] = await self._cleanup_system_metrics()
            
            # Clean up audit logs
            results["audit_logs"] = await self._cleanup_audit_logs()
            
            # Clean up refresh tokens
            results["refresh_tokens"] = await self._cleanup_refresh_tokens()
            
            logger.info("Data retention policies applied successfully")
            
        except Exception as e:
            logger.error(f"Error applying retention policies: {e}")
            results["error"] = str(e)
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": results
        }
    
    async def _cleanup_conversations(self) -> Dict[str, int]:
        """Clean up old conversations"""
        async with AsyncSessionLocal() as session:
            policies = self.retention_policies["conversations"]
            results = {"archived": 0, "deleted": 0, "cleaned": 0}
            
            # Archive old conversations
            archive_date = datetime.now(timezone.utc) - timedelta(days=policies["archive_after_days"])
            archive_result = await session.execute(
                select(Conversation).where(
                    and_(
                        Conversation.last_message_at < archive_date,
                        Conversation.status == ConversationStatus.ACTIVE
                    )
                )
            )
            
            for conv in archive_result.scalars():
                conv.archive()
                results["archived"] += 1
            
            # Delete very old conversations
            delete_date = datetime.now(timezone.utc) - timedelta(days=policies["delete_after_days"])
            delete_conversations = await session.execute(
                select(Conversation).where(
                    Conversation.last_message_at < delete_date
                )
            )
            
            for conv in delete_conversations.scalars():
                await session.delete(conv)
                results["deleted"] += 1
            
            await session.commit()
            
        return results
    
    async def _cleanup_documents(self) -> Dict[str, int]:
        """Clean up old documents"""
        async with AsyncSessionLocal() as session:
            policies = self.retention_policies["documents"]
            results = {"archived": 0, "deleted": 0}
            
            # Archive old documents
            archive_date = datetime.now(timezone.utc) - timedelta(days=policies["archive_after_days"])
            archive_docs = await session.execute(
                select(Document).where(
                    and_(
                        Document.last_accessed < archive_date,
                        Document.status != DocumentStatus.ARCHIVED
                    )
                )
            )
            
            for doc in archive_docs.scalars():
                doc.archive()
                results["archived"] += 1
            
            # Delete very old documents
            delete_date = datetime.now(timezone.utc) - timedelta(days=policies["delete_after_days"])
            delete_result = await session.execute(
                delete(Document).where(
                    Document.created_at < delete_date
                )
            )
            results["deleted"] = delete_result.rowcount
            
            await session.commit()
            
        return results
    
    async def _cleanup_user_sessions(self) -> Dict[str, int]:
        """Clean up old user sessions"""
        async with AsyncSessionLocal() as session:
            policies = self.retention_policies["user_sessions"]
            results = {"expired": 0, "deleted": 0}
            
            # Mark old sessions as expired
            expire_date = datetime.now(timezone.utc) - timedelta(hours=policies["cleanup_inactive_after_hours"])
            expire_result = await session.execute(
                select(UserSession).where(
                    and_(
                        UserSession.last_activity < expire_date,
                        UserSession.is_active == True
                    )
                )
            )
            
            for session_obj in expire_result.scalars():
                session_obj.end_session()
                results["expired"] += 1
            
            # Delete very old sessions
            delete_date = datetime.now(timezone.utc) - timedelta(days=policies["delete_after_days"])
            delete_result = await session.execute(
                delete(UserSession).where(
                    UserSession.created_at < delete_date
                )
            )
            results["deleted"] = delete_result.rowcount
            
            await session.commit()
            
        return results
    
    async def _cleanup_agent_usage(self) -> Dict[str, int]:
        """Clean up old agent usage data"""
        async with AsyncSessionLocal() as session:
            policies = self.retention_policies["agent_usage"]
            results = {"deleted": 0}
            
            # Delete old usage data
            delete_date = datetime.now(timezone.utc) - timedelta(days=policies["delete_after_days"])
            delete_result = await session.execute(
                delete(AgentUsage).where(
                    AgentUsage.started_at < delete_date
                )
            )
            results["deleted"] = delete_result.rowcount
            
            await session.commit()
            
        return results
    
    async def _cleanup_system_metrics(self) -> Dict[str, int]:
        """Clean up old system metrics"""
        async with AsyncSessionLocal() as session:
            policies = self.retention_policies["system_metrics"]
            results = {"deleted": 0}
            
            # Delete old metrics
            delete_date = datetime.now(timezone.utc) - timedelta(days=policies["delete_after_days"])
            delete_result = await session.execute(
                delete(SystemMetrics).where(
                    SystemMetrics.timestamp < delete_date
                )
            )
            results["deleted"] = delete_result.rowcount
            
            await session.commit()
            
        return results
    
    async def _cleanup_audit_logs(self) -> Dict[str, int]:
        """Clean up old audit logs (with compliance considerations)"""
        async with AsyncSessionLocal() as session:
            policies = self.retention_policies["audit_logs"]
            results = {"deleted": 0}
            
            # Only delete extremely old logs (7 years for compliance)
            delete_date = datetime.now(timezone.utc) - timedelta(days=policies["delete_after_days"])
            delete_result = await session.execute(
                delete(AuditLog).where(
                    AuditLog.timestamp < delete_date
                )
            )
            results["deleted"] = delete_result.rowcount
            
            await session.commit()
            
        return results
    
    async def _cleanup_refresh_tokens(self) -> Dict[str, int]:
        """Clean up expired refresh tokens"""
        async with AsyncSessionLocal() as session:
            results = {"deleted": 0}
            
            # Delete expired or revoked tokens
            delete_result = await session.execute(
                delete(RefreshToken).where(
                    or_(
                        RefreshToken.expires_at < datetime.now(timezone.utc),
                        RefreshToken.is_active == False,
                        RefreshToken.revoked_at.isnot(None)
                    )
                )
            )
            results["deleted"] = delete_result.rowcount
            
            await session.commit()
            
        return results
    
    async def get_data_usage_report(self) -> Dict[str, Any]:
        """Generate data usage report"""
        async with AsyncSessionLocal() as session:
            report = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tables": {}
            }
            
            # Define tables to analyze
            tables = [
                ("users", "User accounts"),
                ("user_roles", "User roles"),
                ("refresh_tokens", "Refresh tokens"),
                ("conversations", "Conversations"),
                ("messages", "Messages"),
                ("message_metadata", "Message metadata"),
                ("documents", "Documents"),
                ("document_metadata", "Document metadata"),
                ("document_processing_jobs", "Document processing jobs"),
                ("user_sessions", "User sessions"),
                ("agent_usage", "Agent usage logs"),
                ("system_metrics", "System metrics"),
                ("audit_logs", "Audit logs")
            ]
            
            for table_name, description in tables:
                try:
                    # Get row count
                    count_result = await session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    row_count = count_result.scalar()
                    
                    # Get table size
                    size_result = await session.execute(text(f"""
                        SELECT pg_size_pretty(pg_total_relation_size('{table_name}'))
                    """))
                    table_size = size_result.scalar()
                    
                    report["tables"][table_name] = {
                        "description": description,
                        "row_count": row_count,
                        "size": table_size
                    }
                    
                except Exception as e:
                    report["tables"][table_name] = {
                        "description": description,
                        "error": str(e)
                    }
            
            return report
    
    async def vacuum_all_tables(self) -> Dict[str, Any]:
        """Run VACUUM on all tables for maintenance"""
        async with AsyncSessionLocal() as session:
            results = []
            
            # Get all user tables
            tables_result = await session.execute(text("""
                SELECT tablename FROM pg_tables WHERE schemaname = 'public'
            """))
            
            tables = [row[0] for row in tables_result.fetchall()]
            
            for table in tables:
                try:
                    start_time = datetime.now()
                    await session.execute(text(f"VACUUM ANALYZE {table}"))
                    execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
                    
                    results.append({
                        "table": table,
                        "success": True,
                        "execution_time_ms": execution_time
                    })
                    
                except Exception as e:
                    results.append({
                        "table": table,
                        "success": False,
                        "error": str(e)
                    })
            
            await session.commit()
            
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "results": results
            }

# Global retention manager instance
retention_manager = DataRetentionManager()

def get_retention_manager() -> DataRetentionManager:
    """Get the global retention manager instance"""
    return retention_manager