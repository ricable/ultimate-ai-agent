# File: backend/database/service.py
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc, asc
from sqlalchemy.orm import selectinload, joinedload
import uuid
import hashlib
import logging

from .connection import AsyncSessionLocal
from .session import SessionManager
from .migrations.manager import MigrationManager
from .retention import DataRetentionManager
from ..models.user import User, UserRole, RefreshToken
from ..models.conversation import Conversation, Message, MessageType, MessageStatus
from ..models.document import Document, DocumentMetadata, DocumentProcessingJob
from ..models.analytics import UserSession, AgentUsage, SystemMetrics, AuditLog

logger = logging.getLogger(__name__)

class DatabaseService:
    """High-level database service for application operations"""
    
    def __init__(self):
        self.session_manager = SessionManager()
        self.migration_manager = MigrationManager()
        self.retention_manager = DataRetentionManager()
    
    async def initialize(self):
        """Initialize database service"""
        await self.migration_manager.create_migration_table()
        await self.session_manager.start_cleanup_task()
        logger.info("Database service initialized")
    
    async def shutdown(self):
        """Shutdown database service"""
        await self.session_manager.stop_cleanup_task()
        logger.info("Database service shutdown")
    
    # User Management
    async def create_user(self, username: str, email: str, hashed_password: str, 
                         full_name: str = None, roles: List[str] = None) -> User:
        """Create a new user"""
        async with AsyncSessionLocal() as session:
            user = User(
                username=username,
                email=email,
                hashed_password=hashed_password,
                full_name=full_name,
                roles=roles or ["user"]
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(User).where(User.username == username)
            )
            return result.scalar_one_or_none()
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            return result.scalar_one_or_none()
    
    async def update_user_last_login(self, user_id: str):
        """Update user's last login timestamp"""
        async with AsyncSessionLocal() as session:
            user = await session.get(User, user_id)
            if user:
                user.last_login = datetime.now(timezone.utc)
                await session.commit()
    
    # Refresh Token Management
    async def create_refresh_token(self, user_id: str, token_hash: str, 
                                 expires_at: datetime, device_info: Dict = None) -> RefreshToken:
        """Create a refresh token"""
        async with AsyncSessionLocal() as session:
            refresh_token = RefreshToken(
                user_id=user_id,
                token_hash=token_hash,
                expires_at=expires_at,
                device_info=device_info or {}
            )
            session.add(refresh_token)
            await session.commit()
            await session.refresh(refresh_token)
            return refresh_token
    
    async def get_refresh_token(self, token_hash: str) -> Optional[RefreshToken]:
        """Get refresh token by hash"""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(RefreshToken).where(
                    and_(
                        RefreshToken.token_hash == token_hash,
                        RefreshToken.is_active == True
                    )
                ).options(joinedload(RefreshToken.user))
            )
            return result.scalar_one_or_none()
    
    async def revoke_refresh_token(self, token_hash: str):
        """Revoke a refresh token"""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(RefreshToken).where(RefreshToken.token_hash == token_hash)
            )
            token = result.scalar_one_or_none()
            if token:
                token.revoke()
                await session.commit()
    
    # Conversation Management
    async def create_conversation(self, user_id: str, title: str = None, 
                                framework: str = None, agent_id: str = None,
                                context: Dict = None) -> Conversation:
        """Create a new conversation"""
        async with AsyncSessionLocal() as session:
            conversation = Conversation(
                user_id=user_id,
                title=title,
                framework=framework,
                agent_id=agent_id,
                context=context or {}
            )
            session.add(conversation)
            await session.commit()
            await session.refresh(conversation)
            return conversation
    
    async def get_conversation(self, conversation_id: str, 
                             include_messages: bool = False) -> Optional[Conversation]:
        """Get conversation by ID"""
        async with AsyncSessionLocal() as session:
            query = select(Conversation).where(Conversation.id == conversation_id)
            
            if include_messages:
                query = query.options(selectinload(Conversation.messages))
            
            result = await session.execute(query)
            return result.scalar_one_or_none()
    
    async def get_user_conversations(self, user_id: str, limit: int = 50, 
                                   offset: int = 0) -> List[Conversation]:
        """Get user's conversations"""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .order_by(desc(Conversation.last_message_at))
                .limit(limit)
                .offset(offset)
            )
            return result.scalars().all()
    
    async def create_message(self, conversation_id: str, content: str, 
                           message_type: MessageType, sender_id: str = None,
                           framework: str = None, metadata: Dict = None) -> Message:
        """Create a new message"""
        async with AsyncSessionLocal() as session:
            message = Message(
                conversation_id=conversation_id,
                content=content,
                message_type=message_type,
                sender_id=sender_id,
                framework=framework,
                metadata=metadata or {}
            )
            session.add(message)
            
            # Update conversation message count and timestamp
            conversation = await session.get(Conversation, conversation_id)
            if conversation:
                conversation.increment_message_count()
            
            await session.commit()
            await session.refresh(message)
            return message
    
    # Document Management
    async def create_document(self, filename: str, file_size: int, 
                            content_type: str, uploaded_by: str,
                            file_hash: str = None, metadata: Dict = None) -> Document:
        """Create a new document record"""
        async with AsyncSessionLocal() as session:
            document = Document(
                filename=filename,
                original_filename=filename,
                file_size=file_size,
                content_type=content_type,
                uploaded_by=uploaded_by,
                file_hash=file_hash,
                metadata=metadata or {}
            )
            session.add(document)
            await session.commit()
            await session.refresh(document)
            return document
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID"""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Document).where(Document.id == document_id)
                .options(selectinload(Document.document_metadata))
            )
            return result.scalar_one_or_none()
    
    async def get_user_documents(self, user_id: str, limit: int = 50, 
                               offset: int = 0) -> List[Document]:
        """Get user's documents"""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Document)
                .where(Document.uploaded_by == user_id)
                .order_by(desc(Document.created_at))
                .limit(limit)
                .offset(offset)
            )
            return result.scalars().all()
    
    async def update_document_content(self, document_id: str, content: str, 
                                    extracted_metadata: Dict = None):
        """Update document content and metadata"""
        async with AsyncSessionLocal() as session:
            document = await session.get(Document, document_id)
            if document:
                document.content = content
                document.content_preview = content[:1000] if content else None
                document.word_count = len(content.split()) if content else 0
                if extracted_metadata:
                    document.extracted_metadata = extracted_metadata
                document.mark_completed()
                await session.commit()
    
    # Analytics and Usage Tracking
    async def create_user_session(self, user_id: str, session_token: str,
                                ip_address: str = None, user_agent: str = None,
                                device_info: Dict = None) -> UserSession:
        """Create a new user session"""
        async with AsyncSessionLocal() as session:
            user_session = UserSession(
                user_id=user_id,
                session_token=session_token,
                ip_address=ip_address,
                user_agent=user_agent,
                device_info=device_info or {}
            )
            session.add(user_session)
            await session.commit()
            await session.refresh(user_session)
            return user_session
    
    async def record_agent_usage(self, framework: str, user_id: str = None,
                               request_type: str = "chat", input_tokens: int = 0,
                               output_tokens: int = 0, response_time_ms: int = None,
                               success: bool = True, metadata: Dict = None) -> AgentUsage:
        """Record agent usage metrics"""
        async with AsyncSessionLocal() as session:
            usage = AgentUsage(
                framework=framework,
                user_id=user_id,
                request_type=request_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                response_time_ms=response_time_ms,
                success=success,
                metadata=metadata or {}
            )
            session.add(usage)
            await session.commit()
            await session.refresh(usage)
            return usage
    
    async def record_system_metric(self, metric_name: str, value: float,
                                  metric_type: str = "gauge", component: str = None,
                                  labels: Dict = None) -> SystemMetrics:
        """Record system metric"""
        async with AsyncSessionLocal() as session:
            metric = SystemMetrics(
                metric_name=metric_name,
                value=value,
                metric_type=metric_type,
                component=component,
                labels=labels or {}
            )
            session.add(metric)
            await session.commit()
            await session.refresh(metric)
            return metric
    
    async def create_audit_log(self, user_id: str, action: str, resource_type: str,
                             description: str, success: bool = True,
                             metadata: Dict = None) -> AuditLog:
        """Create audit log entry"""
        async with AsyncSessionLocal() as session:
            audit_log = AuditLog(
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                description=description,
                success=success,
                metadata=metadata or {}
            )
            session.add(audit_log)
            await session.commit()
            await session.refresh(audit_log)
            return audit_log
    
    # Statistics and Reports
    async def get_usage_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get usage statistics for the last N days"""
        async with AsyncSessionLocal() as session:
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # User statistics
            user_count = await session.execute(select(func.count(User.id)))
            active_users = await session.execute(
                select(func.count(User.id)).where(User.last_login >= start_date)
            )
            
            # Conversation statistics
            conversation_count = await session.execute(
                select(func.count(Conversation.id)).where(Conversation.created_at >= start_date)
            )
            message_count = await session.execute(
                select(func.count(Message.id)).where(Message.created_at >= start_date)
            )
            
            # Document statistics
            document_count = await session.execute(
                select(func.count(Document.id)).where(Document.created_at >= start_date)
            )
            
            # Agent usage statistics
            agent_usage_count = await session.execute(
                select(func.count(AgentUsage.id)).where(AgentUsage.started_at >= start_date)
            )
            
            return {
                "period_days": days,
                "start_date": start_date.isoformat(),
                "users": {
                    "total": user_count.scalar(),
                    "active": active_users.scalar()
                },
                "conversations": {
                    "total": conversation_count.scalar(),
                    "messages": message_count.scalar()
                },
                "documents": {
                    "uploaded": document_count.scalar()
                },
                "agent_usage": {
                    "requests": agent_usage_count.scalar()
                }
            }
    
    # Database Maintenance
    async def get_database_health(self) -> Dict[str, Any]:
        """Get database health status"""
        return await self.migration_manager.get_database_status()
    
    async def optimize_database(self) -> Dict[str, Any]:
        """Optimize database performance"""
        return await self.migration_manager.optimize_database()
    
    async def apply_retention_policies(self) -> Dict[str, Any]:
        """Apply data retention policies"""
        return await self.retention_manager.apply_retention_policies()
    
    async def backup_database(self, backup_path: str = None) -> Dict[str, Any]:
        """Create database backup"""
        return await self.migration_manager.backup_database(backup_path)

# Global database service instance
database_service = DatabaseService()

def get_database_service() -> DatabaseService:
    """Get the global database service instance"""
    return database_service