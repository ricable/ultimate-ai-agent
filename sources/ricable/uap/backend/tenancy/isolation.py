# File: backend/tenancy/isolation.py
"""
Data isolation manager for multi-tenant architecture
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from fastapi import HTTPException, status
import logging
import hashlib
import os
from abc import ABC, abstractmethod

from .tenant_context import TenantContext, get_current_tenant

logger = logging.getLogger(__name__)

class IsolationLevel:
    """Data isolation levels"""
    DATABASE = "database"  # Separate database per tenant
    SCHEMA = "schema"     # Separate schema per tenant  
    ROW = "row"          # Row-level security with tenant_id column

class DataIsolationStrategy(ABC):
    """Abstract base class for data isolation strategies"""
    
    @abstractmethod
    async def get_connection_params(self, tenant_context: TenantContext) -> Dict[str, Any]:
        """Get database connection parameters for tenant"""
        pass
    
    @abstractmethod
    async def apply_filters(self, query: str, tenant_context: TenantContext) -> str:
        """Apply tenant filters to database query"""
        pass
    
    @abstractmethod
    async def validate_access(self, table_name: str, tenant_context: TenantContext) -> bool:
        """Validate tenant access to table/resource"""
        pass

class DatabaseIsolationStrategy(DataIsolationStrategy):
    """Database-level isolation - separate database per tenant"""
    
    def __init__(self, base_connection_string: str):
        self.base_connection_string = base_connection_string
    
    async def get_connection_params(self, tenant_context: TenantContext) -> Dict[str, Any]:
        """Generate tenant-specific database connection"""
        db_name = f"uap_tenant_{tenant_context.tenant_id}"
        
        # Replace database name in connection string
        connection_string = self.base_connection_string.replace(
            "database=uap", f"database={db_name}"
        )
        
        return {
            "connection_string": connection_string,
            "database_name": db_name,
            "isolation_level": IsolationLevel.DATABASE
        }
    
    async def apply_filters(self, query: str, tenant_context: TenantContext) -> str:
        """No additional filters needed for database isolation"""
        return query
    
    async def validate_access(self, table_name: str, tenant_context: TenantContext) -> bool:
        """All tables accessible within tenant database"""
        return True

class SchemaIsolationStrategy(DataIsolationStrategy):
    """Schema-level isolation - separate schema per tenant"""
    
    def __init__(self, base_connection_string: str):
        self.base_connection_string = base_connection_string
    
    async def get_connection_params(self, tenant_context: TenantContext) -> Dict[str, Any]:
        """Generate tenant-specific schema connection"""
        schema_name = f"tenant_{tenant_context.tenant_id}"
        
        return {
            "connection_string": self.base_connection_string,
            "schema_name": schema_name,
            "isolation_level": IsolationLevel.SCHEMA
        }
    
    async def apply_filters(self, query: str, tenant_context: TenantContext) -> str:
        """Add schema prefix to table names"""
        schema_name = f"tenant_{tenant_context.tenant_id}"
        
        # Simple table name replacement (production would use proper SQL parsing)
        common_tables = [
            "users", "documents", "conversations", "agents", 
            "workflows", "integrations", "settings", "audit_logs"
        ]
        
        modified_query = query
        for table in common_tables:
            modified_query = modified_query.replace(
                f" {table} ", f" {schema_name}.{table} "
            )
            modified_query = modified_query.replace(
                f" {table}.", f" {schema_name}.{table}."
            )
        
        return modified_query
    
    async def validate_access(self, table_name: str, tenant_context: TenantContext) -> bool:
        """Validate table access within tenant schema"""
        allowed_tables = [
            "users", "documents", "conversations", "agents",
            "workflows", "integrations", "settings", "audit_logs"
        ]
        return table_name in allowed_tables

class RowLevelIsolationStrategy(DataIsolationStrategy):
    """Row-level isolation - tenant_id column filtering"""
    
    def __init__(self, base_connection_string: str):
        self.base_connection_string = base_connection_string
    
    async def get_connection_params(self, tenant_context: TenantContext) -> Dict[str, Any]:
        """Use shared database with row-level security"""
        return {
            "connection_string": self.base_connection_string,
            "tenant_id": tenant_context.tenant_id,
            "isolation_level": IsolationLevel.ROW
        }
    
    async def apply_filters(self, query: str, tenant_context: TenantContext) -> str:
        """Add tenant_id filters to WHERE clauses"""
        tenant_id = tenant_context.tenant_id
        
        # Add tenant_id filter to WHERE clause
        if "WHERE" in query.upper():
            # Add to existing WHERE clause
            modified_query = query.replace(
                "WHERE", f"WHERE tenant_id = '{tenant_id}' AND"
            )
        else:
            # Add WHERE clause before ORDER BY, GROUP BY, etc.
            for keyword in ["ORDER BY", "GROUP BY", "HAVING", "LIMIT"]:
                if keyword in query.upper():
                    query = query.replace(
                        keyword, f"WHERE tenant_id = '{tenant_id}' {keyword}"
                    )
                    break
            else:
                # Add WHERE clause at the end
                modified_query = f"{query} WHERE tenant_id = '{tenant_id}'"
        
        return modified_query
    
    async def validate_access(self, table_name: str, tenant_context: TenantContext) -> bool:
        """Validate table has tenant_id column"""
        tenant_aware_tables = [
            "users", "documents", "conversations", "agents",
            "workflows", "integrations", "settings", "audit_logs",
            "tenant_users", "organizations", "tenants"
        ]
        return table_name in tenant_aware_tables

class DataIsolationManager:
    """Manages data isolation across different tenancy strategies"""
    
    def __init__(self):
        self.strategies: Dict[str, DataIsolationStrategy] = {}
        self.default_strategy = IsolationLevel.SCHEMA
        
        # Initialize with default connection string
        base_connection = os.getenv(
            "DATABASE_URL", 
            "postgresql://uap:password@localhost:5432/uap"
        )
        
        # Register isolation strategies
        self.register_strategy(
            IsolationLevel.DATABASE, 
            DatabaseIsolationStrategy(base_connection)
        )
        self.register_strategy(
            IsolationLevel.SCHEMA, 
            SchemaIsolationStrategy(base_connection)
        )
        self.register_strategy(
            IsolationLevel.ROW, 
            RowLevelIsolationStrategy(base_connection)
        )
    
    def register_strategy(self, level: str, strategy: DataIsolationStrategy):
        """Register isolation strategy"""
        self.strategies[level] = strategy
        logger.info(f"Registered isolation strategy: {level}")
    
    async def get_strategy(self, tenant_context: TenantContext) -> DataIsolationStrategy:
        """Get isolation strategy for tenant"""
        isolation_level = tenant_context.isolation_level or self.default_strategy
        
        if isolation_level not in self.strategies:
            logger.warning(f"Unknown isolation level: {isolation_level}, using default")
            isolation_level = self.default_strategy
        
        return self.strategies[isolation_level]
    
    async def get_connection_params(self, tenant_context: TenantContext = None) -> Dict[str, Any]:
        """Get database connection parameters for current tenant"""
        if not tenant_context:
            tenant_context = get_current_tenant()
            if not tenant_context:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Tenant context required for data isolation"
                )
        
        strategy = await self.get_strategy(tenant_context)
        return await strategy.get_connection_params(tenant_context)
    
    async def apply_query_filters(self, query: str, tenant_context: TenantContext = None) -> str:
        """Apply tenant-specific filters to database query"""
        if not tenant_context:
            tenant_context = get_current_tenant()
            if not tenant_context:
                # Return query as-is if no tenant context (for system operations)
                return query
        
        strategy = await self.get_strategy(tenant_context)
        filtered_query = await strategy.apply_filters(query, tenant_context)
        
        logger.debug(f"Applied tenant filters - Original: {query[:100]}...")
        logger.debug(f"Filtered: {filtered_query[:100]}...")
        
        return filtered_query
    
    async def validate_table_access(self, table_name: str, tenant_context: TenantContext = None) -> bool:
        """Validate tenant access to table/resource"""
        if not tenant_context:
            tenant_context = get_current_tenant()
            if not tenant_context:
                return True  # Allow system access when no tenant context
        
        strategy = await self.get_strategy(tenant_context)
        return await strategy.validate_access(table_name, tenant_context)
    
    async def create_tenant_resources(self, tenant_context: TenantContext):
        """Create tenant-specific database resources"""
        strategy = await self.get_strategy(tenant_context)
        
        if tenant_context.isolation_level == IsolationLevel.DATABASE:
            await self._create_tenant_database(tenant_context)
        elif tenant_context.isolation_level == IsolationLevel.SCHEMA:
            await self._create_tenant_schema(tenant_context)
        # Row-level isolation doesn't need separate resources
        
        logger.info(f"Created tenant resources for {tenant_context.tenant_id}")
    
    async def _create_tenant_database(self, tenant_context: TenantContext):
        """Create separate database for tenant"""
        db_name = f"uap_tenant_{tenant_context.tenant_id}"
        
        # In production, this would execute actual database creation
        logger.info(f"Creating tenant database: {db_name}")
        
        # TODO: Execute database creation SQL
        # CREATE DATABASE {db_name};
        # Run schema migrations on new database
    
    async def _create_tenant_schema(self, tenant_context: TenantContext):
        """Create separate schema for tenant"""
        schema_name = f"tenant_{tenant_context.tenant_id}"
        
        # In production, this would execute actual schema creation
        logger.info(f"Creating tenant schema: {schema_name}")
        
        # TODO: Execute schema creation SQL
        # CREATE SCHEMA {schema_name};
        # CREATE TABLE {schema_name}.users (...);
        # etc.
    
    async def destroy_tenant_resources(self, tenant_context: TenantContext):
        """Destroy tenant-specific database resources"""
        strategy = await self.get_strategy(tenant_context)
        
        if tenant_context.isolation_level == IsolationLevel.DATABASE:
            await self._destroy_tenant_database(tenant_context)
        elif tenant_context.isolation_level == IsolationLevel.SCHEMA:
            await self._destroy_tenant_schema(tenant_context)
        # Row-level isolation: delete rows with tenant_id
        elif tenant_context.isolation_level == IsolationLevel.ROW:
            await self._delete_tenant_rows(tenant_context)
        
        logger.info(f"Destroyed tenant resources for {tenant_context.tenant_id}")
    
    async def _destroy_tenant_database(self, tenant_context: TenantContext):
        """Drop tenant database"""
        db_name = f"uap_tenant_{tenant_context.tenant_id}"
        logger.info(f"Dropping tenant database: {db_name}")
        # TODO: DROP DATABASE {db_name};
    
    async def _destroy_tenant_schema(self, tenant_context: TenantContext):
        """Drop tenant schema"""
        schema_name = f"tenant_{tenant_context.tenant_id}"
        logger.info(f"Dropping tenant schema: {schema_name}")
        # TODO: DROP SCHEMA {schema_name} CASCADE;
    
    async def _delete_tenant_rows(self, tenant_context: TenantContext):
        """Delete all rows for tenant"""
        tenant_id = tenant_context.tenant_id
        logger.info(f"Deleting all data for tenant: {tenant_id}")
        # TODO: DELETE FROM table WHERE tenant_id = '{tenant_id}';

class EncryptionManager:
    """Manages tenant-specific data encryption"""
    
    def __init__(self):
        self.master_key = os.getenv("ENCRYPTION_MASTER_KEY", "default-master-key")
    
    def generate_tenant_key(self, tenant_id: str) -> str:
        """Generate unique encryption key for tenant"""
        combined = f"{self.master_key}:{tenant_id}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def encrypt_data(self, data: str, tenant_id: str) -> str:
        """Encrypt data with tenant-specific key"""
        # Simplified encryption (use proper crypto library in production)
        tenant_key = self.generate_tenant_key(tenant_id)
        
        # TODO: Implement actual encryption
        # For now, just return base64 encoded data with key hash
        import base64
        encrypted = base64.b64encode(data.encode()).decode()
        return f"encrypted:{tenant_key[:8]}:{encrypted}"
    
    def decrypt_data(self, encrypted_data: str, tenant_id: str) -> str:
        """Decrypt data with tenant-specific key"""
        if not encrypted_data.startswith("encrypted:"):
            return encrypted_data  # Not encrypted
        
        parts = encrypted_data.split(":", 2)
        if len(parts) != 3:
            raise ValueError("Invalid encrypted data format")
        
        tenant_key = self.generate_tenant_key(tenant_id)
        if parts[1] != tenant_key[:8]:
            raise ValueError("Invalid encryption key for tenant")
        
        # TODO: Implement actual decryption
        import base64
        return base64.b64decode(parts[2]).decode()

# Global instances
data_isolation_manager = DataIsolationManager()
encryption_manager = EncryptionManager()