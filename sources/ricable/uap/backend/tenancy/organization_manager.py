# File: backend/tenancy/organization_manager.py
"""
Organization and tenant management service with enterprise features
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from fastapi import HTTPException, status
import uuid
import asyncio
import logging
from dataclasses import dataclass

from .models import (
    Organization, Tenant, TenantUser, TenantInvitation, TenantSettings,
    OrganizationCreate, TenantCreate, TenantUpdate, TenantUserCreate,
    TenantInvitationCreate, TenantUsageReport, TenantStatus, OrganizationType
)
from ..services.auth import User, UserInDB

logger = logging.getLogger(__name__)

@dataclass
class TenantContext:
    """Current tenant context for request processing"""
    tenant_id: str
    organization_id: str
    user_id: str
    roles: List[str]
    permissions: List[str]
    limits: Dict[str, Any]

class OrganizationManager:
    """Enterprise organization and tenant management"""
    
    def __init__(self):
        # In-memory storage for demo (replace with database in production)
        self.organizations: Dict[str, Organization] = {}
        self.tenants: Dict[str, Tenant] = {}
        self.tenant_users: Dict[str, List[TenantUser]] = {}  # tenant_id -> [users]
        self.user_tenants: Dict[str, List[str]] = {}  # user_id -> [tenant_ids]
        self.invitations: Dict[str, TenantInvitation] = {}
        self.tenant_settings: Dict[str, TenantSettings] = {}
        
        # Current request context
        self.current_tenant_context: Optional[TenantContext] = None
        
        # Usage tracking
        self.usage_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default organization for backwards compatibility
        self._create_default_organization()
    
    def _create_default_organization(self):
        """Create default organization for existing users"""
        default_org = Organization(
            id="default",
            name="Default Organization",
            slug="default",
            email="admin@uap.local",
            organization_type=OrganizationType.BUSINESS,
            status=TenantStatus.ACTIVE,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        default_tenant = Tenant(
            id="default-tenant",
            organization_id="default",
            name="Default Tenant",
            slug="default",
            status=TenantStatus.ACTIVE,
            tier="enterprise",
            features=["all"],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        self.organizations["default"] = default_org
        self.tenants["default-tenant"] = default_tenant
        self.tenant_users["default-tenant"] = []
    
    # Organization Management
    
    async def create_organization(self, org_data: OrganizationCreate, creator_user: UserInDB) -> Organization:
        """Create a new organization"""
        # Check if slug is already taken
        for org in self.organizations.values():
            if org.slug == org_data.slug:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Organization slug '{org_data.slug}' is already taken"
                )
        
        # Create organization
        org_id = str(uuid.uuid4())
        organization = Organization(
            id=org_id,
            name=org_data.name,
            slug=org_data.slug,
            email=org_data.email,
            organization_type=org_data.organization_type,
            phone=org_data.phone,
            address=org_data.address or {},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        self.organizations[org_id] = organization
        
        # Create default tenant for the organization
        tenant = await self.create_tenant(
            org_id,
            TenantCreate(
                name=f"{org_data.name} Main",
                slug="main",
                tier="basic"
            ),
            creator_user
        )
        
        # Make creator an admin of the organization
        await self.add_user_to_tenant(tenant.id, creator_user, ["admin"], is_tenant_admin=True)
        
        logger.info(f"Created organization {org_id} with tenant {tenant.id}")
        return organization
    
    async def get_organization(self, org_id: str) -> Optional[Organization]:
        """Get organization by ID"""
        return self.organizations.get(org_id)
    
    async def get_organization_by_slug(self, slug: str) -> Optional[Organization]:
        """Get organization by slug"""
        for org in self.organizations.values():
            if org.slug == slug:
                return org
        return None
    
    async def update_organization(self, org_id: str, updates: Dict[str, Any]) -> Organization:
        """Update organization"""
        if org_id not in self.organizations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )
        
        org = self.organizations[org_id]
        
        # Update fields
        for field, value in updates.items():
            if hasattr(org, field):
                setattr(org, field, value)
        
        org.updated_at = datetime.now(timezone.utc)
        self.organizations[org_id] = org
        
        return org
    
    async def delete_organization(self, org_id: str):
        """Delete organization and all its tenants"""
        if org_id not in self.organizations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )
        
        # Delete all tenants
        org_tenants = [t for t in self.tenants.values() if t.organization_id == org_id]
        for tenant in org_tenants:
            await self.delete_tenant(tenant.id)
        
        # Delete organization
        del self.organizations[org_id]
        logger.info(f"Deleted organization {org_id}")
    
    # Tenant Management
    
    async def create_tenant(self, org_id: str, tenant_data: TenantCreate, creator_user: UserInDB) -> Tenant:
        """Create a new tenant within an organization"""
        if org_id not in self.organizations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )
        
        # Check if slug is unique within organization
        for tenant in self.tenants.values():
            if tenant.organization_id == org_id and tenant.slug == tenant_data.slug:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Tenant slug '{tenant_data.slug}' already exists in this organization"
                )
        
        # Create tenant
        tenant_id = str(uuid.uuid4())
        tenant = Tenant(
            id=tenant_id,
            organization_id=org_id,
            name=tenant_data.name,
            slug=tenant_data.slug,
            subdomain=tenant_data.subdomain,
            tier=tenant_data.tier,
            features=tenant_data.features,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        self.tenants[tenant_id] = tenant
        self.tenant_users[tenant_id] = []
        
        # Create default tenant settings
        settings = TenantSettings(
            tenant_id=tenant_id,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        self.tenant_settings[tenant_id] = settings
        
        logger.info(f"Created tenant {tenant_id} in organization {org_id}")
        return tenant
    
    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID"""
        return self.tenants.get(tenant_id)
    
    async def get_tenant_by_slug(self, org_id: str, slug: str) -> Optional[Tenant]:
        """Get tenant by organization and slug"""
        for tenant in self.tenants.values():
            if tenant.organization_id == org_id and tenant.slug == slug:
                return tenant
        return None
    
    async def update_tenant(self, tenant_id: str, updates: TenantUpdate) -> Tenant:
        """Update tenant"""
        if tenant_id not in self.tenants:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        tenant = self.tenants[tenant_id]
        
        # Update fields
        update_data = updates.dict(exclude_unset=True)
        for field, value in update_data.items():
            if hasattr(tenant, field):
                setattr(tenant, field, value)
        
        tenant.updated_at = datetime.now(timezone.utc)
        self.tenants[tenant_id] = tenant
        
        return tenant
    
    async def delete_tenant(self, tenant_id: str):
        """Delete tenant and all associated data"""
        if tenant_id not in self.tenants:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        # Remove all users from tenant
        if tenant_id in self.tenant_users:
            for tenant_user in self.tenant_users[tenant_id]:
                await self.remove_user_from_tenant(tenant_id, tenant_user.user_id)
        
        # Delete tenant data
        if tenant_id in self.tenant_users:
            del self.tenant_users[tenant_id]
        if tenant_id in self.tenant_settings:
            del self.tenant_settings[tenant_id]
        del self.tenants[tenant_id]
        
        logger.info(f"Deleted tenant {tenant_id}")
    
    # User-Tenant Management
    
    async def add_user_to_tenant(
        self, 
        tenant_id: str, 
        user: UserInDB, 
        roles: List[str] = None,
        is_tenant_admin: bool = False
    ) -> TenantUser:
        """Add user to tenant with specific roles"""
        if tenant_id not in self.tenants:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        tenant = self.tenants[tenant_id]
        
        # Check if user already exists in tenant
        existing_users = self.tenant_users.get(tenant_id, [])
        for existing_user in existing_users:
            if existing_user.user_id == user.id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="User already exists in this tenant"
                )
        
        # Create tenant user
        tenant_user = TenantUser(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            organization_id=tenant.organization_id,
            user_id=user.id,
            tenant_roles=roles or ["user"],
            is_tenant_admin=is_tenant_admin,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        # Add to mappings
        if tenant_id not in self.tenant_users:
            self.tenant_users[tenant_id] = []
        self.tenant_users[tenant_id].append(tenant_user)
        
        if user.id not in self.user_tenants:
            self.user_tenants[user.id] = []
        self.user_tenants[user.id].append(tenant_id)
        
        # Update tenant user count
        tenant.current_users += 1
        self.tenants[tenant_id] = tenant
        
        return tenant_user
    
    async def remove_user_from_tenant(self, tenant_id: str, user_id: str):
        """Remove user from tenant"""
        if tenant_id not in self.tenant_users:
            return
        
        # Remove from tenant users
        self.tenant_users[tenant_id] = [
            u for u in self.tenant_users[tenant_id] if u.user_id != user_id
        ]
        
        # Remove from user tenants
        if user_id in self.user_tenants:
            self.user_tenants[user_id] = [
                tid for tid in self.user_tenants[user_id] if tid != tenant_id
            ]
        
        # Update tenant user count
        if tenant_id in self.tenants:
            self.tenants[tenant_id].current_users -= 1
    
    async def get_user_tenants(self, user_id: str) -> List[Tenant]:
        """Get all tenants for a user"""
        tenant_ids = self.user_tenants.get(user_id, [])
        return [self.tenants[tid] for tid in tenant_ids if tid in self.tenants]
    
    async def get_tenant_users(self, tenant_id: str) -> List[TenantUser]:
        """Get all users in a tenant"""
        return self.tenant_users.get(tenant_id, [])
    
    # Invitation Management
    
    async def create_invitation(
        self, 
        tenant_id: str, 
        invitation_data: TenantInvitationCreate,
        inviter: UserInDB
    ) -> TenantInvitation:
        """Create invitation to join tenant"""
        if tenant_id not in self.tenants:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        tenant = self.tenants[tenant_id]
        
        invitation = TenantInvitation(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            organization_id=tenant.organization_id,
            inviter_id=inviter.id,
            email=invitation_data.email,
            roles=invitation_data.roles,
            message=invitation_data.message,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=invitation_data.expires_in_hours),
            created_at=datetime.now(timezone.utc)
        )
        
        self.invitations[invitation.id] = invitation
        return invitation
    
    async def accept_invitation(self, invitation_id: str, user: UserInDB) -> TenantUser:
        """Accept tenant invitation"""
        if invitation_id not in self.invitations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invitation not found"
            )
        
        invitation = self.invitations[invitation_id]
        
        # Check if invitation is still valid
        if invitation.status != "pending":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invitation is no longer valid"
            )
        
        if datetime.now(timezone.utc) > invitation.expires_at:
            invitation.status = "expired"
            self.invitations[invitation_id] = invitation
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invitation has expired"
            )
        
        # Add user to tenant
        tenant_user = await self.add_user_to_tenant(
            invitation.tenant_id,
            user,
            invitation.roles
        )
        
        # Mark invitation as accepted
        invitation.status = "accepted"
        invitation.accepted_at = datetime.now(timezone.utc)
        self.invitations[invitation_id] = invitation
        
        return tenant_user
    
    # Usage Tracking and Reporting
    
    async def track_usage(self, tenant_id: str, metric: str, value: Union[int, float]):
        """Track usage metrics for billing"""
        if tenant_id not in self.usage_metrics:
            self.usage_metrics[tenant_id] = {}
        
        current_time = datetime.now(timezone.utc)
        month_key = current_time.strftime("%Y-%m")
        
        if month_key not in self.usage_metrics[tenant_id]:
            self.usage_metrics[tenant_id][month_key] = {}
        
        if metric not in self.usage_metrics[tenant_id][month_key]:
            self.usage_metrics[tenant_id][month_key][metric] = 0
        
        self.usage_metrics[tenant_id][month_key][metric] += value
    
    async def get_usage_report(self, tenant_id: str, period: str = None) -> TenantUsageReport:
        """Generate usage report for tenant"""
        if tenant_id not in self.tenants:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        tenant = self.tenants[tenant_id]
        current_time = datetime.now(timezone.utc)
        
        if not period:
            period = current_time.strftime("%Y-%m")
        
        # Get usage metrics for period
        usage_data = self.usage_metrics.get(tenant_id, {}).get(period, {})
        
        # Calculate period boundaries
        period_start = datetime.strptime(f"{period}-01", "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if period == current_time.strftime("%Y-%m"):
            period_end = current_time
        else:
            next_month = period_start.replace(month=period_start.month + 1) if period_start.month < 12 else period_start.replace(year=period_start.year + 1, month=1)
            period_end = next_month - timedelta(seconds=1)
        
        return TenantUsageReport(
            tenant_id=tenant_id,
            reporting_period=period,
            active_users=len(self.tenant_users.get(tenant_id, [])),
            total_api_calls=usage_data.get("api_calls", 0),
            storage_used_gb=tenant.storage_used_gb,
            bandwidth_used_gb=usage_data.get("bandwidth_gb", 0.0),
            feature_usage=usage_data.get("features", {}),
            billing_tier=tenant.tier,
            generated_at=current_time,
            period_start=period_start,
            period_end=period_end
        )
    
    # Context Management
    
    def set_tenant_context(self, context: TenantContext):
        """Set current tenant context for request"""
        self.current_tenant_context = context
    
    def get_tenant_context(self) -> Optional[TenantContext]:
        """Get current tenant context"""
        return self.current_tenant_context
    
    def clear_tenant_context(self):
        """Clear current tenant context"""
        self.current_tenant_context = None
    
    # Utility Methods
    
    async def list_organizations(self, limit: int = 50, offset: int = 0) -> Tuple[List[Organization], int]:
        """List organizations with pagination"""
        orgs = list(self.organizations.values())
        total = len(orgs)
        
        # Sort by created_at desc
        orgs.sort(key=lambda x: x.created_at, reverse=True)
        
        # Paginate
        paginated_orgs = orgs[offset:offset + limit]
        
        return paginated_orgs, total
    
    async def list_tenants(self, org_id: str = None, limit: int = 50, offset: int = 0) -> Tuple[List[Tenant], int]:
        """List tenants with optional organization filter"""
        tenants = list(self.tenants.values())
        
        if org_id:
            tenants = [t for t in tenants if t.organization_id == org_id]
        
        total = len(tenants)
        
        # Sort by created_at desc
        tenants.sort(key=lambda x: x.created_at, reverse=True)
        
        # Paginate
        paginated_tenants = tenants[offset:offset + limit]
        
        return paginated_tenants, total

# Global organization manager instance
organization_manager = OrganizationManager()