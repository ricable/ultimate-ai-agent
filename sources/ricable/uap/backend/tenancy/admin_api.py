# File: backend/tenancy/admin_api.py
"""
Enterprise admin API endpoints for multi-tenancy management
"""

from fastapi import APIRouter, HTTPException, Depends, status, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, EmailStr
import uuid
import logging

from .models import (
    Organization, Tenant, TenantUser, TenantInvitation, TenantSettings,
    OrganizationCreate, TenantCreate, TenantUpdate, TenantUserCreate,
    TenantInvitationCreate, TenantUsageReport, TenantStatus, OrganizationType
)
from .organization_manager import organization_manager
from .tenant_context import (
    require_tenant_context, require_tenant_admin, get_current_tenant,
    TenantContext
)
from ..billing.models import (
    Subscription, BillingPlan, Invoice, Payment, BillingReport,
    SubscriptionCreate, SubscriptionUpdate
)
from ..billing.subscription_manager import subscription_manager
from ..services.auth import UserInDB, require_permission

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["Enterprise Admin"])

# Request/Response Models

class OrganizationResponse(BaseModel):
    """Organization response model"""
    id: str
    name: str
    slug: str
    email: EmailStr
    organization_type: OrganizationType
    status: TenantStatus
    tenant_count: int
    user_count: int
    created_at: datetime
    updated_at: datetime
    branding: Dict[str, Any] = {}
    white_label_enabled: bool = False

class TenantResponse(BaseModel):
    """Tenant response model"""
    id: str
    organization_id: str
    name: str
    slug: str
    subdomain: Optional[str] = None
    status: TenantStatus
    tier: str
    features: List[str]
    current_users: int
    storage_used_gb: float
    api_calls_this_month: int
    created_at: datetime
    updated_at: datetime

class TenantUserResponse(BaseModel):
    """Tenant user response model"""
    id: str
    tenant_id: str
    user_id: str
    tenant_roles: List[str]
    is_tenant_admin: bool
    is_active: bool
    last_login: Optional[datetime] = None
    login_count: int
    created_at: datetime

class AdminDashboardResponse(BaseModel):
    """Admin dashboard summary"""
    total_organizations: int
    total_tenants: int
    total_users: int
    active_subscriptions: int
    monthly_revenue: float
    top_organizations: List[Dict[str, Any]]
    recent_activity: List[Dict[str, Any]]
    system_metrics: Dict[str, Any]

class WhiteLabelConfig(BaseModel):
    """White-label configuration"""
    enabled: bool = False
    logo_url: Optional[str] = None
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None
    custom_css: Optional[str] = None
    custom_domain: Optional[str] = None
    brand_name: Optional[str] = None

# Organization Management Endpoints

@router.get("/dashboard", response_model=AdminDashboardResponse)
async def get_admin_dashboard(
    current_user: UserInDB = Depends(require_permission("admin:read"))
):
    """Get enterprise admin dashboard summary"""
    try:
        # Get organization stats
        orgs, total_orgs = await organization_manager.list_organizations(limit=1000)
        tenants, total_tenants = await organization_manager.list_tenants(limit=1000)
        
        # Count total users across all tenants
        total_users = sum(len(users) for users in organization_manager.tenant_users.values())
        
        # Get subscription stats
        active_subs = [s for s in subscription_manager.subscriptions.values() 
                      if s.status in ['active', 'trial']]
        
        # Calculate monthly revenue
        monthly_revenue = sum(float(s.total_amount) for s in active_subs)
        
        # Get top organizations by user count
        org_stats = []
        for org in orgs[:10]:  # Top 10
            org_tenants = [t for t in tenants if t.organization_id == org.id]
            user_count = sum(len(organization_manager.tenant_users.get(t.id, [])) 
                           for t in org_tenants)
            org_stats.append({
                "id": org.id,
                "name": org.name,
                "tenant_count": len(org_tenants),
                "user_count": user_count,
                "status": org.status
            })
        
        # Sort by user count
        top_organizations = sorted(org_stats, key=lambda x: x["user_count"], reverse=True)[:5]
        
        # Recent activity (last 10 organizations created)
        recent_orgs = sorted(orgs, key=lambda x: x.created_at, reverse=True)[:10]
        recent_activity = [{
            "type": "organization_created",
            "description": f"Organization '{org.name}' created",
            "timestamp": org.created_at,
            "organization_id": org.id
        } for org in recent_orgs]
        
        return AdminDashboardResponse(
            total_organizations=total_orgs,
            total_tenants=total_tenants,
            total_users=total_users,
            active_subscriptions=len(active_subs),
            monthly_revenue=monthly_revenue,
            top_organizations=top_organizations,
            recent_activity=recent_activity,
            system_metrics={
                "avg_users_per_org": total_users / max(total_orgs, 1),
                "avg_tenants_per_org": total_tenants / max(total_orgs, 1),
                "subscription_rate": len(active_subs) / max(total_tenants, 1) * 100
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get admin dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get dashboard data")

@router.get("/organizations", response_model=List[OrganizationResponse])
async def list_organizations(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: UserInDB = Depends(require_permission("admin:read"))
):
    """List all organizations"""
    try:
        orgs, total = await organization_manager.list_organizations(limit, offset)
        
        response_orgs = []
        for org in orgs:
            # Count tenants and users for this organization
            org_tenants = [t for t in organization_manager.tenants.values() 
                          if t.organization_id == org.id]
            tenant_count = len(org_tenants)
            user_count = sum(len(organization_manager.tenant_users.get(t.id, [])) 
                           for t in org_tenants)
            
            response_orgs.append(OrganizationResponse(
                id=org.id,
                name=org.name,
                slug=org.slug,
                email=org.email,
                organization_type=org.organization_type,
                status=org.status,
                tenant_count=tenant_count,
                user_count=user_count,
                created_at=org.created_at,
                updated_at=org.updated_at,
                branding=org.branding,
                white_label_enabled=org.white_label_enabled
            ))
        
        return response_orgs
        
    except Exception as e:
        logger.error(f"Failed to list organizations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list organizations")

@router.post("/organizations", response_model=OrganizationResponse)
async def create_organization(
    org_data: OrganizationCreate,
    current_user: UserInDB = Depends(require_permission("admin:write"))
):
    """Create new organization"""
    try:
        org = await organization_manager.create_organization(org_data, current_user)
        
        return OrganizationResponse(
            id=org.id,
            name=org.name,
            slug=org.slug,
            email=org.email,
            organization_type=org.organization_type,
            status=org.status,
            tenant_count=1,  # Default tenant created
            user_count=1,    # Creator added
            created_at=org.created_at,
            updated_at=org.updated_at,
            branding=org.branding,
            white_label_enabled=org.white_label_enabled
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create organization: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create organization")

@router.get("/organizations/{org_id}", response_model=OrganizationResponse)
async def get_organization(
    org_id: str,
    current_user: UserInDB = Depends(require_permission("admin:read"))
):
    """Get organization details"""
    try:
        org = await organization_manager.get_organization(org_id)
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        # Count tenants and users
        org_tenants = [t for t in organization_manager.tenants.values() 
                      if t.organization_id == org.id]
        tenant_count = len(org_tenants)
        user_count = sum(len(organization_manager.tenant_users.get(t.id, [])) 
                       for t in org_tenants)
        
        return OrganizationResponse(
            id=org.id,
            name=org.name,
            slug=org.slug,
            email=org.email,
            organization_type=org.organization_type,
            status=org.status,
            tenant_count=tenant_count,
            user_count=user_count,
            created_at=org.created_at,
            updated_at=org.updated_at,
            branding=org.branding,
            white_label_enabled=org.white_label_enabled
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get organization: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get organization")

@router.put("/organizations/{org_id}/white-label")
async def configure_white_label(
    org_id: str,
    config: WhiteLabelConfig,
    current_user: UserInDB = Depends(require_permission("admin:write"))
):
    """Configure white-label settings for organization"""
    try:
        org = await organization_manager.get_organization(org_id)
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        # Update organization branding and white-label settings
        branding = {}
        if config.logo_url:
            branding["logo_url"] = config.logo_url
        if config.primary_color:
            branding["primary_color"] = config.primary_color
        if config.secondary_color:
            branding["secondary_color"] = config.secondary_color
        if config.custom_css:
            branding["custom_css"] = config.custom_css
        if config.brand_name:
            branding["name"] = config.brand_name
        
        updates = {
            "white_label_enabled": config.enabled,
            "branding": branding,
            "custom_domain": config.custom_domain
        }
        
        updated_org = await organization_manager.update_organization(org_id, updates)
        
        return {"message": "White-label configuration updated", "organization_id": org_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to configure white-label: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to configure white-label")

# Tenant Management Endpoints

@router.get("/tenants", response_model=List[TenantResponse])
async def list_tenants(
    org_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: UserInDB = Depends(require_permission("admin:read"))
):
    """List tenants with optional organization filter"""
    try:
        tenants, total = await organization_manager.list_tenants(org_id, limit, offset)
        
        response_tenants = []
        for tenant in tenants:
            response_tenants.append(TenantResponse(
                id=tenant.id,
                organization_id=tenant.organization_id,
                name=tenant.name,
                slug=tenant.slug,
                subdomain=tenant.subdomain,
                status=tenant.status,
                tier=tenant.tier,
                features=tenant.features,
                current_users=tenant.current_users,
                storage_used_gb=tenant.storage_used_gb,
                api_calls_this_month=tenant.api_calls_this_month,
                created_at=tenant.created_at,
                updated_at=tenant.updated_at
            ))
        
        return response_tenants
        
    except Exception as e:
        logger.error(f"Failed to list tenants: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list tenants")

@router.put("/tenants/{tenant_id}/status")
async def update_tenant_status(
    tenant_id: str,
    status: TenantStatus,
    current_user: UserInDB = Depends(require_permission("admin:write"))
):
    """Update tenant status"""
    try:
        updates = TenantUpdate(status=status)
        tenant = await organization_manager.update_tenant(tenant_id, updates)
        
        return {"message": "Tenant status updated", "tenant_id": tenant_id, "new_status": status}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update tenant status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update tenant status")

# Billing Management Endpoints

@router.get("/billing/subscriptions")
async def list_all_subscriptions(
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    current_user: UserInDB = Depends(require_permission("admin:read"))
):
    """List all subscriptions across all tenants"""
    try:
        subscriptions = list(subscription_manager.subscriptions.values())
        
        if status:
            subscriptions = [s for s in subscriptions if s.status == status]
        
        # Sort by created date and limit
        subscriptions.sort(key=lambda x: x.created_at, reverse=True)
        subscriptions = subscriptions[:limit]
        
        # Enhance with organization info
        enhanced_subs = []
        for sub in subscriptions:
            tenant = await organization_manager.get_tenant(sub.tenant_id)
            org = await organization_manager.get_organization(sub.organization_id) if tenant else None
            
            enhanced_subs.append({
                "id": sub.id,
                "tenant_id": sub.tenant_id,
                "tenant_name": tenant.name if tenant else "Unknown",
                "organization_id": sub.organization_id,
                "organization_name": org.name if org else "Unknown",
                "plan_id": sub.plan_id,
                "status": sub.status,
                "total_amount": float(sub.total_amount),
                "currency": sub.currency,
                "current_period_start": sub.current_period_start,
                "current_period_end": sub.current_period_end,
                "created_at": sub.created_at
            })
        
        return enhanced_subs
        
    except Exception as e:
        logger.error(f"Failed to list subscriptions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list subscriptions")

@router.get("/billing/revenue")
async def get_revenue_analytics(
    period: str = Query("month"),  # month, quarter, year
    current_user: UserInDB = Depends(require_permission("admin:read"))
):
    """Get revenue analytics"""
    try:
        now = datetime.now(timezone.utc)
        
        # Calculate period boundaries
        if period == "month":
            start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif period == "quarter":
            quarter_start_month = ((now.month - 1) // 3) * 3 + 1
            start_date = now.replace(month=quarter_start_month, day=1, hour=0, minute=0, second=0, microsecond=0)
        elif period == "year":
            start_date = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Get active subscriptions
        active_subs = [s for s in subscription_manager.subscriptions.values() 
                      if s.status in ['active', 'trial'] and s.created_at >= start_date]
        
        # Calculate metrics
        total_revenue = sum(float(s.total_amount) for s in active_subs)
        subscription_count = len(active_subs)
        
        # Revenue by plan
        revenue_by_plan = {}
        for sub in active_subs:
            plan_id = sub.plan_id
            if plan_id not in revenue_by_plan:
                revenue_by_plan[plan_id] = {"revenue": 0, "count": 0}
            revenue_by_plan[plan_id]["revenue"] += float(sub.total_amount)
            revenue_by_plan[plan_id]["count"] += 1
        
        return {
            "period": period,
            "start_date": start_date,
            "end_date": now,
            "total_revenue": total_revenue,
            "subscription_count": subscription_count,
            "average_revenue_per_subscription": total_revenue / max(subscription_count, 1),
            "revenue_by_plan": revenue_by_plan
        }
        
    except Exception as e:
        logger.error(f"Failed to get revenue analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get revenue analytics")

# User Management Endpoints

@router.get("/users")
async def list_all_users(
    tenant_id: Optional[str] = Query(None),
    org_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    current_user: UserInDB = Depends(require_permission("admin:read"))
):
    """List users across tenants and organizations"""
    try:
        all_users = []
        
        if tenant_id:
            # Get users for specific tenant
            tenant_users = await organization_manager.get_tenant_users(tenant_id)
            for tenant_user in tenant_users:
                # Would need to fetch full user details from auth service
                all_users.append({
                    "tenant_user_id": tenant_user.id,
                    "user_id": tenant_user.user_id,
                    "tenant_id": tenant_user.tenant_id,
                    "organization_id": tenant_user.organization_id,
                    "tenant_roles": tenant_user.tenant_roles,
                    "is_tenant_admin": tenant_user.is_tenant_admin,
                    "is_active": tenant_user.is_active,
                    "last_login": tenant_user.last_login,
                    "login_count": tenant_user.login_count,
                    "created_at": tenant_user.created_at
                })
        elif org_id:
            # Get users for all tenants in organization
            org_tenants = [t for t in organization_manager.tenants.values() 
                          if t.organization_id == org_id]
            for tenant in org_tenants:
                tenant_users = await organization_manager.get_tenant_users(tenant.id)
                for tenant_user in tenant_users:
                    all_users.append({
                        "tenant_user_id": tenant_user.id,
                        "user_id": tenant_user.user_id,
                        "tenant_id": tenant_user.tenant_id,
                        "organization_id": tenant_user.organization_id,
                        "tenant_roles": tenant_user.tenant_roles,
                        "is_tenant_admin": tenant_user.is_tenant_admin,
                        "is_active": tenant_user.is_active,
                        "last_login": tenant_user.last_login,
                        "login_count": tenant_user.login_count,
                        "created_at": tenant_user.created_at
                    })
        else:
            # Get all users across all tenants
            for tenant_id, tenant_users in organization_manager.tenant_users.items():
                for tenant_user in tenant_users:
                    all_users.append({
                        "tenant_user_id": tenant_user.id,
                        "user_id": tenant_user.user_id,
                        "tenant_id": tenant_user.tenant_id,
                        "organization_id": tenant_user.organization_id,
                        "tenant_roles": tenant_user.tenant_roles,
                        "is_tenant_admin": tenant_user.is_tenant_admin,
                        "is_active": tenant_user.is_active,
                        "last_login": tenant_user.last_login,
                        "login_count": tenant_user.login_count,
                        "created_at": tenant_user.created_at
                    })
        
        # Sort by created date and limit
        all_users.sort(key=lambda x: x["created_at"], reverse=True)
        return all_users[:limit]
        
    except Exception as e:
        logger.error(f"Failed to list users: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list users")

# System Maintenance Endpoints

@router.post("/maintenance/cleanup-expired")
async def cleanup_expired_data(
    background_tasks: BackgroundTasks,
    current_user: UserInDB = Depends(require_permission("admin:write"))
):
    """Cleanup expired invitations, trials, etc."""
    try:
        def cleanup_task():
            logger.info("Starting system cleanup task")
            
            # Cleanup expired invitations
            expired_count = 0
            current_time = datetime.now(timezone.utc)
            
            for inv_id, invitation in list(organization_manager.invitations.items()):
                if (invitation.status == "pending" and 
                    current_time > invitation.expires_at):
                    invitation.status = "expired"
                    organization_manager.invitations[inv_id] = invitation
                    expired_count += 1
            
            logger.info(f"Marked {expired_count} invitations as expired")
            
            # Could add more cleanup tasks here:
            # - Expired trials
            # - Old usage records
            # - Temporary files
            # - Audit logs older than retention period
        
        background_tasks.add_task(cleanup_task)
        
        return {"message": "Cleanup task scheduled"}
        
    except Exception as e:
        logger.error(f"Failed to schedule cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to schedule cleanup")

@router.get("/health")
async def admin_health_check(
    current_user: UserInDB = Depends(require_permission("admin:read"))
):
    """Get system health status for admin monitoring"""
    try:
        # Check various system components
        health_status = {
            "organization_manager": "healthy",
            "subscription_manager": "healthy",
            "total_organizations": len(organization_manager.organizations),
            "total_tenants": len(organization_manager.tenants),
            "total_subscriptions": len(subscription_manager.subscriptions),
            "active_subscriptions": len([
                s for s in subscription_manager.subscriptions.values()
                if s.status in ['active', 'trial']
            ]),
            "timestamp": datetime.now(timezone.utc)
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc)
        }
