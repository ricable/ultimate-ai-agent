"""
Workflow Marketplace

Advanced marketplace system for sharing, discovering, and installing workflow templates
with ratings, categories, certification, and monetization support.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from ..database.session import get_db
from ..monitoring.metrics.prometheus_metrics import workflow_metrics
from .models import (
    Workflow, WorkflowTemplate, WorkflowStatus,
    ExecutionStatus, WorkflowExecution
)
import uuid
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class WorkflowMarketplace:
    """Advanced workflow marketplace with comprehensive template management."""
    
    def __init__(self):
        self.featured_templates: List[str] = []
        self.categories = self._init_categories()
        
    def _init_categories(self) -> Dict[str, List[str]]:
        """Initialize workflow categories and subcategories."""
        return {
            "Business Process": [
                "Customer Support",
                "Sales Automation",
                "Marketing Campaigns",
                "HR Processes",
                "Financial Operations",
                "Compliance & Audit"
            ],
            "Data Processing": [
                "Data Integration",
                "ETL Pipelines",
                "Data Validation",
                "Report Generation",
                "Analytics",
                "Data Cleanup"
            ],
            "Development & DevOps": [
                "CI/CD Pipelines",
                "Code Reviews",
                "Testing Automation",
                "Deployment",
                "Monitoring & Alerts",
                "Infrastructure"
            ],
            "Content & Media": [
                "Content Creation",
                "Media Processing",
                "Publishing",
                "Social Media",
                "Documentation",
                "Translation"
            ],
            "Communication": [
                "Notifications",
                "Email Automation",
                "Chat Integration",
                "Meeting Management",
                "Customer Communication",
                "Internal Communication"
            ],
            "Integration": [
                "API Integration",
                "Database Sync",
                "File Processing",
                "Third-party Services",
                "Legacy Systems",
                "Cloud Services"
            ],
            "AI & Machine Learning": [
                "Model Training",
                "Data Preprocessing",
                "Prediction Workflows",
                "NLP Processing",
                "Computer Vision",
                "MLOps"
            ],
            "Utilities": [
                "File Management",
                "System Administration",
                "Backup & Recovery",
                "Monitoring",
                "Maintenance",
                "Troubleshooting"
            ]
        }
    
    async def publish_template(
        self, 
        workflow_id: str,
        template_name: str,
        description: str,
        category: str,
        subcategory: str = None,
        tags: List[str] = None,
        keywords: List[str] = None,
        documentation: str = "",
        price: int = 0,
        created_by: str = None,
        organization_id: str = None
    ) -> str:
        """Publish a workflow as a template to the marketplace."""
        
        db = next(get_db())
        try:
            # Get workflow
            workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Validate category
            if category not in self.categories:
                raise ValueError(f"Invalid category: {category}")
            
            if subcategory and subcategory not in self.categories[category]:
                raise ValueError(f"Invalid subcategory: {subcategory}")
            
            # Create template
            template = WorkflowTemplate(
                name=template_name,
                description=description,
                category=category,
                subcategory=subcategory,
                definition=workflow.definition,
                variables=workflow.variables,
                documentation=documentation,
                created_by=created_by,
                organization_id=organization_id,
                price=price,
                tags=tags or [],
                keywords=keywords or []
            )
            
            db.add(template)
            db.commit()
            
            # Update metrics
            workflow_metrics.templates_published_total.labels(
                category=category,
                subcategory=subcategory or "none"
            ).inc()
            
            logger.info(f"Published workflow template {template.id}: {template_name}")
            return template.id
            
        finally:
            db.close()
    
    async def search_templates(
        self,
        query: str = None,
        category: str = None,
        subcategory: str = None,
        tags: List[str] = None,
        min_rating: float = None,
        max_price: int = None,
        is_free_only: bool = False,
        is_featured: bool = False,
        is_verified: bool = False,
        sort_by: str = "relevance",  # relevance, rating, downloads, date, price
        limit: int = 20,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Search workflow templates with advanced filtering and sorting."""
        
        db = next(get_db())
        try:
            # Build query
            query_obj = db.query(WorkflowTemplate)
            
            # Apply filters
            if query:
                query_obj = query_obj.filter(
                    func.lower(WorkflowTemplate.name).contains(query.lower()) |
                    func.lower(WorkflowTemplate.description).contains(query.lower())
                )
            
            if category:
                query_obj = query_obj.filter(WorkflowTemplate.category == category)
            
            if subcategory:
                query_obj = query_obj.filter(WorkflowTemplate.subcategory == subcategory)
            
            if tags:
                # Filter by tags (PostgreSQL JSON contains)
                for tag in tags:
                    query_obj = query_obj.filter(WorkflowTemplate.tags.contains([tag]))
            
            if min_rating:
                query_obj = query_obj.filter(WorkflowTemplate.rating_average >= min_rating * 100)
            
            if max_price is not None:
                query_obj = query_obj.filter(WorkflowTemplate.price <= max_price)
            
            if is_free_only:
                query_obj = query_obj.filter(WorkflowTemplate.price == 0)
            
            if is_featured:
                query_obj = query_obj.filter(WorkflowTemplate.is_featured == True)
            
            if is_verified:
                query_obj = query_obj.filter(WorkflowTemplate.is_verified == True)
            
            # Apply sorting
            if sort_by == "rating":
                query_obj = query_obj.order_by(desc(WorkflowTemplate.rating_average))
            elif sort_by == "downloads":
                query_obj = query_obj.order_by(desc(WorkflowTemplate.download_count))
            elif sort_by == "date":
                query_obj = query_obj.order_by(desc(WorkflowTemplate.created_at))
            elif sort_by == "price":
                query_obj = query_obj.order_by(WorkflowTemplate.price)
            elif sort_by == "name":
                query_obj = query_obj.order_by(WorkflowTemplate.name)
            else:  # relevance (default)
                # Simple relevance scoring based on downloads and rating
                query_obj = query_obj.order_by(
                    desc(WorkflowTemplate.download_count + WorkflowTemplate.rating_average)
                )
            
            # Get total count
            total_count = query_obj.count()
            
            # Apply pagination
            templates = query_obj.offset(offset).limit(limit).all()
            
            return {
                "templates": [template.to_dict() for template in templates],
                "total_count": total_count,
                "page_count": (total_count + limit - 1) // limit,
                "current_page": offset // limit + 1,
                "has_more": offset + limit < total_count
            }
            
        finally:
            db.close()
    
    async def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed template information."""
        
        db = next(get_db())
        try:
            template = db.query(WorkflowTemplate).filter(WorkflowTemplate.id == template_id).first()
            if not template:
                return None
            
            template_dict = template.to_dict()
            
            # Add additional information
            template_dict["statistics"] = await self._get_template_statistics(template_id, db)
            template_dict["similar_templates"] = await self._get_similar_templates(template, db)
            
            return template_dict
            
        finally:
            db.close()
    
    async def _get_template_statistics(self, template_id: str, db: Session) -> Dict[str, Any]:
        """Get template usage statistics."""
        
        # Get workflows created from this template
        workflows_from_template = db.query(Workflow).filter(
            Workflow.definition.contains({"template_id": template_id})
        ).all()
        
        total_executions = sum(w.execution_count for w in workflows_from_template)
        total_success = sum(w.success_count for w in workflows_from_template)
        avg_success_rate = (total_success / total_executions * 100) if total_executions > 0 else 0
        
        return {
            "workflows_created": len(workflows_from_template),
            "total_executions": total_executions,
            "success_rate": round(avg_success_rate, 1),
            "avg_duration_ms": sum(w.avg_duration_ms for w in workflows_from_template) // len(workflows_from_template) if workflows_from_template else 0
        }
    
    async def _get_similar_templates(self, template: WorkflowTemplate, db: Session, limit: int = 5) -> List[Dict[str, Any]]:
        """Get similar templates based on category, tags, and keywords."""
        
        similar_query = db.query(WorkflowTemplate).filter(
            WorkflowTemplate.id != template.id
        )
        
        # Same category gets higher weight
        if template.category:
            similar_query = similar_query.filter(WorkflowTemplate.category == template.category)
        
        # Order by rating and downloads
        similar_templates = similar_query.order_by(
            desc(WorkflowTemplate.rating_average),
            desc(WorkflowTemplate.download_count)
        ).limit(limit).all()
        
        return [
            {
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "rating": t.rating_average / 100.0 if t.rating_average else 0,
                "downloads": t.download_count,
                "price": t.price
            }
            for t in similar_templates
        ]
    
    async def install_template(
        self, 
        template_id: str, 
        workflow_name: str,
        user_id: str,
        organization_id: str = None,
        customizations: Dict[str, Any] = None
    ) -> str:
        """Install a template as a new workflow."""
        
        db = next(get_db())
        try:
            # Get template
            template = db.query(WorkflowTemplate).filter(WorkflowTemplate.id == template_id).first()
            if not template:
                raise ValueError(f"Template {template_id} not found")
            
            # Create workflow from template
            workflow_definition = template.definition.copy()
            workflow_variables = template.variables.copy() if template.variables else {}
            
            # Apply customizations
            if customizations:
                if "variables" in customizations:
                    workflow_variables.update(customizations["variables"])
                
                if "definition_updates" in customizations:
                    workflow_definition.update(customizations["definition_updates"])
            
            # Add template metadata
            workflow_definition["template_id"] = template_id
            workflow_definition["template_version"] = template.version
            workflow_definition["installed_at"] = datetime.utcnow().isoformat()
            
            # Create workflow
            workflow = Workflow(
                name=workflow_name,
                description=f"Created from template: {template.name}",
                definition=workflow_definition,
                variables=workflow_variables,
                created_by=user_id,
                organization_id=organization_id,
                status=WorkflowStatus.DRAFT
            )
            
            db.add(workflow)
            db.commit()
            
            # Update template download count
            template.download_count += 1
            db.commit()
            
            # Update metrics
            workflow_metrics.templates_installed_total.labels(
                template_id=template_id,
                category=template.category
            ).inc()
            
            logger.info(f"Installed template {template_id} as workflow {workflow.id}")
            return workflow.id
            
        finally:
            db.close()
    
    async def rate_template(self, template_id: str, user_id: str, rating: int, review: str = "") -> bool:
        """Rate and review a template."""
        
        if rating < 1 or rating > 5:
            raise ValueError("Rating must be between 1 and 5")
        
        db = next(get_db())
        try:
            # Get template
            template = db.query(WorkflowTemplate).filter(WorkflowTemplate.id == template_id).first()
            if not template:
                return False
            
            # For simplicity, we'll update the template's rating directly
            # In a full implementation, you'd have a separate ratings table
            
            # Calculate new average rating
            total_rating_points = template.rating_average * template.rating_count
            new_total_points = total_rating_points + (rating * 100)  # Store as integer
            new_count = template.rating_count + 1
            new_average = new_total_points // new_count
            
            template.rating_average = new_average
            template.rating_count = new_count
            
            db.commit()
            
            logger.info(f"Template {template_id} rated {rating}/5 by user {user_id}")
            return True
            
        finally:
            db.close()
    
    async def get_featured_templates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get featured templates."""
        
        db = next(get_db())
        try:
            templates = db.query(WorkflowTemplate).filter(
                WorkflowTemplate.is_featured == True
            ).order_by(
                desc(WorkflowTemplate.rating_average),
                desc(WorkflowTemplate.download_count)
            ).limit(limit).all()
            
            return [template.to_dict() for template in templates]
            
        finally:
            db.close()
    
    async def get_popular_templates(self, limit: int = 10, days: int = 30) -> List[Dict[str, Any]]:
        """Get popular templates based on recent downloads."""
        
        db = next(get_db())
        try:
            # For simplicity, using total downloads
            # In production, you'd track downloads by date
            templates = db.query(WorkflowTemplate).order_by(
                desc(WorkflowTemplate.download_count)
            ).limit(limit).all()
            
            return [template.to_dict() for template in templates]
            
        finally:
            db.close()
    
    async def get_categories(self) -> Dict[str, List[str]]:
        """Get all categories and subcategories."""
        return self.categories
    
    async def get_category_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for each category."""
        
        db = next(get_db())
        try:
            stats = {}
            
            for category in self.categories:
                category_templates = db.query(WorkflowTemplate).filter(
                    WorkflowTemplate.category == category
                ).all()
                
                total_downloads = sum(t.download_count for t in category_templates)
                avg_rating = sum(t.rating_average for t in category_templates if t.rating_average) / len([t for t in category_templates if t.rating_average]) if category_templates else 0
                
                stats[category] = {
                    "template_count": len(category_templates),
                    "total_downloads": total_downloads,
                    "average_rating": round(avg_rating / 100.0, 2) if avg_rating else 0,
                    "featured_count": len([t for t in category_templates if t.is_featured]),
                    "verified_count": len([t for t in category_templates if t.is_verified])
                }
            
            return stats
            
        finally:
            db.close()
    
    async def feature_template(self, template_id: str, featured: bool = True) -> bool:
        """Feature or unfeature a template."""
        
        db = next(get_db())
        try:
            template = db.query(WorkflowTemplate).filter(WorkflowTemplate.id == template_id).first()
            if not template:
                return False
            
            template.is_featured = featured
            db.commit()
            
            action = "featured" if featured else "unfeatured"
            logger.info(f"Template {template_id} {action}")
            return True
            
        finally:
            db.close()
    
    async def verify_template(self, template_id: str, verified: bool = True) -> bool:
        """Verify or unverify a template."""
        
        db = next(get_db())
        try:
            template = db.query(WorkflowTemplate).filter(WorkflowTemplate.id == template_id).first()
            if not template:
                return False
            
            template.is_verified = verified
            db.commit()
            
            action = "verified" if verified else "unverified"
            logger.info(f"Template {template_id} {action}")
            return True
            
        finally:
            db.close()
    
    async def update_template(
        self, 
        template_id: str, 
        updates: Dict[str, Any],
        created_by: str = None
    ) -> bool:
        """Update a template."""
        
        db = next(get_db())
        try:
            template = db.query(WorkflowTemplate).filter(WorkflowTemplate.id == template_id).first()
            if not template:
                return False
            
            # Check permissions (simplified)
            if created_by and template.created_by != created_by:
                raise ValueError("Not authorized to update this template")
            
            # Apply updates
            allowed_fields = ['name', 'description', 'documentation', 'tags', 'keywords', 'price']
            for field, value in updates.items():
                if field in allowed_fields and hasattr(template, field):
                    setattr(template, field, value)
            
            template.updated_at = datetime.utcnow()
            db.commit()
            
            logger.info(f"Updated template {template_id}")
            return True
            
        finally:
            db.close()
    
    async def delete_template(self, template_id: str, created_by: str = None) -> bool:
        """Delete a template."""
        
        db = next(get_db())
        try:
            template = db.query(WorkflowTemplate).filter(WorkflowTemplate.id == template_id).first()
            if not template:
                return False
            
            # Check permissions (simplified)
            if created_by and template.created_by != created_by:
                raise ValueError("Not authorized to delete this template")
            
            db.delete(template)
            db.commit()
            
            logger.info(f"Deleted template {template_id}")
            return True
            
        finally:
            db.close()
    
    async def get_user_templates(self, user_id: str) -> List[Dict[str, Any]]:
        """Get templates created by a user."""
        
        db = next(get_db())
        try:
            templates = db.query(WorkflowTemplate).filter(
                WorkflowTemplate.created_by == user_id
            ).order_by(desc(WorkflowTemplate.created_at)).all()
            
            return [template.to_dict() for template in templates]
            
        finally:
            db.close()
    
    async def export_template(self, template_id: str) -> Dict[str, Any]:
        """Export a template for sharing or backup."""
        
        db = next(get_db())
        try:
            template = db.query(WorkflowTemplate).filter(WorkflowTemplate.id == template_id).first()
            if not template:
                raise ValueError(f"Template {template_id} not found")
            
            export_data = {
                "template_info": {
                    "name": template.name,
                    "description": template.description,
                    "category": template.category,
                    "subcategory": template.subcategory,
                    "version": template.version,
                    "tags": template.tags,
                    "keywords": template.keywords,
                    "documentation": template.documentation
                },
                "workflow_definition": template.definition,
                "variables": template.variables,
                "export_timestamp": datetime.utcnow().isoformat(),
                "export_version": "1.0"
            }
            
            return export_data
            
        finally:
            db.close()
    
    async def import_template(
        self, 
        template_data: Dict[str, Any], 
        user_id: str,
        organization_id: str = None
    ) -> str:
        """Import a template from exported data."""
        
        db = next(get_db())
        try:
            template_info = template_data["template_info"]
            
            # Create new template
            template = WorkflowTemplate(
                name=template_info["name"],
                description=template_info["description"],
                category=template_info["category"],
                subcategory=template_info.get("subcategory"),
                definition=template_data["workflow_definition"],
                variables=template_data.get("variables", {}),
                documentation=template_info.get("documentation", ""),
                tags=template_info.get("tags", []),
                keywords=template_info.get("keywords", []),
                created_by=user_id,
                organization_id=organization_id
            )
            
            db.add(template)
            db.commit()
            
            logger.info(f"Imported template {template.id}: {template.name}")
            return template.id
            
        finally:
            db.close()
    
    def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get overall marketplace statistics."""
        
        db = next(get_db())
        try:
            total_templates = db.query(WorkflowTemplate).count()
            featured_templates = db.query(WorkflowTemplate).filter(WorkflowTemplate.is_featured == True).count()
            verified_templates = db.query(WorkflowTemplate).filter(WorkflowTemplate.is_verified == True).count()
            free_templates = db.query(WorkflowTemplate).filter(WorkflowTemplate.price == 0).count()
            
            total_downloads = db.query(func.sum(WorkflowTemplate.download_count)).scalar() or 0
            avg_rating = db.query(func.avg(WorkflowTemplate.rating_average)).scalar() or 0
            
            return {
                "total_templates": total_templates,
                "featured_templates": featured_templates,
                "verified_templates": verified_templates,
                "free_templates": free_templates,
                "paid_templates": total_templates - free_templates,
                "total_downloads": total_downloads,
                "average_rating": round(avg_rating / 100.0, 2) if avg_rating else 0,
                "categories": len(self.categories),
                "subcategories": sum(len(subs) for subs in self.categories.values())
            }
            
        finally:
            db.close()