# File: backend/sales/pipeline_automation.py
"""
Sales Pipeline Automation System for UAP Platform

Provides comprehensive sales pipeline management, lead tracking,
opportunity management, and automated sales processes.
"""

import asyncio
import json
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import uuid
from threading import Lock

from ..analytics.usage_analytics import usage_analytics
from ..analytics.predictive_analytics import predictive_analytics

class LeadSource(Enum):
    """Lead source types"""
    WEBSITE = "website"
    REFERRAL = "referral"
    SOCIAL_MEDIA = "social_media"
    EMAIL_CAMPAIGN = "email_campaign"
    CONTENT_MARKETING = "content_marketing"
    PAID_ADVERTISING = "paid_advertising"
    EVENTS = "events"
    PARTNERSHIPS = "partnerships"
    COLD_OUTREACH = "cold_outreach"
    INBOUND = "inbound"
    DIRECT = "direct"

class LeadStatus(Enum):
    """Lead status in pipeline"""
    NEW = "new"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    PROPOSAL = "proposal"
    NEGOTIATION = "negotiation"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"
    NURTURING = "nurturing"
    UNQUALIFIED = "unqualified"

class CompanySize(Enum):
    """Company size categories"""
    STARTUP = "startup"          # 1-10 employees
    SMALL_BUSINESS = "small"     # 11-50 employees
    MID_MARKET = "mid_market"    # 51-500 employees
    ENTERPRISE = "enterprise"    # 500+ employees

class Priority(Enum):
    """Lead/opportunity priority"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    URGENT = "urgent"

@dataclass
class Lead:
    """Lead data structure"""
    lead_id: str
    email: str
    first_name: str
    last_name: str
    company: str
    title: str
    phone: Optional[str] = None
    source: LeadSource = LeadSource.DIRECT
    status: LeadStatus = LeadStatus.NEW
    lead_score: float = 0.0
    company_size: Optional[CompanySize] = None
    industry: Optional[str] = None
    budget: Optional[float] = None
    timeline: Optional[str] = None
    pain_points: List[str] = None
    interests: List[str] = None
    last_contact: Optional[datetime] = None
    next_follow_up: Optional[datetime] = None
    assigned_to: Optional[str] = None
    tags: List[str] = None
    notes: str = ""
    created_at: datetime = None
    updated_at: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.pain_points is None:
            self.pain_points = []
        if self.interests is None:
            self.interests = []
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lead_id": self.lead_id,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": f"{self.first_name} {self.last_name}",
            "company": self.company,
            "title": self.title,
            "phone": self.phone,
            "source": self.source.value,
            "status": self.status.value,
            "lead_score": self.lead_score,
            "company_size": self.company_size.value if self.company_size else None,
            "industry": self.industry,
            "budget": self.budget,
            "timeline": self.timeline,
            "pain_points": self.pain_points,
            "interests": self.interests,
            "last_contact": self.last_contact.isoformat() if self.last_contact else None,
            "next_follow_up": self.next_follow_up.isoformat() if self.next_follow_up else None,
            "assigned_to": self.assigned_to,
            "tags": self.tags,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }

@dataclass
class Opportunity:
    """Sales opportunity data structure"""
    opportunity_id: str
    lead_id: str
    name: str
    value: float
    probability: float  # 0-100
    stage: LeadStatus
    close_date: datetime
    assigned_to: str
    products_interested: List[str] = None
    competitors: List[str] = None
    decision_makers: List[str] = None
    next_steps: str = ""
    notes: str = ""
    priority: Priority = Priority.MEDIUM
    created_at: datetime = None
    updated_at: datetime = None
    closed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.products_interested is None:
            self.products_interested = []
        if self.competitors is None:
            self.competitors = []
        if self.decision_makers is None:
            self.decision_makers = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "opportunity_id": self.opportunity_id,
            "lead_id": self.lead_id,
            "name": self.name,
            "value": self.value,
            "probability": self.probability,
            "stage": self.stage.value,
            "close_date": self.close_date.isoformat(),
            "assigned_to": self.assigned_to,
            "products_interested": self.products_interested,
            "competitors": self.competitors,
            "decision_makers": self.decision_makers,
            "next_steps": self.next_steps,
            "notes": self.notes,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "metadata": self.metadata,
            "expected_value": self.value * (self.probability / 100),
            "days_to_close": (self.close_date - datetime.utcnow()).days,
            "age_days": (datetime.utcnow() - self.created_at).days
        }

@dataclass
class SalesActivity:
    """Sales activity tracking"""
    activity_id: str
    lead_id: Optional[str]
    opportunity_id: Optional[str]
    activity_type: str  # call, email, meeting, demo, proposal, etc.
    subject: str
    description: str
    assigned_to: str
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    outcome: Optional[str] = None
    next_activity: Optional[str] = None
    created_at: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "activity_id": self.activity_id,
            "lead_id": self.lead_id,
            "opportunity_id": self.opportunity_id,
            "activity_type": self.activity_type,
            "subject": self.subject,
            "description": self.description,
            "assigned_to": self.assigned_to,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "outcome": self.outcome,
            "next_activity": self.next_activity,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "is_completed": self.completed_at is not None,
            "is_overdue": self.due_date and self.due_date < datetime.utcnow() and not self.completed_at
        }

@dataclass
class PipelineStage:
    """Pipeline stage configuration"""
    stage_id: str
    name: str
    order: int
    conversion_rate: float  # Historical conversion rate to next stage
    avg_duration_days: int  # Average time in this stage
    required_activities: List[str] = None
    automated_actions: List[str] = None
    
    def __post_init__(self):
        if self.required_activities is None:
            self.required_activities = []
        if self.automated_actions is None:
            self.automated_actions = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "name": self.name,
            "order": self.order,
            "conversion_rate": self.conversion_rate,
            "avg_duration_days": self.avg_duration_days,
            "required_activities": self.required_activities,
            "automated_actions": self.automated_actions
        }

class SalesPipelineAutomation:
    """Main sales pipeline automation system"""
    
    def __init__(self):
        self.leads: Dict[str, Lead] = {}
        self.opportunities: Dict[str, Opportunity] = {}
        self.activities: Dict[str, SalesActivity] = {}
        self.pipeline_stages: Dict[str, PipelineStage] = {}
        self.automation_rules: List[Dict[str, Any]] = []
        self.lock = Lock()
        
        # Initialize default pipeline stages
        self._initialize_pipeline_stages()
        
        # Initialize automation rules
        self._initialize_automation_rules()
        
        # Performance tracking
        self.pipeline_metrics = defaultdict(lambda: defaultdict(float))
        
    def _initialize_pipeline_stages(self):
        """Initialize default pipeline stages"""
        stages = [
            {"name": "New Lead", "order": 1, "conversion_rate": 0.4, "avg_duration": 3},
            {"name": "Contacted", "order": 2, "conversion_rate": 0.6, "avg_duration": 5},
            {"name": "Qualified", "order": 3, "conversion_rate": 0.7, "avg_duration": 7},
            {"name": "Proposal", "order": 4, "conversion_rate": 0.5, "avg_duration": 14},
            {"name": "Negotiation", "order": 5, "conversion_rate": 0.8, "avg_duration": 10},
            {"name": "Closed Won", "order": 6, "conversion_rate": 1.0, "avg_duration": 0},
            {"name": "Closed Lost", "order": 7, "conversion_rate": 0.0, "avg_duration": 0}
        ]
        
        for stage_data in stages:
            stage = PipelineStage(
                stage_id=str(uuid.uuid4()),
                name=stage_data["name"],
                order=stage_data["order"],
                conversion_rate=stage_data["conversion_rate"],
                avg_duration_days=stage_data["avg_duration"]
            )
            self.pipeline_stages[stage.stage_id] = stage
    
    def _initialize_automation_rules(self):
        """Initialize automation rules"""
        self.automation_rules = [
            {
                "name": "Auto-assign high-value leads",
                "condition": {"lead_score": {">=": 80}},
                "action": {"assign_to": "senior_sales_rep", "priority": "high"}
            },
            {
                "name": "Follow-up reminder",
                "condition": {"days_since_contact": {">=": 3}},
                "action": {"create_activity": "follow_up_call"}
            },
            {
                "name": "Move qualified leads to opportunity",
                "condition": {"status": "qualified", "budget": {">": 0}},
                "action": {"create_opportunity": True}
            },
            {
                "name": "Nurture low-score leads",
                "condition": {"lead_score": {"<": 30}},
                "action": {"add_to_nurture_campaign": True}
            }
        ]
    
    def create_lead(self, lead_data: Dict[str, Any]) -> str:
        """Create a new lead"""
        lead_id = str(uuid.uuid4())
        
        lead = Lead(
            lead_id=lead_id,
            email=lead_data["email"],
            first_name=lead_data["first_name"],
            last_name=lead_data["last_name"],
            company=lead_data["company"],
            title=lead_data.get("title", ""),
            phone=lead_data.get("phone"),
            source=LeadSource(lead_data.get("source", "direct")),
            company_size=CompanySize(lead_data["company_size"]) if lead_data.get("company_size") else None,
            industry=lead_data.get("industry"),
            budget=lead_data.get("budget"),
            timeline=lead_data.get("timeline"),
            pain_points=lead_data.get("pain_points", []),
            interests=lead_data.get("interests", []),
            tags=lead_data.get("tags", []),
            notes=lead_data.get("notes", ""),
            metadata=lead_data.get("metadata", {})
        )
        
        with self.lock:
            self.leads[lead_id] = lead
        
        # Calculate initial lead score
        asyncio.create_task(self.calculate_lead_score(lead_id))
        
        # Apply automation rules
        asyncio.create_task(self.apply_automation_rules(lead_id))
        
        return lead_id
    
    def update_lead(self, lead_id: str, updates: Dict[str, Any]) -> bool:
        """Update lead information"""
        with self.lock:
            if lead_id not in self.leads:
                return False
            
            lead = self.leads[lead_id]
            
            # Update fields
            for field, value in updates.items():
                if hasattr(lead, field):
                    if field == "source":
                        setattr(lead, field, LeadSource(value))
                    elif field == "status":
                        setattr(lead, field, LeadStatus(value))
                    elif field == "company_size":
                        setattr(lead, field, CompanySize(value) if value else None)
                    else:
                        setattr(lead, field, value)
            
            lead.updated_at = datetime.utcnow()
        
        # Recalculate lead score if relevant fields changed
        score_relevant_fields = ["company_size", "industry", "budget", "pain_points", "interests"]
        if any(field in updates for field in score_relevant_fields):
            asyncio.create_task(self.calculate_lead_score(lead_id))
        
        # Apply automation rules
        asyncio.create_task(self.apply_automation_rules(lead_id))
        
        return True
    
    def create_opportunity(self, opportunity_data: Dict[str, Any]) -> str:
        """Create a new opportunity"""
        opportunity_id = str(uuid.uuid4())
        
        opportunity = Opportunity(
            opportunity_id=opportunity_id,
            lead_id=opportunity_data["lead_id"],
            name=opportunity_data["name"],
            value=opportunity_data["value"],
            probability=opportunity_data.get("probability", 50.0),
            stage=LeadStatus(opportunity_data.get("stage", "qualified")),
            close_date=datetime.fromisoformat(opportunity_data["close_date"]),
            assigned_to=opportunity_data["assigned_to"],
            products_interested=opportunity_data.get("products_interested", []),
            competitors=opportunity_data.get("competitors", []),
            decision_makers=opportunity_data.get("decision_makers", []),
            next_steps=opportunity_data.get("next_steps", ""),
            notes=opportunity_data.get("notes", ""),
            priority=Priority(opportunity_data.get("priority", "medium")),
            metadata=opportunity_data.get("metadata", {})
        )
        
        with self.lock:
            self.opportunities[opportunity_id] = opportunity
        
        # Create initial activity
        self.create_activity({
            "opportunity_id": opportunity_id,
            "activity_type": "opportunity_created",
            "subject": f"New opportunity: {opportunity.name}",
            "description": f"Opportunity created with value ${opportunity.value:,.2f}",
            "assigned_to": opportunity.assigned_to
        })
        
        return opportunity_id
    
    def update_opportunity(self, opportunity_id: str, updates: Dict[str, Any]) -> bool:
        """Update opportunity information"""
        with self.lock:
            if opportunity_id not in self.opportunities:
                return False
            
            opportunity = self.opportunities[opportunity_id]
            
            # Track stage changes
            old_stage = opportunity.stage
            
            # Update fields
            for field, value in updates.items():
                if hasattr(opportunity, field):
                    if field == "stage":
                        setattr(opportunity, field, LeadStatus(value))
                    elif field == "priority":
                        setattr(opportunity, field, Priority(value))
                    elif field == "close_date":
                        setattr(opportunity, field, datetime.fromisoformat(value))
                    else:
                        setattr(opportunity, field, value)
            
            opportunity.updated_at = datetime.utcnow()
            
            # Handle stage changes
            new_stage = opportunity.stage
            if old_stage != new_stage:
                self._handle_stage_change(opportunity_id, old_stage, new_stage)
        
        return True
    
    def _handle_stage_change(self, opportunity_id: str, old_stage: LeadStatus, new_stage: LeadStatus):
        """Handle opportunity stage changes"""
        opportunity = self.opportunities[opportunity_id]
        
        # Create activity for stage change
        self.create_activity({
            "opportunity_id": opportunity_id,
            "activity_type": "stage_change",
            "subject": f"Stage changed: {old_stage.value} → {new_stage.value}",
            "description": f"Opportunity moved from {old_stage.value} to {new_stage.value}",
            "assigned_to": opportunity.assigned_to
        })
        
        # Handle closed stages
        if new_stage in [LeadStatus.CLOSED_WON, LeadStatus.CLOSED_LOST]:
            opportunity.closed_at = datetime.utcnow()
            
            # Update lead status
            if opportunity.lead_id in self.leads:
                self.update_lead(opportunity.lead_id, {"status": new_stage.value})
    
    def create_activity(self, activity_data: Dict[str, Any]) -> str:
        """Create a new sales activity"""
        activity_id = str(uuid.uuid4())
        
        activity = SalesActivity(
            activity_id=activity_id,
            lead_id=activity_data.get("lead_id"),
            opportunity_id=activity_data.get("opportunity_id"),
            activity_type=activity_data["activity_type"],
            subject=activity_data["subject"],
            description=activity_data["description"],
            assigned_to=activity_data["assigned_to"],
            due_date=datetime.fromisoformat(activity_data["due_date"]) if activity_data.get("due_date") else None,
            metadata=activity_data.get("metadata", {})
        )
        
        with self.lock:
            self.activities[activity_id] = activity
        
        return activity_id
    
    def complete_activity(self, activity_id: str, outcome: str, next_activity: Optional[str] = None) -> bool:
        """Complete a sales activity"""
        with self.lock:
            if activity_id not in self.activities:
                return False
            
            activity = self.activities[activity_id]
            activity.completed_at = datetime.utcnow()
            activity.outcome = outcome
            activity.next_activity = next_activity
        
        # Update last contact for lead/opportunity
        if activity.lead_id:
            self.update_lead(activity.lead_id, {"last_contact": datetime.utcnow()})
        
        return True
    
    async def calculate_lead_score(self, lead_id: str) -> float:
        """Calculate lead score based on various factors"""
        if lead_id not in self.leads:
            return 0.0
        
        lead = self.leads[lead_id]
        score = 0.0
        
        # Company size scoring
        if lead.company_size:
            size_scores = {
                CompanySize.STARTUP: 20,
                CompanySize.SMALL_BUSINESS: 40,
                CompanySize.MID_MARKET: 70,
                CompanySize.ENTERPRISE: 100
            }
            score += size_scores.get(lead.company_size, 0)
        
        # Budget scoring
        if lead.budget:
            if lead.budget >= 100000:
                score += 30
            elif lead.budget >= 50000:
                score += 20
            elif lead.budget >= 10000:
                score += 10
            else:
                score += 5
        
        # Industry scoring (tech-friendly industries get higher scores)
        tech_industries = ["technology", "software", "ai", "fintech", "healthcare", "finance"]
        if lead.industry and any(industry in lead.industry.lower() for industry in tech_industries):
            score += 15
        
        # Pain points scoring
        relevant_pain_points = ["automation", "efficiency", "ai", "integration", "scaling"]
        matching_pain_points = sum(1 for pain in lead.pain_points 
                                 if any(keyword in pain.lower() for keyword in relevant_pain_points))
        score += matching_pain_points * 5
        
        # Interest scoring
        relevant_interests = ["ai agents", "automation", "api", "integration", "workflow"]
        matching_interests = sum(1 for interest in lead.interests 
                               if any(keyword in interest.lower() for keyword in relevant_interests))
        score += matching_interests * 8
        
        # Source scoring
        source_scores = {
            LeadSource.REFERRAL: 15,
            LeadSource.INBOUND: 12,
            LeadSource.CONTENT_MARKETING: 10,
            LeadSource.WEBSITE: 8,
            LeadSource.SOCIAL_MEDIA: 6,
            LeadSource.PAID_ADVERTISING: 5,
            LeadSource.COLD_OUTREACH: 3
        }
        score += source_scores.get(lead.source, 0)
        
        # Timeline urgency scoring
        if lead.timeline:
            if "immediate" in lead.timeline.lower() or "asap" in lead.timeline.lower():
                score += 20
            elif "month" in lead.timeline.lower():
                score += 15
            elif "quarter" in lead.timeline.lower():
                score += 10
            elif "year" in lead.timeline.lower():
                score += 5
        
        # Cap score at 100
        score = min(100, score)
        
        # Update lead with calculated score
        with self.lock:
            lead.lead_score = score
            lead.updated_at = datetime.utcnow()
        
        return score
    
    async def apply_automation_rules(self, lead_id: str):
        """Apply automation rules to a lead"""
        if lead_id not in self.leads:
            return
        
        lead = self.leads[lead_id]
        
        for rule in self.automation_rules:
            if self._evaluate_rule_condition(lead, rule["condition"]):
                await self._execute_rule_action(lead, rule["action"])
    
    def _evaluate_rule_condition(self, lead: Lead, condition: Dict[str, Any]) -> bool:
        """Evaluate if a rule condition is met"""
        for field, criteria in condition.items():
            if field == "lead_score":
                value = lead.lead_score
            elif field == "days_since_contact":
                if not lead.last_contact:
                    value = 999  # Large number if never contacted
                else:
                    value = (datetime.utcnow() - lead.last_contact).days
            elif field == "status":
                value = lead.status.value
            elif field == "budget":
                value = lead.budget or 0
            else:
                value = getattr(lead, field, None)
            
            # Evaluate criteria
            if isinstance(criteria, dict):
                for operator, threshold in criteria.items():
                    if operator == ">=" and value < threshold:
                        return False
                    elif operator == ">" and value <= threshold:
                        return False
                    elif operator == "<=" and value > threshold:
                        return False
                    elif operator == "<" and value >= threshold:
                        return False
                    elif operator == "==" and value != threshold:
                        return False
            else:
                if value != criteria:
                    return False
        
        return True
    
    async def _execute_rule_action(self, lead: Lead, action: Dict[str, Any]):
        """Execute automation rule action"""
        if "assign_to" in action:
            self.update_lead(lead.lead_id, {"assigned_to": action["assign_to"]})
        
        if "priority" in action:
            # Add priority tag
            if "priority" not in lead.tags:
                tags = lead.tags + [f"priority_{action['priority']}"]
                self.update_lead(lead.lead_id, {"tags": tags})
        
        if "create_activity" in action:
            activity_type = action["create_activity"]
            self.create_activity({
                "lead_id": lead.lead_id,
                "activity_type": activity_type,
                "subject": f"Automated {activity_type.replace('_', ' ').title()}",
                "description": f"Automated activity triggered by rule",
                "assigned_to": lead.assigned_to or "system"
            })
        
        if "create_opportunity" in action and action["create_opportunity"]:
            # Create opportunity if one doesn't exist
            existing_opportunity = any(
                opp.lead_id == lead.lead_id for opp in self.opportunities.values()
            )
            
            if not existing_opportunity:
                self.create_opportunity({
                    "lead_id": lead.lead_id,
                    "name": f"{lead.company} - {lead.first_name} {lead.last_name}",
                    "value": lead.budget or 10000,  # Default value
                    "close_date": (datetime.utcnow() + timedelta(days=30)).isoformat(),
                    "assigned_to": lead.assigned_to or "system"
                })
        
        if "add_to_nurture_campaign" in action:
            # Add nurture tag
            if "nurture" not in lead.tags:
                tags = lead.tags + ["nurture"]
                self.update_lead(lead.lead_id, {"tags": tags})
    
    def get_pipeline_overview(self) -> Dict[str, Any]:
        """Get pipeline overview and metrics"""
        overview = {
            "total_leads": len(self.leads),
            "total_opportunities": len(self.opportunities),
            "total_activities": len(self.activities),
            "pipeline_value": 0.0,
            "weighted_pipeline_value": 0.0,
            "leads_by_stage": defaultdict(int),
            "opportunities_by_stage": defaultdict(int),
            "activities_by_type": defaultdict(int),
            "conversion_rates": {},
            "avg_deal_size": 0.0,
            "avg_sales_cycle": 0.0
        }
        
        # Calculate pipeline metrics
        for lead in self.leads.values():
            overview["leads_by_stage"][lead.status.value] += 1
        
        opportunity_values = []
        closed_won_opportunities = []
        
        for opportunity in self.opportunities.values():
            overview["opportunities_by_stage"][opportunity.stage.value] += 1
            overview["pipeline_value"] += opportunity.value
            overview["weighted_pipeline_value"] += opportunity.value * (opportunity.probability / 100)
            
            opportunity_values.append(opportunity.value)
            
            if opportunity.stage == LeadStatus.CLOSED_WON:
                closed_won_opportunities.append(opportunity)
        
        # Calculate averages
        if opportunity_values:
            overview["avg_deal_size"] = statistics.mean(opportunity_values)
        
        if closed_won_opportunities:
            sales_cycles = []
            for opp in closed_won_opportunities:
                if opp.closed_at:
                    cycle_days = (opp.closed_at - opp.created_at).days
                    sales_cycles.append(cycle_days)
            
            if sales_cycles:
                overview["avg_sales_cycle"] = statistics.mean(sales_cycles)
        
        # Activity metrics
        for activity in self.activities.values():
            overview["activities_by_type"][activity.activity_type] += 1
        
        # Conversion rates
        for stage in LeadStatus:
            stage_leads = [lead for lead in self.leads.values() if lead.status == stage]
            if stage_leads:
                # Calculate conversion to next stage (simplified)
                overview["conversion_rates"][stage.value] = 0.5  # Placeholder
        
        return overview
    
    def get_sales_forecast(self, months: int = 3) -> Dict[str, Any]:
        """Generate sales forecast"""
        forecast_end = datetime.utcnow() + timedelta(days=months * 30)
        
        # Get opportunities closing within the forecast period
        forecast_opportunities = [
            opp for opp in self.opportunities.values()
            if opp.close_date <= forecast_end and opp.stage not in [LeadStatus.CLOSED_WON, LeadStatus.CLOSED_LOST]
        ]
        
        # Calculate forecast by month
        monthly_forecast = defaultdict(lambda: {"count": 0, "value": 0.0, "weighted_value": 0.0})
        
        for opportunity in forecast_opportunities:
            month_key = opportunity.close_date.strftime("%Y-%m")
            monthly_forecast[month_key]["count"] += 1
            monthly_forecast[month_key]["value"] += opportunity.value
            monthly_forecast[month_key]["weighted_value"] += opportunity.value * (opportunity.probability / 100)
        
        # Calculate total forecast
        total_forecast = {
            "total_opportunities": len(forecast_opportunities),
            "total_value": sum(opp.value for opp in forecast_opportunities),
            "weighted_value": sum(opp.value * (opp.probability / 100) for opp in forecast_opportunities),
            "monthly_breakdown": dict(monthly_forecast)
        }
        
        # Add confidence levels
        if len(forecast_opportunities) > 10:
            total_forecast["confidence"] = "high"
        elif len(forecast_opportunities) > 5:
            total_forecast["confidence"] = "medium"
        else:
            total_forecast["confidence"] = "low"
        
        return total_forecast
    
    def get_lead_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get lead analytics"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        recent_leads = [lead for lead in self.leads.values() if lead.created_at >= cutoff_date]
        
        analytics = {
            "total_leads": len(recent_leads),
            "avg_lead_score": 0.0,
            "leads_by_source": defaultdict(int),
            "leads_by_industry": defaultdict(int),
            "leads_by_company_size": defaultdict(int),
            "conversion_funnel": defaultdict(int),
            "top_performing_sources": [],
            "lead_quality_distribution": {"high": 0, "medium": 0, "low": 0}
        }
        
        if not recent_leads:
            return analytics
        
        # Calculate metrics
        lead_scores = [lead.lead_score for lead in recent_leads]
        analytics["avg_lead_score"] = statistics.mean(lead_scores)
        
        for lead in recent_leads:
            analytics["leads_by_source"][lead.source.value] += 1
            analytics["conversion_funnel"][lead.status.value] += 1
            
            if lead.industry:
                analytics["leads_by_industry"][lead.industry] += 1
            
            if lead.company_size:
                analytics["leads_by_company_size"][lead.company_size.value] += 1
            
            # Lead quality distribution
            if lead.lead_score >= 70:
                analytics["lead_quality_distribution"]["high"] += 1
            elif lead.lead_score >= 40:
                analytics["lead_quality_distribution"]["medium"] += 1
            else:
                analytics["lead_quality_distribution"]["low"] += 1
        
        # Top performing sources (by average lead score)
        source_performance = defaultdict(list)
        for lead in recent_leads:
            source_performance[lead.source.value].append(lead.lead_score)
        
        source_avg_scores = {
            source: statistics.mean(scores)
            for source, scores in source_performance.items()
        }
        
        analytics["top_performing_sources"] = sorted(
            source_avg_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return analytics
    
    async def get_recommendations(self) -> List[Dict[str, Any]]:
        """Get AI-powered sales recommendations"""
        recommendations = []
        
        # Analyze pipeline health
        overview = self.get_pipeline_overview()
        
        # Stalled opportunities
        stalled_opportunities = []
        for opp in self.opportunities.values():
            days_in_stage = (datetime.utcnow() - opp.updated_at).days
            expected_duration = 14  # Default expected duration
            
            if days_in_stage > expected_duration * 1.5:
                stalled_opportunities.append(opp)
        
        if stalled_opportunities:
            recommendations.append({
                "type": "stalled_opportunities",
                "priority": "high",
                "title": f"{len(stalled_opportunities)} Stalled Opportunities",
                "description": "Several opportunities have been inactive for longer than expected",
                "actions": [
                    "Schedule follow-up calls",
                    "Send check-in emails",
                    "Reassess opportunity viability"
                ],
                "opportunities": [opp.opportunity_id for opp in stalled_opportunities[:5]]
            })
        
        # High-value leads not contacted
        uncontacted_high_value = [
            lead for lead in self.leads.values()
            if lead.lead_score >= 70 and not lead.last_contact
        ]
        
        if uncontacted_high_value:
            recommendations.append({
                "type": "high_value_leads",
                "priority": "urgent",
                "title": f"{len(uncontacted_high_value)} High-Value Leads Need Attention",
                "description": "High-scoring leads that haven't been contacted yet",
                "actions": [
                    "Immediate outreach required",
                    "Assign to senior sales rep",
                    "Personalize initial contact"
                ],
                "leads": [lead.lead_id for lead in uncontacted_high_value[:5]]
            })
        
        # Forecast risk
        forecast = self.get_sales_forecast()
        if forecast["confidence"] == "low":
            recommendations.append({
                "type": "forecast_risk",
                "priority": "medium",
                "title": "Low Forecast Confidence",
                "description": "Pipeline lacks sufficient qualified opportunities",
                "actions": [
                    "Increase lead generation efforts",
                    "Accelerate qualification process",
                    "Review and adjust sales targets"
                ]
            })
        
        # Lead source optimization
        analytics = self.get_lead_analytics()
        top_sources = analytics["top_performing_sources"]
        
        if top_sources:
            best_source = top_sources[0]
            recommendations.append({
                "type": "lead_source_optimization",
                "priority": "low",
                "title": f"Optimize Lead Generation",
                "description": f"Best performing source: {best_source[0]} (avg score: {best_source[1]:.1f})",
                "actions": [
                    f"Increase investment in {best_source[0]}",
                    "Analyze what makes this source successful",
                    "Replicate success factors in other channels"
                ]
            })
        
        return recommendations

# Global sales pipeline automation instance
sales_pipeline = SalesPipelineAutomation()

# Convenience functions
def create_lead(lead_data: Dict[str, Any]) -> str:
    """Create a new lead"""
    return sales_pipeline.create_lead(lead_data)

def create_opportunity(opportunity_data: Dict[str, Any]) -> str:
    """Create a new opportunity"""
    return sales_pipeline.create_opportunity(opportunity_data)

def get_pipeline_overview() -> Dict[str, Any]:
    """Get pipeline overview"""
    return sales_pipeline.get_pipeline_overview()

def get_sales_forecast(months: int = 3) -> Dict[str, Any]:
    """Get sales forecast"""
    return sales_pipeline.get_sales_forecast(months)

async def get_sales_recommendations() -> List[Dict[str, Any]]:
    """Get sales recommendations"""
    return await sales_pipeline.get_recommendations()