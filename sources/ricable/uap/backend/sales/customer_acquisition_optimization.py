# File: backend/sales/customer_acquisition_optimization.py
"""
Customer Acquisition Cost (CAC) Optimization System for UAP Platform

Provides comprehensive CAC analysis, channel optimization,
marketing spend optimization, and customer lifetime value analysis.
"""

import asyncio
import json
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import uuid
import warnings
from threading import Lock

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .pipeline_automation import sales_pipeline, LeadSource
from .lead_scoring import lead_scoring_engine
from ..market.pricing_optimization import pricing_optimizer

class MarketingChannel(Enum):
    """Marketing channel types"""
    ORGANIC_SEARCH = "organic_search"
    PAID_SEARCH = "paid_search"
    SOCIAL_MEDIA_ORGANIC = "social_media_organic"
    SOCIAL_MEDIA_PAID = "social_media_paid"
    EMAIL_MARKETING = "email_marketing"
    CONTENT_MARKETING = "content_marketing"
    REFERRALS = "referrals"
    PARTNERSHIPS = "partnerships"
    EVENTS = "events"
    DIRECT = "direct"
    DISPLAY_ADVERTISING = "display_advertising"
    VIDEO_ADVERTISING = "video_advertising"
    INFLUENCER_MARKETING = "influencer_marketing"
    AFFILIATE_MARKETING = "affiliate_marketing"

class CampaignStatus(Enum):
    """Campaign status types"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    DRAFT = "draft"

class OptimizationGoal(Enum):
    """Optimization goal types"""
    MINIMIZE_CAC = "minimize_cac"
    MAXIMIZE_ROI = "maximize_roi"
    MAXIMIZE_CONVERSIONS = "maximize_conversions"
    MAXIMIZE_REVENUE = "maximize_revenue"
    IMPROVE_QUALITY = "improve_quality"

@dataclass
class MarketingCampaign:
    """Marketing campaign data"""
    campaign_id: str
    name: str
    channel: MarketingChannel
    status: CampaignStatus
    budget: float
    spent: float
    start_date: datetime
    end_date: Optional[datetime]
    target_audience: str
    goals: List[str]
    impressions: int = 0
    clicks: int = 0
    leads_generated: int = 0
    conversions: int = 0
    revenue_generated: float = 0.0
    cost_per_click: float = 0.0
    cost_per_lead: float = 0.0
    cost_per_acquisition: float = 0.0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "name": self.name,
            "channel": self.channel.value,
            "status": self.status.value,
            "budget": self.budget,
            "spent": self.spent,
            "budget_utilization": (self.spent / self.budget) * 100 if self.budget > 0 else 0,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "target_audience": self.target_audience,
            "goals": self.goals,
            "impressions": self.impressions,
            "clicks": self.clicks,
            "leads_generated": self.leads_generated,
            "conversions": self.conversions,
            "revenue_generated": self.revenue_generated,
            "cost_per_click": self.cost_per_click,
            "cost_per_lead": self.cost_per_lead,
            "cost_per_acquisition": self.cost_per_acquisition,
            "click_through_rate": (self.clicks / self.impressions) * 100 if self.impressions > 0 else 0,
            "conversion_rate": (self.conversions / self.clicks) * 100 if self.clicks > 0 else 0,
            "lead_conversion_rate": (self.conversions / self.leads_generated) * 100 if self.leads_generated > 0 else 0,
            "return_on_ad_spend": (self.revenue_generated / self.spent) * 100 if self.spent > 0 else 0,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class ChannelPerformance:
    """Channel performance metrics"""
    channel: MarketingChannel
    total_spend: float
    total_leads: int
    total_conversions: int
    total_revenue: float
    avg_cac: float
    avg_ltv: float
    roi: float
    lead_quality_score: float
    conversion_rate: float
    time_to_conversion_days: float
    retention_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel.value,
            "total_spend": self.total_spend,
            "total_leads": self.total_leads,
            "total_conversions": self.total_conversions,
            "total_revenue": self.total_revenue,
            "avg_cac": self.avg_cac,
            "avg_ltv": self.avg_ltv,
            "ltv_cac_ratio": self.avg_ltv / self.avg_cac if self.avg_cac > 0 else 0,
            "roi": self.roi,
            "lead_quality_score": self.lead_quality_score,
            "conversion_rate": self.conversion_rate,
            "time_to_conversion_days": self.time_to_conversion_days,
            "retention_rate": self.retention_rate,
            "efficiency_score": self._calculate_efficiency_score()
        }
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate overall channel efficiency score"""
        # Weighted combination of key metrics
        roi_score = min(100, self.roi * 10) if self.roi > 0 else 0  # Cap at 100
        quality_score = self.lead_quality_score
        conversion_score = min(100, self.conversion_rate * 5)  # Cap at 100
        ltv_cac_score = min(100, (self.avg_ltv / self.avg_cac) * 20) if self.avg_cac > 0 else 0
        
        # Weighted average
        efficiency_score = (
            roi_score * 0.3 +
            quality_score * 0.25 +
            conversion_score * 0.25 +
            ltv_cac_score * 0.2
        )
        
        return efficiency_score

@dataclass
class OptimizationRecommendation:
    """CAC optimization recommendation"""
    recommendation_id: str
    channel: MarketingChannel
    recommendation_type: str
    current_spend: float
    recommended_spend: float
    expected_impact: Dict[str, float]
    confidence: float
    reasoning: List[str]
    priority: str
    implementation_steps: List[str]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        spend_change = self.recommended_spend - self.current_spend
        spend_change_percent = (spend_change / self.current_spend) * 100 if self.current_spend > 0 else 0
        
        return {
            "recommendation_id": self.recommendation_id,
            "channel": self.channel.value,
            "recommendation_type": self.recommendation_type,
            "current_spend": self.current_spend,
            "recommended_spend": self.recommended_spend,
            "spend_change": spend_change,
            "spend_change_percent": spend_change_percent,
            "expected_impact": self.expected_impact,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "priority": self.priority,
            "implementation_steps": self.implementation_steps,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class CustomerCohort:
    """Customer cohort data for LTV analysis"""
    cohort_id: str
    acquisition_period: str  # e.g., "2024-01"
    channel: MarketingChannel
    initial_customers: int
    revenue_by_month: List[float]
    retention_by_month: List[float]
    avg_cac: float
    calculated_ltv: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cohort_id": self.cohort_id,
            "acquisition_period": self.acquisition_period,
            "channel": self.channel.value,
            "initial_customers": self.initial_customers,
            "revenue_by_month": self.revenue_by_month,
            "retention_by_month": self.retention_by_month,
            "avg_cac": self.avg_cac,
            "calculated_ltv": self.calculated_ltv,
            "ltv_cac_ratio": self.calculated_ltv / self.avg_cac if self.avg_cac > 0 else 0,
            "payback_period_months": self._calculate_payback_period()
        }
    
    def _calculate_payback_period(self) -> float:
        """Calculate payback period in months"""
        cumulative_revenue = 0
        for month, revenue in enumerate(self.revenue_by_month):
            cumulative_revenue += revenue
            if cumulative_revenue >= self.avg_cac:
                return month + 1
        return len(self.revenue_by_month)  # If never paid back

class CustomerAcquisitionOptimizer:
    """Customer acquisition cost optimization engine"""
    
    def __init__(self):
        self.campaigns: Dict[str, MarketingCampaign] = {}
        self.channel_performance: Dict[MarketingChannel, ChannelPerformance] = {}
        self.customer_cohorts: Dict[str, CustomerCohort] = {}
        self.optimization_history: deque = deque(maxlen=1000)
        self.budget_allocations: Dict[MarketingChannel, float] = defaultdict(float)
        self.lock = Lock()
        
        # ML models
        self.cac_prediction_model = None
        self.ltv_prediction_model = None
        self.channel_performance_model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
        # Configuration
        self.target_ltv_cac_ratio = 3.0  # Target LTV:CAC ratio
        self.max_cac_budget_percent = 30.0  # Max % of revenue for CAC
        self.min_campaign_budget = 1000.0  # Minimum campaign budget
        self.optimization_frequency_days = 7  # How often to optimize
        
        # Initialize with sample campaigns
        self._initialize_sample_campaigns()
    
    def _initialize_sample_campaigns(self):
        """Initialize with sample marketing campaigns"""
        sample_campaigns = [
            {
                "name": "Google Ads - AI Agents",
                "channel": MarketingChannel.PAID_SEARCH,
                "budget": 10000,
                "spent": 7500,
                "target_audience": "developers",
                "goals": ["lead_generation", "brand_awareness"],
                "impressions": 50000,
                "clicks": 2500,
                "leads_generated": 125,
                "conversions": 15
            },
            {
                "name": "LinkedIn Content Marketing",
                "channel": MarketingChannel.CONTENT_MARKETING,
                "budget": 5000,
                "spent": 3200,
                "target_audience": "enterprise_decision_makers",
                "goals": ["thought_leadership", "lead_generation"],
                "impressions": 25000,
                "clicks": 1200,
                "leads_generated": 80,
                "conversions": 12
            },
            {
                "name": "Developer Conference Sponsorship",
                "channel": MarketingChannel.EVENTS,
                "budget": 15000,
                "spent": 15000,
                "target_audience": "technical_professionals",
                "goals": ["brand_awareness", "lead_generation"],
                "impressions": 5000,
                "clicks": 500,
                "leads_generated": 200,
                "conversions": 25
            },
            {
                "name": "Referral Program",
                "channel": MarketingChannel.REFERRALS,
                "budget": 8000,
                "spent": 4800,
                "target_audience": "existing_customers",
                "goals": ["customer_acquisition", "retention"],
                "impressions": 10000,
                "clicks": 800,
                "leads_generated": 60,
                "conversions": 18
            }
        ]
        
        for campaign_data in sample_campaigns:
            campaign = MarketingCampaign(
                campaign_id=str(uuid.uuid4()),
                name=campaign_data["name"],
                channel=campaign_data["channel"],
                status=CampaignStatus.ACTIVE,
                budget=campaign_data["budget"],
                spent=campaign_data["spent"],
                start_date=datetime.utcnow() - timedelta(days=30),
                end_date=None,
                target_audience=campaign_data["target_audience"],
                goals=campaign_data["goals"],
                impressions=campaign_data["impressions"],
                clicks=campaign_data["clicks"],
                leads_generated=campaign_data["leads_generated"],
                conversions=campaign_data["conversions"]
            )
            
            # Calculate metrics
            if campaign.clicks > 0:
                campaign.cost_per_click = campaign.spent / campaign.clicks
            if campaign.leads_generated > 0:
                campaign.cost_per_lead = campaign.spent / campaign.leads_generated
            if campaign.conversions > 0:
                campaign.cost_per_acquisition = campaign.spent / campaign.conversions
                # Estimate revenue (would come from actual data)
                campaign.revenue_generated = campaign.conversions * 500  # $500 avg revenue per conversion
            
            self.campaigns[campaign.campaign_id] = campaign
        
        # Calculate initial channel performance
        asyncio.create_task(self._update_channel_performance())
    
    def create_campaign(self, campaign_data: Dict[str, Any]) -> str:
        """Create a new marketing campaign"""
        campaign_id = str(uuid.uuid4())
        
        campaign = MarketingCampaign(
            campaign_id=campaign_id,
            name=campaign_data["name"],
            channel=MarketingChannel(campaign_data["channel"]),
            status=CampaignStatus(campaign_data.get("status", "active")),
            budget=campaign_data["budget"],
            spent=campaign_data.get("spent", 0.0),
            start_date=datetime.fromisoformat(campaign_data["start_date"]),
            end_date=datetime.fromisoformat(campaign_data["end_date"]) if campaign_data.get("end_date") else None,
            target_audience=campaign_data["target_audience"],
            goals=campaign_data.get("goals", [])
        )
        
        with self.lock:
            self.campaigns[campaign_id] = campaign
        
        return campaign_id
    
    def update_campaign_metrics(self, campaign_id: str, metrics: Dict[str, Any]) -> bool:
        """Update campaign performance metrics"""
        with self.lock:
            if campaign_id not in self.campaigns:
                return False
            
            campaign = self.campaigns[campaign_id]
            
            # Update metrics
            for metric, value in metrics.items():
                if hasattr(campaign, metric):
                    setattr(campaign, metric, value)
            
            # Recalculate derived metrics
            if campaign.clicks > 0 and campaign.spent > 0:
                campaign.cost_per_click = campaign.spent / campaign.clicks
            if campaign.leads_generated > 0 and campaign.spent > 0:
                campaign.cost_per_lead = campaign.spent / campaign.leads_generated
            if campaign.conversions > 0 and campaign.spent > 0:
                campaign.cost_per_acquisition = campaign.spent / campaign.conversions
        
        # Update channel performance
        asyncio.create_task(self._update_channel_performance())
        
        return True
    
    async def _update_channel_performance(self):
        """Update channel performance metrics"""
        channel_data = defaultdict(lambda: {
            "total_spend": 0,
            "total_leads": 0,
            "total_conversions": 0,
            "total_revenue": 0,
            "campaigns": []
        })
        
        # Aggregate data by channel
        for campaign in self.campaigns.values():
            channel = campaign.channel
            channel_data[channel]["total_spend"] += campaign.spent
            channel_data[channel]["total_leads"] += campaign.leads_generated
            channel_data[channel]["total_conversions"] += campaign.conversions
            channel_data[channel]["total_revenue"] += campaign.revenue_generated
            channel_data[channel]["campaigns"].append(campaign)
        
        # Calculate channel performance metrics
        with self.lock:
            for channel, data in channel_data.items():
                if data["total_conversions"] > 0:
                    avg_cac = data["total_spend"] / data["total_conversions"]
                else:
                    avg_cac = 0
                
                # Calculate average LTV (would use actual customer data)
                avg_ltv = await self._calculate_channel_ltv(channel)
                
                # Calculate ROI
                roi = ((data["total_revenue"] - data["total_spend"]) / data["total_spend"]) * 100 if data["total_spend"] > 0 else 0
                
                # Calculate lead quality score
                lead_quality_score = await self._calculate_channel_lead_quality(channel)
                
                # Calculate conversion rate
                total_clicks = sum(campaign.clicks for campaign in data["campaigns"])
                conversion_rate = (data["total_conversions"] / total_clicks) * 100 if total_clicks > 0 else 0
                
                # Calculate other metrics (simplified)
                time_to_conversion = 14  # Default 14 days
                retention_rate = 85  # Default 85%
                
                self.channel_performance[channel] = ChannelPerformance(
                    channel=channel,
                    total_spend=data["total_spend"],
                    total_leads=data["total_leads"],
                    total_conversions=data["total_conversions"],
                    total_revenue=data["total_revenue"],
                    avg_cac=avg_cac,
                    avg_ltv=avg_ltv,
                    roi=roi,
                    lead_quality_score=lead_quality_score,
                    conversion_rate=conversion_rate,
                    time_to_conversion_days=time_to_conversion,
                    retention_rate=retention_rate
                )
    
    async def _calculate_channel_ltv(self, channel: MarketingChannel) -> float:
        """Calculate average LTV for a channel"""
        # This would integrate with actual customer data
        # For now, use channel-based estimates
        
        channel_ltv_estimates = {
            MarketingChannel.REFERRALS: 2500,
            MarketingChannel.ORGANIC_SEARCH: 2000,
            MarketingChannel.CONTENT_MARKETING: 1800,
            MarketingChannel.EVENTS: 1600,
            MarketingChannel.PAID_SEARCH: 1400,
            MarketingChannel.SOCIAL_MEDIA_ORGANIC: 1200,
            MarketingChannel.EMAIL_MARKETING: 1000,
            MarketingChannel.SOCIAL_MEDIA_PAID: 900,
            MarketingChannel.DISPLAY_ADVERTISING: 800
        }
        
        return channel_ltv_estimates.get(channel, 1200)
    
    async def _calculate_channel_lead_quality(self, channel: MarketingChannel) -> float:
        """Calculate average lead quality score for a channel"""
        # Get leads from this channel and calculate average quality
        channel_leads = []
        
        for lead in sales_pipeline.leads.values():
            if self._map_lead_source_to_channel(lead.source) == channel:
                channel_leads.append(lead)
        
        if not channel_leads:
            return 50.0  # Default score
        
        # Calculate average lead score
        lead_scores = [lead.lead_score for lead in channel_leads if lead.lead_score > 0]
        
        if lead_scores:
            return statistics.mean(lead_scores)
        
        return 50.0
    
    def _map_lead_source_to_channel(self, lead_source: LeadSource) -> MarketingChannel:
        """Map lead source to marketing channel"""
        mapping = {
            LeadSource.WEBSITE: MarketingChannel.ORGANIC_SEARCH,
            LeadSource.REFERRAL: MarketingChannel.REFERRALS,
            LeadSource.SOCIAL_MEDIA: MarketingChannel.SOCIAL_MEDIA_ORGANIC,
            LeadSource.EMAIL_CAMPAIGN: MarketingChannel.EMAIL_MARKETING,
            LeadSource.CONTENT_MARKETING: MarketingChannel.CONTENT_MARKETING,
            LeadSource.PAID_ADVERTISING: MarketingChannel.PAID_SEARCH,
            LeadSource.EVENTS: MarketingChannel.EVENTS,
            LeadSource.PARTNERSHIPS: MarketingChannel.PARTNERSHIPS,
            LeadSource.COLD_OUTREACH: MarketingChannel.EMAIL_MARKETING,
            LeadSource.INBOUND: MarketingChannel.ORGANIC_SEARCH,
            LeadSource.DIRECT: MarketingChannel.DIRECT
        }
        
        return mapping.get(lead_source, MarketingChannel.DIRECT)
    
    async def optimize_budget_allocation(self, total_budget: float, 
                                       goal: OptimizationGoal) -> Dict[MarketingChannel, float]:
        """Optimize budget allocation across channels"""
        if not self.channel_performance:
            await self._update_channel_performance()
        
        # Get channel efficiency scores
        channel_scores = {}
        for channel, performance in self.channel_performance.items():
            if goal == OptimizationGoal.MINIMIZE_CAC:
                # Prioritize channels with low CAC
                score = 1 / (performance.avg_cac + 1) if performance.avg_cac > 0 else 0
            elif goal == OptimizationGoal.MAXIMIZE_ROI:
                # Prioritize channels with high ROI
                score = performance.roi / 100
            elif goal == OptimizationGoal.MAXIMIZE_CONVERSIONS:
                # Prioritize channels with high conversion rates
                score = performance.conversion_rate / 100
            elif goal == OptimizationGoal.IMPROVE_QUALITY:
                # Prioritize channels with high lead quality
                score = performance.lead_quality_score / 100
            else:  # MAXIMIZE_REVENUE
                # Prioritize channels with high revenue efficiency
                score = performance.total_revenue / max(1, performance.total_spend)
            
            channel_scores[channel] = max(0, score)
        
        # Normalize scores
        total_score = sum(channel_scores.values())
        if total_score == 0:
            # Equal allocation if no performance data
            allocation_per_channel = total_budget / len(channel_scores)
            return {channel: allocation_per_channel for channel in channel_scores.keys()}
        
        # Allocate budget proportionally to scores
        budget_allocation = {}
        for channel, score in channel_scores.items():
            proportion = score / total_score
            allocated_budget = total_budget * proportion
            
            # Ensure minimum budget for each channel
            budget_allocation[channel] = max(self.min_campaign_budget, allocated_budget)
        
        # Adjust if total exceeds budget due to minimum allocations
        total_allocated = sum(budget_allocation.values())
        if total_allocated > total_budget:
            adjustment_factor = total_budget / total_allocated
            budget_allocation = {
                channel: budget * adjustment_factor
                for channel, budget in budget_allocation.items()
            }
        
        return budget_allocation
    
    async def generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate CAC optimization recommendations"""
        if not self.channel_performance:
            await self._update_channel_performance()
        
        recommendations = []
        
        for channel, performance in self.channel_performance.items():
            current_spend = performance.total_spend
            
            # Skip channels with no spend
            if current_spend == 0:
                continue
            
            # Analyze channel performance
            ltv_cac_ratio = performance.avg_ltv / performance.avg_cac if performance.avg_cac > 0 else 0
            roi = performance.roi
            efficiency_score = performance._calculate_efficiency_score()
            
            # Generate recommendations based on performance
            if ltv_cac_ratio > 5 and roi > 200:
                # High performing channel - increase spend
                recommendation_type = "increase_budget"
                recommended_spend = current_spend * 1.5
                expected_impact = {
                    "additional_conversions": performance.total_conversions * 0.5,
                    "additional_revenue": performance.total_revenue * 0.5,
                    "roi_change": 0  # Maintain current ROI
                }
                reasoning = [
                    f"Excellent LTV:CAC ratio of {ltv_cac_ratio:.1f}",
                    f"Strong ROI of {roi:.1f}%",
                    "Channel has proven efficiency"
                ]
                priority = "high"
                
            elif ltv_cac_ratio > 3 and roi > 100:
                # Good performing channel - moderate increase
                recommendation_type = "optimize_budget"
                recommended_spend = current_spend * 1.2
                expected_impact = {
                    "additional_conversions": performance.total_conversions * 0.2,
                    "additional_revenue": performance.total_revenue * 0.2,
                    "cac_reduction_percent": 5
                }
                reasoning = [
                    f"Good LTV:CAC ratio of {ltv_cac_ratio:.1f}",
                    f"Positive ROI of {roi:.1f}%",
                    "Room for scaling with optimization"
                ]
                priority = "medium"
                
            elif ltv_cac_ratio < 2 or roi < 50:
                # Poor performing channel - reduce or optimize
                if efficiency_score < 30:
                    recommendation_type = "reduce_budget"
                    recommended_spend = current_spend * 0.5
                    reasoning = [
                        f"Low LTV:CAC ratio of {ltv_cac_ratio:.1f}",
                        f"Poor ROI of {roi:.1f}%",
                        "Channel underperforming significantly"
                    ]
                else:
                    recommendation_type = "optimize_targeting"
                    recommended_spend = current_spend
                    reasoning = [
                        f"Suboptimal LTV:CAC ratio of {ltv_cac_ratio:.1f}",
                        "Potential for improvement with better targeting"
                    ]
                
                expected_impact = {
                    "cac_reduction_percent": 20,
                    "quality_improvement": 15,
                    "roi_improvement": 25
                }
                priority = "high"
                
            else:
                # Average performing channel - maintain
                recommendation_type = "maintain_budget"
                recommended_spend = current_spend
                expected_impact = {
                    "status": "maintain_current_performance"
                }
                reasoning = [
                    f"Adequate LTV:CAC ratio of {ltv_cac_ratio:.1f}",
                    "Channel performing within acceptable range"
                ]
                priority = "low"
            
            # Generate implementation steps
            implementation_steps = self._generate_implementation_steps(recommendation_type, channel)
            
            # Calculate confidence based on data quality
            confidence = self._calculate_recommendation_confidence(performance)
            
            recommendation = OptimizationRecommendation(
                recommendation_id=str(uuid.uuid4()),
                channel=channel,
                recommendation_type=recommendation_type,
                current_spend=current_spend,
                recommended_spend=recommended_spend,
                expected_impact=expected_impact,
                confidence=confidence,
                reasoning=reasoning,
                priority=priority,
                implementation_steps=implementation_steps,
                created_at=datetime.utcnow()
            )
            
            recommendations.append(recommendation)
        
        # Sort by priority
        priority_order = {"high": 3, "medium": 2, "low": 1}
        recommendations.sort(key=lambda x: priority_order.get(x.priority, 0), reverse=True)
        
        return recommendations
    
    def _generate_implementation_steps(self, recommendation_type: str, 
                                     channel: MarketingChannel) -> List[str]:
        """Generate implementation steps for recommendation"""
        steps = []
        
        if recommendation_type == "increase_budget":
            steps = [
                f"Increase {channel.value} budget allocation by 50%",
                "Monitor performance metrics closely",
                "Scale successful ad groups/campaigns",
                "Expand to similar audiences"
            ]
        
        elif recommendation_type == "reduce_budget":
            steps = [
                f"Reduce {channel.value} budget by 50%",
                "Pause underperforming campaigns",
                "Reallocate budget to better channels",
                "Analyze failure reasons"
            ]
        
        elif recommendation_type == "optimize_targeting":
            steps = [
                "Analyze audience segments performance",
                "Refine targeting parameters",
                "A/B test different messaging",
                "Improve landing page conversion"
            ]
        
        elif recommendation_type == "optimize_budget":
            steps = [
                "Increase budget by 20%",
                "Optimize bid strategies",
                "Expand high-performing keywords/audiences",
                "Test new creative variations"
            ]
        
        else:  # maintain_budget
            steps = [
                "Continue current strategy",
                "Monitor for performance changes",
                "Test minor optimizations",
                "Maintain current targeting"
            ]
        
        return steps
    
    def _calculate_recommendation_confidence(self, performance: ChannelPerformance) -> float:
        """Calculate confidence in recommendation based on data quality"""
        confidence = 0.5  # Base confidence
        
        # More conversions = higher confidence
        if performance.total_conversions >= 50:
            confidence += 0.3
        elif performance.total_conversions >= 20:
            confidence += 0.2
        elif performance.total_conversions >= 10:
            confidence += 0.1
        
        # Higher spend = more reliable data
        if performance.total_spend >= 10000:
            confidence += 0.2
        elif performance.total_spend >= 5000:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def analyze_customer_cohorts(self) -> Dict[str, Any]:
        """Analyze customer cohorts for LTV insights"""
        cohort_analysis = {
            "cohorts_by_channel": {},
            "overall_metrics": {
                "avg_ltv": 0,
                "avg_cac": 0,
                "avg_ltv_cac_ratio": 0,
                "avg_payback_period": 0
            },
            "trends": {
                "ltv_trend": "stable",
                "cac_trend": "increasing",
                "retention_trend": "improving"
            }
        }
        
        # Generate sample cohort data (would use actual customer data)
        for channel in MarketingChannel:
            if channel in self.channel_performance:
                performance = self.channel_performance[channel]
                
                # Create sample cohort
                cohort = CustomerCohort(
                    cohort_id=str(uuid.uuid4()),
                    acquisition_period="2024-01",
                    channel=channel,
                    initial_customers=max(10, performance.total_conversions),
                    revenue_by_month=self._generate_sample_revenue_curve(performance.avg_ltv),
                    retention_by_month=self._generate_sample_retention_curve(),
                    avg_cac=performance.avg_cac,
                    calculated_ltv=performance.avg_ltv
                )
                
                cohort_analysis["cohorts_by_channel"][channel.value] = cohort.to_dict()
        
        # Calculate overall metrics
        if self.channel_performance:
            total_ltv = sum(p.avg_ltv * p.total_conversions for p in self.channel_performance.values())
            total_cac = sum(p.avg_cac * p.total_conversions for p in self.channel_performance.values())
            total_customers = sum(p.total_conversions for p in self.channel_performance.values())
            
            if total_customers > 0:
                cohort_analysis["overall_metrics"]["avg_ltv"] = total_ltv / total_customers
                cohort_analysis["overall_metrics"]["avg_cac"] = total_cac / total_customers
                cohort_analysis["overall_metrics"]["avg_ltv_cac_ratio"] = (total_ltv / total_cac) if total_cac > 0 else 0
                cohort_analysis["overall_metrics"]["avg_payback_period"] = 8  # Placeholder
        
        return cohort_analysis
    
    def _generate_sample_revenue_curve(self, total_ltv: float) -> List[float]:
        """Generate sample monthly revenue curve"""
        # Typical SaaS revenue curve - higher in early months, declining over time
        months = 24
        curve = []
        
        for month in range(months):
            # Exponential decay with higher initial values
            monthly_revenue = (total_ltv / 6) * (0.8 ** month)  # Front-loaded
            curve.append(max(0, monthly_revenue))
        
        # Normalize to match total LTV
        total_generated = sum(curve)
        if total_generated > 0:
            normalizer = total_ltv / total_generated
            curve = [revenue * normalizer for revenue in curve]
        
        return curve
    
    def _generate_sample_retention_curve(self) -> List[float]:
        """Generate sample retention curve"""
        # Typical SaaS retention curve
        months = 24
        curve = []
        retention = 100.0
        
        for month in range(months):
            if month == 0:
                curve.append(retention)
            else:
                # Monthly churn rate around 5-8%
                churn_rate = 5 + (month * 0.1)  # Increasing churn over time
                retention *= (1 - churn_rate / 100)
                curve.append(max(0, retention))
        
        return curve
    
    def get_cac_analytics(self) -> Dict[str, Any]:
        """Get comprehensive CAC analytics"""
        analytics = {
            "campaigns": {campaign_id: campaign.to_dict() for campaign_id, campaign in self.campaigns.items()},
            "channel_performance": {channel.value: performance.to_dict() 
                                  for channel, performance in self.channel_performance.items()},
            "overall_metrics": self._calculate_overall_metrics(),
            "efficiency_rankings": self._rank_channels_by_efficiency(),
            "budget_recommendations": {},
            "trends": self._analyze_cac_trends()
        }
        
        return analytics
    
    def _calculate_overall_metrics(self) -> Dict[str, Any]:
        """Calculate overall CAC metrics"""
        total_spend = sum(campaign.spent for campaign in self.campaigns.values())
        total_conversions = sum(campaign.conversions for campaign in self.campaigns.values())
        total_revenue = sum(campaign.revenue_generated for campaign in self.campaigns.values())
        total_leads = sum(campaign.leads_generated for campaign in self.campaigns.values())
        
        return {
            "total_marketing_spend": total_spend,
            "total_conversions": total_conversions,
            "total_leads_generated": total_leads,
            "total_revenue_generated": total_revenue,
            "overall_cac": total_spend / total_conversions if total_conversions > 0 else 0,
            "overall_roi": ((total_revenue - total_spend) / total_spend) * 100 if total_spend > 0 else 0,
            "lead_to_customer_rate": (total_conversions / total_leads) * 100 if total_leads > 0 else 0,
            "cost_per_lead": total_spend / total_leads if total_leads > 0 else 0
        }
    
    def _rank_channels_by_efficiency(self) -> List[Dict[str, Any]]:
        """Rank channels by efficiency score"""
        rankings = []
        
        for channel, performance in self.channel_performance.items():
            efficiency_dict = performance.to_dict()
            rankings.append({
                "channel": channel.value,
                "efficiency_score": efficiency_dict["efficiency_score"],
                "ltv_cac_ratio": efficiency_dict["ltv_cac_ratio"],
                "roi": performance.roi,
                "conversion_rate": performance.conversion_rate
            })
        
        # Sort by efficiency score
        rankings.sort(key=lambda x: x["efficiency_score"], reverse=True)
        
        return rankings
    
    def _analyze_cac_trends(self) -> Dict[str, Any]:
        """Analyze CAC trends over time"""
        # This would analyze historical data - simplified for now
        return {
            "cac_trend": "increasing",
            "cac_change_percent": 15,
            "ltv_trend": "stable",
            "roi_trend": "declining",
            "conversion_rate_trend": "improving",
            "lead_quality_trend": "stable"
        }
    
    async def run_optimization_cycle(self, total_budget: float) -> Dict[str, Any]:
        """Run complete optimization cycle"""
        # Update channel performance
        await self._update_channel_performance()
        
        # Generate recommendations
        recommendations = await self.generate_optimization_recommendations()
        
        # Optimize budget allocation
        optimal_allocation = await self.optimize_budget_allocation(total_budget, OptimizationGoal.MAXIMIZE_ROI)
        
        # Analyze cohorts
        cohort_analysis = self.analyze_customer_cohorts()
        
        # Create optimization report
        optimization_report = {
            "optimization_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "total_budget": total_budget,
            "recommendations": [rec.to_dict() for rec in recommendations],
            "optimal_budget_allocation": {channel.value: budget for channel, budget in optimal_allocation.items()},
            "current_performance": {channel.value: perf.to_dict() for channel, perf in self.channel_performance.items()},
            "cohort_analysis": cohort_analysis,
            "summary": {
                "total_recommendations": len(recommendations),
                "high_priority_recommendations": len([r for r in recommendations if r.priority == "high"]),
                "expected_cac_reduction": statistics.mean([r.expected_impact.get("cac_reduction_percent", 0) 
                                                         for r in recommendations if "cac_reduction_percent" in r.expected_impact]),
                "expected_roi_improvement": statistics.mean([r.expected_impact.get("roi_improvement", 0) 
                                                           for r in recommendations if "roi_improvement" in r.expected_impact])
            }
        }
        
        # Store optimization history
        self.optimization_history.append(optimization_report)
        
        return optimization_report

# Global customer acquisition optimizer instance
cac_optimizer = CustomerAcquisitionOptimizer()

# Convenience functions
def create_campaign(campaign_data: Dict[str, Any]) -> str:
    """Create marketing campaign"""
    return cac_optimizer.create_campaign(campaign_data)

def update_campaign_metrics(campaign_id: str, metrics: Dict[str, Any]) -> bool:
    """Update campaign metrics"""
    return cac_optimizer.update_campaign_metrics(campaign_id, metrics)

async def optimize_budget_allocation(total_budget: float, goal: OptimizationGoal) -> Dict[MarketingChannel, float]:
    """Optimize budget allocation"""
    return await cac_optimizer.optimize_budget_allocation(total_budget, goal)

async def generate_cac_recommendations() -> List[OptimizationRecommendation]:
    """Generate CAC recommendations"""
    return await cac_optimizer.generate_optimization_recommendations()

def get_cac_analytics() -> Dict[str, Any]:
    """Get CAC analytics"""
    return cac_optimizer.get_cac_analytics()

async def run_optimization_cycle(total_budget: float) -> Dict[str, Any]:
    """Run optimization cycle"""
    return await cac_optimizer.run_optimization_cycle(total_budget)