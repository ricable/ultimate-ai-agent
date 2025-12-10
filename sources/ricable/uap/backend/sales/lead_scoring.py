# File: backend/sales/lead_scoring.py
"""
Lead Scoring System for UAP Platform

Provides advanced lead scoring using machine learning models,
behavioral analysis, and predictive analytics to identify
high-quality leads and optimize sales efforts.
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
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .pipeline_automation import sales_pipeline, LeadSource, CompanySize
from ..analytics.usage_analytics import usage_analytics

class ScoringModel(Enum):
    """Lead scoring model types"""
    RULE_BASED = "rule_based"
    MACHINE_LEARNING = "machine_learning"
    HYBRID = "hybrid"
    BEHAVIORAL = "behavioral"

class LeadQuality(Enum):
    """Lead quality categories"""
    HOT = "hot"          # 80-100 score
    WARM = "warm"        # 60-79 score
    COLD = "cold"        # 40-59 score
    UNQUALIFIED = "unqualified"  # 0-39 score

@dataclass
class ScoringFactor:
    """Individual scoring factor"""
    factor_id: str
    name: str
    category: str
    weight: float
    max_points: int
    description: str
    calculation_method: str
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "name": self.name,
            "category": self.category,
            "weight": self.weight,
            "max_points": self.max_points,
            "description": self.description,
            "calculation_method": self.calculation_method,
            "is_active": self.is_active
        }

@dataclass
class LeadScoreBreakdown:
    """Detailed lead score breakdown"""
    lead_id: str
    total_score: float
    quality_category: LeadQuality
    factor_scores: Dict[str, float]
    behavioral_score: float
    demographic_score: float
    firmographic_score: float
    engagement_score: float
    intent_score: float
    confidence: float
    last_calculated: datetime
    model_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lead_id": self.lead_id,
            "total_score": self.total_score,
            "quality_category": self.quality_category.value,
            "factor_scores": self.factor_scores,
            "behavioral_score": self.behavioral_score,
            "demographic_score": self.demographic_score,
            "firmographic_score": self.firmographic_score,
            "engagement_score": self.engagement_score,
            "intent_score": self.intent_score,
            "confidence": self.confidence,
            "last_calculated": self.last_calculated.isoformat(),
            "model_version": self.model_version
        }

@dataclass
class BehavioralEvent:
    """Behavioral tracking event"""
    event_id: str
    lead_id: str
    event_type: str
    event_data: Dict[str, Any]
    timestamp: datetime
    score_impact: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "lead_id": self.lead_id,
            "event_type": self.event_type,
            "event_data": self.event_data,
            "timestamp": self.timestamp.isoformat(),
            "score_impact": self.score_impact
        }

class LeadScoringEngine:
    """Advanced lead scoring engine"""
    
    def __init__(self):
        self.scoring_factors: Dict[str, ScoringFactor] = {}
        self.lead_scores: Dict[str, LeadScoreBreakdown] = {}
        self.behavioral_events: Dict[str, List[BehavioralEvent]] = defaultdict(list)
        self.ml_model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.label_encoders = {}
        self.lock = Lock()
        
        # Scoring configuration
        self.model_type = ScoringModel.HYBRID
        self.min_training_samples = 100
        self.model_version = "1.0"
        self.score_decay_days = 30  # How long behavioral scores remain relevant
        
        # Initialize scoring factors
        self._initialize_scoring_factors()
        
        # Performance tracking
        self.model_performance = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "last_trained": None
        }
    
    def _initialize_scoring_factors(self):
        """Initialize default scoring factors"""
        factors = [
            # Demographic factors
            {
                "name": "Job Title Relevance",
                "category": "demographic",
                "weight": 0.15,
                "max_points": 20,
                "description": "Score based on decision-making authority",
                "calculation_method": "title_keywords"
            },
            {
                "name": "Industry Fit",
                "category": "demographic",
                "weight": 0.10,
                "max_points": 15,
                "description": "How well the industry aligns with our target market",
                "calculation_method": "industry_matching"
            },
            
            # Firmographic factors
            {
                "name": "Company Size",
                "category": "firmographic",
                "weight": 0.20,
                "max_points": 25,
                "description": "Company size alignment with target market",
                "calculation_method": "size_scoring"
            },
            {
                "name": "Budget Authority",
                "category": "firmographic",
                "weight": 0.15,
                "max_points": 20,
                "description": "Stated or estimated budget capacity",
                "calculation_method": "budget_analysis"
            },
            {
                "name": "Technology Stack",
                "category": "firmographic",
                "weight": 0.10,
                "max_points": 15,
                "description": "Existing technology compatibility",
                "calculation_method": "tech_compatibility"
            },
            
            # Behavioral factors
            {
                "name": "Website Engagement",
                "category": "behavioral",
                "weight": 0.10,
                "max_points": 15,
                "description": "Level of engagement with our website",
                "calculation_method": "engagement_scoring"
            },
            {
                "name": "Content Interaction",
                "category": "behavioral",
                "weight": 0.08,
                "max_points": 12,
                "description": "Interaction with marketing content",
                "calculation_method": "content_scoring"
            },
            {
                "name": "Email Engagement",
                "category": "behavioral",
                "weight": 0.07,
                "max_points": 10,
                "description": "Email open rates and click-through rates",
                "calculation_method": "email_scoring"
            },
            
            # Intent factors
            {
                "name": "Pain Point Alignment",
                "category": "intent",
                "weight": 0.12,
                "max_points": 18,
                "description": "How well stated pain points align with our solutions",
                "calculation_method": "pain_point_matching"
            },
            {
                "name": "Timeline Urgency",
                "category": "intent",
                "weight": 0.08,
                "max_points": 12,
                "description": "Urgency of stated timeline",
                "calculation_method": "timeline_scoring"
            },
            {
                "name": "Solution Interest",
                "category": "intent",
                "weight": 0.10,
                "max_points": 15,
                "description": "Expressed interest in specific solutions",
                "calculation_method": "interest_matching"
            }
        ]
        
        for factor_data in factors:
            factor = ScoringFactor(
                factor_id=str(uuid.uuid4()),
                name=factor_data["name"],
                category=factor_data["category"],
                weight=factor_data["weight"],
                max_points=factor_data["max_points"],
                description=factor_data["description"],
                calculation_method=factor_data["calculation_method"]
            )
            self.scoring_factors[factor.factor_id] = factor
    
    def track_behavioral_event(self, lead_id: str, event_type: str, 
                              event_data: Dict[str, Any]) -> str:
        """Track a behavioral event for lead scoring"""
        event_id = str(uuid.uuid4())
        
        event = BehavioralEvent(
            event_id=event_id,
            lead_id=lead_id,
            event_type=event_type,
            event_data=event_data,
            timestamp=datetime.utcnow()
        )
        
        # Calculate immediate score impact
        event.score_impact = self._calculate_event_impact(event_type, event_data)
        
        with self.lock:
            self.behavioral_events[lead_id].append(event)
            
            # Keep only recent events (last 90 days)
            cutoff_date = datetime.utcnow() - timedelta(days=90)
            self.behavioral_events[lead_id] = [
                e for e in self.behavioral_events[lead_id]
                if e.timestamp >= cutoff_date
            ]
        
        # Trigger score recalculation
        asyncio.create_task(self.calculate_lead_score(lead_id))
        
        return event_id
    
    def _calculate_event_impact(self, event_type: str, event_data: Dict[str, Any]) -> float:
        """Calculate the scoring impact of a behavioral event"""
        impact_scores = {
            "page_view": 1.0,
            "document_download": 3.0,
            "video_watch": 2.0,
            "demo_request": 8.0,
            "pricing_page_view": 4.0,
            "contact_form_submit": 6.0,
            "email_open": 0.5,
            "email_click": 1.5,
            "webinar_attend": 5.0,
            "trial_signup": 10.0,
            "api_documentation_view": 3.0,
            "github_repo_view": 2.0,
            "case_study_view": 2.5,
            "integration_page_view": 3.5
        }
        
        base_score = impact_scores.get(event_type, 0.0)
        
        # Adjust based on event data
        if event_type == "page_view":
            time_on_page = event_data.get("time_on_page", 0)
            if time_on_page > 120:  # 2 minutes
                base_score *= 2
            elif time_on_page > 60:  # 1 minute
                base_score *= 1.5
        
        elif event_type == "video_watch":
            completion_rate = event_data.get("completion_rate", 0)
            base_score *= completion_rate
        
        elif event_type == "email_open":
            # Multiple opens within short time = higher engagement
            recent_opens = event_data.get("recent_opens", 1)
            base_score *= min(3, recent_opens)
        
        return base_score
    
    async def calculate_lead_score(self, lead_id: str) -> Optional[LeadScoreBreakdown]:
        """Calculate comprehensive lead score"""
        if lead_id not in sales_pipeline.leads:
            return None
        
        lead = sales_pipeline.leads[lead_id]
        
        # Calculate different score components
        demographic_score = self._calculate_demographic_score(lead)
        firmographic_score = self._calculate_firmographic_score(lead)
        behavioral_score = self._calculate_behavioral_score(lead_id)
        intent_score = self._calculate_intent_score(lead)
        engagement_score = self._calculate_engagement_score(lead_id)
        
        # Individual factor scores
        factor_scores = {}
        
        # Calculate each scoring factor
        for factor_id, factor in self.scoring_factors.items():
            if not factor.is_active:
                continue
            
            if factor.calculation_method == "title_keywords":
                score = self._score_job_title(lead.title)
            elif factor.calculation_method == "industry_matching":
                score = self._score_industry(lead.industry)
            elif factor.calculation_method == "size_scoring":
                score = self._score_company_size(lead.company_size)
            elif factor.calculation_method == "budget_analysis":
                score = self._score_budget(lead.budget)
            elif factor.calculation_method == "tech_compatibility":
                score = self._score_tech_compatibility(lead)
            elif factor.calculation_method == "engagement_scoring":
                score = behavioral_score * 0.4  # Portion of behavioral score
            elif factor.calculation_method == "content_scoring":
                score = self._score_content_interaction(lead_id)
            elif factor.calculation_method == "email_scoring":
                score = self._score_email_engagement(lead_id)
            elif factor.calculation_method == "pain_point_matching":
                score = self._score_pain_points(lead.pain_points)
            elif factor.calculation_method == "timeline_scoring":
                score = self._score_timeline(lead.timeline)
            elif factor.calculation_method == "interest_matching":
                score = self._score_interests(lead.interests)
            else:
                score = 0.0
            
            # Normalize to factor's max points
            normalized_score = min(factor.max_points, score * factor.max_points / 100)
            factor_scores[factor.name] = normalized_score
        
        # Calculate total weighted score
        total_score = 0.0
        for factor_id, factor in self.scoring_factors.items():
            if factor.is_active and factor.name in factor_scores:
                weighted_score = factor_scores[factor.name] * factor.weight
                total_score += weighted_score
        
        # Normalize to 0-100 scale
        max_possible_score = sum(f.max_points * f.weight for f in self.scoring_factors.values() if f.is_active)
        if max_possible_score > 0:
            total_score = (total_score / max_possible_score) * 100
        
        # Determine quality category
        if total_score >= 80:
            quality = LeadQuality.HOT
        elif total_score >= 60:
            quality = LeadQuality.WARM
        elif total_score >= 40:
            quality = LeadQuality.COLD
        else:
            quality = LeadQuality.UNQUALIFIED
        
        # Calculate confidence based on data completeness
        confidence = self._calculate_confidence(lead, lead_id)
        
        # Create score breakdown
        score_breakdown = LeadScoreBreakdown(
            lead_id=lead_id,
            total_score=total_score,
            quality_category=quality,
            factor_scores=factor_scores,
            behavioral_score=behavioral_score,
            demographic_score=demographic_score,
            firmographic_score=firmographic_score,
            engagement_score=engagement_score,
            intent_score=intent_score,
            confidence=confidence,
            last_calculated=datetime.utcnow(),
            model_version=self.model_version
        )
        
        # Store the breakdown
        with self.lock:
            self.lead_scores[lead_id] = score_breakdown
        
        # Update lead score in pipeline
        sales_pipeline.update_lead(lead_id, {"lead_score": total_score})
        
        return score_breakdown
    
    def _calculate_demographic_score(self, lead) -> float:
        """Calculate demographic score component"""
        score = 0.0
        
        # Job title scoring
        title_score = self._score_job_title(lead.title)
        score += title_score * 0.6
        
        # Industry scoring
        industry_score = self._score_industry(lead.industry)
        score += industry_score * 0.4
        
        return min(100, score)
    
    def _calculate_firmographic_score(self, lead) -> float:
        """Calculate firmographic score component"""
        score = 0.0
        
        # Company size scoring
        size_score = self._score_company_size(lead.company_size)
        score += size_score * 0.5
        
        # Budget scoring
        budget_score = self._score_budget(lead.budget)
        score += budget_score * 0.3
        
        # Tech compatibility
        tech_score = self._score_tech_compatibility(lead)
        score += tech_score * 0.2
        
        return min(100, score)
    
    def _calculate_behavioral_score(self, lead_id: str) -> float:
        """Calculate behavioral score component"""
        events = self.behavioral_events.get(lead_id, [])
        
        if not events:
            return 0.0
        
        # Calculate time-decay weighted score
        current_time = datetime.utcnow()
        total_score = 0.0
        
        for event in events:
            days_ago = (current_time - event.timestamp).days
            decay_factor = max(0, 1 - (days_ago / self.score_decay_days))
            
            weighted_impact = event.score_impact * decay_factor
            total_score += weighted_impact
        
        # Normalize to 0-100 scale (assuming max 50 points from behavioral)
        return min(100, total_score * 2)
    
    def _calculate_intent_score(self, lead) -> float:
        """Calculate intent score component"""
        score = 0.0
        
        # Pain point alignment
        pain_score = self._score_pain_points(lead.pain_points)
        score += pain_score * 0.5
        
        # Timeline urgency
        timeline_score = self._score_timeline(lead.timeline)
        score += timeline_score * 0.3
        
        # Solution interest
        interest_score = self._score_interests(lead.interests)
        score += interest_score * 0.2
        
        return min(100, score)
    
    def _calculate_engagement_score(self, lead_id: str) -> float:
        """Calculate engagement score component"""
        events = self.behavioral_events.get(lead_id, [])
        
        if not events:
            return 0.0
        
        # Count different types of engagement
        engagement_types = set(event.event_type for event in events)
        engagement_frequency = len(events)
        
        # Recent activity bonus
        recent_events = [e for e in events if (datetime.utcnow() - e.timestamp).days <= 7]
        recent_activity_bonus = len(recent_events) * 5
        
        # Calculate engagement score
        variety_score = len(engagement_types) * 10  # Up to 100 for 10 different types
        frequency_score = min(50, engagement_frequency * 2)  # Up to 50 for frequent engagement
        
        total_score = variety_score + frequency_score + recent_activity_bonus
        return min(100, total_score)
    
    def _score_job_title(self, title: str) -> float:
        """Score job title based on decision-making authority"""
        if not title:
            return 0.0
        
        title_lower = title.lower()
        
        # C-level executives
        if any(keyword in title_lower for keyword in ['ceo', 'cto', 'cio', 'chief']):
            return 90.0
        
        # VP level
        if any(keyword in title_lower for keyword in ['vp', 'vice president', 'head of']):
            return 80.0
        
        # Director level
        if any(keyword in title_lower for keyword in ['director', 'lead']):
            return 70.0
        
        # Manager level
        if any(keyword in title_lower for keyword in ['manager', 'supervisor']):
            return 60.0
        
        # Technical roles
        if any(keyword in title_lower for keyword in ['engineer', 'developer', 'architect', 'analyst']):
            return 50.0
        
        # Other roles
        return 30.0
    
    def _score_industry(self, industry: str) -> float:
        """Score industry fit"""
        if not industry:
            return 50.0  # Neutral score
        
        industry_lower = industry.lower()
        
        # High-fit industries
        high_fit = ['technology', 'software', 'ai', 'machine learning', 'fintech', 
                   'healthcare', 'finance', 'consulting', 'saas']
        
        if any(keyword in industry_lower for keyword in high_fit):
            return 90.0
        
        # Medium-fit industries
        medium_fit = ['manufacturing', 'retail', 'education', 'media', 'telecom']
        
        if any(keyword in industry_lower for keyword in medium_fit):
            return 70.0
        
        # Lower-fit industries
        return 40.0
    
    def _score_company_size(self, company_size: Optional[CompanySize]) -> float:
        """Score company size alignment"""
        if not company_size:
            return 50.0
        
        size_scores = {
            CompanySize.STARTUP: 60.0,      # Good fit but budget constraints
            CompanySize.SMALL_BUSINESS: 75.0,  # Good balance
            CompanySize.MID_MARKET: 90.0,   # Sweet spot
            CompanySize.ENTERPRISE: 80.0    # High value but complex sales
        }
        
        return size_scores.get(company_size, 50.0)
    
    def _score_budget(self, budget: Optional[float]) -> float:
        """Score budget capacity"""
        if not budget:
            return 30.0  # Unknown budget gets low score
        
        if budget >= 100000:
            return 95.0
        elif budget >= 50000:
            return 85.0
        elif budget >= 25000:
            return 75.0
        elif budget >= 10000:
            return 65.0
        elif budget >= 5000:
            return 50.0
        else:
            return 30.0
    
    def _score_tech_compatibility(self, lead) -> float:
        """Score technology stack compatibility"""
        # This would integrate with actual tech stack detection
        # For now, use interests and industry as proxy
        
        score = 50.0  # Base score
        
        tech_interests = ['api', 'integration', 'automation', 'ai', 'ml', 'workflow']
        
        if lead.interests:
            matching_interests = sum(1 for interest in lead.interests 
                                   if any(tech in interest.lower() for tech in tech_interests))
            score += matching_interests * 10
        
        return min(100, score)
    
    def _score_content_interaction(self, lead_id: str) -> float:
        """Score content interaction behavior"""
        events = self.behavioral_events.get(lead_id, [])
        
        content_events = [e for e in events if e.event_type in 
                         ['document_download', 'case_study_view', 'video_watch', 'webinar_attend']]
        
        if not content_events:
            return 0.0
        
        return min(100, len(content_events) * 15)
    
    def _score_email_engagement(self, lead_id: str) -> float:
        """Score email engagement behavior"""
        events = self.behavioral_events.get(lead_id, [])
        
        email_events = [e for e in events if e.event_type in ['email_open', 'email_click']]
        
        if not email_events:
            return 0.0
        
        opens = len([e for e in email_events if e.event_type == 'email_open'])
        clicks = len([e for e in email_events if e.event_type == 'email_click'])
        
        # Clicks are more valuable than opens
        score = (opens * 5) + (clicks * 15)
        
        return min(100, score)
    
    def _score_pain_points(self, pain_points: List[str]) -> float:
        """Score pain point alignment"""
        if not pain_points:
            return 0.0
        
        relevant_pain_points = [
            'automation', 'efficiency', 'integration', 'scaling', 'ai', 'workflow',
            'manual processes', 'data silos', 'complexity', 'time-consuming'
        ]
        
        matching_points = 0
        for pain in pain_points:
            pain_lower = pain.lower()
            if any(keyword in pain_lower for keyword in relevant_pain_points):
                matching_points += 1
        
        return min(100, matching_points * 25)
    
    def _score_timeline(self, timeline: str) -> float:
        """Score timeline urgency"""
        if not timeline:
            return 30.0
        
        timeline_lower = timeline.lower()
        
        if any(keyword in timeline_lower for keyword in ['immediate', 'asap', 'urgent']):
            return 95.0
        elif any(keyword in timeline_lower for keyword in ['month', '30 days']):
            return 85.0
        elif any(keyword in timeline_lower for keyword in ['quarter', '3 months']):
            return 70.0
        elif any(keyword in timeline_lower for keyword in ['6 months', 'half year']):
            return 55.0
        elif any(keyword in timeline_lower for keyword in ['year', '12 months']):
            return 40.0
        else:
            return 30.0
    
    def _score_interests(self, interests: List[str]) -> float:
        """Score solution interest alignment"""
        if not interests:
            return 0.0
        
        relevant_interests = [
            'ai agents', 'automation', 'api', 'integration', 'workflow',
            'machine learning', 'copilot', 'chatbot', 'assistant'
        ]
        
        matching_interests = 0
        for interest in interests:
            interest_lower = interest.lower()
            if any(keyword in interest_lower for keyword in relevant_interests):
                matching_interests += 1
        
        return min(100, matching_interests * 20)
    
    def _calculate_confidence(self, lead, lead_id: str) -> float:
        """Calculate confidence in the lead score"""
        confidence = 0.0
        
        # Data completeness (40% of confidence)
        completeness_score = 0
        fields_to_check = ['title', 'company', 'industry', 'company_size', 'budget', 'timeline']
        
        for field in fields_to_check:
            if getattr(lead, field):
                completeness_score += 1
        
        confidence += (completeness_score / len(fields_to_check)) * 0.4
        
        # Behavioral data availability (30% of confidence)
        behavioral_events = self.behavioral_events.get(lead_id, [])
        if len(behavioral_events) >= 5:
            confidence += 0.3
        elif len(behavioral_events) >= 2:
            confidence += 0.2
        elif len(behavioral_events) >= 1:
            confidence += 0.1
        
        # Pain points and interests (20% of confidence)
        if lead.pain_points and len(lead.pain_points) >= 2:
            confidence += 0.1
        if lead.interests and len(lead.interests) >= 2:
            confidence += 0.1
        
        # Time since creation (10% of confidence)
        days_since_creation = (datetime.utcnow() - lead.created_at).days
        if days_since_creation >= 7:
            confidence += 0.1
        elif days_since_creation >= 3:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    async def train_ml_model(self) -> Dict[str, float]:
        """Train machine learning model for lead scoring"""
        if not SKLEARN_AVAILABLE:
            return {"error": "sklearn not available"}
        
        # Collect training data
        training_data = []
        labels = []
        
        for lead_id, score_breakdown in self.lead_scores.items():
            if lead_id in sales_pipeline.leads:
                lead = sales_pipeline.leads[lead_id]
                
                # Create feature vector
                features = self._extract_ml_features(lead, lead_id)
                
                # Use quality category as label
                label = 1 if score_breakdown.quality_category in [LeadQuality.HOT, LeadQuality.WARM] else 0
                
                training_data.append(features)
                labels.append(label)
        
        if len(training_data) < self.min_training_samples:
            return {"error": f"Insufficient training data: {len(training_data)} samples"}
        
        # Prepare data
        X = np.array(training_data)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ml_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.ml_model.predict(X_test_scaled)
        
        performance = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "training_samples": len(training_data),
            "last_trained": datetime.utcnow().isoformat()
        }
        
        self.model_performance.update(performance)
        
        return performance
    
    def _extract_ml_features(self, lead, lead_id: str) -> List[float]:
        """Extract features for machine learning model"""
        features = []
        
        # Demographic features
        features.append(self._score_job_title(lead.title))
        features.append(self._score_industry(lead.industry))
        
        # Firmographic features
        features.append(self._score_company_size(lead.company_size))
        features.append(self._score_budget(lead.budget))
        
        # Intent features
        features.append(len(lead.pain_points) if lead.pain_points else 0)
        features.append(len(lead.interests) if lead.interests else 0)
        features.append(self._score_timeline(lead.timeline))
        
        # Behavioral features
        behavioral_events = self.behavioral_events.get(lead_id, [])
        features.append(len(behavioral_events))
        features.append(len(set(e.event_type for e in behavioral_events)))
        
        # Recent activity
        recent_events = [e for e in behavioral_events 
                        if (datetime.utcnow() - e.timestamp).days <= 7]
        features.append(len(recent_events))
        
        # Source scoring
        source_scores = {
            LeadSource.REFERRAL: 90,
            LeadSource.INBOUND: 80,
            LeadSource.CONTENT_MARKETING: 70,
            LeadSource.WEBSITE: 60,
            LeadSource.SOCIAL_MEDIA: 50,
            LeadSource.PAID_ADVERTISING: 40,
            LeadSource.COLD_OUTREACH: 30
        }
        features.append(source_scores.get(lead.source, 50))
        
        return features
    
    def get_scoring_insights(self) -> Dict[str, Any]:
        """Get insights about lead scoring performance"""
        insights = {
            "total_scored_leads": len(self.lead_scores),
            "quality_distribution": defaultdict(int),
            "avg_scores_by_source": defaultdict(list),
            "avg_scores_by_industry": defaultdict(list),
            "behavioral_engagement_stats": {
                "total_events": 0,
                "avg_events_per_lead": 0,
                "most_common_events": defaultdict(int)
            },
            "model_performance": self.model_performance,
            "scoring_factors": {f.name: f.to_dict() for f in self.scoring_factors.values()}
        }
        
        # Quality distribution
        for score_breakdown in self.lead_scores.values():
            insights["quality_distribution"][score_breakdown.quality_category.value] += 1
        
        # Score analysis by source and industry
        for lead_id, score_breakdown in self.lead_scores.items():
            if lead_id in sales_pipeline.leads:
                lead = sales_pipeline.leads[lead_id]
                
                insights["avg_scores_by_source"][lead.source.value].append(score_breakdown.total_score)
                
                if lead.industry:
                    insights["avg_scores_by_industry"][lead.industry].append(score_breakdown.total_score)
        
        # Calculate averages
        for source, scores in insights["avg_scores_by_source"].items():
            insights["avg_scores_by_source"][source] = statistics.mean(scores)
        
        for industry, scores in insights["avg_scores_by_industry"].items():
            insights["avg_scores_by_industry"][industry] = statistics.mean(scores)
        
        # Behavioral engagement stats
        total_events = sum(len(events) for events in self.behavioral_events.values())
        insights["behavioral_engagement_stats"]["total_events"] = total_events
        
        if self.behavioral_events:
            insights["behavioral_engagement_stats"]["avg_events_per_lead"] = total_events / len(self.behavioral_events)
        
        # Most common events
        for events in self.behavioral_events.values():
            for event in events:
                insights["behavioral_engagement_stats"]["most_common_events"][event.event_type] += 1
        
        return insights
    
    def get_lead_recommendations(self, lead_id: str) -> List[Dict[str, Any]]:
        """Get recommendations for improving lead score"""
        if lead_id not in self.lead_scores:
            return [{"message": "Lead not found or not scored yet"}]
        
        score_breakdown = self.lead_scores[lead_id]
        recommendations = []
        
        # Low score recommendations
        if score_breakdown.total_score < 40:
            recommendations.append({
                "type": "qualification",
                "priority": "high",
                "message": "Lead appears unqualified - consider moving to nurture campaign",
                "actions": ["Verify contact information", "Confirm budget authority", "Reassess fit"]
            })
        
        # Behavioral engagement recommendations
        if score_breakdown.behavioral_score < 20:
            recommendations.append({
                "type": "engagement",
                "priority": "medium",
                "message": "Low behavioral engagement - increase touchpoints",
                "actions": ["Send relevant content", "Invite to webinar", "Schedule demo"]
            })
        
        # Missing information recommendations
        if score_breakdown.confidence < 0.7:
            recommendations.append({
                "type": "data_completion",
                "priority": "medium",
                "message": "Incomplete lead data affecting score confidence",
                "actions": ["Gather missing company information", "Clarify budget and timeline", "Understand pain points"]
            })
        
        # High potential recommendations
        if score_breakdown.total_score >= 80:
            recommendations.append({
                "type": "acceleration",
                "priority": "urgent",
                "message": "High-quality lead - accelerate sales process",
                "actions": ["Immediate outreach", "Schedule executive meeting", "Prepare custom proposal"]
            })
        
        return recommendations

# Global lead scoring engine instance
lead_scoring_engine = LeadScoringEngine()

# Convenience functions
def track_behavioral_event(lead_id: str, event_type: str, event_data: Dict[str, Any]) -> str:
    """Track behavioral event"""
    return lead_scoring_engine.track_behavioral_event(lead_id, event_type, event_data)

async def calculate_lead_score(lead_id: str) -> Optional[LeadScoreBreakdown]:
    """Calculate lead score"""
    return await lead_scoring_engine.calculate_lead_score(lead_id)

def get_scoring_insights() -> Dict[str, Any]:
    """Get scoring insights"""
    return lead_scoring_engine.get_scoring_insights()

def get_lead_recommendations(lead_id: str) -> List[Dict[str, Any]]:
    """Get lead recommendations"""
    return lead_scoring_engine.get_lead_recommendations(lead_id)

async def train_scoring_model() -> Dict[str, float]:
    """Train ML scoring model"""
    return await lead_scoring_engine.train_ml_model()