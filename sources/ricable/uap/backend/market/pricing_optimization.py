# File: backend/market/pricing_optimization.py
"""
Pricing Optimization and Revenue Forecasting System for UAP Platform

Provides dynamic pricing optimization, revenue forecasting,
competitive pricing analysis, and pricing strategy recommendations.
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
    from sklearn.linear_model import LinearRegression, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .trend_analysis import market_intelligence
from ..sales.pipeline_automation import sales_pipeline
from ..analytics.predictive_analytics import predictive_analytics

class PricingStrategy(Enum):
    """Pricing strategy types"""
    VALUE_BASED = "value_based"
    COMPETITIVE = "competitive"
    COST_PLUS = "cost_plus"
    PENETRATION = "penetration"
    SKIMMING = "skimming"
    DYNAMIC = "dynamic"
    FREEMIUM = "freemium"
    USAGE_BASED = "usage_based"

class PricingModel(Enum):
    """Pricing model types"""
    SUBSCRIPTION = "subscription"
    PAY_PER_USE = "pay_per_use"
    TIERED = "tiered"
    FLAT_RATE = "flat_rate"
    HYBRID = "hybrid"
    CUSTOM = "custom"

class RevenueStream(Enum):
    """Revenue stream types"""
    SUBSCRIPTION_FEES = "subscription_fees"
    USAGE_FEES = "usage_fees"
    SETUP_FEES = "setup_fees"
    SUPPORT_FEES = "support_fees"
    TRAINING_FEES = "training_fees"
    CUSTOM_DEVELOPMENT = "custom_development"
    MARKETPLACE_COMMISSION = "marketplace_commission"

@dataclass
class PricingTier:
    """Pricing tier definition"""
    tier_id: str
    name: str
    price: float
    billing_period: str  # monthly, yearly, etc.
    features: List[str]
    limits: Dict[str, Any]
    target_segment: str
    discount_percent: float = 0.0
    is_active: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier_id": self.tier_id,
            "name": self.name,
            "price": self.price,
            "billing_period": self.billing_period,
            "features": self.features,
            "limits": self.limits,
            "target_segment": self.target_segment,
            "discount_percent": self.discount_percent,
            "effective_price": self.price * (1 - self.discount_percent / 100),
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class PriceTestResult:
    """A/B price test result"""
    test_id: str
    test_name: str
    control_price: float
    variant_price: float
    control_conversions: int
    variant_conversions: int
    control_revenue: float
    variant_revenue: float
    statistical_significance: float
    winner: str  # control, variant, or inconclusive
    confidence_level: float
    test_duration_days: int
    created_at: datetime
    ended_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        control_rate = (self.control_conversions / max(1, self.control_conversions + 100)) * 100
        variant_rate = (self.variant_conversions / max(1, self.variant_conversions + 100)) * 100
        
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "control_price": self.control_price,
            "variant_price": self.variant_price,
            "control_conversions": self.control_conversions,
            "variant_conversions": self.variant_conversions,
            "control_conversion_rate": control_rate,
            "variant_conversion_rate": variant_rate,
            "control_revenue": self.control_revenue,
            "variant_revenue": self.variant_revenue,
            "revenue_lift": ((self.variant_revenue - self.control_revenue) / max(1, self.control_revenue)) * 100,
            "statistical_significance": self.statistical_significance,
            "winner": self.winner,
            "confidence_level": self.confidence_level,
            "test_duration_days": self.test_duration_days,
            "created_at": self.created_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None
        }

@dataclass
class RevenueForcast:
    """Revenue forecast result"""
    forecast_id: str
    period_start: datetime
    period_end: datetime
    forecasted_revenue: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    revenue_breakdown: Dict[str, float]
    assumptions: Dict[str, Any]
    model_accuracy: float
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "forecast_id": self.forecast_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "period_days": (self.period_end - self.period_start).days,
            "forecasted_revenue": self.forecasted_revenue,
            "confidence_interval_lower": self.confidence_interval_lower,
            "confidence_interval_upper": self.confidence_interval_upper,
            "revenue_breakdown": self.revenue_breakdown,
            "assumptions": self.assumptions,
            "model_accuracy": self.model_accuracy,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class PricingRecommendation:
    """Pricing optimization recommendation"""
    recommendation_id: str
    recommendation_type: str
    current_price: float
    recommended_price: float
    expected_impact: Dict[str, float]  # revenue, conversion_rate, etc.
    confidence: float
    reasoning: List[str]
    implementation_effort: str  # low, medium, high
    priority: str  # high, medium, low
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "recommendation_type": self.recommendation_type,
            "current_price": self.current_price,
            "recommended_price": self.recommended_price,
            "price_change_percent": ((self.recommended_price - self.current_price) / self.current_price) * 100,
            "expected_impact": self.expected_impact,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "implementation_effort": self.implementation_effort,
            "priority": self.priority,
            "created_at": self.created_at.isoformat()
        }

class PricingOptimizer:
    """Pricing optimization engine"""
    
    def __init__(self):
        self.pricing_tiers: Dict[str, PricingTier] = {}
        self.price_tests: Dict[str, PriceTestResult] = {}
        self.revenue_forecasts: Dict[str, RevenueForcast] = {}
        self.pricing_history: deque = deque(maxlen=1000)
        self.customer_price_sensitivity: Dict[str, float] = {}
        self.lock = Lock()
        
        # ML models
        self.demand_model = None
        self.revenue_model = None
        self.churn_model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
        # Pricing configuration
        self.min_price_change_percent = 5.0  # Minimum price change for optimization
        self.max_price_change_percent = 30.0  # Maximum price change per optimization
        self.price_test_min_sample_size = 100
        self.confidence_threshold = 0.95
        
        # Initialize default pricing tiers
        self._initialize_default_tiers()
        
        # Revenue tracking
        self.revenue_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
    def _initialize_default_tiers(self):
        """Initialize default pricing tiers"""
        default_tiers = [
            {
                "name": "Starter",
                "price": 29.0,
                "billing_period": "monthly",
                "features": ["Basic AI Agents", "API Access", "Community Support"],
                "limits": {"agents": 3, "requests_per_month": 10000, "storage_gb": 5},
                "target_segment": "individual_developers"
            },
            {
                "name": "Professional",
                "price": 99.0,
                "billing_period": "monthly",
                "features": ["Advanced AI Agents", "Multiple Frameworks", "Priority Support", "Analytics"],
                "limits": {"agents": 10, "requests_per_month": 50000, "storage_gb": 50},
                "target_segment": "small_teams"
            },
            {
                "name": "Business",
                "price": 299.0,
                "billing_period": "monthly",
                "features": ["Unlimited Agents", "Custom Integrations", "Advanced Analytics", "SLA"],
                "limits": {"agents": -1, "requests_per_month": 250000, "storage_gb": 500},
                "target_segment": "mid_market"
            },
            {
                "name": "Enterprise",
                "price": 999.0,
                "billing_period": "monthly",
                "features": ["White-label", "Dedicated Support", "Custom Development", "On-premise"],
                "limits": {"agents": -1, "requests_per_month": -1, "storage_gb": -1},
                "target_segment": "enterprise"
            }
        ]
        
        for tier_data in default_tiers:
            tier = PricingTier(
                tier_id=str(uuid.uuid4()),
                name=tier_data["name"],
                price=tier_data["price"],
                billing_period=tier_data["billing_period"],
                features=tier_data["features"],
                limits=tier_data["limits"],
                target_segment=tier_data["target_segment"]
            )
            self.pricing_tiers[tier.tier_id] = tier
    
    def add_pricing_tier(self, tier_data: Dict[str, Any]) -> str:
        """Add a new pricing tier"""
        tier_id = str(uuid.uuid4())
        
        tier = PricingTier(
            tier_id=tier_id,
            name=tier_data["name"],
            price=tier_data["price"],
            billing_period=tier_data["billing_period"],
            features=tier_data["features"],
            limits=tier_data["limits"],
            target_segment=tier_data["target_segment"],
            discount_percent=tier_data.get("discount_percent", 0.0)
        )
        
        with self.lock:
            self.pricing_tiers[tier_id] = tier
        
        # Log pricing change
        self._log_pricing_change("tier_added", {"tier_id": tier_id, "tier_data": tier.to_dict()})
        
        return tier_id
    
    def update_pricing_tier(self, tier_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing pricing tier"""
        with self.lock:
            if tier_id not in self.pricing_tiers:
                return False
            
            tier = self.pricing_tiers[tier_id]
            old_price = tier.price
            
            # Update fields
            for field, value in updates.items():
                if hasattr(tier, field):
                    setattr(tier, field, value)
            
            # Log price changes
            if tier.price != old_price:
                change_percent = ((tier.price - old_price) / old_price) * 100
                self._log_pricing_change("price_change", {
                    "tier_id": tier_id,
                    "old_price": old_price,
                    "new_price": tier.price,
                    "change_percent": change_percent
                })
        
        return True
    
    def _log_pricing_change(self, change_type: str, data: Dict[str, Any]):
        """Log pricing changes for analysis"""
        log_entry = {
            "timestamp": datetime.utcnow(),
            "change_type": change_type,
            "data": data
        }
        
        self.pricing_history.append(log_entry)
    
    def start_price_test(self, test_config: Dict[str, Any]) -> str:
        """Start an A/B price test"""
        test_id = str(uuid.uuid4())
        
        test = PriceTestResult(
            test_id=test_id,
            test_name=test_config["test_name"],
            control_price=test_config["control_price"],
            variant_price=test_config["variant_price"],
            control_conversions=0,
            variant_conversions=0,
            control_revenue=0.0,
            variant_revenue=0.0,
            statistical_significance=0.0,
            winner="inconclusive",
            confidence_level=0.0,
            test_duration_days=test_config.get("duration_days", 30),
            created_at=datetime.utcnow()
        )
        
        with self.lock:
            self.price_tests[test_id] = test
        
        return test_id
    
    def record_price_test_conversion(self, test_id: str, variant: str, 
                                   conversion_value: float) -> bool:
        """Record a conversion for a price test"""
        with self.lock:
            if test_id not in self.price_tests:
                return False
            
            test = self.price_tests[test_id]
            
            if variant == "control":
                test.control_conversions += 1
                test.control_revenue += conversion_value
            elif variant == "variant":
                test.variant_conversions += 1
                test.variant_revenue += conversion_value
            else:
                return False
            
            # Update statistical analysis
            self._update_test_analysis(test)
            
        return True
    
    def _update_test_analysis(self, test: PriceTestResult):
        """Update statistical analysis for price test"""
        # Simple statistical significance calculation
        # In production, would use more sophisticated statistical tests
        
        total_control = test.control_conversions + 100  # Assuming 100 visitors for simplicity
        total_variant = test.variant_conversions + 100
        
        if total_control < 50 or total_variant < 50:
            test.statistical_significance = 0.0
            test.confidence_level = 0.0
            return
        
        control_rate = test.control_conversions / total_control
        variant_rate = test.variant_conversions / total_variant
        
        # Simplified z-test calculation
        pooled_rate = (test.control_conversions + test.variant_conversions) / (total_control + total_variant)
        standard_error = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/total_control + 1/total_variant))
        
        if standard_error > 0:
            z_score = abs(variant_rate - control_rate) / standard_error
            
            # Convert z-score to confidence level (simplified)
            if z_score >= 2.58:  # 99% confidence
                test.confidence_level = 0.99
                test.statistical_significance = 0.99
            elif z_score >= 1.96:  # 95% confidence
                test.confidence_level = 0.95
                test.statistical_significance = 0.95
            elif z_score >= 1.65:  # 90% confidence
                test.confidence_level = 0.90
                test.statistical_significance = 0.90
            else:
                test.confidence_level = z_score / 1.96 * 0.95
                test.statistical_significance = test.confidence_level
        
        # Determine winner
        if test.confidence_level >= self.confidence_threshold:
            if test.variant_revenue > test.control_revenue:
                test.winner = "variant"
            elif test.control_revenue > test.variant_revenue:
                test.winner = "control"
            else:
                test.winner = "inconclusive"
        else:
            test.winner = "inconclusive"
    
    async def optimize_pricing(self, tier_id: str) -> Optional[PricingRecommendation]:
        """Generate pricing optimization recommendation"""
        if tier_id not in self.pricing_tiers:
            return None
        
        tier = self.pricing_tiers[tier_id]
        current_price = tier.price
        
        # Analyze demand elasticity
        elasticity = await self._calculate_price_elasticity(tier_id)
        
        # Analyze competitive pricing
        competitive_analysis = await self._analyze_competitive_pricing(tier.target_segment)
        
        # Analyze customer willingness to pay
        willingness_to_pay = await self._analyze_willingness_to_pay(tier.target_segment)
        
        # Generate recommendation based on analysis
        recommendation = self._generate_pricing_recommendation(
            tier, elasticity, competitive_analysis, willingness_to_pay
        )
        
        return recommendation
    
    async def _calculate_price_elasticity(self, tier_id: str) -> float:
        """Calculate price elasticity of demand"""
        # Analyze historical pricing changes and their impact on demand
        price_changes = [entry for entry in self.pricing_history 
                        if entry["change_type"] == "price_change" and 
                        entry["data"].get("tier_id") == tier_id]
        
        if len(price_changes) < 2:
            return -1.0  # Default elasticity assumption
        
        # Calculate elasticity from recent price changes
        # This is a simplified calculation - in production would use more sophisticated methods
        elasticities = []
        
        for i in range(1, len(price_changes)):
            prev_change = price_changes[i-1]
            curr_change = price_changes[i]
            
            price_change_percent = curr_change["data"]["change_percent"]
            
            # Would need demand data here - using placeholder
            # demand_change_percent = self._get_demand_change(prev_change, curr_change)
            demand_change_percent = -price_change_percent * 0.8  # Placeholder
            
            if price_change_percent != 0:
                elasticity = demand_change_percent / price_change_percent
                elasticities.append(elasticity)
        
        return statistics.mean(elasticities) if elasticities else -1.0
    
    async def _analyze_competitive_pricing(self, target_segment: str) -> Dict[str, Any]:
        """Analyze competitive pricing for target segment"""
        # Get competitive analysis from market intelligence
        competitive_data = await market_intelligence.get_competitive_analysis()
        
        # Extract pricing information (this would be enhanced with real competitive data)
        segment_pricing = {
            "individual_developers": {"min": 0, "avg": 25, "max": 50},
            "small_teams": {"min": 50, "avg": 150, "max": 300},
            "mid_market": {"min": 200, "avg": 500, "max": 1000},
            "enterprise": {"min": 1000, "avg": 2500, "max": 10000}
        }
        
        pricing_data = segment_pricing.get(target_segment, {"min": 0, "avg": 100, "max": 500})
        
        return {
            "min_market_price": pricing_data["min"],
            "avg_market_price": pricing_data["avg"],
            "max_market_price": pricing_data["max"],
            "competitive_position": "competitive",  # Would calculate based on actual data
            "market_share_leaders": ["OpenAI", "Anthropic", "Google"],
            "pricing_trends": "stable"  # Would analyze from trend data
        }
    
    async def _analyze_willingness_to_pay(self, target_segment: str) -> Dict[str, Any]:
        """Analyze customer willingness to pay"""
        # Analyze sales pipeline data for price sensitivity
        pipeline_overview = sales_pipeline.get_pipeline_overview()
        
        # Get leads in target segment
        segment_leads = []
        for lead in sales_pipeline.leads.values():
            if self._match_target_segment(lead, target_segment):
                segment_leads.append(lead)
        
        # Analyze budget data
        budgets = [lead.budget for lead in segment_leads if lead.budget]
        
        if budgets:
            willingness_data = {
                "min_budget": min(budgets),
                "avg_budget": statistics.mean(budgets),
                "max_budget": max(budgets),
                "median_budget": statistics.median(budgets),
                "sample_size": len(budgets)
            }
        else:
            # Default assumptions based on segment
            segment_budgets = {
                "individual_developers": {"min": 10, "avg": 50, "max": 200},
                "small_teams": {"min": 100, "avg": 300, "max": 1000},
                "mid_market": {"min": 500, "avg": 1500, "max": 5000},
                "enterprise": {"min": 2000, "avg": 10000, "max": 50000}
            }
            
            willingness_data = segment_budgets.get(target_segment, {"min": 50, "avg": 200, "max": 1000})
            willingness_data["sample_size"] = 0
        
        return willingness_data
    
    def _match_target_segment(self, lead, target_segment: str) -> bool:
        """Check if lead matches target segment"""
        if target_segment == "individual_developers":
            return not lead.company_size or lead.company_size.value == "startup"
        elif target_segment == "small_teams":
            return lead.company_size and lead.company_size.value in ["startup", "small"]
        elif target_segment == "mid_market":
            return lead.company_size and lead.company_size.value == "mid_market"
        elif target_segment == "enterprise":
            return lead.company_size and lead.company_size.value == "enterprise"
        return False
    
    def _generate_pricing_recommendation(self, tier: PricingTier, elasticity: float,
                                       competitive_analysis: Dict[str, Any],
                                       willingness_to_pay: Dict[str, Any]) -> PricingRecommendation:
        """Generate pricing recommendation based on analysis"""
        current_price = tier.price
        market_avg = competitive_analysis["avg_market_price"]
        customer_avg_budget = willingness_to_pay["avg_budget"]
        
        reasoning = []
        
        # Price vs market analysis
        if current_price < market_avg * 0.8:
            # Significantly below market - consider increase
            recommended_price = min(market_avg * 0.9, current_price * 1.2)
            reasoning.append("Current price significantly below market average")
            recommendation_type = "price_increase"
            
        elif current_price > market_avg * 1.2:
            # Significantly above market - consider decrease
            recommended_price = max(market_avg * 1.1, current_price * 0.9)
            reasoning.append("Current price significantly above market average")
            recommendation_type = "price_decrease"
            
        else:
            # Within market range - optimize based on elasticity and willingness to pay
            if elasticity > -0.5:  # Low elasticity - can increase price
                recommended_price = min(customer_avg_budget * 0.8, current_price * 1.1)
                reasoning.append("Low price elasticity suggests room for price increase")
                recommendation_type = "elasticity_optimization"
            elif elasticity < -2.0:  # High elasticity - consider decrease
                recommended_price = max(current_price * 0.9, market_avg * 0.8)
                reasoning.append("High price elasticity suggests price reduction could increase volume")
                recommendation_type = "elasticity_optimization"
            else:
                # Maintain current price
                recommended_price = current_price
                reasoning.append("Current price is optimally positioned")
                recommendation_type = "maintain_price"
        
        # Calculate expected impact
        price_change_percent = ((recommended_price - current_price) / current_price) * 100
        expected_demand_change = -price_change_percent * abs(elasticity)
        expected_revenue_change = price_change_percent + expected_demand_change
        
        expected_impact = {
            "revenue_change_percent": expected_revenue_change,
            "demand_change_percent": expected_demand_change,
            "price_change_percent": price_change_percent
        }
        
        # Determine confidence and priority
        data_quality_score = min(1.0, willingness_to_pay.get("sample_size", 0) / 50)
        confidence = 0.5 + (data_quality_score * 0.3)  # 50-80% confidence
        
        if abs(price_change_percent) > 15:
            priority = "high"
            implementation_effort = "high"
        elif abs(price_change_percent) > 5:
            priority = "medium"
            implementation_effort = "medium"
        else:
            priority = "low"
            implementation_effort = "low"
        
        return PricingRecommendation(
            recommendation_id=str(uuid.uuid4()),
            recommendation_type=recommendation_type,
            current_price=current_price,
            recommended_price=recommended_price,
            expected_impact=expected_impact,
            confidence=confidence,
            reasoning=reasoning,
            implementation_effort=implementation_effort,
            priority=priority,
            created_at=datetime.utcnow()
        )
    
    async def forecast_revenue(self, months_ahead: int = 12) -> RevenueForcast:
        """Generate revenue forecast"""
        forecast_id = str(uuid.uuid4())
        
        # Get historical revenue data
        historical_data = self._get_historical_revenue_data()
        
        # Build forecasting model
        if SKLEARN_AVAILABLE and len(historical_data) >= 12:
            forecasted_revenue, confidence_interval, model_accuracy = await self._ml_revenue_forecast(
                historical_data, months_ahead
            )
        else:
            # Simple trend-based forecast
            forecasted_revenue, confidence_interval, model_accuracy = self._simple_revenue_forecast(
                historical_data, months_ahead
            )
        
        # Calculate revenue breakdown by stream
        revenue_breakdown = await self._forecast_revenue_breakdown(forecasted_revenue)
        
        # Define assumptions
        assumptions = {
            "customer_growth_rate": 0.15,  # 15% monthly growth
            "churn_rate": 0.05,  # 5% monthly churn
            "pricing_changes": "current_pricing_maintained",
            "market_conditions": "stable",
            "seasonal_adjustments": "included"
        }
        
        forecast = RevenueForcast(
            forecast_id=forecast_id,
            period_start=datetime.utcnow(),
            period_end=datetime.utcnow() + timedelta(days=months_ahead * 30),
            forecasted_revenue=forecasted_revenue,
            confidence_interval_lower=confidence_interval[0],
            confidence_interval_upper=confidence_interval[1],
            revenue_breakdown=revenue_breakdown,
            assumptions=assumptions,
            model_accuracy=model_accuracy,
            created_at=datetime.utcnow()
        )
        
        with self.lock:
            self.revenue_forecasts[forecast_id] = forecast
        
        return forecast
    
    def _get_historical_revenue_data(self) -> List[Dict[str, Any]]:
        """Get historical revenue data"""
        # In production, this would query actual revenue data
        # For now, generate simulated historical data
        
        historical_data = []
        base_revenue = 50000  # Starting monthly revenue
        
        for i in range(24):  # 24 months of history
            month_date = datetime.utcnow() - timedelta(days=(24-i) * 30)
            
            # Add growth trend with some noise
            growth_factor = 1 + (i * 0.08)  # 8% compound monthly growth
            noise_factor = 1 + (np.random.random() - 0.5) * 0.2  # ±10% noise
            
            monthly_revenue = base_revenue * growth_factor * noise_factor
            
            historical_data.append({
                "date": month_date,
                "revenue": monthly_revenue,
                "customers": int(monthly_revenue / 200),  # Avg $200 per customer
                "new_customers": int(monthly_revenue / 2000),  # 10% new customers
                "churned_customers": int(monthly_revenue / 4000)  # 5% churn
            })
        
        return historical_data
    
    async def _ml_revenue_forecast(self, historical_data: List[Dict[str, Any]], 
                                 months_ahead: int) -> Tuple[float, Tuple[float, float], float]:
        """Machine learning-based revenue forecast"""
        # Prepare features and targets
        features = []
        targets = []
        
        for i, data_point in enumerate(historical_data):
            feature_vector = [
                i,  # Time index
                data_point["customers"],
                data_point["new_customers"],
                data_point["churned_customers"],
                data_point["date"].month,  # Seasonal component
                data_point["date"].year - 2020  # Year offset
            ]
            features.append(feature_vector)
            targets.append(data_point["revenue"])
        
        X = np.array(features)
        y = np.array(targets)
        
        # Train model
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Calculate model accuracy using cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        model_accuracy = cv_scores.mean()
        
        # Make forecast
        last_data = historical_data[-1]
        future_features = []
        
        for month_offset in range(1, months_ahead + 1):
            future_date = datetime.utcnow() + timedelta(days=month_offset * 30)
            
            # Estimate future customer metrics (simplified)
            growth_rate = 0.15  # 15% monthly growth
            churn_rate = 0.05   # 5% monthly churn
            
            estimated_customers = last_data["customers"] * ((1 + growth_rate - churn_rate) ** month_offset)
            estimated_new_customers = estimated_customers * growth_rate
            estimated_churn = estimated_customers * churn_rate
            
            feature_vector = [
                len(historical_data) + month_offset - 1,
                estimated_customers,
                estimated_new_customers,
                estimated_churn,
                future_date.month,
                future_date.year - 2020
            ]
            future_features.append(feature_vector)
        
        X_future = np.array(future_features)
        future_predictions = model.predict(X_future)
        
        # Calculate total forecasted revenue
        total_forecasted_revenue = np.sum(future_predictions)
        
        # Estimate confidence interval (simplified)
        prediction_std = np.std(future_predictions)
        confidence_interval = (
            total_forecasted_revenue - 1.96 * prediction_std,
            total_forecasted_revenue + 1.96 * prediction_std
        )
        
        return total_forecasted_revenue, confidence_interval, model_accuracy
    
    def _simple_revenue_forecast(self, historical_data: List[Dict[str, Any]], 
                               months_ahead: int) -> Tuple[float, Tuple[float, float], float]:
        """Simple trend-based revenue forecast"""
        if len(historical_data) < 3:
            # Not enough data - use current revenue * months
            current_revenue = 100000  # Default assumption
            forecasted_revenue = current_revenue * months_ahead
            confidence_interval = (forecasted_revenue * 0.7, forecasted_revenue * 1.3)
            return forecasted_revenue, confidence_interval, 0.5
        
        # Calculate trend
        revenues = [data["revenue"] for data in historical_data[-12:]]  # Last 12 months
        
        # Simple linear trend
        x = list(range(len(revenues)))
        if len(revenues) >= 2:
            # Calculate linear regression manually
            x_mean = statistics.mean(x)
            y_mean = statistics.mean(revenues)
            
            numerator = sum((x[i] - x_mean) * (revenues[i] - y_mean) for i in range(len(x)))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
            
            if denominator != 0:
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean
                
                # Forecast future months
                future_revenues = []
                for month_offset in range(1, months_ahead + 1):
                    future_x = len(revenues) + month_offset - 1
                    future_revenue = slope * future_x + intercept
                    future_revenues.append(max(0, future_revenue))  # Ensure non-negative
                
                forecasted_revenue = sum(future_revenues)
                
                # Calculate confidence interval based on historical variance
                revenue_std = statistics.stdev(revenues)
                confidence_interval = (
                    forecasted_revenue - 1.96 * revenue_std * months_ahead,
                    forecasted_revenue + 1.96 * revenue_std * months_ahead
                )
                
                # Simple accuracy estimate
                model_accuracy = max(0.3, 1 - (revenue_std / y_mean))
                
                return forecasted_revenue, confidence_interval, model_accuracy
        
        # Fallback to average-based forecast
        avg_revenue = statistics.mean(revenues)
        forecasted_revenue = avg_revenue * months_ahead
        confidence_interval = (forecasted_revenue * 0.8, forecasted_revenue * 1.2)
        
        return forecasted_revenue, confidence_interval, 0.6
    
    async def _forecast_revenue_breakdown(self, total_revenue: float) -> Dict[str, float]:
        """Forecast revenue breakdown by stream"""
        # Revenue distribution assumptions (would be based on actual data)
        revenue_streams = {
            RevenueStream.SUBSCRIPTION_FEES.value: 0.75,  # 75% of revenue
            RevenueStream.USAGE_FEES.value: 0.15,         # 15% of revenue
            RevenueStream.SETUP_FEES.value: 0.05,         # 5% of revenue
            RevenueStream.SUPPORT_FEES.value: 0.03,       # 3% of revenue
            RevenueStream.TRAINING_FEES.value: 0.02       # 2% of revenue
        }
        
        breakdown = {}
        for stream, percentage in revenue_streams.items():
            breakdown[stream] = total_revenue * percentage
        
        return breakdown
    
    def get_pricing_analytics(self) -> Dict[str, Any]:
        """Get comprehensive pricing analytics"""
        analytics = {
            "pricing_tiers": {tier_id: tier.to_dict() for tier_id, tier in self.pricing_tiers.items()},
            "active_price_tests": [],
            "completed_price_tests": [],
            "recent_pricing_changes": [],
            "tier_performance": {},
            "revenue_metrics": self._calculate_revenue_metrics(),
            "competitive_position": self._analyze_competitive_position()
        }
        
        # Price test analytics
        for test in self.price_tests.values():
            test_data = test.to_dict()
            
            if test.ended_at:
                analytics["completed_price_tests"].append(test_data)
            else:
                analytics["active_price_tests"].append(test_data)
        
        # Recent pricing changes
        recent_changes = [entry for entry in list(self.pricing_history)[-10:]]
        analytics["recent_pricing_changes"] = [
            {
                "timestamp": entry["timestamp"].isoformat(),
                "change_type": entry["change_type"],
                "data": entry["data"]
            }
            for entry in recent_changes
        ]
        
        # Tier performance (would integrate with actual sales data)
        for tier_id, tier in self.pricing_tiers.items():
            analytics["tier_performance"][tier_id] = {
                "tier_name": tier.name,
                "current_price": tier.price,
                "estimated_monthly_revenue": tier.price * 100,  # Placeholder
                "estimated_customers": 100,  # Placeholder
                "conversion_rate": 0.15,  # Placeholder
                "customer_lifetime_value": tier.price * 12,  # Placeholder
            }
        
        return analytics
    
    def _calculate_revenue_metrics(self) -> Dict[str, Any]:
        """Calculate key revenue metrics"""
        # This would integrate with actual financial data
        return {
            "monthly_recurring_revenue": 250000,
            "annual_recurring_revenue": 3000000,
            "average_revenue_per_user": 150,
            "customer_lifetime_value": 1800,
            "revenue_growth_rate": 0.15,  # 15% monthly
            "churn_impact_on_revenue": 0.05  # 5% loss
        }
    
    def _analyze_competitive_position(self) -> Dict[str, Any]:
        """Analyze competitive pricing position"""
        return {
            "position": "competitive",
            "pricing_vs_market": {
                "starter_tier": "below_market",
                "professional_tier": "market_aligned",
                "business_tier": "above_market",
                "enterprise_tier": "competitive"
            },
            "value_proposition_strength": "high",
            "differentiation_factors": [
                "Multi-framework support",
                "Local inference capability",
                "Unified protocol"
            ]
        }
    
    async def get_pricing_recommendations(self) -> List[PricingRecommendation]:
        """Get all pricing recommendations"""
        recommendations = []
        
        for tier_id in self.pricing_tiers.keys():
            recommendation = await self.optimize_pricing(tier_id)
            if recommendation:
                recommendations.append(recommendation)
        
        # Sort by priority
        priority_order = {"high": 3, "medium": 2, "low": 1}
        recommendations.sort(key=lambda x: priority_order.get(x.priority, 0), reverse=True)
        
        return recommendations

# Global pricing optimizer instance
pricing_optimizer = PricingOptimizer()

# Convenience functions
def add_pricing_tier(tier_data: Dict[str, Any]) -> str:
    """Add pricing tier"""
    return pricing_optimizer.add_pricing_tier(tier_data)

def start_price_test(test_config: Dict[str, Any]) -> str:
    """Start price test"""
    return pricing_optimizer.start_price_test(test_config)

async def optimize_pricing(tier_id: str) -> Optional[PricingRecommendation]:
    """Optimize pricing"""
    return await pricing_optimizer.optimize_pricing(tier_id)

async def forecast_revenue(months_ahead: int = 12) -> RevenueForcast:
    """Forecast revenue"""
    return await pricing_optimizer.forecast_revenue(months_ahead)

def get_pricing_analytics() -> Dict[str, Any]:
    """Get pricing analytics"""
    return pricing_optimizer.get_pricing_analytics()

async def get_pricing_recommendations() -> List[PricingRecommendation]:
    """Get pricing recommendations"""
    return await pricing_optimizer.get_pricing_recommendations()