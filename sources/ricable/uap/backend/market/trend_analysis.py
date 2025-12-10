# File: backend/market/trend_analysis.py
"""
Market Trend Analysis System for UAP Platform

Provides comprehensive market trend analysis, competitive intelligence,
and industry insights for strategic decision making.
"""

import asyncio
import json
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import uuid
import warnings
from threading import Lock
import requests
from urllib.parse import quote

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from textblob import TextBlob
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..analytics.usage_analytics import usage_analytics
from ..analytics.predictive_analytics import predictive_analytics

class TrendType(Enum):
    """Types of market trends"""
    MARKET_GROWTH = "market_growth"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    CUSTOMER_SENTIMENT = "customer_sentiment"
    TECHNOLOGY_ADOPTION = "technology_adoption"
    PRICING_TRENDS = "pricing_trends"
    FEATURE_DEMAND = "feature_demand"
    INDUSTRY_INSIGHTS = "industry_insights"
    SEASONAL_PATTERNS = "seasonal_patterns"

class TrendDirection(Enum):
    """Direction of trend movement"""
    UPWARD = "upward"
    DOWNWARD = "downward"
    STABLE = "stable"
    VOLATILE = "volatile"

class CompetitorTier(Enum):
    """Competitor categorization"""
    DIRECT = "direct"
    INDIRECT = "indirect"
    EMERGING = "emerging"
    SUBSTITUTE = "substitute"

@dataclass
class MarketTrendPoint:
    """Market trend data point"""
    timestamp: datetime
    metric_name: str
    value: float
    source: str
    confidence: float
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_name": self.metric_name,
            "value": self.value,
            "source": self.source,
            "confidence": self.confidence,
            "metadata": self.metadata or {}
        }

@dataclass
class TrendAnalysisResult:
    """Result of trend analysis"""
    trend_id: str
    trend_type: TrendType
    metric_name: str
    direction: TrendDirection
    strength: float  # 0-1 scale
    confidence: float  # 0-1 scale
    time_period: timedelta
    start_value: float
    end_value: float
    growth_rate: float
    insights: List[str]
    recommendations: List[str]
    created_at: datetime
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trend_id": self.trend_id,
            "trend_type": self.trend_type.value,
            "metric_name": self.metric_name,
            "direction": self.direction.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "time_period_days": self.time_period.days,
            "start_value": self.start_value,
            "end_value": self.end_value,
            "growth_rate": self.growth_rate,
            "insights": self.insights,
            "recommendations": self.recommendations,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata or {}
        }

@dataclass
class CompetitorProfile:
    """Competitor profile data"""
    competitor_id: str
    name: str
    tier: CompetitorTier
    market_share: float
    pricing_model: str
    key_features: List[str]
    strengths: List[str]
    weaknesses: List[str]
    last_updated: datetime
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "competitor_id": self.competitor_id,
            "name": self.name,
            "tier": self.tier.value,
            "market_share": self.market_share,
            "pricing_model": self.pricing_model,
            "key_features": self.key_features,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata or {}
        }

@dataclass
class MarketInsight:
    """Market insight data"""
    insight_id: str
    category: str
    title: str
    description: str
    impact_score: float  # 0-10 scale
    urgency: str  # high, medium, low
    actionable_items: List[str]
    data_sources: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_id": self.insight_id,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "impact_score": self.impact_score,
            "urgency": self.urgency,
            "actionable_items": self.actionable_items,
            "data_sources": self.data_sources,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }

class TrendAnalyzer:
    """Trend analysis engine"""
    
    def __init__(self):
        self.lock = Lock()
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.trend_models = {}
        
    def detect_trend_direction(self, data_points: List[float]) -> Tuple[TrendDirection, float]:
        """Detect trend direction and strength"""
        if len(data_points) < 2:
            return TrendDirection.STABLE, 0.0
        
        # Calculate linear regression if available
        if SKLEARN_AVAILABLE and len(data_points) >= 3:
            X = np.array(range(len(data_points))).reshape(-1, 1)
            y = np.array(data_points)
            
            model = LinearRegression()
            model.fit(X, y)
            
            slope = model.coef_[0]
            r_squared = model.score(X, y)
            
            # Determine direction based on slope
            if abs(slope) < 0.01:
                direction = TrendDirection.STABLE
            elif slope > 0:
                direction = TrendDirection.UPWARD
            else:
                direction = TrendDirection.DOWNWARD
            
            # Calculate volatility
            volatility = np.std(data_points) / np.mean(data_points) if np.mean(data_points) != 0 else 0
            
            if volatility > 0.3:
                direction = TrendDirection.VOLATILE
            
            strength = min(1.0, abs(slope) * r_squared)
            
        else:
            # Simple statistical approach
            first_half = data_points[:len(data_points)//2]
            second_half = data_points[len(data_points)//2:]
            
            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)
            
            change = (second_avg - first_avg) / first_avg if first_avg != 0 else 0
            
            if abs(change) < 0.05:
                direction = TrendDirection.STABLE
            elif change > 0:
                direction = TrendDirection.UPWARD
            else:
                direction = TrendDirection.DOWNWARD
            
            strength = min(1.0, abs(change))
        
        return direction, strength
    
    def calculate_growth_rate(self, start_value: float, end_value: float, 
                            time_period: timedelta) -> float:
        """Calculate growth rate"""
        if start_value == 0:
            return 0.0
        
        days = time_period.days
        if days == 0:
            return 0.0
        
        # Calculate compound annual growth rate (CAGR)
        years = days / 365.25
        if years == 0:
            return 0.0
        
        growth_rate = ((end_value / start_value) ** (1 / years)) - 1
        return growth_rate
    
    def analyze_trend(self, data_points: List[MarketTrendPoint], 
                     trend_type: TrendType, metric_name: str) -> TrendAnalysisResult:
        """Analyze trend from data points"""
        if not data_points:
            return None
        
        # Sort by timestamp
        data_points.sort(key=lambda x: x.timestamp)
        
        # Extract values
        values = [point.value for point in data_points]
        timestamps = [point.timestamp for point in data_points]
        
        # Detect trend direction and strength
        direction, strength = self.detect_trend_direction(values)
        
        # Calculate basic statistics
        start_value = values[0]
        end_value = values[-1]
        time_period = timestamps[-1] - timestamps[0]
        
        # Calculate growth rate
        growth_rate = self.calculate_growth_rate(start_value, end_value, time_period)
        
        # Calculate confidence based on data quality
        confidence = min(1.0, len(data_points) / 30)  # Higher confidence with more data
        avg_confidence = statistics.mean([point.confidence for point in data_points])
        confidence = (confidence + avg_confidence) / 2
        
        # Generate insights
        insights = self._generate_insights(trend_type, direction, strength, growth_rate, values)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(trend_type, direction, strength, growth_rate)
        
        return TrendAnalysisResult(
            trend_id=str(uuid.uuid4()),
            trend_type=trend_type,
            metric_name=metric_name,
            direction=direction,
            strength=strength,
            confidence=confidence,
            time_period=time_period,
            start_value=start_value,
            end_value=end_value,
            growth_rate=growth_rate,
            insights=insights,
            recommendations=recommendations,
            created_at=datetime.utcnow(),
            metadata={
                "data_points": len(data_points),
                "value_range": [min(values), max(values)],
                "volatility": statistics.stdev(values) if len(values) > 1 else 0
            }
        )
    
    def _generate_insights(self, trend_type: TrendType, direction: TrendDirection, 
                          strength: float, growth_rate: float, values: List[float]) -> List[str]:
        """Generate insights based on trend analysis"""
        insights = []
        
        # Direction insights
        if direction == TrendDirection.UPWARD:
            insights.append(f"Strong upward trend detected with {strength:.1%} strength")
            if growth_rate > 0.5:
                insights.append("Exceptional growth rate indicates market opportunity")
        elif direction == TrendDirection.DOWNWARD:
            insights.append(f"Downward trend detected with {strength:.1%} strength")
            if growth_rate < -0.2:
                insights.append("Significant decline requires immediate attention")
        elif direction == TrendDirection.VOLATILE:
            insights.append("High volatility detected - market uncertainty present")
            insights.append("Consider risk mitigation strategies")
        else:
            insights.append("Stable trend - steady market conditions")
        
        # Trend type specific insights
        if trend_type == TrendType.MARKET_GROWTH:
            if direction == TrendDirection.UPWARD:
                insights.append("Market expansion presents scaling opportunities")
            elif direction == TrendDirection.DOWNWARD:
                insights.append("Market contraction requires defensive strategies")
        
        elif trend_type == TrendType.COMPETITIVE_ANALYSIS:
            if direction == TrendDirection.UPWARD:
                insights.append("Competitive pressure increasing")
            else:
                insights.append("Competitive advantage opportunity identified")
        
        elif trend_type == TrendType.PRICING_TRENDS:
            if direction == TrendDirection.UPWARD:
                insights.append("Price increases possible due to market conditions")
            elif direction == TrendDirection.DOWNWARD:
                insights.append("Price pressure - consider value-based positioning")
        
        # Value-based insights
        if len(values) > 1:
            volatility = statistics.stdev(values) / statistics.mean(values)
            if volatility > 0.3:
                insights.append("High volatility suggests market instability")
            elif volatility < 0.1:
                insights.append("Low volatility indicates stable market conditions")
        
        return insights
    
    def _generate_recommendations(self, trend_type: TrendType, direction: TrendDirection, 
                                strength: float, growth_rate: float) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Direction-based recommendations
        if direction == TrendDirection.UPWARD and strength > 0.7:
            recommendations.append("Accelerate investment in this area")
            recommendations.append("Increase marketing efforts to capitalize on growth")
            recommendations.append("Scale resources to meet growing demand")
        
        elif direction == TrendDirection.DOWNWARD and strength > 0.7:
            recommendations.append("Reassess strategy and pivot if necessary")
            recommendations.append("Reduce investment in declining areas")
            recommendations.append("Focus on customer retention strategies")
        
        elif direction == TrendDirection.VOLATILE:
            recommendations.append("Implement risk management protocols")
            recommendations.append("Diversify portfolio to reduce exposure")
            recommendations.append("Monitor closely for trend stabilization")
        
        # Trend type specific recommendations
        if trend_type == TrendType.MARKET_GROWTH:
            if direction == TrendDirection.UPWARD:
                recommendations.append("Prepare for market expansion")
                recommendations.append("Invest in scalable infrastructure")
            else:
                recommendations.append("Focus on market share protection")
        
        elif trend_type == TrendType.COMPETITIVE_ANALYSIS:
            recommendations.append("Conduct competitor feature analysis")
            recommendations.append("Identify differentiation opportunities")
            recommendations.append("Monitor competitive pricing strategies")
        
        elif trend_type == TrendType.PRICING_TRENDS:
            if direction == TrendDirection.UPWARD:
                recommendations.append("Consider strategic price increases")
                recommendations.append("Communicate value proposition clearly")
            else:
                recommendations.append("Optimize cost structure")
                recommendations.append("Enhance value delivery")
        
        return recommendations

class MarketIntelligence:
    """Main market intelligence system"""
    
    def __init__(self):
        self.trend_analyzer = TrendAnalyzer()
        self.trend_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.competitors: Dict[str, CompetitorProfile] = {}
        self.insights: Dict[str, MarketInsight] = {}
        self.analysis_cache: Dict[str, TrendAnalysisResult] = {}
        self.lock = Lock()
        
        # Configuration
        self.data_retention_days = 365
        self.analysis_interval_hours = 6
        self.min_data_points = 10
        
        # Initialize with sample competitors
        self._initialize_competitors()
    
    def _initialize_competitors(self):
        """Initialize competitor profiles"""
        sample_competitors = [
            {
                "name": "OpenAI Platform",
                "tier": CompetitorTier.DIRECT,
                "market_share": 25.0,
                "pricing_model": "API Usage + Subscription",
                "key_features": ["GPT Models", "API Access", "Fine-tuning", "Embeddings"],
                "strengths": ["Model Quality", "Developer Tools", "Documentation"],
                "weaknesses": ["Cost", "Rate Limits", "Dependency Risk"]
            },
            {
                "name": "Anthropic Claude",
                "tier": CompetitorTier.DIRECT,
                "market_share": 15.0,
                "pricing_model": "API Usage",
                "key_features": ["Claude Models", "Constitutional AI", "Long Context"],
                "strengths": ["Safety", "Reasoning", "Context Length"],
                "weaknesses": ["Limited Availability", "Newer Platform"]
            },
            {
                "name": "Google Vertex AI",
                "tier": CompetitorTier.DIRECT,
                "market_share": 20.0,
                "pricing_model": "Pay-per-use",
                "key_features": ["PaLM Models", "AutoML", "MLOps", "Vertex Search"],
                "strengths": ["Google Ecosystem", "Enterprise Tools", "Scaling"],
                "weaknesses": ["Complexity", "Vendor Lock-in"]
            },
            {
                "name": "AWS Bedrock",
                "tier": CompetitorTier.DIRECT,
                "market_share": 18.0,
                "pricing_model": "Pay-per-use",
                "key_features": ["Multi-Model Access", "Fine-tuning", "Guardrails"],
                "strengths": ["AWS Integration", "Model Choice", "Enterprise"],
                "weaknesses": ["AWS Dependency", "Cost Complexity"]
            },
            {
                "name": "Hugging Face",
                "tier": CompetitorTier.INDIRECT,
                "market_share": 12.0,
                "pricing_model": "Open Source + Paid Services",
                "key_features": ["Model Hub", "Transformers", "Datasets", "Spaces"],
                "strengths": ["Open Source", "Community", "Model Variety"],
                "weaknesses": ["Self-hosted Complexity", "Support"]
            }
        ]
        
        for comp_data in sample_competitors:
            competitor = CompetitorProfile(
                competitor_id=str(uuid.uuid4()),
                name=comp_data["name"],
                tier=comp_data["tier"],
                market_share=comp_data["market_share"],
                pricing_model=comp_data["pricing_model"],
                key_features=comp_data["key_features"],
                strengths=comp_data["strengths"],
                weaknesses=comp_data["weaknesses"],
                last_updated=datetime.utcnow()
            )
            self.competitors[competitor.competitor_id] = competitor
    
    def add_trend_data(self, metric_name: str, value: float, source: str, 
                      confidence: float = 1.0, metadata: Dict[str, Any] = None):
        """Add trend data point"""
        data_point = MarketTrendPoint(
            timestamp=datetime.utcnow(),
            metric_name=metric_name,
            value=value,
            source=source,
            confidence=confidence,
            metadata=metadata
        )
        
        with self.lock:
            self.trend_data[metric_name].append(data_point)
    
    async def analyze_market_trends(self, metric_name: str, 
                                  trend_type: TrendType) -> Optional[TrendAnalysisResult]:
        """Analyze market trends for a specific metric"""
        with self.lock:
            data_points = list(self.trend_data[metric_name])
        
        if len(data_points) < self.min_data_points:
            return None
        
        # Filter recent data (last 90 days)
        cutoff_date = datetime.utcnow() - timedelta(days=90)
        recent_data = [dp for dp in data_points if dp.timestamp >= cutoff_date]
        
        if not recent_data:
            return None
        
        # Analyze trend
        analysis = self.trend_analyzer.analyze_trend(recent_data, trend_type, metric_name)
        
        if analysis:
            # Cache the analysis
            cache_key = f"{metric_name}_{trend_type.value}"
            self.analysis_cache[cache_key] = analysis
        
        return analysis
    
    async def get_competitive_analysis(self) -> Dict[str, Any]:
        """Get comprehensive competitive analysis"""
        competitors_data = {}
        
        # Analyze each competitor
        for comp_id, competitor in self.competitors.items():
            competitors_data[comp_id] = competitor.to_dict()
        
        # Market share analysis
        total_market_share = sum(comp.market_share for comp in self.competitors.values())
        our_market_share = max(0, 100 - total_market_share)
        
        # Competitive positioning
        strengths_analysis = defaultdict(int)
        weaknesses_analysis = defaultdict(int)
        
        for competitor in self.competitors.values():
            for strength in competitor.strengths:
                strengths_analysis[strength] += 1
            for weakness in competitor.weaknesses:
                weaknesses_analysis[weakness] += 1
        
        # Pricing analysis
        pricing_models = defaultdict(int)
        for competitor in self.competitors.values():
            pricing_models[competitor.pricing_model] += 1
        
        return {
            "market_overview": {
                "total_competitors": len(self.competitors),
                "our_market_share": our_market_share,
                "competitor_market_share": total_market_share,
                "market_concentration": "moderate" if total_market_share < 80 else "high"
            },
            "competitors": competitors_data,
            "competitive_landscape": {
                "common_strengths": dict(strengths_analysis),
                "common_weaknesses": dict(weaknesses_analysis),
                "pricing_models": dict(pricing_models)
            },
            "opportunities": self._identify_market_opportunities(),
            "threats": self._identify_market_threats(),
            "recommendations": self._generate_competitive_recommendations()
        }
    
    def _identify_market_opportunities(self) -> List[str]:
        """Identify market opportunities"""
        opportunities = []
        
        # Analyze competitor weaknesses
        weakness_counts = defaultdict(int)
        for competitor in self.competitors.values():
            for weakness in competitor.weaknesses:
                weakness_counts[weakness] += 1
        
        # Common weaknesses are opportunities
        for weakness, count in weakness_counts.items():
            if count >= 2:
                opportunities.append(f"Address common competitor weakness: {weakness}")
        
        # Pricing opportunities
        pricing_models = [comp.pricing_model for comp in self.competitors.values()]
        if "Freemium" not in pricing_models:
            opportunities.append("Consider freemium pricing model")
        
        if "Enterprise" not in pricing_models:
            opportunities.append("Target enterprise segment")
        
        # Feature gaps
        all_features = set()
        for competitor in self.competitors.values():
            all_features.update(competitor.key_features)
        
        our_features = {"Multi-Framework", "Unified Protocol", "Local Inference", "Distributed Processing"}
        missing_features = all_features - our_features
        
        for feature in list(missing_features)[:3]:  # Top 3 missing features
            opportunities.append(f"Consider adding feature: {feature}")
        
        return opportunities
    
    def _identify_market_threats(self) -> List[str]:
        """Identify market threats"""
        threats = []
        
        # Large market share competitors
        for competitor in self.competitors.values():
            if competitor.market_share > 20:
                threats.append(f"Dominant player: {competitor.name} ({competitor.market_share}% market share)")
        
        # Common competitor strengths
        strength_counts = defaultdict(int)
        for competitor in self.competitors.values():
            for strength in competitor.strengths:
                strength_counts[strength] += 1
        
        for strength, count in strength_counts.items():
            if count >= 3:
                threats.append(f"Industry standard strength: {strength}")
        
        # Pricing pressure
        api_usage_models = [comp for comp in self.competitors.values() 
                           if "API" in comp.pricing_model]
        if len(api_usage_models) >= 3:
            threats.append("Pricing pressure from API usage models")
        
        return threats
    
    def _generate_competitive_recommendations(self) -> List[str]:
        """Generate competitive recommendations"""
        recommendations = []
        
        # Differentiation strategies
        recommendations.append("Focus on multi-framework unified approach as key differentiator")
        recommendations.append("Emphasize local inference capabilities for privacy-conscious customers")
        recommendations.append("Highlight distributed processing for enterprise scalability")
        
        # Pricing strategy
        recommendations.append("Consider competitive pricing for API usage model")
        recommendations.append("Develop enterprise pricing tiers")
        
        # Product strategy
        recommendations.append("Invest in developer experience and documentation")
        recommendations.append("Build strong community and ecosystem")
        recommendations.append("Focus on performance and reliability")
        
        return recommendations
    
    async def generate_market_insights(self) -> List[MarketInsight]:
        """Generate actionable market insights"""
        insights = []
        
        # Analyze recent trends
        for metric_name in self.trend_data.keys():
            for trend_type in [TrendType.MARKET_GROWTH, TrendType.COMPETITIVE_ANALYSIS, 
                             TrendType.PRICING_TRENDS]:
                analysis = await self.analyze_market_trends(metric_name, trend_type)
                
                if analysis and analysis.confidence > 0.7:
                    insight = MarketInsight(
                        insight_id=str(uuid.uuid4()),
                        category=trend_type.value,
                        title=f"{metric_name} Trend Analysis",
                        description=f"Detected {analysis.direction.value} trend with {analysis.strength:.1%} strength",
                        impact_score=min(10, analysis.strength * 10),
                        urgency="high" if analysis.strength > 0.8 else "medium" if analysis.strength > 0.5 else "low",
                        actionable_items=analysis.recommendations[:3],
                        data_sources=[f"trend_analysis_{metric_name}"],
                        created_at=datetime.utcnow(),
                        expires_at=datetime.utcnow() + timedelta(days=30)
                    )
                    insights.append(insight)
        
        # Competitive insights
        competitive_analysis = await self.get_competitive_analysis()
        
        if competitive_analysis["market_overview"]["our_market_share"] < 10:
            insight = MarketInsight(
                insight_id=str(uuid.uuid4()),
                category="competitive_analysis",
                title="Low Market Share Alert",
                description="Current market share below 10% - aggressive growth strategies needed",
                impact_score=9.0,
                urgency="high",
                actionable_items=[
                    "Increase marketing investment",
                    "Enhance product differentiation",
                    "Consider strategic partnerships"
                ],
                data_sources=["competitive_analysis"],
                created_at=datetime.utcnow()
            )
            insights.append(insight)
        
        # Store insights
        with self.lock:
            for insight in insights:
                self.insights[insight.insight_id] = insight
        
        return insights
    
    async def get_market_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive market report"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Collect trend analyses
        trend_analyses = {}
        for metric_name in self.trend_data.keys():
            for trend_type in TrendType:
                analysis = await self.analyze_market_trends(metric_name, trend_type)
                if analysis:
                    key = f"{metric_name}_{trend_type.value}"
                    trend_analyses[key] = analysis.to_dict()
        
        # Get competitive analysis
        competitive_analysis = await self.get_competitive_analysis()
        
        # Get recent insights
        recent_insights = []
        for insight in self.insights.values():
            if insight.created_at >= cutoff_date:
                recent_insights.append(insight.to_dict())
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(trend_analyses, competitive_analysis, recent_insights)
        
        return {
            "report_id": str(uuid.uuid4()),
            "generated_at": datetime.utcnow().isoformat(),
            "time_period_days": days,
            "executive_summary": executive_summary,
            "trend_analyses": trend_analyses,
            "competitive_analysis": competitive_analysis,
            "market_insights": recent_insights[:10],  # Top 10 insights
            "key_metrics": self._calculate_key_metrics(),
            "recommendations": self._generate_strategic_recommendations(trend_analyses, competitive_analysis)
        }
    
    def _generate_executive_summary(self, trend_analyses: Dict, competitive_analysis: Dict, 
                                  insights: List[Dict]) -> str:
        """Generate executive summary"""
        summary_points = []
        
        # Market position
        market_share = competitive_analysis["market_overview"]["our_market_share"]
        if market_share > 15:
            summary_points.append(f"Strong market position with {market_share:.1f}% market share")
        elif market_share > 5:
            summary_points.append(f"Growing market position with {market_share:.1f}% market share")
        else:
            summary_points.append(f"Emerging market position with {market_share:.1f}% market share")
        
        # Trend summary
        upward_trends = sum(1 for ta in trend_analyses.values() if ta.get("direction") == "upward")
        total_trends = len(trend_analyses)
        
        if upward_trends > total_trends * 0.6:
            summary_points.append("Majority of market trends showing positive momentum")
        elif upward_trends > total_trends * 0.4:
            summary_points.append("Mixed market trends with moderate growth signals")
        else:
            summary_points.append("Challenging market conditions with defensive positioning needed")
        
        # Competitive landscape
        competitor_count = competitive_analysis["market_overview"]["total_competitors"]
        summary_points.append(f"Operating in competitive landscape with {competitor_count} major competitors")
        
        # Insights summary
        high_impact_insights = [i for i in insights if i.get("impact_score", 0) > 7]
        if high_impact_insights:
            summary_points.append(f"Identified {len(high_impact_insights)} high-impact strategic opportunities")
        
        return ". ".join(summary_points) + "."
    
    def _calculate_key_metrics(self) -> Dict[str, Any]:
        """Calculate key market metrics"""
        metrics = {}
        
        # Data quality metrics
        total_data_points = sum(len(data) for data in self.trend_data.values())
        metrics["total_data_points"] = total_data_points
        metrics["metrics_tracked"] = len(self.trend_data)
        
        # Analysis metrics
        metrics["cached_analyses"] = len(self.analysis_cache)
        metrics["active_insights"] = len(self.insights)
        
        # Competitive metrics
        metrics["competitors_tracked"] = len(self.competitors)
        metrics["direct_competitors"] = len([c for c in self.competitors.values() 
                                           if c.tier == CompetitorTier.DIRECT])
        
        return metrics
    
    def _generate_strategic_recommendations(self, trend_analyses: Dict, 
                                          competitive_analysis: Dict) -> List[str]:
        """Generate strategic recommendations"""
        recommendations = []
        
        # Based on trend analysis
        upward_trends = [ta for ta in trend_analyses.values() if ta.get("direction") == "upward"]
        downward_trends = [ta for ta in trend_analyses.values() if ta.get("direction") == "downward"]
        
        if len(upward_trends) > len(downward_trends):
            recommendations.append("Capitalize on positive market momentum with aggressive growth strategies")
        else:
            recommendations.append("Focus on defensive positioning and market share protection")
        
        # Based on competitive analysis
        our_market_share = competitive_analysis["market_overview"]["our_market_share"]
        
        if our_market_share < 10:
            recommendations.append("Implement market penetration strategies")
            recommendations.append("Consider strategic partnerships or acquisitions")
        elif our_market_share < 25:
            recommendations.append("Focus on differentiation and customer retention")
            recommendations.append("Expand into adjacent markets")
        else:
            recommendations.append("Maintain market leadership position")
            recommendations.append("Invest in innovation and market development")
        
        # Product strategy
        recommendations.append("Continue investing in multi-framework unified approach")
        recommendations.append("Enhance developer experience and documentation")
        recommendations.append("Build strong enterprise sales channel")
        
        return recommendations
    
    async def collect_market_data(self):
        """Collect market data from various sources"""
        current_time = datetime.utcnow()
        
        # Collect usage-based market data
        usage_summary = usage_analytics.get_usage_summary(24)
        
        # Market growth indicators
        self.add_trend_data("daily_active_users", usage_summary["summary"]["unique_users"], 
                          "internal_analytics", 0.9)
        self.add_trend_data("agent_requests", usage_summary["summary"]["total_events"], 
                          "internal_analytics", 0.9)
        
        # Competitive indicators (would integrate with external data sources)
        # For now, simulate with internal metrics
        self.add_trend_data("market_interest", usage_summary["summary"]["unique_users"] * 1.5, 
                          "market_simulation", 0.7)
        self.add_trend_data("technology_adoption", len(usage_summary["agent_usage"]), 
                          "internal_metrics", 0.8)
        
        # Pricing indicators
        avg_response_time = usage_summary["summary"]["avg_response_time_ms"]
        performance_score = max(0, 100 - (avg_response_time / 10))  # Convert to 0-100 scale
        self.add_trend_data("service_quality", performance_score, "internal_metrics", 0.9)
        
        print(f"Market data collection completed at {current_time}")
    
    async def start_background_tasks(self):
        """Start background market intelligence tasks"""
        while True:
            try:
                # Collect market data every 2 hours
                await self.collect_market_data()
                
                # Generate insights every 6 hours
                await self.generate_market_insights()
                
                # Clean up old data
                await self._cleanup_old_data()
                
                await asyncio.sleep(7200)  # 2 hours
                
            except Exception as e:
                print(f"Error in market intelligence background tasks: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_old_data(self):
        """Clean up old market data"""
        cutoff_date = datetime.utcnow() - timedelta(days=self.data_retention_days)
        
        with self.lock:
            # Clean up trend data
            for metric_name in list(self.trend_data.keys()):
                data_points = self.trend_data[metric_name]
                filtered_data = deque([dp for dp in data_points if dp.timestamp >= cutoff_date], 
                                    maxlen=1000)
                self.trend_data[metric_name] = filtered_data
            
            # Clean up expired insights
            expired_insights = []
            for insight_id, insight in self.insights.items():
                if insight.expires_at and insight.expires_at < datetime.utcnow():
                    expired_insights.append(insight_id)
            
            for insight_id in expired_insights:
                del self.insights[insight_id]

# Global market intelligence instance
market_intelligence = MarketIntelligence()

# Convenience functions
async def analyze_market_trends(metric_name: str, trend_type: TrendType) -> Optional[TrendAnalysisResult]:
    """Analyze market trends"""
    return await market_intelligence.analyze_market_trends(metric_name, trend_type)

async def get_competitive_analysis() -> Dict[str, Any]:
    """Get competitive analysis"""
    return await market_intelligence.get_competitive_analysis()

async def generate_market_report(days: int = 30) -> Dict[str, Any]:
    """Generate market report"""
    return await market_intelligence.get_market_report(days)

async def get_market_insights() -> List[MarketInsight]:
    """Get market insights"""
    return await market_intelligence.generate_market_insights()

def add_market_data(metric_name: str, value: float, source: str, confidence: float = 1.0):
    """Add market trend data"""
    market_intelligence.add_trend_data(metric_name, value, source, confidence)