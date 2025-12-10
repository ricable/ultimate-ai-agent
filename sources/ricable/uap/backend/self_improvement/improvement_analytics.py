"""
Agent 40: Self-Improving AI Metacognition System - Improvement Analytics
Advanced analytics and insights for self-improvement processes.
"""

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class ImprovementTrend:
    """Represents a trend in improvement metrics"""
    metric_name: str
    time_period: timedelta
    trend_direction: str  # "improving", "declining", "stable"
    rate_of_change: float
    confidence: float
    data_points: int
    significance: str  # "high", "medium", "low"


@dataclass
class ImprovementInsight:
    """Represents an analytical insight about improvements"""
    insight_id: str
    timestamp: datetime
    insight_type: str
    description: str
    evidence: List[Dict[str, Any]]
    impact_assessment: Dict[str, float]
    recommendations: List[str]
    confidence: float
    metadata: Dict[str, Any]


class ImprovementAnalytics:
    """Advanced analytics for self-improvement processes"""
    
    def __init__(self):
        self.analytics_id = str(uuid4())
        self.improvement_history: List[Dict[str, Any]] = []
        self.trends_cache: Dict[str, ImprovementTrend] = {}
        self.insights_history: List[ImprovementInsight] = []
        
        # Analytics parameters
        self.trend_analysis_window = timedelta(hours=24)
        self.min_data_points = 5
        self.significance_thresholds = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
    
    async def analyze_improvement_trends(self, 
                                       improvement_data: List[Dict[str, Any]]) -> List[ImprovementTrend]:
        """Analyze trends in improvement data"""
        trends = []
        
        # Group data by metric
        metrics_data = defaultdict(list)
        for data_point in improvement_data:
            for metric, value in data_point.get('improvements', {}).items():
                metrics_data[metric].append({
                    'timestamp': data_point.get('timestamp', datetime.utcnow()),
                    'value': value,
                    'context': data_point.get('context', {})
                })
        
        # Analyze each metric's trend
        for metric_name, data_points in metrics_data.items():
            if len(data_points) >= self.min_data_points:
                trend = await self._calculate_trend(metric_name, data_points)
                if trend:
                    trends.append(trend)
                    self.trends_cache[metric_name] = trend
        
        return trends
    
    async def _calculate_trend(self, metric_name: str, 
                             data_points: List[Dict[str, Any]]) -> Optional[ImprovementTrend]:
        """Calculate trend for a specific metric"""
        try:
            # Sort by timestamp
            sorted_points = sorted(data_points, key=lambda x: x['timestamp'])
            
            # Extract values and calculate trend
            values = [point['value'] for point in sorted_points]
            timestamps = [point['timestamp'] for point in sorted_points]
            
            if len(values) < 2:
                return None
            
            # Simple linear trend calculation
            time_diffs = [(timestamps[i] - timestamps[0]).total_seconds() 
                         for i in range(len(timestamps))]
            
            # Calculate slope using least squares
            n = len(values)
            sum_x = sum(time_diffs)
            sum_y = sum(values)
            sum_xy = sum(time_diffs[i] * values[i] for i in range(n))
            sum_x2 = sum(x * x for x in time_diffs)
            
            if n * sum_x2 - sum_x * sum_x == 0:
                slope = 0
            else:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # Determine trend direction
            if abs(slope) < 0.01:
                direction = "stable"
            elif slope > 0:
                direction = "improving"
            else:
                direction = "declining"
            
            # Calculate confidence based on data consistency
            variance = sum((values[i] - sum_y/n) ** 2 for i in range(n)) / n
            confidence = max(0.1, min(1.0, 1.0 - variance))
            
            # Determine significance
            significance = "low"
            if abs(slope) > self.significance_thresholds['high']:
                significance = "high"
            elif abs(slope) > self.significance_thresholds['medium']:
                significance = "medium"
            
            return ImprovementTrend(
                metric_name=metric_name,
                time_period=timestamps[-1] - timestamps[0],
                trend_direction=direction,
                rate_of_change=slope,
                confidence=confidence,
                data_points=len(values),
                significance=significance
            )
            
        except Exception as e:
            logger.error(f"Error calculating trend for {metric_name}: {e}")
            return None
    
    async def generate_improvement_insights(self, 
                                          trends: List[ImprovementTrend],
                                          execution_history: List[Dict[str, Any]]) -> List[ImprovementInsight]:
        """Generate analytical insights from trends and execution history"""
        insights = []
        
        # Analyze overall improvement velocity
        velocity_insight = await self._analyze_improvement_velocity(trends)
        if velocity_insight:
            insights.append(velocity_insight)
        
        # Analyze improvement effectiveness
        effectiveness_insight = await self._analyze_improvement_effectiveness(execution_history)
        if effectiveness_insight:
            insights.append(effectiveness_insight)
        
        # Analyze improvement patterns
        patterns_insight = await self._analyze_improvement_patterns(trends, execution_history)
        if patterns_insight:
            insights.append(patterns_insight)
        
        # Analyze risk vs reward patterns
        risk_reward_insight = await self._analyze_risk_reward_patterns(execution_history)
        if risk_reward_insight:
            insights.append(risk_reward_insight)
        
        # Store insights
        self.insights_history.extend(insights)
        
        # Maintain insights history size
        if len(self.insights_history) > 100:
            self.insights_history = self.insights_history[-50:]
        
        return insights
    
    async def _analyze_improvement_velocity(self, trends: List[ImprovementTrend]) -> Optional[ImprovementInsight]:
        """Analyze the velocity of improvements"""
        if not trends:
            return None
        
        improving_trends = [t for t in trends if t.trend_direction == "improving"]
        declining_trends = [t for t in trends if t.trend_direction == "declining"]
        
        if not improving_trends and not declining_trends:
            return None
        
        avg_improvement_rate = sum(t.rate_of_change for t in improving_trends) / max(len(improving_trends), 1)
        avg_decline_rate = sum(abs(t.rate_of_change) for t in declining_trends) / max(len(declining_trends), 1)
        
        # Calculate overall velocity score
        velocity_score = (avg_improvement_rate - avg_decline_rate)
        
        insight_description = f"System improvement velocity analysis: {len(improving_trends)} metrics improving, {len(declining_trends)} declining"
        
        recommendations = []
        if velocity_score > 0.1:
            recommendations.extend([
                "Maintain current improvement strategies",
                "Consider increasing improvement frequency",
                "Document successful improvement patterns"
            ])
        elif velocity_score < -0.1:
            recommendations.extend([
                "Review recent changes for negative impacts",
                "Increase safety validation for improvements",
                "Focus on stabilizing declining metrics"
            ])
        else:
            recommendations.extend([
                "System appears stable - consider targeted optimizations",
                "Analyze improvement effectiveness",
                "Identify new areas for enhancement"
            ])
        
        return ImprovementInsight(
            insight_id=str(uuid4()),
            timestamp=datetime.utcnow(),
            insight_type="improvement_velocity",
            description=insight_description,
            evidence=[asdict(trend) for trend in trends],
            impact_assessment={
                'velocity_score': velocity_score,
                'improving_metrics': len(improving_trends),
                'declining_metrics': len(declining_trends)
            },
            recommendations=recommendations,
            confidence=min(1.0, sum(t.confidence for t in trends) / len(trends)),
            metadata={'trends_analyzed': len(trends)}
        )
    
    async def _analyze_improvement_effectiveness(self, 
                                               execution_history: List[Dict[str, Any]]) -> Optional[ImprovementInsight]:
        """Analyze the effectiveness of improvement executions"""
        if not execution_history:
            return None
        
        recent_executions = execution_history[-20:]  # Last 20 executions
        
        success_rate = sum(1 for ex in recent_executions if ex.get('success', False)) / len(recent_executions)
        
        # Analyze improvement magnitudes
        improvement_magnitudes = []
        for execution in recent_executions:
            if execution.get('improvement'):
                total_improvement = sum(abs(imp) for imp in execution['improvement'].values())
                improvement_magnitudes.append(total_improvement)
        
        avg_improvement = sum(improvement_magnitudes) / max(len(improvement_magnitudes), 1)
        
        # Effectiveness score
        effectiveness_score = success_rate * avg_improvement
        
        recommendations = []
        if effectiveness_score > 0.5:
            recommendations.extend([
                "Improvement effectiveness is high - maintain current approach",
                "Consider increasing improvement ambition",
                "Share successful patterns with other systems"
            ])
        elif effectiveness_score > 0.2:
            recommendations.extend([
                "Moderate effectiveness - review improvement selection criteria",
                "Analyze failed improvements for common patterns",
                "Consider smaller, incremental improvements"
            ])
        else:
            recommendations.extend([
                "Low improvement effectiveness detected",
                "Review improvement methodology and safety constraints",
                "Focus on understanding system behavior before major changes"
            ])
        
        return ImprovementInsight(
            insight_id=str(uuid4()),
            timestamp=datetime.utcnow(),
            insight_type="improvement_effectiveness",
            description=f"Improvement effectiveness analysis: {success_rate:.1%} success rate, {avg_improvement:.3f} average improvement",
            evidence=[{'executions_analyzed': len(recent_executions), 'success_rate': success_rate}],
            impact_assessment={
                'effectiveness_score': effectiveness_score,
                'success_rate': success_rate,
                'average_improvement': avg_improvement
            },
            recommendations=recommendations,
            confidence=min(1.0, len(recent_executions) / 20),
            metadata={'executions_analyzed': len(recent_executions)}
        )
    
    async def _analyze_improvement_patterns(self, trends: List[ImprovementTrend],
                                          execution_history: List[Dict[str, Any]]) -> Optional[ImprovementInsight]:
        """Analyze patterns in improvements"""
        if not trends and not execution_history:
            return None
        
        # Analyze timing patterns
        if execution_history:
            execution_times = [ex.get('timestamp', datetime.utcnow()) for ex in execution_history[-10:]]
            time_intervals = []
            for i in range(1, len(execution_times)):
                interval = (execution_times[i] - execution_times[i-1]).total_seconds() / 3600  # hours
                time_intervals.append(interval)
            
            avg_interval = sum(time_intervals) / max(len(time_intervals), 1) if time_intervals else 24
        else:
            avg_interval = 24
        
        # Analyze category patterns
        category_success = defaultdict(list)
        for execution in execution_history[-20:]:
            category = execution.get('category', 'unknown')
            success = execution.get('success', False)
            category_success[category].append(success)
        
        best_category = None
        best_success_rate = 0
        for category, successes in category_success.items():
            success_rate = sum(successes) / len(successes)
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_category = category
        
        recommendations = [
            f"Improvements typically occur every {avg_interval:.1f} hours",
            f"Most successful improvement category: {best_category} ({best_success_rate:.1%} success rate)"
        ]
        
        if avg_interval < 1:
            recommendations.append("High frequency of improvements - monitor for system instability")
        elif avg_interval > 48:
            recommendations.append("Low frequency of improvements - consider more proactive optimization")
        
        return ImprovementInsight(
            insight_id=str(uuid4()),
            timestamp=datetime.utcnow(),
            insight_type="improvement_patterns",
            description=f"Improvement pattern analysis: {avg_interval:.1f}h average interval, best category: {best_category}",
            evidence=[{'category_analysis': dict(category_success), 'timing_analysis': time_intervals}],
            impact_assessment={
                'average_interval_hours': avg_interval,
                'best_category': best_category,
                'best_success_rate': best_success_rate
            },
            recommendations=recommendations,
            confidence=0.8,
            metadata={'patterns_analyzed': len(category_success)}
        )
    
    async def _analyze_risk_reward_patterns(self, 
                                          execution_history: List[Dict[str, Any]]) -> Optional[ImprovementInsight]:
        """Analyze risk vs reward patterns in improvements"""
        if not execution_history:
            return None
        
        risk_reward_data = []
        for execution in execution_history[-15:]:
            risk = execution.get('risk_level', 0.5)
            success = execution.get('success', False)
            improvement = sum(abs(imp) for imp in execution.get('improvement', {}).values())
            
            risk_reward_data.append({
                'risk': risk,
                'success': success,
                'improvement': improvement,
                'reward': improvement if success else 0
            })
        
        if not risk_reward_data:
            return None
        
        # Analyze risk-reward correlation
        high_risk_executions = [d for d in risk_reward_data if d['risk'] > 0.6]
        low_risk_executions = [d for d in risk_reward_data if d['risk'] <= 0.3]
        
        high_risk_success = sum(1 for d in high_risk_executions if d['success']) / max(len(high_risk_executions), 1)
        low_risk_success = sum(1 for d in low_risk_executions if d['success']) / max(len(low_risk_executions), 1)
        
        high_risk_reward = sum(d['reward'] for d in high_risk_executions) / max(len(high_risk_executions), 1)
        low_risk_reward = sum(d['reward'] for d in low_risk_executions) / max(len(low_risk_executions), 1)
        
        recommendations = []
        if high_risk_reward > low_risk_reward * 1.5 and high_risk_success > 0.5:
            recommendations.extend([
                "High-risk improvements show good returns - consider selective high-risk strategies",
                "Ensure robust safety mechanisms for high-risk improvements"
            ])
        elif low_risk_success > high_risk_success:
            recommendations.extend([
                "Low-risk improvements more reliable - focus on incremental improvements",
                "Review high-risk improvement methodology"
            ])
        else:
            recommendations.extend([
                "Balanced risk-reward profile - maintain current risk management",
                "Consider optimizing risk assessment accuracy"
            ])
        
        return ImprovementInsight(
            insight_id=str(uuid4()),
            timestamp=datetime.utcnow(),
            insight_type="risk_reward_analysis",
            description=f"Risk-reward analysis: High-risk success {high_risk_success:.1%}, Low-risk success {low_risk_success:.1%}",
            evidence=[{'risk_reward_data': risk_reward_data}],
            impact_assessment={
                'high_risk_success_rate': high_risk_success,
                'low_risk_success_rate': low_risk_success,
                'high_risk_avg_reward': high_risk_reward,
                'low_risk_avg_reward': low_risk_reward
            },
            recommendations=recommendations,
            confidence=min(1.0, len(risk_reward_data) / 15),
            metadata={'data_points_analyzed': len(risk_reward_data)}
        )
    
    async def get_analytics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_insights = [i for i in self.insights_history if i.timestamp > cutoff_time]
        
        return {
            'analytics_id': self.analytics_id,
            'total_insights': len(recent_insights),
            'insights_by_type': {
                insight_type: len([i for i in recent_insights if i.insight_type == insight_type])
                for insight_type in set(i.insight_type for i in recent_insights)
            },
            'trends_cached': len(self.trends_cache),
            'recent_trends': {
                metric: {
                    'direction': trend.trend_direction,
                    'rate': trend.rate_of_change,
                    'significance': trend.significance
                }
                for metric, trend in self.trends_cache.items()
            },
            'top_recommendations': self._get_top_recommendations(recent_insights),
            'confidence_scores': {
                i.insight_type: i.confidence for i in recent_insights
            }
        }
    
    def _get_top_recommendations(self, insights: List[ImprovementInsight]) -> List[str]:
        """Get top recommendations from insights"""
        all_recommendations = []
        for insight in insights:
            all_recommendations.extend(insight.recommendations)
        
        # Count recommendation frequency
        rec_counts = defaultdict(int)
        for rec in all_recommendations:
            rec_counts[rec] += 1
        
        # Return top 5 most frequent recommendations
        return sorted(rec_counts.keys(), key=lambda x: rec_counts[x], reverse=True)[:5]
    
    async def cleanup(self):
        """Clean up analytics resources"""
        self.trends_cache.clear()
        logger.info(f"Improvement Analytics {self.analytics_id} cleaned up")