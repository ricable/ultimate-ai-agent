# File: backend/config/bi_config.py
"""
Business Intelligence Configuration for UAP Platform

Provides configuration settings for analytics, reporting, and BI dashboards.
"""

import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class AnalyticsLevel(Enum):
    """Analytics detail levels"""
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    ENTERPRISE = "enterprise"

class MetricFrequency(Enum):
    """Metric collection frequencies"""
    REAL_TIME = "real_time"  # Continuous
    MINUTE = "minute"        # Every minute
    FIVE_MINUTES = "5min"    # Every 5 minutes
    HOURLY = "hourly"        # Every hour
    DAILY = "daily"          # Once per day

@dataclass
class KPIConfiguration:
    """Configuration for Key Performance Indicators"""
    name: str
    description: str
    metric_source: str
    calculation_method: str
    target_value: Optional[float] = None
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    unit: str = ""
    display_format: str = "number"  # number, percentage, currency, duration
    frequency: MetricFrequency = MetricFrequency.HOURLY
    enabled: bool = True

@dataclass
class DashboardConfiguration:
    """Configuration for BI dashboards"""
    dashboard_id: str
    name: str
    description: str
    kpis: List[str]  # List of KPI names
    charts: List[Dict[str, Any]]
    refresh_interval_seconds: int = 300  # 5 minutes
    access_roles: List[str] = field(default_factory=lambda: ["admin", "manager"])
    enabled: bool = True

@dataclass
class ReportConfiguration:
    """Configuration for automated reports"""
    report_id: str
    name: str
    description: str
    report_type: str
    schedule: str  # Cron expression
    format: str = "html"
    recipients: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

@dataclass
class PredictiveConfiguration:
    """Configuration for predictive analytics"""
    model_type: str
    prediction_type: str
    training_frequency_hours: int = 24
    min_training_samples: int = 100
    retrain_threshold_accuracy: float = 0.8
    prediction_horizon_hours: int = 24
    confidence_threshold: float = 0.7
    enabled: bool = True

@dataclass
class ABTestConfiguration:
    """Configuration for A/B testing"""
    max_experiments_per_user: int = 5
    default_confidence_level: float = 0.95
    minimum_sample_size: int = 100
    auto_stop_significant: bool = False
    auto_stop_days: Optional[int] = 30
    statistical_engine: str = "frequentist"  # frequentist, bayesian
    enabled: bool = True

@dataclass
class AlertConfiguration:
    """Configuration for BI alerts"""
    alert_id: str
    name: str
    description: str
    metric_source: str
    condition: str  # e.g., "> 0.95", "< 100", "!= expected"
    severity: str = "warning"  # info, warning, critical
    notification_channels: List[str] = field(default_factory=list)  # email, slack, webhook
    cooldown_minutes: int = 60
    enabled: bool = True

class BIConfig:
    """Main Business Intelligence Configuration"""
    
    def __init__(self):
        self.analytics_level = AnalyticsLevel(os.getenv("ANALYTICS_LEVEL", "standard"))
        self.data_retention_days = int(os.getenv("BI_DATA_RETENTION_DAYS", "90"))
        self.real_time_enabled = os.getenv("BI_REAL_TIME_ENABLED", "true").lower() == "true"
        self.export_enabled = os.getenv("BI_EXPORT_ENABLED", "true").lower() == "true"
        self.predictive_enabled = os.getenv("BI_PREDICTIVE_ENABLED", "true").lower() == "true"
        self.ab_testing_enabled = os.getenv("BI_AB_TESTING_ENABLED", "true").lower() == "true"
        
        # Load configurations
        self.kpis = self._load_kpi_configurations()
        self.dashboards = self._load_dashboard_configurations()
        self.reports = self._load_report_configurations()
        self.predictive = self._load_predictive_configurations()
        self.ab_testing = self._load_ab_test_configuration()
        self.alerts = self._load_alert_configurations()
    
    def _load_kpi_configurations(self) -> Dict[str, KPIConfiguration]:
        """Load KPI configurations"""
        return {
            "total_users": KPIConfiguration(
                name="Total Users",
                description="Total number of registered users",
                metric_source="user_sessions",
                calculation_method="count_distinct(user_id)",
                target_value=1000,
                threshold_warning=50,
                threshold_critical=10,
                display_format="number",
                frequency=MetricFrequency.HOURLY
            ),
            "active_sessions": KPIConfiguration(
                name="Active Sessions",
                description="Number of currently active user sessions",
                metric_source="user_sessions",
                calculation_method="count(status='active')",
                target_value=100,
                threshold_warning=10,
                threshold_critical=5,
                display_format="number",
                frequency=MetricFrequency.REAL_TIME
            ),
            "response_time": KPIConfiguration(
                name="Average Response Time",
                description="Average agent response time in milliseconds",
                metric_source="agent_usage",
                calculation_method="avg(response_time_ms)",
                target_value=1000,
                threshold_warning=2000,
                threshold_critical=5000,
                unit="ms",
                display_format="number",
                frequency=MetricFrequency.MINUTE
            ),
            "error_rate": KPIConfiguration(
                name="Error Rate",
                description="Percentage of requests resulting in errors",
                metric_source="agent_usage",
                calculation_method="(count(success=false) / count(*)) * 100",
                target_value=1.0,
                threshold_warning=5.0,
                threshold_critical=10.0,
                unit="%",
                display_format="percentage",
                frequency=MetricFrequency.FIVE_MINUTES
            ),
            "cost_per_request": KPIConfiguration(
                name="Cost per Request",
                description="Average cost per agent request",
                metric_source="agent_usage",
                calculation_method="avg(cost_total)",
                target_value=0.01,
                threshold_warning=0.05,
                threshold_critical=0.10,
                unit="$",
                display_format="currency",
                frequency=MetricFrequency.HOURLY
            ),
            "user_satisfaction": KPIConfiguration(
                name="User Satisfaction",
                description="Average user satisfaction score",
                metric_source="feedback",
                calculation_method="avg(rating)",
                target_value=4.5,
                threshold_warning=3.5,
                threshold_critical=2.5,
                display_format="number",
                frequency=MetricFrequency.DAILY
            ),
            "platform_uptime": KPIConfiguration(
                name="Platform Uptime",
                description="Platform availability percentage",
                metric_source="system_metrics",
                calculation_method="uptime_percentage",
                target_value=99.9,
                threshold_warning=99.0,
                threshold_critical=95.0,
                unit="%",
                display_format="percentage",
                frequency=MetricFrequency.MINUTE
            ),
            "feature_adoption": KPIConfiguration(
                name="Feature Adoption Rate",
                description="Percentage of users using new features",
                metric_source="feature_usage",
                calculation_method="adoption_rate",
                target_value=70.0,
                threshold_warning=50.0,
                threshold_critical=30.0,
                unit="%",
                display_format="percentage",
                frequency=MetricFrequency.DAILY
            )
        }
    
    def _load_dashboard_configurations(self) -> Dict[str, DashboardConfiguration]:
        """Load dashboard configurations"""
        return {
            "executive": DashboardConfiguration(
                dashboard_id="executive",
                name="Executive Dashboard",
                description="High-level KPIs and business metrics",
                kpis=["total_users", "active_sessions", "user_satisfaction", "platform_uptime"],
                charts=[
                    {
                        "type": "metric_cards",
                        "metrics": ["total_users", "active_sessions", "user_satisfaction", "platform_uptime"]
                    },
                    {
                        "type": "line_chart",
                        "title": "User Growth Trend",
                        "metric": "total_users",
                        "time_range": "30d"
                    },
                    {
                        "type": "gauge_chart",
                        "title": "Platform Health",
                        "metric": "platform_uptime"
                    }
                ],
                refresh_interval_seconds=300,
                access_roles=["admin", "executive"]
            ),
            "operations": DashboardConfiguration(
                dashboard_id="operations",
                name="Operations Dashboard",
                description="Technical performance and system metrics",
                kpis=["response_time", "error_rate", "platform_uptime", "cost_per_request"],
                charts=[
                    {
                        "type": "metric_cards",
                        "metrics": ["response_time", "error_rate", "platform_uptime"]
                    },
                    {
                        "type": "line_chart",
                        "title": "Response Time Trend",
                        "metric": "response_time",
                        "time_range": "24h"
                    },
                    {
                        "type": "area_chart",
                        "title": "Error Rate",
                        "metric": "error_rate",
                        "time_range": "24h"
                    },
                    {
                        "type": "bar_chart",
                        "title": "Cost Analysis",
                        "metric": "cost_per_request",
                        "time_range": "7d"
                    }
                ],
                refresh_interval_seconds=60,
                access_roles=["admin", "ops"]
            ),
            "analytics": DashboardConfiguration(
                dashboard_id="analytics",
                name="Analytics Dashboard",
                description="User behavior and feature adoption analytics",
                kpis=["total_users", "active_sessions", "feature_adoption", "user_satisfaction"],
                charts=[
                    {
                        "type": "metric_cards",
                        "metrics": ["total_users", "active_sessions", "feature_adoption"]
                    },
                    {
                        "type": "funnel_chart",
                        "title": "User Journey",
                        "stages": ["registration", "first_session", "feature_use", "retention"]
                    },
                    {
                        "type": "heat_map",
                        "title": "Feature Usage Patterns",
                        "metric": "feature_usage"
                    },
                    {
                        "type": "cohort_chart",
                        "title": "User Retention",
                        "metric": "user_retention"
                    }
                ],
                refresh_interval_seconds=300,
                access_roles=["admin", "analyst"]
            )
        }
    
    def _load_report_configurations(self) -> Dict[str, ReportConfiguration]:
        """Load automated report configurations"""
        return {
            "daily_summary": ReportConfiguration(
                report_id="daily_summary",
                name="Daily Summary Report",
                description="Daily platform performance and usage summary",
                report_type="usage_summary",
                schedule="0 8 * * *",  # Daily at 8 AM
                format="html",
                recipients=["ops@company.com"],
                filters={"time_range": "24h"}
            ),
            "weekly_business": ReportConfiguration(
                report_id="weekly_business",
                name="Weekly Business Report",
                description="Weekly business intelligence and KPI report",
                report_type="business_intelligence",
                schedule="0 9 * * 1",  # Mondays at 9 AM
                format="html",
                recipients=["executives@company.com"],
                filters={"time_range": "7d"}
            ),
            "monthly_analytics": ReportConfiguration(
                report_id="monthly_analytics",
                name="Monthly Analytics Report",
                description="Comprehensive monthly analytics and insights",
                report_type="comprehensive",
                schedule="0 10 1 * *",  # First day of month at 10 AM
                format="pdf",
                recipients=["leadership@company.com"],
                filters={"time_range": "30d", "include_predictions": True}
            )
        }
    
    def _load_predictive_configurations(self) -> Dict[str, PredictiveConfiguration]:
        """Load predictive analytics configurations"""
        return {
            "usage_forecast": PredictiveConfiguration(
                model_type="random_forest",
                prediction_type="usage_forecast",
                training_frequency_hours=24,
                min_training_samples=100,
                retrain_threshold_accuracy=0.8,
                prediction_horizon_hours=24,
                confidence_threshold=0.7
            ),
            "anomaly_detection": PredictiveConfiguration(
                model_type="isolation_forest",
                prediction_type="anomaly_detection",
                training_frequency_hours=12,
                min_training_samples=50,
                retrain_threshold_accuracy=0.85,
                prediction_horizon_hours=1,
                confidence_threshold=0.8
            ),
            "capacity_planning": PredictiveConfiguration(
                model_type="linear_regression",
                prediction_type="capacity_planning",
                training_frequency_hours=48,
                min_training_samples=200,
                retrain_threshold_accuracy=0.75,
                prediction_horizon_hours=168,  # 1 week
                confidence_threshold=0.7
            )
        }
    
    def _load_ab_test_configuration(self) -> ABTestConfiguration:
        """Load A/B testing configuration"""
        return ABTestConfiguration(
            max_experiments_per_user=int(os.getenv("AB_MAX_EXPERIMENTS_PER_USER", "5")),
            default_confidence_level=float(os.getenv("AB_DEFAULT_CONFIDENCE", "0.95")),
            minimum_sample_size=int(os.getenv("AB_MIN_SAMPLE_SIZE", "100")),
            auto_stop_significant=os.getenv("AB_AUTO_STOP_SIGNIFICANT", "false").lower() == "true",
            auto_stop_days=int(os.getenv("AB_AUTO_STOP_DAYS", "30")),
            statistical_engine=os.getenv("AB_STATISTICAL_ENGINE", "frequentist"),
            enabled=self.ab_testing_enabled
        )
    
    def _load_alert_configurations(self) -> Dict[str, AlertConfiguration]:
        """Load alert configurations"""
        return {
            "high_error_rate": AlertConfiguration(
                alert_id="high_error_rate",
                name="High Error Rate",
                description="Error rate exceeds threshold",
                metric_source="error_rate",
                condition="> 5.0",
                severity="critical",
                notification_channels=["email", "slack"],
                cooldown_minutes=30
            ),
            "slow_response_time": AlertConfiguration(
                alert_id="slow_response_time",
                name="Slow Response Time",
                description="Average response time is too high",
                metric_source="response_time",
                condition="> 2000",
                severity="warning",
                notification_channels=["email"],
                cooldown_minutes=60
            ),
            "low_user_satisfaction": AlertConfiguration(
                alert_id="low_user_satisfaction",
                name="Low User Satisfaction",
                description="User satisfaction score is below target",
                metric_source="user_satisfaction",
                condition="< 3.5",
                severity="warning",
                notification_channels=["email"],
                cooldown_minutes=120
            ),
            "system_anomaly": AlertConfiguration(
                alert_id="system_anomaly",
                name="System Anomaly Detected",
                description="Predictive model detected system anomaly",
                metric_source="anomaly_score",
                condition="> 0.7",
                severity="critical",
                notification_channels=["email", "slack", "webhook"],
                cooldown_minutes=15
            )
        }
    
    def get_kpi_config(self, kpi_name: str) -> Optional[KPIConfiguration]:
        """Get configuration for a specific KPI"""
        return self.kpis.get(kpi_name)
    
    def get_dashboard_config(self, dashboard_id: str) -> Optional[DashboardConfiguration]:
        """Get configuration for a specific dashboard"""
        return self.dashboards.get(dashboard_id)
    
    def get_enabled_kpis(self) -> List[KPIConfiguration]:
        """Get all enabled KPIs"""
        return [kpi for kpi in self.kpis.values() if kpi.enabled]
    
    def get_kpis_by_frequency(self, frequency: MetricFrequency) -> List[KPIConfiguration]:
        """Get KPIs by collection frequency"""
        return [kpi for kpi in self.kpis.values() if kpi.frequency == frequency and kpi.enabled]
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a BI feature is enabled"""
        feature_map = {
            "real_time": self.real_time_enabled,
            "export": self.export_enabled,
            "predictive": self.predictive_enabled,
            "ab_testing": self.ab_testing_enabled
        }
        return feature_map.get(feature, False)
    
    def get_retention_policy(self) -> int:
        """Get data retention policy in days"""
        return self.data_retention_days
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "analytics_level": self.analytics_level.value,
            "data_retention_days": self.data_retention_days,
            "features": {
                "real_time_enabled": self.real_time_enabled,
                "export_enabled": self.export_enabled,
                "predictive_enabled": self.predictive_enabled,
                "ab_testing_enabled": self.ab_testing_enabled
            },
            "kpis_count": len(self.kpis),
            "dashboards_count": len(self.dashboards),
            "reports_count": len(self.reports),
            "alerts_count": len(self.alerts)
        }

# Global BI configuration instance
bi_config = BIConfig()

# Helper functions
def get_bi_config() -> BIConfig:
    """Get the global BI configuration"""
    return bi_config

def is_analytics_enabled() -> bool:
    """Check if analytics is enabled"""
    return bi_config.analytics_level != AnalyticsLevel.BASIC

def get_analytics_level() -> AnalyticsLevel:
    """Get current analytics level"""
    return bi_config.analytics_level

def get_kpi_thresholds(kpi_name: str) -> Dict[str, Optional[float]]:
    """Get KPI thresholds for alerting"""
    kpi = bi_config.get_kpi_config(kpi_name)
    if not kpi:
        return {}
    
    return {
        "target": kpi.target_value,
        "warning": kpi.threshold_warning,
        "critical": kpi.threshold_critical
    }
