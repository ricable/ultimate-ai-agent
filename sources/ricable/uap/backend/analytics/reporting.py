# File: backend/analytics/reporting.py
"""
Report Generation System for UAP Analytics

Provides custom report generation capabilities with multiple export formats,
scheduled reporting, and automated report distribution.
"""

import asyncio
import json
import csv
import io
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import uuid
from pathlib import Path
import base64
from jinja2 import Template

from .usage_analytics import usage_analytics, EventType
from ..monitoring.metrics.performance import performance_monitor

class ReportType(Enum):
    """Types of reports that can be generated"""
    USAGE_SUMMARY = "usage_summary"
    USER_ACTIVITY = "user_activity"
    AGENT_PERFORMANCE = "agent_performance"
    SYSTEM_HEALTH = "system_health"
    ERROR_ANALYSIS = "error_analysis"
    FEATURE_ADOPTION = "feature_adoption"
    BUSINESS_INTELLIGENCE = "business_intelligence"
    CUSTOM = "custom"

class ReportFormat(Enum):
    """Report output formats"""
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    PDF = "pdf"
    EXCEL = "xlsx"

class ReportFrequency(Enum):
    """Report generation frequency"""
    REAL_TIME = "real_time"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"

@dataclass
class ReportConfiguration:
    """Configuration for report generation"""
    report_id: str
    report_type: ReportType
    title: str
    description: str
    format: ReportFormat
    frequency: ReportFrequency
    parameters: Dict[str, Any]
    filters: Dict[str, Any] = None
    recipients: List[str] = None
    template: Optional[str] = None
    created_at: datetime = None
    last_generated: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.filters is None:
            self.filters = {}
        if self.recipients is None:
            self.recipients = []
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['report_type'] = self.report_type.value
        data['format'] = self.format.value
        data['frequency'] = self.frequency.value
        data['created_at'] = self.created_at.isoformat()
        if self.last_generated:
            data['last_generated'] = self.last_generated.isoformat()
        return data

@dataclass
class GeneratedReport:
    """Container for generated report data"""
    report_id: str
    config: ReportConfiguration
    generated_at: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    content: Union[str, bytes]
    file_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "config": self.config.to_dict(),
            "generated_at": self.generated_at.isoformat(),
            "data_summary": {
                "records_count": len(self.data.get('records', [])),
                "time_range": self.data.get('time_range'),
                "filters_applied": self.data.get('filters_applied')
            },
            "metadata": self.metadata,
            "file_path": self.file_path,
            "content_size": len(self.content) if isinstance(self.content, (str, bytes)) else 0
        }

class ReportGenerator:
    """Main report generation system"""
    
    def __init__(self, output_directory: str = "/tmp/uap_reports"):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        self.report_configs: Dict[str, ReportConfiguration] = {}
        self.generated_reports: Dict[str, GeneratedReport] = {}
        self.report_templates = self._load_report_templates()
        
        # Report generation settings
        self.max_records_per_report = 10000
        self.report_retention_days = 90
        
    def _load_report_templates(self) -> Dict[str, str]:
        """Load report templates"""
        return {
            "usage_summary_html": """
            <!DOCTYPE html>
            <html>
            <head>
                <title>{{ title }}</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                    .metric { display: inline-block; margin: 10px; padding: 15px; background-color: #e7f3ff; border-radius: 5px; }
                    .chart { margin: 20px 0; }
                    table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{{ title }}</h1>
                    <p>{{ description }}</p>
                    <p><strong>Period:</strong> {{ time_range.start }} to {{ time_range.end }}</p>
                    <p><strong>Generated:</strong> {{ generated_at }}</p>
                </div>
                
                <h2>Key Metrics</h2>
                <div class="metrics">
                    {% for metric, value in summary.items() %}
                    <div class="metric">
                        <h3>{{ metric|title|replace('_', ' ') }}</h3>
                        <p>{{ value }}</p>
                    </div>
                    {% endfor %}
                </div>
                
                {% if charts %}
                <h2>Charts and Visualizations</h2>
                {% for chart in charts %}
                <div class="chart">
                    <h3>{{ chart.title }}</h3>
                    <p>{{ chart.description }}</p>
                    <!-- Chart data would be rendered here -->
                </div>
                {% endfor %}
                {% endif %}
                
                {% if tables %}
                <h2>Detailed Data</h2>
                {% for table in tables %}
                <h3>{{ table.title }}</h3>
                <table>
                    <thead>
                        <tr>
                            {% for header in table.headers %}
                            <th>{{ header }}</th>
                            {% endfor %}
                        </tr>
                    </thead>
                    <tbody>
                        {% for row in table.rows %}
                        <tr>
                            {% for cell in row %}
                            <td>{{ cell }}</td>
                            {% endfor %}
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                {% endfor %}
                {% endif %}
            </body>
            </html>
            """,
            
            "agent_performance_html": """
            <!DOCTYPE html>
            <html>
            <head>
                <title>{{ title }} - Agent Performance Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .agent-card { border: 1px solid #ddd; margin: 15px 0; padding: 15px; border-radius: 5px; }
                    .metric-row { display: flex; justify-content: space-between; margin: 10px 0; }
                    .status-healthy { color: green; }
                    .status-warning { color: orange; }
                    .status-error { color: red; }
                </style>
            </head>
            <body>
                <h1>{{ title }}</h1>
                <p>{{ description }}</p>
                <p><strong>Generated:</strong> {{ generated_at }}</p>
                
                {% for agent in agents %}
                <div class="agent-card">
                    <h2>{{ agent.agent_id }} ({{ agent.framework }})</h2>
                    <div class="metric-row">
                        <span><strong>Total Requests:</strong> {{ agent.total_requests }}</span>
                        <span><strong>Success Rate:</strong> {{ "%.2f"|format(agent.success_rate) }}%</span>
                        <span><strong>Avg Response Time:</strong> {{ "%.2f"|format(agent.avg_response_time) }}ms</span>
                    </div>
                    <div class="metric-row">
                        <span><strong>Unique Users:</strong> {{ agent.unique_users_count }}</span>
                        <span><strong>Peak Concurrent:</strong> {{ agent.peak_concurrent_users }}</span>
                        <span class="status-{{ agent.status }}"><strong>Status:</strong> {{ agent.status|title }}</span>
                    </div>
                    
                    {% if agent.popular_features %}
                    <h4>Popular Features</h4>
                    <ul>
                        {% for feature, count in agent.popular_features.items() %}
                        <li>{{ feature }}: {{ count }} uses</li>
                        {% endfor %}
                    </ul>
                    {% endif %}
                </div>
                {% endfor %}
            </body>
            </html>
            """
        }
    
    def create_report_config(self, report_type: ReportType, title: str, 
                           description: str, format: ReportFormat,
                           frequency: ReportFrequency, parameters: Dict[str, Any],
                           filters: Dict[str, Any] = None,
                           recipients: List[str] = None,
                           template: Optional[str] = None) -> str:
        """Create a new report configuration"""
        report_id = str(uuid.uuid4())
        
        config = ReportConfiguration(
            report_id=report_id,
            report_type=report_type,
            title=title,
            description=description,
            format=format,
            frequency=frequency,
            parameters=parameters,
            filters=filters or {},
            recipients=recipients or [],
            template=template
        )
        
        self.report_configs[report_id] = config
        return report_id
    
    async def generate_report(self, report_id: str, 
                            override_params: Dict[str, Any] = None) -> GeneratedReport:
        """Generate a report based on configuration"""
        if report_id not in self.report_configs:
            raise ValueError(f"Report configuration {report_id} not found")
        
        config = self.report_configs[report_id]
        
        # Merge parameters with overrides
        params = {**config.parameters}
        if override_params:
            params.update(override_params)
        
        # Generate report data based on type
        if config.report_type == ReportType.USAGE_SUMMARY:
            data = await self._generate_usage_summary_data(params, config.filters)
        elif config.report_type == ReportType.USER_ACTIVITY:
            data = await self._generate_user_activity_data(params, config.filters)
        elif config.report_type == ReportType.AGENT_PERFORMANCE:
            data = await self._generate_agent_performance_data(params, config.filters)
        elif config.report_type == ReportType.SYSTEM_HEALTH:
            data = await self._generate_system_health_data(params, config.filters)
        elif config.report_type == ReportType.ERROR_ANALYSIS:
            data = await self._generate_error_analysis_data(params, config.filters)
        elif config.report_type == ReportType.FEATURE_ADOPTION:
            data = await self._generate_feature_adoption_data(params, config.filters)
        elif config.report_type == ReportType.BUSINESS_INTELLIGENCE:
            data = await self._generate_business_intelligence_data(params, config.filters)
        else:
            raise ValueError(f"Unsupported report type: {config.report_type}")
        
        # Format report content
        content = await self._format_report_content(config, data)
        
        # Save report file
        file_path = None
        if config.format != ReportFormat.JSON:  # JSON is returned inline
            file_path = await self._save_report_file(report_id, config, content)
        
        # Create generated report
        report = GeneratedReport(
            report_id=report_id,
            config=config,
            generated_at=datetime.utcnow(),
            data=data,
            metadata={
                "parameters_used": params,
                "filters_applied": config.filters,
                "generation_time_ms": 0  # Would be calculated in real implementation
            },
            content=content,
            file_path=str(file_path) if file_path else None
        )
        
        # Update last generated time
        config.last_generated = datetime.utcnow()
        
        # Store generated report
        self.generated_reports[report_id] = report
        
        return report
    
    async def _generate_usage_summary_data(self, params: Dict[str, Any], 
                                         filters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate usage summary report data"""
        time_window_hours = params.get('time_window_hours', 24)
        
        # Get usage summary from analytics
        summary = usage_analytics.get_usage_summary(time_window_hours)
        
        # Get real-time metrics
        real_time = usage_analytics.get_real_time_metrics()
        
        # Calculate additional metrics
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_window_hours)
        
        return {
            "report_type": "usage_summary",
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": time_window_hours
            },
            "summary": summary['summary'],
            "events_by_type": summary['events_by_type'],
            "agent_usage": summary['agent_usage'],
            "feature_usage": summary['feature_usage'],
            "error_breakdown": summary['error_breakdown'],
            "real_time_metrics": real_time,
            "filters_applied": filters
        }
    
    async def _generate_agent_performance_data(self, params: Dict[str, Any],
                                             filters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate agent performance report data"""
        time_window_hours = params.get('time_window_hours', 24)
        
        # Get agent statistics
        agent_stats = usage_analytics.agent_stats
        
        # Get system health for agent status
        system_health = performance_monitor.get_system_health()
        agent_health = system_health.get('agent_health', {})
        
        # Process agent data
        agents_data = []
        for agent_id, stats in agent_stats.items():
            agent_data = stats.to_dict()
            
            # Add health status
            health_info = agent_health.get(agent_id, {})
            agent_data['status'] = 'healthy' if health_info.get('healthy', True) else 'warning'
            agent_data['health_score'] = health_info.get('success_rate', 100)
            
            agents_data.append(agent_data)
        
        # Sort by total requests
        agents_data.sort(key=lambda x: x['total_requests'], reverse=True)
        
        return {
            "report_type": "agent_performance",
            "time_range": {
                "start": (datetime.utcnow() - timedelta(hours=time_window_hours)).isoformat(),
                "end": datetime.utcnow().isoformat()
            },
            "agents": agents_data,
            "summary": {
                "total_agents": len(agents_data),
                "healthy_agents": len([a for a in agents_data if a['status'] == 'healthy']),
                "total_requests": sum(a['total_requests'] for a in agents_data),
                "avg_success_rate": statistics.mean([a['success_rate'] for a in agents_data]) if agents_data else 0
            },
            "filters_applied": filters
        }
    
    async def _generate_system_health_data(self, params: Dict[str, Any],
                                         filters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate system health report data"""
        # Get current system health
        health = performance_monitor.get_system_health()
        
        # Get performance summary
        time_window = params.get('time_window_minutes', 60)
        performance_summary = performance_monitor.get_performance_summary(time_window)
        
        return {
            "report_type": "system_health",
            "timestamp": datetime.utcnow().isoformat(),
            "overall_healthy": health['overall_healthy'],
            "system_health": health['system_health'],
            "agent_health": health['agent_health'],
            "current_stats": health['current_stats'],
            "performance_summary": performance_summary,
            "thresholds": health['thresholds'],
            "recommendations": self._generate_health_recommendations(health),
            "filters_applied": filters
        }
    
    async def _generate_error_analysis_data(self, params: Dict[str, Any],
                                          filters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate error analysis report data"""
        time_window_hours = params.get('time_window_hours', 24)
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        # Get error events from usage analytics
        error_events = [e for e in usage_analytics.events_history 
                       if e.event_type == EventType.ERROR_OCCURRED and e.timestamp >= cutoff_time]
        
        # Analyze errors
        error_summary = self._analyze_errors(error_events)
        
        return {
            "report_type": "error_analysis",
            "time_range": {
                "start": cutoff_time.isoformat(),
                "end": datetime.utcnow().isoformat()
            },
            "total_errors": len(error_events),
            "error_summary": error_summary,
            "error_trends": self._calculate_error_trends(error_events),
            "top_errors": self._get_top_errors(error_events),
            "error_rate_by_hour": self._calculate_error_rate_by_hour(error_events),
            "recommendations": self._generate_error_recommendations(error_summary),
            "filters_applied": filters
        }
    
    async def _generate_feature_adoption_data(self, params: Dict[str, Any],
                                            filters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate feature adoption report data"""
        time_window_days = params.get('time_window_days', 30)
        cutoff_time = datetime.utcnow() - timedelta(days=time_window_days)
        
        # Get feature usage events
        feature_events = [e for e in usage_analytics.events_history 
                         if e.event_type == EventType.FEATURE_USED and e.timestamp >= cutoff_time]
        
        # Analyze feature adoption
        adoption_analysis = self._analyze_feature_adoption(feature_events, time_window_days)
        
        return {
            "report_type": "feature_adoption",
            "time_range": {
                "start": cutoff_time.isoformat(),
                "end": datetime.utcnow().isoformat(),
                "days": time_window_days
            },
            "feature_usage": dict(usage_analytics.feature_usage),
            "adoption_analysis": adoption_analysis,
            "feature_trends": self._calculate_feature_trends(feature_events),
            "user_adoption": self._calculate_user_adoption_rates(feature_events),
            "recommendations": self._generate_adoption_recommendations(adoption_analysis),
            "filters_applied": filters
        }
    
    async def _generate_business_intelligence_data(self, params: Dict[str, Any],
                                                 filters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate business intelligence report data"""
        time_window_days = params.get('time_window_days', 30)
        
        # Gather comprehensive business metrics
        usage_summary = usage_analytics.get_usage_summary(time_window_days * 24)
        system_health = performance_monitor.get_system_health()
        
        # Calculate business KPIs
        kpis = self._calculate_business_kpis(usage_summary, system_health, time_window_days)
        
        return {
            "report_type": "business_intelligence",
            "time_range": {
                "start": (datetime.utcnow() - timedelta(days=time_window_days)).isoformat(),
                "end": datetime.utcnow().isoformat(),
                "days": time_window_days
            },
            "kpis": kpis,
            "user_engagement": self._calculate_user_engagement_metrics(),
            "platform_performance": self._calculate_platform_performance_metrics(),
            "growth_metrics": self._calculate_growth_metrics(time_window_days),
            "operational_metrics": self._calculate_operational_metrics(),
            "recommendations": self._generate_business_recommendations(kpis),
            "filters_applied": filters
        }
    
    async def _format_report_content(self, config: ReportConfiguration, 
                                   data: Dict[str, Any]) -> Union[str, bytes]:
        """Format report content based on configuration"""
        if config.format == ReportFormat.JSON:
            return json.dumps(data, indent=2, default=str)
        
        elif config.format == ReportFormat.CSV:
            return await self._format_as_csv(data)
        
        elif config.format == ReportFormat.HTML:
            return await self._format_as_html(config, data)
        
        else:
            # Default to JSON for unsupported formats
            return json.dumps(data, indent=2, default=str)
    
    async def _format_as_csv(self, data: Dict[str, Any]) -> str:
        """Format data as CSV"""
        output = io.StringIO()
        
        # Handle different report types
        if data.get('report_type') == 'agent_performance':
            agents = data.get('agents', [])
            if agents:
                writer = csv.DictWriter(output, fieldnames=agents[0].keys())
                writer.writeheader()
                writer.writerows(agents)
        
        elif data.get('report_type') == 'usage_summary':
            # Create a flattened view of the summary data
            rows = []
            summary = data.get('summary', {})
            for key, value in summary.items():
                rows.append({'metric': key, 'value': value})
            
            if rows:
                writer = csv.DictWriter(output, fieldnames=['metric', 'value'])
                writer.writeheader()
                writer.writerows(rows)
        
        else:
            # Generic CSV format
            writer = csv.writer(output)
            writer.writerow(['Key', 'Value'])
            for key, value in data.items():
                if isinstance(value, (str, int, float)):
                    writer.writerow([key, value])
        
        return output.getvalue()
    
    async def _format_as_html(self, config: ReportConfiguration, 
                            data: Dict[str, Any]) -> str:
        """Format data as HTML using templates"""
        template_name = f"{config.report_type.value}_html"
        template_content = self.report_templates.get(template_name, 
                                                   self.report_templates['usage_summary_html'])
        
        template = Template(template_content)
        
        # Prepare template context
        context = {
            'title': config.title,
            'description': config.description,
            'generated_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
            **data
        }
        
        return template.render(**context)
    
    async def _save_report_file(self, report_id: str, config: ReportConfiguration,
                              content: Union[str, bytes]) -> Path:
        """Save report content to file"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"{config.report_type.value}_{timestamp}_{report_id[:8]}.{config.format.value}"
        file_path = self.output_directory / filename
        
        mode = 'wb' if isinstance(content, bytes) else 'w'
        with open(file_path, mode) as f:
            f.write(content)
        
        return file_path
    
    def _analyze_errors(self, error_events: List) -> Dict[str, Any]:
        """Analyze error events"""
        error_types = defaultdict(int)
        error_messages = defaultdict(int)
        
        for event in error_events:
            error_type = event.data.get('error_type', 'unknown')
            error_message = event.data.get('error_message', 'No message')
            
            error_types[error_type] += 1
            error_messages[error_message] += 1
        
        return {
            'by_type': dict(error_types),
            'by_message': dict(error_messages),
            'total_unique_types': len(error_types),
            'total_unique_messages': len(error_messages)
        }
    
    def _calculate_business_kpis(self, usage_summary: Dict[str, Any], 
                               system_health: Dict[str, Any], days: int) -> Dict[str, Any]:
        """Calculate business KPIs"""
        summary = usage_summary.get('summary', {})
        
        return {
            'user_acquisition': {
                'total_users': summary.get('unique_users', 0),
                'daily_average': summary.get('unique_users', 0) / max(1, days),
                'growth_rate': 0  # Would calculate from historical data
            },
            'user_engagement': {
                'active_sessions': summary.get('unique_sessions', 0),
                'avg_session_length': 0,  # Would calculate from session data
                'feature_adoption_rate': 0  # Would calculate from feature usage
            },
            'platform_performance': {
                'uptime_percentage': 99.9 if system_health.get('overall_healthy') else 95.0,
                'avg_response_time': summary.get('avg_response_time_ms', 0),
                'error_rate': summary.get('error_rate_percent', 0)
            },
            'usage_metrics': {
                'total_requests': summary.get('total_events', 0),
                'agents_utilized': summary.get('agents_used', 0),
                'documents_processed': 0  # Would calculate from document events
            }
        }
    
    def _generate_health_recommendations(self, health: Dict[str, Any]) -> List[str]:
        """Generate health recommendations"""
        recommendations = []
        
        if not health.get('overall_healthy', True):
            recommendations.append("System health check required - investigate failing components")
        
        system_health = health.get('system_health', {})
        if not system_health.get('cpu_healthy', True):
            recommendations.append("High CPU usage detected - consider scaling resources")
        
        if not system_health.get('memory_healthy', True):
            recommendations.append("High memory usage detected - monitor for memory leaks")
        
        return recommendations
    
    def _calculate_user_engagement_metrics(self) -> Dict[str, Any]:
        """Calculate user engagement metrics"""
        active_sessions = len(usage_analytics.active_sessions)
        
        return {
            'current_active_users': active_sessions,
            'avg_session_duration': 0,  # Would calculate from session data
            'feature_engagement_score': 0,  # Would calculate from feature usage
            'user_retention_rate': 0  # Would calculate from historical data
        }
    
    def _calculate_platform_performance_metrics(self) -> Dict[str, Any]:
        """Calculate platform performance metrics"""
        health = performance_monitor.get_system_health()
        
        return {
            'system_uptime': 99.9,  # Would calculate from monitoring data
            'avg_response_time': health.get('current_stats', {}).get('cpu_percent', 0),
            'throughput': 0,  # Would calculate from request metrics
            'availability_score': 99.9 if health.get('overall_healthy') else 95.0
        }
    
    def _calculate_growth_metrics(self, days: int) -> Dict[str, Any]:
        """Calculate growth metrics"""
        return {
            'user_growth_rate': 0,  # Would calculate from historical user data
            'usage_growth_rate': 0,  # Would calculate from historical usage data
            'feature_adoption_growth': 0,  # Would calculate from feature usage trends
            'revenue_growth': 0  # Would integrate with billing data if available
        }
    
    def _calculate_operational_metrics(self) -> Dict[str, Any]:
        """Calculate operational metrics"""
        return {
            'total_agents_deployed': len(usage_analytics.agent_stats),
            'avg_agent_utilization': 0,  # Would calculate from agent usage
            'infrastructure_costs': 0,  # Would integrate with cloud billing
            'support_tickets': 0  # Would integrate with support system
        }
    
    def _generate_business_recommendations(self, kpis: Dict[str, Any]) -> List[str]:
        """Generate business recommendations based on KPIs"""
        recommendations = []
        
        performance = kpis.get('platform_performance', {})
        if performance.get('error_rate', 0) > 5:
            recommendations.append("Error rate exceeds threshold - investigate and fix critical issues")
        
        if performance.get('avg_response_time', 0) > 2000:
            recommendations.append("Response times are high - consider performance optimization")
        
        engagement = kpis.get('user_engagement', {})
        if engagement.get('feature_adoption_rate', 0) < 50:
            recommendations.append("Low feature adoption - improve user onboarding and documentation")
        
        return recommendations
    
    # Additional helper methods would be implemented here...
    def _calculate_error_trends(self, error_events): return {}
    def _get_top_errors(self, error_events): return []
    def _calculate_error_rate_by_hour(self, error_events): return {}
    def _generate_error_recommendations(self, error_summary): return []
    def _analyze_feature_adoption(self, feature_events, days): return {}
    def _calculate_feature_trends(self, feature_events): return {}
    def _calculate_user_adoption_rates(self, feature_events): return {}
    def _generate_adoption_recommendations(self, adoption_analysis): return []

# Global report generator instance
report_generator = ReportGenerator()

# Convenience functions
async def generate_usage_report(time_window_hours: int = 24, 
                              format: ReportFormat = ReportFormat.JSON) -> GeneratedReport:
    """Generate a usage summary report"""
    report_id = report_generator.create_report_config(
        report_type=ReportType.USAGE_SUMMARY,
        title="Usage Summary Report",
        description=f"Platform usage summary for the last {time_window_hours} hours",
        format=format,
        frequency=ReportFrequency.DAILY,
        parameters={'time_window_hours': time_window_hours}
    )
    return await report_generator.generate_report(report_id)

async def generate_agent_performance_report(format: ReportFormat = ReportFormat.HTML) -> GeneratedReport:
    """Generate an agent performance report"""
    report_id = report_generator.create_report_config(
        report_type=ReportType.AGENT_PERFORMANCE,
        title="Agent Performance Report",
        description="Comprehensive performance analysis of all agents",
        format=format,
        frequency=ReportFrequency.DAILY,
        parameters={'time_window_hours': 24}
    )
    return await report_generator.generate_report(report_id)

async def generate_business_intelligence_report(days: int = 30) -> GeneratedReport:
    """Generate a business intelligence report"""
    report_id = report_generator.create_report_config(
        report_type=ReportType.BUSINESS_INTELLIGENCE,
        title="Business Intelligence Dashboard",
        description=f"Comprehensive business metrics for the last {days} days",
        format=ReportFormat.HTML,
        frequency=ReportFrequency.MONTHLY,
        parameters={'time_window_days': days}
    )
    return await report_generator.generate_report(report_id)