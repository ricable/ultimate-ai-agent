# File: backend/analytics/usage_analytics.py
"""
Usage Analytics System for UAP Platform

Tracks and analyzes user behavior, agent usage patterns, and system utilization.
Provides insights into platform adoption, feature usage, and performance trends.
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

from ..monitoring.metrics.prometheus_metrics import prometheus_metrics
from ..monitoring.metrics.performance import performance_monitor

class EventType(Enum):
    """Event types for usage tracking"""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    AGENT_REQUEST = "agent_request"
    WEBSOCKET_CONNECT = "websocket_connect"
    WEBSOCKET_DISCONNECT = "websocket_disconnect"
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_PROCESS = "document_process"
    ERROR_OCCURRED = "error_occurred"
    FEATURE_USED = "feature_used"
    PAGE_VIEW = "page_view"
    SESSION_START = "session_start"
    SESSION_END = "session_end"

@dataclass
class UsageEvent:
    """Container for usage tracking events"""
    event_id: str
    event_type: EventType
    user_id: Optional[str]
    session_id: Optional[str]
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "metadata": self.metadata or {}
        }

@dataclass
class UserSessionData:
    """User session tracking data"""
    session_id: str
    user_id: str
    start_time: datetime
    last_activity: datetime
    events_count: int = 0
    pages_visited: int = 0
    agents_used: set = None
    features_used: set = None
    total_response_time: float = 0.0
    errors_encountered: int = 0
    
    def __post_init__(self):
        if self.agents_used is None:
            self.agents_used = set()
        if self.features_used is None:
            self.features_used = set()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "start_time": self.start_time.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "duration_minutes": (self.last_activity - self.start_time).total_seconds() / 60,
            "events_count": self.events_count,
            "pages_visited": self.pages_visited,
            "agents_used": list(self.agents_used),
            "features_used": list(self.features_used),
            "avg_response_time": self.total_response_time / max(1, self.events_count),
            "errors_encountered": self.errors_encountered
        }

@dataclass
class AgentUsageStats:
    """Agent usage statistics"""
    agent_id: str
    framework: str
    total_requests: int = 0
    unique_users: set = None
    total_response_time: float = 0.0
    success_count: int = 0
    error_count: int = 0
    peak_concurrent_users: int = 0
    popular_features: Dict[str, int] = None
    usage_by_hour: Dict[int, int] = None
    
    def __post_init__(self):
        if self.unique_users is None:
            self.unique_users = set()
        if self.popular_features is None:
            self.popular_features = defaultdict(int)
        if self.usage_by_hour is None:
            self.usage_by_hour = defaultdict(int)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "framework": self.framework,
            "total_requests": self.total_requests,
            "unique_users_count": len(self.unique_users),
            "avg_response_time": self.total_response_time / max(1, self.total_requests),
            "success_rate": (self.success_count / max(1, self.total_requests)) * 100,
            "error_rate": (self.error_count / max(1, self.total_requests)) * 100,
            "peak_concurrent_users": self.peak_concurrent_users,
            "popular_features": dict(self.popular_features),
            "usage_by_hour": dict(self.usage_by_hour)
        }

class UsageAnalytics:
    """Main usage analytics system"""
    
    def __init__(self, max_events_history: int = 100000):
        self.max_events_history = max_events_history
        self.events_history: deque = deque(maxlen=max_events_history)
        self.active_sessions: Dict[str, UserSessionData] = {}
        self.agent_stats: Dict[str, AgentUsageStats] = {}
        self.user_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.feature_usage: Dict[str, int] = defaultdict(int)
        self.error_tracking: Dict[str, int] = defaultdict(int)
        self.lock = Lock()
        
        # Analytics configuration
        self.session_timeout_minutes = 30
        self.batch_size = 1000
        self.retention_days = 90
        
        # Start background processing
        self.processing_active = True
    
    def track_event(self, event_type: EventType, user_id: Optional[str] = None,
                   session_id: Optional[str] = None, data: Dict[str, Any] = None,
                   metadata: Dict[str, Any] = None) -> str:
        """Track a usage event"""
        event_id = str(uuid.uuid4())
        event = UsageEvent(
            event_id=event_id,
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.utcnow(),
            data=data or {},
            metadata=metadata or {}
        )
        
        with self.lock:
            self.events_history.append(event)
            
            # Update session data if applicable
            if session_id and session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.last_activity = datetime.utcnow()
                session.events_count += 1
                
                # Track specific event types
                if event_type == EventType.AGENT_REQUEST:
                    agent_id = data.get('agent_id')
                    if agent_id:
                        session.agents_used.add(agent_id)
                        response_time = data.get('response_time', 0)
                        session.total_response_time += response_time
                        
                elif event_type == EventType.ERROR_OCCURRED:
                    session.errors_encountered += 1
                    
                elif event_type == EventType.FEATURE_USED:
                    feature = data.get('feature')
                    if feature:
                        session.features_used.add(feature)
                        
                elif event_type == EventType.PAGE_VIEW:
                    session.pages_visited += 1
            
            # Update agent statistics
            if event_type == EventType.AGENT_REQUEST:
                self._update_agent_stats(event)
            
            # Update feature usage
            if event_type == EventType.FEATURE_USED:
                feature = data.get('feature')
                if feature:
                    self.feature_usage[feature] += 1
            
            # Track errors
            if event_type == EventType.ERROR_OCCURRED:
                error_type = data.get('error_type', 'unknown')
                self.error_tracking[error_type] += 1
        
        return event_id
    
    def start_session(self, user_id: str, session_id: Optional[str] = None,
                     metadata: Dict[str, Any] = None) -> str:
        """Start a user session"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        session = UserSessionData(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.utcnow(),
            last_activity=datetime.utcnow()
        )
        
        with self.lock:
            self.active_sessions[session_id] = session
        
        # Track session start event
        self.track_event(
            EventType.SESSION_START,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata
        )
        
        return session_id
    
    def end_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """End a user session and return session summary"""
        with self.lock:
            if session_id not in self.active_sessions:
                return None
            
            session = self.active_sessions.pop(session_id)
            session.last_activity = datetime.utcnow()
        
        # Track session end event
        self.track_event(
            EventType.SESSION_END,
            user_id=session.user_id,
            session_id=session_id,
            data=session.to_dict()
        )
        
        return session.to_dict()
    
    def _update_agent_stats(self, event: UsageEvent):
        """Update agent usage statistics"""
        data = event.data
        agent_id = data.get('agent_id')
        framework = data.get('framework')
        
        if not agent_id or not framework:
            return
        
        if agent_id not in self.agent_stats:
            self.agent_stats[agent_id] = AgentUsageStats(
                agent_id=agent_id,
                framework=framework
            )
        
        stats = self.agent_stats[agent_id]
        stats.total_requests += 1
        
        if event.user_id:
            stats.unique_users.add(event.user_id)
        
        response_time = data.get('response_time', 0)
        stats.total_response_time += response_time
        
        success = data.get('success', True)
        if success:
            stats.success_count += 1
        else:
            stats.error_count += 1
        
        # Track usage by hour
        hour = event.timestamp.hour
        stats.usage_by_hour[hour] += 1
        
        # Track features used
        features = data.get('features_used', [])
        for feature in features:
            stats.popular_features[feature] += 1
    
    def get_usage_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get usage summary for specified time window"""
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        # Filter recent events
        recent_events = [e for e in self.events_history if e.timestamp >= cutoff_time]
        
        # Calculate summary statistics
        total_events = len(recent_events)
        unique_users = len(set(e.user_id for e in recent_events if e.user_id))
        unique_sessions = len(set(e.session_id for e in recent_events if e.session_id))
        
        # Group events by type
        events_by_type = defaultdict(int)
        for event in recent_events:
            events_by_type[event.event_type.value] += 1
        
        # Calculate agent usage
        agent_requests = [e for e in recent_events if e.event_type == EventType.AGENT_REQUEST]
        agents_used = set(e.data.get('agent_id') for e in agent_requests if e.data.get('agent_id'))
        
        # Calculate error rates
        error_events = [e for e in recent_events if e.event_type == EventType.ERROR_OCCURRED]
        error_rate = (len(error_events) / max(1, total_events)) * 100
        
        # Calculate response times
        response_times = [e.data.get('response_time', 0) for e in agent_requests 
                         if e.data.get('response_time')]
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        return {
            "time_window_hours": time_window_hours,
            "summary": {
                "total_events": total_events,
                "unique_users": unique_users,
                "unique_sessions": unique_sessions,
                "agents_used": len(agents_used),
                "error_rate_percent": error_rate,
                "avg_response_time_ms": avg_response_time
            },
            "events_by_type": dict(events_by_type),
            "agent_usage": {aid: stats.to_dict() for aid, stats in self.agent_stats.items()},
            "feature_usage": dict(self.feature_usage),
            "error_breakdown": dict(self.error_tracking),
            "active_sessions": len(self.active_sessions)
        }
    
    def get_user_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get analytics for a specific user"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        # Filter user events
        user_events = [e for e in self.events_history 
                      if e.user_id == user_id and e.timestamp >= cutoff_time]
        
        # Calculate user statistics
        total_events = len(user_events)
        sessions = set(e.session_id for e in user_events if e.session_id)
        agents_used = set(e.data.get('agent_id') for e in user_events 
                         if e.event_type == EventType.AGENT_REQUEST and e.data.get('agent_id'))
        features_used = set(e.data.get('feature') for e in user_events 
                           if e.event_type == EventType.FEATURE_USED and e.data.get('feature'))
        
        # Calculate activity patterns
        events_by_day = defaultdict(int)
        events_by_hour = defaultdict(int)
        for event in user_events:
            day = event.timestamp.date().isoformat()
            hour = event.timestamp.hour
            events_by_day[day] += 1
            events_by_hour[hour] += 1
        
        # Calculate response time statistics
        agent_events = [e for e in user_events if e.event_type == EventType.AGENT_REQUEST]
        response_times = [e.data.get('response_time', 0) for e in agent_events 
                         if e.data.get('response_time')]
        
        return {
            "user_id": user_id,
            "time_period_days": days,
            "summary": {
                "total_events": total_events,
                "total_sessions": len(sessions),
                "agents_used": list(agents_used),
                "features_used": list(features_used),
                "avg_response_time": statistics.mean(response_times) if response_times else 0,
                "first_activity": min(e.timestamp for e in user_events).isoformat() if user_events else None,
                "last_activity": max(e.timestamp for e in user_events).isoformat() if user_events else None
            },
            "activity_patterns": {
                "events_by_day": dict(events_by_day),
                "events_by_hour": dict(events_by_hour)
            },
            "performance": {
                "response_times": response_times,
                "avg_response_time": statistics.mean(response_times) if response_times else 0,
                "p95_response_time": statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else 0
            }
        }
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time analytics metrics"""
        current_time = datetime.utcnow()
        
        # Active sessions
        active_sessions_count = len(self.active_sessions)
        
        # Recent activity (last 5 minutes)
        recent_cutoff = current_time - timedelta(minutes=5)
        recent_events = [e for e in self.events_history if e.timestamp >= recent_cutoff]
        
        # Current agent usage
        current_agent_usage = defaultdict(int)
        for session in self.active_sessions.values():
            for agent_id in session.agents_used:
                current_agent_usage[agent_id] += 1
        
        # System health indicators
        error_events_recent = [e for e in recent_events if e.event_type == EventType.ERROR_OCCURRED]
        recent_error_rate = (len(error_events_recent) / max(1, len(recent_events))) * 100
        
        return {
            "timestamp": current_time.isoformat(),
            "active_sessions": active_sessions_count,
            "recent_activity": {
                "events_last_5min": len(recent_events),
                "error_rate_percent": recent_error_rate
            },
            "current_agent_usage": dict(current_agent_usage),
            "system_metrics": {
                "total_events_tracked": len(self.events_history),
                "total_agents_used": len(self.agent_stats),
                "total_features_tracked": len(self.feature_usage)
            }
        }
    
    async def cleanup_old_data(self):
        """Clean up old analytics data"""
        cutoff_time = datetime.utcnow() - timedelta(days=self.retention_days)
        
        with self.lock:
            # Clean up old events (already handled by deque maxlen)
            # Clean up expired sessions
            expired_sessions = []
            session_timeout = datetime.utcnow() - timedelta(minutes=self.session_timeout_minutes)
            
            for session_id, session in self.active_sessions.items():
                if session.last_activity < session_timeout:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                self.end_session(session_id)

# Global usage analytics instance
usage_analytics = UsageAnalytics()

# Convenience functions
def track_user_login(user_id: str, session_id: str, metadata: Dict[str, Any] = None):
    """Track user login event"""
    return usage_analytics.track_event(
        EventType.USER_LOGIN,
        user_id=user_id,
        session_id=session_id,
        metadata=metadata
    )

def track_agent_request(user_id: str, session_id: str, agent_id: str, 
                       framework: str, response_time: float, success: bool = True,
                       features_used: List[str] = None):
    """Track agent request event"""
    return usage_analytics.track_event(
        EventType.AGENT_REQUEST,
        user_id=user_id,
        session_id=session_id,
        data={
            "agent_id": agent_id,
            "framework": framework,
            "response_time": response_time,
            "success": success,
            "features_used": features_used or []
        }
    )

def track_document_upload(user_id: str, session_id: str, filename: str, 
                         file_size: int, content_type: str):
    """Track document upload event"""
    return usage_analytics.track_event(
        EventType.DOCUMENT_UPLOAD,
        user_id=user_id,
        session_id=session_id,
        data={
            "filename": filename,
            "file_size": file_size,
            "content_type": content_type
        }
    )

def track_feature_usage(user_id: str, session_id: str, feature: str, 
                       metadata: Dict[str, Any] = None):
    """Track feature usage event"""
    return usage_analytics.track_event(
        EventType.FEATURE_USED,
        user_id=user_id,
        session_id=session_id,
        data={"feature": feature},
        metadata=metadata
    )

def track_error(user_id: str, session_id: str, error_type: str, 
               error_message: str, context: Dict[str, Any] = None):
    """Track error event"""
    return usage_analytics.track_event(
        EventType.ERROR_OCCURRED,
        user_id=user_id,
        session_id=session_id,
        data={
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {}
        }
    )

def get_usage_summary(hours: int = 24) -> Dict[str, Any]:
    """Get usage summary"""
    return usage_analytics.get_usage_summary(hours)

def get_real_time_metrics() -> Dict[str, Any]:
    """Get real-time metrics"""
    return usage_analytics.get_real_time_metrics()