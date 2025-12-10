# File: backend/api_routes/analytics.py
"""
Analytics API Routes for UAP Platform

Provides comprehensive API endpoints for business intelligence, predictive analytics,
A/B testing, and advanced reporting capabilities.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Depends, Query, Body, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import io
import uuid

from ..analytics.usage_analytics import (
    usage_analytics, get_usage_summary, get_real_time_metrics
)
from ..analytics.reporting import (
    report_generator, generate_usage_report, generate_agent_performance_report,
    generate_business_intelligence_report, ReportFormat, ReportType
)
from ..analytics.predictive_analytics import (
    predictive_analytics, forecast_usage, detect_system_anomalies,
    plan_capacity, get_prediction_insights
)
from ..analytics.ab_testing import (
    ab_testing, create_feature_flag_test, assign_user_to_variant,
    record_conversion, get_experiment_results
)
from ..services.auth import get_current_user
from ..services.analytics_service import analytics_service
from ..services.metrics_collector import metrics_collector
from ..models.user import User

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

# Pydantic Models for Request/Response
class BusinessIntelligenceRequest(BaseModel):
    days: int = Field(default=30, ge=1, le=365)
    include_predictions: bool = Field(default=True)
    include_comparisons: bool = Field(default=True)

class PredictionRequest(BaseModel):
    prediction_type: str = Field(..., description="Type of prediction to make")
    features: Dict[str, Any] = Field(..., description="Input features for prediction")
    time_horizon_hours: Optional[int] = Field(default=24, ge=1, le=168)

class ExperimentCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=1000)
    experiment_type: str = Field(..., regex="^(feature_flag|ui_variant|algorithm_test|performance_test)$")
    variants: List[Dict[str, Any]] = Field(..., min_items=2, max_items=10)
    metrics: List[Dict[str, Any]] = Field(..., min_items=1, max_items=20)
    minimum_sample_size: int = Field(default=100, ge=10, le=100000)
    confidence_level: float = Field(default=0.95, ge=0.8, le=0.99)

class MetricEventRequest(BaseModel):
    experiment_id: str = Field(...)
    metric_id: str = Field(...)
    value: Union[float, int, bool, str] = Field(...)
    metadata: Optional[Dict[str, Any]] = Field(default={})

class ReportExportRequest(BaseModel):
    report_type: str = Field(...)
    format: str = Field(default="json", regex="^(json|csv|html|xlsx|pdf)$")
    time_period: str = Field(default="30d")
    filters: Optional[Dict[str, Any]] = Field(default={})
    include_details: bool = Field(default=False)

# ============================================================================
# DASHBOARD ENDPOINTS
# ============================================================================

@router.get("/dashboard")
async def get_dashboard_data(
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive dashboard data for analytics"""
    try:
        dashboard_data = await analytics_service.get_dashboard_data(str(current_user.id))
        return dashboard_data.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch dashboard data: {str(e)}")

@router.get("/metrics/{metric_name}/history")
async def get_metric_history(
    metric_name: str,
    hours: int = Query(default=24, ge=1, le=168),
    current_user: User = Depends(get_current_user)
):
    """Get historical data for a specific metric"""
    try:
        history = await metrics_collector.get_metric_history(metric_name, hours)
        return history
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch metric history: {str(e)}")

@router.get("/metrics/collection/stats")
async def get_collection_stats(
    current_user: User = Depends(get_current_user)
):
    """Get metrics collection statistics"""
    try:
        stats = await metrics_collector.get_collection_stats()
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch collection stats: {str(e)}")

@router.post("/metrics/custom")
async def add_custom_metric(
    metric_name: str = Body(...),
    value: Union[float, int] = Body(...),
    labels: Optional[Dict[str, str]] = Body(default=None),
    current_user: User = Depends(get_current_user)
):
    """Add a custom metric value"""
    try:
        metrics_collector.add_custom_metric(metric_name, value, labels)
        return {"message": "Custom metric added successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add custom metric: {str(e)}")

# ============================================================================
# BUSINESS INTELLIGENCE ENDPOINTS
# ============================================================================

@router.get("/business-intelligence")
async def get_business_intelligence(
    days: int = Query(default=30, ge=1, le=365),
    include_kpis: bool = Query(default=True),
    include_trends: bool = Query(default=True),
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive business intelligence dashboard data"""
    try:
        # Generate business intelligence report
        report = await generate_business_intelligence_report(days=days)
        
        if not report:
            raise HTTPException(status_code=500, detail="Failed to generate BI report")
        
        # Extract data from the report
        bi_data = report.data
        
        # Add real-time metrics
        real_time = get_real_time_metrics()
        
        # Combine all data
        response_data = {
            "time_range": bi_data.get("time_range"),
            "kpis": bi_data.get("kpis", {}),
            "user_engagement": bi_data.get("user_engagement", {}),
            "platform_performance": bi_data.get("platform_performance", {}),
            "growth_metrics": bi_data.get("growth_metrics", {}),
            "operational_metrics": bi_data.get("operational_metrics", {}),
            "recommendations": bi_data.get("recommendations", []),
            "real_time_metrics": real_time,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch business intelligence: {str(e)}")

@router.post("/business-intelligence/export")
async def export_business_intelligence(
    request: ReportExportRequest,
    current_user: User = Depends(get_current_user)
):
    """Export business intelligence report in specified format"""
    try:
        # Parse time period
        days = 30
        if request.time_period == "7d":
            days = 7
        elif request.time_period == "90d":
            days = 90
        elif request.time_period == "1y":
            days = 365
        
        # Generate report
        report_format = ReportFormat(request.format)
        report = await generate_business_intelligence_report(days=days)
        
        if not report:
            raise HTTPException(status_code=500, detail="Failed to generate report")
        
        # Return file for download
        if report.file_path:
            with open(report.file_path, 'rb') as f:
                content = f.read()
            
            media_type = {
                'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'csv': 'text/csv',
                'html': 'text/html',
                'pdf': 'application/pdf',
                'json': 'application/json'
            }.get(request.format, 'application/octet-stream')
            
            return StreamingResponse(
                io.BytesIO(content),
                media_type=media_type,
                headers={"Content-Disposition": f"attachment; filename=bi-report-{datetime.now().strftime('%Y%m%d')}.{request.format}"}
            )
        else:
            # Return JSON data directly
            return report.data
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export report: {str(e)}")

# ============================================================================
# USAGE ANALYTICS ENDPOINTS
# ============================================================================

@router.get("/usage/summary")
async def get_usage_analytics(
    time_window_hours: int = Query(default=24, ge=1, le=168),
    current_user: User = Depends(get_current_user)
):
    """Get usage analytics summary"""
    try:
        summary = get_usage_summary(time_window_hours)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch usage analytics: {str(e)}")

@router.get("/usage/real-time")
async def get_real_time_analytics(
    current_user: User = Depends(get_current_user)
):
    """Get real-time analytics metrics"""
    try:
        metrics = get_real_time_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch real-time metrics: {str(e)}")

@router.get("/usage/user/{user_id}")
async def get_user_analytics(
    user_id: str,
    days: int = Query(default=30, ge=1, le=365),
    current_user: User = Depends(get_current_user)
):
    """Get analytics for a specific user"""
    try:
        analytics = usage_analytics.get_user_analytics(user_id, days)
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch user analytics: {str(e)}")

# ============================================================================
# PREDICTIVE ANALYTICS ENDPOINTS
# ============================================================================

@router.get("/predictions/insights")
async def get_predictive_insights(
    days: int = Query(default=7, ge=1, le=30),
    current_user: User = Depends(get_current_user)
):
    """Get predictive analytics insights"""
    try:
        insights = await get_prediction_insights(days)
        
        # Get recent predictions from the system
        predictions = list(predictive_analytics.prediction_history)
        recent_predictions = [
            pred.to_dict() for pred in predictions[-50:]  # Last 50 predictions
        ]
        
        return {
            "insights": insights,
            "predictions": recent_predictions,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch predictive insights: {str(e)}")

@router.post("/predictions/forecast")
async def create_usage_forecast(
    request: PredictionRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a usage forecast prediction"""
    try:
        # Create forecast based on target time
        target_time = datetime.utcnow() + timedelta(hours=request.time_horizon_hours or 24)
        prediction = await predictive_analytics.predict_usage_forecast(target_time)
        
        if not prediction:
            raise HTTPException(status_code=400, detail="Unable to generate forecast - insufficient data")
        
        return prediction.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create forecast: {str(e)}")

@router.get("/predictions/anomalies")
async def detect_anomalies(
    current_user: User = Depends(get_current_user)
):
    """Detect current system anomalies"""
    try:
        anomaly_prediction = await detect_system_anomalies()
        
        if not anomaly_prediction:
            return {
                "anomalies": [],
                "current_anomaly_score": 0.0,
                "status": "normal",
                "message": "No anomalies detected"
            }
        
        anomaly_score = anomaly_prediction.predicted_value
        status = "normal"
        if anomaly_score > 0.7:
            status = "high_risk"
        elif anomaly_score > 0.4:
            status = "medium_risk"
        elif anomaly_score > 0.2:
            status = "low_risk"
        
        return {
            "anomalies": [anomaly_prediction.to_dict()],
            "current_anomaly_score": anomaly_score,
            "status": status,
            "confidence": anomaly_prediction.confidence_score,
            "timestamp": anomaly_prediction.created_at
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect anomalies: {str(e)}")

@router.get("/predictions/capacity")
async def get_capacity_planning(
    hours_ahead: int = Query(default=12, ge=1, le=168),
    current_user: User = Depends(get_current_user)
):
    """Get capacity planning predictions"""
    try:
        capacity_plan = await plan_capacity(hours_ahead)
        return capacity_plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get capacity planning: {str(e)}")

@router.get("/predictions/models/performance")
async def get_model_performance(
    current_user: User = Depends(get_current_user)
):
    """Get performance metrics for all prediction models"""
    try:
        models_performance = []
        
        for prediction_type, model in predictive_analytics.models.items():
            perf_data = {
                "model_type": model.model_type.value,
                "prediction_type": prediction_type.value,
                "is_trained": model.is_trained,
                "last_trained": model.last_trained.isoformat() if model.last_trained else None,
                "training_data_points": len(predictive_analytics.training_data.get(prediction_type, [])),
                "performance_metrics": model.performance_metrics._asdict() if model.performance_metrics else None,
                "feature_names": model.feature_names
            }
            
            # Calculate accuracy as R² score percentage
            if model.performance_metrics:
                perf_data["accuracy"] = max(0, min(100, model.performance_metrics.r2 * 100))
            else:
                perf_data["accuracy"] = 0
            
            models_performance.append(perf_data)
        
        return {"models": models_performance}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model performance: {str(e)}")

@router.post("/predictions/models/{model_type}/retrain")
async def retrain_model(
    model_type: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Trigger model retraining"""
    try:
        # Map model type string to enum
        from ..analytics.predictive_analytics import PredictionType
        
        prediction_type_map = {
            "usage_forecast": PredictionType.USAGE_FORECAST,
            "anomaly_detection": PredictionType.ANOMALY_DETECTION,
            "capacity_planning": PredictionType.CAPACITY_PLANNING,
            "performance_trend": PredictionType.PERFORMANCE_TREND
        }
        
        prediction_type = prediction_type_map.get(model_type)
        if not prediction_type:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
        
        # Schedule retraining in background
        background_tasks.add_task(predictive_analytics.train_model, prediction_type)
        
        return {
            "message": f"Model retraining scheduled for {model_type}",
            "model_type": model_type,
            "scheduled_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger retraining: {str(e)}")

# ============================================================================
# A/B TESTING ENDPOINTS
# ============================================================================

@router.get("/ab-tests/summary")
async def get_ab_test_summary(
    current_user: User = Depends(get_current_user)
):
    """Get A/B testing summary"""
    try:
        summary = ab_testing.get_experiment_summary()
        
        # Get detailed results for active experiments
        active_experiments = {}
        for exp_id, exp in ab_testing.experiments.items():
            if exp.status.value == 'active':
                results = ab_testing.get_experiment_results(exp_id)
                if results:
                    active_experiments[exp_id] = {
                        "name": exp.name,
                        "results": results.to_dict()
                    }
        
        return {
            "summary": summary,
            "active_experiments": active_experiments,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get A/B test summary: {str(e)}")

@router.get("/ab-tests/experiments")
async def get_experiments(
    status: Optional[str] = Query(default=None),
    current_user: User = Depends(get_current_user)
):
    """Get list of A/B test experiments"""
    try:
        experiments_data = []
        
        for exp_id, exp in ab_testing.experiments.items():
            if status and exp.status.value != status:
                continue
            
            exp_data = exp.to_dict()
            
            # Add results if available
            results = ab_testing.get_experiment_results(exp_id)
            if results:
                exp_data["results"] = results.to_dict()
            
            experiments_data.append(exp_data)
        
        return {"experiments": experiments_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get experiments: {str(e)}")

@router.post("/ab-tests/experiments")
async def create_experiment(
    request: ExperimentCreateRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new A/B test experiment"""
    try:
        from ..analytics.ab_testing import ExperimentType
        
        # Map experiment type
        exp_type_map = {
            "feature_flag": ExperimentType.FEATURE_FLAG,
            "ui_variant": ExperimentType.UI_VARIANT,
            "algorithm_test": ExperimentType.ALGORITHM_TEST,
            "performance_test": ExperimentType.PERFORMANCE_TEST
        }
        
        exp_type = exp_type_map.get(request.experiment_type)
        if not exp_type:
            raise HTTPException(status_code=400, detail=f"Invalid experiment type: {request.experiment_type}")
        
        # Create experiment
        experiment_id = ab_testing.create_experiment(
            name=request.name,
            description=request.description,
            experiment_type=exp_type,
            variants=request.variants,
            metrics=request.metrics,
            minimum_sample_size=request.minimum_sample_size,
            confidence_level=request.confidence_level,
            created_by=current_user.username
        )
        
        return {
            "experiment_id": experiment_id,
            "message": "Experiment created successfully",
            "created_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create experiment: {str(e)}")

@router.post("/ab-tests/experiments/{experiment_id}/start")
async def start_experiment(
    experiment_id: str,
    end_date: Optional[datetime] = Body(default=None),
    current_user: User = Depends(get_current_user)
):
    """Start an A/B test experiment"""
    try:
        success = ab_testing.start_experiment(experiment_id, end_date=end_date)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to start experiment")
        
        return {
            "message": "Experiment started successfully",
            "experiment_id": experiment_id,
            "started_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start experiment: {str(e)}")

@router.post("/ab-tests/experiments/{experiment_id}/pause")
async def pause_experiment(
    experiment_id: str,
    current_user: User = Depends(get_current_user)
):
    """Pause an A/B test experiment"""
    try:
        if experiment_id not in ab_testing.experiments:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        exp = ab_testing.experiments[experiment_id]
        from ..analytics.ab_testing import ExperimentStatus
        exp.status = ExperimentStatus.PAUSED
        
        return {
            "message": "Experiment paused successfully",
            "experiment_id": experiment_id,
            "paused_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to pause experiment: {str(e)}")

@router.post("/ab-tests/experiments/{experiment_id}/stop")
async def stop_experiment(
    experiment_id: str,
    current_user: User = Depends(get_current_user)
):
    """Stop an A/B test experiment"""
    try:
        success = ab_testing.stop_experiment(experiment_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to stop experiment")
        
        return {
            "message": "Experiment stopped successfully",
            "experiment_id": experiment_id,
            "stopped_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop experiment: {str(e)}")

@router.get("/ab-tests/experiments/{experiment_id}/results")
async def get_experiment_results_api(
    experiment_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get results for a specific experiment"""
    try:
        results = get_experiment_results(experiment_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="Experiment results not found")
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get experiment results: {str(e)}")

@router.post("/ab-tests/assign/{experiment_id}")
async def assign_user_to_experiment(
    experiment_id: str,
    user_id: Optional[str] = Body(default=None),
    force_variant: Optional[str] = Body(default=None),
    current_user: User = Depends(get_current_user)
):
    """Assign a user to an experiment variant"""
    try:
        # Use current user if no user_id provided
        target_user_id = user_id or str(current_user.id)
        
        variant = assign_user_to_variant(
            target_user_id, 
            experiment_id,
            # Could add session_id from request headers
        )
        
        if not variant:
            raise HTTPException(status_code=400, detail="Failed to assign user to experiment")
        
        return {
            "experiment_id": experiment_id,
            "user_id": target_user_id,
            "assigned_variant": variant,
            "assigned_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to assign user: {str(e)}")

@router.post("/ab-tests/events")
async def record_metric_event(
    request: MetricEventRequest,
    user_id: Optional[str] = Body(default=None),
    current_user: User = Depends(get_current_user)
):
    """Record a metric event for an experiment"""
    try:
        target_user_id = user_id or str(current_user.id)
        
        success = ab_testing.record_metric_event(
            experiment_id=request.experiment_id,
            user_id=target_user_id,
            metric_id=request.metric_id,
            value=request.value,
            metadata=request.metadata
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to record metric event")
        
        return {
            "message": "Metric event recorded successfully",
            "experiment_id": request.experiment_id,
            "metric_id": request.metric_id,
            "recorded_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record metric event: {str(e)}")

# ============================================================================
# REPORTING ENDPOINTS
# ============================================================================

@router.get("/reports/types")
async def get_report_types(
    current_user: User = Depends(get_current_user)
):
    """Get available report types"""
    return {
        "report_types": [
            {"value": rt.value, "label": rt.value.replace('_', ' ').title()}
            for rt in ReportType
        ],
        "formats": [
            {"value": rf.value, "label": rf.value.upper()}
            for rf in ReportFormat
        ]
    }

@router.post("/reports/generate")
async def generate_custom_report(
    request: ReportExportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Generate a custom analytics report"""
    try:
        # Parse time period
        time_window_hours = 24
        if request.time_period == "7d":
            time_window_hours = 7 * 24
        elif request.time_period == "30d":
            time_window_hours = 30 * 24
        elif request.time_period == "90d":
            time_window_hours = 90 * 24
        
        # Generate report based on type
        if request.report_type == "usage_summary":
            report = await generate_usage_report(
                time_window_hours=time_window_hours,
                format=ReportFormat(request.format)
            )
        elif request.report_type == "agent_performance":
            report = await generate_agent_performance_report(
                format=ReportFormat(request.format)
            )
        elif request.report_type == "business_intelligence":
            days = time_window_hours // 24
            report = await generate_business_intelligence_report(days=days)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown report type: {request.report_type}")
        
        if not report:
            raise HTTPException(status_code=500, detail="Failed to generate report")
        
        # Return report data or file
        if report.file_path:
            with open(report.file_path, 'rb') as f:
                content = f.read()
            
            media_type = {
                'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'csv': 'text/csv',
                'html': 'text/html',
                'pdf': 'application/pdf',
                'json': 'application/json'
            }.get(request.format, 'application/octet-stream')
            
            return StreamingResponse(
                io.BytesIO(content),
                media_type=media_type,
                headers={"Content-Disposition": f"attachment; filename=report-{datetime.now().strftime('%Y%m%d')}.{request.format}"}
            )
        else:
            return report.to_dict()
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

# ============================================================================
# HEALTH AND STATUS ENDPOINTS
# ============================================================================

@router.get("/health")
async def get_analytics_health():
    """Get analytics system health status"""
    try:
        # Check various analytics components
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "usage_analytics": {
                    "status": "healthy",
                    "events_tracked": len(usage_analytics.events_history),
                    "active_sessions": len(usage_analytics.active_sessions)
                },
                "predictive_analytics": {
                    "status": "healthy",
                    "models_active": len([m for m in predictive_analytics.models.values() if m.is_trained]),
                    "predictions_made": len(predictive_analytics.prediction_history)
                },
                "ab_testing": {
                    "status": "healthy",
                    "active_experiments": len([e for e in ab_testing.experiments.values() if e.status.value == 'active']),
                    "total_participants": sum(len(p) for p in ab_testing.participants.values())
                },
                "reporting": {
                    "status": "healthy",
                    "report_configs": len(report_generator.report_configs),
                    "generated_reports": len(report_generator.generated_reports)
                }
            }
        }
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# ============================================================================
# BACKGROUND TASKS
# ============================================================================

@router.post("/maintenance/cleanup")
async def trigger_cleanup(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Trigger analytics data cleanup"""
    try:
        # Schedule cleanup tasks
        background_tasks.add_task(usage_analytics.cleanup_old_data)
        background_tasks.add_task(predictive_analytics.collect_training_data)
        
        return {
            "message": "Cleanup tasks scheduled",
            "scheduled_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to schedule cleanup: {str(e)}")

@router.post("/maintenance/collect-training-data")
async def trigger_data_collection(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Trigger training data collection for ML models"""
    try:
        background_tasks.add_task(predictive_analytics.collect_training_data)
        
        return {
            "message": "Data collection scheduled",
            "scheduled_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to schedule data collection: {str(e)}")
