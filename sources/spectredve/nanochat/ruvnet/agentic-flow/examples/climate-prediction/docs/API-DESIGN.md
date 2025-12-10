# Climate Prediction API Design

## Base URL
```
http://localhost:8080/api/v1
```

## Authentication
API key in header:
```
Authorization: Bearer <api-key>
```

## Endpoints

### 1. Single Prediction
```http
POST /predict
Content-Type: application/json

{
  "location": {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "altitude": 10.0
  },
  "time_range": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-07T00:00:00Z"
  },
  "variables": ["Temperature", "Precipitation"],
  "model_name": "climate-v1"
}
```

**Response**:
```json
{
  "predictions": [
    {
      "location": {
        "latitude": 40.7128,
        "longitude": -74.0060,
        "altitude": 10.0
      },
      "variable": "Temperature",
      "forecast_time": "2024-01-01T00:00:00Z",
      "predicted_value": 15.5,
      "confidence_interval": {
        "lower": 13.2,
        "upper": 17.8,
        "confidence_level": 0.95
      },
      "model_version": "climate-v1.0.0"
    }
  ],
  "request_id": "req_abc123",
  "processing_time_ms": 45
}
```

### 2. Batch Predictions
```http
POST /predict/batch
Content-Type: application/json

{
  "locations": [
    {"latitude": 40.7128, "longitude": -74.0060},
    {"latitude": 34.0522, "longitude": -118.2437}
  ],
  "time_range": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-07T00:00:00Z"
  },
  "variables": ["Temperature"],
  "model_name": "climate-v1"
}
```

**Response**: Array of predictions grouped by location

### 3. List Models
```http
GET /models
```

**Response**:
```json
{
  "models": [
    {
      "name": "climate-v1",
      "version": "1.0.0",
      "trained_on": "2023-12-01T00:00:00Z",
      "variables": ["Temperature", "Precipitation"],
      "spatial_resolution": 0.25,
      "temporal_resolution": "1h",
      "metrics": {
        "rmse": 1.2,
        "mae": 0.9,
        "r2": 0.95
      }
    }
  ]
}
```

### 4. Health Check
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 3600,
  "models_loaded": 2,
  "data_sources_available": 3
}
```

### 5. Metrics
```http
GET /metrics
```

**Response**: Prometheus format
```
# HELP prediction_requests_total Total prediction requests
# TYPE prediction_requests_total counter
prediction_requests_total{status="success"} 1234
prediction_requests_total{status="error"} 12

# HELP prediction_latency_seconds Prediction latency
# TYPE prediction_latency_seconds histogram
prediction_latency_seconds_bucket{le="0.1"} 800
prediction_latency_seconds_bucket{le="0.5"} 1200
prediction_latency_seconds_sum 450
prediction_latency_seconds_count 1234
```

## Error Responses

### 400 Bad Request
```json
{
  "error": {
    "code": "invalid_input",
    "message": "Latitude must be between -90 and 90",
    "field": "location.latitude"
  }
}
```

### 401 Unauthorized
```json
{
  "error": {
    "code": "unauthorized",
    "message": "Invalid API key"
  }
}
```

### 429 Too Many Requests
```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Rate limit exceeded",
    "retry_after_seconds": 60
  }
}
```

### 500 Internal Server Error
```json
{
  "error": {
    "code": "internal_error",
    "message": "Model inference failed",
    "request_id": "req_abc123"
  }
}
```

## Rate Limiting
- 100 requests per minute per API key
- 1000 batch predictions per hour
- Headers:
  - `X-RateLimit-Limit`: Max requests
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Unix timestamp of reset

## Pagination
For endpoints returning large datasets:
```http
GET /predictions?page=1&per_page=50
```

**Response Headers**:
```
Link: <http://api.example.com/predictions?page=2>; rel="next"
X-Total-Count: 500
X-Page: 1
X-Per-Page: 50
```

## OpenAPI Specification
Available at `/openapi.json` for Swagger UI integration.

## WebSocket Streaming (Future)
```
ws://localhost:8080/api/v1/stream/predict
```

Real-time predictions streamed as JSON lines.
