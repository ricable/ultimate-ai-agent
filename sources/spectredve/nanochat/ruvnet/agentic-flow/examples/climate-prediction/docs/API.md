# Climate Prediction API Documentation

Complete REST API reference for the Climate Prediction System.

## Table of Contents

1. [Authentication](#authentication)
2. [Endpoints](#endpoints)
3. [Data Models](#data-models)
4. [Error Handling](#error-handling)
5. [Rate Limiting](#rate-limiting)
6. [OpenAPI Specification](#openapi-specification)

## Base URL

```
Production: https://api.climate-prediction.io/v1
Development: http://localhost:8080/api
```

## Authentication

All API requests require authentication using Bearer tokens.

### Obtain API Key

**Endpoint:** `POST /auth/register`

**Request:**
```json
{
  "email": "user@example.com",
  "password": "secure_password",
  "name": "John Doe"
}
```

**Response:**
```json
{
  "api_key": "cpk_1234567890abcdef",
  "user_id": "usr_abc123",
  "expires_at": "2026-10-14T10:30:00Z"
}
```

**Usage:**
```bash
curl -H "Authorization: Bearer cpk_1234567890abcdef" \
  https://api.climate-prediction.io/v1/predictions
```

## Endpoints

### 1. Get Single Prediction

Get weather prediction for a specific location.

**Endpoint:** `POST /predictions`

**Request:**
```json
{
  "latitude": 40.7128,
  "longitude": -74.0060,
  "days_ahead": 7,
  "model": "ensemble",
  "include_uncertainty": true,
  "units": "metric"
}
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| latitude | float | Yes | Latitude (-90 to 90) |
| longitude | float | Yes | Longitude (-180 to 180) |
| days_ahead | integer | No | Days to predict (1-14, default: 7) |
| model | string | No | Model type: "neural", "arima", "hybrid", "ensemble" |
| include_uncertainty | boolean | No | Include confidence intervals |
| units | string | No | "metric" or "imperial" (default: metric) |

**Response:**
```json
{
  "prediction": {
    "temperature": 22.5,
    "feels_like": 21.8,
    "humidity": 65,
    "precipitation": 2.3,
    "precipitation_probability": 0.35,
    "wind_speed": 12.5,
    "wind_direction": 180,
    "pressure": 1013.2,
    "cloud_cover": 45,
    "uv_index": 6,
    "confidence": 0.89,
    "model_name": "ensemble",
    "uncertainty": 1.2
  },
  "daily_forecast": [
    {
      "date": "2025-10-14",
      "temp_min": 18.5,
      "temp_max": 25.3,
      "precipitation": 1.2,
      "conditions": "partly_cloudy"
    }
  ],
  "metadata": {
    "timestamp": "2025-10-14T10:30:00Z",
    "location": {
      "latitude": 40.7128,
      "longitude": -74.0060,
      "timezone": "America/New_York",
      "elevation": 10
    },
    "processing_time_ms": 42,
    "cache_hit": false
  }
}
```

**cURL Example:**
```bash
curl -X POST https://api.climate-prediction.io/v1/predictions \
  -H "Authorization: Bearer cpk_1234567890abcdef" \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 40.7128,
    "longitude": -74.0060,
    "days_ahead": 7,
    "include_uncertainty": true
  }'
```

### 2. Batch Predictions

Get predictions for multiple locations in one request.

**Endpoint:** `POST /predictions/batch`

**Request:**
```json
{
  "locations": [
    {"latitude": 40.7128, "longitude": -74.0060, "name": "New York"},
    {"latitude": 34.0522, "longitude": -118.2437, "name": "Los Angeles"},
    {"latitude": 41.8781, "longitude": -87.6298, "name": "Chicago"}
  ],
  "days_ahead": 7,
  "model": "ensemble"
}
```

**Response:**
```json
{
  "predictions": [
    {
      "location": {
        "name": "New York",
        "latitude": 40.7128,
        "longitude": -74.0060
      },
      "prediction": { ... },
      "daily_forecast": [ ... ]
    }
  ],
  "metadata": {
    "total_locations": 3,
    "successful": 3,
    "failed": 0,
    "processing_time_ms": 125
  }
}
```

**Rate Limit:** 100 locations per request

### 3. Historical Predictions

Retrieve historical prediction data and accuracy metrics.

**Endpoint:** `GET /predictions/history`

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| lat | float | Yes | Latitude |
| lon | float | Yes | Longitude |
| days | integer | No | Days to retrieve (default: 30, max: 90) |
| start_date | string | No | ISO 8601 date (e.g., "2025-09-01") |
| end_date | string | No | ISO 8601 date |

**Request:**
```bash
GET /predictions/history?lat=40.7128&lon=-74.0060&days=30
```

**Response:**
```json
{
  "history": [
    {
      "date": "2025-09-14",
      "predicted": {
        "temperature": 22.5,
        "precipitation": 2.3
      },
      "actual": {
        "temperature": 23.1,
        "precipitation": 1.8
      },
      "accuracy": {
        "temperature_error": 0.6,
        "precipitation_error": 0.5,
        "confidence": 0.89
      }
    }
  ],
  "statistics": {
    "avg_temperature_error": 1.2,
    "avg_confidence": 0.87,
    "total_predictions": 30
  }
}
```

### 4. Model Information

Get information about available prediction models.

**Endpoint:** `GET /models`

**Response:**
```json
{
  "models": [
    {
      "name": "neural",
      "version": "1.0.0",
      "description": "WASM-based neural network",
      "accuracy": {
        "rmse": 1.1,
        "mae": 0.9
      },
      "performance": {
        "avg_inference_time_ms": 35,
        "memory_usage_mb": 45
      },
      "features": ["temperature", "precipitation", "wind_speed"]
    },
    {
      "name": "arima",
      "version": "1.0.0",
      "description": "Time series ARIMA model",
      "accuracy": {
        "rmse": 1.5,
        "mae": 1.2
      }
    },
    {
      "name": "ensemble",
      "version": "1.0.0",
      "description": "Multi-model ensemble",
      "weights": {
        "neural": 0.5,
        "arima": 0.3,
        "hybrid": 0.2
      },
      "accuracy": {
        "rmse": 0.9,
        "mae": 0.7
      }
    }
  ]
}
```

### 5. Health Check

Check API and model health status.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 86400,
  "models": {
    "neural": "loaded",
    "arima": "loaded",
    "hybrid": "loaded"
  },
  "reasoningbank": {
    "status": "connected",
    "backend": "wasm",
    "memory_usage_mb": 120
  },
  "cache": {
    "hit_rate": 0.75,
    "size": 450,
    "max_size": 1000
  }
}
```

### 6. Metrics

Get system performance metrics (Prometheus format).

**Endpoint:** `GET /metrics`

**Response:**
```
# HELP predictions_total Total number of predictions made
# TYPE predictions_total counter
predictions_total{model="ensemble"} 15234

# HELP prediction_duration_seconds Prediction processing time
# TYPE prediction_duration_seconds histogram
prediction_duration_seconds_bucket{le="0.05"} 12000
prediction_duration_seconds_bucket{le="0.1"} 14500
prediction_duration_seconds_sum 682.5
prediction_duration_seconds_count 15234

# HELP model_accuracy Model accuracy (RMSE)
# TYPE model_accuracy gauge
model_accuracy{model="neural"} 1.1
model_accuracy{model="arima"} 1.5
model_accuracy{model="ensemble"} 0.9
```

### 7. Search Predictions

Search historical predictions using semantic queries.

**Endpoint:** `POST /predictions/search`

**Request:**
```json
{
  "query": "warm sunny weather with low precipitation",
  "location": {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "radius_km": 50
  },
  "date_range": {
    "start": "2025-06-01",
    "end": "2025-08-31"
  },
  "limit": 10
}
```

**Response:**
```json
{
  "results": [
    {
      "date": "2025-07-15",
      "location": {
        "latitude": 40.7234,
        "longitude": -73.9987,
        "distance_km": 5.2
      },
      "prediction": {
        "temperature": 28.5,
        "conditions": "sunny",
        "precipitation": 0.1
      },
      "similarity_score": 0.95
    }
  ],
  "metadata": {
    "total_results": 10,
    "search_time_ms": 23
  }
}
```

## Data Models

### PredictionRequest

```typescript
interface PredictionRequest {
  latitude: number;        // -90 to 90
  longitude: number;       // -180 to 180
  days_ahead?: number;     // 1-14, default: 7
  model?: ModelType;       // "neural" | "arima" | "hybrid" | "ensemble"
  include_uncertainty?: boolean;
  units?: "metric" | "imperial";
}
```

### PredictionResponse

```typescript
interface PredictionResponse {
  prediction: {
    temperature: number;
    feels_like: number;
    humidity: number;           // 0-100
    precipitation: number;      // mm or inches
    precipitation_probability: number;  // 0-1
    wind_speed: number;
    wind_direction: number;     // degrees
    pressure: number;           // hPa
    cloud_cover: number;        // 0-100
    uv_index: number;           // 0-11+
    confidence: number;         // 0-1
    model_name: string;
    uncertainty?: number;
  };
  daily_forecast: DailyForecast[];
  metadata: {
    timestamp: string;
    location: Location;
    processing_time_ms: number;
    cache_hit: boolean;
  };
}
```

### Error Response

```typescript
interface ErrorResponse {
  error: {
    code: string;
    message: string;
    details?: object;
  };
  request_id: string;
  timestamp: string;
}
```

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid or missing API key |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource doesn't exist |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Model loading or maintenance |

### Error Response Examples

**400 Bad Request:**
```json
{
  "error": {
    "code": "INVALID_COORDINATES",
    "message": "Latitude must be between -90 and 90",
    "details": {
      "field": "latitude",
      "value": 120.5,
      "constraint": "range(-90, 90)"
    }
  },
  "request_id": "req_abc123",
  "timestamp": "2025-10-14T10:30:00Z"
}
```

**401 Unauthorized:**
```json
{
  "error": {
    "code": "INVALID_API_KEY",
    "message": "API key is invalid or has expired"
  },
  "request_id": "req_def456",
  "timestamp": "2025-10-14T10:30:00Z"
}
```

**429 Rate Limit:**
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Limit: 100 requests/minute",
    "details": {
      "limit": 100,
      "remaining": 0,
      "reset_at": "2025-10-14T10:31:00Z"
    }
  },
  "request_id": "req_ghi789",
  "timestamp": "2025-10-14T10:30:00Z"
}
```

## Rate Limiting

Rate limits are applied per API key:

| Tier | Requests/Minute | Requests/Day | Batch Size |
|------|-----------------|--------------|------------|
| Free | 60 | 1,000 | 10 locations |
| Basic | 300 | 10,000 | 50 locations |
| Pro | 1,000 | 50,000 | 100 locations |
| Enterprise | Custom | Custom | Custom |

**Rate Limit Headers:**
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1697275860
```

## OpenAPI Specification

Complete OpenAPI 3.0 specification available at:

```
https://api.climate-prediction.io/openapi.json
```

### Interactive Documentation

Swagger UI: `https://api.climate-prediction.io/docs`
ReDoc: `https://api.climate-prediction.io/redoc`

## SDK Examples

### JavaScript/TypeScript

```typescript
import { ClimateAPI } from '@climate-prediction/sdk';

const client = new ClimateAPI('cpk_1234567890abcdef');

// Single prediction
const prediction = await client.predict({
  latitude: 40.7128,
  longitude: -74.0060,
  daysAhead: 7
});

console.log(`Temperature: ${prediction.temperature}°C`);
console.log(`Confidence: ${prediction.confidence * 100}%`);

// Batch predictions
const locations = [
  { latitude: 40.7128, longitude: -74.0060, name: 'NYC' },
  { latitude: 34.0522, longitude: -118.2437, name: 'LA' }
];

const batch = await client.batchPredict(locations, { daysAhead: 7 });
```

### Python

```python
from climate_prediction import ClimateAPI

client = ClimateAPI(api_key='cpk_1234567890abcdef')

# Single prediction
prediction = client.predict(
    latitude=40.7128,
    longitude=-74.0060,
    days_ahead=7
)

print(f"Temperature: {prediction.temperature}°C")
print(f"Confidence: {prediction.confidence * 100}%")

# Batch predictions
locations = [
    {'latitude': 40.7128, 'longitude': -74.0060, 'name': 'NYC'},
    {'latitude': 34.0522, 'longitude': -118.2437, 'name': 'LA'}
]

batch = client.batch_predict(locations, days_ahead=7)
```

### Rust

```rust
use climate_prediction_sdk::{ClimateClient, PredictionConfig};

#[tokio::main]
async fn main() -> Result<()> {
    let client = ClimateClient::new("cpk_1234567890abcdef")?;

    let config = PredictionConfig {
        latitude: 40.7128,
        longitude: -74.0060,
        days_ahead: 7,
        ..Default::default()
    };

    let prediction = client.predict(&config).await?;

    println!("Temperature: {:.1}°C", prediction.temperature);
    println!("Confidence: {:.1}%", prediction.confidence * 100.0);

    Ok(())
}
```

## Webhooks

Subscribe to prediction updates via webhooks.

**Endpoint:** `POST /webhooks`

**Request:**
```json
{
  "url": "https://your-app.com/webhooks/climate",
  "events": ["prediction.completed", "model.updated"],
  "locations": [
    {"latitude": 40.7128, "longitude": -74.0060}
  ]
}
```

**Webhook Payload:**
```json
{
  "event": "prediction.completed",
  "timestamp": "2025-10-14T10:30:00Z",
  "data": {
    "location": {"latitude": 40.7128, "longitude": -74.0060},
    "prediction": { ... }
  }
}
```

---

**For more information, visit [https://docs.climate-prediction.io](https://docs.climate-prediction.io)**
