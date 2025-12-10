# Deployment Migration Guide: Production Deployment Strategies

This guide covers comprehensive strategies for migrating production deployments from Python NeuralForecast to Rust neuro-divergent, ensuring zero-downtime transitions and robust production systems.

## Table of Contents

1. [Deployment Strategies](#deployment-strategies)
2. [Infrastructure Migration](#infrastructure-migration)
3. [Configuration Management](#configuration-management)
4. [Service Architecture](#service-architecture)
5. [Load Balancing and Scaling](#load-balancing-and-scaling)
6. [Security Considerations](#security-considerations)
7. [Monitoring and Alerting](#monitoring-and-alerting)
8. [Rollback Procedures](#rollback-procedures)

## Deployment Strategies

### Blue-Green Deployment

**Strategy Overview**:
- Run both Python and Rust systems in parallel
- Gradually shift traffic from Python (blue) to Rust (green)
- Instant rollback capability

**Implementation**:
```bash
# Phase 1: Deploy Rust system alongside Python
kubectl apply -f k8s/neuro-divergent-green.yaml

# Phase 2: Test Rust system with 1% traffic
kubectl patch ingress forecasting-ingress --patch '
{
  "spec": {
    "rules": [{
      "host": "api.company.com",
      "http": {
        "paths": [
          {
            "path": "/predict",
            "pathType": "Prefix",
            "backend": {
              "service": {
                "name": "traffic-split",
                "port": {"number": 80}
              }
            }
          }
        ]
      }
    }]
  }
}'

# Phase 3: Gradually increase Rust traffic (10%, 25%, 50%, 100%)
# Monitor metrics and performance

# Phase 4: Complete cutover
kubectl delete deployment neuralforecast-python
```

**Traffic Splitting Configuration**:
```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: forecasting-traffic-split
spec:
  hosts:
  - api.company.com
  http:
  - match:
    - uri:
        prefix: "/predict"
    route:
    - destination:
        host: neuralforecast-python
      weight: 90  # Gradually decrease
    - destination:
        host: neuro-divergent-rust
      weight: 10  # Gradually increase
```

### Canary Deployment

**Strategy Overview**:
- Deploy Rust to subset of users/regions
- Monitor performance and error rates
- Gradual rollout based on success metrics

**Implementation**:
```yaml
# Canary deployment with Flagger
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: neuro-divergent-canary
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: neuro-divergent
  service:
    port: 8080
  analysis:
    interval: 30s
    threshold: 5
    stepWeight: 10
    maxWeight: 50
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99
    - name: request-duration
      thresholdRange:
        max: 500
    webhooks:
    - name: load-test
      url: http://load-test.default/
      timeout: 15s
      metadata:
        cmd: "hey -z 10m -c 2 http://neuro-divergent-canary.default:8080/predict"
```

### Rolling Deployment

**Strategy Overview**:
- Replace Python instances one by one with Rust
- Gradual migration with automatic rollback on failure
- Good for stateless services

**Kubernetes Rolling Update**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neuro-divergent
spec:
  replicas: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  template:
    metadata:
      labels:
        app: neuro-divergent
        version: rust
    spec:
      containers:
      - name: neuro-divergent
        image: neuro-divergent:latest
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
```

## Infrastructure Migration

### Container Migration

**Python Container (Before)**:
```dockerfile
# Python container - ~500MB
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Rust Container (After)**:
```dockerfile
# Multi-stage Rust container - ~20MB
FROM rust:1.70 as builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

# Runtime stage
FROM gcr.io/distroless/cc-debian11

WORKDIR /app

COPY --from=builder /app/target/release/neuro-divergent /usr/local/bin/
COPY models/ ./models/

EXPOSE 8080
CMD ["neuro-divergent", "serve"]
```

### Resource Requirements

**Before (Python)**:
```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

**After (Rust)**:
```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

### Storage Migration

**Model Storage Optimization**:
```bash
# Python models (pickle format)
# models/
#   ├── lstm_model.pkl        (50MB)
#   ├── nbeats_model.pkl      (100MB)
#   └── tft_model.pkl         (150MB)
# Total: 300MB

# Rust models (binary format)
# models/
#   ├── lstm_model.bin        (20MB)
#   ├── nbeats_model.bin      (40MB)
#   └── tft_model.bin         (60MB)
# Total: 120MB (60% smaller)
```

## Configuration Management

### Environment Configuration

**Python Configuration**:
```python
# config.py
import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    model_path: str = "/models"
    data_path: str = "/data"
    log_level: str = "INFO"
    port: int = 8000
    workers: int = 4
    
    class Config:
        env_file = ".env"

settings = Settings()
```

**Rust Configuration**:
```rust
// config.rs
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Deserialize, Serialize)]
pub struct Config {
    pub model_path: String,
    pub data_path: String,
    pub log_level: String,
    pub port: u16,
    pub workers: usize,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            model_path: env::var("MODEL_PATH").unwrap_or_else(|_| "/models".to_string()),
            data_path: env::var("DATA_PATH").unwrap_or_else(|_| "/data".to_string()),
            log_level: env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
            port: env::var("PORT").unwrap_or_else(|_| "8080".to_string()).parse().unwrap_or(8080),
            workers: env::var("WORKERS").unwrap_or_else(|_| "4".to_string()).parse().unwrap_or(4),
        }
    }
}
```

### ConfigMap Migration

**Kubernetes ConfigMap**:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: neuro-divergent-config
data:
  config.toml: |
    [server]
    port = 8080
    workers = 4
    
    [models]
    path = "/models"
    cache_size = 10
    
    [logging]
    level = "info"
    format = "json"
    
    [metrics]
    enabled = true
    port = 9090
```

## Service Architecture

### Microservices Migration

**Before: Python Monolith**:
```python
# main.py
from fastapi import FastAPI
from neuralforecast import NeuralForecast

app = FastAPI()
model = None

@app.on_event("startup")
async def startup_event():
    global model
    model = NeuralForecast.load("model.pkl")

@app.post("/predict")
async def predict(data: PredictionRequest):
    return model.predict(data.to_dataframe())

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

**After: Rust Microservices**:
```rust
// src/main.rs
use axum::{extract::State, http::StatusCode, Json, Router};
use neuro_divergent::NeuralForecast;
use std::sync::Arc;
use tokio::net::TcpListener;

#[derive(Clone)]
struct AppState {
    model: Arc<NeuralForecast>,
}

#[tokio::main]
async fn main() {
    let model = Arc::new(NeuralForecast::load("model.bin").unwrap());
    let state = AppState { model };
    
    let app = Router::new()
        .route("/predict", axum::routing::post(predict))
        .route("/health", axum::routing::get(health))
        .route("/metrics", axum::routing::get(metrics))
        .with_state(state);
    
    let listener = TcpListener::bind("0.0.0.0:8080").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn predict(
    State(state): State<AppState>,
    Json(request): Json<PredictionRequest>,
) -> Result<Json<PredictionResponse>, StatusCode> {
    let df = request.to_dataframe().map_err(|_| StatusCode::BAD_REQUEST)?;
    let predictions = state.model.predict(df).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(Json(PredictionResponse { predictions }))
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "healthy"}))
}

async fn metrics() -> String {
    prometheus::gather().into_iter()
        .map(|mf| prometheus::text_format::metric_family_to_text(&mf))
        .collect::<Vec<_>>()
        .join("")
}
```

### Service Mesh Integration

**Istio Configuration**:
```yaml
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: neuro-divergent
spec:
  host: neuro-divergent
  trafficPolicy:
    loadBalancer:
      simple: LEAST_CONN
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        maxRequestsPerConnection: 10
    circuitBreaker:
      consecutiveErrors: 3
      interval: 30s
      baseEjectionTime: 30s
---
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: neuro-divergent
spec:
  hosts:
  - neuro-divergent
  http:
  - timeout: 30s
    retries:
      attempts: 3
      perTryTimeout: 10s
    route:
    - destination:
        host: neuro-divergent
```

## Load Balancing and Scaling

### Horizontal Pod Autoscaler

**HPA Configuration**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: neuro-divergent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: neuro-divergent
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### Vertical Pod Autoscaler

**VPA Configuration**:
```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: neuro-divergent-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: neuro-divergent
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: neuro-divergent
      maxAllowed:
        cpu: 2
        memory: 4Gi
      minAllowed:
        cpu: 100m
        memory: 128Mi
```

## Security Considerations

### Container Security

**Security Scanning**:
```yaml
# .github/workflows/security.yml
name: Security Scan

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'neuro-divergent:latest'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
```

**Security Policies**:
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: neuro-divergent
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
  containers:
  - name: neuro-divergent
    image: neuro-divergent:latest
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
    volumeMounts:
    - name: tmp
      mountPath: /tmp
  volumes:
  - name: tmp
    emptyDir: {}
```

### Network Policies

**Network Security**:
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: neuro-divergent-netpol
spec:
  podSelector:
    matchLabels:
      app: neuro-divergent
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: frontend
    - podSelector:
        matchLabels:
          app: api-gateway
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: data
    ports:
    - protocol: TCP
      port: 5432  # Database
  - to: []
    ports:
    - protocol: TCP
      port: 53   # DNS
    - protocol: UDP
      port: 53
```

## Monitoring and Alerting

### Prometheus Metrics

**Custom Metrics**:
```rust
use prometheus::{Counter, Histogram, Gauge, register_counter, register_histogram, register_gauge};

lazy_static! {
    static ref PREDICTIONS_TOTAL: Counter = register_counter!(
        "neuro_divergent_predictions_total",
        "Total number of predictions made"
    ).unwrap();
    
    static ref PREDICTION_DURATION: Histogram = register_histogram!(
        "neuro_divergent_prediction_duration_seconds",
        "Time spent making predictions"
    ).unwrap();
    
    static ref MODEL_MEMORY_USAGE: Gauge = register_gauge!(
        "neuro_divergent_model_memory_bytes",
        "Memory usage of loaded models"
    ).unwrap();
    
    static ref ERROR_RATE: Counter = register_counter!(
        "neuro_divergent_errors_total",
        "Total number of prediction errors"
    ).unwrap();
}
```

### Alerting Rules

**PrometheusRule**:
```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: neuro-divergent-alerts
spec:
  groups:
  - name: neuro-divergent
    rules:
    - alert: HighErrorRate
      expr: |
        rate(neuro_divergent_errors_total[5m]) / rate(neuro_divergent_predictions_total[5m]) > 0.05
      for: 2m
      labels:
        severity: warning
      annotations:
        summary: High error rate in neuro-divergent
        description: "Error rate is {{ $value | humanizePercentage }} over the last 5 minutes"
    
    - alert: HighLatency
      expr: |
        histogram_quantile(0.95, rate(neuro_divergent_prediction_duration_seconds_bucket[5m])) > 1.0
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: High prediction latency
        description: "95th percentile latency is {{ $value }}s"
    
    - alert: PodCrashLooping
      expr: |
        rate(kube_pod_container_status_restarts_total{container="neuro-divergent"}[15m]) > 0
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: Pod is crash looping
        description: "Pod {{ $labels.pod }} is restarting frequently"
```

## Rollback Procedures

### Automated Rollback

**Rollback Script**:
```bash
#!/bin/bash
# rollback.sh

set -e

NAMESPACE=${1:-default}
DEPLOYMENT=${2:-neuro-divergent}

echo "Starting rollback for $DEPLOYMENT in namespace $NAMESPACE"

# Check current status
kubectl get deployment $DEPLOYMENT -n $NAMESPACE

# Rollback to previous version
kubectl rollout undo deployment/$DEPLOYMENT -n $NAMESPACE

# Wait for rollback to complete
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE --timeout=300s

# Verify health
echo "Checking health endpoints..."
for i in {1..30}; do
    if kubectl exec deployment/$DEPLOYMENT -n $NAMESPACE -- curl -f http://localhost:8080/health; then
        echo "Health check passed"
        break
    fi
    echo "Health check failed, retrying in 10s..."
    sleep 10
done

echo "Rollback completed successfully"
```

### Circuit Breaker Pattern

**Circuit Breaker Implementation**:
```rust
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct CircuitBreaker {
    state: Arc<Mutex<CircuitState>>,
    failure_threshold: usize,
    recovery_timeout: Duration,
}

#[derive(Debug)]
enum CircuitState {
    Closed { failure_count: usize },
    Open { last_failure: Instant },
    HalfOpen,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: usize, recovery_timeout: Duration) -> Self {
        Self {
            state: Arc::new(Mutex::new(CircuitState::Closed { failure_count: 0 })),
            failure_threshold,
            recovery_timeout,
        }
    }
    
    pub async fn call<F, R, E>(&self, f: F) -> Result<R, CircuitBreakerError>
    where
        F: FnOnce() -> Result<R, E>,
        E: std::error::Error,
    {
        // Check if circuit is open
        if self.is_open()? {
            return Err(CircuitBreakerError::Open);
        }
        
        // Execute function
        match f() {
            Ok(result) => {
                self.on_success();
                Ok(result)
            }
            Err(e) => {
                self.on_failure();
                Err(CircuitBreakerError::Failure(Box::new(e)))
            }
        }
    }
    
    fn is_open(&self) -> Result<bool, CircuitBreakerError> {
        let mut state = self.state.lock().unwrap();
        
        match *state {
            CircuitState::Open { last_failure } => {
                if last_failure.elapsed() > self.recovery_timeout {
                    *state = CircuitState::HalfOpen;
                    Ok(false)
                } else {
                    Ok(true)
                }
            }
            _ => Ok(false),
        }
    }
    
    fn on_success(&self) {
        let mut state = self.state.lock().unwrap();
        *state = CircuitState::Closed { failure_count: 0 };
    }
    
    fn on_failure(&self) {
        let mut state = self.state.lock().unwrap();
        
        match *state {
            CircuitState::Closed { failure_count } => {
                let new_count = failure_count + 1;
                if new_count >= self.failure_threshold {
                    *state = CircuitState::Open {
                        last_failure: Instant::now(),
                    };
                } else {
                    *state = CircuitState::Closed {
                        failure_count: new_count,
                    };
                }
            }
            CircuitState::HalfOpen => {
                *state = CircuitState::Open {
                    last_failure: Instant::now(),
                };
            }
            _ => {}
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CircuitBreakerError {
    #[error("Circuit breaker is open")]
    Open,
    #[error("Function call failed: {0}")]
    Failure(Box<dyn std::error::Error + Send + Sync>),
}
```

---

**Migration Success Checklist**:
- ✅ Zero-downtime deployment achieved
- ✅ Performance improvements verified
- ✅ Error rates remain stable or improve
- ✅ Monitoring and alerting functional
- ✅ Rollback procedures tested
- ✅ Security policies applied
- ✅ Scaling behavior validated
- ✅ Documentation updated