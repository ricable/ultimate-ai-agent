# Deployment Guide

Complete guide for deploying the Climate Prediction System to production environments.

## Table of Contents

1. [Deployment Options](#deployment-options)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Cloud Platforms](#cloud-platforms)
5. [Performance Optimization](#performance-optimization)
6. [Monitoring](#monitoring)
7. [Security](#security)
8. [Scaling](#scaling)

## Deployment Options

### Option 1: Docker (Recommended for Single Server)
- **Pros**: Simple, portable, easy to manage
- **Cons**: Limited scaling, single point of failure
- **Best for**: Small to medium deployments (< 1000 req/s)

### Option 2: Kubernetes
- **Pros**: Auto-scaling, high availability, cloud-native
- **Cons**: More complex, requires orchestration knowledge
- **Best for**: Large deployments (> 1000 req/s), production systems

### Option 3: Serverless (AWS Lambda, Cloud Run)
- **Pros**: Pay-per-use, automatic scaling, zero maintenance
- **Cons**: Cold start latency, execution time limits
- **Best for**: Variable workloads, cost optimization

### Option 4: Bare Metal
- **Pros**: Maximum performance, full control
- **Cons**: Manual management, higher operational cost
- **Best for**: High-performance requirements, on-premise deployments

## Docker Deployment

### Step 1: Create Dockerfile

```dockerfile
# Dockerfile
FROM rust:1.70 as builder

WORKDIR /app

# Copy dependency files
COPY Cargo.toml Cargo.lock ./

# Build dependencies (cached layer)
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release && \
    rm -rf src

# Copy source code
COPY . .

# Build application
RUN cargo build --release

# Production image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 climate && \
    mkdir -p /app/models /app/logs && \
    chown -R climate:climate /app

WORKDIR /app

# Copy binary and assets
COPY --from=builder /app/target/release/climate-api ./
COPY --from=builder /app/models ./models/
COPY config.production.toml ./config.toml

USER climate

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["./climate-api"]
```

### Step 2: Create docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    image: climate-prediction:latest
    ports:
      - "8080:8080"
    environment:
      - RUST_LOG=info
      - SERVER_HOST=0.0.0.0
      - SERVER_PORT=8080
      - REASONINGBANK_BACKEND=wasm
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
      - ./config.production.toml:/app/config.toml:ro
    restart: unless-stopped
    networks:
      - climate-net
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 3s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
    restart: unless-stopped
    networks:
      - climate-net

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped
    networks:
      - climate-net

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./grafana/datasources:/etc/grafana/provisioning/datasources:ro
    depends_on:
      - prometheus
    restart: unless-stopped
    networks:
      - climate-net

networks:
  climate-net:
    driver: bridge

volumes:
  prometheus-data:
  grafana-data:
```

### Step 3: Nginx Configuration

```nginx
# nginx.conf
events {
    worker_connections 4096;
}

http {
    upstream climate-api {
        least_conn;
        server api:8080 max_fails=3 fail_timeout=30s;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/m;
    limit_req_status 429;

    # Cache configuration
    proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=api_cache:10m
                     max_size=1g inactive=60m use_temp_path=off;

    server {
        listen 80;
        server_name api.climate-prediction.io;

        # Redirect to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name api.climate-prediction.io;

        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        ssl_prefer_server_ciphers on;

        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Strict-Transport-Security "max-age=31536000" always;

        # Compression
        gzip on;
        gzip_types text/plain application/json;
        gzip_min_length 1000;

        location /api/ {
            # Rate limiting
            limit_req zone=api_limit burst=20 nodelay;

            # Proxy settings
            proxy_pass http://climate-api;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;

            # Timeouts
            proxy_connect_timeout 5s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;

            # Caching for GET requests
            proxy_cache api_cache;
            proxy_cache_methods GET;
            proxy_cache_valid 200 5m;
            proxy_cache_key "$scheme$request_method$host$request_uri";
            add_header X-Cache-Status $upstream_cache_status;
        }

        location /health {
            proxy_pass http://climate-api/health;
            access_log off;
        }

        location /metrics {
            proxy_pass http://climate-api/metrics;
            allow 10.0.0.0/8;  # Internal network only
            deny all;
        }
    }
}
```

### Step 4: Deploy with Docker Compose

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f api

# Scale API instances
docker-compose up -d --scale api=3

# Stop services
docker-compose down

# Update and restart
docker-compose pull
docker-compose up -d --force-recreate
```

## Kubernetes Deployment

### Step 1: Create Namespace

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: climate-prediction
```

### Step 2: Create Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: climate-api
  namespace: climate-prediction
  labels:
    app: climate-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: climate-api
  template:
    metadata:
      labels:
        app: climate-api
    spec:
      containers:
      - name: api
        image: climate-prediction:latest
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: RUST_LOG
          value: "info"
        - name: SERVER_HOST
          value: "0.0.0.0"
        - name: SERVER_PORT
          value: "8080"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 15
          periodSeconds: 20
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
        - name: config
          mountPath: /app/config.toml
          subPath: config.toml
          readOnly: true
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: climate-models-pvc
      - name: config
        configMap:
          name: climate-config
```

### Step 3: Create Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: climate-api
  namespace: climate-prediction
spec:
  selector:
    app: climate-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

### Step 4: Create Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: climate-api-ingress
  namespace: climate-prediction
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.climate-prediction.io
    secretName: climate-tls
  rules:
  - host: api.climate-prediction.io
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: climate-api
            port:
              number: 80
```

### Step 5: Create HorizontalPodAutoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: climate-api-hpa
  namespace: climate-prediction
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: climate-api
  minReplicas: 3
  maxReplicas: 10
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
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 60
```

### Step 6: Deploy to Kubernetes

```bash
# Apply configurations
kubectl apply -f namespace.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml

# Check status
kubectl get pods -n climate-prediction
kubectl get svc -n climate-prediction
kubectl get ingress -n climate-prediction

# View logs
kubectl logs -f deployment/climate-api -n climate-prediction

# Scale manually
kubectl scale deployment climate-api --replicas=5 -n climate-prediction
```

## Cloud Platforms

### AWS Deployment

**Using ECS Fargate:**

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name climate-prediction

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster climate-prediction \
  --service-name climate-api \
  --task-definition climate-api:1 \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### Google Cloud Deployment

**Using Cloud Run:**

```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/climate-api

# Deploy to Cloud Run
gcloud run deploy climate-api \
  --image gcr.io/PROJECT_ID/climate-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --min-instances 2 \
  --max-instances 10 \
  --cpu 2 \
  --memory 2Gi
```

### Azure Deployment

**Using Container Instances:**

```bash
# Create resource group
az group create --name climate-prediction --location eastus

# Deploy container
az container create \
  --resource-group climate-prediction \
  --name climate-api \
  --image climate-prediction:latest \
  --cpu 2 \
  --memory 2 \
  --ports 8080 \
  --dns-name-label climate-api
```

## Performance Optimization

### 1. Database Optimization

```toml
[database]
pool_size = 20
max_overflow = 10
connection_timeout = 30
```

### 2. Caching Strategy

```rust
use moka::future::Cache;
use std::time::Duration;

let cache = Cache::builder()
    .max_capacity(10_000)
    .time_to_live(Duration::from_secs(300))
    .time_to_idle(Duration::from_secs(60))
    .build();
```

### 3. Connection Pooling

```toml
[server]
workers = 8  # CPU cores
keep_alive = 75
client_timeout = 5000
```

## Monitoring

See full monitoring setup in docker-compose.yml above.

### Prometheus Metrics

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'climate-api'
    static_configs:
      - targets: ['api:8080']
```

### Grafana Dashboards

Import dashboard from `/examples/climate-prediction/docs/grafana-dashboard.json`

## Security

### 1. SSL/TLS Configuration

Use Let's Encrypt for free SSL certificates:

```bash
# Install certbot
apt-get install certbot python3-certbot-nginx

# Obtain certificate
certbot --nginx -d api.climate-prediction.io
```

### 2. API Key Management

Store API keys in secrets:

```bash
# Kubernetes secret
kubectl create secret generic api-keys \
  --from-literal=master-key=your-secret-key \
  -n climate-prediction
```

### 3. Network Security

```yaml
# NetworkPolicy for Kubernetes
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: climate-api-netpol
spec:
  podSelector:
    matchLabels:
      app: climate-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8080
```

## Scaling

### Horizontal Scaling

Kubernetes HPA automatically scales based on CPU/memory. Manual scaling:

```bash
# Scale to 10 replicas
kubectl scale deployment climate-api --replicas=10 -n climate-prediction
```

### Vertical Scaling

Update resource limits:

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "4000m"
```

---

**For production support, contact: devops@climate-prediction.io**
