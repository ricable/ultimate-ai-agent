# Production Deployment Guide

This comprehensive guide covers production deployment strategies, scaling, and operations for SAFLA (Self-Aware Feedback Loop Algorithm). Our deployment approach emphasizes reliability, scalability, and maintainability in production environments.

## Table of Contents

1. [Deployment Philosophy](#deployment-philosophy)
2. [Environment Setup](#environment-setup)
3. [Container Deployment](#container-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Platform Deployment](#cloud-platform-deployment)
6. [Scaling Strategies](#scaling-strategies)
7. [Load Balancing](#load-balancing)
8. [Database and Storage](#database-and-storage)
9. [Monitoring and Observability](#monitoring-and-observability)
10. [Security Considerations](#security-considerations)
11. [CI/CD Pipeline](#cicd-pipeline)
12. [Backup and Recovery](#backup-and-recovery)
13. [Performance Optimization](#performance-optimization)
14. [Troubleshooting](#troubleshooting)

## Deployment Philosophy

SAFLA's deployment strategy is built on several core principles:

### Infrastructure as Code
All infrastructure is defined and managed through code:
- **Version Control**: Infrastructure definitions are version controlled
- **Reproducibility**: Environments can be recreated consistently
- **Automation**: Deployments are automated and repeatable
- **Testing**: Infrastructure changes are tested before production

### Microservices Architecture
SAFLA components are deployed as independent services:
- **Component Isolation**: Each component can be scaled independently
- **Fault Tolerance**: Failure in one component doesn't affect others
- **Technology Flexibility**: Different components can use optimal technologies
- **Team Autonomy**: Teams can deploy components independently

### Zero-Downtime Deployments
Production deployments maintain service availability:
- **Rolling Updates**: Gradual replacement of instances
- **Blue-Green Deployments**: Switch between environments
- **Canary Releases**: Gradual traffic shifting to new versions
- **Health Checks**: Automated verification of deployment success

## Environment Setup

### Development Environment
```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  safla-core:
    build:
      context: .
      dockerfile: Dockerfile.dev
    environment:
      - ENVIRONMENT=development
      - LOG_LEVEL=DEBUG
      - ENABLE_HOT_RELOAD=true
    volumes:
      - .:/app
      - /app/node_modules
    ports:
      - "3000:3000"
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: safla_dev
      POSTGRES_USER: safla
      POSTGRES_PASSWORD: dev_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  redis_data:
  postgres_data:
```

### Staging Environment
```yaml
# docker-compose.staging.yml
version: '3.8'
services:
  safla-core:
    image: safla/core:${VERSION}
    environment:
      - ENVIRONMENT=staging
      - LOG_LEVEL=INFO
      - DATABASE_URL=${STAGING_DATABASE_URL}
      - REDIS_URL=${STAGING_REDIS_URL}
    ports:
      - "3000:3000"
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  safla-vector-memory:
    image: safla/vector-memory:${VERSION}
    environment:
      - ENVIRONMENT=staging
      - VECTOR_DIMENSION=768
      - MAX_VECTORS=100000
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 2G
          cpus: '1.0'

  safla-mcp-orchestrator:
    image: safla/mcp-orchestrator:${VERSION}
    environment:
      - ENVIRONMENT=staging
      - MAX_CONCURRENT_TASKS=50
    deploy:
      replicas: 2
```

### Production Environment Configuration
```bash
# production.env
ENVIRONMENT=production
LOG_LEVEL=WARN
NODE_ENV=production

# Database Configuration
DATABASE_URL=postgresql://user:password@prod-db:5432/safla_prod
DATABASE_POOL_SIZE=20
DATABASE_TIMEOUT=30000

# Redis Configuration
REDIS_URL=redis://prod-redis:6379
REDIS_POOL_SIZE=10

# Security
JWT_SECRET=${JWT_SECRET}
ENCRYPTION_KEY=${ENCRYPTION_KEY}
API_RATE_LIMIT=1000

# Performance
MAX_CONCURRENT_REQUESTS=1000
WORKER_PROCESSES=4
MEMORY_LIMIT=2048

# Monitoring
METRICS_ENABLED=true
TRACING_ENABLED=true
HEALTH_CHECK_INTERVAL=30

# External Services
MCP_SERVER_ENDPOINTS=${MCP_ENDPOINTS}
VECTOR_STORE_URL=${VECTOR_STORE_URL}
```

## Container Deployment

### Multi-Stage Dockerfile
```dockerfile
# Dockerfile
FROM node:18-alpine AS base
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

FROM node:18-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build
RUN npm run test

FROM node:18-alpine AS production
WORKDIR /app

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S safla -u 1001

# Copy built application
COPY --from=base /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY --from=build /app/package*.json ./

# Set ownership
RUN chown -R safla:nodejs /app
USER safla

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node dist/health-check.js

EXPOSE 3000
CMD ["node", "dist/index.js"]
```

### Component-Specific Dockerfiles

#### Vector Memory Manager
```dockerfile
# Dockerfile.vector-memory
FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS production
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libblas3 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Copy application
COPY src/ ./src/
COPY config/ ./config/

# Create non-root user
RUN useradd -m -u 1001 safla
RUN chown -R safla:safla /app
USER safla

EXPOSE 8000
CMD ["python", "-m", "src.vector_memory.server"]
```

#### MCP Orchestrator
```dockerfile
# Dockerfile.mcp-orchestrator
FROM node:18-alpine AS base
WORKDIR /app

# Install dependencies for native modules
RUN apk add --no-cache python3 make g++

COPY package*.json ./
RUN npm ci --only=production

FROM node:18-alpine AS production
WORKDIR /app

# Copy dependencies
COPY --from=base /app/node_modules ./node_modules

# Copy application
COPY src/ ./src/
COPY config/ ./config/
COPY package*.json ./

# Create non-root user
RUN adduser -D -s /bin/sh safla
RUN chown -R safla:safla /app
USER safla

EXPOSE 3001
CMD ["node", "src/mcp-orchestrator/server.js"]
```

### Docker Compose for Production
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  safla-core:
    image: safla/core:${VERSION:-latest}
    environment:
      - ENVIRONMENT=production
    env_file:
      - production.env
    ports:
      - "3000:3000"
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - safla-network

  safla-vector-memory:
    image: safla/vector-memory:${VERSION:-latest}
    environment:
      - ENVIRONMENT=production
    env_file:
      - production.env
    ports:
      - "8000:8000"
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
    volumes:
      - vector_data:/app/data
    networks:
      - safla-network

  safla-mcp-orchestrator:
    image: safla/mcp-orchestrator:${VERSION:-latest}
    environment:
      - ENVIRONMENT=production
    env_file:
      - production.env
    ports:
      - "3001:3001"
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    networks:
      - safla-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - safla-core
    networks:
      - safla-network

networks:
  safla-network:
    driver: overlay
    attachable: true

volumes:
  vector_data:
    driver: local
```

## Kubernetes Deployment

### Namespace and ConfigMap
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: safla-production
  labels:
    name: safla-production
    environment: production

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: safla-config
  namespace: safla-production
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  NODE_ENV: "production"
  METRICS_ENABLED: "true"
  HEALTH_CHECK_INTERVAL: "30"
```

### Secrets Management
```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: safla-secrets
  namespace: safla-production
type: Opaque
data:
  DATABASE_URL: <base64-encoded-database-url>
  REDIS_URL: <base64-encoded-redis-url>
  JWT_SECRET: <base64-encoded-jwt-secret>
  ENCRYPTION_KEY: <base64-encoded-encryption-key>
```

### Core Service Deployment
```yaml
# k8s/safla-core-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: safla-core
  namespace: safla-production
  labels:
    app: safla-core
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: safla-core
  template:
    metadata:
      labels:
        app: safla-core
        version: v1
    spec:
      containers:
      - name: safla-core
        image: safla/core:latest
        ports:
        - containerPort: 3000
        env:
        - name: PORT
          value: "3000"
        envFrom:
        - configMapRef:
            name: safla-config
        - secretRef:
            name: safla-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        securityContext:
          runAsNonRoot: true
          runAsUser: 1001
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: tmp
        emptyDir: {}
      - name: logs
        emptyDir: {}
      securityContext:
        fsGroup: 1001

---
# k8s/safla-core-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: safla-core-service
  namespace: safla-production
  labels:
    app: safla-core
spec:
  selector:
    app: safla-core
  ports:
  - port: 80
    targetPort: 3000
    protocol: TCP
  type: ClusterIP
```

### Vector Memory Service
```yaml
# k8s/vector-memory-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: safla-vector-memory
  namespace: safla-production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: safla-vector-memory
  template:
    metadata:
      labels:
        app: safla-vector-memory
    spec:
      containers:
      - name: safla-vector-memory
        image: safla/vector-memory:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: safla-config
        - secretRef:
            name: safla-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: vector-storage
          mountPath: /app/data
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
      volumes:
      - name: vector-storage
        persistentVolumeClaim:
          claimName: vector-storage-pvc

---
# k8s/vector-storage-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vector-storage-pvc
  namespace: safla-production
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
```

### Horizontal Pod Autoscaler
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: safla-core-hpa
  namespace: safla-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: safla-core
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
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

### Ingress Configuration
```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: safla-ingress
  namespace: safla-production
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
spec:
  tls:
  - hosts:
    - api.safla.example.com
    secretName: safla-tls
  rules:
  - host: api.safla.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: safla-core-service
            port:
              number: 80
      - path: /vector
        pathType: Prefix
        backend:
          service:
            name: safla-vector-memory-service
            port:
              number: 80
```

## Cloud Platform Deployment

### AWS EKS Deployment
```yaml
# aws/eks-cluster.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: safla-production
  region: us-west-2
  version: "1.28"

iam:
  withOIDC: true

managedNodeGroups:
  - name: safla-workers
    instanceType: m5.xlarge
    minSize: 3
    maxSize: 10
    desiredCapacity: 5
    volumeSize: 100
    volumeType: gp3
    ssh:
      allow: false
    iam:
      withAddonPolicies:
        autoScaler: true
        cloudWatch: true
        ebs: true
        efs: true
    tags:
      Environment: production
      Application: safla

addons:
  - name: vpc-cni
  - name: coredns
  - name: kube-proxy
  - name: aws-ebs-csi-driver

cloudWatch:
  clusterLogging:
    enableTypes: ["*"]
```

### AWS RDS Configuration
```yaml
# aws/rds.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'SAFLA Production Database'

Parameters:
  Environment:
    Type: String
    Default: production
  
  DBInstanceClass:
    Type: String
    Default: db.r5.xlarge
  
  AllocatedStorage:
    Type: Number
    Default: 500

Resources:
  DBSubnetGroup:
    Type: AWS::RDS::DBSubnetGroup
    Properties:
      DBSubnetGroupDescription: Subnet group for SAFLA database
      SubnetIds:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2
      Tags:
        - Key: Name
          Value: !Sub ${Environment}-safla-db-subnet-group

  DatabaseCluster:
    Type: AWS::RDS::DBCluster
    Properties:
      Engine: aurora-postgresql
      EngineVersion: '15.4'
      DatabaseName: safla
      MasterUsername: safla_admin
      MasterUserPassword: !Ref DBPassword
      DBSubnetGroupName: !Ref DBSubnetGroup
      VpcSecurityGroupIds:
        - !Ref DatabaseSecurityGroup
      BackupRetentionPeriod: 30
      PreferredBackupWindow: "03:00-04:00"
      PreferredMaintenanceWindow: "sun:04:00-sun:05:00"
      StorageEncrypted: true
      DeletionProtection: true
      EnableCloudwatchLogsExports:
        - postgresql
      Tags:
        - Key: Environment
          Value: !Ref Environment
        - Key: Application
          Value: safla

  DatabasePrimaryInstance:
    Type: AWS::RDS::DBInstance
    Properties:
      Engine: aurora-postgresql
      DBInstanceClass: !Ref DBInstanceClass
      DBClusterIdentifier: !Ref DatabaseCluster
      PubliclyAccessible: false
      MonitoringInterval: 60
      MonitoringRoleArn: !GetAtt RDSEnhancedMonitoringRole.Arn
      PerformanceInsightsEnabled: true
      PerformanceInsightsRetentionPeriod: 7
```

### Azure AKS Deployment
```yaml
# azure/aks-cluster.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: aks-deployment-config
data:
  deploy.sh: |
    #!/bin/bash
    
    # Create resource group
    az group create --name safla-production --location eastus
    
    # Create AKS cluster
    az aks create \
      --resource-group safla-production \
      --name safla-cluster \
      --node-count 3 \
      --node-vm-size Standard_D4s_v3 \
      --enable-addons monitoring \
      --enable-cluster-autoscaler \
      --min-count 3 \
      --max-count 10 \
      --kubernetes-version 1.28.0 \
      --enable-managed-identity \
      --attach-acr saflaregistry
    
    # Get credentials
    az aks get-credentials --resource-group safla-production --name safla-cluster
    
    # Install ingress controller
    helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
    helm install ingress-nginx ingress-nginx/ingress-nginx \
      --namespace ingress-nginx \
      --create-namespace \
      --set controller.service.type=LoadBalancer
```

### Google GKE Deployment
```yaml
# gcp/gke-cluster.yaml
apiVersion: container.cnrm.cloud.google.com/v1beta1
kind: ContainerCluster
metadata:
  name: safla-production
  namespace: safla-system
spec:
  location: us-central1
  initialNodeCount: 3
  
  nodeConfig:
    machineType: n1-standard-4
    diskSizeGb: 100
    diskType: pd-ssd
    oauthScopes:
    - "https://www.googleapis.com/auth/cloud-platform"
    
  addonsConfig:
    horizontalPodAutoscaling:
      disabled: false
    httpLoadBalancing:
      disabled: false
    networkPolicyConfig:
      disabled: false
      
  networkPolicy:
    enabled: true
    
  ipAllocationPolicy:
    useIpAliases: true
    
  workloadIdentityConfig:
    workloadPool: PROJECT_ID.svc.id.goog
    
  releaseChannel:
    channel: STABLE
    
  maintenancePolicy:
    window:
      dailyMaintenanceWindow:
        startTime: "03:00"
```

## Scaling Strategies

### Horizontal Scaling
```yaml
# k8s/scaling/horizontal-scaling.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: safla-comprehensive-hpa
  namespace: safla-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: safla-core
  minReplicas: 3
  maxReplicas: 20
  metrics:
  # CPU-based scaling
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  
  # Memory-based scaling
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  
  # Custom metrics scaling
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  
  # External metrics scaling
  - type: External
    external:
      metric:
        name: queue_depth
        selector:
          matchLabels:
            queue: safla-tasks
      target:
        type: Value
        value: "50"
  
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Min
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max
```

### Vertical Scaling
```yaml
# k8s/scaling/vertical-scaling.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: safla-core-vpa
  namespace: safla-production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: safla-core
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: safla-core
      minAllowed:
        cpu: 100m
        memory: 128Mi
      maxAllowed:
        cpu: 2
        memory: 4Gi
      controlledResources: ["cpu", "memory"]
      controlledValues: RequestsAndLimits
```

### Cluster Autoscaling
```yaml
# k8s/scaling/cluster-autoscaler.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
  labels:
    app: cluster-autoscaler
spec:
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
    spec:
      serviceAccountName: cluster-autoscaler
      containers:
      - image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.21.0
        name: cluster-autoscaler
        resources:
          limits:
            cpu: 100m
            memory: 300Mi
          requests:
            cpu: 100m
            memory: 300Mi
        command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=aws
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/safla-production
        - --balance-similar-node-groups
        - --scale-down-enabled=true
        - --scale-down-delay-after-add=10m
        - --scale-down-unneeded-time=10m
        - --scale-down-utilization-threshold=0.5
        env:
        - name: AWS_REGION
          value: us-west-2
```

## Load Balancing

### NGINX Configuration
```nginx
# nginx/nginx.conf
upstream safla_backend {
    least_conn;
    server safla-core-1:3000 max_fails=3 fail_timeout=30s;
    server safla-core-2:3000 max_fails=3 fail_timeout=30s;
    server safla-core-3:3000 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

upstream safla_vector_backend {
    ip_hash;  # Sticky sessions for vector operations
    server safla-vector-1:8000 max_fails=2 fail_timeout=30s;
    server safla-vector-2:8000 max_fails=2 fail_timeout=30s;
    keepalive 16;
}

server {
    listen 80;
    listen 443 ssl http2;
    server_name api.safla.example.com;
    
    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/safla.crt;
    ssl_certificate_key /etc/nginx/ssl/safla.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    # Security Headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=vector:10m rate=5r/s;
    
    # Gzip Compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain application/json application/javascript text/css;
    
    # Main API
    location / {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://safla_backend;
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
        
        # Health check
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
    }
    
    # Vector API
    location /vector/ {
        limit_req zone=vector burst=10 nodelay;
        
        proxy_pass http://safla_vector_backend/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Longer timeouts for vector operations
        proxy_connect_timeout 10s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
        
        # Larger body size for vector uploads
        client_max_body_size 50M;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://safla_backend/health;
        proxy_connect_timeout 2s;
        proxy_send_timeout 2s;
        proxy_read_timeout 2s;
    }
    
    # Static assets
    location /static/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        try_files $uri =404;
    }
}

# Rate limiting configuration
limit_req_status 429;
limit_req_log_level warn;
```

### HAProxy Configuration
```haproxy
# haproxy/haproxy.cfg
global
    daemon
    maxconn 4096
    log stdout local0
    
    # SSL Configuration
    ssl-default-bind-ciphers ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305
    ssl-default-bind-options ssl-min-ver TLSv1.2 no-tls-tickets
    
defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog
    option dontlognull
    option redispatch
    retries 3
    
    # Health checks
    option httpchk GET /health
    http-check expect status 200
    
frontend safla_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/safla.pem
    
    # Redirect HTTP to HTTPS
    redirect scheme https if !{ ssl_fc }
    
    # Rate limiting
    stick-table type ip size 100k expire 30s store http_req_rate(10s)
    http-request track-sc0 src
    http-request reject if { sc_http_req_rate(0) gt 20 }
    
    # Route based on path
    acl is_vector_api path_beg /vector
    acl is_mcp_api path_beg /mcp
    
    use_backend safla_vector_backend if is_vector_api
    use_backend safla_mcp_backend if is_mcp_api
    default_backend safla_core_backend

backend safla_core_backend
    balance roundrobin
    option httpchk GET /health
    
    server core1 safla-core-1:3000 check inter 5s fall 3 rise 2
    server core2 safla-core-2:3000 check inter 5s fall 3 rise 2
    server core3 safla-core-3:3000 check inter 5s fall 3 rise 2

backend safla_vector_backend
    balance source  # Sticky sessions
    option httpchk GET /health
    
    server vector1 safla-vector-1:8000 check inter 10s fall 2 rise 2
    server vector2 safla-vector-2:8000 check inter 10s fall 2 rise 2

backend safla_mcp_backend
    balance leastconn
    option httpchk GET /health
    
    server mcp1 safla-mcp-1:3001 check inter 5s fall 3 rise 2
    server mcp2 safla-mcp-2:3001 check inter 5s fall 3 rise 2
```

This deployment guide provides comprehensive coverage of production deployment strategies for SAFLA, ensuring reliable, scalable, and maintainable operations across different environments and platforms.