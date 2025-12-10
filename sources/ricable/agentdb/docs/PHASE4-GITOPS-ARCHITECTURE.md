# Phase 4 GitOps Architecture

## 🔄 Enterprise-Grade GitOps Deployment Pipeline

**Complete Automated Deployment with ArgoCD and Kustomize**

---

## Overview

The GitOps architecture implements a complete CI/CD pipeline with:

- **ArgoCD** for GitOps-based continuous deployment
- **Kustomize** for environment-specific configurations
- **Helm Charts** for templated application deployment
- **Multi-Environment Support** with promotion workflows
- **Automated Testing** with quality gates
- **Rollback Capabilities** with one-click recovery

## GitOps Architecture Diagram

```mermaid
graph TB
    subgraph "Development Workflow"
        Dev[Developer]
        GitLab[GitLab Repository]
        CI[GitLab CI/CD]
        Registry[Container Registry]
    end

    subgraph "GitOps Control Plane"
        ArgoCD[ArgoCD Application]
        Kustomize[Kustomize Overlays]
        Helm[Helm Charts]
        Repo[GitOps Repository]
    end

    subgraph "Kubernetes Clusters"
        subgraph "Development Cluster"
            DevApps[Dev Applications]
            DevMonitor[Dev Monitoring]
        end

        subgraph "Staging Cluster"
            StagingApps[Staging Apps]
            StagingTests[Integration Tests]
        end

        subgraph "Production Cluster"
            ProdApps[Production Apps]
            ProdMonitor[Production Monitoring]
        end
    end

    Dev --> GitLab
    GitLab --> CI
    CI --> Registry
    CI --> Repo

    Repo --> ArgoCD
    ArgoCD --> Kustomize
    ArgoCD --> Helm

    ArgoCD --> DevApps
    ArgoCD --> StagingApps
    ArgoCD --> ProdApps

    StagingTests -->|Promotion| Repo
```

## GitOps Repository Structure

```
ran-automation-gitops/
├── README.md
├── argocd/
│   ├── applications/
│   │   ├── agentdb-cluster.yaml
│   │   ├── swarm-coordinator.yaml
│   │   ├── optimization-engine.yaml
│   │   ├── monitoring-stack.yaml
│   │   └── infrastructure.yaml
│   ├── projects/
│   │   ├── ran-optimization.yaml
│   │   ├── monitoring.yaml
│   │   └── infrastructure.yaml
│   └── appprojects/
│       ├── ran-optimization.yaml
│       └── monitoring.yaml
├── environments/
│   ├── development/
│   │   ├── kustomization.yaml
│   │   ├── agentdb/
│   │   ├── swarm-coordinator/
│   │   ├── optimization-engine/
│   │   └── monitoring/
│   ├── staging/
│   │   ├── kustomization.yaml
│   │   ├── agentdb/
│   │   ├── swarm-coordinator/
│   │   ├── optimization-engine/
│   │   └── monitoring/
│   └── production/
│       ├── kustomization.yaml
│       ├── agentdb/
│       ├── swarm-coordinator/
│       ├── optimization-engine/
│       └── monitoring/
├── charts/
│   ├── agentdb/
│   │   ├── Chart.yaml
│   │   ├── values.yaml
│   │   ├── values-dev.yaml
│   │   ├── values-staging.yaml
│   │   ├── values-prod.yaml
│   │   └── templates/
│   ├── swarm-coordinator/
│   ├── optimization-engine/
│   ├── monitoring/
│   └── infrastructure/
├── scripts/
│   ├── deploy.sh
│   ├── promote.sh
│   ├── rollback.sh
│   └── health-check.sh
└── docs/
    ├── deployment-guide.md
    ├── troubleshooting.md
    └── architecture.md
```

## ArgoCD Configuration

### ArgoCD Installation
```yaml
# argocd-install.yaml
apiVersion: argoproj.io/v1alpha1
kind: ArgoCD
metadata:
  name: argocd
  namespace: argocd
spec:
  controller:
    processors:
      operation: 10
      status: 20
  dex:
    config: |
      connectors:
      - type: oidc
        id: oidc
        name: Ericsson SSO
        config:
          issuer: https://sso.ericsson.com
          clientID: argocd
          clientSecret: $oidc.clientSecret
          requestedScopes: ["openid", "profile", "email", "groups"]
  redis:
    image: redis:7.0.8-alpine
  repository:
    insecure: false
  server:
    replicas: 3
    grpc:
      web: true
    insecure: false
    service:
      type: LoadBalancer
    resources:
      limits:
        cpu: 500m
        memory: 512Mi
      requests:
        cpu: 250m
        memory: 256Mi
    extraArgs:
    - --insecure
  notifications:
    enabled: true
```

### ArgoCD Projects
```yaml
# argocd-project-ran-optimization.yaml
apiVersion: argoproj.io/v1alpha1
kind: AppProject
metadata:
  name: ran-optimization
  namespace: argocd
spec:
  description: "Ericsson RAN Optimization Platform"
  sourceRepos:
  - https://gitlab.ericsson.com/ran-automation/ran-automation-gitops.git
  - https://gitlab.ericsson.com/ran-automation/ran-automation.git
  destinations:
  - namespace: ran-optimization-dev
    server: https://k8s-dev.ericsson-ran.local
  - namespace: ran-optimization-staging
    server: https://k8s-staging.ericsson-ran.local
  - namespace: ran-optimization
    server: https://k8s-prod.ericsson-ran.local
  clusterResourceWhitelist:
  - group: '*'
    kind: '*'
  namespaceResourceWhitelist:
  - group: '*'
    kind: '*'
  roles:
  - name: developers
    description: "Developer access to RAN optimization project"
    policies:
    - p, proj:ran-optimization:developers, applications, *, ran-optimization/*, allow
    - p, proj:ran-optimization:developers, applications, sync, ran-optimization/*, allow
    groups:
    - ericsson:ran-optimization:developers
  - name: ops
    description: "Operations team access"
    policies:
    - p, proj:ran-optimization:ops, applications, *, ran-optimization/*, allow
    groups:
    - ericsson:ran-optimization:operators
  - name: admins
    description: "Admin access to RAN optimization project"
    policies:
    - p, proj:ran-optimization:admins, applications, *, ran-optimization/*, allow
    - p, proj:ran-optimization:admins, applications, override, ran-optimization/*, allow
    groups:
    - ericsson:ran-optimization:admins
```

### ArgoCD Applications
```yaml
# argocd-application-agentdb.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: agentdb-cluster
  namespace: argocd
  labels:
    app: agentdb
    environment: production
    component: database
  finalizers:
  - resources-finalizer.argocd.argoproj.io
spec:
  project: ran-optimization
  source:
    repoURL: https://gitlab.ericsson.com/ran-automation/ran-automation-gitops.git
    targetRevision: main
    path: environments/production/agentdb
    kustomize:
      images:
      - name: ericsson/agentdb
        newTag: v4.0.0
  destination:
    server: https://k8s-prod.ericsson-ran.local
    namespace: ran-optimization
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    syncOptions:
    - CreateNamespace=true
    - PrunePropagationPolicy=foreground
    - RespectIgnoreDifferences=true
    - PruneLast=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
    ignoreDifferences:
    - group: apps
      kind: Deployment
      jsonPointers:
      - /spec/replicas
    - group: autoscaling
      kind: HorizontalPodAutoscaler
      jsonPointers:
      - /spec/minReplicas
      - /spec/maxReplicas
  revisionHistoryLimit: 3
  info:
  - name: "Application Owner"
    value: "RAN Optimization Team"
  - name: "Environment"
    value: "Production"
  - name: "Component"
    value: "AgentDB Cluster"
  - name: "SLA"
    value: "99.9% Availability"
---
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: swarm-coordinator
  namespace: argocd
  labels:
    app: swarm-coordinator
    environment: production
    component: orchestration
  finalizers:
  - resources-finalizer.argocd.argoproj.io
spec:
  project: ran-optimization
  source:
    repoURL: https://gitlab.ericsson.com/ran-automation/ran-automation-gitops.git
    targetRevision: main
    path: environments/production/swarm-coordinator
    kustomize:
      images:
      - name: ericsson/ran-swarm-coordinator
        newTag: v4.0.0
  destination:
    server: https://k8s-prod.ericsson-ran.local
    namespace: ran-optimization
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    syncOptions:
    - CreateNamespace=true
    - PrunePropagationPolicy=foreground
    - RespectIgnoreDifferences=true
    - PruneLast=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
  revisionHistoryLimit: 3
  info:
  - name: "Application Owner"
    value: "RAN Optimization Team"
  - name: "Environment"
    value: "Production"
  - name: "Component"
    value: "Swarm Coordinator"
---
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: optimization-engine
  namespace: argocd
  labels:
    app: optimization-engine
    environment: production
    component: optimization
  finalizers:
  - resources-finalizer.argocd.argoproj.io
spec:
  project: ran-optimization
  source:
    repoURL: https://gitlab.ericsson.com/ran-automation/ran-automation-gitops.git
    targetRevision: main
    path: environments/production/optimization-engine
    kustomize:
      images:
      - name: ericsson/ran-optimization-engine
        newTag: v4.0.0
  destination:
    server: https://k8s-prod.ericsson-ran.local
    namespace: ran-optimization
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    syncOptions:
    - CreateNamespace=true
    - PrunePropagationPolicy=foreground
    - RespectIgnoreDifferences=true
    - PruneLast=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
  revisionHistoryLimit: 3
  info:
  - name: "Application Owner"
    value: "RAN Optimization Team"
  - name: "Environment"
    value: "Production"
  - name: "Component"
    value: "Optimization Engine"
```

## Kustomize Configurations

### Production Environment AgentDB
```yaml
# environments/production/agentdb/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

metadata:
  name: agentdb-production
  namespace: ran-optimization

resources:
- ../../charts/agentdb/templates/agentdb-statefulset.yaml
- ../../charts/agentdb/templates/agentdb-service.yaml
- ../../charts/agentdb/templates/agentdb-serviceaccount.yaml
- ../../charts/agentdb/templates/agentdb-configmap.yaml
- ../../charts/agentdb/templates/agentdb-pvc.yaml
- ../../charts/agentdb/templates/agentdb-networkpolicy.yaml
- ../../charts/agentdb/templates/agentdb-poddisruptionbudget.yaml
- ../../charts/agentdb/templates/agentdb-hpa.yaml

patchesStrategicMerge:
- agentdb-patch.yaml
- agentdb-resources.yaml
- agentdb-monitoring.yaml

patchesJson6902:
- path: agentdb-replicas.yaml
  target:
    group: apps
    version: v1
    kind: StatefulSet
    name: agentdb-cluster

images:
- name: ericsson/agentdb
  newTag: v4.0.0
  digest: sha256:abc123def456789...

commonLabels:
  app.kubernetes.io/name: agentdb
  app.kubernetes.io/version: v4.0.0
  app.kubernetes.io/component: database
  app.kubernetes.io/managed-by: argocd
  app.kubernetes.io/part-of: ran-optimization

commonAnnotations:
  argocd.argoproj.io/sync-options: CreateNamespace=true
  argocd.argoproj.io/compare-options: IgnoreExtraneousFields
```

### Production Environment Swarm Coordinator
```yaml
# environments/production/swarm-coordinator/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

metadata:
  name: swarm-coordinator-production
  namespace: ran-optimization

resources:
- ../../charts/swarm-coordinator/templates/swarm-coordinator-deployment.yaml
- ../../charts/swarm-coordinator/templates/swarm-coordinator-service.yaml
- ../../charts/swarm-coordinator/templates/swarm-coordinator-configmap.yaml
- ../../charts/swarm-coordinator/templates/swarm-coordinator-secret.yaml
- ../../charts/swarm-coordinator/templates/swarm-coordinator-serviceaccount.yaml
- ../../charts/swarm-coordinator/templates/swarm-coordinator-networkpolicy.yaml
- ../../charts/swarm-coordinator/templates/swarm-coordinator-poddisruptionbudget.yaml
- ../../charts/swarm-coordinator/templates/swarm-coordinator-hpa.yaml
- ../../charts/swarm-coordinator/templates/swarm-coordinator-ingress.yaml

patchesStrategicMerge:
- swarm-coordinator-patch.yaml
- swarm-coordinator-resources.yaml
- swarm-coordinator-monitoring.yaml
- swarm-coordinator-security.yaml

patchesJson6902:
- path: swarm-coordinator-replicas.yaml
  target:
    group: apps
    version: v1
    kind: Deployment
    name: swarm-coordinator

images:
- name: ericsson/ran-swarm-coordinator
  newTag: v4.0.0

commonLabels:
  app.kubernetes.io/name: swarm-coordinator
  app.kubernetes.io/version: v4.0.0
  app.kubernetes.io/component: orchestration
  app.kubernetes.io/managed-by: argocd
  app.kubernetes.io/part-of: ran-optimization

configMapGenerator:
- name: swarm-coordinator-config
  literals:
  - CLAUDE_FLOW_TOPOLOGY=hierarchical
  - CLAUDE_FLOW_MAX_AGENTS=50
  - CLAUDE_FLOW_STRATEGY=adaptive
  - TEMPORAL_EXPANSION_FACTOR=1000
  - OPTIMIZATION_CYCLE_MINUTES=15
  - COGNITIVE_CONSCIOUSNESS_LEVEL=maximum

secretGenerator:
- name: swarm-coordinator-secrets
  envs:
  - .env.secrets

vars:
- name: AGENTDB_ENDPOINTS
  objref:
    kind: ConfigMap
    name: agentdb-config
    apiVersion: v1
  fieldref:
    fieldpath: data.AGENTDB_ENDPOINTS
```

## Helm Charts

### AgentDB Helm Chart
```yaml
# charts/agentdb/Chart.yaml
apiVersion: v2
name: agentdb
description: Ericsson RAN AgentDB - High-performance vector database with QUIC synchronization
type: application
version: 4.0.0
appVersion: "4.0.0"
keywords:
- ericsson
- ran
- agentdb
- vector-database
- optimization
- machine-learning
home: https://github.com/ericsson/ran-automation
sources:
- https://github.com/ericsson/ran-automation
maintainers:
- name: Ericsson RAN Team
  email: ran-team@ericsson.com
dependencies:
- name: common
  version: 2.2.0
  repository: https://charts.bitnami.com/bitnami
annotations:
  category: Database
  licenses: Apache-2.0
```

### AgentDB Values - Production
```yaml
# charts/agentdb/values-prod.yaml
global:
  imageRegistry: registry.ericsson.com
  imagePullSecrets:
  - ericsson-registry-secret
  storageClass: "ssd-premium"

image:
  registry: registry.ericsson.com
  repository: ran/agentdb
  tag: v4.0.0
  pullPolicy: Always

replicaCount: 5

nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations:
    iam.amazonaws.com/role: "arn:aws:iam::123456789012:role/agentdb-cluster-role"
  name: "agentdb"

podAnnotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "9090"
  prometheus.io/path: "/metrics"
  iam.amazonaws.com/role: "arn:aws:iam::123456789012:role/agentdb-cluster-role"

podSecurityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  fsGroup: 1000

securityContext:
  capabilities:
    drop:
    - ALL
  readOnlyRootFilesystem: true
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000

service:
  type: ClusterIP
  ports:
    quicSync:
      port: 4433
      targetPort: 4433
      protocol: UDP
    http:
      port: 8080
      targetPort: 8080
      protocol: TCP
    metrics:
      port: 9090
      targetPort: 9090
      protocol: TCP

ingress:
  enabled: true
  className: "nginx-internal"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/backend-protocol: "HTTP"
  hosts:
    - host: agentdb.ericsson-ran.local
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: agentdb-tls
      hosts:
        - agentdb.ericsson-ran.local

resources:
  limits:
    cpu: 4000m
    memory: 16Gi
    ephemeral-storage: 50Gi
  requests:
    cpu: 2000m
    memory: 8Gi
    ephemeral-storage: 10Gi

autoscaling:
  enabled: false

persistence:
  enabled: true
  storageClass: "ssd-premium"
  size: 500Gi
  accessModes:
  - ReadWriteOnce
  annotations:
    volume.beta.kubernetes.io/storage-class: "ssd-premium"

nodeSelector:
  node-type: database
  performance-class: high

tolerations:
- key: "database"
  operator: "Equal"
  value: "true"
  effect: "NoSchedule"
- key: "performance-class"
  operator: "Equal"
  value: "high"
  effect: "NoSchedule"

affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchLabels:
            app.kubernetes.io/name: agentdb
        topologyKey: "kubernetes.io/hostname"
    - weight: 50
      podAffinityTerm:
        labelSelector:
          matchLabels:
            app.kubernetes.io/name: agentdb
        topologyKey: "topology.kubernetes.io/zone"

config:
  # AgentDB Configuration
  clusterSize: 5
  quicSync: true
  quicPort: 4433
  httpPort: 8080
  metricsPort: 9090
  quantization: "scalar"
  cacheSize: 4000
  hnswM: 32
  hnswEfConstruction: 200
  logLevel: "info"

  # Performance Configuration
  maxConnections: 10000
  connectionTimeout: 30000
  queryTimeout: 60000
  batchSize: 100
  maxBatchSize: 1000

  # Persistence Configuration
  walSync: true
  checkpointInterval: 300000
  maxWALSize: 1073741824  # 1GB
  compactionThreshold: 0.7

  # Backup Configuration
  backupEnabled: true
  backupInterval: 3600
  backupRetention: 168
  backupCompression: true
  backupEncryption: true

  # Security Configuration
  tlsEnabled: true
  authEnabled: true
  rbacEnabled: true
  auditEnabled: true

networkPolicy:
  enabled: true
  ingress:
    - from:
      - namespaceSelector:
          matchLabels:
            name: ran-optimization
      - namespaceSelector:
          matchLabels:
            name: monitoring
      ports:
      - protocol: UDP
        port: 4433
      - protocol: TCP
        port: 8080
      - protocol: TCP
        port: 9090
  egress:
    - to: []
      ports:
      - protocol: UDP
        port: 4433
      - protocol: TCP
        port: 8080
      - protocol: TCP
        port: 53
      - protocol: UDP
        port: 53

podDisruptionBudget:
  enabled: true
  minAvailable: 3

healthChecks:
  livenessProbe:
    httpGet:
      path: /health
      port: http
    initialDelaySeconds: 60
    periodSeconds: 10
    timeoutSeconds: 5
    failureThreshold: 3
  readinessProbe:
    httpGet:
      path: /ready
      port: http
    initialDelaySeconds: 10
    periodSeconds: 5
    timeoutSeconds: 3
    failureThreshold: 3
  startupProbe:
    httpGet:
      path: /startup
      port: http
    initialDelaySeconds: 10
    periodSeconds: 10
    timeoutSeconds: 5
    failureThreshold: 30

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    namespace: monitoring
    labels:
      release: prometheus
    interval: 30s
    scrapeTimeout: 10s
  prometheusRule:
    enabled: true
    namespace: monitoring
    rules:
      - AgentDBDown
      - AgentDBHighMemoryUsage
      - AgentDBHighCPUUsage
      - AgentDBSlowQueries
      - AgentDBReplicationLag
      - AgentDBDiskSpace
  grafanaDashboard:
    enabled: true
    namespace: monitoring

backup:
  enabled: true
  schedule: "0 */6 * * *"  # Every 6 hours
  retention: "30d"
  storageClass: "ssd-premium"
  size: "1Ti"
  encryption: true
  compression: true
  destination:
    type: "s3"
    bucket: "ericsson-ran-backups"
    region: "us-east-1"
    endpoint: "s3.amazonaws.com"

security:
  tls:
    enabled: true
    autoGenerated: false
    secretName: agentdb-tls-secret
  rbac:
    enabled: true
    serviceAccount:
      create: true
      annotations:
        eks.amazonaws.com/role-arn: "arn:aws:iam::123456789012:role/agentdb-service-role"
  podSecurityPolicy:
    enabled: false
```

## CI/CD Pipeline

### GitLab CI Configuration
```yaml
# .gitlab-ci.yml
stages:
  - validate
  - test
  - build
  - security-scan
  - deploy-dev
  - integration-test
  - deploy-staging
  - e2e-test
  - promote-production
  - post-deployment

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"
  KUBECTL_VERSION: "1.28.4"
  HELM_VERSION: "3.13.0"
  KUSTOMIZE_VERSION: "5.2.1"

before_script:
  - echo "Starting pipeline for $CI_COMMIT_BRANCH"
  - echo "Commit SHA: $CI_COMMIT_SHA"
  - echo "Pipeline ID: $CI_PIPELINE_ID"

validate:
  stage: validate
  image: alpine:3.18
  script:
    - apk add --no-cache git yamllint
    - yamllint environments/**/kustomization.yaml
    - yamllint charts/**/values*.yaml
    - yamllint argocd/**/*.yaml
    - find . -name "*.yaml" -exec yamllint {} \;
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

test:
  stage: test
  image: node:18-alpine
  script:
    - apk add --no-cache git
    - npm ci
    - npm run test:unit
    - npm run test:integration
    - npm run lint
    - npm run type-check
  coverage: '/Coverage: \d+\.\d+%/'
  artifacts:
    reports:
      junit: junit-report.xml
      coverage_report:
        coverage_format: cobertura
        path: coverage/cobertura-coverage.xml
    paths:
      - coverage/
    expire_in: 1 week
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

build:
  stage: build
  image: docker:24.0.6
  services:
    - docker:24.0.6-dind
  script:
    - echo $CI_REGISTRY_PASSWORD | docker login -u $CI_REGISTRY_USER --password-stdin $CI_REGISTRY
    - docker buildx create --use
    - docker buildx build --platform linux/amd64,linux/arm64 -t $CI_REGISTRY_IMAGE/agentdb:$CI_COMMIT_SHA -t $CI_REGISTRY_IMAGE/agentdb:latest --push .
    - docker buildx build --platform linux/amd64,linux/arm64 -t $CI_REGISTRY_IMAGE/swarm-coordinator:$CI_COMMIT_SHA -t $CI_REGISTRY_IMAGE/swarm-coordinator:latest --push .
    - docker buildx build --platform linux/amd64,linux/arm64 -t $CI_REGISTRY_IMAGE/optimization-engine:$CI_COMMIT_SHA -t $CI_REGISTRY_IMAGE/optimization-engine:latest --push .
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"

security-scan:
  stage: security-scan
  image: aquasec/trivy:latest
  script:
    - trivy image --exit-code 0 --severity HIGH,CRITICAL $CI_REGISTRY_IMAGE/agentdb:$CI_COMMIT_SHA
    - trivy image --exit-code 0 --severity HIGH,CRITICAL $CI_REGISTRY_IMAGE/swarm-coordinator:$CI_COMMIT_SHA
    - trivy image --exit-code 0 --severity HIGH,CRITICAL $CI_REGISTRY_IMAGE/optimization-engine:$CI_COMMIT_SHA
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"

deploy-dev:
  stage: deploy-dev
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context $KUBE_CONTEXT_DEV
    - git clone https://gitlab-ci-token:${CI_JOB_TOKEN}@gitlab.com/ericsson/ran-automation/ran-automation-gitops.git
    - cd ran-automation-gitops
    - git config user.name "GitLab CI"
    - git config user.email "gitlab-ci@ericsson.com"
    - sed -i "s/newTag: .*/newTag: $CI_COMMIT_SHA/" environments/development/*/kustomization.yaml
    - git add environments/development/*/kustomization.yaml
    - git commit -m "Update development images to $CI_COMMIT_SHA"
    - git push origin main
    - echo "Waiting for ArgoCD to sync..."
    - sleep 60
    - kubectl wait --for=condition=available --timeout=600s deployment/agentdb-cluster -n ran-optimization-dev
    - kubectl wait --for=condition=available --timeout=600s deployment/swarm-coordinator -n ran-optimization-dev
    - kubectl wait --for=condition=available --timeout=600s deployment/optimization-engine -n ran-optimization-dev
  environment:
    name: development
    url: https://ran-optimization-dev.ericsson-ran.local
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
  when: manual

integration-test:
  stage: integration-test
  image: node:18-alpine
  script:
    - npm ci
    - npm run test:integration:dev
    - npm run test:api:dev
  dependencies:
    - deploy-dev
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

deploy-staging:
  stage: deploy-staging
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context $KUBE_CONTEXT_STAGING
    - git clone https://gitlab-ci-token:${CI_JOB_TOKEN}@gitlab.com/ericsson/ran-automation/ran-automation-gitops.git
    - cd ran-automation-gitops
    - git config user.name "GitLab CI"
    - git config user.email "gitlab-ci@ericsson.com"
    - sed -i "s/newTag: .*/newTag: $CI_COMMIT_SHA/" environments/staging/*/kustomization.yaml
    - git add environments/staging/*/kustomization.yaml
    - git commit -m "Update staging images to $CI_COMMIT_SHA"
    - git push origin main
    - echo "Waiting for ArgoCD to sync..."
    - sleep 120
    - kubectl wait --for=condition=available --timeout=900s deployment/agentdb-cluster -n ran-optimization-staging
    - kubectl wait --for=condition=available --timeout=900s deployment/swarm-coordinator -n ran-optimization-staging
    - kubectl wait --for=condition=available --timeout=900s deployment/optimization-engine -n ran-optimization-staging
  environment:
    name: staging
    url: https://ran-optimization-staging.ericsson-ran.local
  dependencies:
    - integration-test
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
  when: manual

e2e-test:
  stage: e2e-test
  image: cypress/included:13.6.1
  script:
    - npm ci
    - npm run test:e2e:staging
  dependencies:
    - deploy-staging
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

promote-production:
  stage: promote-production
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context $KUBE_CONTEXT_PROD
    - git clone https://gitlab-ci-token:${CI_JOB_TOKEN}@gitlab.com/ericsson/ran-automation/ran-automation-gitops.git
    - cd ran-automation-gitops
    - git config user.name "GitLab CI"
    - git config user.email "gitlab-ci@ericsson.com"
    - sed -i "s/newTag: .*/newTag: $CI_COMMIT_SHA/" environments/production/*/kustomization.yaml
    - git add environments/production/*/kustomization.yaml
    - git commit -m "Update production images to $CI_COMMIT_SHA [skip ci]"
    - git push origin main
    - echo "Waiting for ArgoCD to sync..."
    - sleep 180
    - kubectl wait --for=condition=available --timeout=1200s statefulset/agentdb-cluster -n ran-optimization
    - kubectl wait --for=condition=available --timeout=1200s deployment/swarm-coordinator -n ran-optimization
    - kubectl wait --for=condition=available --timeout=1200s deployment/optimization-engine -n ran-optimization
  environment:
    name: production
    url: https://ran-optimization.ericsson-ran.local
  dependencies:
    - e2e-test
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
  when: manual
  allow_failure: false

post-deployment:
  stage: post-deployment
  image: alpine:3.18
  script:
    - apk add --no-cache curl
    - echo "Running post-deployment health checks..."
    - curl -f https://ran-optimization.ericsson-ran.local/health || exit 1
    - curl -f https://agentdb.ericsson-ran.local/health || exit 1
    - echo "All health checks passed!"
  dependencies:
    - promote-production
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
```

## Deployment Scripts

### Deployment Automation
```bash
#!/bin/bash
# scripts/deploy.sh

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-staging}"
DRY_RUN="${DRY_RUN:-false}"
FORCE="${FORCE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validate environment
validate_environment() {
    local env=$1
    case $env in
        development|staging|production)
            log_info "Valid environment: $env"
            ;;
        *)
            log_error "Invalid environment: $env. Must be one of: development, staging, production"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if required tools are installed
    local tools=("kubectl" "helm" "kustomize" "git" "argocd")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool not found: $tool"
            exit 1
        fi
    done

    # Check if we can connect to the cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check ArgoCD connection
    if ! argocd app list &> /dev/null; then
        log_error "Cannot connect to ArgoCD"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Validate configurations
validate_configurations() {
    local env=$1
    log_info "Validating configurations for $env environment..."

    cd "$PROJECT_ROOT"

    # Validate YAML files
    if ! find . -name "*.yaml" -o -name "*.yml" | xargs yamllint; then
        log_error "YAML validation failed"
        exit 1
    fi

    # Validate Kustomize configurations
    if ! kustomize build "environments/$env" > /dev/null; then
        log_error "Kustomize validation failed for $env"
        exit 1
    fi

    # Validate Helm charts
    if ! helm lint "charts/" --values "charts/agentdb/values-$env.yaml"; then
        log_error "Helm chart validation failed for $env"
        exit 1
    fi

    log_success "Configuration validation passed"
}

# Create GitOps repository if it doesn't exist
setup_gitops_repo() {
    local env=$1
    log_info "Setting up GitOps repository for $env..."

    cd "$PROJECT_ROOT"

    # Clone GitOps repo if not already present
    if [ ! -d "ran-automation-gitops" ]; then
        git clone "git@gitlab.com:ericsson/ran-automation/ran-automation-gitops.git"
    fi

    cd "ran-automation-gitops"

    # Configure git user
    git config user.name "Deployment Script"
    git config user.email "deployment@ericsson.com"

    # Pull latest changes
    git pull origin main

    log_success "GitOps repository setup complete"
}

# Update configurations
update_configurations() {
    local env=$1
    local image_tag=$2
    log_info "Updating configurations for $env environment..."

    cd "$PROJECT_ROOT/ran-automation-gitops"

    # Update image tags
    if [ -n "$image_tag" ]; then
        log_info "Updating image tags to $image_tag"
        find "environments/$env" -name "kustomization.yaml" -exec sed -i "s/newTag: .*/newTag: $image_tag/" {} \;
    fi

    # Generate configurations
    log_info "Generating Kubernetes manifests..."
    kustomize build "environments/$env" > "manifests-$env.yaml"

    # Validate generated manifests
    if ! kubectl apply --dry-run=client -f "manifests-$env.yaml"; then
        log_error "Generated manifests validation failed"
        exit 1
    fi

    log_success "Configuration update complete"
}

# Deploy to environment
deploy_environment() {
    local env=$1
    log_info "Deploying to $env environment..."

    cd "$PROJECT_ROOT/ran-automation-gitops"

    if [ "$DRY_RUN" = "true" ]; then
        log_info "DRY RUN: Skipping actual deployment"
        log_info "Would commit and push changes to trigger ArgoCD"
        return 0
    fi

    # Commit and push changes
    git add .
    if git diff --staged --quiet; then
        log_warning "No changes to commit"
        return 0
    fi

    local commit_message="Deploy to $env - $(date '+%Y-%m-%d %H:%M:%S')"
    if [ -n "$image_tag" ]; then
        commit_message="$commit_message (image: $image_tag)"
    fi

    git commit -m "$commit_message"
    git push origin main

    log_success "Changes pushed to trigger ArgoCD deployment"

    # Wait for deployment to complete
    log_info "Waiting for deployment to complete..."
    local max_wait=1800  # 30 minutes
    local wait_interval=30
    local elapsed=0

    while [ $elapsed -lt $max_wait ]; do
        local status
        status=$(argocd app get "ran-optimization-$env" --output json | jq -r '.status.sync.status')

        if [ "$status" = "Synced" ]; then
            log_success "Deployment completed successfully"
            return 0
        elif [ "$status" = "OutOfSync" ]; then
            log_info "Application is out of sync, waiting..."
        elif [ "$status" = "Unknown" ]; then
            log_warning "Application status unknown, continuing to wait..."
        else
            log_error "Deployment failed with status: $status"
            return 1
        fi

        sleep $wait_interval
        elapsed=$((elapsed + wait_interval))

        if [ $((elapsed % 120)) -eq 0 ]; then
            log_info "Still waiting... (${elapsed}s elapsed)"
        fi
    done

    log_error "Deployment timed out after ${max_wait}s"
    return 1
}

# Health check
health_check() {
    local env=$1
    log_info "Running health checks for $env environment..."

    # Determine namespace based on environment
    local namespace="ran-optimization"
    if [ "$env" = "development" ]; then
        namespace="ran-optimization-dev"
    elif [ "$env" = "staging" ]; then
        namespace="ran-optimization-staging"
    fi

    # Check if all pods are running
    local pod_status
    pod_status=$(kubectl get pods -n "$namespace" -o jsonpath='{.items[*].status.phase}')

    for status in $pod_status; do
        if [ "$status" != "Running" ] && [ "$status" != "Succeeded" ]; then
            log_error "Pod not in healthy state: $status"
            return 1
        fi
    done

    # Check service endpoints
    local services=("agentdb" "swarm-coordinator" "optimization-engine")
    for service in "${services[@]}"; do
        if ! kubectl get svc "$service" -n "$namespace" &> /dev/null; then
            log_error "Service not found: $service"
            return 1
        fi
    done

    # Run application-specific health checks
    log_info "Running application health checks..."

    # AgentDB health check
    local agentdb_ready
    agentdb_ready=$(kubectl wait --for=condition=ready pod -l app=agentdb -n "$namespace" --timeout=300s --output=jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "False")
    if [ "$agentdb_ready" != "True" ]; then
        log_error "AgentDB not ready"
        return 1
    fi

    # Swarm coordinator health check
    local swarm_ready
    swarm_ready=$(kubectl wait --for=condition=ready pod -l app=swarm-coordinator -n "$namespace" --timeout=300s --output=jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "False")
    if [ "$swarm_ready" != "True" ]; then
        log_error "Swarm coordinator not ready"
        return 1
    fi

    # Optimization engine health check
    local opt_ready
    opt_ready=$(kubectl wait --for=condition=ready pod -l app=optimization-engine -n "$namespace" --timeout=300s --output=jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "False")
    if [ "$opt_ready" != "True" ]; then
        log_error "Optimization engine not ready"
        return 1
    fi

    log_success "All health checks passed"
}

# Rollback function
rollback() {
    local env=$1
    log_warning "Rolling back $env environment..."

    cd "$PROJECT_ROOT/ran-automation-gitops"

    # Get current and previous revisions
    local current_rev
    current_rev=$(argocd app get "ran-optimization-$env" --output json | jq -r '.status.sync.revision')

    local previous_rev
    previous_rev=$(argocd app history "ran-optimization-$env" --output json | jq -r '.items[1].revision')

    if [ "$previous_rev" = "null" ] || [ -z "$previous_rev" ]; then
        log_error "No previous revision found for rollback"
        return 1
    fi

    log_info "Rolling back from $current_rev to $previous_rev"

    if [ "$DRY_RUN" = "true" ]; then
        log_info "DRY RUN: Would rollback to revision $previous_rev"
        return 0
    fi

    # Perform rollback
    argocd app rollback "ran-optimization-$env" "$previous_rev"

    # Wait for rollback to complete
    log_info "Waiting for rollback to complete..."
    if argocd app wait "ran-optimization-$env" --health --timeout 600s; then
        log_success "Rollback completed successfully"
    else
        log_error "Rollback failed"
        return 1
    fi
}

# Main function
main() {
    local environment="${1:-staging}"
    local image_tag="${2:-}"
    local action="${3:-deploy}"

    log_info "Starting deployment process for $environment environment"

    # Validate environment
    validate_environment "$environment"

    # Check prerequisites
    check_prerequisites

    # Validate configurations
    validate_configurations "$environment"

    # Setup GitOps repository
    setup_gitops_repo "$environment"

    case $action in
        deploy)
            # Update configurations
            update_configurations "$environment" "$image_tag"

            # Deploy
            if deploy_environment "$environment"; then
                # Health check
                health_check "$environment"

                log_success "Deployment to $environment completed successfully"
            else
                log_error "Deployment to $environment failed"
                exit 1
            fi
            ;;
        rollback)
            rollback "$environment"
            ;;
        validate)
            log_info "Validation completed successfully"
            ;;
        *)
            log_error "Unknown action: $action. Supported actions: deploy, rollback, validate"
            exit 1
            ;;
    esac
}

# Help function
help() {
    cat << EOF
Usage: $0 <environment> [image_tag] [action]

Arguments:
  environment    Target environment (development, staging, production)
  image_tag     Docker image tag to deploy (optional)
  action        Action to perform (deploy, rollback, validate) [default: deploy]

Environment Variables:
  DRY_RUN       Set to 'true' for dry run mode
  FORCE         Set to 'true' to skip confirmations

Examples:
  $0 staging latest deploy
  $0 production v4.0.0 deploy
  $0 staging rollback
  $0 development validate

  DRY_RUN=true $0 staging latest deploy
EOF
}

# Check for help flag
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    help
    exit 0
fi

# Check for minimum arguments
if [ $# -lt 1 ]; then
    help
    exit 1
fi

# Run main function
main "$@"
```

---

**Continue with Cloud Integration, Network, Storage, and Security architectures...**