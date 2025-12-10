# RAN Intelligent Automation System - Phase 4 Deployment Playbook

## Overview

This comprehensive deployment playbook provides step-by-step instructions for deploying the RAN Intelligent Automation System to production environments using Kubernetes, GitOps with ArgoCD, and comprehensive monitoring.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Deployment Architecture](#deployment-architecture)
3. [Three-Phase Rollout Strategy](#three-phase-rollout-strategy)
4. [Pre-Deployment Checklist](#pre-deployment-checklist)
5. [Deployment Procedures](#deployment-procedures)
6. [Monitoring and Validation](#monitoring-and-validation)
7. [Rollback Procedures](#rollback-procedures)
8. [Disaster Recovery](#disaster-recovery)
9. [Post-Deployment Activities](#post-deployment-activities)
10. [Troubleshooting Guide](#troubleshooting-guide)

## Prerequisites

### Infrastructure Requirements

#### Kubernetes Cluster
- **Kubernetes Version**: 1.24+ (recommended 1.26+)
- **Nodes**: Minimum 3 worker nodes for production
- **Node Specifications**:
  - Master nodes: 4 CPU, 8GB RAM, 100GB SSD
  - Worker nodes: 8 CPU, 16GB RAM, 200GB SSD
  - Monitoring nodes: 4 CPU, 8GB RAM, 100GB SSD

#### Storage Classes
- **fast-ssd**: Primary storage for applications (SSD, IOPS >= 3000)
- **standard-storage**: Backup and logging storage
- **Monitoring storage**: Prometheus and Grafana data

#### Network Requirements
- **Ingress Controller**: NGINX Ingress Controller
- **Load Balancer**: External load balancer with SSL termination
- **Network Policies**: Calico or similar CNI with policy support
- **DNS**: Internal DNS resolution for service discovery

#### Security Requirements
- **RBAC**: Role-based access control configured
- **Secrets Management**: Encrypted secrets storage
- **TLS Certificates**: Wildcard certificates for domains
- **Network Isolation**: Proper network policies implemented

### Software Requirements

#### Required Tools
```bash
# Kubernetes CLI
kubectl version --client

# Package Manager
helm version

# Container Runtime
docker --version

# Git
git --version

# YAML Processor
yq --version

# Monitoring Tools
# (Optional) Prometheus CLI
```

#### Optional Tools
```bash
# ArgoCD CLI
argocd version

# Monitoring CLI
promtool version

# Security Scanning
trivy version
```

### Access Requirements

#### Kubernetes Permissions
- **Cluster Admin**: For initial setup
- **Namespace Admin**: For application deployment
- **Monitoring Admin**: For monitoring stack management

#### External Services
- **GitHub/GitLab**: Repository access
- **Container Registry**: Docker Hub, ECR, GCR, or similar
- **Certificate Authority**: Let's Encrypt or internal CA
- **Monitoring Service**: Slack, PagerDuty, or similar for alerts

## Deployment Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Internet / DMZ                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
              ┌───────▼───────┐
              │  Ingress      │
              │ Controller    │
              └───────┬───────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                    Kubernetes Cluster                        │
├─────────────────────────────────────────────────────────────┤
│  ran-automation (Application Namespace)                      │
│  ├── ran-automation-api (3 replicas)                        │
│  ├── cognitive-performance-service (2 replicas)             │
│  ├── swarm-coordination-service (2 replicas)                │
│  ├── agentdb-service (1 replica)                            │
│  └── redis-service (1 replica)                              │
├─────────────────────────────────────────────────────────────┤
│  ran-monitoring (Monitoring Namespace)                       │
│  ├── Prometheus (1 replica)                                 │
│  ├── Grafana (1 replica)                                    │
│  ├── AlertManager (1 replica)                               │
│  └── Node Exporter (DaemonSet)                              │
├─────────────────────────────────────────────────────────────┤
│  ran-logging (Logging Namespace)                             │
│  ├── Loki (1 replica)                                       │
│  ├── Promtail (DaemonSet)                                   │
│  └── Fluentd (DaemonSet)                                    │
└─────────────────────────────────────────────────────────────┘
```

### Component Specifications

#### Application Services

| Service | Replicas | CPU Request | CPU Limit | Memory Request | Memory Limit | Storage |
|---------|----------|-------------|-----------|----------------|--------------|---------|
| ran-automation-api | 3 | 500m | 2000m | 1Gi | 4Gi | - |
| cognitive-performance | 2 | 1000m | 4000m | 2Gi | 8Gi | - |
| swarm-coordination | 2 | 500m | 2000m | 1Gi | 4Gi | - |
| agentdb | 1 | 1000m | 4000m | 2Gi | 8Gi | 100Gi |
| redis | 1 | 250m | 1000m | 512Mi | 2Gi | 20Gi |

#### Monitoring Services

| Service | Replicas | CPU Request | CPU Limit | Memory Request | Memory Limit | Storage |
|---------|----------|-------------|-----------|----------------|--------------|---------|
| Prometheus | 1 | 1000m | 4000m | 2Gi | 8Gi | 100Gi |
| Grafana | 1 | 500m | 2000m | 1Gi | 4Gi | 20Gi |
| AlertManager | 1 | 250m | 1000m | 512Mi | 2Gi | 10Gi |

## Three-Phase Rollout Strategy

### Phase 1: Canary Deployment (5-10% traffic)

**Objective**: Validate new version with minimal risk
**Duration**: 1-2 hours
**Success Criteria**:
- All health checks pass
- Error rate < 1%
- Response time < 500ms (95th percentile)
- No critical alerts

**Steps**:
1. Deploy canary version with 1 replica
2. Configure ingress to route 5-10% traffic to canary
3. Monitor metrics and logs for 1-2 hours
4. Run automated validation tests
5. Make go/no-go decision

**Rollback Triggers**:
- Error rate > 1%
- Response time > 500ms
- Health check failures
- Critical alerts triggered

### Phase 2: Partial Deployment (25% traffic)

**Objective**: Gradual traffic increase with continued monitoring
**Duration**: 4-6 hours
**Success Criteria**:
- All health checks pass
- Error rate < 0.5%
- Response time < 300ms (95th percentile)
- Resource utilization < 80%

**Steps**:
1. Scale canary to 2-3 replicas
2. Increase traffic to 25%
3. Monitor extended period (4-6 hours)
4. Validate performance under load
5. Check database and cache performance

**Rollback Triggers**:
- Error rate > 0.5%
- Response time > 300ms
- Resource utilization > 80%
- Database performance degradation

### Phase 3: Full Deployment (100% traffic)

**Objective**: Complete rollout with full monitoring
**Duration**: 24-48 hours
**Success Criteria**:
- All health checks pass
- Error rate < 0.1%
- Response time < 200ms (95th percentile)
- System stability maintained

**Steps**:
1. Scale all services to target replicas
2. Route 100% traffic to new version
3. Monitor for 24-48 hours
4. Validate all integrations
5. Complete deployment documentation

**Rollback Triggers**:
- Error rate > 0.1%
- Response time > 200ms
- System instability
- Integration failures

## Pre-Deployment Checklist

### Infrastructure Readiness

- [ ] Kubernetes cluster is healthy and accessible
- [ ] All nodes are ready and unschedulable status is false
- [ ] Storage classes are properly configured
- [ ] Ingress controller is running and healthy
- [ ] Network policies are tested
- [ ] Load balancer is configured and accessible
- [ ] DNS records are configured and propagating
- [ ] TLS certificates are valid and installed

### Security Configuration

- [ ] RBAC roles and bindings are configured
- [ ] Service accounts are created with minimal permissions
- [ ] Secrets are encrypted and properly stored
- [ ] Network policies are in place
- [ ] Pod security policies are enabled
- [ ] Image security scanning is configured
- [ ] Vulnerability scanning is enabled
- [ ] Security monitoring is active

### Application Configuration

- [ ] Container images are built and pushed to registry
- [ ] Configuration maps are updated and validated
- [ ] Secrets are created and tested
- [ ] Resource limits and requests are set
- [ ] Health checks are configured
- [ ] Liveness and readiness probes are set
- [ ] Environment variables are configured
- [ ] Feature flags are properly set

### Monitoring Setup

- [ ] Prometheus is configured and scraping targets
- [ ] Grafana dashboards are created and tested
- [ ] Alerting rules are configured and tested
- [ ] Notification channels are working
- [ ] Log aggregation is configured
- [ ] Performance monitoring is active
- [ ] Error tracking is enabled
- [ ] Custom metrics are defined

### Backup and Recovery

- [ ] Database backup strategy is configured
- [ ] Application state backup is automated
- [ ] Configuration backup is scheduled
- [ ] Recovery procedures are tested
- [ ] RTO/RPO objectives are defined
- [ ] Disaster recovery plan is documented
- [ ] Backup retention policies are set
- [ ] Restore testing is performed

## Deployment Procedures

### 1. Environment Preparation

```bash
# Clone repository
git clone https://github.com/your-org/ran-automation-agentdb.git
cd ran-automation-agentdb

# Switch to target branch/commit
git checkout main
git pull origin main

# Verify kubectl context
kubectl config current-context
kubectl cluster-info

# Create required directories
mkdir -p logs reports backups
```

### 2. Configuration Validation

```bash
# Validate YAML syntax
find k8s -name "*.yaml" -exec yq eval '.' {} \; > /dev/null

# Validate Kubernetes manifests
kubectl apply --dry-run=client -f k8s/namespaces/
kubectl apply --dry-run=client -f k8s/configmaps/
kubectl apply --dry-run=client -f k8s/secrets/
kubectl apply --dry-run=client -f k8s/deployments/
kubectl apply --dry-run=client -f k8s/services/
```

### 3. Canary Deployment

```bash
# Deploy canary configuration
./scripts/deployment/deploy.sh --phase canary

# Monitor canary health
kubectl get pods -n ran-automation -l phase=canary
kubectl logs -n ran-automation -l phase=canary

# Check application health
curl -H "Host: api.ran-automation.example.com" https://your-ingress-ip/health/live

# Validate metrics
kubectl top pods -n ran-automation
```

### 4. Partial Deployment

```bash
# Scale to partial deployment
./scripts/deployment/deploy.sh --phase partial

# Monitor partial deployment
kubectl get pods -n ran-automation
kubectl get hpa -n ran-automation

# Check traffic distribution
kubectl get ingress -n ran-automation

# Validate performance
./scripts/validation/validate-deployment.sh
```

### 5. Full Deployment

```bash
# Execute full deployment
./scripts/deployment/deploy.sh --phase full

# Monitor all services
kubectl get all -n ran-automation

# Validate full system
./scripts/validation/validate-deployment.sh

# Check monitoring
kubectl get all -n ran-monitoring
```

## Monitoring and Validation

### Health Check Endpoints

| Service | Endpoint | Method | Expected Response |
|---------|----------|--------|------------------|
| API | `/health/live` | GET | 200 OK |
| API | `/health/ready` | GET | 200 OK |
| Cognitive Performance | `/health/live` | GET | 200 OK |
| Swarm Coordination | `/health/live` | GET | 200 OK |
| Database | N/A | TCP Connection | Success |

### Key Metrics

#### Application Metrics
- **Request Rate**: `rate(http_requests_total[5m])`
- **Error Rate**: `rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])`
- **Response Time**: `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))`
- **Active Connections**: `http_connections_active`
- **Queue Length**: `http_queue_length`

#### System Metrics
- **CPU Usage**: `rate(container_cpu_usage_seconds_total[5m])`
- **Memory Usage**: `container_memory_usage_bytes`
- **Disk Usage**: `node_filesystem_avail_bytes`
- **Network I/O**: `rate(container_network_bytes_total[5m])`

#### Business Metrics
- **Cognitive Consciousness Level**: `cognitive_consciousness_level`
- **Temporal Analysis Depth**: `temporal_analysis_depth_factor`
- **Active Agents**: `swarm_active_agents`
- **Task Queue Length**: `swarm_task_queue_length`

### Validation Scripts

```bash
# Run comprehensive validation
./scripts/validation/validate-deployment.sh --namespace ran-automation

# Run smoke tests
./scripts/validation/smoke-tests.sh

# Run performance tests
./scripts/validation/performance-tests.sh

# Run security validation
./scripts/validation/security-validation.sh
```

### Alert Configuration

#### Critical Alerts
- **Service Down**: `up{job="ran-automation-services"} == 0`
- **High Error Rate**: `error_rate > 0.05`
- **High Response Time**: `response_time_p95 > 2s`
- **Database Connection**: `up{job="agentdb"} == 0`

#### Warning Alerts
- **High Memory Usage**: `memory_usage > 0.9`
- **High CPU Usage**: `cpu_usage > 0.8`
- **Disk Space Low**: `disk_available < 10%`
- **Pod Restarts**: `kube_pod_container_status_restarts_total > 5`

## Rollback Procedures

### Immediate Rollback (Emergency)

```bash
# Quick rollback to previous deployment
kubectl rollout undo deployment/ran-automation-api -n ran-automation
kubectl rollout undo deployment/cognitive-performance-service -n ran-automation
kubectl rollout undo deployment/swarm-coordination-service -n ran-automation

# Wait for rollback to complete
kubectl rollout status deployment/ran-automation-api -n ran-automation
kubectl rollout status deployment/cognitive-performance-service -n ran-automation
kubectl rollout status deployment/swarm-coordination-service -n ran-automation
```

### Full Rollback Using Backup

```bash
# Identify latest backup
ls -la backups/ | grep backup- | head -1

# Restore from backup
./scripts/deployment/deploy.sh --rollback

# Validate rollback
./scripts/validation/validate-deployment.sh
```

### ArgoCD Rollback

```bash
# Sync to previous commit
argocd app sync ran-automation-root --revision HEAD~1

# Monitor rollback
argocd app get ran-automation-root
```

### Rollback Validation

```bash
# Check deployment status
kubectl get deployments -n ran-automation
kubectl get pods -n ran-automation

# Validate application health
curl -H "Host: api.ran-automation.example.com" https://your-ingress-ip/health/live

# Check monitoring alerts
kubectl get prometheusrules -n ran-monitoring
```

## Disaster Recovery

### Scenario 1: Node Failure

**Detection**: Node becomes NotReady
**Impact**: Pods on affected node are rescheduled
**Recovery Time**: 5-10 minutes
**Procedure**:
1. Monitor node status: `kubectl get nodes`
2. Check pod rescheduling: `kubectl get pods -o wide`
3. Verify service availability: `kubectl get svc`
4. Monitor automated recovery
5. Escalate if node doesn't recover within 15 minutes

### Scenario 2: Database Failure

**Detection**: Database connection failures
**Impact**: Application becomes unavailable
**Recovery Time**: 15-30 minutes
**Procedure**:
1. Identify database issue: `kubectl logs agentdb-service-...`
2. Check PVC status: `kubectl get pvc`
3. Attempt database restart: `kubectl delete pod agentdb-service-...`
4. If persistent, restore from backup: `kubectl apply -f backups/latest/`
5. Validate data integrity
6. Monitor application recovery

### Scenario 3: Complete Cluster Failure

**Detection**: Cluster inaccessible
**Impact**: Total system outage
**Recovery Time**: 1-4 hours
**Procedure**:
1. Activate disaster recovery site
2. Restore from latest backup
3. Rebuild cluster infrastructure
4. Deploy application stack
5. Validate all services
6. Switch DNS to recovered cluster
7. Monitor system stability

### Backup Strategy

#### Automated Backups
```yaml
# Daily database backup
apiVersion: batch/v1
kind: CronJob
metadata:
  name: database-backup
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:15
            command:
            - pg_dump
            - -h
            - agentdb-service
            - -U
            - postgres
            - ran_automation_prod
            volumeMounts:
            - name: backup-storage
              mountPath: /backups
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
```

#### Manual Backup Commands
```bash
# Create application backup
kubectl get all -n ran-automation -o yaml > backup-manual-$(date +%Y%m%d).yaml

# Database backup
kubectl exec agentdb-service-... -- pg_dump ran_automation_prod > db-backup-$(date +%Y%m%d).sql

# Configuration backup
kubectl get configmaps,secrets -n ran-automation -o yaml > config-backup-$(date +%Y%m%d).yaml
```

## Post-Deployment Activities

### Performance Monitoring

1. **Baseline Establishment**: Record post-deployment performance metrics
2. **Trend Analysis**: Monitor performance trends over 24-48 hours
3. **Capacity Planning**: Review resource utilization and scaling needs
4. **Optimization**: Identify and implement performance improvements

### Security Validation

1. **Vulnerability Scanning**: Run security scans on deployed images
2. **Network Policy Testing**: Verify network isolation rules
3. **Access Control Review**: Audit RBAC configurations
4. **Compliance Check**: Validate security compliance requirements

### Documentation Updates

1. **Runbook Updates**: Update operational procedures
2. **Architecture Documentation**: Reflect any architectural changes
3. **Configuration Changes**: Document all configuration updates
4. **Knowledge Base**: Add lessons learned and best practices

### Training and Handover

1. **Team Training**: Train operations team on new features
2. **Runbook Review**: Review and practice operational procedures
3. **Monitoring Setup**: Configure monitoring dashboards for operations team
4. **Escalation Procedures**: Review and update escalation procedures

## Troubleshooting Guide

### Common Issues

#### Pod Not Starting

**Symptoms**: Pod stuck in Pending or ContainerCreating state
**Causes**: Resource constraints, image pull issues, PVC problems
**Solutions**:
```bash
# Check pod status and events
kubectl describe pod <pod-name> -n ran-automation

# Check resource availability
kubectl top nodes
kubectl describe nodes

# Check image pull issues
kubectl logs <pod-name> -n ran-automation

# Check PVC status
kubectl get pvc -n ran-automation
```

#### Service Not Accessible

**Symptoms**: Service returns connection refused or timeout
**Causes**: Service misconfiguration, network policies, endpoint issues
**Solutions**:
```bash
# Check service configuration
kubectl get svc <service-name> -n ran-automation -o yaml

# Check endpoints
kubectl get endpoints <service-name> -n ran-automation

# Test internal connectivity
kubectl run test-pod --image=busybox --rm -it -- wget -qO- http://<service-name>:<port>

# Check network policies
kubectl get networkpolicies -n ran-automation
```

#### High Memory Usage

**Symptoms**: OOMKilled events, memory pressure
**Causes**: Memory leaks, insufficient resources, traffic spikes
**Solutions**:
```bash
# Check memory usage
kubectl top pods -n ran-automation --sort-by=memory

# Check OOM events
kubectl describe pod <pod-name> -n ran-automation | grep -i oom

# Increase memory limits
kubectl patch deployment <deployment-name> -p '{"spec":{"template":{"spec":{"containers":[{"name":"<container-name>","resources":{"limits":{"memory":"<new-limit>"}}}]}}}}'

# Check for memory leaks
kubectl exec <pod-name> -- top
```

#### Database Connection Issues

**Symptoms**: Application cannot connect to database
**Causes**: Database down, network issues, authentication problems
**Solutions**:
```bash
# Check database pod
kubectl get pods -n ran-automation -l app=agentdb

# Check database logs
kubectl logs agentdb-service-... -n ran-automation

# Test database connectivity
kubectl run db-test --image=postgres:15 --rm -it -- psql -h agentdb-service -U postgres -d ran_automation_prod

# Check database configuration
kubectl get configmap agentdb-config -n ran-automation -o yaml
```

### Debug Commands

```bash
# Comprehensive system status
kubectl get all -n ran-automation
kubectl get events -n ran-automation --sort-by='.lastTimestamp'

# Resource utilization
kubectl top pods -n ran-automation
kubectl top nodes

# Network debugging
kubectl run network-debug --image=nicolaka/netshoot --rm -it -- nslookup ran-automation-api-service.ran-automation.svc.cluster.local

# Application logs
kubectl logs -f deployment/ran-automation-api -n ran-automation
kubectl logs -f deployment/cognitive-performance-service -n ran-automation

# Configuration validation
kubectl get configmaps -n ran-automation -o yaml
kubectl get secrets -n ran-automation -o yaml
```

### Escalation Procedures

#### Level 1: Operations Team (0-30 minutes)
- Initial troubleshooting using runbook
- Check monitoring dashboards
- Verify system status
- Document findings

#### Level 2: Engineering Team (30-60 minutes)
- Deep dive investigation
- Code analysis if needed
- System architecture review
- Implement temporary fixes

#### Level 3: Architecture Team (60+ minutes)
- System-wide impact assessment
- Architecture review
- Long-term solutions
- Incident post-mortem

---

## Conclusion

This deployment playbook provides comprehensive guidance for deploying the RAN Intelligent Automation System to production environments. Following the procedures and guidelines outlined here will ensure a smooth, secure, and reliable deployment process.

Remember to:
- Always validate changes in non-production environments first
- Monitor all aspects of the system during and after deployment
- Maintain up-to-date documentation and runbooks
- Practice rollback and disaster recovery procedures regularly
- Continuously improve processes based on lessons learned

For additional support or questions, refer to the project documentation or contact the engineering team.