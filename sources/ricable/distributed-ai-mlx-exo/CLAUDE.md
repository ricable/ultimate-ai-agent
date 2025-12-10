# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a distributed AI/ML inference system built for Apple Silicon clusters, combining MLX (Apple's machine learning framework) with Exo's peer-to-peer inference capabilities. The system enables running large language models (70B+ parameters) across multiple Mac devices with enterprise-grade monitoring, reliability, and production deployment capabilities.

## Build and Development Commands

### Core System Commands
```bash
# Production deployment
./scripts/deploy_production.sh deploy          # Full cluster deployment
./scripts/deploy_production.sh status          # Check cluster health
./scripts/deploy_production.sh update          # Rolling updates
./scripts/deploy_production.sh validate        # Validate deployment

# Development and testing
python -m pytest tests/test_system_integration.py    # Full system tests
python -m pytest tests/test_phase3.py               # API gateway tests
python -m pytest tests/test_phase1_integration.py   # Foundation tests

# Start monitoring stack
docker-compose up -d                          # Start all monitoring services
docker-compose down                           # Stop monitoring services
docker-compose logs -f grafana               # View Grafana logs

# Manual service testing
python src/enhanced_api_server.py            # Start API server manually
python -m src.monitoring.health_monitor      # Start health monitoring
python -m src.monitoring.prometheus_metrics  # Start metrics collection
```

### Claude-Flow Integration Commands
```bash
./claude-flow --help                         # Show all available commands
./claude-flow start --ui --port 3000        # Start orchestration with web UI
./claude-flow status                         # Show system status
./claude-flow sparc "task description"      # Run orchestrator mode
./claude-flow swarm "objective" --parallel  # Multi-agent coordination
```

## High-Level Architecture

### System Components
The system follows a 5-phase implementation approach:

1. **Phase 1 (Foundation)**: Environment setup, MLX distributed configuration, Exo P2P cluster formation
2. **Phase 2 (Core Integration)**: MLX-Exo bridge, distributed inference engine, model partitioning
3. **Phase 3 (API Gateway)**: FastAPI server, load balancing, authentication, rate limiting
4. **Phase 4 (Performance)**: Network/memory/compute optimization, profiling tools
5. **Phase 5 (Monitoring)**: Health monitoring, Prometheus metrics, Grafana dashboards, production deployment

### Core Architecture Patterns

**Distributed Inference Flow:**
```
API Gateway → Load Balancer → Node Selection → Model Partitioning → Inference Execution → Response Aggregation
```

**Monitoring Stack:**
```
Application Metrics → Prometheus → Grafana Dashboards
Health Monitors → AlertManager → Notification System
Logs → Loki → Grafana Log Explorer
```

**Node Communication:**
- Primary: 10GbE Ethernet for data transfer
- Secondary: Thunderbolt 4 ring for low-latency coordination
- Service Discovery: Exo's P2P auto-discovery with mDNS fallback

### Key Integration Points

**MLX + Exo Bridge** (`src/exo_integration/enhanced_cluster_manager.py`):
- Handles model partitioning across cluster nodes based on memory capacity
- Manages distributed loading of model shards
- Coordinates inference execution across nodes
- Provides failover and recovery mechanisms

**Distributed Memory Management** (`src/memory_manager.py`):
- Ring memory-weighted partitioning strategy
- Tiered caching (L1: GPU memory, L2: unified memory, L3: NVMe, L4: network)
- Dynamic memory allocation based on model requirements

**API Gateway Architecture** (`src/enhanced_api_server.py`):
- OpenAI-compatible REST API endpoints
- Integrated load balancing with health-aware routing
- JWT-based authentication and rate limiting
- Streaming response support for real-time inference

**Health Monitoring System** (`src/monitoring/health_monitor.py`):
- Multi-component health checks (nodes, services, network, resources)
- Automatic failover within 30 seconds of node failure
- Configurable health thresholds and alert callbacks
- Historical health data retention and analysis

## Configuration Management

### Primary Configuration Files
- `config/cluster_nodes.json`: Cluster topology and node specifications
- `config/api_gateway.yaml`: API server configuration
- `config/grafana/dashboards/`: Monitoring dashboard definitions
- `docker-compose.yml`: Complete monitoring stack definition

### Environment-Specific Settings
Production deployment uses centralized configuration with SSH-based distribution. Development mode supports hot-reload and debug logging.

## Development Workflow

### Testing Strategy
The system includes comprehensive testing at multiple levels:
- **Unit Tests**: Component-level validation
- **Integration Tests**: Cross-component workflow validation  
- **System Tests**: End-to-end cluster validation
- **Load Tests**: Performance and scalability validation
- **Chaos Tests**: Fault tolerance and recovery validation

### Performance Benchmarks
Target performance metrics enforced by automated testing:
- 70B models: >10 tokens/second
- API latency: <100ms time-to-first-token
- Model loading: <120 seconds
- System availability: >99.9% uptime
- Error rate: <1%

### Monitoring and Observability

**Access Points:**
- Grafana Dashboards: http://localhost:3000 (admin/mlx-cluster-admin)
- Prometheus Metrics: http://localhost:9091
- AlertManager: http://localhost:9093
- API Health Endpoint: http://localhost:52415/health

**Key Metrics Categories:**
- API Performance: Request rates, latency distributions, error rates
- Inference Metrics: Token generation rates, model performance, queue depths
- System Resources: CPU, memory, disk, network utilization per node
- Cluster Health: Node status, failover events, service availability

## Deployment Architecture

### Production Deployment
The deployment script (`scripts/deploy_production.sh`) handles:
- SSH connectivity validation across all cluster nodes
- Prerequisites checking and dependency installation
- Code distribution and service configuration
- Rolling updates with zero-downtime deployment
- Health validation and rollback capabilities

### Node Specifications
Default cluster configuration supports:
- 2x Mac Studio M1 Max (64GB unified memory, 32-core GPU)
- 1x Mac Studio M2 Max (32GB unified memory, 30-core GPU)  
- Total: 160GB combined memory, heterogeneous compute optimization

### Service Management
Services are managed via systemd on each node:
- `mlx-api-server.service`: Main API gateway service
- `mlx-health-monitor.service`: Node health monitoring
- Automatic restart and failure recovery
- Centralized log aggregation via journald

## Important Implementation Notes

### MLX Distributed Limitations
- Recent MLX versions have ~15% performance regression in distributed mode
- Requires specific version pinning: MLX >=0.22.1, MLX-LM >=0.21.1
- MPI backend requires proper SSH key distribution for inter-node communication

### Exo Integration Constraints  
- Adds latency penalty for single-request inference (optimized for batch processing)
- Network discovery can fail due to mDNSResponder issues (fallback mechanisms implemented)
- Ring memory-weighted partitioning requires manual configuration for optimal performance

### Network Requirements
- 10GbE infrastructure required for adequate bandwidth (minimum 1GbE acceptable for testing)
- Jumbo frames (MTU 9000) recommended for large model transfers
- Firewall configuration for ports: 52415 (API), 8000 (metrics), 3000 (Grafana), 9090-9093 (monitoring)

### Security Considerations
- SSH key-based authentication for cluster communication
- JWT tokens for API authentication
- TLS encryption for external API access
- No sensitive data logging or storage in cluster components

This architecture enables distributed inference of large language models across Apple Silicon hardware while maintaining enterprise-grade reliability, monitoring, and operational capabilities.