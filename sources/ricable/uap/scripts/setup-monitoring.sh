#!/bin/bash
# scripts/setup-monitoring.sh
# UAP Monitoring and Observability Setup Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO]${NC} $@"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $@"; }
error() { echo -e "${RED}[ERROR]${NC} $@"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $@"; }

# Create monitoring directory structure
setup_directories() {
    info "Setting up monitoring directories..."
    
    mkdir -p "$PROJECT_ROOT/monitoring"/{prometheus,grafana,nginx}
    mkdir -p "$PROJECT_ROOT/monitoring/grafana"/{dashboards,datasources}
    
    success "Monitoring directories created"
}

# Create Prometheus configuration
create_prometheus_config() {
    info "Creating Prometheus configuration..."
    
    cat > "$PROJECT_ROOT/monitoring/prometheus.yml" << 'EOF'
# Prometheus configuration for UAP monitoring
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'uap-production'
    environment: 'production'

rule_files:
  - "alerts.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # UAP Backend API metrics
  - job_name: 'uap-backend'
    static_configs:
      - targets: ['uap-backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
    scrape_timeout: 5s

  # Node Exporter for system metrics
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  # cAdvisor for container metrics
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  # Redis metrics
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  # PostgreSQL metrics
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  # NGINX metrics
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']

  # Prometheus self-monitoring
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF

    success "Prometheus configuration created"
}

# Create Grafana datasource configuration
create_grafana_datasources() {
    info "Creating Grafana datasources..."
    
    cat > "$PROJECT_ROOT/monitoring/grafana/datasources/prometheus.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
    jsonData:
      timeInterval: "5s"
EOF

    success "Grafana datasources configured"
}

# Create UAP-specific Grafana dashboard
create_uap_dashboard() {
    info "Creating UAP Grafana dashboard..."
    
    cat > "$PROJECT_ROOT/monitoring/grafana/dashboards/uap-overview.json" << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "UAP - Agent Orchestration Platform",
    "tags": ["uap", "agents", "production"],
    "timezone": "browser",
    "refresh": "30s",
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "panels": [
      {
        "id": 1,
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ],
        "yAxes": [
          {
            "label": "Seconds",
            "min": 0
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 3,
        "title": "Active WebSocket Connections",
        "type": "singlestat",
        "targets": [
          {
            "expr": "websocket_connections_active",
            "legendFormat": "Active Connections"
          }
        ],
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8}
      },
      {
        "id": 4,
        "title": "Agent Framework Health",
        "type": "table",
        "targets": [
          {
            "expr": "agent_framework_status",
            "legendFormat": "{{framework}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 6, "y": 8}
      },
      {
        "id": 5,
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "100 - (avg by (instance) (rate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "CPU Usage %"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16}
      },
      {
        "id": 6,
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100",
            "legendFormat": "Memory Usage %"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16}
      }
    ]
  }
}
EOF

    success "UAP dashboard created"
}

# Create alerting rules
create_alert_rules() {
    info "Creating Prometheus alert rules..."
    
    cat > "$PROJECT_ROOT/monitoring/alerts.yml" << 'EOF'
groups:
  - name: uap.rules
    rules:
      # High response time alert
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }}s"

      # High error rate alert  
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"

      # WebSocket connection issues
      - alert: WebSocketConnectionDrop
        expr: increase(websocket_connections_dropped_total[5m]) > 10
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "WebSocket connections dropping"
          description: "{{ $value }} connections dropped in last 5 minutes"

      # Agent framework down
      - alert: AgentFrameworkDown
        expr: agent_framework_status == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "Agent framework is down"
          description: "Framework {{ $labels.framework }} is not responding"

      # High CPU usage
      - alert: HighCPUUsage
        expr: 100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage"
          description: "CPU usage is {{ $value }}% on {{ $labels.instance }}"

      # High memory usage
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100 > 85
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}% on {{ $labels.instance }}"

      # Disk space low
      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100 < 10
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Disk space low"
          description: "Disk space is {{ $value }}% on {{ $labels.instance }}"
EOF

    success "Alert rules created"
}

# Create NGINX configuration for monitoring
create_nginx_config() {
    info "Creating NGINX configuration..."
    
    cat > "$PROJECT_ROOT/monitoring/nginx/nginx.conf" << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream uap_backend {
        server uap-backend:8000;
    }

    # Enable stub_status for monitoring
    server {
        listen 80;
        server_name _;

        # Serve frontend
        location / {
            root /usr/share/nginx/html;
            try_files $uri $uri/ /index.html;
        }

        # Proxy API requests
        location /api/ {
            proxy_pass http://uap_backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # WebSocket proxy
        location /ws/ {
            proxy_pass http://uap_backend/ws/;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        # Health check endpoint
        location /health {
            proxy_pass http://uap_backend/health;
        }

        # NGINX status for monitoring
        location /nginx_status {
            stub_status on;
            access_log off;
            allow 127.0.0.1;
            allow 172.20.0.0/16;  # Docker network
            deny all;
        }
    }

    # Log format for monitoring
    log_format json_combined escape=json
        '{'
        '"time_local":"$time_local",'
        '"remote_addr":"$remote_addr",'
        '"remote_user":"$remote_user",'
        '"request":"$request",'
        '"status": "$status",'
        '"body_bytes_sent":"$body_bytes_sent",'
        '"request_time":"$request_time",'
        '"http_referrer":"$http_referer",'
        '"http_user_agent":"$http_user_agent"'
        '}';

    access_log /var/log/nginx/access.log json_combined;
    error_log /var/log/nginx/error.log;
}
EOF

    success "NGINX configuration created"
}

# Create monitoring Docker Compose
create_monitoring_compose() {
    info "Creating monitoring Docker Compose configuration..."
    
    cat > "$PROJECT_ROOT/docker-compose.monitoring.yml" << 'EOF'
version: '3.8'

services:
  # Node Exporter for system metrics
  node-exporter:
    image: prom/node-exporter:latest
    container_name: uap-node-exporter
    restart: unless-stopped
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.ignored-mount-points=^/(sys|proc|dev|host|etc)($$|/)'
    ports:
      - "9100:9100"
    networks:
      - uap-network

  # cAdvisor for container metrics
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    container_name: uap-cadvisor
    restart: unless-stopped
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:rw
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
    ports:
      - "8081:8080"
    networks:
      - uap-network

  # Redis Exporter
  redis-exporter:
    image: oliver006/redis_exporter:latest
    container_name: uap-redis-exporter
    restart: unless-stopped
    environment:
      REDIS_ADDR: "redis:6379"
      REDIS_PASSWORD: "${REDIS_PASSWORD}"
    ports:
      - "9121:9121"
    networks:
      - uap-network
    depends_on:
      - redis

  # PostgreSQL Exporter
  postgres-exporter:
    image: prometheuscommunity/postgres-exporter:latest
    container_name: uap-postgres-exporter
    restart: unless-stopped
    environment:
      DATA_SOURCE_NAME: "${DATABASE_URL}"
    ports:
      - "9187:9187"
    networks:
      - uap-network
    depends_on:
      - postgres

  # NGINX Exporter
  nginx-exporter:
    image: nginx/nginx-prometheus-exporter:latest
    container_name: uap-nginx-exporter
    restart: unless-stopped
    command:
      - '-nginx.scrape-uri=http://nginx/nginx_status'
    ports:
      - "9113:9113"
    networks:
      - uap-network
    depends_on:
      - nginx

networks:
  uap-network:
    external: true
    name: uap-production-network
EOF

    success "Monitoring Docker Compose created"
}

# Create health check script
create_health_check_script() {
    info "Creating comprehensive health check script..."
    
    cat > "$PROJECT_ROOT/scripts/health-check.sh" << 'EOF'
#!/bin/bash
# Comprehensive UAP health check script

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $@"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $@"; }
error() { echo -e "${RED}[ERROR]${NC} $@"; }

BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3001}"

# Check backend health
check_backend() {
    info "Checking backend health..."
    
    if curl -f -s "$BACKEND_URL/health" > /dev/null; then
        info "✓ Backend is healthy"
        return 0
    else
        error "✗ Backend health check failed"
        return 1
    fi
}

# Check agent frameworks
check_frameworks() {
    info "Checking agent frameworks..."
    
    local response=$(curl -s "$BACKEND_URL/agents/status" || echo "{}")
    
    if echo "$response" | jq -e '.copilot.status == "active"' > /dev/null 2>&1; then
        info "✓ CopilotKit framework is active"
    else
        warn "⚠ CopilotKit framework status unknown"
    fi
    
    if echo "$response" | jq -e '.agno.status == "active"' > /dev/null 2>&1; then
        info "✓ Agno framework is active"
    else
        warn "⚠ Agno framework status unknown"
    fi
    
    if echo "$response" | jq -e '.mastra.status == "active"' > /dev/null 2>&1; then
        info "✓ Mastra framework is active"
    else
        warn "⚠ Mastra framework status unknown"
    fi
}

# Check monitoring services
check_monitoring() {
    info "Checking monitoring services..."
    
    if curl -f -s "$PROMETHEUS_URL/-/healthy" > /dev/null; then
        info "✓ Prometheus is healthy"
    else
        warn "⚠ Prometheus not accessible"
    fi
    
    if curl -f -s "$GRAFANA_URL/api/health" > /dev/null; then
        info "✓ Grafana is healthy"
    else
        warn "⚠ Grafana not accessible"
    fi
}

# Check database connectivity
check_database() {
    info "Checking database connectivity..."
    
    local db_status=$(curl -s "$BACKEND_URL/health" | jq -r '.database // "unknown"')
    
    if [[ "$db_status" == "healthy" ]]; then
        info "✓ Database is healthy"
    else
        error "✗ Database health check failed: $db_status"
        return 1
    fi
}

# Main health check
main() {
    info "Starting UAP health check..."
    
    local failed=0
    
    check_backend || failed=$((failed + 1))
    check_frameworks
    check_monitoring
    check_database || failed=$((failed + 1))
    
    if [[ $failed -eq 0 ]]; then
        info "✓ All health checks passed"
        exit 0
    else
        error "✗ $failed critical health checks failed"
        exit 1
    fi
}

main "$@"
EOF

    chmod +x "$PROJECT_ROOT/scripts/health-check.sh"
    success "Health check script created"
}

# Main setup function
main() {
    info "Setting up UAP monitoring and observability..."
    
    setup_directories
    create_prometheus_config
    create_grafana_datasources
    create_uap_dashboard
    create_alert_rules
    create_nginx_config
    create_monitoring_compose
    create_health_check_script
    
    success "UAP monitoring setup completed!"
    info ""
    info "Next steps:"
    info "1. Start monitoring services: docker-compose -f docker-compose.monitoring.yml up -d"
    info "2. Access Grafana at: http://localhost:3001 (admin/admin)"
    info "3. Access Prometheus at: http://localhost:9090"
    info "4. Run health checks: ./scripts/health-check.sh"
}

main "$@"