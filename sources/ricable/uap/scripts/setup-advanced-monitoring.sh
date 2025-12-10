#!/bin/bash

# UAP Advanced Monitoring Setup Script
# Sets up comprehensive monitoring infrastructure with Prometheus, Grafana, Jaeger, and alerting

set -e  # Exit on any error

# Configuration
MONITORING_DIR="/opt/uap-monitoring"
PROMETHEUS_VERSION="2.45.0"
GRAFANA_VERSION="10.0.0"
JAEGER_VERSION="1.49.0"
ALERTMANAGER_VERSION="0.25.0"
NODE_EXPORTER_VERSION="1.6.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

check_requirements() {
    log_info "Checking system requirements..."
    
    # Check if running as root or with sudo
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root or with sudo"
        exit 1
    fi
    
    # Check for required commands
    local required_commands=("curl" "tar" "systemctl" "docker" "docker-compose")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Required command '$cmd' is not installed"
            exit 1
        fi
    done
    
    # Check available disk space (need at least 5GB)
    local available_space=$(df / | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 5242880 ]]; then  # 5GB in KB
        log_warning "Low disk space detected. Monitoring setup may fail."
    fi
    
    log_success "System requirements check passed"
}

create_monitoring_user() {
    log_info "Creating monitoring system user..."
    
    if ! id "monitoring" &>/dev/null; then
        useradd --system --shell /bin/false --home-dir /var/lib/monitoring monitoring
        log_success "Created monitoring user"
    else
        log_info "Monitoring user already exists"
    fi
}

setup_directories() {
    log_info "Setting up monitoring directories..."
    
    # Create base monitoring directory
    mkdir -p "$MONITORING_DIR"
    mkdir -p "$MONITORING_DIR/prometheus"
    mkdir -p "$MONITORING_DIR/grafana"
    mkdir -p "$MONITORING_DIR/jaeger"
    mkdir -p "$MONITORING_DIR/alertmanager"
    mkdir -p "$MONITORING_DIR/exporters"
    mkdir -p "$MONITORING_DIR/data"
    mkdir -p "$MONITORING_DIR/logs"
    
    # Create data directories
    mkdir -p "/var/lib/monitoring/prometheus"
    mkdir -p "/var/lib/monitoring/grafana"
    mkdir -p "/var/lib/monitoring/jaeger"
    mkdir -p "/var/lib/monitoring/alertmanager"
    
    # Set permissions
    chown -R monitoring:monitoring "/var/lib/monitoring"
    chown -R monitoring:monitoring "$MONITORING_DIR"
    
    log_success "Monitoring directories created"
}

install_prometheus() {
    log_info "Installing Prometheus $PROMETHEUS_VERSION..."
    
    cd /tmp
    curl -LO "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz"
    tar xzf "prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz"
    
    # Install binaries
    cp "prometheus-${PROMETHEUS_VERSION}.linux-amd64/prometheus" /usr/local/bin/
    cp "prometheus-${PROMETHEUS_VERSION}.linux-amd64/promtool" /usr/local/bin/
    
    # Set permissions
    chown monitoring:monitoring /usr/local/bin/prometheus
    chown monitoring:monitoring /usr/local/bin/promtool
    chmod 755 /usr/local/bin/prometheus
    chmod 755 /usr/local/bin/promtool
    
    # Copy configuration from repository
    cp "$(dirname "$0")/../monitoring/prometheus/prometheus.yml" "$MONITORING_DIR/prometheus/"
    cp -r "$(dirname "$0")/../monitoring/prometheus/alert_rules" "$MONITORING_DIR/prometheus/"
    
    # Create systemd service
    cat > /etc/systemd/system/prometheus.service << EOF
[Unit]
Description=Prometheus
Wants=network-online.target
After=network-online.target

[Service]
User=monitoring
Group=monitoring
Type=simple
ExecStart=/usr/local/bin/prometheus \\
    --config.file=$MONITORING_DIR/prometheus/prometheus.yml \\
    --storage.tsdb.path=/var/lib/monitoring/prometheus/ \\
    --web.console.templates=/usr/local/share/prometheus/consoles \\
    --web.console.libraries=/usr/local/share/prometheus/console_libraries \\
    --web.listen-address=0.0.0.0:9090 \\
    --web.external-url= \\
    --storage.tsdb.retention.time=30d \\
    --storage.tsdb.retention.size=50GB

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    systemctl enable prometheus
    
    log_success "Prometheus installed and configured"
}

install_grafana() {
    log_info "Installing Grafana $GRAFANA_VERSION..."
    
    # Add Grafana repository
    curl -fsSL https://packages.grafana.com/gpg.key | apt-key add -
    echo "deb https://packages.grafana.com/oss/deb stable main" | tee -a /etc/apt/sources.list.d/grafana.list
    
    # Update and install
    apt-get update
    apt-get install -y grafana
    
    # Configure Grafana
    cat > /etc/grafana/grafana.ini << EOF
[server]
protocol = http
http_addr = 0.0.0.0
http_port = 3001
root_url = http://localhost:3001/

[database]
type = sqlite3
path = /var/lib/grafana/grafana.db

[session]
provider = file
provider_config = sessions

[analytics]
reporting_enabled = false
check_for_updates = false

[security]
admin_user = admin
admin_password = admin123!
disable_gravatar = true

[users]
allow_sign_up = false
allow_org_create = false
auto_assign_org = true
auto_assign_org_role = Viewer

[auth.anonymous]
enabled = false

[log]
mode = file
level = info

[paths]
data = /var/lib/grafana
logs = /var/log/grafana
plugins = /var/lib/grafana/plugins
provisioning = /etc/grafana/provisioning
EOF
    
    # Copy dashboard configuration
    mkdir -p /etc/grafana/provisioning/dashboards
    mkdir -p /etc/grafana/provisioning/datasources
    
    # Configure Prometheus datasource
    cat > /etc/grafana/provisioning/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9090
    isDefault: true
    editable: true
EOF
    
    # Configure dashboard provisioning
    cat > /etc/grafana/provisioning/dashboards/uap-dashboards.yml << EOF
apiVersion: 1

providers:
  - name: 'UAP Dashboards'
    orgId: 1
    folder: 'UAP'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF
    
    # Copy dashboard files
    mkdir -p /var/lib/grafana/dashboards
    cp "$(dirname "$0")/../monitoring/grafana/dashboards/"*.json /var/lib/grafana/dashboards/
    
    # Set permissions
    chown -R grafana:grafana /var/lib/grafana
    chown -R grafana:grafana /etc/grafana
    
    systemctl enable grafana-server
    
    log_success "Grafana installed and configured"
}

install_jaeger() {
    log_info "Installing Jaeger $JAEGER_VERSION..."
    
    cd /tmp
    curl -LO "https://github.com/jaegertracing/jaeger/releases/download/v${JAEGER_VERSION}/jaeger-${JAEGER_VERSION}-linux-amd64.tar.gz"
    tar xzf "jaeger-${JAEGER_VERSION}-linux-amd64.tar.gz"
    
    # Install binaries
    cp "jaeger-${JAEGER_VERSION}-linux-amd64/jaeger-all-in-one" /usr/local/bin/
    cp "jaeger-${JAEGER_VERSION}-linux-amd64/jaeger-agent" /usr/local/bin/
    cp "jaeger-${JAEGER_VERSION}-linux-amd64/jaeger-collector" /usr/local/bin/
    cp "jaeger-${JAEGER_VERSION}-linux-amd64/jaeger-query" /usr/local/bin/
    
    # Set permissions
    chown monitoring:monitoring /usr/local/bin/jaeger-*
    chmod 755 /usr/local/bin/jaeger-*
    
    # Create systemd service for Jaeger all-in-one
    cat > /etc/systemd/system/jaeger.service << EOF
[Unit]
Description=Jaeger Tracing
Wants=network-online.target
After=network-online.target

[Service]
User=monitoring
Group=monitoring
Type=simple
ExecStart=/usr/local/bin/jaeger-all-in-one \\
    --memory.max-traces=10000 \\
    --query.base-path=/ \\
    --admin.http.host-port=:14269

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    systemctl enable jaeger
    
    log_success "Jaeger installed and configured"
}

install_node_exporter() {
    log_info "Installing Node Exporter $NODE_EXPORTER_VERSION..."
    
    cd /tmp
    curl -LO "https://github.com/prometheus/node_exporter/releases/download/v${NODE_EXPORTER_VERSION}/node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz"
    tar xzf "node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz"
    
    # Install binary
    cp "node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64/node_exporter" /usr/local/bin/
    
    # Set permissions
    chown monitoring:monitoring /usr/local/bin/node_exporter
    chmod 755 /usr/local/bin/node_exporter
    
    # Create systemd service
    cat > /etc/systemd/system/node_exporter.service << EOF
[Unit]
Description=Node Exporter
Wants=network-online.target
After=network-online.target

[Service]
User=monitoring
Group=monitoring
Type=simple
ExecStart=/usr/local/bin/node_exporter

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    systemctl enable node_exporter
    
    log_success "Node Exporter installed and configured"
}

install_alertmanager() {
    log_info "Installing Alertmanager $ALERTMANAGER_VERSION..."
    
    cd /tmp
    curl -LO "https://github.com/prometheus/alertmanager/releases/download/v${ALERTMANAGER_VERSION}/alertmanager-${ALERTMANAGER_VERSION}.linux-amd64.tar.gz"
    tar xzf "alertmanager-${ALERTMANAGER_VERSION}.linux-amd64.tar.gz"
    
    # Install binaries
    cp "alertmanager-${ALERTMANAGER_VERSION}.linux-amd64/alertmanager" /usr/local/bin/
    cp "alertmanager-${ALERTMANAGER_VERSION}.linux-amd64/amtool" /usr/local/bin/
    
    # Set permissions
    chown monitoring:monitoring /usr/local/bin/alertmanager
    chown monitoring:monitoring /usr/local/bin/amtool
    chmod 755 /usr/local/bin/alertmanager
    chmod 755 /usr/local/bin/amtool
    
    # Create configuration
    cat > "$MONITORING_DIR/alertmanager/alertmanager.yml" << EOF
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alertmanager@uap.local'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'http://localhost:8000/api/monitoring/webhooks/alerts'
        send_resolved: true

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']
EOF
    
    # Create systemd service
    cat > /etc/systemd/system/alertmanager.service << EOF
[Unit]
Description=Alertmanager
Wants=network-online.target
After=network-online.target

[Service]
User=monitoring
Group=monitoring
Type=simple
ExecStart=/usr/local/bin/alertmanager \\
    --config.file=$MONITORING_DIR/alertmanager/alertmanager.yml \\
    --storage.path=/var/lib/monitoring/alertmanager/ \\
    --web.listen-address=0.0.0.0:9093

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    systemctl enable alertmanager
    
    log_success "Alertmanager installed and configured"
}

setup_nginx_proxy() {
    log_info "Setting up Nginx reverse proxy for monitoring services..."
    
    # Install nginx if not present
    if ! command -v nginx &> /dev/null; then
        apt-get update
        apt-get install -y nginx
    fi
    
    # Create monitoring site configuration
    cat > /etc/nginx/sites-available/uap-monitoring << EOF
server {
    listen 80;
    server_name monitoring.uap.local localhost;
    
    # Grafana
    location /grafana/ {
        proxy_pass http://localhost:3001/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # Prometheus
    location /prometheus/ {
        proxy_pass http://localhost:9090/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # Jaeger
    location /jaeger/ {
        proxy_pass http://localhost:16686/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # Alertmanager
    location /alertmanager/ {
        proxy_pass http://localhost:9093/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # UAP Backend API
    location /api/ {
        proxy_pass http://localhost:8000/api/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # Default monitoring dashboard
    location / {
        return 301 /grafana/;
    }
}
EOF
    
    # Enable site
    ln -sf /etc/nginx/sites-available/uap-monitoring /etc/nginx/sites-enabled/
    
    # Test nginx configuration
    nginx -t
    
    # Reload nginx
    systemctl reload nginx
    
    log_success "Nginx reverse proxy configured"
}

install_python_dependencies() {
    log_info "Installing Python monitoring dependencies..."
    
    # Install Python packages for monitoring integration
    pip install -r "$(dirname "$0")/../backend/requirements.txt"
    
    # Install additional monitoring packages
    pip install opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi
    pip install opentelemetry-exporter-jaeger opentelemetry-exporter-otlp
    pip install prometheus-client psutil aiofiles
    
    log_success "Python dependencies installed"
}

start_monitoring_services() {
    log_info "Starting monitoring services..."
    
    # Start services in order
    local services=("node_exporter" "prometheus" "alertmanager" "jaeger" "grafana-server")
    
    for service in "${services[@]}"; do
        log_info "Starting $service..."
        systemctl start "$service"
        
        # Wait for service to be ready
        sleep 2
        
        if systemctl is-active --quiet "$service"; then
            log_success "$service started successfully"
        else
            log_error "Failed to start $service"
            systemctl status "$service"
        fi
    done
    
    log_success "All monitoring services started"
}

verify_installation() {
    log_info "Verifying monitoring installation..."
    
    # Check service status
    local services=("node_exporter" "prometheus" "alertmanager" "jaeger" "grafana-server")
    local failed_services=()
    
    for service in "${services[@]}"; do
        if ! systemctl is-active --quiet "$service"; then
            failed_services+=("$service")
        fi
    done
    
    if [[ ${#failed_services[@]} -eq 0 ]]; then
        log_success "All monitoring services are running"
    else
        log_error "The following services failed to start: ${failed_services[*]}"
        return 1
    fi
    
    # Check endpoint availability
    local endpoints=(
        "http://localhost:9090/-/ready"  # Prometheus
        "http://localhost:3001/api/health"  # Grafana
        "http://localhost:16686/"  # Jaeger
        "http://localhost:9093/-/ready"  # Alertmanager
        "http://localhost:9100/metrics"  # Node Exporter
    )
    
    for endpoint in "${endpoints[@]}"; do
        if curl -s "$endpoint" > /dev/null; then
            log_success "Endpoint $endpoint is accessible"
        else
            log_warning "Endpoint $endpoint is not accessible"
        fi
    done
    
    log_success "Installation verification completed"
}

print_access_info() {
    log_info "\n=== UAP Advanced Monitoring Setup Complete ==="
    echo
    echo "Access your monitoring tools at:"
    echo "  • Grafana:       http://localhost/grafana/ (admin/admin123!)"
    echo "  • Prometheus:    http://localhost/prometheus/"
    echo "  • Jaeger:        http://localhost/jaeger/"
    echo "  • Alertmanager:  http://localhost/alertmanager/"
    echo "  • UAP API:       http://localhost/api/monitoring/"
    echo
    echo "Service endpoints:"
    echo "  • Grafana:       http://localhost:3001"
    echo "  • Prometheus:    http://localhost:9090"
    echo "  • Jaeger:        http://localhost:16686"
    echo "  • Alertmanager:  http://localhost:9093"
    echo "  • Node Exporter: http://localhost:9100"
    echo
    echo "Configuration files:"
    echo "  • Prometheus:    $MONITORING_DIR/prometheus/prometheus.yml"
    echo "  • Grafana:       /etc/grafana/grafana.ini"
    echo "  • Alertmanager:  $MONITORING_DIR/alertmanager/alertmanager.yml"
    echo
    echo "Log files:"
    echo "  • Prometheus:    journalctl -u prometheus"
    echo "  • Grafana:       journalctl -u grafana-server"
    echo "  • Jaeger:        journalctl -u jaeger"
    echo "  • Alertmanager:  journalctl -u alertmanager"
    echo
    log_success "Monitoring infrastructure is ready!"
}

cleanup_temp_files() {
    log_info "Cleaning up temporary files..."
    
    cd /tmp
    rm -f prometheus-*.tar.gz
    rm -f jaeger-*.tar.gz
    rm -f node_exporter-*.tar.gz
    rm -f alertmanager-*.tar.gz
    rm -rf prometheus-*
    rm -rf jaeger-*
    rm -rf node_exporter-*
    rm -rf alertmanager-*
    
    log_success "Temporary files cleaned up"
}

main() {
    log_info "Starting UAP Advanced Monitoring Setup..."
    
    # Run setup steps
    check_requirements
    create_monitoring_user
    setup_directories
    install_prometheus
    install_grafana
    install_jaeger
    install_node_exporter
    install_alertmanager
    setup_nginx_proxy
    install_python_dependencies
    start_monitoring_services
    verify_installation
    cleanup_temp_files
    
    print_access_info
}

# Handle script arguments
case "${1:-}" in
    "--verify")
        verify_installation
        ;;
    "--start")
        start_monitoring_services
        ;;
    "--stop")
        log_info "Stopping monitoring services..."
        systemctl stop grafana-server jaeger alertmanager prometheus node_exporter
        log_success "Monitoring services stopped"
        ;;
    "--restart")
        log_info "Restarting monitoring services..."
        systemctl restart node_exporter prometheus alertmanager jaeger grafana-server
        log_success "Monitoring services restarted"
        ;;
    "--status")
        log_info "Monitoring services status:"
        systemctl status node_exporter prometheus alertmanager jaeger grafana-server --no-pager
        ;;
    "--uninstall")
        log_warning "Uninstalling monitoring infrastructure..."
        systemctl stop grafana-server jaeger alertmanager prometheus node_exporter
        systemctl disable grafana-server jaeger alertmanager prometheus node_exporter
        rm -f /etc/systemd/system/{prometheus,jaeger,alertmanager,node_exporter}.service
        rm -f /usr/local/bin/{prometheus,promtool,jaeger-*,alertmanager,amtool,node_exporter}
        rm -rf "$MONITORING_DIR"
        rm -rf /var/lib/monitoring
        userdel monitoring 2>/dev/null || true
        systemctl daemon-reload
        log_success "Monitoring infrastructure uninstalled"
        ;;
    "")
        main
        ;;
    *)
        echo "Usage: $0 [--verify|--start|--stop|--restart|--status|--uninstall]"
        echo "  (no args)    - Full installation"
        echo "  --verify     - Verify installation"
        echo "  --start      - Start services"
        echo "  --stop       - Stop services"
        echo "  --restart    - Restart services"
        echo "  --status     - Show service status"
        echo "  --uninstall  - Remove installation"
        exit 1
        ;;
esac