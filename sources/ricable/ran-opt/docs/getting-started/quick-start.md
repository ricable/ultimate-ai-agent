# Quick Start Guide

This guide will get you up and running with RAN-OPT in just a few minutes.

## Prerequisites

Before starting, ensure you have:
- ✅ Rust 1.70+ installed
- ✅ 8GB+ RAM available
- ✅ RAN-OPT successfully built (see [Installation Guide](installation.md))

## Step 1: Basic Setup

### Clone and Build (if not done already)

```bash
git clone https://github.com/ran-opt/ran-opt.git
cd ran-opt
cargo build --release
```

### Create Configuration

```bash
# Create basic configuration
cat > config.toml << EOF
[platform]
gpu_enabled = false  # Set to true if you have CUDA
worker_threads = 4
max_batch_size = 256

[logging]
level = "info"
format = "pretty"

[monitoring]
metrics_port = 9090

[networking]
grpc_port = 50051
http_port = 8080

[agents]
# Enable core agents for quick start
pfs_core = true
pfs_data = true
dtm_traffic = true
afm_detect = true
EOF
```

## Step 2: Start the Platform

### Basic Startup

```bash
# Start with default configuration
./target/release/ran-opt

# Or start with custom config
./target/release/ran-opt --config config.toml
```

You should see output similar to:
```
[INFO] RAN-OPT v1.0.0 starting...
[INFO] Initializing Platform Foundation Services...
[INFO] Starting PFS-Core agent...
[INFO] Starting PFS-Data agent...
[INFO] Starting DTM-Traffic agent...
[INFO] Starting AFM-Detect agent...
[INFO] All agents started successfully
[INFO] gRPC server listening on 0.0.0.0:50051
[INFO] HTTP API server listening on 0.0.0.0:8080
[INFO] Metrics available at http://localhost:9090/metrics
```

## Step 3: Verify Installation

### Health Check

In a new terminal, verify the platform is running:

```bash
# Check platform health
curl http://localhost:8080/health

# Expected response:
# {"status":"healthy","agents":{"pfs_core":"running","pfs_data":"running","dtm_traffic":"running","afm_detect":"running"}}
```

### View Metrics

```bash
# View basic metrics
curl http://localhost:9090/metrics | grep ran_opt

# Example output:
# ran_opt_agents_active 4
# ran_opt_requests_total 0
# ran_opt_memory_usage_bytes 134217728
```

## Step 4: Run Your First Example

### Traffic Prediction Example

```bash
# Generate sample traffic data
cargo run --example dtm_traffic_prediction

# This will:
# 1. Generate synthetic traffic patterns
# 2. Train a simple LSTM model
# 3. Make predictions for the next 4 hours
# 4. Display results and AMOS scripts
```

Expected output:
```
Training traffic prediction model...
Epoch 1/50: Loss = 0.1234
Epoch 50/50: Loss = 0.0123

Making predictions...
Predicted PRB utilization for Cell_001:
  +1h: 65.3% (±5.2%)
  +2h: 72.1% (±6.8%)
  +4h: 68.9% (±7.1%)

Generated AMOS scripts:
  - Handover trigger: L2100 → N78 (30% traffic)
  - Antenna tilt adjustment: +2.0°
```

### Anomaly Detection Example

```bash
# Run anomaly detection
cargo run --example afm_detection_example

# This will:
# 1. Load sample KPI data
# 2. Train an autoencoder
# 3. Detect anomalies
# 4. Generate alerts
```

## Step 5: Basic API Usage

### REST API Examples

```bash
# Get agent status
curl http://localhost:8080/api/v1/agents/status

# Submit traffic data for prediction
curl -X POST http://localhost:8080/api/v1/traffic/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cell_id": "Cell_001",
    "timestamp": "2024-01-01T12:00:00Z",
    "prb_utilization": 0.65,
    "layer": "N78",
    "service_type": "eMBB"
  }'

# Get prediction results
curl http://localhost:8080/api/v1/traffic/predictions/Cell_001
```

### gRPC API Examples (using grpcurl)

```bash
# Install grpcurl if not available
go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest

# List available services
grpcurl -plaintext localhost:50051 list

# Call traffic prediction service
grpcurl -plaintext -d '{
  "cell_id": "Cell_001",
  "historical_data": [
    {"timestamp": "2024-01-01T12:00:00Z", "prb_util": 0.65}
  ]
}' localhost:50051 ran_opt.traffic.TrafficService/Predict
```

## Step 6: View Results

### Web Dashboard (Optional)

If you have a web browser, you can view the built-in dashboard:

```bash
# Open in browser
open http://localhost:8080/dashboard

# Or use curl to see the HTML
curl http://localhost:8080/dashboard
```

### Log Analysis

```bash
# View real-time logs
tail -f logs/ran-opt.log

# Search for specific events
grep "prediction" logs/ran-opt.log
grep "anomaly" logs/ran-opt.log
grep "ERROR" logs/ran-opt.log
```

## Step 7: Advanced Configuration

### Enable GPU Acceleration (if available)

```bash
# Update config.toml
cat >> config.toml << EOF

[cuda]
device_id = 0
memory_pool_size = 1073741824  # 1GB
num_streams = 4
EOF

# Restart with GPU support
./target/release/ran-opt --config config.toml --features gpu
```

### Enable More Agents

```bash
# Add more agents to config.toml
cat >> config.toml << EOF

[agents]
pfs_core = true
pfs_data = true
pfs_twin = true
pfs_logs = true
dtm_traffic = true
dtm_power = true
dtm_mobility = true
afm_detect = true
afm_correlate = true
afm_rca = true
aos_heal = true
EOF

# Restart platform
./target/release/ran-opt --config config.toml
```

## Common Use Cases

### 1. Network Traffic Monitoring

```bash
# Start traffic monitoring
cargo run --example pfs_data_demo

# Monitor in real-time
curl http://localhost:8080/api/v1/traffic/realtime
```

### 2. Anomaly Detection

```bash
# Configure anomaly detection
curl -X POST http://localhost:8080/api/v1/afm/configure \
  -H "Content-Type: application/json" \
  -d '{
    "threshold": 0.95,
    "window_size": 60,
    "detection_method": "autoencoder"
  }'

# Get anomaly alerts
curl http://localhost:8080/api/v1/afm/alerts
```

### 3. Self-Healing Actions

```bash
# Enable self-healing
curl -X POST http://localhost:8080/api/v1/aos/enable \
  -H "Content-Type: application/json" \
  -d '{
    "auto_execute": false,
    "approval_required": true
  }'

# View suggested actions
curl http://localhost:8080/api/v1/aos/suggestions
```

## Performance Tuning

### For Development

```toml
[platform]
worker_threads = 2
max_batch_size = 128

[cuda]
memory_pool_size = 268435456  # 256MB
```

### For Production

```toml
[platform]
worker_threads = 16
max_batch_size = 2048

[cuda]
memory_pool_size = 2147483648  # 2GB
optimize_memory = true
enable_fusion = true
```

## Troubleshooting

### Platform Won't Start

```bash
# Check port conflicts
netstat -tuln | grep -E ':(50051|8080|9090)'

# Check permissions
ls -la target/release/ran-opt
chmod +x target/release/ran-opt

# Check dependencies
ldd target/release/ran-opt
```

### High Memory Usage

```bash
# Monitor memory usage
watch -n 1 'curl -s http://localhost:9090/metrics | grep memory'

# Reduce batch size
sed -i 's/max_batch_size = .*/max_batch_size = 128/' config.toml
```

### Slow Performance

```bash
# Check CPU usage
top -p $(pgrep ran-opt)

# Enable SIMD optimizations
echo 'simd_enabled = true' >> config.toml

# Use release build
cargo build --release
```

## Next Steps

Now that you have RAN-OPT running, explore these areas:

1. **[Examples](examples.md)** - More detailed examples and use cases
2. **[Module Documentation](../modules/README.md)** - Learn about specific modules
3. **[API Reference](../apis/rust-api.md)** - Complete API documentation
4. **[Configuration](../deployment/configuration.md)** - Advanced configuration options
5. **[Performance Tuning](../deployment/performance-tuning.md)** - Optimize for your environment

## Getting Help

- 📖 **Documentation**: [Full documentation](../README.md)
- 🐛 **Issues**: [GitHub Issues](https://github.com/ran-opt/ran-opt/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/ran-opt/ran-opt/discussions)
- 🔧 **Troubleshooting**: [Common issues](../reference/troubleshooting.md)