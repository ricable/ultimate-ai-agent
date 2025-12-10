# RAN Optimizer LLM Agent

WASI-NN based LLM agent for Ericsson RAN (Radio Access Network) optimization automation. Built with Rust, compiled to WebAssembly, and runs on WasmEdge with the GGML backend for efficient local inference.

## Overview

This agent provides intelligent analysis and optimization recommendations for:

- **Coverage Optimization** - Antenna tilt, power adjustments, coverage hole remediation
- **Capacity Optimization** - Load balancing, PRB utilization, carrier aggregation
- **Interference Mitigation** - ICIC/eICIC configuration, PCI optimization
- **Mobility/Handover** - Time-to-trigger, hysteresis, CIO tuning
- **Energy Efficiency** - Cell sleep modes, carrier shutdown scheduling
- **Anomaly Detection** - KPI deviation analysis, root cause identification

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAN Optimizer Agent                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Rust WASM     │    │  WASI-NN API    │                    │
│  │   Application   │───▶│  GraphBuilder   │                    │
│  └─────────────────┘    └────────┬────────┘                    │
│                                  │                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    WasmEdge Runtime                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │   │
│  │  │ GGML Plugin │  │   Llama.cpp │  │  GPU/CPU Accel  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                  │                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              GGUF Model (Qwen2.5/Llama/Mistral)          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

```bash
# Install WasmEdge with GGML plugin
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | \
  bash -s -- --plugins wasi_nn-ggml

# Install Rust with WASM target
rustup target add wasm32-wasi
```

### Build

```bash
# Using mise
mise run ran-build

# Or directly
cd agents/ran-optimizer
cargo build --release --target wasm32-wasi
```

### Download Model

```bash
# Download recommended model (Qwen2.5-7B)
mise run ran-download-model qwen2.5-7b

# Or smaller model for limited resources
mise run ran-download-model llama-3.2-3b
```

### Run

```bash
# Interactive mode
mise run ran-run

# Analyze specific metrics
mise run ran-analyze models/qwen2.5-7b-instruct-q5_k_m.gguf examples/sample_metrics.json

# Start API server
mise run ran-server models/qwen2.5-7b-instruct-q5_k_m.gguf 8090
```

## Usage Modes

### Interactive Chat

```bash
wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:models/qwen2.5-7b-instruct-q5_k_m.gguf \
  ran-optimizer.wasm default interactive
```

Commands:
- `/analyze <cell_id>` - Analyze specific cell
- `/metrics <json>` - Input custom metrics
- `/mode <type>` - Set optimization mode (coverage/capacity/interference/energy)
- `/clear` - Clear conversation history
- `/exit` - Exit session

### Batch Analysis

```bash
# Analyze metrics file
wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:models/model.gguf \
  ran-optimizer.wasm default analyze "$(cat metrics.json)"
```

### API Server

The agent can run as an OpenAI-compatible API server:

```bash
# Start server
mise run ran-server

# Send request
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ran-optimizer",
    "messages": [
      {"role": "user", "content": "Analyze cell ENB123 with RSRP -112dBm and recommend coverage improvements"}
    ]
  }'
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CTX_SIZE` | Context window size | 8192 |
| `N_PREDICT` | Maximum output tokens | 2048 |
| `N_GPU_LAYERS` | GPU offload layers | 35 |
| `TEMPERATURE` | Sampling temperature | 0.3 |
| `TOP_P` | Top-p sampling | 0.9 |
| `REPEAT_PENALTY` | Token repeat penalty | 1.1 |
| `STREAM_STDOUT` | Stream output to stdout | true |

### Recommended Models

| Model | Size | Context | Best For |
|-------|------|---------|----------|
| Qwen2.5-7B-Instruct | 5.1 GB | 32K | Structured JSON output, reasoning |
| Llama-3.2-3B-Instruct | 2.3 GB | 131K | Fast inference, limited resources |
| Mistral-7B-Instruct | 5.0 GB | 32K | General purpose, balanced |

## Kubernetes Deployment

```bash
# Deploy to cluster
mise run ran-deploy

# Check status
kubectl -n ran-optimizer get pods

# View logs
kubectl -n ran-optimizer logs -f deployment/ran-optimizer
```

### GPU Acceleration

The deployment supports NVIDIA GPUs via the `llamaedge-gpu` runtime class:

```yaml
spec:
  runtimeClassName: llamaedge-gpu
  containers:
    - resources:
        limits:
          nvidia.com/gpu: "1"
```

## Integration with LiteLLM

The RAN Optimizer is registered in LiteLLM gateway as:
- `ran-optimizer` - Standard CPU inference
- `ran-optimizer-gpu` - GPU-accelerated inference

```python
import litellm

response = litellm.completion(
    model="ran-optimizer",
    messages=[{
        "role": "user",
        "content": "Analyze cell with high PRB utilization and recommend load balancing"
    }]
)
```

## Example Scenarios

See the `examples/` directory for sample scenarios:

- `sample_metrics.json` - Multi-cell network snapshot
- `coverage_analysis.json` - Coverage optimization case
- `capacity_optimization.json` - Load balancing scenario
- `handover_optimization.json` - Mobility parameter tuning

## RAN KPIs Supported

### Radio Quality
- RSRP (Reference Signal Received Power)
- RSRQ (Reference Signal Received Quality)
- SINR (Signal to Interference plus Noise Ratio)
- CQI (Channel Quality Indicator)
- PRB Utilization

### Throughput
- DL/UL Throughput (user and cell level)
- Data Volume
- Spectral Efficiency

### Mobility
- Handover Success Rate
- Ping-pong Rate
- Too-early/Too-late Handover

### Accessibility
- RRC Setup Success Rate
- E-RAB Setup Success Rate
- RACH Success Rate

### Retainability
- Call Drop Rate
- E-RAB Drop Rate
- Abnormal Release Rate

## Ericsson ENM Parameters

The agent understands and can recommend changes to common ENM parameters:

- `referenceSignalPower` - RS power adjustment
- `pZeroNominalPusch/Pucch` - Uplink power control
- `timeToTrigger` - HO trigger delay
- `hysteresis` - HO hysteresis
- `a3Offset` - A3 event offset
- `cellIndividualOffset` - Per-relation offset
- `loadBalancingActive` - MLB activation
- `energySavingState` - Energy saving mode

## Testing

```bash
# Run unit tests
mise run ran-test

# Or directly
cd agents/ran-optimizer
cargo test --release
```

## License

MIT License - See LICENSE file for details.
