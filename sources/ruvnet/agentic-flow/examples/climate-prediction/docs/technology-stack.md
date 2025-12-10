# Technology Stack Specification

**Version:** 1.0.0
**Date:** 2025-10-14
**Purpose:** Comprehensive library, framework, and dependency specifications

## Technology Selection Philosophy

**Key Principles:**
1. **Use existing libraries** - Don't build from scratch what already exists
2. **Rust for production** - Python for prototyping, Rust for deployment
3. **Cloud-native** - Kubernetes, containerization, scalable infrastructure
4. **Open-source first** - Prefer OSS with commercial support available
5. **Battle-tested** - Choose mature libraries with active maintenance

## Core Technology Stack

### 1. Machine Learning Frameworks

#### 1.1 Python (Prototyping & Training)

**Primary Framework: PyTorch**

```bash
# Installation
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning==2.1.0
pip install torchmetrics==1.2.0
```

**Why PyTorch:**
- Industry standard for research (80%+ of papers)
- Excellent GPU support (CUDA, ROCm)
- Dynamic computation graphs (debugging friendly)
- Extensive ecosystem (HuggingFace, timm, etc.)
- Strong distributed training (FSDP, DDP)

**Alternative: JAX** (for advanced users)

```bash
pip install jax[cuda12_pip]==0.4.20 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax==0.7.5
pip install optax==0.1.7
```

**JAX Advantages:**
- Functional programming paradigm
- Auto-vectorization (vmap)
- Better XLA compilation
- Used by Google's climate models (GraphCast, NeuralGCM)

#### 1.2 Rust (Production Inference)

**Primary Framework: Burn**

```toml
[dependencies]
burn = { version = "0.13.0", features = ["std", "wgpu"] }
burn-ndarray = "0.13.0"
burn-tch = { version = "0.13.0", optional = true }  # PyTorch backend
burn-candle = { version = "0.13.0", optional = true }  # Candle backend

# GPU backends
wgpu = "0.18"  # Cross-platform (Vulkan, Metal, DX12, WebGPU)
tch = { version = "0.15.0", optional = true }  # CUDA via LibTorch
```

**Why Burn:**
- Backend-agnostic (single codebase for all platforms)
- Production-ready (v0.13+, stable API)
- Excellent type safety (Rust's strong typing)
- Native mixed precision support
- Active development (50+ contributors)
- Best documentation among Rust ML frameworks

**Alternative 1: tch-rs** (PyTorch bindings)

```toml
[dependencies]
tch = "0.15.0"
torch-sys = "0.15.0"
```

**tch-rs Advantages:**
- Direct PyTorch compatibility (load .pt files directly)
- 5.5x faster CPU training than Python
- Full CUDA support
- Minimal porting effort from PyTorch

**Alternative 2: Candle** (HuggingFace)

```toml
[dependencies]
candle-core = "0.3.0"
candle-nn = "0.3.0"
candle-transformers = "0.3.0"
```

**Candle Advantages:**
- Optimized for inference (not training)
- Excellent ONNX support
- WebAssembly ready
- Small binary size (10-50MB)

#### 1.3 ONNX (Interoperability)

**Export from PyTorch:**

```python
import torch

# Export trained model to ONNX
dummy_input = torch.randn(1, 13, 256, 256)
torch.onnx.export(
    model,
    dummy_input,
    "climate-model.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

**Load in Rust (tract):**

```toml
[dependencies]
tract-onnx = "0.20.0"
tract-hir = "0.20.0"
```

```rust
use tract_onnx::prelude::*;

// Load and optimize ONNX model
let model = tract_onnx::onnx()
    .model_for_path("climate-model.onnx")?
    .with_input_fact(0, f32::fact(&[1, 13, 256, 256]).into())?
    .into_optimized()?
    .into_runnable()?;

// Run inference
let result = model.run(tvec![input_tensor.into()])?;
```

### 2. Neural Operator Libraries

#### 2.1 Fourier Neural Operators

**NeuralOperator (Python)**

```bash
pip install neuraloperator==0.3.0
pip install torch-harmonics==0.6.2  # Spherical harmonics
```

**Usage:**

```python
from neuralop.models import SFNO

model = SFNO(
    n_modes=(32, 32),        # Fourier modes (lat, lon)
    hidden_channels=256,
    projection_channels=128,
    n_layers=8,
    domain_padding=0.1,
    use_spectral_norm=True,
)

# Train on ERA5 data
output = model(input_data)  # [batch, channels, lat, lon]
```

**NVIDIA FourCastNet (Reference Implementation)**

```bash
git clone https://github.com/NVlabs/FourCastNet.git
pip install -e FourCastNet
```

**Key Features:**
- Pre-configured SFNO for weather
- ERA5 data loaders
- Distributed training scripts
- Benchmark evaluation tools

#### 2.2 Graph Neural Networks

**PyTorch Geometric**

```bash
pip install torch-geometric==2.4.0
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

**Usage:**

```python
import torch_geometric as pyg
from torch_geometric.nn import MessagePassing

class MultiMeshGNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, out_channels),
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return self.mlp(torch.cat([x_i, x_j], dim=-1))
```

**GraphCast Reference:**

```bash
git clone https://github.com/google-deepmind/graphcast.git
pip install -e graphcast
```

#### 2.3 Physics-Informed Neural Networks

**DeepXDE**

```bash
pip install deepxde==1.10.0
```

**Usage:**

```python
import deepxde as dde

# Define PDE (e.g., heat equation)
def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=0)
    dy_xx = dde.grad.hessian(y, x, i=1, j=1)
    return dy_t - alpha * dy_xx

# Define domain and boundary conditions
geom = dde.geometry.Rectangle([0, 0], [1, 1])
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Train physics-informed model
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
model.train(epochs=10000)
```

**ClimODE (Physics-Informed Neural ODE)**

```bash
git clone https://github.com/Aalto-QuML/ClimODE.git
pip install -e ClimODE
```

### 3. Data Processing & Storage

#### 3.1 Data Formats

**Zarr (Cloud-Optimized Array Storage)**

```bash
pip install zarr==2.16.0
pip install fsspec==2023.12.0  # Abstract filesystems
pip install s3fs==2023.12.0    # S3 support
pip install gcsfs==2023.12.0   # Google Cloud Storage
```

**Usage:**

```python
import zarr
import fsspec

# Open ERA5 on cloud storage
store = fsspec.get_mapper('gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3')
root = zarr.open(store, mode='r')

# Access data efficiently
temperature = root['2m_temperature'][
    '2020-01-01':'2020-12-31',  # Time
    :,  # All latitudes
    :   # All longitudes
]
```

**NetCDF (Legacy Climate Data)**

```bash
pip install netCDF4==1.6.5
pip install xarray==2023.12.0
```

```python
import xarray as xr

# Open multi-file dataset
ds = xr.open_mfdataset(
    'era5_*.nc',
    combine='by_coords',
    parallel=True,
    chunks={'time': 24, 'latitude': 100, 'longitude': 100}
)
```

#### 3.2 Distributed Data Processing

**Dask (Parallel Computing)**

```bash
pip install dask[complete]==2023.12.0
pip install dask-ml==2023.12.0
```

**Usage:**

```python
import dask.array as da
import dask.dataframe as dd

# Lazy loading large datasets
data = da.from_zarr('era5.zarr', chunks=(24, 100, 100, 13))

# Parallel preprocessing
normalized = (data - data.mean(axis=0)) / data.std(axis=0)

# Compute only when needed
result = normalized.compute()  # Triggers parallel execution
```

**Ray (Distributed ML)**

```bash
pip install ray[default]==2.9.0
pip install ray[train]==2.9.0
```

**Usage:**

```python
import ray
from ray import train
from ray.train import ScalingConfig

# Distributed training
trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(
        num_workers=16,
        use_gpu=True,
        resources_per_worker={"GPU": 1, "CPU": 8}
    ),
)
trainer.fit()
```

### 4. Weather Data APIs

#### 4.1 ERA5 (Training Data)

**Copernicus Climate Data Store API**

```bash
pip install cdsapi==0.6.1
```

**Setup:**

```bash
# ~/.cdsapirc
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_UID:YOUR_API_KEY
```

**Usage:**

```python
import cdsapi

c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': ['2m_temperature', '10m_u_component_of_wind'],
        'year': '2023',
        'month': '01',
        'day': list(range(1, 32)),
        'time': list(range(0, 24)),
        'format': 'netcdf',
    },
    'era5_2023_01.nc'
)
```

**Alternative: ARCO-ERA5 (Cloud-Optimized)**

```python
# Direct access via Zarr (no API key)
import xarray as xr

ds = xr.open_zarr(
    'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
    chunks=None,
    storage_options={'token': 'anon'}
)
```

#### 4.2 Real-Time Weather APIs

**OpenWeatherMap**

```bash
pip install pyowm==3.3.0
```

```python
from pyowm import OWM

owm = OWM('YOUR_API_KEY')
mgr = owm.weather_manager()

# Current weather
observation = mgr.weather_at_place('London,GB')
weather = observation.weather
temp = weather.temperature('celsius')['temp']

# Forecast
forecast = mgr.forecast_at_place('London,GB', '3h')
```

**Open-Meteo (Free, No Auth)**

```bash
pip install openmeteo-requests==1.1.0
```

```python
import openmeteo_requests

om = openmeteo_requests.Client()
params = {
    "latitude": 52.52,
    "longitude": 13.41,
    "hourly": ["temperature_2m", "wind_speed_10m"],
    "forecast_days": 7
}
response = om.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
```

**NOAA API**

```bash
pip install noaa-sdk==0.1.25
```

```python
from noaa_sdk import NOAA

n = NOAA()
observations = n.get_observations('KJFK', 'US')  # JFK Airport
forecasts = n.get_forecasts('02134', 'US', type='forecastGridData')
```

### 5. Web Framework & API

#### 5.1 Rust Web Server

**Axum (Tokio Ecosystem)**

```toml
[dependencies]
axum = "0.7.0"
tokio = { version = "1.35", features = ["full"] }
tower = "0.4"
tower-http = { version = "0.5", features = ["cors", "compression-gzip"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

**Example API Server:**

```rust
use axum::{
    Router,
    routing::{get, post},
    Json, extract::State,
};
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::main]
async fn main() {
    let state = AppState::new().await;

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/predict", post(predict))
        .route("/forecast/:location", get(get_forecast))
        .with_state(Arc::new(RwLock::new(state)))
        .layer(tower_http::cors::CorsLayer::permissive())
        .layer(tower_http::compression::CompressionLayer::new());

    axum::Server::bind(&"0.0.0.0:8080".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
```

**Alternative: Actix-Web**

```toml
[dependencies]
actix-web = "4.4"
actix-rt = "2.9"
```

### 6. Containerization & Orchestration

#### 6.1 Docker

**Multi-Stage Build (Rust Inference Server):**

```dockerfile
# Stage 1: Build
FROM rust:1.75-slim as builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN apt-get update && apt-get install -y pkg-config libssl-dev
RUN cargo build --release

# Stage 2: Runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y libssl3 ca-certificates
COPY --from=builder /app/target/release/climate-server /usr/local/bin/
COPY models /models

EXPOSE 8080
CMD ["climate-server"]
```

**Result:** 50-100MB image (vs 1GB+ Python)

#### 6.2 Kubernetes

**Helm Chart:**

```yaml
# values.yaml
replicaCount: 3
image:
  repository: climate-inference
  tag: v1.0.0
  pullPolicy: IfNotPresent

resources:
  requests:
    memory: "4Gi"
    cpu: "2"
    nvidia.com/gpu: "1"
  limits:
    memory: "8Gi"
    cpu: "4"
    nvidia.com/gpu: "1"

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

service:
  type: LoadBalancer
  port: 80
  targetPort: 8080
```

### 7. Monitoring & Observability

#### 7.1 Metrics

**Prometheus (Rust Client)**

```toml
[dependencies]
prometheus = "0.13"
lazy_static = "1.4"
```

```rust
use prometheus::{Encoder, TextEncoder, Registry, Histogram, Counter};

lazy_static! {
    static ref INFERENCE_DURATION: Histogram = Histogram::new(
        "inference_duration_seconds",
        "Model inference time"
    ).unwrap();
}
```

#### 7.2 Logging

**Tracing (Structured Logging)**

```toml
[dependencies]
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
```

```rust
use tracing::{info, warn, error, instrument};

#[instrument]
async fn predict(input: Input) -> Result<Output> {
    info!("Starting prediction");
    let result = model.predict(input).await;
    info!("Prediction complete");
    result
}
```

### 8. ML Operations (MLOps)

#### 8.1 Experiment Tracking

**MLflow**

```bash
pip install mlflow==2.9.0
```

```python
import mlflow

mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("climate-prediction")

with mlflow.start_run():
    mlflow.log_params({"lr": 1e-4, "batch_size": 32})
    mlflow.log_metrics({"rmse": 304.5, "latency_ms": 35})
    mlflow.pytorch.log_model(model, "model")
```

#### 8.2 Model Registry

**HuggingFace Hub (Open Source Models)**

```bash
pip install huggingface-hub==0.20.0
```

```python
from huggingface_hub import HfApi, ModelCard

api = HfApi()
api.upload_file(
    path_or_fileobj="climate-model.onnx",
    path_in_repo="climate-model.onnx",
    repo_id="your-org/climate-sfno",
    repo_type="model",
)
```

### 9. GPU & Acceleration

#### 9.1 CUDA (NVIDIA)

```toml
[dependencies]
cudarc = "0.9"  # Rust CUDA bindings
```

**Check CUDA availability:**

```rust
use cudarc::driver::CudaDevice;

let device = CudaDevice::new(0)?;  // GPU 0
println!("Using GPU: {}", device.name());
```

#### 9.2 WGPU (Cross-Platform)

```toml
[dependencies]
wgpu = "0.18"
```

**Advantages:**
- Works on NVIDIA, AMD, Apple Silicon
- Same code for all platforms
- WebGPU support for browsers

### 10. Testing & Validation

#### 10.1 Unit Testing (Rust)

```toml
[dev-dependencies]
approx = "0.5"  # Floating point comparison
proptest = "1.4"  # Property-based testing
```

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_model_output_shape() {
        let input = Tensor::zeros(&[1, 13, 256, 256]);
        let output = model.forward(input);
        assert_eq!(output.shape(), &[1, 13, 256, 256]);
    }
}
```

#### 10.2 Integration Testing

**Pytest**

```bash
pip install pytest==7.4.0
pip install pytest-cov==4.1.0
```

```python
def test_end_to_end_prediction():
    # Load model
    model = load_model("climate-model.onnx")

    # Create test input
    input_data = create_test_input()

    # Run inference
    output = model.predict(input_data)

    # Validate output
    assert output.shape == (1, 13, 256, 256)
    assert -80 < output['temperature'].mean() < 60  # Physical bounds
```

## Dependency Management

### Python (pyproject.toml)

```toml
[project]
name = "climate-prediction"
version = "1.0.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1.0",
    "pytorch-lightning>=2.1.0",
    "neuraloperator>=0.3.0",
    "xarray>=2023.12.0",
    "zarr>=2.16.0",
    "dask[complete]>=2023.12.0",
    "mlflow>=2.9.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.12.0",
    "ruff>=0.1.0",
]
```

### Rust (Cargo.toml)

```toml
[workspace]
members = ["inference-server", "preprocessing", "common"]

[workspace.dependencies]
burn = "0.13"
tokio = "1.35"
axum = "0.7"
serde = "1.0"
tract-onnx = "0.20"

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
```

## Total Stack Summary

| Layer | Python (Training) | Rust (Inference) |
|-------|-------------------|------------------|
| **ML Framework** | PyTorch 2.1 / JAX 0.4 | Burn 0.13 / tch-rs 0.15 |
| **Neural Ops** | neuraloperator, torch-harmonics | Custom SFNO impl |
| **Data** | xarray, Zarr, Dask | serde, bincode |
| **APIs** | requests, httpx | reqwest, hyper |
| **Web** | FastAPI, Flask | Axum, Actix-Web |
| **Inference** | ONNX Runtime | tract-onnx |
| **GPU** | CUDA 12.1, cuDNN 8.9 | cudarc, wgpu |
| **Container** | python:3.11-slim (400MB) | debian:bookworm-slim (50MB) |
| **Monitoring** | prometheus-client | prometheus crate |

**Next Steps:**
1. Set up development environment with these dependencies
2. Implement data pipeline (data-pipeline.md)
3. Build ML models (ml-models.md)
