# Installation Guide

This comprehensive guide covers installing SAFLA in various environments, from development setups to production deployments. Choose the installation method that best fits your needs.

## 📋 System Requirements

### Minimum Requirements
- **Operating System**: Linux (Ubuntu 18.04+), macOS (10.14+), or Windows 10+
- **Python**: 3.8 or higher (3.9+ recommended)
- **Memory**: 4GB RAM minimum
- **Storage**: 2GB free disk space
- **Network**: Internet connection for initial setup

### Recommended Requirements
- **Operating System**: Linux (Ubuntu 20.04+) or macOS (11.0+)
- **Python**: 3.10 or 3.11
- **Memory**: 8GB+ RAM for optimal performance
- **Storage**: 10GB+ free disk space
- **CPU**: Multi-core processor (4+ cores recommended)
- **GPU**: Optional, for accelerated vector operations

### Production Requirements
- **Memory**: 16GB+ RAM
- **Storage**: 50GB+ SSD storage
- **CPU**: 8+ cores
- **Network**: High-bandwidth connection
- **Monitoring**: System monitoring tools
- **Backup**: Automated backup solutions

## 🚀 Quick Installation

### Using pip (Recommended)

```bash
# Install from PyPI (when available)
pip install safla

# Or install from source
pip install git+https://github.com/ruvnet/SAFLA.git
```

### Using conda

```bash
# Create conda environment
conda create -n safla python=3.10
conda activate safla

# Install SAFLA
pip install safla
```

## 🛠️ Development Installation

### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/ruvnet/SAFLA.git
cd SAFLA

# Verify the clone
ls -la
```

### Step 2: Set Up Virtual Environment

#### Using venv (Python built-in)
```bash
# Create virtual environment
python -m venv safla-env

# Activate environment
# On Linux/macOS:
source safla-env/bin/activate
# On Windows:
safla-env\Scripts\activate

# Verify activation
which python  # Should point to safla-env
```

#### Using conda
```bash
# Create conda environment
conda create -n safla python=3.10 -y
conda activate safla

# Install additional conda packages (optional)
conda install numpy scipy scikit-learn -y
```

#### Using pyenv (Advanced)
```bash
# Install specific Python version
pyenv install 3.10.12
pyenv local 3.10.12

# Create virtual environment
python -m venv safla-env
source safla-env/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install SAFLA in editable mode
pip install -e .
```

### Step 4: Verify Installation

```bash
# Test basic import
python -c "from safla.core.hybrid_memory import HybridMemoryArchitecture; print('✅ SAFLA installed successfully!')"

# Run basic tests
python -m pytest tests/test_basic_functionality.py -v

# Check version
python -c "import safla; print(f'SAFLA version: {safla.__version__}')"
```

## 🐳 Docker Installation

### Using Pre-built Image

```bash
# Pull the official SAFLA image
docker pull ruvnet/safla:latest

# Run SAFLA container
docker run -it --name safla-container \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  ruvnet/safla:latest
```

### Building from Source

```bash
# Clone repository
git clone https://github.com/ruvnet/SAFLA.git
cd SAFLA

# Build Docker image
docker build -t safla:local .

# Run container
docker run -it --name safla-dev \
  -p 8000:8000 \
  -v $(pwd):/app \
  safla:local bash
```

### Docker Compose Setup

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  safla:
    build: .
    container_name: safla-app
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - SAFLA_ENVIRONMENT=development
      - SAFLA_LOG_LEVEL=INFO
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    container_name: safla-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15-alpine
    container_name: safla-postgres
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=safla
      - POSTGRES_USER=safla
      - POSTGRES_PASSWORD=safla_password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  redis_data:
  postgres_data:
```

Run with Docker Compose:
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f safla

# Stop services
docker-compose down
```

## ☁️ Cloud Installation

### AWS EC2

#### Launch Instance
```bash
# Launch EC2 instance (using AWS CLI)
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type t3.large \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxxx \
  --subnet-id subnet-xxxxxxxxx \
  --user-data file://install-script.sh
```

#### Install Script (`install-script.sh`)
```bash
#!/bin/bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10
sudo apt install python3.10 python3.10-venv python3.10-dev -y

# Install Git
sudo apt install git -y

# Clone SAFLA
cd /opt
sudo git clone https://github.com/ruvnet/SAFLA.git
sudo chown -R ubuntu:ubuntu SAFLA
cd SAFLA

# Set up virtual environment
python3.10 -m venv safla-env
source safla-env/bin/activate

# Install SAFLA
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Create systemd service
sudo tee /etc/systemd/system/safla.service > /dev/null <<EOF
[Unit]
Description=SAFLA Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/SAFLA
Environment=PATH=/opt/SAFLA/safla-env/bin
ExecStart=/opt/SAFLA/safla-env/bin/python -m safla.server
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable safla
sudo systemctl start safla
```

### Google Cloud Platform

```bash
# Create VM instance
gcloud compute instances create safla-instance \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --metadata-from-file startup-script=install-script.sh
```

### Azure

```bash
# Create resource group
az group create --name safla-rg --location eastus

# Create VM
az vm create \
  --resource-group safla-rg \
  --name safla-vm \
  --image UbuntuLTS \
  --size Standard_D4s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --custom-data install-script.sh
```

## 🔧 Configuration

### Environment Variables

Create `.env` file:

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration
nano .env  # or your preferred editor
```

#### Core Configuration
```bash
# Environment
SAFLA_ENVIRONMENT=development  # development, staging, production
SAFLA_LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR
SAFLA_DEBUG=true              # Enable debug mode

# Memory Configuration
SAFLA_VECTOR_DIMENSIONS=512,768,1024,1536
SAFLA_MAX_MEMORIES=10000
SAFLA_SIMILARITY_THRESHOLD=0.8
SAFLA_MEMORY_CONSOLIDATION_INTERVAL=3600  # seconds

# Safety Configuration
SAFLA_MEMORY_LIMIT=1000000000    # 1GB in bytes
SAFLA_CPU_LIMIT=0.9              # 90% CPU usage limit
SAFLA_SAFETY_MONITORING_INTERVAL=1.0  # seconds
SAFLA_EMERGENCY_STOP_ENABLED=true

# Performance Configuration
SAFLA_PERFORMANCE_MONITORING=true
SAFLA_METRICS_COLLECTION_INTERVAL=10  # seconds
SAFLA_CACHE_SIZE=1000
SAFLA_BATCH_SIZE=32

# MCP Configuration
SAFLA_MCP_TIMEOUT=30
SAFLA_MCP_MAX_RETRIES=3
SAFLA_MCP_HEALTH_CHECK_INTERVAL=60
SAFLA_MCP_SERVERS_CONFIG_PATH=.roo/mcp.json
```

#### Database Configuration
```bash
# Vector Database (Optional)
SAFLA_VECTOR_DB_URL=postgresql://user:pass@localhost:5432/safla_vectors
SAFLA_VECTOR_DB_POOL_SIZE=10

# Redis (Optional)
SAFLA_REDIS_URL=redis://localhost:6379/0
SAFLA_REDIS_MAX_CONNECTIONS=20

# File Storage
SAFLA_DATA_DIR=./data
SAFLA_LOGS_DIR=./logs
SAFLA_CHECKPOINTS_DIR=./checkpoints
```

#### Security Configuration
```bash
# API Security
SAFLA_API_KEY=your-secure-api-key
SAFLA_JWT_SECRET=your-jwt-secret
SAFLA_CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# Encryption
SAFLA_ENCRYPTION_KEY=your-encryption-key
SAFLA_ENCRYPT_MEMORIES=true
SAFLA_ENCRYPT_CHECKPOINTS=true
```

### MCP Server Configuration

Create `.roo/mcp.json`:

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"],
      "env": {
        "CONTEXT7_API_KEY": "your-api-key"
      }
    },
    "perplexity": {
      "command": "python",
      "args": ["-m", "mcp_perplexity"],
      "env": {
        "PERPLEXITY_API_KEY": "your-api-key"
      }
    },
    "filesystem": {
      "command": "python",
      "args": ["-m", "mcp_filesystem"],
      "env": {
        "ALLOWED_PATHS": "/safe/path1,/safe/path2"
      }
    }
  }
}
```

### Logging Configuration

Create `logging.yaml`:

```yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
  detailed:
    format: '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: logs/safla.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: detailed
    filename: logs/safla_errors.log
    maxBytes: 10485760
    backupCount: 5

loggers:
  safla:
    level: DEBUG
    handlers: [console, file, error_file]
    propagate: false

root:
  level: INFO
  handlers: [console]
```

## 🧪 Testing Installation

### Basic Functionality Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_hybrid_memory.py -v
python -m pytest tests/test_meta_cognitive.py -v
python -m pytest tests/test_safety_validation.py -v
python -m pytest tests/test_mcp_orchestration.py -v

# Run tests with coverage
python -m pytest --cov=safla --cov-report=html tests/

# Run performance tests
python -m pytest tests/test_performance.py -v --benchmark-only
```

### Integration Tests

```bash
# Test database connections
python -c "
from safla.core.hybrid_memory import HybridMemoryArchitecture
import asyncio

async def test_db():
    memory = HybridMemoryArchitecture()
    await memory.start()
    print('✅ Database connection successful')
    await memory.stop()

asyncio.run(test_db())
"

# Test MCP servers
python -c "
from safla.core.mcp_orchestration import MCPOrchestrator
import asyncio

async def test_mcp():
    orchestrator = MCPOrchestrator()
    await orchestrator.start()
    print('✅ MCP orchestration successful')
    await orchestrator.stop()

asyncio.run(test_mcp())
"
```

### Performance Benchmarks

```bash
# Run memory performance tests
python scripts/benchmark_memory.py

# Run safety validation benchmarks
python scripts/benchmark_safety.py

# Run end-to-end performance tests
python scripts/benchmark_e2e.py
```

## 🔧 Post-Installation Setup

### Create Required Directories

```bash
# Create data directories
mkdir -p data/{memories,checkpoints,logs,exports}

# Create configuration directories
mkdir -p .roo/{configs,plugins,extensions}

# Set permissions
chmod 755 data/
chmod 700 .roo/
```

### Initialize Database Schema

```bash
# Initialize vector database (if using external DB)
python -m safla.scripts.init_db

# Create initial safety checkpoints
python -m safla.scripts.create_checkpoint --name "initial_install"

# Verify schema
python -m safla.scripts.verify_schema
```

### Set Up Monitoring

```bash
# Install monitoring dependencies
pip install prometheus-client grafana-api

# Start monitoring services
python -m safla.monitoring.prometheus_exporter &
python -m safla.monitoring.health_checker &
```

## 🚨 Troubleshooting

### Common Installation Issues

#### Python Version Issues
```bash
# Check Python version
python --version

# Install specific Python version (Ubuntu)
sudo apt install python3.10 python3.10-venv python3.10-dev

# Update alternatives (if needed)
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
```

#### Dependency Conflicts
```bash
# Clear pip cache
pip cache purge

# Reinstall with no cache
pip install --no-cache-dir -r requirements.txt

# Use pip-tools for dependency resolution
pip install pip-tools
pip-compile requirements.in
pip-sync requirements.txt
```

#### Memory Issues
```bash
# Reduce memory usage in .env
SAFLA_MAX_MEMORIES=1000
SAFLA_VECTOR_DIMENSIONS=512
SAFLA_BATCH_SIZE=16

# Enable memory monitoring
SAFLA_MEMORY_MONITORING=true
```

#### Permission Issues
```bash
# Fix file permissions
sudo chown -R $USER:$USER /path/to/SAFLA
chmod -R 755 /path/to/SAFLA

# Fix virtual environment permissions
sudo chown -R $USER:$USER safla-env/
```

### Performance Issues

#### Slow Startup
```bash
# Enable startup profiling
SAFLA_PROFILE_STARTUP=true python -m safla.server

# Reduce initial memory allocation
SAFLA_INITIAL_MEMORY_SIZE=100
```

#### High Memory Usage
```bash
# Enable memory profiling
pip install memory-profiler
python -m memory_profiler your_script.py

# Optimize memory settings
SAFLA_MEMORY_OPTIMIZATION=aggressive
SAFLA_GC_THRESHOLD=100
```

### Network Issues

#### MCP Connection Problems
```bash
# Test MCP server connectivity
python -m safla.scripts.test_mcp_connection

# Check firewall settings
sudo ufw status
sudo ufw allow 8000/tcp

# Test with curl
curl -X GET http://localhost:8000/health
```

## 📚 Next Steps

After successful installation:

1. **[Quick Start Guide](02-quickstart.md)** - Test basic functionality
2. **[Configuration Guide](16-configuration.md)** - Customize your setup
3. **[System Architecture](04-architecture.md)** - Understand the system design
4. **[Development Guide](20-development.md)** - Start developing with SAFLA

## 🆘 Getting Help

If you encounter issues:

- **Check the [Troubleshooting Guide](27-troubleshooting.md)**
- **Review [FAQ](33-faq.md)** for common questions
- **Search [GitHub Issues](https://github.com/ruvnet/SAFLA/issues)**
- **Join the community forums** for support

---

**Next**: [System Architecture](04-architecture.md) - Understanding SAFLA's design  
**Previous**: [Quick Start Guide](02-quickstart.md) - Get up and running quickly