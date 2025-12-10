# Migration Guide

## Overview

This guide provides comprehensive instructions for migrating between SAFLA versions, upgrading from legacy systems, and ensuring compatibility across different environments. It includes step-by-step procedures, compatibility matrices, and troubleshooting guidance.

## Table of Contents

- [Version Migration](#version-migration)
- [Legacy System Migration](#legacy-system-migration)
- [Data Migration](#data-migration)
- [Configuration Migration](#configuration-migration)
- [API Migration](#api-migration)
- [Deployment Migration](#deployment-migration)
- [Rollback Procedures](#rollback-procedures)
- [Testing and Validation](#testing-and-validation)

## Version Migration

### Compatibility Matrix

| From Version | To Version | Migration Type | Complexity | Estimated Time |
|--------------|------------|----------------|------------|----------------|
| 0.7.x | 0.8.x | Major | High | 4-8 hours |
| 0.8.x | 0.9.x | Major | Medium | 2-4 hours |
| 0.9.x | 1.0.0 | Major | Medium | 2-4 hours |
| 1.0.x | 1.1.x | Minor | Low | 30-60 minutes |
| 1.1.x | 1.2.x | Minor | Low | 30-60 minutes |

### Migration from 0.9.x to 1.0.0

This is the most common migration path for current users.

#### Pre-Migration Checklist

1. **Backup your data**:
   ```bash
   # Backup configuration
   cp -r config/ config-backup-$(date +%Y%m%d)/
   
   # Backup memory data
   safla backup create --output backup-$(date +%Y%m%d).tar.gz
   
   # Backup custom extensions
   cp -r extensions/ extensions-backup-$(date +%Y%m%d)/
   ```

2. **Check system requirements**:
   ```bash
   safla system-check --target-version 1.0.0
   ```

3. **Review breaking changes**:
   - Memory API changes
   - Configuration format updates
   - MCP protocol updates
   - Safety constraint modifications

#### Step-by-Step Migration

**Step 1: Update Dependencies**

```bash
# Update Python dependencies
pip install --upgrade safla==1.0.0

# Update Node.js dependencies (if using Node.js components)
npm update safla@1.0.0
```

**Step 2: Migrate Configuration**

```bash
# Use the migration tool
safla migrate config --from 0.9 --to 1.0 --config-dir ./config

# Manual migration if needed
safla config validate --version 1.0
```

**Configuration Changes (0.9.x → 1.0.0)**:

```yaml
# OLD (0.9.x)
memory:
  vector_db:
    type: "faiss"
    index_type: "flat"
  
  episodic:
    storage: "sqlite"
    max_size: 1000000

# NEW (1.0.0)
memory:
  vector:
    provider: "faiss"
    index_type: "hnsw"
    dimensions: 768
  
  episodic:
    provider: "sqlite"
    max_episodes: 1000000
    retention_policy: "time_based"
```

**Step 3: Migrate Memory Data**

```bash
# Migrate vector memory
safla migrate memory vector --from-version 0.9 --to-version 1.0

# Migrate episodic memory
safla migrate memory episodic --from-version 0.9 --to-version 1.0

# Migrate semantic memory
safla migrate memory semantic --from-version 0.9 --to-version 1.0
```

**Step 4: Update Custom Code**

API changes that require code updates:

```python
# OLD (0.9.x)
from safla.memory import MemorySystem
memory = MemorySystem()
result = memory.store_vector(data, metadata)

# NEW (1.0.0)
from safla.memory import VectorMemory
memory = VectorMemory()
result = await memory.store(data, metadata=metadata)
```

```python
# OLD (0.9.x)
from safla.agents import AgentManager
manager = AgentManager()
manager.add_agent(agent_config)

# NEW (1.0.0)
from safla.coordination import AgentCoordinator
coordinator = AgentCoordinator()
await coordinator.register_agent(agent_instance)
```

**Step 5: Update MCP Servers**

```yaml
# OLD (0.9.x)
mcp:
  servers:
    - name: "custom-server"
      command: ["node", "server.js"]
      
# NEW (1.0.0)
mcp:
  servers:
    custom-server:
      command: "node"
      args: ["server.js"]
      env:
        LOG_LEVEL: "info"
```

**Step 6: Validate Migration**

```bash
# Run validation tests
safla validate --comprehensive

# Check memory integrity
safla memory validate --all-types

# Test agent coordination
safla agents test --all
```

### Migration from 0.8.x to 0.9.x

#### Key Changes

1. **Agent API Restructure**: Complete rewrite of agent coordination
2. **Memory Consolidation**: New consolidation algorithms
3. **Safety Framework**: Enhanced safety constraints

#### Migration Steps

```bash
# 1. Backup and prepare
safla backup create --version 0.8

# 2. Update configuration
safla migrate config --from 0.8 --to 0.9

# 3. Migrate agent definitions
safla migrate agents --from 0.8 --to 0.9

# 4. Update memory settings
safla migrate memory --from 0.8 --to 0.9

# 5. Validate
safla validate --target-version 0.9
```

### Migration from 0.7.x to 0.8.x

This is a major architectural migration requiring significant changes.

#### Breaking Changes

1. **Complete API Rewrite**: All APIs have changed
2. **Memory System Overhaul**: New memory architecture
3. **Configuration Format**: YAML-based configuration
4. **Agent System**: Introduction of MCP-based coordination

#### Migration Strategy

Due to the extensive changes, this migration requires a **fresh installation** approach:

```bash
# 1. Export data from 0.7.x
safla-0.7 export --format json --output legacy-data.json

# 2. Install SAFLA 0.8.x
pip install safla==0.8.0

# 3. Import data using migration tool
safla import legacy --source legacy-data.json --format 0.7

# 4. Rewrite custom code for new APIs
# (Manual process - see API migration guide)
```

## Legacy System Migration

### From Traditional ML Frameworks

#### From TensorFlow/PyTorch

```python
# Migration helper for TensorFlow models
from safla.migration import TensorFlowMigrator

migrator = TensorFlowMigrator()

# Convert TensorFlow model to SAFLA agent
agent = migrator.convert_model(
    model_path="path/to/model.pb",
    agent_type="classification",
    capabilities=["data_analysis"]
)

# Register in SAFLA
coordinator = AgentCoordinator()
await coordinator.register_agent(agent)
```

#### From Scikit-learn

```python
# Migration helper for scikit-learn models
from safla.migration import SklearnMigrator
import joblib

# Load existing model
model = joblib.load('model.pkl')

# Convert to SAFLA agent
migrator = SklearnMigrator()
agent = migrator.convert_model(
    model=model,
    model_type="regression",
    feature_names=feature_names
)
```

### From Vector Databases

#### From Pinecone

```python
from safla.migration import PineconeMigrator

migrator = PineconeMigrator(
    api_key="your-pinecone-key",
    environment="us-west1-gcp"
)

# Migrate vectors to SAFLA
await migrator.migrate_index(
    index_name="source-index",
    target_memory="vector",
    batch_size=1000
)
```

#### From Weaviate

```python
from safla.migration import WeaviateMigrator

migrator = WeaviateMigrator(url="http://localhost:8080")

# Migrate schema and data
await migrator.migrate_class(
    class_name="Document",
    target_memory="semantic",
    include_vectors=True
)
```

## Data Migration

### Memory Data Migration

#### Vector Memory Migration

```python
from safla.migration import VectorMemoryMigrator

async def migrate_vector_data():
    migrator = VectorMemoryMigrator()
    
    # Configure source and target
    await migrator.configure_source(
        provider="faiss",
        path="./old_vectors.index"
    )
    
    await migrator.configure_target(
        provider="qdrant",
        url="http://localhost:6333"
    )
    
    # Perform migration with progress tracking
    async for progress in migrator.migrate():
        print(f"Migration progress: {progress.percentage}%")
        print(f"Vectors migrated: {progress.vectors_migrated}")
        print(f"Estimated time remaining: {progress.eta}")
```

#### Episodic Memory Migration

```python
from safla.migration import EpisodicMemoryMigrator

async def migrate_episodic_data():
    migrator = EpisodicMemoryMigrator()
    
    # Migrate from SQLite to PostgreSQL
    await migrator.migrate(
        source_db="sqlite:///episodes.db",
        target_db="postgresql://user:pass@localhost/safla",
        batch_size=10000,
        preserve_timestamps=True
    )
```

### Large Dataset Migration

For datasets larger than available memory:

```python
from safla.migration import StreamingMigrator

async def migrate_large_dataset():
    migrator = StreamingMigrator()
    
    # Configure streaming migration
    await migrator.configure(
        source_path="./large_dataset/",
        target_memory="vector",
        chunk_size="1GB",
        parallel_workers=4
    )
    
    # Start streaming migration
    async for chunk_result in migrator.stream_migrate():
        print(f"Chunk {chunk_result.chunk_id} completed")
        print(f"Records processed: {chunk_result.records}")
        
        # Optional: pause migration if needed
        if chunk_result.chunk_id % 10 == 0:
            await migrator.pause(duration=30)  # 30 second pause
```

## Configuration Migration

### Automated Configuration Migration

```bash
# Use the built-in migration tool
safla config migrate \
  --from-version 0.9 \
  --to-version 1.0 \
  --config-dir ./config \
  --backup-dir ./config-backup \
  --validate
```

### Manual Configuration Migration

#### Memory Configuration

```yaml
# OLD FORMAT (0.9.x)
memory_config:
  vector_store:
    backend: "faiss"
    index: "flat"
    metric: "cosine"
  
  episode_store:
    backend: "sqlite"
    file: "episodes.db"

# NEW FORMAT (1.0.0)
memory:
  vector:
    provider: "faiss"
    index_type: "hnsw"
    similarity_metric: "cosine"
    dimensions: 768
  
  episodic:
    provider: "sqlite"
    database_url: "sqlite:///episodes.db"
    max_episodes: 1000000
```

#### Security Configuration

```yaml
# OLD FORMAT (0.9.x)
security:
  auth_enabled: true
  jwt_secret: "secret"
  
# NEW FORMAT (1.0.0)
security:
  authentication:
    provider: "jwt"
    secret_key: "${JWT_SECRET}"
    token_expiry: "1h"
    mfa_required: false
  
  authorization:
    rbac_enabled: true
    default_role: "viewer"
```

### Environment Variable Migration

```bash
# OLD (0.9.x)
export SAFLA_MEMORY_BACKEND=faiss
export SAFLA_AUTH_SECRET=mysecret

# NEW (1.0.0)
export SAFLA_MEMORY_VECTOR_PROVIDER=faiss
export SAFLA_SECURITY_JWT_SECRET=mysecret
export SAFLA_MEMORY_VECTOR_DIMENSIONS=768
```

## API Migration

### REST API Changes

#### Endpoint Changes

| Old Endpoint (0.9.x) | New Endpoint (1.0.0) | Notes |
|----------------------|----------------------|-------|
| `/api/memory/store` | `/api/v1/memory/vector/store` | Separated by memory type |
| `/api/memory/search` | `/api/v1/memory/vector/search` | Enhanced search parameters |
| `/api/agents/list` | `/api/v1/coordination/agents` | New coordination API |
| `/api/safety/check` | `/api/v1/safety/validate` | Enhanced safety validation |

#### Request/Response Format Changes

```python
# OLD (0.9.x)
import requests

response = requests.post('/api/memory/store', {
    'data': 'text data',
    'type': 'vector'
})

# NEW (1.0.0)
response = requests.post('/api/v1/memory/vector/store', {
    'content': 'text data',
    'metadata': {'source': 'user_input'},
    'embedding_model': 'text-embedding-ada-002'
})
```

### Python SDK Changes

```python
# OLD (0.9.x)
from safla import SAFLA

safla = SAFLA()
result = safla.store_memory(data, memory_type='vector')
search_results = safla.search_memory(query, memory_type='vector')

# NEW (1.0.0)
from safla import SAFLA
from safla.memory import VectorMemory

safla = SAFLA()
vector_memory = VectorMemory()

result = await vector_memory.store(data, metadata={'type': 'user_input'})
search_results = await vector_memory.search(query, limit=10, threshold=0.7)
```

### JavaScript/TypeScript SDK Changes

```typescript
// OLD (0.9.x)
import { SAFLA } from 'safla';

const safla = new SAFLA();
const result = await safla.storeMemory(data, 'vector');

// NEW (1.0.0)
import { SAFLA, VectorMemory } from 'safla';

const safla = new SAFLA();
const vectorMemory = new VectorMemory();

const result = await vectorMemory.store(data, {
  metadata: { source: 'user_input' }
});
```

## Deployment Migration

### Docker Migration

#### Old Docker Setup (0.9.x)

```dockerfile
FROM safla/safla:0.9
COPY config.json /app/config.json
EXPOSE 8080
CMD ["safla", "start"]
```

#### New Docker Setup (1.0.0)

```dockerfile
FROM safla/safla:1.0
COPY config/ /app/config/
COPY .env /app/.env
EXPOSE 8080 8081
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD safla health-check
CMD ["safla", "start", "--config-dir", "/app/config"]
```

### Kubernetes Migration

#### Migration Script

```bash
#!/bin/bash
# k8s-migration.sh

# Backup current deployment
kubectl get deployment safla -o yaml > safla-deployment-backup.yaml

# Scale down old deployment
kubectl scale deployment safla --replicas=0

# Apply new configuration
kubectl apply -f k8s/safla-1.0-deployment.yaml

# Wait for rollout
kubectl rollout status deployment/safla

# Verify migration
kubectl exec -it deployment/safla -- safla validate
```

#### New Kubernetes Configuration

```yaml
# k8s/safla-1.0-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: safla
  labels:
    app: safla
    version: "1.0"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: safla
  template:
    metadata:
      labels:
        app: safla
        version: "1.0"
    spec:
      containers:
      - name: safla
        image: safla/safla:1.0.0
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 8081
          name: metrics
        env:
        - name: SAFLA_CONFIG_DIR
          value: "/app/config"
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: data
          mountPath: /app/data
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: safla-config
      - name: data
        persistentVolumeClaim:
          claimName: safla-data
```

## Rollback Procedures

### Automated Rollback

```bash
# Create rollback point before migration
safla backup create --name "pre-migration-$(date +%Y%m%d)"

# If migration fails, rollback
safla rollback --to "pre-migration-$(date +%Y%m%d)" --force
```

### Manual Rollback Steps

1. **Stop SAFLA services**:
   ```bash
   safla stop --all-services
   ```

2. **Restore configuration**:
   ```bash
   cp -r config-backup-20250101/ config/
   ```

3. **Restore data**:
   ```bash
   safla restore --backup backup-20250101.tar.gz
   ```

4. **Downgrade software**:
   ```bash
   pip install safla==0.9.5  # Previous version
   ```

5. **Restart services**:
   ```bash
   safla start --validate-config
   ```

### Kubernetes Rollback

```bash
# Rollback to previous deployment
kubectl rollout undo deployment/safla

# Rollback to specific revision
kubectl rollout undo deployment/safla --to-revision=2

# Check rollback status
kubectl rollout status deployment/safla
```

## Testing and Validation

### Pre-Migration Testing

```bash
# Test current system
safla test --comprehensive --output pre-migration-test.json

# Validate data integrity
safla validate data --all-types --checksum

# Performance baseline
safla benchmark --output pre-migration-benchmark.json
```

### Post-Migration Validation

```bash
# Validate migration success
safla validate migration --from-version 0.9 --to-version 1.0

# Test all functionality
safla test --comprehensive --output post-migration-test.json

# Compare performance
safla benchmark --compare-with pre-migration-benchmark.json

# Validate data integrity
safla validate data --all-types --verify-checksums
```

### Custom Validation Scripts

```python
# custom-validation.py
import asyncio
from safla import SAFLA
from safla.testing import ValidationSuite

async def custom_validation():
    safla = SAFLA()
    validator = ValidationSuite()
    
    # Test memory operations
    memory_results = await validator.test_memory_operations()
    print(f"Memory tests: {memory_results.passed}/{memory_results.total}")
    
    # Test agent coordination
    agent_results = await validator.test_agent_coordination()
    print(f"Agent tests: {agent_results.passed}/{agent_results.total}")
    
    # Test safety mechanisms
    safety_results = await validator.test_safety_mechanisms()
    print(f"Safety tests: {safety_results.passed}/{safety_results.total}")
    
    # Custom business logic tests
    custom_results = await validator.run_custom_tests([
        test_custom_workflow,
        test_integration_points,
        test_performance_requirements
    ])
    
    return validator.generate_report()

if __name__ == "__main__":
    report = asyncio.run(custom_validation())
    print(report)
```

## Migration Troubleshooting

### Common Issues

#### Memory Migration Failures

**Issue**: Vector migration fails with dimension mismatch
```bash
Error: Vector dimension mismatch. Source: 512, Target: 768
```

**Solution**:
```python
# Use dimension transformation
from safla.migration import DimensionTransformer

transformer = DimensionTransformer()
await transformer.transform_vectors(
    source_dim=512,
    target_dim=768,
    method="pad_zeros"  # or "truncate", "interpolate"
)
```

#### Configuration Validation Errors

**Issue**: Invalid configuration after migration
```bash
Error: Invalid memory configuration. Missing required field: dimensions
```

**Solution**:
```bash
# Use configuration fixer
safla config fix --auto-fix --backup

# Or manually update
safla config validate --fix-suggestions
```

#### Agent Registration Failures

**Issue**: Custom agents fail to register after migration
```bash
Error: Agent capability 'custom_analysis' not recognized
```

**Solution**:
```python
# Update agent capabilities
from safla.agents import AgentCapabilities

# Register custom capability
AgentCapabilities.register_custom('custom_analysis')

# Or use standard capabilities
class CustomAgent(Agent):
    def __init__(self):
        super().__init__(
            'custom-agent',
            capabilities=[AgentCapabilities.DATA_ANALYSIS]  # Use standard
        )
```

### Migration Support

#### Getting Help

1. **Documentation**: Check migration-specific docs
2. **Community Forum**: Ask migration questions
3. **GitHub Issues**: Report migration bugs
4. **Professional Support**: Enterprise migration assistance

#### Migration Tools

```bash
# Migration health check
safla migration health-check --from 0.9 --to 1.0

# Migration dry run
safla migration dry-run --config ./config --data ./data

# Migration progress monitoring
safla migration status --watch

# Migration cleanup
safla migration cleanup --remove-backups --older-than 30d
```

---

This comprehensive migration guide ensures smooth transitions between SAFLA versions while maintaining data integrity and system functionality. Always test migrations in a staging environment before applying to production systems.

**Last Updated**: January 2025  
**Version**: 1.0.0  
**Maintained by**: SAFLA Migration Team