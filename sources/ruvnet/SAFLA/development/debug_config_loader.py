#!/usr/bin/env python3

import sys
import json
import tempfile
sys.path.insert(0, '/workspaces/SAFLA')

from safla.utils.config_loader import ConfigLoader

# Create the same config data as the test
config_data = {
    "debug": True,
    "log_level": "DEBUG",
    "performance": {
        "worker_threads": 8,
        "batch_size": 64
    },
    "memory": {
        "max_memories": 5000,
        "similarity_threshold": 0.9
    }
}

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(config_data, f)
    config_file = f.name

print(f"Created config file: {config_file}")
print(f"Config data: {config_data}")

try:
    loader = ConfigLoader()
    config = loader.load_config(config_file=config_file, create_dirs=False)
    
    print("\n=== Loaded Config Values ===")
    print(f"config.debug: {config.debug}")
    print(f"config.log_level: {config.log_level}")
    print(f"config.performance.worker_threads: {config.performance.worker_threads}")
    print(f"config.performance.batch_size: {config.performance.batch_size}")
    print(f"config.memory.max_memories: {config.memory.max_memories}")
    print(f"config.memory.similarity_threshold: {config.memory.similarity_threshold}")
    
    print(f"\n=== Full Config Dict ===")
    print(config.model_dump())
    
finally:
    import os
    os.unlink(config_file)