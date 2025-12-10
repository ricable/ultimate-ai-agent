#!/usr/bin/env python3

import sys
sys.path.insert(0, '/workspaces/SAFLA')

from safla.utils.pydantic_config import SAFLAConfig, IntegrationConfig, DevelopmentConfig

# Create config exactly like the test
config = SAFLAConfig(
    integration=IntegrationConfig(database_url="sqlite:///prod.db"),
    development=DevelopmentConfig(
        test_mode=False,
        test_database_url="sqlite:///test.db"
    )
)

print("=== Debug Config Values ===")
print(f"config.integration.database_url: {config.integration.database_url}")
print(f"config.development.test_database_url: {config.development.test_database_url}")
print(f"config.development.test_mode: {config.development.test_mode}")
print(f"config.get_database_url(): {config.get_database_url()}")

# Check if there are any environment variables affecting this
import os
print("\n=== Environment Variables ===")
for key, value in os.environ.items():
    if 'DATABASE' in key.upper() or 'SAFLA' in key.upper():
        print(f"{key}: {value}")

# Check the integration config directly
print(f"\n=== Integration Config ===")
print(f"Type: {type(config.integration)}")
print(f"Dict: {config.integration.model_dump()}")