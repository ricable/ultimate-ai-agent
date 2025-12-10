#!/bin/bash
# Example: Using SAFLA via MCP with cURL

curl -X POST https://safla.fly.dev/api/safla \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "embed_001", 
    "method": "generate_embeddings",
    "params": {
      "texts": ["Hello SAFLA!", "This is a test"],
      "config": {
        "batch_size": 256,
        "cache_embeddings": true
      }
    }
  }'
