#!/usr/bin/env python3
"""Debug script to test enhanced endpoint availability."""

import requests
import json

def test_debug_endpoint():
    """Test debug endpoint to see what's available."""
    url = "https://safla.fly.dev/api/safla"
    
    # Test a simple method first
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "debug_info",
        "params": {}
    }
    
    print("Testing debug_info endpoint...")
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

    # Test if enhanced endpoints module is available
    payload2 = {
        "jsonrpc": "2.0", 
        "id": 2,
        "method": "enhanced_status",
        "params": {}
    }
    
    print("\nTesting enhanced_status endpoint...")
    try:
        response = requests.post(url, json=payload2, timeout=30)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_debug_endpoint()